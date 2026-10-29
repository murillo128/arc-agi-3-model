"""Train independent game sessions with cross-level recurrent context.

Run: python -m model.sequence_training --training-config CONFIG --output DIR
Use --source CHECKPOINT to warm-start weights, or --resume to retain optimizers.
SIGINT/SIGTERM finish the current batch and atomically save its completed state.
"""
import argparse
import copy
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import random
import signal
import time

import torch

from . import checkpoint
from .config import NetworkConfig, TrainConfig, SequenceConfig
from .environment import Factory
from .game_sequences import (SequenceLearner, EncodedSessions, collect_session,
                             evaluate_sessions, schedule_games, merge_evaluations)
from .learning import Engine
from .networks import WorldModel
from .online_learning import OnlineRepresentation
from .training_config import digest, load_plan, run_owner
from .sequence_control import SequenceControlConfig, SequencePhaseControl, TrainingBatchDiagnostics
from .sequence_exploration import SequenceExplorationConfig, ChangeExplorer, interaction_counts, sum_counts
from .sequence_upgrade import (normalized_plan, upgrade_plan_matches, upgraded_network,
                               load_upgraded_weights, upgraded_optimizer)
from .sequence_sampling import reweighted_plan_matches


def synchronize(engine):
    if engine.device.type == 'cuda':
        torch.cuda.synchronize(engine.device)


def factories_for(plan):
    sources = {}
    for stage in plan.stages:
        item = sources.setdefault(stage['game'], dict(source=stage['source'], levels=[]))
        item['levels'].append(stage['level'])
    result = {}
    try:
        for game, item in sources.items():
            if item['source'] == 'sdk':
                if plan.payload['phase'] != 'development':
                    raise ValueError('Sequence SDK experiments require a development split')
                if item['levels'] != list(range(1, max(item['levels']) + 1)):
                    raise ValueError('SDK sequence levels must be the allowed prefix starting at 1')
                sdk = plan.payload['sdk']
                factory = Factory(game, sdk['environments_dir'], sdk['mode'], reserve_final_level=True)
            else:
                factory = Factory(game, Path(__file__).resolve().parents[1] / 'games')
            result[game] = (factory, item['levels'])
    except Exception:
        close_factories(result)
        raise
    return result


def close_factories(factories):
    for factory, _ in factories.values():
        factory.arcade.close_scorecard()


def collect_batch(factories, config, batch_index, seed, *, evaluation=False, exploration=None, explorer=None,
                  game_weights=None, sampling_start_batch=0, planning=None):
    games = schedule_games(list(factories), batch_index - sampling_start_batch, config,
                           seed=seed, weights=game_weights)
    sessions = []
    for i, game in enumerate(games):
        # Even training / odd evaluation seeds, including intra-session resets.
        # Reserve a full batch's seed slots even when a shuffled round ends short.
        session_seed = 4_000_000 + 2 * (seed * 10_000 + batch_index * config.batch_games + i) + int(evaluation)
        factory, levels = factories[game]
        options = {} if evaluation or exploration is None else dict(exploration=exploration, explorer=explorer)
        if planning is not None and not evaluation:
            options['planning'] = planning
        sessions.append(collect_session(factory, levels, session_seed, config, **options))
    if any(not any(s.supervised for s in row.transitions) for row in sessions):
        raise ValueError('A session has no eligible real transitions; check level access and budgets')
    return sessions


def load_engine(plan, source=None, *, device='cpu', compile_model=False, upgrade_predictor=False,
                fix_click_sigma=False):
    if upgrade_predictor and fix_click_sigma:
        raise ValueError('Predictor upgrade and fixed click sigma cannot be combined')
    initial = plan.payload['initialization']
    saved = checkpoint.load(source) if source else None
    if saved and saved.get('format') == 'arc3-agent':
        saved = saved['world']
    network = NetworkConfig(**(saved['network'] if saved else initial['network']))
    if upgrade_predictor or fix_click_sigma:
        if not saved or saved.get('kind') != 'game_sequences':
            raise ValueError('Predictor upgrade requires a sequence checkpoint')
        network = upgraded_network(saved, initial, fix_click_sigma=fix_click_sigma)
    training = TrainConfig(**(saved['training'] if saved else initial.get('training', {})))
    training = replace(training, learning_rate=plan.settings['learning_rate'],
                       prediction_delta_loss=plan.settings['prediction_delta_loss'])
    net = WorldModel(network).to(device)
    if saved:
        if upgrade_predictor or fix_click_sigma:
            load_upgraded_weights(net, saved, fix_click_sigma=fix_click_sigma)
        else:
            net.load_state_dict(saved['weights'], strict=True)
    return Engine(net, training, compile_model=compile_model)


def representation_for(engine, plan):
    learner = OnlineRepresentation(engine, plan.settings['representation_learning_rate'],
        freeze_encoder=plan.settings['freeze_encoder'],
        encoder_learning_rate_after_predictor=plan.settings['encoder_learning_rate_after_predictor'])
    if 'sequence_control' not in plan.payload:
        learner.activate_predictor()
    return learner


def predictor_learning_rate(learner, value=None):
    """Override only the predictor rate, retaining Adam state and the run identity."""
    if value is not None:
        if not math.isfinite(value) or not 0 < value < 1:
            raise ValueError('Predictor learning rate must be finite and between 0 and 1')
        for group in learner.optimizer.param_groups:
            group['lr'] = value
    # Ordinary resume restores the optimizer's effective rate, including an earlier
    # explicit override. Keep checkpoint training metadata consistent with it.
    learner.engine.config = replace(learner.engine.config,
        learning_rate=learner.optimizer.param_groups[0]['lr'])


def fit_batch(learner, sessions, representation=None, *, telemetry=None,
              train_predictor=True, training_probe=None):
    """One representation phase, then causal predictor blocks on fixed latents."""
    e, c = learner.engine, learner.config
    synchronize(e)
    begin = time.perf_counter()
    reconstruction = []
    frozen = bool(representation is not None and getattr(representation, 'frozen', False))
    if representation is not None and not frozen:
        # Rotate games evenly. Frame selection is independent of prediction error.
        rng = random.Random(sum(s.seed for s in sessions))
        for i in range(c.representation_updates):
            candidates = [s for s in sessions[i % len(sessions)].transitions if s.supervised]
            if candidates:
                step = rng.choice(candidates)
                result = representation.update([step.before, step.after], telemetry=telemetry)
                if result is not None:
                    reconstruction.append(result)
    synchronize(e)
    representation_seconds = 0. if frozen else time.perf_counter() - begin
    drift_seconds = 0.
    if training_probe is not None:
        start = time.perf_counter()
        training_probe.after_representation(frozen=frozen)
        synchronize(e)
        drift_seconds = 0. if frozen else time.perf_counter() - start
    start = time.perf_counter()
    e.model.eval()
    encoded = EncodedSessions(e, sessions) if train_predictor else None
    synchronize(e)
    encode_seconds = time.perf_counter() - start
    start = time.perf_counter()
    updates = []
    length = max(len(s.transitions) for s in sessions)
    last_supervised = max((i for s in sessions for i, step in enumerate(s.transitions) if step.supervised), default=-1)
    for position in range(0, length if train_predictor else 0, c.block_steps):
        result = learner.update(encoded, position, min(length, position + c.block_steps),
            telemetry=telemetry, last=position <= last_supervised < position + c.block_steps)
        if result is not None:
            updates.append(result)
    synchronize(e)
    predictor_seconds = time.perf_counter() - start
    probe_seconds = 0.
    if telemetry and telemetry.probe_requested and train_predictor:
        from .optimization_probe import memory_probe
        probe_start = time.perf_counter()
        telemetry.probe = dict(memory_probe(learner, encoded, telemetry.probe_session),
                               batch=telemetry.batch, event_id=telemetry.tick)
        telemetry.probe_requested = False
        synchronize(e)
        probe_seconds = time.perf_counter() - probe_start
    return dict(representation_seconds=representation_seconds, encode_seconds=encode_seconds,
        predictor_seconds=predictor_seconds, training_seconds=time.perf_counter() - begin,
        drift_seconds=drift_seconds,
        probe_seconds=probe_seconds,
        updates=len(updates), examples=sum(x['examples'] for x in updates),
        loss=sum(x['loss'] for x in updates)/len(updates) if updates else None,
        predictor_terms={key: sum(x[key] for x in updates)/len(updates)
                         for key in ('latent', 'visual')} if updates else {},
        reconstruction_loss=sum(x['total'] for x in reconstruction)/len(reconstruction) if reconstruction else None,
        representation_terms={key: sum(x['terms'][key] for x in reconstruction if key in x['terms']) /
                              sum(key in x['terms'] for x in reconstruction)
                              for key in {k for x in reconstruction for k in x['terms']}},
        examples_per_second=sum(x['examples'] for x in updates)/max(1e-9, predictor_seconds))


def save_run(path, plan, learner, representation, batches, actions, control=None, *, extensions=(),
             restrictions=(), explorer=None, exploration_adoptions=(), predictor_adaptations=(),
             game_weight_adoptions=(), planning=None):
    e = learner.engine
    checkpoint.write(path, dict(kind='game_sequences', version=1, plan=plan.payload,
        network=asdict(e.model.config), training=asdict(e.config), weights=e.model.state_dict(),
        sequence=asdict(learner.config), predictor_optimizer=learner.optimizer.state_dict(),
        representation_optimizer=representation.optimizer.state_dict(),
        progress=dict(batches=batches, actions=actions, updates=learner.updates),
        curriculum_extensions=list(extensions), curriculum_restrictions=list(restrictions),
        exploration_adoptions=list(exploration_adoptions),
        predictor_adaptations=list(predictor_adaptations),
        game_weight_adoptions=list(game_weight_adoptions),
        **(dict(exploration=explorer.snapshot()) if explorer is not None else {}),
        **(dict(planning=planning.snapshot()) if planning is not None else {}),
        torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if e.device.type == 'cuda' else [],
        **(dict(sequence_control=control.snapshot(), representation_state=dict(
            sigreg_enabled=representation.sigreg_enabled, updates=representation.updates)) if control else {})))


def extended_plan_matches(previous, current):
    """Only an appended curriculum and the first SDK setup may differ."""
    count = len(previous['stages'])
    if len(current['stages']) <= count or current['stages'][:count] != previous['stages']:
        return False
    expected = copy.deepcopy(current)
    expected['stages'] = previous['stages']
    if not previous['sdk']:
        expected['sdk'] = {}
    if 'sequence_control' in previous and 'sequence_control' in current:
        old_window = previous['sequence_control']['window_batches']
        if current['sequence_control']['window_batches'] < old_window:
            return False
        expected['sequence_control']['window_batches'] = old_window
    return expected == previous


def restricted_plan_matches(previous, current):
    """Only remove stages, retaining their order, definitions and all settings."""
    if not 0 < len(current['stages']) < len(previous['stages']):
        return False
    remaining = iter(previous['stages'])
    if not all(any(old == stage for old in remaining) for stage in current['stages']):
        return False
    expected = copy.deepcopy(current)
    expected['stages'] = previous['stages']
    return expected == previous


def restore_run(saved, plan, learner, representation, control=None, *, adopt=False, extend=False, restrict=False,
                adopt_exploration=False, explorer=None, disable_undo=False, upgrade_predictor=False,
                fix_click_sigma=False, adopt_game_weights=False, planning=None, adopt_planning=False):
    if fix_click_sigma and (disable_undo or upgrade_predictor):
        raise ValueError('Fixed click sigma cannot be combined with other adaptations')
    adaptation = disable_undo or upgrade_predictor or fix_click_sigma
    if saved.get('sequence_control', {}).get('representation_frozen_at_batch') is not None and any(
            (adopt, extend, restrict, adopt_exploration, adaptation, adopt_game_weights)):
        raise ValueError('Frozen representation cannot be combined with other adoptions or curriculum changes')
    if sum((adopt, extend, restrict, adopt_exploration, adaptation, adopt_game_weights)) > 1:
        raise ValueError('Explicit adoptions/extensions cannot be combined')
    if adopt_planning and (any((adopt, extend, restrict, adopt_exploration, adaptation, adopt_game_weights))
            or planning is None or 'planning' in saved or 'sequence_planning' in saved['plan']):
        raise ValueError('Planning adoption requires a new head and no other adoption')
    if planning is not None and (saved.get('sequence_control', {}).get('phase') != 'predictor'
            or saved.get('sequence_control', {}).get('representation_frozen_at_batch') is None):
        raise ValueError('Sequence planning requires frozen representation in predictor phase')
    if planning is None and 'planning' in saved:
        raise ValueError('Cannot discard saved planning heads')
    expected = normalized_plan(plan.payload)
    if adopt:
        if control is None or 'sequence_control' in saved or 'sequence_control' in saved['plan']:
            raise ValueError('Phase adoption requires a legacy sequence checkpoint and a controlled plan')
        expected.pop('sequence_control')
        expected['settings']['encoder_learning_rate_after_predictor'] = saved['plan']['settings']['encoder_learning_rate_after_predictor']
    previous_plan = normalized_plan(saved['plan'])
    if adopt_planning:
        expected.pop('sequence_planning', None)
    if adopt_exploration:
        # Initial adoption only: never silently reinitialize an existing head.
        if saved.get('exploration') is not None or explorer is None:
            raise ValueError('Exploration adoption requires a new learned head')
        previous_plan = copy.deepcopy(previous_plan)
        previous_plan.pop('sequence_exploration', None)
        expected.pop('sequence_exploration', None)
    changed = previous_plan != expected
    if (saved.get('kind') != 'game_sequences' or saved.get('version') != 1 or
            (changed and not (extend and extended_plan_matches(previous_plan, expected)) and
             not (restrict and restricted_plan_matches(previous_plan, expected)) and
             not (adopt_game_weights and reweighted_plan_matches(previous_plan, expected)) and
             not (adaptation and upgrade_plan_matches(previous_plan, expected,
                  disable_undo=disable_undo, upgrade_predictor=upgrade_predictor,
                  fix_click_sigma=fix_click_sigma))) or
            saved['sequence'] != asdict(learner.config)):
        raise ValueError('Resume requires the same sequence configuration and training plan')
    if control is not None and not adopt:
        if 'sequence_control' not in saved:
            raise ValueError('Missing checkpoint phase state; use explicit adoption for legacy runs')
        control.restore(saved['sequence_control'], extend=extend and changed, restrict=restrict and changed)
    if explorer is not None and not adopt_exploration:
        if 'exploration' not in saved:
            raise ValueError('Missing change-head state; use --adopt-exploration for initial adoption')
        explorer.restore(saved['exploration'], disable_undo=disable_undo)
        if restrict and changed:
            # Replay has no game labels: discard old examples, retaining the
            # learned head, Adam moments, counters and random generator.
            explorer.replay.clear()
    if upgrade_predictor or fix_click_sigma:
        target = upgraded_network(saved, plan.payload['initialization'], fix_click_sigma=fix_click_sigma)
        if learner.engine.model.config != target:
            raise ValueError('Predictor upgrade must load the expanded model first')
        learner.optimizer.load_state_dict(upgraded_optimizer(saved, learner.engine.model.dynamics,
                                                            fix_click_sigma=fix_click_sigma))
    else:
        learner.optimizer.load_state_dict(saved['predictor_optimizer'])
    representation.optimizer.load_state_dict(saved['representation_optimizer'])
    learner.updates = saved['progress']['updates']
    if control is not None:
        state = saved.get('representation_state', {})
        representation.sigreg_enabled = state.get('sigreg_enabled',
            representation.sigreg_enabled and learner.updates == 0)
        representation.updates = state.get('updates', 0)
        control.apply(representation)
    if planning is not None:
        if adopt_planning:
            planning.adopted_batch = saved['progress']['batches']
        else:
            if 'planning' not in saved:
                raise ValueError('Missing planner weights; use --adopt-planning once')
            planning.restore(saved['planning'])
    torch.set_rng_state(saved['torch_rng'])
    if learner.engine.device.type == 'cuda' and saved['cuda_rng']:
        if len(saved['cuda_rng']) != torch.cuda.device_count():
            raise ValueError('CUDA device count changed on resume')
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
    return saved['progress']['batches'], saved['progress']['actions']


def write_report(path, report):
    report['updated_at'] = time.time()
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, allow_nan=False) + '\n')
    temporary.replace(path)


def click_map_report(engine):
    if not engine.model.config.rotation_adapter_width:
        return None
    sigma = engine.model.dynamics.rotation_adapter.click_sigma()
    return dict(sigma_pixels=float(sigma.detach().cpu()) if isinstance(sigma, torch.Tensor) else sigma,
                learnable=engine.model.config.learnable_click_sigma, peak=1.,
                minimum_sigma_pixels=.375, maximum_sigma_pixels=6.)


def run(args):
    rate = getattr(args, 'predictor_learning_rate', None)
    if rate is not None and (not math.isfinite(rate) or not 0 < rate < 1):
        raise ValueError('Predictor learning rate must be finite and between 0 and 1')
    plan = load_plan(args.training_config)
    if 'sequence' not in plan.payload:
        raise ValueError('Add sequence settings to the training configuration')
    config = SequenceConfig(**plan.payload['sequence'])
    exploration = SequenceExplorationConfig(**plan.payload.get('sequence_exploration', {}))
    output = args.output.resolve()
    if args.resume and args.source:
        raise ValueError('Choose --source for a fresh run or --resume, not both')
    adopt = getattr(args, 'adopt_training_phases', False)
    extend = getattr(args, 'extend_curriculum', False)
    restrict = getattr(args, 'restrict_curriculum', False)
    adopt_exploration = getattr(args, 'adopt_exploration', False)
    disable_undo = getattr(args, 'disable_undo', False)
    upgrade_predictor = getattr(args, 'upgrade_predictor', False)
    fix_click_sigma = getattr(args, 'fix_click_sigma', False)
    adopt_game_weights = getattr(args, 'adopt_game_weights', False)
    freeze_representation = getattr(args, 'freeze_representation', False)
    adopt_planning = getattr(args, 'adopt_planning', False)
    if adopt_planning and (not args.resume or 'sequence_planning' not in plan.payload or rate is not None or freeze_representation):
        raise ValueError('--adopt-planning requires a planner-enabled resume without LR/freeze override')
    if 'sequence_planning' in plan.payload and not args.resume:
        raise ValueError('Adopt planning from an existing frozen sequence run')
    if freeze_representation and (not args.resume or 'sequence_control' not in plan.payload or any(
            (adopt, extend, restrict, adopt_exploration, disable_undo, upgrade_predictor,
             fix_click_sigma, adopt_game_weights, rate is not None))):
        raise ValueError('--freeze-representation requires controlled --resume without other adoptions or LR override')
    if fix_click_sigma and (disable_undo or upgrade_predictor):
        raise ValueError('--fix-click-sigma cannot be combined with other adaptations')
    adaptation = disable_undo or upgrade_predictor or fix_click_sigma
    if adopt_game_weights and (not args.resume or adopt or extend or restrict or
                              adopt_exploration or adaptation or rate is not None):
        raise ValueError('--adopt-game-weights requires --resume without other adoptions or LR override')
    if adaptation and (not args.resume or adopt or extend or restrict or adopt_exploration):
        raise ValueError('Predictor/Undo adaptation requires --resume without other adoptions')
    if disable_undo and exploration.allow_undo:
        raise ValueError('--disable-undo requires sequence_exploration.allow_undo=false')
    exploration_source = getattr(args, 'exploration_source', None)
    if adopt_exploration and (not args.resume or adopt or extend or exploration.policy != 'learned'):
        raise ValueError('--adopt-exploration requires a learned policy and --resume without other adoptions')
    if exploration_source and (exploration.policy != 'learned' or (args.resume and not adopt_exploration)):
        raise ValueError('Auxiliary bootstrap is only allowed for a fresh head')
    if adopt and not args.resume:
        raise ValueError('--adopt-training-phases requires --resume')
    if extend and (not args.resume or adopt):
        raise ValueError('--extend-curriculum requires --resume without --adopt-training-phases')
    if restrict and (not args.resume or adopt or extend or adopt_exploration):
        raise ValueError('--restrict-curriculum requires --resume without other adoptions/extensions')
    if args.max_batches is not None and args.max_batches < 1:
        raise ValueError('--max-batches must be positive')
    action_budget = None if getattr(args, 'unlimited', False) else plan.settings['actions']
    with run_owner(output, args.resume):
        source = output / 'latest.pt' if args.resume else args.source
        torch.manual_seed(plan.settings['seed'])
        engine = load_engine(plan, source, device=args.device, compile_model=args.compile,
                             upgrade_predictor=upgrade_predictor, fix_click_sigma=fix_click_sigma)
        learner, representation = SequenceLearner(engine, config), representation_for(engine, plan)
        control = (SequencePhaseControl(plan.stages, plan.settings,
            SequenceControlConfig(**plan.payload['sequence_control'])) if 'sequence_control' in plan.payload else None)
        explorer = ChangeExplorer(exploration) if exploration.policy == 'learned' else None
        objects = None
        if 'sequence_objects' in plan.payload:
            from .segmentation import SAM2Segmenter
            from .object_training import ObjectTrainer, ObjectTrainingConfig
            object_config = ObjectTrainingConfig(**plan.payload['sequence_objects'])
            provider = SAM2Segmenter(object_config.checkpoint, device=object_config.device,
                capacity=engine.model.config.object_slots, cache_frames=object_config.cache_frames,
                points_per_side=object_config.points_per_side)
            objects = ObjectTrainer(engine, object_config, provider)
            if explorer is not None:
                explorer.segmenter = provider
        from .sequence_planning import SequencePlanning, SequencePlanningConfig
        planning = (SequencePlanning(engine, SequencePlanningConfig(**plan.payload['sequence_planning']),
                    explorer=explorer, seed=plan.settings['seed']) if 'sequence_planning' in plan.payload else None)
        if exploration_source:
            bootstrap = checkpoint.load(exploration_source)
            if bootstrap.get('kind') != 'sequence_change_head':
                raise ValueError('Expected an auxiliary change-head checkpoint')
            explorer.restore(bootstrap['exploration'])
        batches, actions, extensions, exploration_adoptions = 0, 0, [], []
        restrictions, restricted = [], False
        predictor_adaptations, adapted = [], False
        game_weight_adoptions = []
        if args.resume:
            saved = checkpoint.load(source)
            batches, actions = restore_run(saved, plan, learner, representation, control,
                                           adopt=adopt, extend=extend, restrict=restrict,
                                           adopt_exploration=adopt_exploration, explorer=explorer,
                                           disable_undo=disable_undo, upgrade_predictor=upgrade_predictor,
                                           fix_click_sigma=fix_click_sigma, adopt_game_weights=adopt_game_weights,
                                           planning=planning, adopt_planning=adopt_planning)
            game_weight_adoptions = list(saved.get('game_weight_adoptions', []))
            if adopt_game_weights and saved['plan'].get('sequence_game_weights', {}) != plan.payload.get('sequence_game_weights', {}):
                game_weight_adoptions.append(dict(batch=batches, actions=actions,
                    before=saved['plan'].get('sequence_game_weights', {}),
                    after=plan.payload.get('sequence_game_weights', {})))
            predictor_adaptations = list(saved.get('predictor_adaptations', []))
            adapted = adaptation and normalized_plan(saved['plan']) != normalized_plan(plan.payload)
            if adapted:
                predictor_adaptations.append(dict(batch=batches, actions=actions,
                    allow_undo=exploration.allow_undo,
                    rotation_adapter_width=engine.model.config.rotation_adapter_width,
                    learnable_click_sigma=engine.model.config.learnable_click_sigma,
                    fix_click_sigma=fix_click_sigma,
                    previous_plan=digest(saved['plan']), plan=plan.identity))
            extensions = list(saved.get('curriculum_extensions', []))
            restrictions = list(saved.get('curriculum_restrictions', []))
            exploration_adoptions = list(saved.get('exploration_adoptions', []))
            if adopt_exploration:
                exploration_adoptions.append(dict(batch=batches, actions=actions, config=asdict(exploration)))
            if extend and saved['plan'] != plan.payload:
                extensions.append(dict(batch=batches, actions=actions,
                    previous_plan=digest(saved['plan']), plan=plan.identity,
                    games_before=list(dict.fromkeys(s['game'] for s in saved['plan']['stages'])),
                    games_after=list(dict.fromkeys(s['game'] for s in plan.stages)),
                    added_stages=list(plan.stages[len(saved['plan']['stages']):])))
            restricted = restrict and saved['plan'] != plan.payload
            if restricted:
                restrictions.append(dict(batch=batches, actions=actions,
                    previous_plan=digest(saved['plan']), plan=plan.identity,
                    games_before=list(dict.fromkeys(s['game'] for s in saved['plan']['stages'])),
                    games_after=list(dict.fromkeys(s['game'] for s in plan.stages)),
                    removed_stages=[s for s in saved['plan']['stages'] if s not in plan.stages]))
            del saved
        sampling_start_batch = game_weight_adoptions[-1]['batch'] if game_weight_adoptions else 0
        predictor_learning_rate(learner, rate)
        if freeze_representation:
            control.freeze_representation(batches)
        if control:
            control.apply(representation)
        telemetry = None
        if getattr(args, 'optimization', False):
            from .optimization_telemetry import OptimizationTelemetry
            telemetry = OptimizationTelemetry(engine.model, output, every=args.optimization_every,
                restored_batch=batches,
                extra_modules=dict(goal_planner=planning.goal, path_planner=planning.path) if planning else None,
                identity=json.dumps(
                    {k:v for k,v in plan.payload.items() if k not in ('sequence_exploration', 'sequence_game_weights', 'sequence_planning')}, sort_keys=True))
        stop = False
        def interrupt(signum, frame):
            nonlocal stop
            stop = True
        handlers = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
        factories = {}
        report_path = output / 'report.json'
        report = dict(kind='game_sequences', status='running', config=asdict(config), history=[],
                      progress=dict(batches=batches, actions=actions), games=list(dict.fromkeys(s['game'] for s in plan.stages)))
        report['schedule'] = list(plan.stages)
        report['game_weights'] = plan.payload.get('sequence_game_weights', {})
        report['game_weight_adoptions'] = game_weight_adoptions
        report['sampling_start_batch'] = sampling_start_batch
        report['curriculum_extensions'] = extensions
        report['curriculum_restrictions'] = restrictions
        report['exploration_config'] = asdict(exploration)
        report['exploration_adoptions'] = exploration_adoptions
        report['predictor_adaptations'] = predictor_adaptations
        report['network'] = asdict(engine.model.config)
        if planning is not None:
            report['planning'] = dict(config=asdict(planning.config), **planning.status())
        report['click_map'] = click_map_report(engine)
        report['budget'] = action_budget
        report['measurement_source'] = 'training_pre_update' if control else 'fixed_evaluation'
        if control:
            report['phase_control'] = control.report()
        report['learning_rates'] = dict(predictor=learner.optimizer.param_groups[0]['lr'],
                                        **representation.learning_rates)
        if telemetry:
            report['optimization_identity'] = telemetry.identity
        if args.resume and report_path.exists():
            previous = json.loads(report_path.read_text())
            if adopt or restricted or adapted:
                # Do not join different metric sources or game populations.
                boundary = 'predictor-adaptation' if adapted else ('training-phases' if adopt else 'curriculum-restriction')
                archive = output / f'report-before-{boundary}-{batches}.json'
                if not archive.exists():
                    archive.write_text(json.dumps(previous, allow_nan=False) + '\n')
                if adopt:
                    report['adopted_at_batch'] = batches
            else:
                # Adding telemetry modules starts a new parameter-space segment,
                # but does not start a new training run or discard metric history.
                report['history_identity'] = previous.get('history_identity', previous.get('optimization_identity'))
                report['history'] = [r for r in previous.get('history', []) if r['batch'] <= batches][-200:]
                report['prediction_samples'] = previous.get('prediction_samples', [])
                report['prediction_sample_batch'] = previous.get('prediction_sample_batch')
                if planning is not None:
                    report['planning_sample'] = previous.get('planning_sample')
                if 'adopted_at_batch' in previous:
                    report['adopted_at_batch'] = previous['adopted_at_batch']
        completed_here = 0
        from .training_telemetry import PlannerTrajectory
        trajectories = PlannerTrajectory(output / 'planner_trajectory.json') if planning is not None else None
        # Sidecars belong to a capture process, not to checkpointed learning state.
        report['planner_trajectory'] = None
        report['trajectory_error'] = None
        try:
            factories = factories_for(plan)
            # Cover the entire configured catalog in bounded GPU batches.
            evaluation = [] if control else [collect_batch(factories, replace(config, game_schedule='interleaved'), i,
                                        plan.settings['seed'], evaluation=True)
                          for i in range((len(factories) + config.batch_games - 1)//config.batch_games)]
            from .sequence_diagnostics import SequenceDiagnostics
            diagnostics = SequenceDiagnostics(engine, [s for batch in evaluation for s in batch])
            write_report(report_path, report)
            while not stop and (action_budget is None or actions < action_budget):
                begin = time.perf_counter()
                if telemetry:
                    telemetry.start_batch(batches + 1)
                if planning is not None:
                    planning.begin_capture(trajectories, batch=batches+1)
                if objects is not None:
                    objects.begin_collection()
                sessions = collect_batch(factories, config, batches, plan.settings['seed'],
                                         exploration=exploration, explorer=explorer,
                                         game_weights=report['game_weights'], sampling_start_batch=sampling_start_batch, planning=planning)
                collection_seconds = time.perf_counter() - begin
                if control:
                    phase = control.phase
                    control.apply(representation)
                    learning_rates = dict(predictor=learner.optimizer.param_groups[0]['lr'], **representation.learning_rates)
                    probe = TrainingBatchDiagnostics(engine, sessions, control.config.diagnostic_samples)
                    probe.start(batches+1)
                    evaluation_start = time.perf_counter()
                    quality = merge_evaluations([evaluate_sessions(engine, sessions,
                        early_steps=config.loss_warmup_steps, reference_loss=bool(telemetry), diagnostics=probe)])
                    synchronize(engine)
                    evaluation_seconds = time.perf_counter() - evaluation_start
                    measured = fit_batch(learner, sessions, representation, telemetry=telemetry,
                        train_predictor=phase == 'predictor', training_probe=probe)
                    measured['diagnostics'], report['prediction_samples'] = probe.finish()
                    report['prediction_sample_batch'] = batches+1
                    measured.update(evaluation_seconds=evaluation_seconds, diagnostics_seconds=probe.seconds,
                        phase=phase, learning_rates=learning_rates, measurement_source='training_pre_update')
                    control.observe(batches+1, {k: dict(v) for k, v in probe.pairs.items()})
                    control.apply(representation)
                    report['phase_control'] = control.report()
                    report['learning_rates'] = dict(predictor=learner.optimizer.param_groups[0]['lr'], **representation.learning_rates)
                    measured['next_phase'] = control.phase
                else:
                    measured = (fit_batch(learner, sessions, representation, telemetry=telemetry) if telemetry
                                else fit_batch(learner, sessions, representation))
                    evaluation_start = time.perf_counter()
                    measure_diagnostics = completed_here == 0 or (batches+1) % diagnostics.every == 0
                    if measure_diagnostics:
                        diagnostics.start(batches+1)
                    quality = merge_evaluations([evaluate_sessions(engine, batch,
                        early_steps=config.loss_warmup_steps, reference_loss=bool(telemetry),
                        diagnostics=diagnostics if measure_diagnostics else None) for batch in evaluation])
                    if measure_diagnostics:
                        measured['diagnostics'], report['prediction_samples'] = diagnostics.finish()
                        report['prediction_sample_batch'] = batches+1
                        measured['diagnostics_seconds'] = diagnostics.seconds
                    synchronize(engine)
                    measured['evaluation_seconds'] = time.perf_counter() - evaluation_start
                head_start = time.perf_counter()
                if objects is not None and (control is None or control.phase == 'predictor'):
                    measured['objects'], object_sample = objects.fit(learner, sessions)
                    measured['object_seconds'] = measured['objects']['seconds']
                    report['objects'] = dict(config=plan.payload['sequence_objects'], **measured['objects'])
                    if object_sample is not None:
                        report['object_sample'] = dict(object_sample, batch=batches+1)
                    head_start = time.perf_counter()
                head_metrics = explorer.fit(sessions) if explorer is not None else {}
                measured['exploration_seconds'] = time.perf_counter() - head_start if explorer is not None else 0.
                interaction_rows = [interaction_counts(session) for session in sessions]
                measured['exploration'] = dict(policy=exploration.policy, identity=digest(asdict(exploration)),
                    **sum_counts(interaction_rows), head=head_metrics,
                    games=[dict(game=session.game, **counts) for session, counts in zip(sessions, interaction_rows)])
                if planning is not None:
                    measured['planning'], sample = planning.fit(sessions, telemetry=telemetry)
                    measured['planning_seconds'] = measured['planning']['seconds']
                    measured['planning']['identity'] = str(planning.adopted_batch)
                    report['planning'] = dict(config=asdict(planning.config), **planning.status())
                    if sample is not None:
                        report['planning_sample'] = dict(sample, batch=batches+1)
                real_actions = sum(s.environment_actions for s in sessions)
                batches += 1
                actions += real_actions
                save_start = time.perf_counter()
                save_run(output / 'latest.pt', plan, learner, representation, batches, actions, control,
                         extensions=extensions, restrictions=restrictions,
                         explorer=explorer, exploration_adoptions=exploration_adoptions,
                         predictor_adaptations=predictor_adaptations, game_weight_adoptions=game_weight_adoptions, planning=planning)
                measured['save_seconds'] = time.perf_counter() - save_start
                if telemetry:
                    telemetry_start = time.perf_counter()
                    capture_seconds = telemetry.seconds
                    telemetry.reference_source = report['measurement_source']
                    telemetry.finish_batch(quality['reference']['nll'], actions,
                        module_losses={name+'_planner': measured['planning'][name+'_loss']
                                       for name in ('goal', 'path')} if planning else None)
                    measured['telemetry_seconds'] = capture_seconds + time.perf_counter() - telemetry_start
                seconds = time.perf_counter() - begin
                row = dict(batch=batches, actions=actions, recent_tps=real_actions/seconds,
                           collection_seconds=collection_seconds, total_seconds=seconds,
                           quality=quality, sessions=[dict(game=s.game, seed=s.seed, steps=len(s.transitions),
                               boundaries=sum(t.reset_before for t in s.transitions), end=s.end_reason) for s in sessions], **measured)
                game_errors = [quality[g]['delta_error'] for g in factories
                               if g in quality and quality[g]['delta_error'] is not None]
                row['macro_delta_error'] = sum(game_errors)/len(game_errors) if game_errors else None
                report['click_map'] = click_map_report(engine)
                if report['click_map'] is not None:
                    row['click_sigma_pixels'] = report['click_map']['sigma_pixels']
                if engine.device.type == 'cuda':
                    row['peak_gpu_gib'] = torch.cuda.max_memory_allocated(engine.device)/1024**3
                report['history'] = (report['history'] + [row])[-200:]
                report['progress'] = dict(batches=batches, actions=actions, updates=learner.updates)
                if trajectories is not None:
                    report['planner_trajectory'] = trajectories.metadata
                    report['trajectory_error'] = trajectories.error
                write_report(report_path, report)
                print(json.dumps(dict(event='sequence_batch', **row), allow_nan=False), flush=True)
                completed_here += 1
                if args.max_batches is not None and completed_here >= args.max_batches:
                    break
            report['status'] = 'paused' if stop or action_budget is None or actions < action_budget else 'budget_complete'
            write_report(report_path, report)
        except BaseException:
            report['status'] = 'error'
            write_report(report_path, report)
            raise
        finally:
            close_factories(factories)
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--freeze-representation', action='store_true',
                        help='Persistently freeze encoder and decoder on predictor-phase resume, retaining learned state')
    parser.add_argument('--disable-undo', action='store_true',
                        help='Explicitly adopt allow_undo=false while preserving learned state')
    parser.add_argument('--upgrade-predictor', action='store_true',
                        help='Enable the rotation adapter or learnable click width, preserving weights and AdamW moments')
    parser.add_argument('--predictor-learning-rate', type=float,
                        help='Explicitly override only the predictor LR, preserving optimizer state on resume')
    parser.add_argument('--extend-curriculum', action='store_true',
                        help='Explicitly append training games/levels on resume, preserving weights and optimizers')
    parser.add_argument('--restrict-curriculum', action='store_true',
                        help='Explicitly remove training games/levels on resume, preserving weights, optimizers and phase')
    parser.add_argument('--adopt-training-phases', action='store_true',
                        help='Explicitly adopt sequence_control on a legacy checkpoint, retaining weights and optimizers')
    parser.add_argument('--adopt-planning', action='store_true',
                        help='Add goal/path heads once to a frozen representation sequence run')
    parser.add_argument('--adopt-exploration', action='store_true',
                        help='Adopt the first learned exploration head without resetting the world model')
    parser.add_argument('--adopt-game-weights', action='store_true',
                        help='Change only shuffled game frequencies on resume, preserving all learned state')
    parser.add_argument('--exploration-source', type=Path, help='Auxiliary bootstrap checkpoint for a new head only')
    parser.add_argument('--fix-click-sigma', action='store_true',
                        help='Explicitly return an existing adapter to fixed sigma=1.5, preserving all other learned state')
    parser.add_argument('--unlimited', action='store_true',
                        help='Ignore the total action budget; run until interrupted or --max-batches is reached')
    parser.add_argument('--optimization', action='store_true', help='Record bounded optimizer diagnostics')
    parser.add_argument('--optimization-every', type=int, default=8, help='Predictor updates between captures')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--max-batches', type=int)
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error('--cpu-threads must be positive')
    torch.set_num_threads(args.cpu_threads)
    run(args)


if __name__ == '__main__':
    main()
