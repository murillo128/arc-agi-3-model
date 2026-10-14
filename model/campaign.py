"""Bounded, resumable architecture comparisons on the two synthetic training games.

One sequential GPU job. Large artifacts are local, replay is ephemeral, and a
completed comparison is never inferred from a launched process. The source
checkpoint is immutable; unfinished training restarts from that same source,
while failed evaluations resume their saved trained agent.
"""
import argparse
from dataclasses import asdict, replace
import gc
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch

from . import agent_checkpoint, checkpoint
from .agent import Agent, AgentConfig
from .behavior_experiment import episode_seeds, evaluate, evaluation_summary, fit, save_heads
from .config import NetworkConfig, TrainConfig
from .data import Episode, Replay, random_action
from .environment import Factory, SYNTHETIC_LEVELS
from .learning import Engine, Learner
from .networks import WorldModel
from .planning import OutcomeValue, PlanConfig, Planner, fingerprint, outcome, returns
from .representation_gate import representation_gate
from .runner import merge_rollout_reports, validation
from .value_learning import RealRecord


WORLD_VARIANTS = ('world_reference', 'rollout2', 'rollout2_visual', 'local_decoder', 'staged_freeze', 'sigreg_only')
BEHAVIOR_VARIANTS = ('mc', 'double_q', 'click_mixed', 'actor_imitation', 'actor_imagined')
CONTROLS = dict(rollout2='world_reference', rollout2_visual='rollout2', local_decoder='rollout2', sigreg_only='staged_freeze',
                staged_freeze='rollout2', double_q='mc', click_mixed='mc',
                actor_imitation='mc', actor_imagined='actor_imitation')
ROOT = Path(__file__).resolve().parents[1]


def emit(event, **values):
    print(json.dumps({'event': event, **values}, allow_nan=False), flush=True)


def write_report(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def copy_artifact(source, destination):
    temporary = destination.with_suffix('.tmp')
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def quotas(total, pairs):
    if total < len(pairs) or len(set(pairs)) != len(pairs):
        raise ValueError('Budget must cover every distinct game/level')
    return {pair: total // len(pairs) + (i < total % len(pairs)) for i, pair in enumerate(pairs)}


def world_config(source, variant, total):
    if variant not in WORLD_VARIANTS:
        raise ValueError('Unknown world comparison')
    # All variants use the same balanced collection protocol and bounded replay.
    common = dict(retain_replay_across_levels=True, validation_episodes=2,
                  validation_steps=64, rollout_diagnostics=True, cache_frozen_latents=True,
                  rollout_weight=0., rollout_visual_weight=0.)
    if variant != 'world_reference':
        common['rollout_weight'] = .5
    if variant == 'rollout2_visual':
        common['rollout_visual_weight'] = .5
    elif variant == 'local_decoder':
        common['decoder_readout'] = 'local'
    elif variant == 'sigreg_only':
        common['variance_weight'] = 0.
    if variant in {'staged_freeze', 'sigreg_only'}:
        common.update(auto_representation=False, freeze_representation=False, representation_joint_steps=total // 2)
    return replace(source, **common)


def load_source(path, device, config=None):
    saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    net = WorldModel(NetworkConfig(**saved['network'])).to(device)
    net.load_state_dict(saved['weights'], strict=True)
    training = config or TrainConfig(**saved['training'])
    engine = Engine(net, training, compile_model=device == 'cuda',
                    optimizer_bits=saved.get('runtime', {}).get('optimizer_bits', 32) if device == 'cuda' else 32)
    return engine


def load_heads(path, engine, *, verify_world=True):
    saved = checkpoint.load(path)
    if verify_world and saved['source_fingerprint'] != fingerprint(engine.model):
        raise ValueError('Initial outcome heads belong to another world checkpoint')
    heads = OutcomeValue(engine.model.config.latent, engine.model.config.memory,
                         saved['weights']['spatial.0.weight'].shape[0]).to(engine.device)
    heads.load_compatible(saved['weights'])
    return heads


@torch.no_grad()
def replay_records(engine, archive, discount):
    """Re-encode only real observations with the final weights; no stale latents."""
    result = []
    was_training = engine.model.training
    engine.model.eval()
    action_budget = min(engine.config.max_episode_steps, 128)
    def cpu(value):
        return value.detach().to(device='cpu', dtype=torch.float16) if value is not None else None
    try:
        for items in archive.values():
            for episode, reason in items:
                if any(o.terminal for o in episode.observations[:-1]):
                    raise ValueError('Archived real episodes cannot continue after terminal observations')
                with engine.autocast():
                    states = engine.encode(episode.observations)
                # Adjacent records share immutable CPU states, including h.
                # The archive never owns these tensors or any device caches.
                cpu_states = [cpu(z) for z in states]
                events = [outcome(a, b) for a, b in zip(episode.observations, episode.observations[1:])]
                targets = returns(events, discount, end_reason=reason)
                h = torch.zeros(1, engine.model.config.memory, device=engine.device)
                cpu_h = cpu(h[0])
                z, cpu_z, geometry = None, None, None
                for index, action in enumerate(episode.actions):
                    before, after = episode.observations[index:index+2]
                    if states[index] is not None:
                        z, cpu_z, geometry = states[index], cpu_states[index], tuple(before.final.shape)
                    if z is None:
                        continue
                    with engine.autocast():
                        next_h, _ = engine.advance(z[None], h, engine.actions([action]))
                    # As in Agent.observe: an empty response preserves visual
                    # belief while advancing memory. It is not a new visual
                    # target; world learning excludes those transitions.
                    next_z = cpu_states[index+1] if states[index+1] is not None else cpu_z
                    next_geometry = tuple(after.final.shape) if after.final is not None else geometry
                    next_cpu_h = cpu(next_h[0])
                    result.append(RealRecord(cpu_z, cpu_h, torch.tensor(action.features()), events[index],
                        targets[index], geometry, action_budget-index, next_z, next_cpu_h,
                        next_geometry, before.available_actions, after.available_actions,
                        reason if index == len(episode.actions)-1 else 'ongoing'))
                    h, cpu_h = next_h, next_cpu_h
    finally:
        engine.model.train(was_training)
    return result


def evaluate_world(engine, factories, pairs, seed, episodes=2, *, diagnostics=True):
    old = engine.config
    engine.config = replace(old, validation_episodes=episodes, rollout_diagnostics=diagnostics)
    reports = []
    try:
        for game, level in pairs:
            # Disjoint odd environment seeds; identical schedules across variants.
            report = validation(engine, factories[game], [level], 600000 + seed,
                diagnostic_episode_limit=2, diagnostic_max_actions=16, stop_on_progress=True)[str(level)]
            reports.append(dict(game=game, level=level, **report))
    finally:
        engine.config = old
    transitions = sum(r['transitions'] for r in reports)
    # Per-game/level macro error: all benchmark pairs have equal weight.
    result = {'delta_error': float(np.mean([r['delta_error'] for r in reports])),
              'decoder_delta_error': float(np.mean([r['decoder_delta_error'] for r in reports])),
              'latent_delta_mse': float(np.mean([r['latent_delta_mse'] for r in reports])),
              'transitions': transitions, 'per_pair': reports}
    if diagnostics:
        result['rollouts'] = merge_rollout_reports([r['rollouts'] for r in reports])
        result['diagnostic_episodes'] = sum(r['rollout_diagnostic_episodes'] for r in reports)
        result['diagnostic_actions'] = sum(r['rollout_diagnostic_actions'] for r in reports)
    return result


def evaluate_cell_world(engine, factories, pairs, variant, seed, episodes, cells):
    """Reuse only MC's identical frozen-world measurement in this same campaign.

    Campaign resume fixes evaluation configuration and seed schedules. Behavior
    variants keep that world/config; their actions still require a fresh separate
    evaluation. World-training variants always collect their own measurement.
    """
    source = fingerprint(engine.model)
    reference = next((row for row in cells
                      if row['variant'] == 'mc' and row['seed'] == seed
                      and row.get('world_fingerprint') == source and row.get('world')), None)
    reuse = variant in BEHAVIOR_VARIANTS and variant != 'mc' and reference is not None
    measured = (reference['world'] if reuse else
                evaluate_world(engine, factories, pairs, seed, episodes=episodes))
    provenance = {'reused': reuse, 'source_variant': 'mc' if reuse else variant,
                  'source_seed': seed, 'world_fingerprint': source,
                  'episodes_per_pair': episodes, 'validation_seed': 600000 + seed}
    return measured, provenance


def imagination_gate(engine, factories, pairs, seed, reference_cell=None):
    """Conservative H2 admission, not a claim of sufficient planning accuracy.

    Every admitted pair needs real H2 anchors and finite diagnostics. Prediction
    must be no worse than keeping the initial latent, apart from roundoff. A
    passing macro average cannot conceal an unevaluated or worse individual pair.
    """
    source = fingerprint(engine.model)
    reuse = (reference_cell is not None and reference_cell.get('variant') == 'mc'
             and reference_cell.get('seed') == seed
             and reference_cell.get('world_fingerprint') == source
             and bool(reference_cell.get('world', {}).get('per_pair')))
    measured = (reference_cell['world'] if reuse else
                evaluate_world(engine, factories, pairs, seed, episodes=2, diagnostics=True))
    if fingerprint(engine.model) != source:
        raise RuntimeError('Imagination admission must not change world weights')
    index = {(row['game'], row['level']): row for row in measured['per_pair']}
    evidence = []
    tolerance = 1e-8
    for game, level in pairs:
        h2 = index.get((game, level), {}).get('rollouts', {}).get('2', {})
        samples = h2.get('samples', 0)
        metrics = {key: h2.get(key) for key in ('latent_mse', 'identity_latent_mse',
                   'shuffled_action_latent_mse', 'action_sensitivity_mse')}
        for name in ('contextual', 'local'):
            for branch in ('prediction', 'reconstruction'):
                metrics[f'{name}_{branch}_delta_error'] = h2.get('readouts', {}).get(name, {}).get(branch + '_delta_error')
        finite = all(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0
                     for value in metrics.values())
        reasons = []
        if type(samples) is not int or samples <= 0:
            reasons.append('missing_real_h2_samples')
        if not finite:
            reasons.append('invalid_or_nonfinite_h2_metrics')
        elif metrics['latent_mse'] > metrics['identity_latent_mse'] + tolerance:
            reasons.append('h2_worse_than_identity')
        # Keep failed numerical diagnostics JSON-safe and distinguish them from
        # zero error. These are the first two diagnostic seeds of validation().
        safe_metrics = {key: float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None
                        for key, value in metrics.items()}
        validation_seed = 600000 + seed
        evaluation_seeds = [2 * ((validation_seed * 100003 + level * 1009 + i) % (2 ** 30)) + 1
                            for i in range(2)]
        evidence.append(dict(game=game, level=level, samples=samples if type(samples) is int else 0,
                             passed=not reasons, reasons=reasons, evaluation_seeds=evaluation_seeds, **safe_metrics))
    return {'passed': bool(evidence) and all(row['passed'] for row in evidence),
            'horizon': 2, 'criterion': 'finite_real_h2_no_worse_than_identity_per_pair',
            'absolute_tolerance': tolerance, 'world_fingerprint': source, 'training_seed': seed,
            'validation_seed': 600000 + seed, 'diagnostic_episode_limit': 2, 'diagnostic_max_actions': 16,
            'evidence_source': 'mc_development' if reuse else 'fresh_source_world_diagnostic',
            'per_pair': evidence}


def calibration_coverage(archive):
    """Actual collected episodes, never imagined successes or oversampled labels."""
    rows = []
    for (game, level), items in archive.items():
        rows.append({'game': game, 'level': level, 'episodes': len(items),
            'actions': sum(len(ep.actions) for ep, _ in items),
            'successes': sum(reason == 'progress' for _, reason in items),
            'failures': sum(reason == 'game_over' for _, reason in items),
            'budget_ends': sum(reason == 'budget' for _, reason in items),
            'censored_episodes': sum(reason == 'interrupted' for _, reason in items)})
    return {'sampling': 'all_collected_real_episodes', 'per_pair': rows,
            **{key: sum(row[key] for row in rows) for key in
               ('episodes', 'actions', 'successes', 'failures', 'budget_ends', 'censored_episodes')}}


def train_world(engine, factories, pairs, total_actions, seed, variant, callback=None, *,
                calibration_bytes=512 * 1024 * 1024):
    c = engine.config
    learner = Learner(engine)
    rng = random.Random(seed)
    sampling_rng = random.Random(seed ^ 0x5245504C)
    action_budget = min(c.max_episode_steps, 128)
    budget, counts = quotas(total_actions, pairs), dict.fromkeys(pairs, 0)
    replay = Replay(c.replay_episodes, c.replay_bytes)
    # Calibration needs the entire bounded collection, including early successes.
    # Raw CPU observations have a separate explicit budget; they are not replay
    # caches and are never persisted. Fail visibly instead of dropping labels.
    archive = {pair: [] for pair in pairs}
    archive_bytes = 0
    actions = updates = episodes = 0
    joint_updates = frozen_updates = 0
    started, next_check, gate = time.monotonic(), min(4096, total_actions), None
    while actions < total_actions:
        pair = min((p for p in pairs if counts[p] < budget[p]), key=lambda p: counts[p] / budget[p])
        game, level = pair
        env_seed = 2_000_000 + 200_000 * seed + 2 * episodes
        env = factories[game].make(env_seed, level)
        observation = env.observation
        if observation.terminal:
            raise ValueError('A campaign attempt must start from a nonterminal observation')
        episode = Episode([observation])
        real_observations, real_actions = [observation], []
        replay.add(episode)
        reason = 'interrupted'
        pending_update = False
        loss = None
        for position in range(min(action_budget, budget[pair] - counts[pair])):
            if observation.terminal:
                break
            if c.representation_joint_steps:
                frozen = actions >= c.representation_joint_steps
                if frozen != engine.config.freeze_representation:
                    learner.set_representation_frozen(frozen)
            action = random_action(observation, rng)
            after = env.step(action)
            real_observations.append(after)
            real_actions.append(action)
            if not after.starts_new_level(observation):
                episode.append(action, after)
                replay.prepare(episode, engine.device, engine.model.config.patch,
                               torch.bfloat16 if engine.amp else torch.float32)
                pending_update = True
            actions += 1
            counts[pair] += 1
            event = outcome(observation, after)
            ended = bool(event) or position + 1 == action_budget or counts[pair] == budget[pair]
            updated = False
            if pending_update and (actions % c.train_every == 0 or ended):
                batch = replay.sample(sampling_rng, c.batch_episodes, 2, latest=episode, random_prefix=True)
                for _ in range(c.updates_per_train):
                    current_loss = learner.update(batch)
                    if current_loss is not None:
                        loss = current_loss
                        updates += 1
                        if engine.config.freeze_representation:
                            frozen_updates += 1
                        else:
                            joint_updates += 1
                        updated = True
                del batch
                pending_update = False
            if updated or ended or actions == 1:
                emit('campaign_progress', variant=variant, seed=seed, stage='world', actions=actions,
                     total_actions=total_actions, game=game, level=level, updates=updates,
                     joint_updates=joint_updates, frozen_updates=frozen_updates,
                     actions_per_second=actions / (time.monotonic()-started), loss=loss)
            observation = after
            if ended:
                reason = ('progress' if event == 1 else 'game_over' if event == 2 else
                          'budget' if position + 1 == action_budget else 'interrupted')
                break
        # The archive owns raw observations only, never orphaned GPU caches.
        collected = Episode(real_observations, real_actions)
        archive_bytes += collected.nbytes
        if archive_bytes > calibration_bytes:
            raise MemoryError('Real calibration archive exceeds calibration_bytes; reduce collection or raise its explicit RAM limit')
        archive[pair].append((collected, reason))
        episodes += 1
        replay.trim()
        if actions >= next_check and actions < total_actions:
            reports = evaluate_world(engine, factories, pairs, seed, diagnostics=False)
            # Keep the configured evidence thresholds; insufficient coverage holds the gate.
            gate_reports = {f'{r["game"]}:{r["level"]}': r for r in reports['per_pair']}
            gate = representation_gate(engine.config, gate_reports, list(gate_reports), gate)
            if gate['changed']:
                learner.set_representation_frozen(gate['freeze_representation'])
            emit('campaign_evaluation', variant=variant, seed=seed, actions=actions,
                 split='intermediate', world=reports, representation_gate=gate)
            if callback:
                callback(actions)
            next_check += 4096
    # Drop optimizer/GPU feature caches before final frozen evaluation.
    for item in replay.episodes:
        item.prepared.clear()
        item.latent_cache.clear()
    del learner, replay
    gc.collect()
    if engine.device.type == 'cuda':
        torch.cuda.empty_cache()
    records = replay_records(engine, archive, .95)
    coverage = calibration_coverage(archive)
    coverage.update(raw_bytes=archive_bytes, records=len(records),
                    positive_records=sum(row.target is not None and row.target > 0 for row in records),
                    censored_records=sum(row.target is None for row in records))
    emit('campaign_calibration_data', variant=variant, seed=seed, calibration=coverage)
    return records, {'actions': actions, 'updates': updates, 'episodes': episodes,
                     'joint_updates': joint_updates, 'frozen_updates': frozen_updates,
                     'seconds': time.monotonic()-started, 'calibration': coverage,
                     'per_pair': [{'game': g, 'level': l, 'actions': counts[(g, l)]} for g, l in pairs]}


def evaluate_behavior(engine, planner, factories, pairs, seed, episodes, split):
    results, started = [], time.monotonic()
    for pair_index, (game, level) in enumerate(pairs, 1):
        seeds = episode_seeds(split, episodes, offset=seed * 10000)
        report = evaluate(engine, planner, factories[game], level, seeds, 128,
                          'value_mpc', split=split, role='campaign', game=game,
                          pair_index=pair_index, pair_count=len(pairs))
        results.extend(dict(r, game=game, training_seed=seed) for r in report['results'])
    return evaluation_summary(results, time.monotonic()-started)


def compare(reference, candidate, seed=13):
    """Paired stratified bootstrap; preserves equal game/level/seed weights."""
    def index(reports):
        return {(r['training_seed'], r['game'], r['level'], r['seed']): r
                for report in reports for r in report['behavior']['results']}
    old, new = index(reference), index(candidate)
    if old.keys() != new.keys() or not old:
        raise ValueError('Comparison requires identical nonempty evaluation pairs')
    keys = sorted(old)
    blocks = {}
    for key in keys:
        blocks.setdefault(key[:3], []).append(float(new[key]['success']) - float(old[key]['success']))
    rng = np.random.default_rng(seed)
    boot = sum(rng.choice(values, (2000, len(values))).mean(1) for values in blocks.values()) / len(blocks)
    interval = np.quantile(boot, [.025, .975]).tolist()
    delta = float(np.mean([np.mean(v) for v in blocks.values()]))
    seconds_old = sum(r['training']['seconds'] for r in reference)
    seconds_new = sum(r['training']['seconds'] for r in candidate)
    noninferior = interval[0] >= -.02
    # Local readout keeps legacy tower weights for reversible comparison; it
    # does not yet qualify as removal of a whole saved module.
    simplified = candidate[0]['variant'] == 'sigreg_only'
    old_wins = sum(r['behavior'].get('wins', sum(x['success'] for x in r['behavior']['results'])) for r in reference)
    new_wins = sum(r['behavior'].get('wins', sum(x['success'] for x in r['behavior']['results'])) for r in candidate)
    action_ratio = ((sum(r['behavior']['actions'] for r in candidate) / new_wins) /
                    (sum(r['behavior']['actions'] for r in reference) / old_wins)) if min(old_wins, new_wins) else None
    latency_ratio = sum(r['behavior']['decision_ms'] for r in candidate) / max(1e-9, sum(r['behavior']['decision_ms'] for r in reference))
    # Sequential JIT warmup/cache effects are reported, never mistaken for an
    # isolated speed comparison. Actions are unaffected by compilation time.
    comparable_costs = all(r.get('warm_cost_comparison', False) for r in reference + candidate)
    efficient = ((action_ratio is not None and action_ratio <= .9) or
                 (comparable_costs and (seconds_new <= .8 * seconds_old or latency_ratio <= .8)))
    promote = interval[0] > 0 or (noninferior and (simplified or efficient))
    return {'success_delta': delta, 'paired95': interval, 'noninferior': noninferior,
            'training_seconds_ratio': seconds_new/max(seconds_old, 1e-9),
            'actions_per_success_ratio': action_ratio, 'decision_latency_ratio': latency_ratio,
            'warm_cost_comparison': comparable_costs,
            'simplification': simplified, 'promote': promote,
            'reason': 'success' if interval[0] > 0 else 'simplification' if noninferior and simplified
                      else 'measured_cost' if noninferior and efficient else 'inconclusive_or_inferior'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--heads', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variants', nargs='+', choices=WORLD_VARIANTS+BEHAVIOR_VARIANTS,
                        default=list(WORLD_VARIANTS+BEHAVIOR_VARIANTS))
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--actions', type=int, default=16384)
    parser.add_argument('--dev-episodes', type=int, default=32)
    parser.add_argument('--final-episodes', type=int, default=128)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--games-dir', type=Path, default=ROOT/'games')
    parser.add_argument('--calibration-bytes', type=int, default=512 * 1024 * 1024,
                        help='Maximum raw CPU observation bytes retained for complete head calibration')
    parser.add_argument('--pilot', action='store_true', help='Diagnostic only: no promotion or final confirmation')
    args = parser.parse_args(argv)
    if (min(args.actions, args.dev_episodes, args.final_episodes, args.calibration_bytes) < 1 or len(set(args.seeds)) != len(args.seeds)
            or any(seed not in (0, 1, 2) for seed in args.seeds)):
        parser.error('Positive budgets and distinct seeds in {0, 1, 2} required')
    if not args.pilot and (args.actions != 16384 or args.seeds != [0, 1, 2] or args.dev_episodes != 32 or args.final_episodes != 128):
        parser.error('Reduced protocols require explicit --pilot and cannot promote weights')
    args.output.mkdir(parents=True, exist_ok=True)
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
    report_path = args.output/'report.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else {
        'config': config, 'status': 'running', 'cells': [], 'comparisons': {}, 'selected': None,
        'processed_variants': []}
    if report['config'] != config:
        raise ValueError('Resume requires the same configuration; use a new output directory')
    if report['status'] in {'complete', 'pilot_complete'}:
        emit('campaign_finished', status=report['status'], selected=report['selected'])
        return
    if not args.pilot:
        for variant in args.variants:
            if variant in CONTROLS and (CONTROLS[variant] not in args.variants or
                                       args.variants.index(CONTROLS[variant]) > args.variants.index(variant)):
                parser.error(f'{variant} requires earlier control {CONTROLS[variant]}')
    report['status'] = 'running'
    report.pop('error', None)
    write_report(report_path, report)
    emit('campaign_started', config=config)
    factories = {game: Factory(game, args.games_dir) for game in SYNTHETIC_LEVELS}
    pairs = [(game, level) for game, count in SYNTHETIC_LEVELS.items() for level in range(1, count+1)]
    # Preserve immutable inode references without duplicating multi-GB artifacts.
    source_path, heads_path = args.output/'source-world.pt', args.output/'source-heads.pt'
    for source, target in ((args.checkpoint, source_path), (args.heads, heads_path)):
        if not target.exists():
            os.link(source, target)
    source = torch.load(source_path, map_location='cpu', weights_only=True, mmap=True)
    source_config = TrainConfig(**source['training'])
    del source
    completed = {(r['variant'], r['seed']) for r in report['cells'] + report.get('deferred_cells', [])}
    candidate_path, selected_path, reference_path = (args.output/f'{name}.pt' for name in ('candidate', 'selected', 'reference'))
    try:
        for variant in args.variants:
            if variant in report['processed_variants']:
                continue
            for seed in args.seeds:
                if (variant, seed) in completed:
                    continue
                # Each cell owns different module instances / gradient modes.
                # Discard their Dynamo guards between cells; Inductor's disk
                # kernels remain reusable. Never fall back to eager on failure.
                torch.compiler.reset()
                torch.manual_seed(seed)
                random.seed(seed)
                emit('campaign_phase', variant=variant, seed=seed, stage='initializing',
                     actions=0, total_actions=args.actions)
                pending_path = args.output / 'pending-agent.pt'
                pending = report.get('pending_evaluation')
                if pending and (pending['variant'], pending['seed']) == (variant, seed):
                    resumed = agent_checkpoint.load(pending_path, device=args.device)
                    engine, heads, planner = resumed.engine, resumed.heads, resumed.planner
                    proposal, training_report = resumed.proposal_policy, pending['training']
                    del resumed
                    emit('campaign_resumed_evaluation', variant=variant, seed=seed, actions=args.actions)
                else:
                    training = world_config(source_config, variant, args.actions) if variant in WORLD_VARIANTS else source_config
                    engine = load_source(source_path, args.device, training)
                    initial_heads_path = args.output/f'mc-heads-{seed}.pt' if variant.startswith('actor_') else heads_path
                    if variant.startswith('actor_') and not initial_heads_path.exists():
                        raise ValueError('Actor comparisons require completed MC calibration for the same seed')
                    heads = load_heads(initial_heads_path, engine)
                    gate = None
                    if variant == 'actor_imagined':
                        reference_cell = next((r for r in report['cells']
                                               if r['variant'] == 'mc' and r['seed'] == seed), None)
                        gate = imagination_gate(engine, factories, pairs, seed, reference_cell)
                        emit('campaign_imagination_gate', variant=variant, seed=seed, **gate)
                        if not gate['passed']:
                            report.setdefault('deferred_cells', []).append({
                                'variant': variant, 'seed': seed, 'status': 'deferred',
                                'reason': 'source_world_h2_not_admitted', 'fallback': 'actor_imitation',
                                'training': {'actions': 0, 'updates': 0}, 'imagination_gate': gate})
                            write_report(report_path, report)
                            del engine, heads
                            gc.collect()
                            if args.device == 'cuda':
                                torch.cuda.empty_cache()
                            continue
                    if args.device == 'cuda':
                        torch.cuda.reset_peak_memory_stats()
                    started = time.monotonic()
                    proposal = None
                    if variant in WORLD_VARIANTS:
                        records, training_report = train_world(engine, factories, pairs, args.actions, seed, variant,
                                                              calibration_bytes=args.calibration_bytes)
                        emit('campaign_phase', variant=variant, seed=seed, stage='head_calibration',
                             actions=args.actions, total_actions=args.actions)
                        fit(engine, heads, records, 12, 32, 3e-4, seed, compiled=args.device == 'cuda')
                        del records
                        planner = Planner(engine, heads, PlanConfig(clicks=16), compile_heads=args.device == 'cuda')
                    else:
                        from .campaign_behavior import train_behavior
                        def check_behavior(actions, current_planner, current_proposal, current_report):
                            if actions >= args.actions:
                                return
                            measured = evaluate_behavior(engine, current_planner, factories, pairs, seed, 2, 'dev')
                            emit('campaign_evaluation', variant=variant, seed=seed, actions=actions,
                                 split='intermediate', behavior=measured)
                        planner, proposal, training_report = train_behavior(engine, heads, PlanConfig(clicks=16), factories,
                            pairs, total_actions=args.actions, seed=seed, variant=variant, emit=emit,
                            check_callback=check_behavior)
                        if variant == 'mc':
                            save_heads(args.output/f'mc-heads-{seed}.pt', heads, fingerprint(engine.model), planner.config,
                                       {'config': {'scope': 'synthetic_campaign', 'seed': seed}})
                    training_report['seconds'] = time.monotonic()-started
                    if gate is not None:
                        training_report['imagination_gate'] = gate
                    training_report['stage_seconds'] = training_report['seconds']
                    training_report['cumulative_actions'] = args.actions
                    if variant.startswith('actor_'):
                        calibration = next(r for r in report['cells'] if r['variant'] == 'mc' and r['seed'] == seed)
                        training_report['seconds'] += calibration['training']['seconds']
                        training_report['cumulative_actions'] += calibration['training']['actions']
                    # Save every completed training cell BEFORE fallible evaluation.
                    # This is a recovery snapshot, not a selected/promoted candidate.
                    agent = Agent(engine, heads, AgentConfig(mode='value_mpc', planner=planner.config),
                                  seed=seed, proposal_policy=proposal)
                    agent_checkpoint.save(pending_path, agent)
                    del agent
                    report['pending_evaluation'] = {'variant': variant, 'seed': seed,
                                                    'training': training_report}
                    write_report(report_path, report)
                emit('campaign_phase', variant=variant, seed=seed, stage='world_evaluation',
                     actions=args.actions, total_actions=args.actions)
                world_report, world_evaluation = evaluate_cell_world(
                    engine, factories, pairs, variant, seed, args.dev_episodes, report['cells'])
                emit('campaign_phase', variant=variant, seed=seed, stage='behavior_evaluation',
                     actions=args.actions, total_actions=args.actions)
                behavior = evaluate_behavior(engine, planner, factories, pairs, seed, args.dev_episodes, 'dev')
                cell = {'variant': variant, 'seed': seed, 'training': training_report, 'world': world_report,
                        'world_evaluation': world_evaluation,
                        'behavior': behavior, 'world_fingerprint': world_evaluation['world_fingerprint'],
                        'gpu_peak_bytes': torch.cuda.max_memory_allocated() if args.device == 'cuda' else None}
                if seed == args.seeds[0]:
                    # Predetermined representative seed, never the best-looking seed.
                    copy_artifact(pending_path, candidate_path)
                report['cells'].append(cell)
                report.pop('pending_evaluation', None)
                write_report(report_path, report)
                pending_path.unlink(missing_ok=True)
                emit('campaign_evaluation', variant=variant, seed=seed, actions=args.actions,
                     split='dev', world=world_report, behavior=behavior)
                del engine, heads, planner, proposal
                gc.collect()
                if args.device == 'cuda':
                    torch.cuda.empty_cache()
            cells = [r for r in report['cells'] if r['variant'] == variant]
            deferred = [r for r in report.get('deferred_cells', []) if r['variant'] == variant]
            if deferred:
                report.setdefault('deferred_variants', {})[variant] = {
                    'reason': 'incomplete_h2_admission', 'seeds': [r['seed'] for r in deferred],
                    'fallback': 'actor_imitation', 'status': 'deferred'}
                emit('campaign_variant_deferred', variant=variant, **report['deferred_variants'][variant])
            if not args.pilot and not deferred:
                baseline = report['selected']
                if baseline is None:
                    report['selected'] = variant
                    copy_artifact(candidate_path, selected_path)
                    if not reference_path.exists():
                        os.link(candidate_path, reference_path)
                else:
                    control = CONTROLS.get(variant)
                    if control is not None:
                        controlled = compare([r for r in report['cells'] if r['variant'] == control], cells)
                        report.setdefault('controlled_comparisons', {})[variant] = dict(control=control, **controlled)
                    reference = [r for r in report['cells'] if r['variant'] == baseline]
                    comparison = compare(reference, cells)
                    report['comparisons'][variant] = comparison
                    if comparison['promote']:
                        report['selected'] = variant
                        copy_artifact(candidate_path, selected_path)
            report['processed_variants'].append(variant)
            write_report(report_path, report)
        # Final confirmation is held out of selection and happens only once.
        if not args.pilot and 'final' not in report:
            chosen = agent_checkpoint.load(selected_path, device=args.device)
            selected_final = evaluate_behavior(chosen.engine, chosen.planner, factories, pairs, 0, args.final_episodes, 'final')
            del chosen
            gc.collect()
            if args.device == 'cuda':
                torch.cuda.empty_cache()
            original = agent_checkpoint.load(reference_path, device=args.device)
            report['final'] = {
                'selected': selected_final,
                'reference': evaluate_behavior(original.engine, original.planner, factories, pairs, 0, args.final_episodes, 'final'),
                'representative_training_seed': 0}
        report['status'] = 'pilot_complete' if args.pilot else 'complete'
        write_report(report_path, report)
        emit('campaign_finished', status=report['status'], selected=report['selected'])
    except BaseException as error:
        report['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        report['error'] = f'{type(error).__name__}: {error}'
        write_report(report_path, report)
        emit('campaign_error', status=report['status'], error=report['error'])
        raise


if __name__ == '__main__':
    main()
