"""Single-game level curriculum and incremental representation/dynamics/path learning.

--init-config initializes every module; --source transfers world weights.
--resume restores a saved curriculum and all weights. Only actual SDK
transitions train dynamics; only finished real goal segments train PathValue.
The environment's teacher stays outside the portable Agent.
SIGUSR1 requests an atomic checkpoint at the next episode/evaluation boundary
without stopping training.
"""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch

from . import agent_checkpoint, checkpoint
from .agent import Agent, AgentConfig
from .behavior_experiment import episode_seeds
from .campaign import write_report
from .config import NetworkConfig, TrainConfig
from .data import Episode, random_action
from .environment import Factory, SYNTHETIC_LEVELS
from .learning import Engine
from .networks import WorldModel
from .online_learning import OnlinePredictor, OnlineRepresentation
from .online_replay import PredictorReplay
from .curriculum import Curriculum, CurriculumConfig
from .metrics import delta_counts
from .path_planning import PathConfig, PathValue, PathLearner, real_states
from .planning import GoalValue
from .prediction_samples import PredictionSamples
from .training_telemetry import PlannerTrajectory, RecentThroughput
from .latent_diagnostics import LatentProbe, summarize_latents

def error_counts(before, actual, predicted):
    wrong, union, changed = delta_counts(before, actual, predicted)
    real, guessed = actual != before, predicted != before
    return dict(wrong=wrong, union=union, changed=changed,
                invented=int((~real & guessed).sum()), missed=int((real & ~guessed).sum()),
                color=int((real & guessed & (actual != predicted)).sum()),
                correct=int((real & (actual == predicted)).sum()))


class QualityProbe:
    """Fixed random real trajectories; one-step and fully imagined H-step error."""
    def __init__(self, engine, horizon, initial):
        self.engine, self.horizon = engine, horizon
        self.states = [initial]
        self.anchor = initial.z.detach().float().cpu().clone()
        self.latent = LatentProbe(engine.model.config.patch)
        self.latent.sample(initial)
        self.counts = {name: dict(wrong=0, union=0, samples=0, changed=0)
                       for name in ('reconstruction', 'prediction', 'rollout')}

    def add(self, name, counts):
        if counts is not None:
            for key in ('wrong', 'union'):
                self.counts[name][key] += counts[key]
            self.counts[name]['samples'] += 1
            self.counts[name]['changed'] += int(counts['changed'] > 0)

    @torch.no_grad()
    def observe(self, episode, previous, current, counts):
        self.states.append(current)
        self.add('prediction', counts)
        if counts is None or current.z is None:
            return
        e, t = self.engine, len(episode.actions)
        actual, reference = episode.observations[-1].final, episode.observations[-2].final
        self.latent.sample(current)
        self.latent.transition(previous, current, reference, actual)
        with e.autocast():
            reconstructed = e.decode(current.z[None], *current.geometry).argmax(-1)[0].cpu().numpy()
        self.add('reconstruction', error_counts(reference, actual, reconstructed))
        # Disjoint windows bound evaluation cost, with an observed starting
        # state/memory and then ONLY predicted states/memories until the endpoint.
        if t >= self.horizon and t % self.horizon == 0:
            anchor = self.states[t - self.horizon]
            z, h = anchor.z[None], anchor.h
            with e.autocast():
                for action in episode.actions[t - self.horizon:t]:
                    z, h, _ = e.predict(z, h, e.actions([action]), *anchor.geometry)
                image = e.decode(z, *anchor.geometry).argmax(-1)[0].cpu().numpy()
            self.add('rollout', error_counts(episode.observations[t - self.horizon].final, actual, image))

    def result(self):
        return dict(counts=self.counts, anchor=self.anchor, latent=self.latent)


def summarize_quality(summaries, previous_anchors=None):
    """No missing evidence is fabricated as zero. Latent drift uses fixed boards."""
    values = [s['quality'] for s in summaries if s.get('quality') is not None]
    result = {'completed_episodes': sum(s.get('reason') in ('success', 'game_over') for s in summaries)}
    for name in ('reconstruction', 'prediction', 'rollout'):
        counts = {key: sum(v['counts'][name][key] for v in values)
                  for key in ('wrong', 'union', 'samples', 'changed')}
        result[name + '_error'] = counts['wrong'] / max(1, counts['union']) if counts['samples'] else None
        if name != 'reconstruction':
            prefix = 'rollout_' if name == 'rollout' else ''
            result[prefix + 'samples'] = counts['samples']
            result[prefix + 'changed'] = counts['changed']
    anchors = [v['anchor'] for v in values]
    result['latent_drift'] = None
    result['latent'] = summarize_latents([v['latent'] for v in values if v.get('latent') is not None])
    if previous_anchors is not None and len(previous_anchors) == len(anchors) and anchors:
        numerator = sum(float((a - b).square().sum()) for a, b in zip(anchors, previous_anchors))
        denominator = sum(float(b.square().sum()) for b in previous_anchors)
        result['latent_drift'] = (numerator / max(denominator, 1e-12)) ** .5
    return result, anchors


def play(agent, factory, level, seed, steps, *, world=None, learner=None, epsilon=0.,
         random_policy=False, progress=None, action_limit=None, representation=None, quality_horizon=None,
         sample=None, predictor_replay=None, trajectory=None, trajectory_step=None):
    """A synthetic teacher supplies only one visual goal to the shared agent.

    Evaluation calls omit all learners. Goal predicates/distances are used only
    by this trainer, never by action selection or hypothetical-state scoring.
    """
    if quality_horizon is not None and any(x is not None for x in (world, learner, representation, predictor_replay)):
        raise ValueError('Quality evaluation must not update weights')
    env = factory.make(seed, level)
    teacher = factory.teacher(env)
    goals = teacher.training_subgoals()
    agent.reset(seed=seed)
    agent.observe(env.observation)
    episode = Episode([env.observation])
    rng = random.Random(seed)
    goal_index, start = 0, 0
    goal = goals[goal_index]
    minimum = teacher.training_goal_distance(goal)
    if minimum is None or minimum < 1:
        raise ValueError('Initial synthetic subgoal must be reachable and unfinished')
    agent.set_goal(goal.frame, goal.mask)
    segments, totals, records = [], dict.fromkeys(('wrong', 'union', 'changed', 'invented', 'missed', 'color', 'correct'), 0), []
    quality = QualityProbe(agent.engine, quality_horizon, agent.state) if quality_horizon else None
    decision_ms = 0.
    end_reason = 'budget'
    count = min(steps, action_limit if action_limit is not None else steps)
    for position in range(count):
        active_goal_index = goal_index
        before, state = env.observation, agent.state
        begin = time.monotonic()
        exploring = random_policy or rng.random() < epsilon
        if exploring:
            action = random_action(before, rng)
            agent.pending_action = action
        else:
            action = agent.act(diagnostics=True)
        decision_ms += (time.monotonic() - begin) * 1000
        prediction = agent.engine.prediction(state, action, before.final)
        if trajectory_step is not None and not random_policy:
            trajectory_step(agent, action, exploring)
        after = env.step(action)
        episode.append(action, after)
        agent.observe(after, action)
        boundary = after.starts_new_level(before)
        counts, error, losses = None, None, None
        if not boundary and prediction is not None and after.final is not None:
            counts = error_counts(before.final, after.final, prediction[0])
            error = counts['wrong'] / max(1, counts['union'])
            for key in totals:
                totals[key] += counts[key]
        capture = None
        if counts is not None and sample is not None:
            def capture(reconstruction):
                sample(before.final, after.final, prediction[0], game=factory.game_id, level=level,
                       episode_seed=seed, episode_action=position + 1,
                       reconstruction=reconstruction,
                       action=dict(id=action.id, data=dict(action.data)))
        if quality is not None:
            quality.observe(episode, state, agent.state, counts)
        representation_loss = None
        if not boundary and representation is not None and not representation.frozen:
            representation_loss = representation.update(episode.observations[-2:], preview=capture)
        elif capture is not None:
            # Frozen representation/evaluation has no update to reuse. Decode
            # the same previous latent, without touching weights or random state.
            with torch.no_grad(), agent.engine.autocast():
                reconstructed = agent.engine.decode(state.z[None], *before.final.shape).argmax(-1)[0]
            capture(reconstructed.to(torch.uint8).cpu().numpy())
        refreshed = None
        if representation_loss is not None:
            # Both endpoints and the current goal must use the UPDATED encoder.
            refreshed = real_states(agent.engine, episode)
            agent.state = refreshed[-1]
            agent.set_goal(goal.frame, goal.mask)
        if not boundary and world is not None and error is not None:
            refreshed = refreshed or real_states(agent.engine, episode)
            rollout = None
            if len(episode.actions) >= world.horizon:
                anchor = len(episode.actions) - world.horizon
                rollout = (refreshed[anchor], episode.actions[anchor:], episode.observations[anchor].final)
            if predictor_replay is not None and predictor_replay.record(episode):
                losses = world.update_batch(predictor_replay.examples(predictor_replay.sample()))
            else:
                losses = world.update(refreshed[-2], action, before.final, after, refreshed[-1].z, error,
                                      rollout=rollout)
            if losses is not None:
                agent.state = real_states(agent.engine, episode)[-1]
                agent.planner.clear_paths()
        level_success = after.state == 'WIN' or boundary
        success = not boundary and teacher.training_goal_reached(goal)
        failure = after.state == 'GAME_OVER' or (position + 1 == steps and not success)
        path_loss = None
        finished_segment = None
        if success or failure:
            length = len(episode.actions) - start
            if learner is not None and not boundary:
                path_loss = learner.update(episode, start, success, steps)
            segment = dict(goal=goal.id, success=bool(success), actions=length, minimum_actions=minimum,
                excess_actions=length - minimum, reward=-(length - minimum) - (0 if success else steps),
                loss=path_loss)
            segments.append(segment)
            finished_segment = segment
            if success:
                goal_index += 1
                # A real action can satisfy several ordered goals at once (or
                # satisfy a later one first). Do not require a fictitious action
                # or lose the remaining completions when native WIN arrives.
                while goal_index < len(goals) and teacher.training_goal_reached(goals[goal_index]):
                    reached = goals[goal_index]
                    agent.set_goal(reached.frame, reached.mask)
                    extra_loss = learner.update(episode, len(episode.actions), True, steps) if learner else None
                    segments.append(dict(goal=reached.id, success=True, actions=0, minimum_actions=0,
                        excess_actions=0, reward=0, loss=extra_loss))
                    goal_index += 1
        row = dict(action=position + 1, action_source='exploration' if exploring else 'planner',
            goal_index=active_goal_index + 1, goals=len(goals),
            goal=goal.label, minimum_actions=minimum, goal_actions=len(episode.actions) - start,
            delta_error=error, counts=counts, predictor_loss=losses, path_loss=path_loss,
            representation_loss=representation_loss,
            predictor_updated=losses is not None, segment=finished_segment,
            epsilon=epsilon, decision_ms=decision_ms / (position + 1),
            observation=after.final.tolist() if after.final is not None else None,
            goal_frame=goal.frame.tolist(), goal_mask=goal.mask.astype(int).tolist())
        records.append({k: v for k, v in row.items() if k not in ('observation', 'goal_frame', 'goal_mask')})
        if predictor_replay is not None:
            row['predictor_replay'] = dict(predictor_replay.status(),
                                          actual_batch_size=(losses or {}).get('batch_size', 0))
        if progress:
            progress(row)
        if failure or after.terminal or boundary or goal_index == len(goals):
            end_reason = 'success' if level_success else 'game_over' if after.state == 'GAME_OVER' else 'budget'
            break
        if success:
            goal, start = goals[goal_index], len(episode.actions)
            minimum = teacher.training_goal_distance(goal)
            if minimum is None:
                end_reason = 'unreachable_subgoal'
                break
            agent.set_goal(goal.frame, goal.mask)
    else:
        if count < steps:
            end_reason = 'interrupted'
    result = dict(success=end_reason == 'success', reason=end_reason, actions=len(episode.actions),
                segments=segments, counts=totals, decision_ms=decision_ms / max(1, len(episode.actions)), records=records,
                quality=None if quality is None else quality.result())
    if trajectory is not None and not random_policy:
        trajectory(episode, records, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    initialization = parser.add_mutually_exclusive_group(required=True)
    initialization.add_argument('--source', type=Path, help='Transfer pretrained world weights')
    initialization.add_argument('--init-config', type=Path,
                                help='Initialize all weights from scratch using this network/training JSON')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--game', choices=tuple(SYNTHETIC_LEVELS), default='sc01-v1',
                        help='One game per campaign; only its unlocked levels are practiced and evaluated')
    parser.add_argument('--resume', action='store_true',
                        help='Continue latest.pt weights/curriculum/counters; optimizer moments restart')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--actions', type=int, default=262144)
    parser.add_argument('--steps', type=int, default=128)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--evaluate-every', type=int, default=256)
    parser.add_argument('--update-evaluation-interval', action='store_true',
                        help='With --resume, explicitly change the evaluation interval and reset qualification streaks')
    parser.add_argument('--evaluation-episodes', type=int, default=8)
    parser.add_argument('--checkpoint-every', type=int, default=2048)
    parser.add_argument('--representation-error', type=float, default=.10)
    parser.add_argument('--update-representation-threshold', action='store_true',
                        help='With --resume, explicitly change the reconstruction threshold and reset qualification streaks')
    parser.add_argument('--predictor-error', type=float, default=.10)
    parser.add_argument('--latent-drift', type=float, default=.15)
    parser.add_argument('--update-latent-drift-threshold', action='store_true',
                        help='With --resume, explicitly change the latent drift threshold and reset qualification streaks')
    parser.add_argument('--success', type=float, default=.80)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--learning-rate', type=float, default=3e-5)
    parser.add_argument('--representation-learning-rate', type=float, default=3e-6)
    parser.add_argument('--encoder-learning-rate-after-predictor', type=float,
                        help='Lower encoder LR after first predictor admission; decoder unchanged. Resume inherits checkpoint.')
    parser.add_argument('--migrate-path-value', action='store_true', help='Upgrade old path evaluators to normalized success-first outputs; preserve world weights')
    parser.add_argument('--prediction-delta-loss', choices=('balanced', 'union', 'focal', 'recall_focal'),
                        help='Override predictor NLL, including on resume; otherwise inherit checkpoint/config')
    parser.add_argument('--freeze-encoder', action=argparse.BooleanOptionalAction, default=None,
                        help='Freeze only the encoder; decoder keeps learning. Resume inherits the saved mode unless overridden')
    parser.add_argument('--freeze-representation-after-planner', action=argparse.BooleanOptionalAction, default=None,
                        help='Freeze encoder/decoder on planner levels; train again on new levels. Resume inherits policy')
    parser.add_argument('--predictor-replay-episodes', type=int,
                        help='RAM-only real episode capacity; default 0 (disabled), or inherit checkpoint on resume')
    parser.add_argument('--predictor-batch-size', type=int,
                        help='Distinct real episodes per predictor update; default 1, or inherit checkpoint on resume')
    parser.add_argument('--path-hindsight', action=argparse.BooleanOptionalAction, default=None,
                        help='Additional goals actually reached in real planner replay; resume inherits policy')
    parser.add_argument('--path-predicted-inputs', action=argparse.BooleanOptionalAction, default=None,
                        help='Calibrate predicted inputs against real outcomes; resume inherits policy')
    args = parser.parse_args(argv)
    if args.update_representation_threshold and not args.resume:
        parser.error('--update-representation-threshold requires --resume')
    if args.update_latent_drift_threshold and not args.resume:
        parser.error('--update-latent-drift-threshold requires --resume')
    if args.update_evaluation_interval and not args.resume:
        parser.error('--update-evaluation-interval requires --resume')
    if (min(args.actions, args.steps, args.evaluate_every, args.evaluation_episodes, args.checkpoint_every) < 1
            or args.steps > 128 or not 0 <= args.seed < 100
            or not 0 < args.learning_rate < 1 or not 0 < args.representation_learning_rate < 1):
        parser.error('Invalid budgets, learning rates or seed')
    if (args.encoder_learning_rate_after_predictor is not None
            and not 0 < args.encoder_learning_rate_after_predictor <= args.representation_learning_rate):
        parser.error('Post-predictor encoder LR must be positive and no greater than representation LR')
    try:
        gates = CurriculumConfig(representation_open=args.representation_error,
            prediction_open=args.predictor_error, latent_drift=args.latent_drift,
            success=args.success, patience=args.patience)
    except ValueError as error:
        parser.error(str(error))
    curriculum = Curriculum({args.game: SYNTHETIC_LEVELS[args.game]}, gates)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / 'report.json'
    if path.exists() and not args.resume:
        parser.error('Use an empty run directory, or --resume to continue its latest checkpoint')
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(args.seed)
    # Keep each game's evaluation scenarios stable when campaign coverage changes.
    evaluation_pairs = [(g, n) for g, count in SYNTHETIC_LEVELS.items() for n in range(1, count + 1)]
    report_config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                     if k not in ('update_representation_threshold', 'update_latent_drift_threshold',
                                  'update_evaluation_interval')}
    report_config['curriculum'] = asdict(gates)
    report_config['horizon'] = PathConfig().horizon
    report_config['initialization'] = 'random' if args.init_config else 'checkpoint'
    report = dict(kind='synthetic_subgoals', status='running', config=report_config,
                  progress={'phase': 'loading', 'actions': 0, 'total_actions': args.actions},
                  curriculum=curriculum.snapshot(), curves=[], episodes=[], evaluations=[])
    restored = None
    if args.resume:
        if not path.exists() or not (args.output / 'latest.pt').exists():
            parser.error('--resume requires report.json and latest.pt in --output')
        report = json.loads(path.read_text())
        for key, value in report_config.items():
            previous = report['config'].get(key)
            if args.update_representation_threshold:
                if key == 'representation_error':
                    continue
                if key == 'curriculum' and previous is not None:
                    previous = dict(previous, representation_open=gates.representation_open)
            if args.update_latent_drift_threshold:
                if key == 'latent_drift':
                    continue
                if key == 'curriculum' and previous is not None:
                    previous = dict(previous, latent_drift=gates.latent_drift)
            if key == 'evaluate_every' and args.update_evaluation_interval:
                continue
            if key not in ('resume', 'actions', 'source', 'output', 'prediction_delta_loss', 'freeze_encoder', 'migrate_path_value',
                           'predictor_replay_episodes', 'predictor_batch_size', 'path_hindsight', 'path_predicted_inputs',
                           'encoder_learning_rate_after_predictor', 'freeze_representation_after_planner') and previous != value:
                parser.error(f'Cannot change {key} while resuming this run')
        saved = checkpoint.load(args.output / 'latest.pt')
        restored = saved['training_progress']
        del saved
        curriculum.restore(restored['curriculum'],
                           update_representation_threshold=args.update_representation_threshold,
                           update_latent_drift_threshold=args.update_latent_drift_threshold)
        old_interval = report['config']['evaluate_every']
        if old_interval != args.evaluate_every:
            # The drift window has changed: retain learned permissions but
            # require new consecutive evidence before another activation.
            for state in curriculum.readiness.values():
                state.representation_passes = state.prediction_passes = state.promotion_passes = 0
        if args.actions <= restored['actions']:
            parser.error('--actions must exceed the saved action counter')
        # Report publication can be ahead of the atomic weight snapshot.
        # Discard those unsaved measurements instead of pairing them with old weights.
        for key, counter in (('curves', 'actions'), ('episodes', 'end_action'),
                             ('evaluations', 'actions'), ('promotions', 'actions')):
            report[key] = [r for r in report.get(key, []) if r[counter] <= restored['actions']
                           and r.get('split') != 'final']
        report.update(status='running', config=report_config,
            progress={'phase': 'loading', 'actions': restored['actions'], 'total_actions': args.actions},
            checkpoint_action=restored['actions'],
            resumed_from={'actions': restored['actions'], 'optimizer_moments': 'reset'})
        old_threshold = restored['curriculum']['thresholds']['representation_open']
        if old_threshold != gates.representation_open:
            report['resumed_from']['representation_threshold'] = dict(old=old_threshold, new=gates.representation_open)
        old_drift = restored['curriculum']['thresholds']['latent_drift']
        if old_drift != gates.latent_drift:
            report['resumed_from']['latent_drift_threshold'] = dict(old=old_drift, new=gates.latent_drift)
        if old_interval != args.evaluate_every:
            report['resumed_from']['evaluation_interval'] = dict(old=old_interval, new=args.evaluate_every)
        report.pop('error', None)
        report.pop('prediction_sample', None)
    previous_encoder_rate = (restored or {}).get('encoder_learning_rate_after_predictor')
    encoder_rate = (previous_encoder_rate if args.encoder_learning_rate_after_predictor is None
                    else args.encoder_learning_rate_after_predictor)
    if encoder_rate is not None and not 0 < encoder_rate <= args.representation_learning_rate:
        parser.error('Post-predictor encoder LR must be positive and no greater than representation LR')
    changed_encoder_rate = encoder_rate != previous_encoder_rate
    report['config']['encoder_learning_rate_after_predictor'] = encoder_rate
    if args.resume and changed_encoder_rate:
        report['resumed_from']['encoder_learning_rate_after_predictor'] = dict(old=previous_encoder_rate, new=encoder_rate)
    previous_planner_freeze = (restored or {}).get('freeze_representation_after_planner', False)
    freeze_after_planner = (previous_planner_freeze if args.freeze_representation_after_planner is None
                            else args.freeze_representation_after_planner)
    changed_planner_freeze = freeze_after_planner != previous_planner_freeze
    report['config']['freeze_representation_after_planner'] = freeze_after_planner
    if args.resume and changed_planner_freeze:
        report['resumed_from']['freeze_representation_after_planner'] = dict(old=previous_planner_freeze, new=freeze_after_planner)
    previous_replay = (restored or {}).get('predictor_replay', dict(episodes=0, batch_size=1))
    replay_config = dict(episodes=(previous_replay['episodes'] if args.predictor_replay_episodes is None
                                  else args.predictor_replay_episodes),
                         batch_size=(previous_replay['batch_size'] if args.predictor_batch_size is None
                                     else args.predictor_batch_size))
    if replay_config['episodes'] < 0 or not 1 <= replay_config['batch_size'] <= max(1, replay_config['episodes']):
        parser.error('Replay requires a nonnegative capacity and batch size between 1 and capacity (1 when disabled)')
    changed_replay = replay_config != previous_replay
    report['config'].update(predictor_replay_episodes=replay_config['episodes'],
                            predictor_batch_size=replay_config['batch_size'])
    if args.resume:
        report['resumed_from']['predictor_replay_buffer'] = 'empty'
        if changed_replay:
            report['resumed_from']['predictor_replay'] = dict(old=previous_replay, new=replay_config)
    previous_path_learning = (restored or {}).get('path_learning', dict(hindsight=False, predicted_inputs=False))
    path_learning = {name: (previous_path_learning[name] if value is None else value)
                     for name, value in (('hindsight', args.path_hindsight),
                                         ('predicted_inputs', args.path_predicted_inputs))}
    changed_path_learning = path_learning != previous_path_learning
    report['config'].update({f'path_{name}': value for name, value in path_learning.items()})
    if args.resume:
        report['resumed_from']['path_replay_buffer'] = 'empty'
        if changed_path_learning:
            report['resumed_from']['path_learning'] = dict(old=previous_path_learning, new=path_learning)
    recovery = (restored or {}).get('path_recovery')
    if recovery:
        report['path_recovery'] = recovery
    previews = PredictionSamples()
    trajectories = PlannerTrajectory(args.output / 'planner_trajectory.json')
    throughput = RecentThroughput()
    report.pop('planner_trajectory', None)
    report.pop('trajectory_error', None)
    last_write, started = 0., time.monotonic()
    checkpoint_requests = saved_requests = 0
    control_requests = []
    report['manual_controls'] = list((restored or {}).get('manual_controls', []))
    active_pair = tuple((restored or {}).get('active_pair', (args.game, curriculum.unlocked[args.game])))
    if active_pair not in curriculum.pairs:
        parser.error('Saved active training level is not unlocked')

    def request_checkpoint(signum, frame):
        # No disk/CUDA work in a signal handler. Count requests so a signal
        # received DURING a save remains pending for the next safe boundary.
        nonlocal checkpoint_requests
        checkpoint_requests += 1

    def request_control(signum, frame):
        control_requests.append('phase' if signum == signal.SIGUSR2 else 'level')

    previous_handlers = {}
    report['control_signals'] = {}
    for name, operation, handler in (('SIGUSR1', 'checkpoint', request_checkpoint),
                                    ('SIGUSR2', 'phase', request_control),
                                    ('SIGHUP', 'level', request_control)):
        number = getattr(signal, name, None)
        if number is not None:
            previous_handlers[number] = signal.signal(number, handler)
            report['control_signals'][operation] = name
    report['checkpoint_signal'] = report['control_signals'].get('checkpoint')

    def publish(force=False, **values):
        nonlocal last_write
        report['progress'].update(values)
        now = time.monotonic()
        if not force and now - last_write < 2:
            return
        last_write = now
        report['updated_at'] = time.time()
        report['curriculum'] = curriculum.snapshot()
        report['prediction_sample'] = previews.draw()
        report['planner_trajectory'] = trajectories.metadata
        report['trajectory_error'] = trajectories.error
        report['progress']['recent_tps'] = throughput.value()
        write_report(path, report)
        print(json.dumps({'event': 'subgoal_progress', **{k: v for k, v in report['progress'].items()
            if k not in ('observation', 'goal_frame', 'goal_mask')}}, allow_nan=False), flush=True)

    try:
        publish(force=True)
        if args.resume or args.source:
            source = agent_checkpoint.load(args.output / 'latest.pt' if args.resume else args.source, device=args.device,
                                           migrate_path_value=args.migrate_path_value)
            engine, heads = source.engine, source.heads
            evaluator = source.path_evaluator if args.resume else None
            del source
        else:
            initial = json.loads(args.init_config.read_text())
            net = WorldModel(NetworkConfig(**initial['network'])).to(args.device)
            engine = Engine(net, TrainConfig(**initial.get('training', {})), compile_model=args.device == 'cuda')
            heads = GoalValue(net.config.latent, net.config.memory,
                              width=initial.get('outcome_width', 64)).to(args.device)
            evaluator = None
        migrated_path = bool(evaluator is not None and getattr(evaluator, 'migrated_from_legacy', False))
        path_version = evaluator.config['objective_version'] if evaluator is not None else 4
        if args.resume and migrated_path:
            report['resumed_from']['path_objective'] = dict(old_version=evaluator.previous_objective_version,
                new_version=path_version, reset_outputs=True)
        report['config'].update(path_objective_version=path_version, path_success_margin=PathConfig().success_margin)
        engine.optimizer_bits = 8 if args.device == 'cuda' else 32
        previous_loss = engine.config.prediction_delta_loss
        selected_loss = args.prediction_delta_loss or previous_loss
        changed_loss = selected_loss != previous_loss
        previous_freeze = (restored or {}).get('freeze_encoder', False)
        freeze_encoder = previous_freeze if args.freeze_encoder is None else args.freeze_encoder
        changed_freeze = freeze_encoder != previous_freeze
        engine.config = replace(engine.config, freeze_representation=False, cache_frozen_latents=True,
            prediction_delta_loss=selected_loss,
            learning_rate=args.learning_rate, train_every=1, loss_steps=1,
            retain_replay_across_levels=False, replay_episodes=1, burn_in_steps=0, representation_joint_steps=0)
        report['config']['prediction_delta_loss'] = selected_loss
        report['config']['freeze_encoder'] = freeze_encoder
        if changed_loss or migrated_path:
            at_action = (restored or {}).get('actions', 0)
            changes = [r for r in report.get('objective_changes', []) if r['actions'] <= at_action]
            changes.append(dict(actions=at_action, prediction_delta_loss=selected_loss,
                                path_objective_version=path_version, path_heads_reset=migrated_path))
            report['objective_changes'] = changes[-32:]
        if args.resume and changed_loss:
            report['resumed_from']['prediction_delta_loss'] = dict(old=previous_loss, new=selected_loss)
        if args.resume and changed_freeze:
            report['resumed_from']['freeze_encoder'] = dict(old=previous_freeze, new=freeze_encoder)
        if args.resume and (changed_loss or migrated_path or changed_freeze or changed_replay or changed_encoder_rate or changed_planner_freeze):
            for state in curriculum.readiness.values():
                state.representation_passes = state.prediction_passes = state.promotion_passes = 0
        if evaluator is None:
            if args.resume:
                raise ValueError('Resuming requires the saved PathValue weights')
            evaluator = PathValue(engine.model.config.latent, engine.model.config.memory).to(args.device)
        agent = Agent(engine, heads, AgentConfig(mode='path_mpc', max_actions=args.steps, path=PathConfig()),
                      seed=args.seed, frozen=False, compile_heads=args.device == 'cuda', path_evaluator=evaluator)
        representation = OnlineRepresentation(engine, args.representation_learning_rate, freeze_encoder=freeze_encoder,
            encoder_learning_rate_after_predictor=encoder_rate)
        # A shared encoder must not resume distribution shaping when a later
        # level is still in representation-only warmup. Restored permissions
        # also restore this global phase without adding a checkpoint counter.
        if any(state.predictor for state in curriculum.readiness.values()):
            representation.activate_predictor()
        world = OnlinePredictor(engine, None, args.learning_rate, agent.config.path.horizon)
        predictor_replay = (PredictorReplay(engine, **replay_config, horizon=world.horizon, seed=args.seed)
                            if replay_config['episodes'] else None)
        path_learner = PathLearner(agent.planner, compile_model=args.device == 'cuda', seed=args.seed, **path_learning)
        factory = Factory(args.game, Path(__file__).resolve().parents[1] / 'games')
        actions, episode_index, planner_actions = 0, 0, dict.fromkeys(curriculum.readiness, 0)
        if restored:
            actions, episode_index = restored['actions'], restored['episodes']
            representation.updates = restored['representation_updates']
            world.updates, path_learner.updates = restored['updates'], restored['path_updates']
            if 'planner_actions' in restored:
                planner_actions.update({(r['game'], r['level']): r['actions'] for r in restored['planner_actions']})
            else:
                # Older snapshots did not save this exploration counter. Recover
                # it from completed, learner-enabled real episodes when available.
                for episode in report['episodes']:
                    if any(s.get('loss') is not None for s in episode['segments']):
                        planner_actions[episode['game'], episode['level']] += episode['actions']
        anchors = {}
        next_eval = (actions // args.evaluate_every + 1) * args.evaluate_every
        next_checkpoint = (actions // args.checkpoint_every + 1) * args.checkpoint_every
        process_start_actions = actions
        began_learning = None

        def select_representation_phase():
            if any(state.predictor for state in curriculum.readiness.values()):
                representation.activate_predictor()
            if freeze_after_planner:
                state = curriculum.readiness[active_pair]
                representation.select_level(predictor=state.predictor, planner=state.planner)
            publish(force=True, sigreg_enabled=representation.sigreg_enabled,
                    representation_frozen=representation.frozen, encoder_frozen=representation.freeze_encoder,
                    representation_learning_rates=representation.learning_rates,
                    representation_level=dict(game=active_pair[0], level=active_pair[1]))

        def evaluate(split):
            # Freeze the coverage of this sweep; promotion happens AFTER it.
            pairs = curriculum.pairs
            coverage = [dict(game=g, level=n) for g, n in pairs]
            for game, level in pairs:
                pair = game, level
                readiness = curriculum.readiness[pair]
                seeds = episode_seeds(split, args.evaluation_episodes,
                                      offset=700000 + evaluation_pairs.index(pair) * 100)
                quality, success = None, None
                for mode in ('random', 'path_mpc'):
                    if mode == 'path_mpc' and not readiness.planner:
                        continue
                    summaries = []
                    for i, seed in enumerate(seeds):
                        def progress(row):
                            publish(phase='evaluation', policy=mode, evaluation_game=game, evaluation_level=level,
                                evaluation_episode=i + 1, evaluation_episodes=len(seeds), evaluation_action=row['action'],
                                predictor_enabled=readiness.predictor, planner_enabled=readiness.planner,
                                **{k: row[k] for k in ('goal', 'observation', 'goal_frame', 'goal_mask')})
                        capture = trajectories.begin() if mode == 'path_mpc' else None
                        summaries.append(play(agent, factory, level, seed, args.steps,
                            random_policy=mode == 'random', progress=progress,
                            sample=lambda before, actual, predicted, **meta: previews.add(
                                before, actual, predicted, phase='evaluation', training_action=actions, **meta),
                            quality_horizon=agent.config.path.horizon if mode == 'random' else None,
                            trajectory_step=capture,
                            trajectory=(lambda episode, records, result: trajectories.add(
                                episode, records, result, game=game, level=level, episode_seed=seed,
                                phase='evaluation', split=split, training_action=actions)) if capture else None))
                    if mode == 'random':
                        quality, current_anchors = summarize_quality(
                            summaries, anchors.get(pair) if split == 'dev' else None)
                        if split == 'dev':
                            anchors[pair] = current_anchors
                    else:
                        success = sum(r['success'] for r in summaries) / len(summaries)
                    report['evaluations'].append(dict(actions=actions, stage=curriculum.stage,
                        evaluate_every=args.evaluate_every,
                        coverage=coverage, split=split, policy=mode, game=game, level=level,
                        episodes=len(summaries), successes=sum(r['success'] for r in summaries),
                        subgoals=sum(s['success'] for r in summaries for s in r['segments']),
                        attempted_subgoals=sum(len(r['segments']) for r in summaries),
                        real_actions=sum(r['actions'] for r in summaries),
                        delta_error=sum(r['counts']['wrong'] for r in summaries) / max(1, sum(r['counts']['union'] for r in summaries)),
                        quality=quality if mode == 'random' else None,
                        decision_ms=sum(r['decision_ms'] for r in summaries) / len(summaries)))
                    publish(force=True)
                if split == 'dev':
                    readiness.observe(quality, success, gates)
            if split == 'dev':
                promotions = curriculum.advance()
                if promotions:
                    report.setdefault('promotions', []).append(dict(actions=actions, levels=promotions))
                publish(force=True)
                select_representation_phase()
            save_requested()

        def save(reason='scheduled'):
            nonlocal saved_requests
            requests = checkpoint_requests
            if requests != saved_requests:
                reason = 'signal'
            previous_phase = report['progress']['phase']
            publish(force=True, phase='checkpoint')
            agent_checkpoint.save(args.output / 'latest.pt', agent, training_progress={
                'actions': actions, 'updates': world.updates, 'representation_updates': representation.updates,
                'path_updates': path_learner.updates, 'episodes': episode_index,
                'freeze_encoder': freeze_encoder,
                'freeze_representation_after_planner': freeze_after_planner,
                'encoder_learning_rate_after_predictor': encoder_rate,
                'predictor_replay': replay_config,
                'path_learning': path_learning,
                'path_recovery': recovery,
                'curriculum': curriculum.snapshot(),
                'active_pair': active_pair,
                'manual_controls': list(report['manual_controls']),
                'planner_actions': [dict(game=g, level=n, actions=count)
                                   for (g, n), count in planner_actions.items()]})
            saved_requests = requests
            report['checkpoint_action'] = actions
            report['last_checkpoint'] = dict(actions=actions, reason=reason, saved_at=time.time())
            print(json.dumps(dict(event='checkpoint_saved', **report['last_checkpoint'],
                                  path=str(args.output / 'latest.pt'))), flush=True)
            publish(force=True, phase=previous_phase)

        def save_requested():
            nonlocal active_pair
            # Consume a snapshot: new signals during IO wait for another boundary.
            requests = control_requests[:]
            del control_requests[:len(requests)]
            for operation in requests:
                game, level = active_pair
                state = curriculum.readiness[active_pair]
                old_phase = 'planner' if state.planner else 'predictor' if state.predictor else 'representation'
                result = 'budget_exhausted'
                if actions < args.actions:
                    if operation == 'phase':
                        result = 'advanced' if curriculum.advance_phase(active_pair) else 'already_planner'
                    else:
                        new_pair = curriculum.advance_level(game)
                        result = 'advanced' if new_pair else 'last_level'
                        if new_pair:
                            active_pair = new_pair
                new_state = curriculum.readiness[active_pair]
                event = dict(actions=actions, operation=operation, result=result, game=game,
                             old_level=level, level=active_pair[1], old_phase=old_phase,
                             phase='planner' if new_state.planner else 'predictor' if new_state.predictor else 'representation')
                report['manual_controls'].append(event)
                report['manual_controls'] = report['manual_controls'][-128:]
                print(json.dumps(dict(event='curriculum_control', **event)), flush=True)
            if requests:
                select_representation_phase()
                save('manual_control')
                return
            if checkpoint_requests != saved_requests:
                save()

        select_representation_phase()
        publish(force=True, predictor_updates=world.updates, representation_updates=representation.updates,
                path_updates=path_learner.updates, sigreg_enabled=representation.sigreg_enabled,
                representation_learning_rates=representation.learning_rates,
                encoder_frozen=representation.freeze_encoder, representation_frozen=representation.frozen,
                path_replay=path_learner.status(),
                predictor_replay=(dict(predictor_replay.status(), actual_batch_size=0) if predictor_replay else None))
        if args.resume and (args.update_representation_threshold or args.update_latent_drift_threshold
                            or args.update_evaluation_interval or changed_loss or migrated_path or changed_freeze or changed_replay
                            or changed_encoder_rate or changed_planner_freeze or changed_path_learning):
            save('resume')
        if not args.resume:
            save('initial')
            evaluate('dev')  # Baseline and drift anchors; never grants readiness by itself.
        save_requested()
        while actions < args.actions:
            pair = curriculum.next_pair()
            active_pair = pair
            select_representation_phase()
            game, level = pair
            readiness = curriculum.readiness[pair]
            seed = episode_seeds('train', 1, offset=args.seed * 10000 + episode_index)[0]
            epsilon = max(.2, 1. - .8 * planner_actions[pair] / 8192) if readiness.planner else 1.
            def progress(row):
                nonlocal actions, began_learning
                actions += 1
                if readiness.planner:
                    planner_actions[pair] += 1
                if began_learning is None:
                    began_learning = time.monotonic()
                seconds = time.monotonic() - began_learning
                recent_tps = throughput.record()
                point = dict(actions=actions, game=game, level=level, stage=curriculum.stage,
                    error=row['delta_error'], path_loss=row['path_loss'],
                    predictor_loss=(row['predictor_loss'] or {}).get('total'),
                    representation_loss=(row['representation_loss'] or {}).get('total'),
                    predictor_terms=(row['predictor_loss'] or {}).get('terms'),
                    representation_terms=(row['representation_loss'] or {}).get('terms'),
                    representation_updates=representation.updates,
                    predictor_updates=world.updates, path_updates=path_learner.updates,
                    counts=row['counts'], recent_tps=recent_tps,
                    tps=(actions - process_start_actions - 1) / seconds if seconds else None)
                report['curves'].append(point)
                report['curves'] = report['curves'][-4096:]
                publish(phase='training', game=game, level=level, actions=actions, episodes=episode_index + 1,
                        representation_updates=representation.updates,
                        predictor_updates=world.updates, path_updates=path_learner.updates,
                        path_replay=path_learner.status(),
                        predictor_enabled=readiness.predictor, planner_enabled=readiness.planner,
                        tps=point['tps'], seconds=seconds, **{k: v for k, v in row.items() if k not in ('counts', 'segment')})
            publish(force=True, phase='warmup' if began_learning is None else 'training', game=game, level=level,
                    predictor_enabled=readiness.predictor, planner_enabled=readiness.planner)
            capture = trajectories.begin() if readiness.planner else None
            result = play(agent, factory, level, seed, args.steps,
                representation=representation, world=world if readiness.predictor else None,
                predictor_replay=predictor_replay if readiness.predictor else None,
                learner=path_learner if readiness.planner else None,
                random_policy=not readiness.planner, epsilon=epsilon, progress=progress,
                sample=lambda before, actual, predicted, **meta: previews.add(
                    before, actual, predicted, phase='training', training_action=actions + 1, **meta),
                trajectory_step=capture,
                trajectory=(lambda episode, records, result: trajectories.add(
                    episode, records, result, game=game, level=level, episode_seed=seed,
                    phase='training', training_action=actions)) if capture else None,
                action_limit=args.actions - actions)
            report['episodes'].append(dict(game=game, level=level, end_action=actions,
                **{k: v for k, v in result.items() if k not in ('records', 'quality')}))
            report['episodes'] = report['episodes'][-2000:]
            episode_index += 1
            # The snapshot does not contain an unfinished episode's buffer.
            # Save only once its learning, counters and seed have all advanced.
            save_requested()
            if actions >= next_eval:
                evaluate('dev')
                next_eval = (actions // args.evaluate_every + 1) * args.evaluate_every
            if actions >= next_checkpoint or actions == args.actions:
                save()
                next_checkpoint = (actions // args.checkpoint_every + 1) * args.checkpoint_every
            publish(force=True, phase='training')
        evaluate('final')
        report['status'] = 'complete'
        publish(force=True, phase='complete', elapsed=time.monotonic() - started)
    except BaseException as error:
        report['status'], report['error'] = 'failed', str(error)
        publish(force=True, phase='failed')
        raise
    finally:
        for number, handler in previous_handlers.items():
            signal.signal(number, handler)


if __name__ == '__main__':
    main()
