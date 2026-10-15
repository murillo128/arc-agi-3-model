"""One long synthetic-subgoal run, online dynamics and terminal path learning.

Fresh counters and path weights, existing world weights. Only actual SDK
transitions train dynamics; only finished real goal segments train PathValue.
The environment's teacher stays outside the portable Agent.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from . import agent_checkpoint
from .agent import Agent, AgentConfig
from .behavior_experiment import episode_seeds
from .campaign import write_report
from .data import Episode, random_action
from .environment import Factory, SYNTHETIC_LEVELS
from .learning import optimizer, visual_delta_loss, soft_delta_error
from .metrics import delta_counts
from .path_planning import PathConfig, PathValue, PathLearner, real_states


class OnlinePredictor:
    def __init__(self, engine, threshold=.10, learning_rate=3e-5):
        self.engine, self.threshold = engine, threshold
        engine.model.encoder.eval().requires_grad_(False)
        engine.model.decoder.eval().requires_grad_(False)
        engine.model.dynamics.requires_grad_(True)
        self.optimizer = optimizer(engine.model.dynamics.parameters(), learning_rate,
                                   engine.device.type == 'cuda', engine.optimizer_bits)
        self.updates = 0

    def update(self, state, action, reference, after, target_z, error):
        if error is None or error <= self.threshold or after.final is None or target_z is None:
            return None
        e = self.engine
        self.optimizer.zero_grad(set_to_none=True)
        e.model.dynamics.train()
        with e.autocast():
            predicted, _, delta = e.predict(state.z[None].detach(), state.h.detach(), e.actions([action]), *state.geometry)
            logits = e.decode(predicted, *state.geometry, checkpoint_blocks=True)
        before = torch.tensor(np.array(reference, copy=True), device=e.device, dtype=torch.long)[None]
        actual = torch.tensor(np.array(after.final, copy=True), device=e.device, dtype=torch.long)[None]
        latent = F.mse_loss(delta.float(), (target_z - state.z).detach()[None].float())
        visual = visual_delta_loss(logits, before, actual, e.config.false_change_weight).mean()
        overlap = soft_delta_error(logits, before, actual).mean()
        loss = latent + visual + e.config.delta_overlap_weight * overlap
        loss.backward()
        torch.nn.utils.clip_grad_norm_(e.model.dynamics.parameters(), 1., error_if_nonfinite=True)
        self.optimizer.step()
        self.updates += 1
        return {'total': float(loss.detach()), 'latent': float(latent.detach()), 'visual': float(visual.detach())}


def error_counts(before, actual, predicted):
    wrong, union, changed = delta_counts(before, actual, predicted)
    real, guessed = actual != before, predicted != before
    return dict(wrong=wrong, union=union, changed=changed,
                invented=int((~real & guessed).sum()), missed=int((real & ~guessed).sum()),
                color=int((real & guessed & (actual != predicted)).sum()),
                correct=int((real & (actual == predicted)).sum()))


def play(agent, factory, level, seed, steps, *, world=None, learner=None, epsilon=0.,
         random_policy=False, progress=None, action_limit=None):
    """A synthetic teacher supplies only one visual goal to the shared agent.

    Evaluation calls omit both learners. Goal predicates/distances are used only
    by this trainer, never by action selection or hypothetical-state scoring.
    """
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
    decision_ms = 0.
    end_reason = 'budget'
    count = min(steps, action_limit if action_limit is not None else steps)
    for position in range(count):
        before, state = env.observation, agent.state
        begin = time.monotonic()
        if random_policy or rng.random() < epsilon:
            action = random_action(before, rng)
            agent.pending_action = action
        else:
            action = agent.act(diagnostics=True)
        decision_ms += (time.monotonic() - begin) * 1000
        prediction = agent.engine.prediction(state, action, before.final)
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
            if world is not None:
                losses = world.update(state, action, before.final, after, agent.state.z, error)
                if losses is not None:
                    # No cached imagined state survives a dynamics update. The
                    # real recurrent memory is refreshed from this episode only.
                    agent.state = real_states(agent.engine, episode)[-1]
                    agent.planner.last_plan, agent.planner.last_states = [], []
        level_success = after.state == 'WIN' or boundary
        success = not boundary and teacher.training_goal_reached(goal)
        failure = after.state == 'GAME_OVER' or (position + 1 == steps and not success)
        path_loss = None
        if success or failure:
            length = len(episode.actions) - start
            if learner is not None and not boundary:
                path_loss = learner.update(episode, start, success, steps)
            segment = dict(goal=goal.id, success=bool(success), actions=length, minimum_actions=minimum,
                excess_actions=length - minimum, reward=-(length - minimum) - (0 if success else steps),
                loss=path_loss)
            segments.append(segment)
            if success:
                goal_index += 1
        row = dict(action=position + 1, goal_index=goal_index + 1 - int(success), goals=len(goals),
            goal=goal.label, minimum_actions=minimum, goal_actions=len(episode.actions) - start,
            delta_error=error, counts=counts, predictor_loss=losses, path_loss=path_loss,
            predictor_updated=losses is not None, segment=segments[-1] if success or failure else None,
            epsilon=epsilon, decision_ms=decision_ms / (position + 1),
            observation=after.final.tolist() if after.final is not None else None,
            goal_frame=goal.frame.tolist(), goal_mask=goal.mask.astype(int).tolist())
        records.append({k: v for k, v in row.items() if k not in ('observation', 'goal_frame', 'goal_mask')})
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
    return dict(success=end_reason == 'success', reason=end_reason, actions=len(episode.actions),
                segments=segments, counts=totals, decision_ms=decision_ms / max(1, len(episode.actions)), records=records)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--actions', type=int, default=65536)
    parser.add_argument('--steps', type=int, default=128)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--evaluate-every', type=int, default=4096)
    parser.add_argument('--evaluation-episodes', type=int, default=2)
    parser.add_argument('--checkpoint-every', type=int, default=2048)
    parser.add_argument('--predictor-error', type=float, default=.10)
    parser.add_argument('--learning-rate', type=float, default=3e-5)
    args = parser.parse_args(argv)
    if (min(args.actions, args.steps, args.evaluate_every, args.evaluation_episodes, args.checkpoint_every) < 1
            or args.steps > 128 or not 0 <= args.predictor_error <= 1 or not 0 <= args.seed < 100
            or not 0 < args.learning_rate < 1):
        parser.error('Invalid budgets, error threshold or seed')
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / 'report.json'
    if path.exists():
        parser.error('Use an empty run directory; previous results are not resumed')
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(args.seed)
    pairs = [(game, level) for game, n in SYNTHETIC_LEVELS.items() for level in range(1, n + 1)]
    report_config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    report_config['evaluation_pairs'] = [dict(game=game, level=level) for game, level in pairs]
    report = dict(kind='synthetic_subgoals', status='running', config=report_config,
                  progress={'phase': 'loading', 'actions': 0, 'total_actions': args.actions}, curves=[], episodes=[], evaluations=[])
    last_write, started = 0., time.monotonic()

    def publish(force=False, **values):
        nonlocal last_write
        report['progress'].update(values)
        now = time.monotonic()
        if not force and now - last_write < 2:
            return
        last_write = now
        report['updated_at'] = time.time()
        write_report(path, report)
        print(json.dumps({'event': 'subgoal_progress', **{k: v for k, v in report['progress'].items()
            if k not in ('observation', 'goal_frame', 'goal_mask')}}, allow_nan=False), flush=True)

    agent = None
    try:
        publish(force=True)
        source = agent_checkpoint.load(args.source, device=args.device)
        engine = source.engine
        engine.config = replace(engine.config, freeze_representation=True, cache_frozen_latents=True,
            learning_rate=args.learning_rate, train_every=1, loss_steps=1, rollout_weight=0., rollout_visual_weight=0.,
            retain_replay_across_levels=False, replay_episodes=1, burn_in_steps=0, representation_joint_steps=0)
        evaluator = PathValue(engine.model.config.latent, engine.model.config.memory).to(args.device)
        agent = Agent(engine, source.heads, AgentConfig(mode='path_mpc', max_actions=args.steps, path=PathConfig()),
                      seed=args.seed, frozen=False, compile_heads=args.device == 'cuda', path_evaluator=evaluator)
        del source
        world = OnlinePredictor(engine, args.predictor_error, args.learning_rate)
        path_learner = PathLearner(agent.planner, compile_model=args.device == 'cuda')
        factories = {g: Factory(g, Path(__file__).resolve().parents[1] / 'games') for g in SYNTHETIC_LEVELS}
        counts, actions, episode_index = dict.fromkeys(pairs, 0), 0, 0
        next_eval, next_checkpoint = args.evaluate_every, args.checkpoint_every
        began_learning = None

        def evaluate(split):
            for mode in ('random', 'path_mpc'):
                for pair_index, (game, level) in enumerate(pairs):
                    seeds = episode_seeds(split, args.evaluation_episodes, offset=700000 + pair_index * 100)
                    summaries = []
                    for i, seed in enumerate(seeds):
                        def progress(row):
                            publish(phase='evaluation', policy=mode, evaluation_game=game, evaluation_level=level,
                                evaluation_episode=i + 1, evaluation_episodes=len(seeds), evaluation_action=row['action'],
                                **{k: row[k] for k in ('goal', 'observation', 'goal_frame', 'goal_mask')})
                        summaries.append(play(agent, factories[game], level, seed, args.steps,
                                              random_policy=mode == 'random', progress=progress))
                    report['evaluations'].append(dict(actions=actions, split=split, policy=mode, game=game, level=level,
                        episodes=len(summaries), successes=sum(r['success'] for r in summaries),
                        subgoals=sum(s['success'] for r in summaries for s in r['segments']),
                        attempted_subgoals=sum(len(r['segments']) for r in summaries),
                        real_actions=sum(r['actions'] for r in summaries),
                        delta_error=sum(r['counts']['wrong'] for r in summaries) / max(1, sum(r['counts']['union'] for r in summaries)),
                        decision_ms=sum(r['decision_ms'] for r in summaries) / len(summaries)))
                    publish(force=True)

        publish(force=True, phase='checkpoint', predictor_updates=0, path_updates=0)
        agent_checkpoint.save(args.output / 'latest.pt', agent, training_progress={
            'actions': 0, 'updates': 0, 'path_updates': 0, 'episodes': 0})
        report['checkpoint_action'] = 0
        publish(force=True, phase='warmup')
        while actions < args.actions:
            pair = min(pairs, key=lambda p: counts[p])
            game, level = pair
            seed = episode_seeds('train', 1, offset=args.seed * 10000 + episode_index)[0]
            epsilon = max(.2, 1. - .8 * actions / 8192)
            def progress(row):
                nonlocal actions, began_learning
                actions += 1
                counts[pair] += 1
                if began_learning is None:
                    began_learning = time.monotonic()
                seconds = time.monotonic() - began_learning
                point = dict(actions=actions, error=row['delta_error'], path_loss=row['path_loss'],
                    predictor_loss=(row['predictor_loss'] or {}).get('total'),
                    predictor_updates=world.updates, path_updates=path_learner.updates,
                    counts=row['counts'], tps=(actions - 1) / seconds if seconds else None)
                report['curves'].append(point)
                report['curves'] = report['curves'][-4096:]
                publish(phase='training', game=game, level=level, actions=actions, episodes=episode_index + 1,
                        predictor_updates=world.updates, path_updates=path_learner.updates,
                        tps=point['tps'], seconds=seconds, **{k: v for k, v in row.items() if k not in ('counts', 'segment')})
            result = play(agent, factories[game], level, seed, args.steps, world=world, learner=path_learner,
                epsilon=epsilon, progress=progress, action_limit=args.actions - actions)
            report['episodes'].append(dict(game=game, level=level, end_action=actions,
                **{k: v for k, v in result.items() if k != 'records'}))
            report['episodes'] = report['episodes'][-2000:]
            episode_index += 1
            if actions >= next_checkpoint or actions == args.actions:
                publish(force=True, phase='checkpoint')
                agent_checkpoint.save(args.output / 'latest.pt', agent, training_progress={
                    'actions': actions, 'updates': world.updates, 'path_updates': path_learner.updates,
                    'episodes': episode_index})
                report['checkpoint_action'] = actions
                next_checkpoint = (actions // args.checkpoint_every + 1) * args.checkpoint_every
            if actions >= next_eval:
                evaluate('dev')
                next_eval = (actions // args.evaluate_every + 1) * args.evaluate_every
            publish(force=True, phase='training')
        evaluate('final')
        report['status'] = 'complete'
        publish(force=True, phase='complete', elapsed=time.monotonic() - started)
    except BaseException as error:
        report['status'], report['error'] = 'failed', str(error)
        publish(force=True, phase='failed')
        raise


if __name__ == '__main__':
    main()
