"""Iterate new goal/value heads against random behavior; never train the world.

Ephemeral real experience only. Development and final evaluation have disjoint
seeds and never update weights. This entry point currently admits bundled
synthetic training games only, failing closed for public/held-out SDK games.
"""
import argparse
from collections import deque
from dataclasses import asdict, replace
import json
from pathlib import Path
import random
import time

import torch

from . import checkpoint
from .config import NetworkConfig, TrainConfig
from .data import random_action
from .environment import Factory, SYNTHETIC_LEVELS
from .learning import Engine
from .networks import WorldModel
from .planning import GoalValue, PlanConfig, Planner, fingerprint, freeze, head_loss, outcome, returns


def emit(event, **data):
    print(json.dumps({'event': event, **data}), flush=True)


def episode_seeds(split, count, offset=0):
    bases = {'train': 2_000_000, 'dev': 6_000_001, 'final': 8_000_001}
    if split not in bases or offset < 0 or count < 1 or offset + count >= 1_000_000:
        raise ValueError('Invalid disjoint seed partition')
    return [bases[split] + 2 * index for index in range(offset, offset + count)]


@torch.no_grad()
def collect(engine, factory, level, seeds, steps, discount, records, policy=None, mode='q', epsilon=.25, round_index=None):
    """Store only causal pre-action latents; labels arrive from actual steps."""
    started, wins, actions = time.monotonic(), 0, 0
    for index, seed in enumerate(seeds):
        env, rng = factory.make(seed, level), random.Random(seed)
        obs = env.observation
        observations, chosen, events = [], [], []
        state = engine.initial(obs) if policy else None
        for _ in range(steps):
            if obs.terminal:
                break
            action = (random_action(obs, rng) if policy is None or state.z is None or rng.random() < epsilon
                      else policy.choose(state, obs, rng, mode))
            after = env.step(action)
            observations.append(obs)
            chosen.append(action)
            events.append(outcome(obs, after))
            actions += 1
            if events[-1]:
                wins += events[-1] == 1
                break
            if policy:
                state = engine.observe(state, action, after)
            obs = after
        if observations:
            # Batch visual work over this *real* episode. Recurrent memory is
            # still advanced in causal order with the executed actions.
            with engine.autocast():
                zs = engine.encode(observations)
                h = torch.zeros(1, engine.model.config.memory, device=engine.device)
                last_z, geometry = None, None
                for obs, z, action, event, target in zip(observations, zs, chosen, events, returns(events, discount)):
                    if z is not None:
                        last_z, geometry = z, tuple(obs.final.shape)
                    if last_z is None:
                        continue
                    records.append((last_z.to(device='cpu', dtype=torch.float16),
                                    h[0].to(device='cpu', dtype=torch.float16),
                                    torch.tensor(action.features()), event, target, geometry))
                    h, _ = engine.advance(last_z[None], h, engine.actions([action]))
        if (index + 1) % 16 == 0 or index + 1 == len(seeds):
            emit('behavior_collection', round=round_index, episodes=index + 1, wins=wins, actions=actions,
                 records=len(records), seconds=round(time.monotonic() - started, 2))
    return {'episodes': len(seeds), 'wins': wins, 'actions': actions}


def fit(engine, heads, records, epochs, batch_size, learning_rate, seed, compiled=False, round_index=None):
    if not records:
        raise ValueError('No real training records')
    heads.train()
    forward = torch.compile(heads, fullgraph=True, dynamic=True) if compiled else heads
    opt = torch.optim.AdamW(heads.parameters(), lr=learning_rate, weight_decay=.001)
    groups = {}
    for row in records:
        groups.setdefault(row[-1], []).append(row)
    rng, started = random.Random(seed), time.monotonic()
    updates = 0
    for epoch in range(epochs):
        loss_sum, samples = 0., 0
        component_sum = torch.zeros(3, device=engine.device)
        for geometry, rows in groups.items():
            indices = list(range(len(rows)))
            rng.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                # Fixed batch shape, cycling within the shuffled real records.
                selected = [rows[indices[i % len(indices)]] for i in range(start, start + batch_size)]
                z, h, action = [torch.stack([r[k] for r in selected]).to(engine.device) for k in range(3)]
                events = torch.tensor([r[3] for r in selected], device=engine.device)
                targets = torch.tensor([r[4] for r in selected], device=engine.device)
                p = engine.model.config.patch
                opt.zero_grad(set_to_none=True)
                with engine.autocast():
                    loss, parts = head_loss(forward, z, h, action, events, targets, *((s + p - 1) // p for s in geometry), components=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(heads.parameters(), 10.)
                opt.step()
                updates += 1
                component_sum += parts * len(selected)
                loss_sum += float(loss.detach()) * len(selected)
                samples += len(selected)
        event_ce, value_bce, q_bce = (component_sum / samples).tolist()
        emit('behavior_fit', round=round_index, epoch=epoch + 1, epochs=epochs, updates=updates,
             loss=loss_sum / samples, event_ce=event_ce, value_bce=value_bce, q_bce=q_bce,
             records=len(records), seconds=round(time.monotonic() - started, 2))
    heads.eval()


@torch.no_grad()
def evaluate(engine, planner, factory, level, seeds, steps, mode, *, split='unspecified', round_index=None, role='candidate'):
    results, started = [], time.monotonic()
    for index, seed in enumerate(seeds):
        rng, env = random.Random(seed), factory.make(seed, level)
        obs, actions, event = env.observation, 0, 0
        state = engine.initial(obs) if mode != 'random' else None
        decision_seconds = 0.
        for _ in range(steps):
            if obs.terminal:
                break
            decision_start = time.monotonic()
            action = (random_action(obs, rng) if mode == 'random' or state.z is None
                      else planner.choose(state, obs, rng, mode))
            decision_seconds += time.monotonic() - decision_start
            after = env.step(action)
            actions += 1
            event = outcome(obs, after)
            if event:
                break
            if mode != 'random':
                state = engine.observe(state, action, after)
            obs = after
        results.append({'seed': seed, 'success': event == 1, 'game_over': event == 2,
                        'truncated': event == 0 and not obs.terminal, 'actions': actions,
                        'decision_seconds': decision_seconds})
        if (index + 1) % 16 == 0:
            emit('behavior_evaluation_progress', split=split, round=round_index, role=role, policy=mode, horizon=planner.config.horizon,
                 episodes=index + 1, wins=sum(r['success'] for r in results))
    total_actions = sum(r['actions'] for r in results)
    wins = sum(r['success'] for r in results)
    seconds = time.monotonic() - started
    # Wilson interval over episodes, not over pixels or actions.
    n, z = len(results), 1.96
    rate = wins / n
    center = (rate + z*z/(2*n)) / (1 + z*z/n)
    radius = z * (rate*(1-rate)/n + z*z/(4*n*n))**.5 / (1+z*z/n)
    report = {'split': split, 'round': round_index, 'role': role, 'policy': mode, 'horizon': planner.config.horizon, 'episodes': n, 'wins': wins,
              'win_rate': rate, 'wilson95': [center-radius, center+radius], 'actions': total_actions,
              'actions_per_success': total_actions / wins if wins else None,
              'seconds': seconds, 'actions_per_second': total_actions / max(seconds, 1e-9),
              'decision_ms': 1000 * sum(r['decision_seconds'] for r in results) / max(total_actions, 1),
              'results': results}
    emit('behavior_evaluation', **{k: v for k, v in report.items() if k != 'results'})
    return report


def save_heads(path, heads, source, config, report):
    checkpoint.write(path, {'weights': heads.state_dict(), 'source_fingerprint': source,
                            'planner': asdict(config), 'report': report})


def load_heads(path, heads, source, *, game=None, level=None):
    payload = checkpoint.load(path)
    if payload['source_fingerprint'] != source:
        raise ValueError('Behavior checkpoint belongs to different world-model weights')
    context = payload['report'].get('config', {})
    if game is not None and (context.get('game') != game or context.get('level') != level):
        raise ValueError('Behavior continuation requires the same game and level')
    heads.load_state_dict(payload['weights'])
    return payload


def selection_key(result):
    return result['wins'], -result['actions'], -result['decision_ms']


def copy_weights(heads):
    return {k: v.detach().cpu().clone() for k, v in heads.state_dict().items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--game', default='sc01-v1')
    parser.add_argument('--level', type=int, default=2)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--train-episodes', type=int, default=128)
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-records', type=int, default=8192)
    parser.add_argument('--steps', type=int, default=128)
    parser.add_argument('--dev-episodes', type=int, default=32)
    parser.add_argument('--final-episodes', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--horizons', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--beam', type=int, default=4)
    parser.add_argument('--seed-offset', type=int, default=0,
                        help='Use fresh disjoint train/dev/final seeds for another experiment')
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument('--heads', type=Path, help='Saved behavior heads for evaluation only')
    initialization.add_argument('--init-heads', type=Path, help='Initialize training weights; fresh optimizer and real experience')
    args = parser.parse_args(argv)
    if args.game not in SYNTHETIC_LEVELS or args.level not in range(1, SYNTHETIC_LEVELS.get(args.game, 0) + 1):
        parser.error('This prototype admits bundled synthetic training games only; public-game split admission is not implemented')
    for key in ('train_episodes', 'rounds', 'epochs', 'batch_size', 'max_records', 'steps', 'dev_episodes', 'final_episodes'):
        if getattr(args, key) < 1:
            parser.error(f'{key} must be positive')
    if args.output.resolve() == args.checkpoint.resolve().parent:
        parser.error('Behavior output must be separate from the world-model checkpoint directory')
    PlanConfig(beam=args.beam)
    for h in args.horizons:
        PlanConfig(horizon=h)
    if not 0 < args.learning_rate < 1:
        parser.error('learning rate must lie in (0, 1)')
    for split, count in [('train', args.train_episodes * args.rounds),
                         ('dev', args.dev_episodes), ('final', args.final_episodes)]:
        episode_seeds(split, count, args.seed_offset)
    if (args.output / 'heads.pt').exists() or (args.output / 'report.json').exists():
        parser.error('Output already contains an experiment; choose a fresh output directory')
    torch.set_num_threads(4)
    torch.manual_seed(123)
    payload = checkpoint.load(args.checkpoint)
    net = WorldModel(NetworkConfig(**payload['network'])).to(args.device)
    net.load_state_dict(payload['weights'])
    engine = Engine(net, TrainConfig(**payload['training']), compile_model=args.device == 'cuda')
    freeze(engine)
    source = fingerprint(net)
    args.output.mkdir(parents=True, exist_ok=True)
    source_actions = payload['progress']['actions']
    del payload
    factory = Factory(args.game, Path(__file__).resolve().parents[1] / 'games')
    heads = GoalValue(net.config.latent, net.config.memory).to(args.device)
    config = PlanConfig(beam=args.beam)
    initial_weights, initial_policy, initial_config = None, None, None
    if args.init_heads:
        saved = load_heads(args.init_heads, heads, source, game=args.game, level=args.level)
        initial_config = PlanConfig(**saved['planner'])
        initial_policy = saved['report']['selected']['policy']
        if initial_policy not in {'q', 'mpc'}:
            raise ValueError('Unknown initial behavior policy')
        initial_weights = copy_weights(heads)
        config = replace(initial_config, beam=args.beam)
    planner = Planner(engine, heads, config, compile_heads=args.device == 'cuda')
    records = deque(maxlen=args.max_records)
    report = {'source_fingerprint': source, 'source_actions': source_actions,
              'config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'development': [], 'collection': []}
    if initial_weights is not None:
        report['initial_head_fingerprint'] = fingerprint(heads)
    emit('behavior_started', game=args.game, level=args.level, device=args.device,
         compile=args.device == 'cuda', mode='evaluation_only' if args.heads else 'training',
         source=source, source_actions=source_actions, world_trainable=sum(p.numel() for p in net.parameters() if p.requires_grad))
    if args.heads:
        saved = load_heads(args.heads, heads, source)
        planner.config = PlanConfig(**saved['planner'])
        seeds = episode_seeds('final', args.final_episodes, args.seed_offset)
        mode = saved['report']['selected']['policy']
        report['selected'] = saved['report']['selected']
        report['final_random'] = evaluate(engine, planner, factory, args.level, seeds, args.steps, 'random', split='evaluation_only')
        report['final_learned'] = evaluate(engine, planner, factory, args.level, seeds, args.steps, mode, split='evaluation_only')
        report['world_unchanged'] = fingerprint(net) == source
        if not report['world_unchanged']:
            raise RuntimeError('Frozen world-model invariant violated')
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        emit('behavior_finished', evaluation_only=True, world_unchanged=True,
             random_wins=report['final_random']['wins'], learned_wins=report['final_learned']['wins'])
        return
    development = episode_seeds('dev', args.dev_episodes, args.seed_offset)
    baseline = evaluate(engine, planner, factory, args.level, development, args.steps, 'random', split='development')
    report['development_random'] = baseline
    best, best_weights, chosen, best_config = None, None, None, None
    if initial_weights is not None:
        planner.config = initial_config
        result = evaluate(engine, planner, factory, args.level, development, args.steps, initial_policy,
                          split='development', round_index=0, role='initial')
        result['iteration'] = 0
        report['development_initial'] = result
        best, best_weights, chosen, best_config = selection_key(result), initial_weights, result, initial_config
    for iteration in range(args.rounds):
        # Random only when no learned initialization exists. Later rounds use
        # the best development policy with epsilon exploration. No eval reuse.
        policy = planner if best_weights is not None else None
        if best_weights is not None:
            heads.load_state_dict(best_weights)
            planner.config = best_config
        collection = collect(engine, factory, args.level,
                             episode_seeds('train', args.train_episodes, args.seed_offset + iteration * args.train_episodes),
                             args.steps, config.discount, records, policy,
                             mode=chosen['policy'] if chosen else 'q', round_index=iteration + 1)
        report['collection'].append(collection)
        fit(engine, heads, records, args.epochs, args.batch_size, args.learning_rate,
            123 + iteration, compiled=args.device == 'cuda', round_index=iteration + 1)
        for mode, horizon in [('q', 1)] + [('mpc', h) for h in args.horizons]:
            planner.config = replace(config, horizon=horizon)
            result = evaluate(engine, planner, factory, args.level, development, args.steps, mode, split='development', round_index=iteration + 1)
            result['iteration'] = iteration + 1
            report['development'].append(result)
            key = selection_key(result)
            if best is None or key > best:
                best, chosen, best_config = key, result, planner.config
                best_weights = copy_weights(heads)
        heads.load_state_dict(best_weights)
        report['selected'] = {k: chosen[k] for k in ('iteration', 'policy', 'horizon')}
        report['head_fingerprint'] = fingerprint(heads)
        save_heads(args.output / 'heads.pt', heads, source, best_config, report)
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    heads.load_state_dict(best_weights)
    planner.config = best_config
    final_seeds = episode_seeds('final', args.final_episodes, args.seed_offset)
    report['final_random'] = evaluate(engine, planner, factory, args.level, final_seeds, args.steps, 'random', split='final')
    if initial_weights is not None:
        heads.load_state_dict(initial_weights)
        planner.config = initial_config
        report['final_initial'] = evaluate(engine, planner, factory, args.level, final_seeds, args.steps,
                                          initial_policy, split='final', role='initial')
        heads.load_state_dict(best_weights)
        planner.config = best_config
    report['final_learned'] = evaluate(engine, planner, factory, args.level, final_seeds, args.steps, chosen['policy'], split='final')
    report['world_unchanged'] = fingerprint(net) == source
    if not report['world_unchanged'] or any(p.grad is not None for p in net.parameters()):
        raise RuntimeError('Frozen world-model invariant violated')
    save_heads(args.output / 'heads.pt', heads, source, planner.config, report)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    emit('behavior_finished', selected=report['selected'], world_unchanged=True,
         random_wins=report['final_random']['wins'], learned_wins=report['final_learned']['wins'])


if __name__ == '__main__':
    main()
