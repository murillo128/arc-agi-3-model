"""Bounded, real-environment A/B probe; never mutates the source checkpoint.

python -m model.paired_click_experiment --source COPY.pt --output IGNORED_DIR
Alternative clicks are replayed from public actions, not environment snapshots.
"""
import argparse
import copy
from dataclasses import replace
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from . import checkpoint
from .config import NetworkConfig, TrainConfig, SequenceConfig
from .data import Action
from .environment import Factory
from .game_sequences import (GameSession, Transition, EncodedSessions, SequenceLearner,
    collect_session, memory_inputs)
from .learning import Engine
from .networks import WorldModel
from .sequence_exploration import ChangeExplorer, SequenceExplorationConfig, visible_clicks


def same_observation(a, b):
    return (np.array_equal(a.frames, b.frames) and a.state == b.state and
            a.available_actions == b.available_actions and
            (a.levels_completed, a.win_levels) == (b.levels_completed, b.win_levels))


def alternatives(observation, original, rng, explorer, count=4):
    """Original, diverse visible candidates, then a background control."""
    if 6 not in observation.available_actions or original.id != 6:
        raise ValueError('Paired probe requires a real allowed click')
    candidates = visible_clicks(observation, rng, 256)
    scores = explorer.probabilities(observation, candidates) if candidates else []
    ranked = [a for _, a in sorted(zip(scores, candidates), key=lambda pair: -pair[0])]
    result = [original]
    for action in ranked:
        if all((action.data['x']-a.data['x'])**2 +
               (action.data['y']-a.data['y'])**2 >= 64 for a in result):
            result.append(action)
        if len(result) == count-1:
            break
    frame = observation.final
    color = int(np.bincount(frame.ravel(), minlength=16).argmax())
    background = list(zip(*np.nonzero(frame == color)))
    rng.shuffle(background)
    for y, x in background:
        action = Action(6, dict(x=int(x), y=int(y)))
        if action not in result:
            result.append(action)
        if len(result) == count:
            return result
    raise ValueError('Not enough distinct public click locations')


def replay_branch(factory, session, action):
    """Rebuild the same real prefix, checking every observed frame and boundary."""
    original = session.transitions[-1]
    level = original.level_number
    env, attempts, calls = None, 0, 0
    for index, expected in enumerate(session.transitions):
        if env is None or env.observation.terminal:
            env = factory.make(session.seed + attempts*1_000_000, level,
                               palette_seed=session.seed)
            attempts += 1
        before = env.observation
        if before.state == 'NOT_PLAYED':
            before = env.step(Action(0)).level_initial()
            calls += 1
        if not same_observation(before, expected.before):
            raise ValueError(f'Replay diverged before transition {index}')
        selected = action if index == len(session.transitions)-1 else expected.action
        after = env.step(selected)
        calls += 1
        if index < len(session.transitions)-1:
            if not same_observation(after, expected.after):
                raise ValueError(f'Replay diverged after transition {index}')
            if after.starts_new_level(before):
                env.observation = after.level_initial()
    # Reuse verified immutable prefix observations: encoding is shared, memory is not.
    branch = GameSession(session.game, session.seed,
        [*session.transitions[:-1], Transition(original.before, action, after,
            original.reset_before, original.level)], environment_actions=calls)
    return branch


def seed_sets(train_per_level, validation_per_level):
    if min(train_per_level, validation_per_level) < 1:
        raise ValueError('Both seed sets must be nonempty')
    return ({level: [4_400_000 + 2*(level*1000+i) for i in range(train_per_level*4)]
             for level in range(1, 7)},
            {level: [4_600_001 + 2*(level*1000+i) for i in range(validation_per_level)]
             for level in range(1, 7)})


def collect_data(saved, *, train_per_level=8, validation_per_level=4, context=16):
    config = SequenceExplorationConfig(**saved['exploration']['config'])
    explorer = ChangeExplorer(config)
    explorer.restore(saved['exploration'])
    sequence = SequenceConfig(session_steps=context+1, level_steps=context+2,
        block_steps=1, context_steps=context, loss_start_weight=1.)
    seeds, held = seed_sets(train_per_level, validation_per_level)
    groups, normal, validation, ordinary, guards = [], [], [], [], []
    counters = dict(original_actions=0, replay_actions=0, original_sessions=0,
                    replay_branches=0)
    factory = Factory('bf02-v1', Path(__file__).resolve().parents[1]/'games')
    def scene(seed, level, paired):
        row = collect_session(factory, [level], seed, sequence,
                              exploration=config, explorer=explorer)
        counters['original_actions'] += row.environment_actions
        counters['original_sessions'] += 1
        if len(row.transitions) != context+1 or not row.transitions[-1].supervised:
            raise ValueError('Incomplete or unsupervised anchor')
        if not paired:
            return [row]
        actions = alternatives(row.transitions[-1].before, row.transitions[-1].action,
                               random.Random(seed+91), explorer)
        result = [row]
        for action in actions[1:]:
            branch = replay_branch(factory, row, action)
            counters['replay_actions'] += branch.environment_actions
            counters['replay_branches'] += 1
            result.append(branch)
        return result
    try:
        for level in range(1, 7):
            for i, seed in enumerate(seeds[level]):
                rows = scene(seed, level, i < train_per_level)
                normal.append(rows[0])
                if i < train_per_level:
                    groups.append(rows)
            for seed in held[level]:
                validation.extend(scene(seed, level, True))
            row = collect_session(factory, [level], 4_800_001+level*200,
                replace(sequence, session_steps=64, level_steps=65),
                exploration=config, explorer=explorer)
            ordinary.append(row)
            counters['original_actions'] += row.environment_actions
    finally:
        factory.arcade.close_scorecard()
    for game in ('bf01-v1', 'bf03-v1', 'sc01-v1'):
        factory = Factory(game, Path(__file__).resolve().parents[1]/'games')
        try:
            for level in (1, 3):
                row = collect_session(factory, [level], 4_900_001+level*200,
                    replace(sequence, session_steps=32, level_steps=33),
                    exploration=config, explorer=explorer)
                guards.append(row)
                counters['original_actions'] += row.environment_actions
        finally:
            factory.arcade.close_scorecard()
    assert len(normal) == sum(map(len, groups))
    composition = {}
    for name, rows in (('normal', normal), ('paired', [s for g in groups for s in g]),
                       ('validation', validation)):
        changed = [int(np.count_nonzero(s.transitions[-1].before.final !=
                                       s.transitions[-1].after.final)) for s in rows]
        composition[name] = dict(examples=len(rows), changed_examples=sum(v > 0 for v in changed),
                                 changed_pixels=sum(changed))
    return dict(normal=normal, paired=groups, validation=validation, ordinary=ordinary,
                guards=guards, counters=counters, composition=composition, context=context)


def encode_cache(engine, data):
    observations = {}
    rows = data['normal'] + [s for group in data['paired'] for s in group]
    rows += data['validation'] + data['ordinary'] + data['guards']
    for row in rows:
        for step in row.transitions:
            for obs in (step.before, step.after):
                if obs.final is None:
                    raise ValueError('This bounded probe requires observed frames')
                observations[id(obs)] = obs
    items, cache = list(observations.items()), {}
    engine.model.eval()
    with torch.no_grad(), engine.autocast():
        for start in range(0, len(items), engine.config.frame_chunk):
            chunk = items[start:start+engine.config.frame_chunk]
            for (key, _), value in zip(chunk, engine.encode([o for _, o in chunk])):
                cache[key] = value.detach().cpu()
    return cache


def cached_sessions(engine, sessions, cache):
    """Only this small batch's latents live on the device; no stale-weight reuse."""
    encoded = EncodedSessions.__new__(EncodedSessions)
    encoded.sessions = sessions
    ids = {id(o) for row in sessions for s in row.transitions for o in (s.before, s.after)}
    encoded.latents = {key: cache[key].to(engine.device) for key in ids}
    encoded.encoder_versions = tuple(p._version for p in engine.model.encoder.parameters())
    encoded.context = [[(encoded.z(s.before), tuple(s.before.final.shape))
                        for s in row.transitions] for row in sessions]
    return encoded


def counts(before, actual, predicted):
    real, guessed, correct = actual != before, predicted != before, predicted == actual
    union = real | guessed
    return dict(transitions=1, changed=int(real.sum()), union=int(union.sum()),
        wrong=int((union & ~correct).sum()), correct=int((real & correct).sum()),
        missed=int((real & ~guessed).sum()), invented=int((~real & guessed).sum()),
        color=int((real & guessed & ~correct).sum()))


def add_counts(total, row):
    for key, value in row.items():
        total[key] = total.get(key, 0) + value


def summarize(total):
    return dict(total, error=total['wrong']/max(1, total['union']),
                recall=total['correct']/max(1, total['changed']),
                invented_per_action=total['invented']/max(1, total['transitions']))


@torch.no_grad()
def evaluate(engine, sessions, cache, *, anchor_only=False, batch_size=8):
    engine.model.eval()
    total, levels, games, per_scene = {}, {}, {}, []
    for start in range(0, len(sessions), batch_size):
        batch = sessions[start:start+batch_size]
        encoded = cached_sessions(engine, batch, cache)
        memories = [torch.zeros(1, engine.model.config.memory, device=engine.device) for _ in batch]
        with engine.autocast():
            for t in range(max(len(s.transitions) for s in batch)):
                for ids, steps, z, h, action, geometry in memory_inputs(engine, encoded, t, memories):
                    if anchor_only and t < len(batch[0].transitions)-1:
                        next_h, _ = engine.advance(z, h, action)
                        pixels = None
                    else:
                        predicted, next_h, _ = engine.predict(z, h, action, *geometry)
                        pixels = engine.decode(predicted, *geometry).argmax(-1).cpu().numpy()
                    for k, (i, step, nh) in enumerate(zip(ids, steps, next_h.split(1))):
                        memories[i] = nh.float()
                        if pixels is None or not step.supervised:
                            continue
                        row = counts(step.before.final, step.after.final, pixels[k])
                        add_counts(total, row)
                        add_counts(levels.setdefault(str(step.level_number), {}), row)
                        add_counts(games.setdefault(batch[i].game, {}), row)
                        per_scene.append(dict(game=batch[i].game, seed=batch[i].seed,
                            level=step.level_number, t=t, **row))
    return dict(all=summarize(total), levels={k:summarize(v) for k,v in levels.items()},
                games={k:summarize(v) for k,v in games.items()}, scenes=per_scene)


def batches_for(data, variant, updates, batch_size, seed=124):
    if batch_size % 4:
        raise ValueError('Batch size must be divisible by four')
    rng = random.Random(seed)
    normal = variant.startswith('normal')
    if 'balanced' in variant:
        changed = lambda s: not np.array_equal(s.transitions[-1].before.final, s.transitions[-1].after.final)
        pools = {}
        for level in range(1, 7):
            if normal:
                rows = [s for s in data['normal'] if s.transitions[-1].level_number == level]
                pools[level] = ([s for s in rows if changed(s)], [s for s in rows if not changed(s)])
                if not all(pools[level]): raise ValueError(f'No effect/control samples at level {level}')
            else:
                groups = [g for g in data['paired'] if g[0].transitions[-1].level_number == level]
                pools[level] = [([s for s in g if changed(s)], [s for s in g if not changed(s)]) for g in groups]
                pools[level] = [g for g in pools[level] if all(g)]
                if not pools[level]: raise ValueError(f'No paired effect/control scene at level {level}')
        # Independent of variant-dependent random draws: identical level counts per update.
        level_rng = random.Random(f'{seed}:levels')
        levels = []
        while len(levels) < updates*batch_size//2:
            order = list(range(1, 7)); level_rng.shuffle(order); levels.extend(order)
        for index in range(updates):
            rows = []
            for level in levels[index*batch_size//2:(index+1)*batch_size//2]:
                positive, negative = pools[level] if normal else rng.choice(pools[level])
                rows.extend((rng.choice(positive), rng.choice(negative)))
            yield rows
        return
    source = data['normal'] if normal else data['paired']
    take = batch_size if normal else batch_size//4
    if len(source) < take:
        raise ValueError('Not enough training scenes')
    pending = []
    for _ in range(updates):
        if len(pending) < take:
            order = list(range(len(source))); rng.shuffle(order); pending.extend(order)
        selected, pending = pending[:take], pending[take:]
        yield ([source[i] for i in selected] if normal
               else [s for i in selected for s in source[i]])


def decision(baseline, candidate, initial=None):
    initial = baseline if initial is None else initial
    a, b = baseline['paired'], candidate['paired']
    checks = dict(recall=b['all']['recall'] >= a['all']['recall']+.05,
        error=b['all']['error'] <= a['all']['error']-.03,
        levels=sum(b['levels'][l]['error'] < a['levels'][l]['error'] for l in a['levels']) >= 4,
        ordinary=candidate['ordinary']['all']['error'] <= baseline['ordinary']['all']['error']+.01,
        inventions=b['all']['invented_per_action'] <= a['all']['invented_per_action'] +
                   max(.5, .2*a['all']['invented_per_action']),
        retained_games=candidate['guards']['all']['error'] <= baseline['guards']['all']['error']+.02,
        source_recall=b['all']['recall'] >= initial['paired']['all']['recall'],
        source_ordinary=candidate['ordinary']['all']['error'] <= initial['ordinary']['all']['error']+.01,
        source_retained_games=candidate['guards']['all']['error'] <= initial['guards']['all']['error']+.02)
    return dict(passed=all(checks.values()), checks=checks)


def run(args):
    if min(args.updates, args.cpu_threads, args.train_scenes, args.validation_scenes) < 1 or args.context < 0:
        raise ValueError('Positive budgets and nonnegative context are required')
    if args.batch_size < 4 or args.batch_size % 4:
        raise ValueError('Batch size must be a positive multiple of four')
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new output directory')
    if output == args.source.resolve().parent:
        raise ValueError('Experiment output must be separate from the source run')
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.cpu_threads)
    saved = checkpoint.load(args.source)
    started = time.perf_counter()
    data = collect_data(saved, train_per_level=args.train_scenes,
                        validation_per_level=args.validation_scenes, context=args.context)
    print(json.dumps(dict(event='collected', counters=data['counters'],
                         composition=data['composition'],
                         seconds=time.perf_counter()-started)), flush=True)
    engine = Engine(WorldModel(NetworkConfig(**saved['network'])).to(args.device),
                    TrainConfig(**saved['training']))
    engine.model.load_state_dict(saved['weights'])
    cache = encode_cache(engine, data)
    # Every later arm restores the same frozen encoder and reads cached latents.
    # Its weights need no GPU residency during predictor updates or evaluation.
    if engine.device.type == 'cuda':
        engine.model.encoder.cpu()
        torch.cuda.empty_cache()
    config = SequenceConfig(batch_games=args.batch_size, session_steps=args.context+1,
        block_steps=1, context_steps=args.context, loss_start_weight=1.)
    report = dict(source=str(args.source.resolve()), source_progress=saved['progress'],
        counters=data['counters'], composition=data['composition'],
        device=args.device, amp=engine.amp, context=args.context,
        batch_size=args.batch_size, updates=args.updates, train_scenes=args.train_scenes,
        validation_scenes=args.validation_scenes, trials={})
    def measured():
        return dict(paired=evaluate(engine, data['validation'], cache, anchor_only=True),
                    ordinary=evaluate(engine, data['ordinary'], cache),
                    guards=evaluate(engine, data['guards'], cache))
    report['initial'] = measured()
    print(json.dumps(dict(event='initial', paired=report['initial']['paired']['all'],
        ordinary=report['initial']['ordinary']['all'])), flush=True)
    baseline_name = 'normal_balanced' if args.balanced else 'normal'
    paired_name = 'paired_balanced' if args.balanced else 'paired'
    for variant in (baseline_name, paired_name) + ((paired_name+'_spatial_lr',) if args.spatial_lr else ()):
        engine.model.load_state_dict(saved['weights'])
        learner = SequenceLearner(engine, config)
        learner.optimizer.load_state_dict(copy.deepcopy(saved['predictor_optimizer']))
        if variant.endswith('_spatial_lr'):
            # Partition every restored group: adapter weights and biases have
            # different decay settings. Parameter-keyed Adam moments stay intact.
            adapter = set(engine.model.dynamics.rotation_adapter.parameters())
            for group in list(learner.optimizer.param_groups):
                local = [p for p in group['params'] if p in adapter]
                if not local:
                    continue
                group['params'] = [p for p in group['params'] if p not in adapter]
                learner.optimizer.add_param_group({**{k:v for k,v in group.items() if k != 'params'},
                                                   'params':local, 'lr':1e-4})
        torch.manual_seed(124)
        trace = []; begin = time.perf_counter()
        if engine.device.type == 'cuda': torch.cuda.reset_peak_memory_stats()
        for index, rows in enumerate(batches_for(data, variant, args.updates, args.batch_size)):
            encoded = cached_sessions(engine, rows, cache)
            result = learner.update(encoded, args.context, args.context+1)
            result['changed_examples'] = sum(not np.array_equal(s.transitions[-1].before.final,
                                               s.transitions[-1].after.final) for s in rows)
            trace.append(result)
            if (index+1) % 6 == 0:
                print(json.dumps(dict(event='update', variant=variant, update=index+1, **result)), flush=True)
        if engine.device.type == 'cuda': torch.cuda.synchronize()
        elapsed = time.perf_counter()-begin
        result = measured()
        frozen = all(torch.equal(v.detach().cpu(), saved['weights'][k])
                     for k,v in engine.model.state_dict().items() if k.startswith(('encoder.', 'decoder.')))
        if not frozen: raise RuntimeError('Representation weights changed')
        result.update(seconds=elapsed, trace=trace, frozen_representation=frozen,
            peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3 if engine.device.type=='cuda' else None)
        report['trials'][variant] = result
        if variant != baseline_name:
            result['decision'] = decision(report['trials'][baseline_name], result, report['initial'])
        if result.get('decision', {}).get('passed'):
            checkpoint.write(output/f'candidate-{variant}.pt', dict(
                kind='paired_click_candidate', source_progress=saved['progress'],
                predictor=engine.model.dynamics.state_dict(), optimizer=learner.optimizer.state_dict(),
                optimizer_updates=args.updates, decision=result['decision']))
        (output/'results.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(dict(event='evaluated', variant=variant, paired=result['paired']['all'],
            ordinary=result['ordinary']['all'], guards=result['guards']['all'],
            decision=result.get('decision'), seconds=elapsed)), flush=True)
        del learner
        if engine.device.type == 'cuda': torch.cuda.empty_cache()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu', choices=('cpu', 'cuda'))
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--train-scenes', type=int, default=8)
    parser.add_argument('--validation-scenes', type=int, default=4)
    parser.add_argument('--context', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--updates', type=int, default=24)
    parser.add_argument('--spatial-lr', action='store_true')
    parser.add_argument('--balanced', action='store_true', help='Match level and effect/control counts in both arms')
    run(parser.parse_args())
