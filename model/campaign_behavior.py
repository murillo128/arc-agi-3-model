"""Bounded, balanced behavior experiments using the same frozen agent modules.

This module owns no game rules, deployment adapter, checkpoint publication or
held-out split admission. Its factories are supplied by the campaign controller.
"""
from collections import deque
from copy import deepcopy
from dataclasses import replace
import random
import time

import torch

from .behavior_experiment import collect, episode_seeds, fit
from .planning import PlanConfig, Planner, fingerprint, freeze
from .proposals import ProposalPolicy, TeacherRecord, imagined_loss, imitation_loss
from .value_learning import update_target


class TeacherPlanner:
    """Capture only decisions actually selected by MPC, before environment.step.

    Epsilon/random decisions are made outside choose by collect, so they cannot
    silently become imitation labels. No imagined successor enters this store.
    """
    def __init__(self, planner, records):
        self.planner, self.records = planner, records

    def __getattr__(self, name):
        return getattr(self.planner, name)

    def choose(self, state, observation, rng, mode='value_mpc', diagnostics=None, *, remaining_budget=None):
        if mode not in {'mpc', 'value_mpc'}:
            raise ValueError('Teacher labels require an MPC decision')
        action = self.planner.choose(state, observation, rng, mode, diagnostics,
                                     remaining_budget=remaining_budget)
        self.records.append(TeacherRecord(state.z.detach().to('cpu', torch.float16),
            state.h[0].detach().to('cpu', torch.float16), state.geometry,
            tuple(observation.available_actions), remaining_budget, action, 'mpc_teacher'))
        return action


def fit_proposals(engine, heads, policy, records, optimizer, *, imagined=False, seed=0, epochs=12,
                  batch_size=32, progress_callback=None, target_policy=None):
    if not records:
        return {'updates': 0, 'loss': None, 'teacher_records': 0}
    freeze(engine)
    heads.eval().requires_grad_(False)
    policy.train()
    forward = torch.compile(policy, fullgraph=True, dynamic=True) if str(engine.device).startswith('cuda') else policy
    target_policy = target_policy or deepcopy(policy).eval().requires_grad_(False)
    target_forward = (torch.compile(target_policy, fullgraph=True, dynamic=True)
                      if str(engine.device).startswith('cuda') else target_policy)
    heads_forward = (torch.compile(heads, fullgraph=True, dynamic=True)
                     if imagined and str(engine.device).startswith('cuda') else heads)
    groups = {}
    for row in records:
        groups.setdefault(row.geometry, []).append(row)
    rng, updates, total_loss, count = random.Random(seed), 0, 0., 0
    metrics = {}
    for epoch in range(epochs):
        for rows in groups.values():
            indices = list(range(len(rows)))
            rng.shuffle(indices)
            for start in range(0, len(rows), batch_size):
                batch = [rows[indices[i % len(indices)]] for i in range(start, start + batch_size)]
                optimizer.zero_grad(set_to_none=True)
                with engine.autocast():
                    teacher_loss = imitation_loss(policy, batch, engine.device, forward)
                    if imagined:
                        loss, metrics = imagined_loss(engine, heads, policy, batch, horizon=2, forward=forward,
                                                       target_policy=target_forward, heads_forward=heads_forward)
                        loss = loss + .1 * teacher_loss
                    else:
                        loss = teacher_loss
                        metrics = {'imitation_loss': float(teacher_loss.detach())}
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.)
                optimizer.step()
                update_target(target_policy, policy, .005)
                updates += 1
                total_loss += float(loss.detach()) * len(batch)
                count += len(batch)
        if progress_callback is not None:
            progress_callback({'epoch': epoch + 1, 'updates': updates, 'loss': total_loss / count, **metrics})
    policy.eval()
    return {'updates': updates, 'loss': total_loss / count, 'teacher_records': len(records), **metrics}


def train_behavior(engine, heads, config: PlanConfig, factories, pairs, total_actions=16384, seed=0,
                   variant='mc', emit=None, check_callback=None):
    """Train a bounded stage; return (planner, optional proposal policy, report).

    check_callback(actions, planner, proposal_policy, report) runs at each 4096
    action boundary and at the final boundary. Evaluation callbacks must not
    mutate modules or reuse training seeds. Events use emit(name, **fields).

    Actor variants are an additional stage from a common calibrated MPC head
    checkpoint. The caller supplies that same checkpoint to both actor variants
    and their control comparison; it is not fitted during actor training.
    """
    supported = {'mc', 'double_q', 'click_mixed', 'availability', 'actor_imitation', 'actor_imagined'}
    if variant not in supported or type(total_actions) is not int or total_actions < 1:
        raise ValueError('Unknown variant or nonpositive action budget')
    if not pairs or len(set(pairs)) != len(pairs) or any(game not in factories for game, _ in pairs):
        raise ValueError('Expected unique admitted game/level pairs with factories')
    if type(seed) is not int or not 0 <= seed <= 2:
        raise ValueError('Campaign seed must lie in [0, 2] for disjoint training partitions')
    if total_actions > 32768 or len(pairs) > 10:
        raise ValueError('This bounded stage supports at most 32768 actions and ten pairs')
    emit = emit or (lambda event, **data: None)
    freeze(engine)
    source = fingerprint(engine.model)
    original_heads = fingerprint(heads)
    torch.manual_seed(123 + seed)
    actor_stage = variant.startswith('actor_')
    settings = replace(config, clicks=max(16, config.clicks),
                       click_proposals='mixed' if variant == 'click_mixed' else 'uniform',
                       future_availability=False,
                       leaf_value='q' if variant == 'double_q' else 'v')
    planner = Planner(engine, heads, settings, compile_heads=str(engine.device).startswith('cuda'))
    policy = (ProposalPolicy(engine.model.config.latent, engine.model.config.memory,
                             patch=engine.model.config.patch).to(engine.device) if actor_stage else None)
    optimizer = torch.optim.AdamW((policy if actor_stage else heads).parameters(), lr=3e-4, weight_decay=.001)
    target = deepcopy(heads).eval().requires_grad_(False) if variant == 'double_q' else None
    target_policy = deepcopy(policy).eval().requires_grad_(False) if actor_stage else None
    capacity = max(1, 1024 // len(pairs))
    records = {pair: deque(maxlen=capacity) for pair in pairs}
    teachers = {pair: deque(maxlen=max(1, 512 // len(pairs))) for pair in pairs}
    quotas = {pair: total_actions // len(pairs) + int(i < total_actions % len(pairs)) for i, pair in enumerate(pairs)}
    counts, episodes = {pair: 0 for pair in pairs}, {pair: 0 for pair in pairs}
    actions = updates = rounds = 0
    boundaries = set(range(4096, total_actions, 4096)) | {total_actions}
    if actor_stage and total_actions > 1:
        boundaries.add(total_actions // 2)
    checkpoints = iter(sorted(boundaries))
    started, next_check = time.monotonic(), next(checkpoints)
    report = {'variant': variant, 'seed': seed, 'total_actions': total_actions, 'actions': 0,
              'updates': 0, 'world_fingerprint': source, 'source_heads': original_heads,
              'value_target': 'double_q' if variant == 'double_q' else 'mc',
              'teacher_provenance': 'mpc_teacher' if actor_stage else None, 'checks': [],
              'per_pair': [], 'head_updates': not actor_stage}
    while actions < total_actions:
        for pair_index, pair in enumerate(pairs):
            if actions >= next_check:
                break
            quota = min(128, quotas[pair] - counts[pair], next_check - actions)
            if quota <= 0:
                continue
            game, level = pair
            offset = seed * 330000 + pair_index * 32768 + episodes[pair]
            seeds = episode_seeds('train', quota, offset)
            active = TeacherPlanner(planner, teachers[pair]) if actor_stage else planner
            collected = collect(engine, factories[game], level, seeds, 128, settings.discount, records[pair],
                                policy=active, mode='value_mpc', epsilon=.25, action_limit=quota)
            if collected['actions'] == 0:
                raise RuntimeError('Admitted environment produced no actionable transition')
            actions += collected['actions']
            counts[pair] += collected['actions']
            episodes[pair] += collected['episodes']
            emit('campaign_progress', stage='actor' if actor_stage else 'behavior', variant=variant, seed=seed,
                 actions=actions, total_actions=total_actions, game=game, level=level, updates=updates,
                 actions_per_second=actions / max(time.monotonic() - started, 1e-9), loss=None)
        if actions < next_check:
            continue
        rounds += 1
        def progress(data):
            loss = {k: v for k, v in data.items()
                    if k == 'loss' or k.endswith('_loss') or k in {'event_ce', 'value_bce', 'q_bce', 'entropy'}}
            emit('campaign_progress', stage='actor' if actor_stage else 'behavior', variant=variant, seed=seed,
                 actions=actions, total_actions=total_actions, game=None, level=None,
                 updates=updates + data['updates'], actions_per_second=actions / max(time.monotonic() - started, 1e-9),
                 loss=loss, **{k: v for k, v in data.items() if k != 'updates' and k not in loss})
        if actor_stage:
            # The first half is pure teacher imitation; later collection may use
            # actor proposals while MPC still ranks and selects the real action.
            imagined = variant == 'actor_imagined' and actions > total_actions // 2
            learned = fit_proposals(engine, heads, policy, [r for rows in teachers.values() for r in rows],
                                    optimizer, imagined=imagined, seed=seed + rounds, progress_callback=progress,
                                    target_policy=target_policy)
            if actions >= total_actions // 2:
                planner.proposal_policy = policy
        else:
            learned = fit(engine, heads, [r for rows in records.values() for r in rows], 12, 32, 3e-4,
                          seed + rounds, compiled=str(engine.device).startswith('cuda'), value_target=report['value_target'],
                          discount=settings.discount, train_availability=variant == 'availability',
                          optimizer=optimizer, target_head=target, progress_callback=progress)
        updates += learned['updates']
        report.update(actions=actions, updates=updates, loss=learned['loss'], seconds=time.monotonic() - started,
                      per_pair=[{'game': game, 'level': level, 'actions': counts[(game, level)],
                                 'episodes': episodes[(game, level)]} for game, level in pairs])
        report['checks'].append({'actions': actions, 'updates': updates, 'loss': learned['loss']})
        if fingerprint(engine.model) != source:
            raise RuntimeError('Behavior training changed frozen world weights')
        if actor_stage and fingerprint(heads) != original_heads:
            raise RuntimeError('Actor training changed its fixed MPC evaluator')
        if check_callback is not None and (actions % 4096 == 0 or actions == total_actions):
            check_callback(actions, planner, policy, report)
        next_check = next(checkpoints, total_actions)
    report['world_unchanged'] = True
    report['teacher_records'] = sum(map(len, teachers.values()))
    report['heads_unchanged'] = fingerprint(heads) == original_heads
    report['seconds'] = time.monotonic() - started
    return planner, policy, report
