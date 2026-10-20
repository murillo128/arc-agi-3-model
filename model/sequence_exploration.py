"""A cheap auxiliary action-change prior, learned only from real observations.

This is not a success value or a world-model loss. Uniform exploration and all
real no-op transitions remain available. The head never updates the world model.
"""
from collections import deque
from dataclasses import asdict, dataclass
import math
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import Action, random_action


@dataclass(frozen=True)
class SequenceExplorationConfig:
    policy: str = 'uniform'
    uniform_probability: float = .25
    candidate_clicks: int = 64
    learning_rate: float = .001
    updates: int = 16
    batch_size: int = 256
    replay_size: int = 8192
    noop_cooldown: int = 8

    def __post_init__(self):
        if self.policy not in ('uniform', 'mixed', 'learned'):
            raise ValueError('Exploration policy must be uniform, mixed or learned')
        for key in ('uniform_probability', 'learning_rate'):
            value = getattr(self, key)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value < 1:
                raise ValueError(f'{key} must be finite and between zero and one')
        for key in ('candidate_clicks', 'updates', 'batch_size', 'replay_size', 'noop_cooldown'):
            value = getattr(self, key)
            if type(value) is not int or value < 1:
                raise ValueError(f'{key} must be a positive integer')
        if (not 4 <= self.candidate_clicks <= 256 or self.batch_size > self.replay_size
                or self.noop_cooldown > 64):
            raise ValueError('Use 4..256 candidates, cooldown <= 64 and batch_size <= replay_size')


def visible_clicks(observation, rng, count):
    """Palette-balanced pixels, without a designated target/background color."""
    frame = observation.final
    if frame is None or not frame.size:
        return []
    visible = frame[:64, :64]
    colors, first, counts = np.unique(visible, return_index=True, return_counts=True)
    colors = [int(colors[i]) for i in np.argsort(first) if counts[i] < counts.max()]
    if not colors:
        return []
    pixels = {color: np.flatnonzero(visible == color) for color in colors}
    result = []
    for _ in range(count):
        color = rng.choice(colors)
        y, x = divmod(int(rng.choice(pixels[color])), visible.shape[1])
        result.append(Action(6, dict(x=x, y=y)))
    return result


def features(observation, actions, previous=None):
    """Palette-invariant public image statistics and native local neighborhoods.

    No game identity or known rules. Previous context is (actual action, visible
    change), not a future target. Equal colors remain equal after palette swaps.
    """
    frame = observation.final
    context = np.zeros(34, dtype=np.float32)
    context[18:25] = [i in observation.available_actions for i in range(1, 8)]
    if previous is not None:
        action, changed = previous
        if 1 <= action.id <= 7:
            context[25 + action.id - 1] = 1
        context[32:] = (float(changed), 1.)
    result = np.zeros((len(actions), 48), dtype=np.float32)
    if frame is not None and frame.size:
        height, width = frame.shape
        counts = np.bincount(frame.ravel(), minlength=16).astype(np.float32) / frame.size
        context[:16] = np.sort(counts)
        context[16:18] = height / 64, width / 64
    result[:, :34] = context
    for i, action in enumerate(actions):
        if 1 <= action.id <= 7:
            result[i, 34 + action.id - 1] = 1
        if action.id != 6:
            continue
        x, y = action.data['x'], action.data['y']
        result[i, 41:43] = x / 63, y / 63
        if frame is None or not frame.size or not (0 <= x < width and 0 <= y < height):
            continue
        color = frame[y, x]
        result[i, 43:46] = 1, counts[color], counts[color] / counts.max()
        for column, radius in ((46, 1), (47, 3)):
            patch = frame[max(0, y-radius):y+radius+1, max(0, x-radius):x+radius+1]
            result[i, column] = np.mean(patch == color)
    return result


def real_change(before, after):
    if before.final is None or after.final is None or before.final.shape != after.final.shape:
        return None
    return not np.array_equal(before.final, after.final)


def training_examples(sessions):
    for session in sessions:
        previous = None
        for step in session.transitions:
            if step.reset_before:
                previous = None
            changed = real_change(step.before, step.after)
            if step.supervised and not step.after.terminal:
                yield features(step.before, [step.action], previous)[0], float(changed)
            previous = (step.action, changed) if changed is not None else None


class ChangeHead(nn.Module):
    """Estimate visual change, not reward magnitude or probability of success."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(48, 32), nn.ReLU(), nn.Linear(32, 32),
                                    nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.layers(x).squeeze(-1)


class ChangeExplorer:
    def __init__(self, config):
        self.config = config
        # Auxiliary initialization must not advance the world model's RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(114)
            self.head = ChangeHead().cpu()
        self.optimizer = torch.optim.Adam(self.head.parameters(), lr=config.learning_rate)
        self.rng = np.random.default_rng(114)
        self.replay = deque(maxlen=config.replay_size)
        self.updates = self.examples = 0

    @torch.no_grad()
    def probabilities(self, observation, actions, previous=None):
        self.head.eval()
        values = self.head(torch.from_numpy(features(observation, actions, previous))).sigmoid()
        return values.numpy()

    def fit(self, sessions):
        rows = list(training_examples(sessions))
        self.examples += len(rows)
        self.replay.extend(rows)
        return self.fit_replay()

    def fit_replay(self, updates=None):
        if not self.replay:
            return dict(loss=None, updates=0, total_updates=self.updates, examples_seen=self.examples)
        x = torch.from_numpy(np.stack([r[0] for r in self.replay]))
        y = torch.tensor([r[1] for r in self.replay], dtype=torch.float32)
        losses = []
        self.head.train()
        for _ in range(self.config.updates if updates is None else updates):
            ids = self.rng.integers(len(x), size=min(len(x), self.config.batch_size))
            self.optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(self.head(x[ids]), y[ids])
            loss.backward()
            self.optimizer.step()
            losses.append(float(loss.detach()))
            self.updates += 1
        return dict(loss=sum(losses)/len(losses) if losses else None, updates=len(losses),
                    total_updates=self.updates, examples_seen=self.examples)

    def snapshot(self):
        return dict(version=1, config=asdict(self.config), weights=self.head.state_dict(),
            optimizer=self.optimizer.state_dict(), rng=self.rng.bit_generator.state,
            updates=self.updates, examples=self.examples,
            replay_features=torch.from_numpy(np.stack([r[0] for r in self.replay])) if self.replay else torch.empty(0, 48),
            replay_targets=torch.tensor([r[1] for r in self.replay], dtype=torch.float32))

    def restore(self, saved):
        if saved.get('version') != 1 or saved.get('config') != asdict(self.config):
            raise ValueError('Change-head checkpoint configuration mismatch')
        self.head.load_state_dict(saved['weights'], strict=True)
        self.optimizer.load_state_dict(saved['optimizer'])
        self.rng.bit_generator.state = saved['rng']
        self.updates, self.examples = saved['updates'], saved['examples']
        self.replay = deque(zip(saved['replay_features'].numpy(), saved['replay_targets'].tolist()),
                            maxlen=self.config.replay_size)

    def candidates(self, observation, rng, clicks=16):
        """Optional planner prior: ordinary actions and uniform clicks survive."""
        if type(clicks) is not int or not 1 <= clicks <= 4096:
            raise ValueError('clicks must be in 1..4096')
        ordinary = [Action(i) for i in sorted(set(observation.available_actions)) if 1 <= i <= 7 and i != 6]
        if 6 not in observation.available_actions:
            if not ordinary:
                raise ValueError('No legal ordinary actions')
            return ordinary
        reserved = max(1, math.ceil(clicks * self.config.uniform_probability))
        uniform = [Action(6, dict(x=xy % 64, y=xy // 64))
                   for xy in rng.sample(range(4096), reserved)]
        pool = click_pool(observation, rng, max(clicks, self.config.candidate_clicks))
        probabilities = self.probabilities(observation, pool)
        # Randomize exact score ties; do not prefer low coordinates.
        order = list(range(len(pool))); rng.shuffle(order)
        order.sort(key=lambda i: float(probabilities[i]), reverse=True)
        result, used = list(uniform), {(a.data['x'], a.data['y']) for a in uniform}
        for i in order:
            action = pool[i]; key = action.data['x'], action.data['y']
            if key not in used and len(result) < clicks:
                result.append(action); used.add(key)
        for xy in rng.sample(range(4096), min(4096, clicks)):
            if len(result) >= clicks:
                break
            if (xy % 64, xy // 64) not in used:
                result.append(Action(6, dict(x=xy % 64, y=xy // 64)))
                used.add((xy % 64, xy // 64))
        return ordinary + result


def click_pool(observation, rng, count):
    visual = visible_clicks(observation, rng, count)
    return visual + [Action(6, dict(x=xy % 64, y=xy // 64))
                     for xy in rng.sample(range(4096), count - len(visual))]


class ExplorationSession:
    def __init__(self, config, seed, explorer=None):
        if config.policy == 'learned' and explorer is None:
            raise ValueError('Learned collection requires a change head')
        self.config, self.explorer = config, explorer
        self.rng = random.Random(f'sequence-exploration:{seed}')
        self.reset()

    def reset(self):
        self.previous, self.noops, self.tick = None, {}, 0

    def action(self, observation, rng):
        # Always consume exactly the legacy ID/coordinate draws on this stream.
        action = random_action(observation, rng)
        if (action.id != 6 or self.config.policy == 'uniform' or
                self.rng.random() < self.config.uniform_probability):
            return action
        if self.config.policy == 'mixed':
            choices = visible_clicks(observation, self.rng, 1)
            return choices[0] if choices else action
        if observation.final is None or np.unique(observation.final).size < 2:
            return action
        choices = click_pool(observation, self.rng, self.config.candidate_clicks)
        probabilities = self.explorer.probabilities(observation, choices, self.previous)
        key = observation.final.tobytes() if observation.final is not None else b''
        weights = [float(.05 + p)**4 * (.02 if self.noops.get((key, a.data['x'], a.data['y']), 0) > self.tick else 1)
                   for a, p in zip(choices, probabilities)]
        return self.rng.choices(choices, weights=weights)[0]

    def observe(self, before, action, after):
        self.tick += 1
        self.noops = {k: expiry for k, expiry in self.noops.items() if expiry > self.tick}
        changed = real_change(before, after)
        if (changed is False and action.id == 6 and not after.terminal and not after.starts_new_level(before)):
            self.noops[(before.final.tobytes(), action.data['x'], action.data['y'])] = self.tick + self.config.noop_cooldown
        self.previous = (action, changed) if changed is not None else None


COUNT_FIELDS = (
    'transitions', 'eligible_transitions', 'changed_actions', 'unchanged_actions',
    'eligible_clicks', 'changed_clicks', 'unchanged_clicks',
    'progress_actions', 'levels_gained', 'terminal_actions', 'wins', 'game_overs',
)


def interaction_counts(session):
    """Visual changes on eligible real pairs; native progress is separate."""
    counts = dict.fromkeys(COUNT_FIELDS, 0)
    counts['transitions'] = len(session.transitions)
    for step in session.transitions:
        progress = max(0, step.after.levels_completed - step.before.levels_completed)
        counts['progress_actions'] += int(progress > 0)
        counts['levels_gained'] += progress
        counts['terminal_actions'] += int(step.after.terminal)
        counts['wins'] += int(step.after.state == 'WIN')
        counts['game_overs'] += int(step.after.state == 'GAME_OVER')
        if not step.supervised:
            continue
        changed = real_change(step.before, step.after)
        counts['eligible_transitions'] += 1
        counts['changed_actions' if changed else 'unchanged_actions'] += 1
        if step.action.id == 6:
            counts['eligible_clicks'] += 1
            counts['changed_clicks' if changed else 'unchanged_clicks'] += 1
    return counts


def sum_counts(rows):
    rows = list(rows)
    return {key: sum(row[key] for row in rows) for key in COUNT_FIELDS}
