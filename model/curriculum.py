"""Training permissions and a cumulative curriculum; never used by inference."""
from dataclasses import asdict, dataclass, field
import math


@dataclass(frozen=True)
class CurriculumConfig:
    representation_open: float = .10
    prediction_open: float = .10
    latent_drift: float = .15
    success: float = .80
    patience: int = 3
    minimum_samples: int = 8
    minimum_changed: int = 4

    def __post_init__(self):
        if not all(0 <= threshold <= 1 for threshold in
                   (self.representation_open, self.prediction_open)):
            raise ValueError('Quality thresholds must lie in [0, 1]')
        if not 0 < self.latent_drift <= 1 or not 0 < self.success <= 1:
            raise ValueError('Invalid stability/success threshold')
        if any(type(n) is not int or n < 1 for n in
               (self.patience, self.minimum_samples, self.minimum_changed)):
            raise ValueError('Gate evidence counts must be positive integers')


def measured(value):
    return isinstance(value, (float, int)) and math.isfinite(value) and value >= 0


def planner_exploration(actions, warmup=8192):
    """Per-level collection schedule; fixed evaluations never use exploration."""
    return max(.2, 1. - .8 * actions / warmup)


@dataclass
class LevelReadiness:
    predictor: bool = False
    planner: bool = False
    representation_passes: int = 0
    prediction_passes: int = 0
    promotion_passes: int = 0
    measurements: dict = field(default_factory=dict)

    def observe(self, quality, success, config, *, world_only=False):
        """Automatic permissions require a completed development evaluation."""
        self.measurements = dict(quality, success=success)
        recon, drift = quality.get('reconstruction_error'), quality.get('latent_drift')
        evidence = quality.get('samples', 0) >= config.minimum_samples
        stable = measured(drift) and drift <= config.latent_drift
        good_rep = evidence and measured(recon) and recon <= config.representation_open and stable
        self.representation_passes = self.representation_passes + 1 if good_rep else 0
        # Permissions latch; a later regression prevents promotion. The trainer
        # separately owns the optional representation freeze at planner admission.
        if self.representation_passes >= config.patience:
            self.predictor = True
        one, rollout = quality.get('prediction_error'), quality.get('rollout_error')
        # Some SDK goals are events (e.g. a click) with no pixel change, even on
        # success. Require an observed native completion before accepting such
        # a stationary validation set; never infer it from a game ID/private rule.
        stationary = (quality.get('changed') == 0 and quality.get('rollout_changed') == 0
                      and quality.get('completed_episodes', 0) > 0)
        predictive_evidence = (evidence and quality.get('rollout_samples', 0) >= config.minimum_samples
            and (stationary or (quality.get('changed', 0) >= config.minimum_changed
                                and quality.get('rollout_changed', 0) >= config.minimum_changed)))
        good_pred = (good_rep and self.predictor and predictive_evidence and measured(one) and measured(rollout)
                     and max(one, rollout) <= config.prediction_open)
        self.prediction_passes = self.prediction_passes + 1 if good_pred else 0
        if self.prediction_passes >= config.patience and not world_only:
            self.planner = True
        good_level = (good_rep and good_pred and (world_only or
                      (self.planner and measured(success) and success >= config.success)))
        self.promotion_passes = self.promotion_passes + 1 if good_level else 0


class Curriculum:
    def __init__(self, levels, config=CurriculumConfig()):
        self.levels, self.config = dict(levels), config
        self.unlocked = dict.fromkeys(levels, 1)
        self.readiness = {(g, level): LevelReadiness() for g, n in levels.items()
                          for level in range(1, n + 1)}
        self.episode_counts = dict.fromkeys(self.readiness, 0)
        self.stage = 1
        self.preferred_pair = None

    @property
    def pairs(self):
        return [(g, level) for g, n in self.unlocked.items() for level in range(1, n + 1)]

    def next_pair(self):
        # Fresh episodes, balanced across unlocked levels. Scheduling counts are
        # reset at promotion so the new level cannot monopolize the next stage.
        pair = self.preferred_pair or min(self.pairs, key=self.episode_counts.get)
        self.preferred_pair = None
        self.episode_counts[pair] += 1
        return pair

    def advance_phase(self, pair):
        """Explicit operator override; never invent quality or passed checks."""
        state = self.readiness[pair]
        if state.planner:
            return False
        if state.predictor:
            state.planner = True
        else:
            state.predictor = True
        return True

    def advance_level(self, game):
        """Unlock and schedule the next level, retaining cumulative practice."""
        n = self.unlocked[game]
        if n == self.levels[game]:
            return None
        self.unlocked[game] = n + 1
        self.stage += 1
        self.episode_counts = dict.fromkeys(self.readiness, 0)
        self.preferred_pair = (game, n + 1)
        return self.preferred_pair

    def advance(self, *, latest_only=False):
        promoted = []
        for game, n in self.unlocked.items():
            levels = [n] if latest_only else range(1, n + 1)
            if n < self.levels[game] and all(self.readiness[game, level].promotion_passes
                    >= self.config.patience for level in levels):
                self.unlocked[game] += 1
                promoted.append((game, n + 1))
                for level in range(1, n + 1):
                    self.readiness[game, level].promotion_passes = 0
        if promoted:
            self.stage += 1
            self.episode_counts = dict.fromkeys(self.readiness, 0)
        return promoted

    def snapshot(self):
        return dict(stage=self.stage, unlocked=dict(self.unlocked), thresholds=asdict(self.config),
                    preferred_pair=self.preferred_pair,
                    levels=[dict(game=g, level=n, unlocked=n <= self.unlocked[g],
                                 episodes=self.episode_counts[g, n], **asdict(state))
                            for (g, n), state in self.readiness.items()])

    def restore(self, snapshot, *, update_representation_threshold=False,
                update_latent_drift_threshold=False):
        rows = {(r['game'], r['level']): r for r in snapshot['levels']}
        thresholds = dict(snapshot['thresholds'])
        if update_representation_threshold:
            thresholds['representation_open'] = self.config.representation_open
        if update_latent_drift_threshold:
            thresholds['latent_drift'] = self.config.latent_drift
        if (thresholds != asdict(self.config) or set(rows) != set(self.readiness)
                or set(snapshot['unlocked']) != set(self.levels)
                or any(not 1 <= n <= self.levels[g] for g, n in snapshot['unlocked'].items())):
            raise ValueError('Checkpoint curriculum does not match this run')
        self.stage, self.unlocked = snapshot['stage'], dict(snapshot['unlocked'])
        preferred = snapshot.get('preferred_pair')
        self.preferred_pair = tuple(preferred) if preferred else None
        if self.preferred_pair is not None and self.preferred_pair not in self.pairs:
            raise ValueError('Preferred training level is not unlocked')
        for pair, row in rows.items():
            self.readiness[pair] = LevelReadiness(**{k: row[k] for k in LevelReadiness.__dataclass_fields__})
            if snapshot['thresholds'] != asdict(self.config):
                # Keep learned permissions, but require fresh evidence under the
                # explicitly changed threshold before another activation/promotion.
                state = self.readiness[pair]
                state.representation_passes = state.prediction_passes = state.promotion_passes = 0
            self.episode_counts[pair] = row.get('episodes', 0)


class OrderedCurriculum(Curriculum):
    """Explicit stages, with no implicit sorting or earlier-level practice."""
    def __init__(self, stages):
        self.stages = list(stages)
        self.index, self.completed = 0, False
        self.readiness = {(s['game'], s['level']): LevelReadiness() for s in stages}
        self.episode_counts = dict.fromkeys(self.readiness, 0)
        self.preferred_pair = None

    @property
    def stage(self):
        return self.index + 1

    @property
    def config(self):
        return CurriculumConfig(**self.stages[self.index]['thresholds'])

    @property
    def pairs(self):
        entry = self.stages[self.index]
        return [(entry['game'], entry['level'])]

    @property
    def unlocked(self):
        return dict(self.pairs)

    def advance_phase(self, pair):
        entry = self.stages[self.index]
        if entry.get('objective') == 'world_model' or entry['source'] == 'sdk':
            state = self.readiness[pair]
            if state.predictor:
                return False
            state.predictor = True
            return True
        return super().advance_phase(pair)

    def advance_level(self, game):
        if self.index + 1 == len(self.stages):
            return None
        self.index += 1
        return self.pairs[0]

    def advance(self, *, latest_only=True):
        if self.completed or self.readiness[self.pairs[0]].promotion_passes < self.config.patience:
            return []
        if self.index + 1 == len(self.stages):
            self.completed = True
            return []
        return [self.advance_level(self.pairs[0][0])]

    def snapshot(self):
        rows = []
        for i, entry in enumerate(self.stages):
            pair = entry['game'], entry['level']
            rows.append(dict(game=pair[0], level=pair[1], source=entry['source'],
                             objective=entry.get('objective', 'world_model' if entry['source'] == 'sdk' else 'subgoals'),
                             thresholds=entry['thresholds'], unlocked=i <= self.index,
                             active=i == self.index, episodes=self.episode_counts[pair],
                             **asdict(self.readiness[pair])))
        return dict(stage=self.stage, unlocked=self.unlocked, thresholds=asdict(self.config),
                    ordered_stages=self.stages, index=self.index, completed=self.completed,
                    levels=rows)

    def restore(self, snapshot, **overrides):
        if any(overrides.values()) or snapshot.get('ordered_stages') != self.stages:
            raise ValueError('Checkpoint ordered curriculum does not match this run')
        index = snapshot.get('index')
        completed = snapshot.get('completed')
        if (type(index) is not int or not 0 <= index < len(self.stages) or
                type(completed) is not bool or (completed and index != len(self.stages) - 1)):
            raise ValueError('Invalid ordered curriculum cursor')
        rows = snapshot['levels']
        if [(r['game'], r['level']) for r in rows] != list(self.readiness):
            raise ValueError('Checkpoint levels do not match ordered curriculum')
        self.index, self.completed = index, completed
        for row in rows:
            pair = row['game'], row['level']
            self.readiness[pair] = LevelReadiness(**{k: row[k] for k in LevelReadiness.__dataclass_fields__})
            self.episode_counts[pair] = row['episodes']
