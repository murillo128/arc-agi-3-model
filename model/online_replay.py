"""Bounded observed episode prefixes for online predictor minibatches, RAM only."""
import random

from .data import Episode, Replay
from .path_planning import real_states


class PredictorReplay:
    def __init__(self, engine, *, episodes=16, batch_size=4, horizon=3, seed=0):
        if not 1 <= batch_size <= episodes:
            raise ValueError('Predictor replay requires 1 <= batch size <= episode capacity')
        self.engine, self.batch_size, self.horizon = engine, batch_size, horizon
        self.buffer = Replay(episodes, engine.config.replay_bytes)
        self.rng = random.Random(seed)
        self.current = None

    def record(self, episode):
        """Snapshot only a confirmed, same-level prefix, never a live mutable list."""
        obs = episode.observations
        if not episode.actions or any(o.final is None for o in obs):
            return False
        if any(before.terminal or after.starts_new_level(before) or
               after.final.shape != before.final.shape for before, after in zip(obs, obs[1:])):
            return False
        if self.current is episode:
            self.buffer.episodes.pop()  # Replace this game's previous observed prefix.
        snapshot = Episode(list(obs), list(episode.actions), list(episode.prepared), episode.latent_cache)
        self.buffer.add(snapshot)
        episode.replay_owner = snapshot.replay_owner
        self.current = episode
        return True

    def sample(self):
        """Latest endpoint plus random endpoints from distinct older episodes."""
        if not self.buffer.episodes:
            return []
        return self.buffer.sample(self.rng, self.batch_size, 2,
                                  latest=self.buffer.episodes[-1], random_prefix=True)

    def examples(self, prefixes):
        # z caches are checked against encoder weights. h is always rebuilt from
        # the entire real prefix with current predictor parameters, never stored.
        self.engine.model.eval()
        examples = []
        for prefix in prefixes:
            states = real_states(self.engine, prefix)
            rollout = None
            if len(prefix.actions) >= self.horizon:
                start = len(prefix.actions) - self.horizon
                rollout = (states[start], prefix.actions[start:], prefix.observations[start].final)
            examples.append(dict(state=states[-2], action=prefix.actions[-1],
                reference=prefix.observations[-2].final, after=prefix.observations[-1],
                target_z=states[-1].z, rollout=rollout))
        return examples

    def status(self):
        return dict(capacity=self.buffer.capacity, episodes=len(self.buffer.episodes),
                    transitions=sum(len(e.actions) for e in self.buffer.episodes),
                    batch_size=self.batch_size)
