"""Campaign collection/evaluation boundaries with small real neural modules."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
import unittest
from unittest.mock import patch

import numpy as np
import torch

from model.campaign import evaluate_world, replay_records, train_world
from model.config import NetworkConfig, TrainConfig
from model.data import Action, Episode, Observation
from model.learning import Engine
from model.networks import WorldModel


torch.set_num_threads(2)


def obs(value=1, **kwargs):
    return replace(Observation(np.full((1, 4, 4), value, dtype=np.uint8), (1,),
                               'NOT_FINISHED', 0, 3), **kwargs)


class SyntheticFixture:
    """A public-observation environment fixture; no engine or model mocks."""
    def __init__(self, steps=3, native_progress=False):
        self.steps, self.native_progress, self.calls = steps, native_progress, []

    def make(self, seed, level):
        self.calls.append((seed, level))
        parent = self
        class Environment:
            observation = obs()
            count = 0
            def step(self, action):
                self.count += 1
                completed = self.count == parent.steps
                self.observation = obs(self.count % 10 + 1,
                    state='GAME_OVER' if completed and not parent.native_progress else 'NOT_FINISHED',
                    levels_completed=int(completed and parent.native_progress), action_input=action)
                return self.observation
        return Environment()


def engine(**changes):
    return Engine(WorldModel(NetworkConfig.small()), replace(TrainConfig(),
        **({'max_episode_steps': 4, 'replay_episodes': 4, 'frame_chunk': 2,
            'sigreg_weight': 0, 'variance_weight': 0} | changes)))


class CampaignWorldTests(unittest.TestCase):
    def test_head_calibration_keeps_early_successes_after_replay_eviction(self):
        class EarlySuccesses(SyntheticFixture):
            def make(self, seed, level):
                self.native_progress = len(self.calls) < 2
                return super().make(seed, level)
        model = engine(replay_episodes=1)
        with redirect_stdout(io.StringIO()):
            records, report = train_world(model, {'a': EarlySuccesses(steps=2)}, [('a', 1)], 12, 0, 'world_reference')
        self.assertEqual(sum(r.event == 1 for r in records), 2)
        self.assertEqual(len(records), 12)
        self.assertEqual(report['calibration']['episodes'], 6)
        self.assertEqual(report['calibration']['successes'], 2)
        self.assertEqual(report['calibration']['failures'], 4)
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(MemoryError, 'calibration'):
            train_world(engine(), {'a': SyntheticFixture()}, [('a', 1)], 3, 0, 'world_reference', calibration_bytes=1)

    def test_staged_schedule_reports_actual_joint_and_frozen_updates(self):
        model = engine(representation_joint_steps=4, auto_representation=False,
                       freeze_representation=False, train_every=1)
        with redirect_stdout(io.StringIO()):
            _, report = train_world(model, {'a': SyntheticFixture(steps=4)}, [('a', 1)], 8, 0, 'staged_freeze')
        self.assertEqual(report['joint_updates'], 4)
        self.assertEqual(report['frozen_updates'], 4)
        self.assertEqual(report['updates'], report['joint_updates'] + report['frozen_updates'])
        self.assertTrue(model.config.freeze_representation)

    def test_exact_pair_quotas_and_raw_archive_do_not_own_device_inputs(self):
        model = engine()
        factories = {'a': SyntheticFixture(), 'b': SyntheticFixture()}
        pairs = [('a', 1), ('b', 1)]
        seen = []
        def inspect_archive(e, archive, discount):
            for items in archive.values():
                for episode, reason in items:
                    self.assertEqual(episode.prepared, [])
                    self.assertEqual(episode.latent_cache.values, [])
                    seen.append(reason)
            return replay_records(e, archive, discount)
        with redirect_stdout(io.StringIO()), patch('model.campaign.replay_records', side_effect=inspect_archive):
            records, report = train_world(model, factories, pairs, 9, 0, 'rollout2')
        self.assertEqual(report['actions'], 9)
        self.assertEqual([p['actions'] for p in report['per_pair']], [5, 4])
        self.assertTrue(records)
        self.assertIn('interrupted', seen)
        self.assertTrue(any(row.target is None for row in records))
        self.assertTrue(all(row.z.device.type == 'cpu' and row.z.dtype == torch.float16 for row in records))

    def test_native_progress_does_not_retrain_last_pair_or_enter_world_targets(self):
        model = engine()
        factory = SyntheticFixture(steps=2, native_progress=True)
        with redirect_stdout(io.StringIO()):
            records, report = train_world(model, {'a': factory}, [('a', 1)], 6, 0, 'rollout2')
        self.assertEqual(report['updates'], 3)  # One observed visual pair per attempt.
        self.assertEqual(report['episodes'], 3)
        self.assertEqual(sum(row.event == 1 for row in records), 3)
        self.assertTrue(all(row.target == 1 for row in records if row.event == 1))

    def test_reencoding_shares_successors_and_keeps_empty_response_belief(self):
        model = engine(max_episode_steps=4)
        empty = obs(frames=np.empty((0, 0, 0), dtype=np.uint8))
        episode = Episode([obs(1), empty, obs(2), obs(3, state='WIN')], [Action(1)] * 3)
        model.model.train()
        rows = replay_records(model, {('a', 1): [(episode, 'progress')]}, .95)
        self.assertTrue(model.model.training)
        self.assertEqual([row.remaining_budget for row in rows], [4, 3, 2])
        self.assertAlmostEqual(rows[0].target, .95 ** 2)
        self.assertEqual(rows[-1].target, 1.)
        self.assertIs(rows[0].z, rows[0].next_z)
        self.assertIs(rows[0].next_z, rows[1].z)
        self.assertIs(rows[0].next_h, rows[1].h)
        self.assertIs(rows[1].next_z, rows[2].z)
        self.assertFalse(torch.equal(rows[0].h, rows[0].next_h))

    def test_full_visual_validation_caps_diagnostics_without_extra_actions(self):
        model = engine(validation_steps=20)
        factory = SyntheticFixture(steps=20)
        saved_config = model.config
        with redirect_stdout(io.StringIO()):
            report = evaluate_world(model, {'a': factory}, [('a', 1)], 0, episodes=3)
        self.assertIs(model.config, saved_config)
        self.assertEqual(len(factory.calls), 3)
        self.assertEqual(report['per_pair'][0]['actions'], 60)
        self.assertEqual(report['diagnostic_episodes'], 2)
        self.assertEqual(report['diagnostic_actions'], 32)
        self.assertEqual(report['rollouts']['1']['samples'], 32)
        self.assertEqual(report['rollouts']['2']['samples'], 30)
        self.assertEqual(report['rollouts']['4']['samples'], 26)

    def test_campaign_validation_stops_at_first_progress_without_fake_win(self):
        model = engine(validation_steps=20)
        factory = SyntheticFixture(steps=2, native_progress=True)
        with redirect_stdout(io.StringIO()):
            report = evaluate_world(model, {'a': factory}, [('a', 1)], 0, episodes=2)
        pair = report['per_pair'][0]
        self.assertEqual(pair['actions'], 4)
        self.assertEqual(pair['level_advances'], 2)
        self.assertEqual(pair['wins'], 0)
        self.assertEqual(pair['transitions'], 2)
        self.assertEqual(report['diagnostic_actions'], 2)


if __name__ == '__main__':
    unittest.main()
