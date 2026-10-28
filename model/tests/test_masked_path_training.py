"""Path fitting pairs masked targets with each state's real visual context."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from model.data import Action, Episode, Observation
from model.game_sequences import GameSession, Transition
from model.path_planning import PathLearner, PathValue
from model.sequence_planning import SequencePlanning, SequencePlanningConfig
from model.tests.test_agent import make_agent, obs


def target(context, pixel):
    frame = np.full((4, 4), context, np.uint8)
    frame[0, 0] = pixel
    return frame


def observation(frame):
    return Observation(frame[None], (), 'NOT_FINISHED', 0, 0)


class MaskedPathTrainingTests(unittest.TestCase):
    def setUp(self):
        self.mask = np.zeros((4, 4), bool)
        self.mask[0, 0] = True

    def capture_encoder(self, engine, batches):
        encode = engine.encode

        def captured(observations, *args, **kwargs):
            batches.append([o.final.copy() for o in observations])
            return encode(observations, *args, **kwargs)

        return patch.object(engine, 'encode', side_effect=captured)

    def test_sequence_before_and_after_targets_match_their_real_state_context(self):
        engine = make_agent().engine
        heads = SequencePlanning(engine, SequencePlanningConfig(
            width=8, samples=4, minibatch=4, passes=1))
        frames = [obs(1), obs(2), obs(3)]
        raw = target(9, 7)
        session = GameSession('sc01-v1', 0,
            [Transition(frames[0], Action(1), frames[1]), Transition(frames[1], Action(2), frames[2])],
            planning=dict(counts={}, seconds=0., labels=[dict(index=0, frame=raw, mask=self.mask)],
                segments=[dict(start=0, end=2, deadline=8, success=True, frame=raw, mask=self.mask)]))
        inputs, encoded = [], []
        handle = heads.path.register_forward_pre_hook(lambda _, args: inputs.append(args))
        before = deepcopy(session)
        try:
            with self.capture_encoder(engine, encoded):
                metrics, sample = heads.fit([session])
        finally:
            handle.remove()
        self.assertEqual(metrics['path_examples'], 3)
        np.testing.assert_array_equal(sample['target'], raw)
        z, _, goals, _, _, _, budgets = inputs[0]
        with torch.no_grad():
            states = engine.encode(frames)
            expected = engine.encode([observation(target(context, 7)) for context in (1, 2, 3)])
        self.assertEqual(sorted(budgets.tolist()), [6, 7, 8])
        for index, budget in enumerate(budgets.tolist()):
            position = 8 - budget
            np.testing.assert_array_equal(encoded[-1][index], target(position + 1, 7))
            torch.testing.assert_close(z[index], states[position], rtol=0, atol=0)
            torch.testing.assert_close(goals[index], expected[position], rtol=0, atol=0)
        np.testing.assert_array_equal(session.planning['segments'][0]['frame'], before.planning['segments'][0]['frame'])
        np.testing.assert_array_equal(session.planning['labels'][0]['mask'], before.planning['labels'][0]['mask'])
        for actual, original in zip(session.transitions, before.transitions):
            np.testing.assert_array_equal(actual.before.frames, original.before.frames)
            np.testing.assert_array_equal(actual.after.frames, original.after.frames)

    def learner(self, **options):
        engine = make_agent().engine
        c = engine.model.config
        evaluator = PathValue(c.latent, c.memory, width=4)
        planner = SimpleNamespace(engine=engine, evaluator=evaluator,
            goal=dict(frame=target(9, 7), pixel_mask=self.mask))
        return PathLearner(planner, learning_rate=0., **options)

    def test_legacy_replay_keeps_per_state_goals_batched_through_readout(self):
        learner = self.learner(capacity=2, batch_segments=2)
        engine = learner.planner.engine
        first = Episode([obs(1), obs(2)], [Action(1)])
        learner.update(first, 0, False, 8, passes=1)
        learner.planner.goal['frame'] = target(10, 8)
        encoded, inputs = [], []
        handle = learner.planner.evaluator.register_forward_pre_hook(lambda _, args: inputs.append(args))
        try:
            with self.capture_encoder(engine, encoded):
                learner.update(Episode([obs(4), obs(5)], [Action(2)]), 0, True, 8, passes=1)
        finally:
            handle.remove()
        z, _, goals, _, _, _, budgets = inputs[0]
        expected_frames = [target(1, 7), target(2, 7), target(4, 8), target(5, 8)]
        with torch.no_grad():
            # Preserve batch sizes for exact floating-point encoder comparisons.
            states = engine.encode([obs(1), obs(2)]) + engine.encode([obs(4), obs(5)])
            expected = (engine.encode([observation(frame) for frame in expected_frames[:2]])
                        + engine.encode([observation(frame) for frame in expected_frames[2:]]))
        self.assertEqual(len(goals), 4)
        self.assertEqual(len(encoded), 4, 'Each replay segment encodes its states and paired goals')
        for row, (state, goal, budget) in enumerate(zip(z, goals, budgets.tolist())):
            item, position = row // 2, 8 - budget
            expected_index = item * 2 + position
            np.testing.assert_array_equal(encoded[item * 2 + 1][row % 2], expected_frames[expected_index])
            torch.testing.assert_close(state, states[expected_index], rtol=0, atol=0)
            torch.testing.assert_close(goal, expected[expected_index], rtol=0, atol=0)
        np.testing.assert_array_equal(learner.segments[0]['frame'], target(9, 7))
        np.testing.assert_array_equal(learner.segments[1]['frame'], target(10, 8))

    def test_hindsight_and_imagined_targets_use_matching_real_rollout_origins(self):
        for hindsight, imagined in ((True, False), (False, True), (True, True)):
            with self.subTest(hindsight=hindsight, imagined=imagined):
                learner = self.learner(hindsight=hindsight, predicted_inputs=imagined)
                # The only admissible hindsight destination is observation 1.
                # One-step imagined inputs start at max(0, destination - 1).
                learner.rng.choice = lambda values: values[0]
                engine = learner.planner.engine
                episode = Episode([obs(1), obs(2), replace(obs(3), state='GAME_OVER')],
                                  [Action(1), Action(2)])
                encoded, inputs = [], []
                handle = learner.planner.evaluator.register_forward_pre_hook(lambda _, args: inputs.append(args))
                try:
                    with self.capture_encoder(engine, encoded):
                        learner.update(episode, 0, False, 8, passes=1)
                finally:
                    handle.remove()
                z, _, goals, _, _, _, budgets = inputs[0]
                expected_rows = []
                expected_batches = []
                offset = 0
                for pixel, count in ([(7, 3), (2, 2)] if hindsight else [(7, 3)]):
                    for predicted in ([False, True] if imagined else [False]):
                        for budget in budgets[offset:offset + count].tolist():
                            position = 8 - budget
                            origin = max(0, position - 1) if predicted else position
                            expected_rows.append(target(origin + 1, pixel))
                        expected_batches.append(expected_rows[-count:])
                        offset += count
                self.assertEqual(len(expected_rows), len(z))
                # Inspect the encoder's actual literal input frames, not only
                # downstream equality through a mocked encoder.
                for frame in expected_rows:
                    self.assertTrue(any(np.array_equal(frame, actual) for batch in encoded[1:] for actual in batch),
                                    f'Expected encoded goal with real context {frame.tolist()}')
                with torch.no_grad():
                    expected = torch.cat([torch.stack(engine.encode([observation(frame) for frame in batch]))
                                          for batch in expected_batches])
                torch.testing.assert_close(goals, expected, rtol=0, atol=0)
                self.assertEqual(learner.last_hindsight, int(hindsight))
                self.assertFalse(learner.segments[0]['success'])
                np.testing.assert_array_equal(learner.segments[0]['frame'], target(9, 7))
                self.assertTrue(all(p.grad is None for p in engine.model.parameters()))


if __name__ == '__main__':
    unittest.main()
