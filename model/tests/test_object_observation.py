"""Object extraction replay and comparable, pre-fit forecasting measurements."""
from copy import deepcopy
from dataclasses import replace
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

from model.config import SequenceConfig, TrainConfig
from model.data import Action
from model.game_sequences import GameSession, SequenceLearner, Transition
from model.learning import Engine
from model.networks import WorldModel
from model.object_observation import ObservationReplay
from model.object_training import ObjectTrainer, ObjectTrainingConfig, object_rollout
from model.segmentation import Segmentation
from model.tests.test_objects import FakeTeacher, frame, network


class CountingTeacher(FakeTeacher):
    def __init__(self):
        self.calls = 0

    def segment(self, image):
        self.calls += 1
        return super().segment(image)


class ObjectObservationTests(unittest.TestCase):
    def engine(self):
        torch.manual_seed(23)
        engine = Engine(WorldModel(network()), TrainConfig())
        learner = SequenceLearner(engine, SequenceConfig(batch_games=1, session_steps=4, block_steps=2))
        return engine, learner

    def test_labels_of_before_after_and_endpoint_train_without_more_teacher_calls(self):
        engine, learner = self.engine()
        # Seed 1 samples the first of these three contiguous transitions.
        session = GameSession('fixture', 1, [Transition(frame(x), Action(4), frame(x+1)) for x in range(3)])
        teacher = CountingTeacher()
        trainer = ObjectTrainer(engine, ObjectTrainingConfig('/fake', samples=1, passes=1,
            observer_steps=6, observer_replay_frames=4), teacher)
        metrics, preview = trainer.fit(learner, [session])
        self.assertEqual(teacher.calls, 3)
        self.assertEqual(metrics['observer_frames'], 3)
        self.assertEqual(metrics['observer_updates'], 6)
        columns = {int(np.nonzero(row.labels.masks[0])[1].min()) for row in trainer.observations.rows.values()}
        self.assertEqual(columns, {0, 1, 3})
        self.assertEqual(metrics['metric_version'], 2)
        self.assertEqual(preview['conditioning'], 'student')
        self.assertEqual(preview['measurement'], 'before auxiliary fitting')
        for name in ('observed_iou', 'observed_post_iou', 'sam_iou', 'student_iou',
                     'sam_rollout_iou', 'student_rollout_iou', 'copy_iou', 'copy_rollout_iou'):
            self.assertIsNotNone(metrics[name], name)
            self.assertTrue(0 <= metrics[name] <= 1, name)
        for row in trainer.observations.rows.values():
            self.assertEqual(row.latent.device.type, 'cpu')
            self.assertIsNone(row.latent.grad_fn)
            self.assertFalse(row.latent.requires_grad)

    def test_teacher_forecast_is_not_reported_as_student_quality_or_admitted(self):
        engine, learner = self.engine()
        head = engine.model.dynamics.objects
        with torch.no_grad():
            for p in head.parameters(): p.zero_()
            head.presence.bias.fill_(10.)
        teacher = CountingTeacher()
        trainer = ObjectTrainer(engine, ObjectTrainingConfig('/fake', samples=1, passes=1,
            observer_steps=0, activation_iou=.5, warmup_updates=1, activate_residual=True), teacher)
        same = frame(1)
        session = GameSession('fixture', 1, [Transition(same, Action(1), same) for _ in range(3)])
        # Its hard current mask is exact, but its two .5025 probabilities give
        # a soft source occupancy of .2525. Identity transport cannot restore it.
        def low_confidence_current(z, height, width):
            logits = z.new_full((1, head.slots, height, width), -20.)
            logits[0, 0, 2:4, 1:3] = .01
            presence = z.new_full((1, head.slots), -20.); presence[0, 0] = .01
            return logits, presence
        with patch.object(head, 'observe', side_effect=low_confidence_current):
            metrics, _ = trainer.fit(learner, [session])
        self.assertEqual(metrics['observed_iou'], 1.)
        self.assertEqual(metrics['sam_iou'], 1.)
        self.assertEqual(metrics['sam_rollout_iou'], 1.)
        self.assertEqual(metrics['copy_iou'], 1.)
        self.assertEqual(metrics['copy_rollout_iou'], 1.)
        self.assertEqual(metrics['student_iou'], 0.)
        self.assertEqual(metrics['student_rollout_iou'], 0.)
        self.assertFalse(metrics['ready'])
        self.assertEqual(float(head.gain), 0.)

    def test_replay_is_bounded_diverse_and_invalidates_changed_encoder(self):
        engine, _ = self.engine()
        replay = ObservationReplay(3)
        replay.check_encoder(engine)
        z = torch.randn(1, 4, 64, requires_grad=True)
        for game, x in [('rare', 0), ('common', 1), ('common', 2), ('common', 3)]:
            obs = frame(x)
            replay.add(game, obs, z, FakeTeacher().segment(obs.final))
        self.assertEqual(len(replay.rows), 3)
        self.assertEqual(sum(row.game == 'rare' for row in replay.rows.values()), 1)
        self.assertTrue(all(row.latent.data_ptr() != z.data_ptr() for row in replay.rows.values()))
        current = [next(iter(replay.rows.values()))]
        schedule = replay.schedule(current, 8, random.Random(1))
        self.assertEqual(len(schedule), 8)
        self.assertEqual(schedule[0], current[0])
        self.assertEqual({row.game for row in schedule}, {'rare', 'common'})
        replay.check_encoder(engine)
        self.assertEqual(len(replay.rows), 3)
        with torch.no_grad(): next(engine.model.encoder.parameters()).add_(1.)
        replay.check_encoder(engine)
        self.assertFalse(replay.rows)

    def test_gain_zero_cached_forecasts_preserve_masks_and_auxiliary_gradients(self):
        engine, _ = self.engine()
        head = engine.model.dynamics.objects
        engine.model.eval().requires_grad_(False); head.requires_grad_(True)
        trainer = ObjectTrainer(engine, ObjectTrainingConfig('/fake'), FakeTeacher())
        z = torch.randn(1, 4, 64)
        h = torch.randn(1, engine.model.config.memory)
        masks = torch.rand(1, head.slots, 8, 8)
        actions = [engine.actions([Action(i)]) for i in (1, 4, 2)]
        ordinary = object_rollout(engine, z, h, actions, (8, 8), masks, warmup=True)
        sum(part.sum() for row in ordinary for part in row[2]).backward()
        gradients = {name: None if p.grad is None else p.grad.clone() for name, p in head.named_parameters()}
        expected = [tuple(part.detach().clone() for part in row[2]) for row in ordinary]
        head.zero_grad(set_to_none=True)
        frozen, zz, hh = [], z, h
        with torch.no_grad():
            for action in actions:
                next_z, next_h, _ = engine.predict(zz, hh, action, 8, 8)
                frozen.append((zz, next_h, action, next_z))
                zz, hh = next_z, next_h
        actual = trainer._future(dict(frozen=frozen, geometry=(8, 8)), masks)
        for row, reference in zip(actual, expected):
            for part, wanted in zip(row[2], reference):
                torch.testing.assert_close(part, wanted, rtol=1e-5, atol=1e-6)
        sum(part.sum() for row in actual for part in row[2]).backward()
        for name, p in head.named_parameters():
            before = gradients[name]
            if before is None:
                self.assertIsNone(p.grad, name)
            elif p.grad is None:
                self.assertEqual(float(before.abs().max()), 0., name)
            else:
                torch.testing.assert_close(p.grad, before, rtol=1e-5, atol=1e-5, msg=name)

    def test_observer_budget_bounds(self):
        for key, bad in [('observer_steps', -1), ('observer_steps', 257),
                         ('observer_replay_frames', 0), ('observer_replay_frames', 257)]:
            with self.assertRaises(ValueError): ObjectTrainingConfig('/fake', **{key: bad})
        self.assertEqual(ObjectTrainingConfig('/fake').observer_steps, 32)
