"""Causal object learning, weak teacher labels and weight-preserving adoption."""
from copy import deepcopy
from dataclasses import asdict, replace
import random
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from model.config import NetworkConfig, TrainConfig, SequenceConfig
from model.data import Action, Observation
from model.game_sequences import GameSession, Transition, SequenceLearner
from model.learning import Engine, optimizer
from model.networks import WorldModel
from model.object_dynamics import assignment, mask_loss, probabilities
from model.object_training import ObjectTrainer, ObjectTrainingConfig, object_rollout, rollout_window
from model.object_upgrade import migrate
from model.segmentation import Segmentation, native_masks, interior_clicks, imagined_candidates, detail_prompts

torch.set_num_threads(2)


def network(slots=3):
    return replace(NetworkConfig.small(), latent=64, pixel_latent=True, local_dynamics=True,
                   rotation_adapter_width=2, object_slots=slots, object_width=4, residual_scale=.1)


def frame(x):
    value = np.zeros((8, 8), np.uint8)
    value[2:4, x:x+2] = 7
    return Observation(value[None], (1, 2, 3, 4, 6), 'NOT_FINISHED', 0, 0)


def session():
    return GameSession('sc01-v1', 7, [Transition(frame(x), Action(4), frame(x+1)) for x in range(4)])


class FakeTeacher:
    """Geometric fixture only; real SAM is exercised separately, offline."""
    remaining = None
    def segment(self, image):
        return Segmentation((image == 7)[None], np.array([1.], np.float32))
    def report(self):
        return dict(provider='fixture', calls=0, cache_hits=0, seconds=0)


class ObjectTests(unittest.TestCase):
    def test_sequence_run_and_resume_retain_object_learning_state(self):
        from contextlib import redirect_stdout
        import io
        import json
        from pathlib import Path
        import tempfile
        from unittest.mock import patch
        from model import checkpoint
        from model.sequence_training import run
        from model.tests.test_training_config import config
        class Teacher(FakeTeacher):
            def segment(self, image):
                mask = np.zeros_like(image, dtype=bool); mask[1:3, 2:4] = True
                return Segmentation(mask[None], np.ones(1, np.float32))
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            path, value = config(root, [dict(source='synthetic', game='sc01-v1', levels=[1])],
                sequence=dict(batch_games=1, session_steps=4, block_steps=2, level_steps=4, representation_updates=0),
                sequence_objects=dict(checkpoint='/fake', samples=1, passes=1))
            (root/'initial.json').write_text(json.dumps(dict(network=asdict(network()),
                training=asdict(TrainConfig(freeze_representation=True)), outcome_width=4)))
            value['settings']['actions'] = 8; path.write_text(json.dumps(value))
            args = SimpleNamespace(training_config=path, output=root/'run', source=None, resume=False,
                                   device='cpu', compile=False, max_batches=1)
            with patch('model.segmentation.SAM2Segmenter', return_value=Teacher()):
                run(args)
                first = checkpoint.load(root/'run/latest.pt')
                args.resume = True
                run(args)
            second = checkpoint.load(root/'run/latest.pt')
            self.assertGreater(int(second['weights']['dynamics.objects.updates']),
                               int(first['weights']['dynamics.objects.updates']))
            report = json.loads((root/'run/report.json').read_text())
            self.assertEqual(report['objects']['samples'], 1)
            self.assertIn('objects', report['history'][-1])

    def test_detail_prompts_touch_tiny_regions_without_assuming_a_palette_color(self):
        image = np.full((8, 8), 7); image[1, 2] = 4; image[5:7, 4:6] = 0
        points = detail_prompts(image)
        self.assertEqual(points.shape, (2, 2))
        for x, y in points:
            self.assertNotEqual(image[int(y*8), int(x*8)], 7)
        np.testing.assert_array_equal(points, detail_prompts(15-image))

    def test_architecture_binds_only_enabled_object_tensors(self):
        import json
        from pathlib import Path
        from model.object_architecture import bind_objects
        document = json.loads(Path('model/architecture.json').read_text())
        self.assertIs(bind_objects(document, asdict(network(0))), document)
        updated = bind_objects(document, asdict(network()))
        node = next(n for n in updated['nodes'] if n['id'] == 'objects.student')
        names = {p['name']: [d['value'] for d in p['shape']] for p in updated['parameters'] if p['id'] in node['parameter_ids']}
        model = WorldModel(network())
        expected = {k: list(v.shape) for k, v in model.state_dict().items() if k.startswith('dynamics.objects.')}
        self.assertEqual(names, expected)
        self.assertTrue(all(n.get('operation') for n in updated['nodes'] if n['kind'] == 'operation'))

    def test_native_projection_and_clicks_respect_holes_empty_masks_and_capacity(self):
        ring = np.zeros((8, 8), bool); ring[1:7, 1:7] = 1; ring[2:6, 2:6] = 0
        proposal = dict(segmentation=ring.repeat(2, 0).repeat(2, 1), predicted_iou=.9, stability_score=.95)
        tiny = np.zeros((16, 16), bool); tiny[4:6, 4:6] = 1
        result = native_masks([proposal, proposal, dict(proposal, segmentation=tiny),
                               dict(proposal, predicted_iou=.3)], (8, 8), 8)
        self.assertEqual(len(result.masks), 2)
        self.assertEqual(result.masks[0].sum(), 1)
        np.testing.assert_array_equal(result.masks[1], ring)
        clicks = interior_clicks(Segmentation(ring[None], np.ones(1)), random.Random(1), 12)
        self.assertTrue(clicks)
        self.assertTrue(all(ring[a.data['y'], a.data['x']] for a in clicks))
        self.assertEqual(interior_clicks(result, random.Random(1), 0), [])
        self.assertEqual(native_masks([proposal], (8, 8), 1).masks.shape, (1, 8, 8))
        self.assertEqual(native_masks([], (8, 8), 2).masks.shape, (0, 8, 8))

    def test_future_clicks_use_predicted_positions_and_preserve_ordinary_and_uniform_actions(self):
        choices = [Action(1), Action(4)] + [Action(6, dict(x=i, y=0)) for i in range(8)]
        masks = torch.zeros(3, 8, 8); masks[0, 5:7, 3:5] = 1
        result = imagined_candidates(choices, masks, random.Random(22))
        self.assertEqual(len(result), 10)
        self.assertEqual(result[:2], choices[:2])
        self.assertTrue(any(a.id == 6 and 5 <= a.data['y'] <= 6 and 3 <= a.data['x'] <= 4 for a in result))
        self.assertTrue(any(a.id == 6 and a.data['y'] >= 8 for a in result), 'Uniform SDK-coordinate reserve survives')
        self.assertEqual(imagined_candidates(choices, masks*0, random.Random(1)), choices)
        self.assertEqual(imagined_candidates(choices[:2], masks, random.Random(1)), choices[:2])

    def test_enabled_path_rollout_calls_teacher_only_for_the_real_root(self):
        from unittest.mock import Mock
        from model.learning import State
        from model.path_planning import PathConfig, PathPlanner, PathValue
        engine = Engine(WorldModel(network()).eval(), TrainConfig())
        teacher = Mock(wraps=FakeTeacher()); engine.segmenter = teacher
        head = engine.model.dynamics.objects
        with torch.no_grad():
            head.gain.fill_(1)
            head.refine.weight.zero_(); head.refine.bias.zero_()
            head.presence.weight.zero_(); head.presence.bias.fill_(8.)
            z = engine.encode([frame(0)])[0]
        state = State(z, torch.zeros(1, 32), (8, 8))
        planner = PathPlanner(engine, PathValue(64, 32, 4), PathConfig(horizon=3, beam=2, clicks=4))
        planner.set_goal(frame(5).final, np.ones((8, 8), bool))
        action = planner.choose(state, frame(0), random.Random(9))
        self.assertIn(action.id, frame(0).available_actions)
        self.assertEqual(len(planner.last_forecast_plan), 3)
        self.assertEqual(teacher.segment.call_count, 1)

    def test_assignment_is_optimal_and_loss_is_permutation_invariant(self):
        costs = torch.tensor([[1., 2.], [1.1, 100.], [10., 10.]])
        self.assertEqual(set(assignment(costs)), {(0, 1), (1, 0)})
        with self.assertRaises(ValueError): assignment(torch.tensor([[float('nan')]]))
        truth = torch.zeros(1, 3, 8, 8); truth[0, 0, 1, 1] = 1; truth[0, 1, 4:6, 4:6] = 1
        confidence = torch.tensor([[1., .8, 0.]])
        logits = truth*16-8; presence = torch.tensor([[8., 8., -8.]])
        loss, quality = mask_loss((logits, presence), truth, confidence)
        other, _ = mask_loss((logits, presence), truth[:, [1, 2, 0]], confidence[:, [1, 2, 0]])
        self.assertAlmostEqual(loss.item(), other.item(), places=6)
        self.assertEqual(quality, dict(matched=2, mean_iou=1.))
        omitted, _ = mask_loss((logits*0-8, presence), truth, confidence)
        self.assertGreater(omitted.item(), loss.item()+2)

    def test_tiny_foreground_receives_useful_gradient_and_empty_teacher_is_unknown(self):
        truth = torch.zeros(1, 2, 64, 64); truth[0, 0, 3, 5] = 1
        logits = torch.full_like(truth, -6., requires_grad=True)
        presence = torch.zeros(1, 2, requires_grad=True)
        loss, _ = mask_loss((logits, presence), truth, torch.tensor([[1., 0.]]))
        loss.backward()
        self.assertLess(float(logits.grad[0, 0, 3, 5]), -.45)
        self.assertLess(presence.grad[0, 0].item(), 0, 'Matched instance must become present')
        self.assertGreater(presence.grad[0, 1].item(), 0, 'Unmatched presence receives a weak negative')
        empty, quality = mask_loss((logits, presence), truth*0, torch.zeros(1, 2))
        self.assertEqual(empty.item(), 0)
        self.assertEqual(quality['matched'], 0)

    def test_zero_start_and_three_step_gradient_through_predicted_masks(self):
        torch.manual_seed(3)
        old = WorldModel(network(0)).eval()
        new = WorldModel(network()).eval()
        weights = new.state_dict(); weights.update(old.state_dict()); new.load_state_dict(weights)
        e = Engine(new, TrainConfig())
        z = torch.randn(1, 4, 64); h = torch.zeros(1, 32); action = e.actions([Action(4)])
        with torch.no_grad():
            a, b = old.dynamics(z, h, action, 8, 8), e.predict(z, h, action, 8, 8)
            for x, y in zip(a, b): torch.testing.assert_close(x, y, atol=0, rtol=0)
            torch.testing.assert_close(e.planning_features(z, 8, 8), z, atol=0, rtol=0)
        masks = torch.zeros(1, 3, 8, 8); masks[0, 0, 2:4, 2:4] = .8; masks.requires_grad_()
        result = object_rollout(e, z, h, [action]*3, (8, 8), masks)
        self.assertEqual(len(result), 3)
        target = torch.zeros_like(masks); target[0, 0, 3:5, 4:6] = 1
        target[0, 1, 6, 1] = 1  # A new object may occupy a previously empty slot.
        loss, quality = mask_loss(result[-1][2], target, torch.tensor([[1., 1., 0.]]))
        loss.backward()
        self.assertEqual(quality['matched'], 2)
        self.assertGreater(masks.grad.abs().sum().item(), 0)
        for name in ('motion.weight', 'refine.weight', 'memory.weight'):
            gradient = dict(new.dynamics.objects.named_parameters())[name].grad
            self.assertTrue(torch.isfinite(gradient).all(), name)
            self.assertGreater(gradient.abs().sum().item(), 0, name)

    def test_native_transport_can_translate_rotate_and_remove_a_tiny_instance(self):
        head = WorldModel(network()).dynamics.objects
        with torch.no_grad():
            head.refine.weight.zero_(); head.refine.bias.zero_()
            head.presence.weight.zero_(); head.presence.bias.fill_(8.)
            head.motion.bias[0] = torch.atanh(torch.tensor(-.25))
        masks = torch.zeros(1, 3, 8, 8); masks[0, 0, 3, 2] = 1
        args = (torch.zeros(1, 4, 64), torch.zeros(1, 32), torch.zeros(1, 12), 8, 8, masks)
        moved = probabilities(head(*args)[1])[0, 0] > .5
        self.assertEqual(moved.nonzero().tolist(), [[3, 3]], 'One native pixel to the right')
        with torch.no_grad():
            head.motion.bias.zero_(); head.motion.bias[2] = torch.atanh(torch.tensor(.5))
        rotated = probabilities(head(*args)[1])[0, 0] > .5
        self.assertEqual(rotated.nonzero().tolist(), [[2, 4]], 'Quarter turn about image center')
        with torch.no_grad(): head.presence.bias.fill_(-8.)
        self.assertFalse((probabilities(head(*args)[1]) > .5).any())

    def test_copy_migration_preserves_weights_and_adam_by_name(self):
        model = WorldModel(network(0))
        adam = optimizer(model.dynamics.parameters(), .001, False)
        sum(p.square().sum() for p in model.dynamics.parameters()).backward(); adam.step()
        old_plan = dict(initialization=dict(network=asdict(model.config)), settings=dict(seed=1))
        saved = dict(kind='game_sequences', version=1, network=asdict(model.config), plan=old_plan,
                     weights=deepcopy(model.state_dict()), predictor_optimizer=adam.state_dict(),
                     progress=dict(updates=1, batches=2), sequence_control=dict(representation_frozen_at_batch=1))
        plan = deepcopy(old_plan); plan['initialization']['network'] = asdict(network())
        plan['sequence_objects'] = dict(checkpoint='/offline/sam.pt')
        migrated = migrate(saved, SimpleNamespace(payload=plan))
        target = WorldModel(network()); target.load_state_dict(migrated['weights'], strict=True)
        new_adam = optimizer(target.dynamics.parameters(), .001, False)
        new_adam.load_state_dict(migrated['predictor_optimizer'])
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, target.state_dict()[name], atol=0, rtol=0)
        new_params = dict(target.dynamics.named_parameters())
        for name, parameter in model.dynamics.named_parameters():
            for key, value in adam.state[parameter].items():
                torch.testing.assert_close(value, new_adam.state[new_params[name]][key], atol=0, rtol=0)
        self.assertNotIn('sequence_objects', saved['plan'])
        enabled = deepcopy(plan); enabled['sequence_objects']['activate_residual'] = True
        requested = migrate(migrated, SimpleNamespace(payload=enabled))
        self.assertIs(requested['weights'], migrated['weights'])
        self.assertIs(requested['predictor_optimizer'], migrated['predictor_optimizer'])
        self.assertEqual(float(requested['weights']['dynamics.objects.gain']), 0)
        changed = deepcopy(plan); changed['settings']['seed'] = 2
        with self.assertRaisesRegex(ValueError, 'may not change'): migrate(saved, SimpleNamespace(payload=changed))

    def test_auxiliary_fit_updates_only_new_weights_and_retains_optimizer_rates(self):
        torch.manual_seed(9)
        engine = Engine(WorldModel(network()), TrainConfig())
        learner = SequenceLearner(engine, SequenceConfig(batch_games=1, session_steps=4, block_steps=2))
        before = deepcopy(engine.model.state_dict())
        trainer = ObjectTrainer(engine, ObjectTrainingConfig('/fake', samples=2, passes=2), FakeTeacher())
        rates = [g['lr'] for g in learner.optimizer.param_groups]
        metrics, preview = trainer.fit(learner, [session()])
        self.assertEqual(metrics['updates'], 4)
        self.assertTrue(np.isfinite(metrics['loss']))
        self.assertIsNotNone(preview)
        self.assertEqual([g['lr'] for g in learner.optimizer.param_groups], rates)
        after = engine.model.state_dict()
        self.assertTrue(any(not torch.equal(value, after[name]) for name, value in before.items() if 'objects.' in name))
        for name, value in before.items():
            if 'objects.' not in name: torch.testing.assert_close(value, after[name], rtol=0, atol=0)
        self.assertEqual(float(engine.model.dynamics.objects.gain), 0)

    def test_rollout_never_crosses_reset_or_new_level(self):
        s = session()
        self.assertEqual(len(rollout_window(s, 0)), 3)
        s.transitions[1] = replace(s.transitions[1], reset_before=True)
        self.assertEqual(rollout_window(s, 0), [])
        s = session(); s.transitions[1] = replace(s.transitions[1], level=2)
        self.assertEqual(rollout_window(s, 0), [])


if __name__ == '__main__':
    unittest.main()
