"""Portable policy consumers share the sequence model's object context."""
from contextlib import contextmanager
from dataclasses import replace
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from model import adaptation
from model.adaptation_checkpoint import export
from model.adaptation_heads import goal_loss, path_loss
from model.adaptation_smoke import base_digest
from model.goal_observation import compose_goal_observation
from model.tests.test_adaptation import no_learning, observed
from model.tests.test_adaptation_checkpoint import sequence_snapshot


@contextmanager
def portable_policy(*, slots=3, gain=1.):
    with tempfile.TemporaryDirectory() as directory, torch.random.fork_rng(devices=[]):
        torch.manual_seed(145)
        snapshot = sequence_snapshot(object_slots=slots)
        if slots:
            weights = snapshot['weights']
            weights['dynamics.objects.gain'].fill_(gain)
            weights['dynamics.objects.updates'].fill_(100)
            weights['dynamics.objects.output.weight'].fill_(.05)
            # Vary channels: a uniform shift can disappear under head LayerNorm.
            weights['dynamics.objects.output.bias'].copy_(torch.arange(4).float()*.25)
        # Exercise a learned Path readout, not its constant initial prediction.
        for key in ('success.weight', 'readout.2.weight'):
            value = snapshot['planning']['path'][key]
            value.copy_(torch.linspace(-.1, .1, value.numel()).reshape_as(value))
        source, asset = Path(directory)/'source.pt', Path(directory)/'base.pt'
        torch.save(snapshot, source)
        export(source, asset)
        yield adaptation.load(asset, config=no_learning(), seed=23)


class AdaptiveObjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_goal_inference_uses_active_context_and_preserves_disabled_predictions(self):
        for slots, gain in ((0, 0.), (3, 0.), (3, 1.)):
            with self.subTest(slots=slots, gain=gain), portable_policy(slots=slots, gain=gain) as policy:
                runtime, seen = policy.runtime, []
                handle = runtime.goal.register_forward_hook(lambda module, args, output:
                    seen.append((args[0].detach().clone(), tuple(x.detach().clone() for x in output))))
                try:
                    action = policy.choose_action(observed())
                finally:
                    handle.remove()
                self.assertEqual(action.id, 4)
                self.assertEqual(len(seen), 1)
                state, engine = policy.state, runtime.engine
                with runtime.router.activate(policy.bank), torch.no_grad():
                    features = engine.planning_features(state.z[None], *state.geometry)
                    wanted = runtime.goal(features, state.h, *state.geometry)
                    raw = runtime.goal(state.z[None], state.h, *state.geometry)
                if slots and gain:
                    self.assertFalse(torch.equal(features, state.z[None]))
                    self.assertFalse(torch.equal(wanted[0], raw[0]), 'Fixture must detect missing context')
                else:
                    torch.testing.assert_close(features, state.z[None], rtol=0, atol=0)
                torch.testing.assert_close(seen[0][0], features, rtol=0, atol=0)
                for actual, expected in zip(seen[0][1], wanted):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_goal_and_path_objectives_use_context_with_existing_labels_and_private_updates(self):
        for name in ('goal', 'path'):
            with self.subTest(name=name), portable_policy() as policy:
                # Public confirmation and native success supply the existing
                # accepted labels; the terminal display is not a Goal target.
                policy.choose_action(observed(1))
                policy.confirm_result(observed(3))
                if name == 'goal':
                    policy.choose_action(observed(3))
                    policy.confirm_result(observed(4, state='WIN'))
                row = next(r for r in policy._examples(name) if (r['start'], r['end']) == (0, 1))
                np.testing.assert_array_equal(row['frame'], observed(3).final)
                runtime, engine = policy.runtime, policy.runtime.engine
                base_before = base_digest(runtime)
                bank_before = {g: [p.detach().clone() for p in policy.bank.parameters((g,))]
                               for g in policy.bank.groups}
                with runtime.router.activate(policy.bank, groups=(name,)):
                    states = policy._states()
                    state = states[row['start']]
                    with torch.no_grad():
                        features = engine.planning_features(state.z[None], *state.geometry)
                        self.assertFalse(torch.equal(features, state.z[None]))
                        if name == 'goal':
                            logits, masks = runtime.goal(features, state.h, *state.geometry)
                            expected = goal_loss(logits, masks, torch.tensor(observed(1).final)[None].long(),
                                torch.tensor(row['frame'])[None].long(), torch.tensor(row['mask'])[None])
                        else:
                            frame = compose_goal_observation(row['frame'], row['mask'], observed(1).final)
                            raw_goal = engine.encode([replace(observed(1), frames=frame[None])])[0][None]
                            goal_features = engine.planning_features(raw_goal, *state.geometry)
                            self.assertFalse(torch.equal(goal_features, raw_goal))
                            patch = engine.model.config.patch
                            mask = F.avg_pool2d(torch.tensor(row['mask'])[None, None].float(),
                                               patch, patch, ceil_mode=True).flatten(2).transpose(1, 2)
                            height, width = state.geometry
                            logits, cost = runtime.path(features, state.h, goal_features, mask,
                                (height+patch-1)//patch, (width+patch-1)//patch, torch.tensor([row['budget']]))
                            expected = path_loss(logits, cost, torch.tensor([float(row['success'])]),
                                                 torch.tensor([math.log1p(row['distance'])]))
                    consumed = []
                    head = getattr(runtime, name)
                    handle = head.register_forward_pre_hook(lambda module, args:
                        consumed.append(tuple(x.detach().clone() if torch.is_tensor(x) else x for x in args)))
                    try:
                        objective = policy._objective(name, samples=[row], states=states)
                        self.assertTrue(policy._step(name, (name,), objective))
                    finally:
                        handle.remove()
                    self.assertEqual(len(consumed), 2, 'Measure before and after the optimizer update')
                    for inputs in consumed:
                        torch.testing.assert_close(inputs[0], features, rtol=0, atol=0)
                        if name == 'path':
                            torch.testing.assert_close(inputs[2], goal_features, rtol=0, atol=0)
                    self.assertAlmostEqual(policy.diagnostics['losses'][name]['pre'], float(expected), places=6)
                    self.assertTrue(math.isfinite(policy.diagnostics['losses'][name]['post']))
                self.assertEqual(policy.diagnostics['updates'][name], 1)
                self.assertEqual(base_digest(runtime), base_before)
                for group, before in bank_before.items():
                    changed = any(not torch.equal(a, b) for a, b in
                                  zip(before, policy.bank.parameters((group,))))
                    self.assertEqual(changed, group == name, group)


if __name__ == '__main__':
    unittest.main()
