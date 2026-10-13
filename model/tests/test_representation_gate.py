"""Hysteresis and real update/resume boundaries, independent of game mechanics."""
from contextlib import redirect_stdout
from dataclasses import replace
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from model import checkpoint
from model.config import NetworkConfig, TrainConfig
from model.learning import Engine, Learner
from model.networks import WorldModel
from model.representation_gate import representation_gate
from model.runner import main
from model.tests.test_model import episode


def report(error, **overrides):
    return dict(decoder_delta_error=error, decoder_transitions=128, transitions=128,
                changed_transitions=32, latent_std=.2, delta_error=.9, passes=False) | overrides


class GateTests(unittest.TestCase):
    def test_hysteresis_uses_reconstruction_and_consecutive_evidence(self):
        c = TrainConfig(auto_representation=True, freeze_representation=True)
        state = None
        # Exactly 2% does not open; exactly 1% may close. Dead band resets streaks.
        for error, expected, streak in [(.02, True, 0), (.03, True, 1), (.015, True, 0),
                (.03, True, 1), (.04, False, 0), (.01, False, 1), (.015, False, 0),
                (.01, False, 1), (.01, True, 0)]:
            state = representation_gate(c, {'1': report(error)}, [1], state)
            self.assertEqual(state['freeze_representation'], expected)
            self.assertEqual(max(state['open_streak'], state['close_streak']), streak)
            c = replace(c, freeze_representation=expected)

    def test_all_levels_must_be_eligible_and_low_to_freeze(self):
        c = TrainConfig(auto_representation=True, representation_gate_patience=1)
        for reports, levels, expected in [
            ({'1': report(0.), '2': report(.015)}, [1, 2], False),
            ({'1': report(0.), '2': report(.005)}, [1, 2], True),
            ({'1': report(0.)}, [1, 2], False),
            ({'None': report(.01)}, [None], True),
            ({}, [], False),
            ({'1': report(0., latent_std=.01)}, [1], False),
        ]:
            self.assertEqual(representation_gate(c, reports, levels)['freeze_representation'], expected)
        c = replace(c, freeze_representation=True)
        self.assertFalse(representation_gate(c, {'1': report(.04), '2': report(0.)}, [1, 2])['freeze_representation'])

    def test_invalid_or_insufficient_evidence_breaks_streak(self):
        c = TrainConfig(auto_representation=True, freeze_representation=True)
        first = representation_gate(c, {'1': report(.1)}, [1])
        for bad in [report(float('nan')), report(float('inf')), report(-.1), report(1.1),
                    report(None), report(.1, latent_std=float('nan')),
                    report(.1, decoder_transitions=127), report(.1, changed_transitions=31),
                    report(.1, transitions=127), {}, None]:
            state = representation_gate(c, {'1': bad}, [1], first)
            self.assertFalse(state['eligible'])
            self.assertEqual(state['open_streak'], 0)
            state = representation_gate(c, {'1': report(.1)}, [1], state)
            self.assertTrue(state['freeze_representation'])

    def test_disabled_policy_and_new_scope_reset(self):
        c = TrainConfig(auto_representation=True, freeze_representation=True)
        first = representation_gate(c, {'1': report(.1)}, [1])
        state = representation_gate(c, {'1': report(.1), '2': report(.1)}, [1, 2], first)
        self.assertTrue(state['freeze_representation'])
        self.assertEqual(state['open_streak'], 1)
        state = representation_gate(replace(c, auto_representation=False), {'1': report(.1)}, [1], first)
        self.assertEqual(state['reason'], 'disabled')
        self.assertTrue(state['freeze_representation'])
        state = representation_gate(replace(c, representation_unfreeze_error=.03), {'1': report(.1)}, [1], first)
        self.assertEqual(state['open_streak'], 1)

    def test_config_rejects_invalid_policy(self):
        for kwargs in [dict(representation_freeze_error=.02), dict(representation_unfreeze_error=2.),
                       dict(representation_freeze_error=-.1), dict(representation_gate_patience=0),
                       dict(representation_gate_patience=1.5), dict(auto_representation=1)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TrainConfig(**kwargs)

    def exercise_updates(self, device='cpu', compiled=False, bits=32):
        torch.manual_seed(18)
        learner = Learner(Engine(WorldModel(NetworkConfig.small()).to(device),
            TrainConfig(frame_chunk=2, loss_steps=2), compiled, optimizer_bits=bits))
        learner.update([episode(2)])
        original_parameters = list(learner.model.parameters())
        original_optimizers = list(learner.optimizers)
        original = {n: p.detach().clone() for n, p in learner.model.named_parameters()}
        moments = [(opt, p, copy.deepcopy(opt.state[p])) for opt, module in (
            (learner.optimizers[0], learner.model.encoder), (learner.optimizers[1], learner.model.decoder))
            for p in module.parameters()]
        learner.set_representation_frozen(True)
        result = learner.update([episode(2)])
        self.assertIsNone(result['reconstruction_ce'])
        for name, p in learner.model.named_parameters():
            if not name.startswith('dynamics.'):
                self.assertTrue(torch.equal(original[name], p), name)
                self.assertIsNone(p.grad)
        self.assertTrue(any(not torch.equal(original[n], p) for n, p in learner.model.named_parameters()
                            if n.startswith('dynamics.')))
        for opt, p, previous in moments:
            for key, value in previous.items():
                if isinstance(value, torch.Tensor):
                    self.assertTrue(torch.equal(opt.state[p][key], value))
                else:
                    self.assertEqual(opt.state[p][key], value)
        learner.set_representation_frozen(False)
        self.assertTrue(all(a is b for a,b in zip(original_parameters, learner.model.parameters())))
        self.assertTrue(all(a is b for a,b in zip(original_optimizers, learner.optimizers)))
        result = learner.update([episode(2)])
        self.assertIsNotNone(result['reconstruction_ce'])
        for prefix in ('encoder.', 'decoder.', 'dynamics.'):
            self.assertTrue(any(not torch.equal(original[n], p) for n, p in learner.model.named_parameters()
                                if n.startswith(prefix)), prefix)
        # Enter the frozen compiled specialization again; stale gradients must clear.
        original = {n: p.detach().clone() for n, p in learner.model.named_parameters()}
        learner.set_representation_frozen(True)
        learner.update([episode(2)])
        self.assertTrue(all(torch.equal(original[n], p) for n,p in learner.model.named_parameters()
                            if not n.startswith('dynamics.')))

    def test_live_mode_transitions_keep_optimizer_ownership(self):
        self.exercise_updates()

    @unittest.skipUnless(os.environ.get('ARC3_TEST_COMPILED_GATE') == '1' and torch.cuda.is_available(),
                         'explicit bounded CUDA compile smoke')
    def test_compiled_cuda_mode_transitions(self):
        self.exercise_updates('cuda', True, 8)

    def test_cli_resume_keeps_partial_streak_and_checkpoints_actual_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            common = ['--device', 'cpu', '--no-compile', '--output', directory]
            path = Path(directory) / 'latest.pt'
            with patch('model.runner.validation', return_value={'1': report(0.)}), redirect_stdout(io.StringIO()):
                main(['train', '--profile', 'small', '--levels', '1', '--auto-representation',
                      '--validate-every', '1', '--checkpoint-every', '100', '--max-steps', '1', *common])
            saved = checkpoint.load(path)
            self.assertEqual(saved['progress']['representation_gate']['close_streak'], 1)
            self.assertFalse(saved['training']['freeze_representation'])
            output = io.StringIO()
            with patch('model.runner.validation', return_value={'1': report(0.)}), redirect_stdout(output):
                main(['train', '--checkpoint', str(path), '--max-steps', '2', '--auto-representation', *common])
            saved = checkpoint.load(path)
            self.assertTrue(saved['training']['freeze_representation'])
            events = [json.loads(line) for line in output.getvalue().splitlines()]
            self.assertTrue(any(e['event']=='checkpoint' and e['actions']==2 for e in events))
            self.assertTrue(any(e['event']=='representation_gate' and e['changed'] for e in events))
            with patch('model.runner.validation', return_value={'1': report(.1)}), redirect_stdout(io.StringIO()):
                main(['train', '--checkpoint', str(path), '--max-steps', '3', *common])
                main(['train', '--checkpoint', str(path), '--max-steps', '4', *common])
            saved = checkpoint.load(path)
            self.assertFalse(saved['training']['freeze_representation'])
            self.assertEqual(saved['progress']['passes'], 0)
            self.assertEqual(saved['progress']['representation_gate']['reason'], 'reconstruction_high')


if __name__ == '__main__':
    unittest.main()
