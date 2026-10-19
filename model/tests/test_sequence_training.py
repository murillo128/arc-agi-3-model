"""Sequence semantics: independent games, causal gradients and public boundaries."""
from contextlib import nullcontext
from contextlib import redirect_stdout
from dataclasses import asdict, replace
import json
import io
import os
import signal
import shutil
import subprocess
from pathlib import Path
import tempfile
import threading
from urllib.request import urlopen
from unittest.mock import patch
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

from model.agent import AgentConfig
from model.config import NetworkConfig, TrainConfig, SequenceConfig
from model.data import Action, Observation
from model.game_sequences import (GameSession, Transition, EncodedSessions, SequenceLearner,
    collect_session, evaluate_sessions, loss_coefficients, schedule_games)
from model.learning import Engine
from model.networks import WorldModel
from model.tests.test_agent import make_agent, obs


torch.set_num_threads(2)


def session(values=(1, 2, 3, 4), *, game='test', seed=0):
    observations = [obs(x) for x in values]
    return GameSession(game, seed, [Transition(a, Action(1), b)
        for a, b in zip(observations, observations[1:])])


class ScalarDynamics(nn.Module):
    def __init__(self):
        super().__init__()
        self.rho = nn.Parameter(torch.tensor(.5))

    def advance(self, z, h, action):
        return self.rho * h + z.mean(1), action


class ScalarLearner(SequenceLearner):
    def _terms(self, z, h, action, target, before, actual):
        next_h, _ = self.engine.advance(z, h, action)
        value = (next_h[:, 0] - 1).square()
        return torch.stack((value, value * 0), -1)


def scalar_engine():
    model = nn.Module()
    model.config = SimpleNamespace(memory=1)
    model.encoder, model.decoder, model.dynamics = nn.Linear(1, 1), nn.Linear(1, 1), ScalarDynamics()
    return SimpleNamespace(model=model, device=torch.device('cpu'), config=TrainConfig(),
        optimizer_bits=32, autocast=nullcontext, advance=model.dynamics.advance,
        actions=lambda actions: torch.zeros(len(actions), 1),
        encode=lambda observations: [torch.tensor([[float(o.final[0, 0])]]) for o in observations])


class SequenceTrainingTests(unittest.TestCase):
    def test_dashboard_runtime_uses_recent_speed_and_keeps_missing_distinct_from_zero(self):
        node = shutil.which('node')
        if not node:
            self.skipTest('Node is not installed')
        result = subprocess.run([node, str(Path(__file__).with_name('sequences_runtime.cjs'))],
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_gradient_reaches_previous_real_steps_but_stops_at_block_boundary(self):
        e = scalar_engine()
        row = session()
        # Only the last step is supervised. Earlier inputs still build context.
        row.transitions[:2] = [replace(x, action=Action(0)) for x in row.transitions[:2]]
        learner = ScalarLearner(e, SequenceConfig(block_steps=3, context_steps=0))
        encoded = EncodedSessions(e, [row])
        value = learner.block_loss(encoded, 0, 3).sum()
        # h = 1, 2.5, 4.25; L=(4.25-1)^2, dh3/drho=3.
        self.assertAlmostEqual(float(value.detach()), 10.5625)
        value.backward()
        self.assertAlmostEqual(float(e.model.dynamics.rho.grad), 19.5)
        e.model.zero_grad()
        learner.block_loss(encoded, 2, 3).sum().backward()
        # The same carry is rebuilt, but its prefix is intentionally detached.
        self.assertAlmostEqual(float(e.model.dynamics.rho.grad), 16.25)

    def test_differentiable_context_crosses_an_aligned_level_boundary(self):
        e, row = scalar_engine(), session()
        row.transitions[:2] = [replace(x, action=Action(0)) for x in row.transitions[:2]]
        row.transitions[2] = replace(row.transitions[2], reset_before=True)
        learner = ScalarLearner(e, SequenceConfig(block_steps=3, context_steps=2))
        value = learner.block_loss(EncodedSessions(e, [row]), 2, 3).sum()
        # h1=1,h2=2.5; boundary h=4.25; prediction h=5.125.
        # d h_prediction/d rho=5.75 includes BOTH earlier real steps.
        self.assertAlmostEqual(float(value.detach()), 17.015625)
        value.backward()
        self.assertAlmostEqual(float(e.model.dynamics.rho.grad), 47.4375)

    def test_weight_ramp_survives_block_normalization_and_ignores_boundaries(self):
        a, b = session(), session((4, 5, 6))
        a.transitions[1] = replace(a.transitions[1], action=Action(0))
        c = SequenceConfig(block_steps=1, loss_start_weight=.25, loss_warmup_steps=2)
        # A weights .25,0,1; B .25,.625. Three blocks, two active games.
        first = loss_coefficients([a, b], 0, 1, c)
        last = loss_coefficients([a, b], 2, 3, c)
        self.assertAlmostEqual(first[0][0], .3)
        self.assertAlmostEqual(first[1][0], 3/7)
        self.assertAlmostEqual(last[0][0], 1.2)
        self.assertEqual(last[1], [])
        self.assertEqual(loss_coefficients([a, b], 1, 2, c)[0], [0.])

    def test_batch_games_are_independent_and_future_never_changes_earlier_loss(self):
        e = scalar_engine()
        learner = ScalarLearner(e, SequenceConfig(block_steps=3))
        a, b = session(), session((7, 8, 9, 10))
        separate = [learner.block_loss(EncodedSessions(e, [row]), 0, 3) for row in (a, b)]
        together = learner.block_loss(EncodedSessions(e, [a, b]), 0, 3)
        torch.testing.assert_close(together, (separate[0] + separate[1])/2)
        first = learner.block_loss(EncodedSessions(e, [a]), 0, 1)
        a.transitions[2] = replace(a.transitions[2], before=obs(14), after=obs(15))
        torch.testing.assert_close(first, learner.block_loss(EncodedSessions(e, [a]), 0, 1))

    def test_real_agent_preserves_level_context_and_reset_but_new_game_clears_it(self):
        agent = make_agent()
        agent.config = replace(agent.config, memory_across_levels=True)
        agent.observe(obs(1))
        agent.observe(obs(2), Action(1))
        before = agent.state
        with torch.no_grad():
            h, _ = agent.engine.advance(before.z[None], before.h, agent.engine.actions([Action(1)]))
            expected = agent.engine.initial(obs(3, level=1), memory=h)
        agent.observe(obs(3, level=1), Action(1))
        torch.testing.assert_close(agent.state.h, expected.h)
        self.assertGreater(float(agent.state.h.abs().sum()), 0)
        agent.observe(obs(4), Action(0))
        self.assertGreater(float(agent.state.h.abs().sum()), 0)
        from model import agent_checkpoint
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'agent.pt'
            agent_checkpoint.save(path, agent, include_runtime=True)
            restored = agent_checkpoint.load(path, resume_runtime=True)
            self.assertTrue(restored.config.memory_across_levels)
            torch.testing.assert_close(restored.state.h, agent.state.h)
        agent.reset()
        agent.observe(obs())
        self.assertEqual(float(agent.state.h.abs().sum()), 0)

    def test_collector_masks_native_level_jump_and_excludes_reserved_target(self):
        class Env:
            observation = obs(1, level=0)
            def step(self, action):
                self.observation = obs(2, level=self.observation.levels_completed + 1)
                return self.observation
        factory = SimpleNamespace(game_id='native', reserve_final_level=True,
                                  make=lambda seed, level: Env())
        row = collect_session(factory, [1, 2], 0,
            SequenceConfig(session_steps=4, block_steps=2, level_steps=1))
        self.assertEqual(row.environment_actions, 2)
        self.assertEqual(len(row.transitions), 1)
        self.assertFalse(row.transitions[0].supervised)
        self.assertEqual(row.end_reason, 'reserved_level')
        with self.assertRaises(ValueError):
            collect_session(factory, [2], 0, SequenceConfig())

    def test_synthetic_level_rotation_and_partial_sessions_keep_game_identity(self):
        class Env:
            observation = obs(1)
            def step(self, action):
                self.observation = obs(2)
                return self.observation
        seen = []
        def make(seed, level):
            seen.append(level)
            return Env()
        factory = SimpleNamespace(game_id='synthetic', reserve_final_level=False, make=make)
        for levels, expected in (([1], [1, 1, 1, 1]), ([1, 3], [1, 3, 1, 3])):
            with self.subTest(levels=levels):
                seen.clear()
                row = collect_session(factory, levels, 0,
                    SequenceConfig(session_steps=7, block_steps=2, level_steps=2))
                self.assertEqual(seen, expected)
                self.assertEqual([s.reset_before for s in row.transitions],
                                 [False, False, True, False, True, False, True])
                self.assertEqual(row.environment_actions, 7)
                self.assertEqual(row.game, 'synthetic')
                self.assertEqual([s.level_number for s in row.transitions],
                                 [levels[(t//2) % len(levels)] for t in range(7)])
                e = scalar_engine()
                memory = ScalarLearner(e).prefix(EncodedSessions(e, [row]), 3)[0]
                # h1=1, h2=2.5; new board/reset -> 2.25; next action -> 2.125.
                # Clearing session memory at the quota would instead give 1.5.
                self.assertEqual(float(memory), 2.125)
        self.assertEqual(schedule_games(['a', 'b'], 0, SequenceConfig()), ['a', 'b', 'a', 'b'])
        self.assertEqual(schedule_games(['a', 'b'], 1, SequenceConfig(game_schedule='blocked')), ['b']*4)

    def test_real_network_update_variable_geometry_and_stale_encoder_guard(self):
        torch.manual_seed(4)
        engine = Engine(WorldModel(NetworkConfig.small()), TrainConfig(prediction_delta_loss='recall_focal_capped'))
        a, b = session((1, 2, 3)), session((2, 3))
        b.transitions = [replace(t, before=replace(t.before, frames=np.ones((1, 4, 8), dtype=np.uint8)),
                               after=replace(t.after, frames=np.full((1, 4, 8), 2, dtype=np.uint8))) for t in b.transitions]
        learner = SequenceLearner(engine, SequenceConfig(block_steps=2))
        encoded = EncodedSessions(engine, [a, b])
        before = {k: p.clone() for k, p in engine.model.encoder.state_dict().items()}
        result = learner.update(encoded, 0, 2)
        self.assertEqual(result['examples'], 3)
        self.assertTrue(np.isfinite(result['loss']))
        for k, p in engine.model.encoder.state_dict().items():
            torch.testing.assert_close(before[k], p, rtol=0, atol=0)
        with torch.no_grad():
            next(engine.model.encoder.parameters()).add_(1)
        with self.assertRaisesRegex(ValueError, 'Encoder changed'):
            learner.update(encoded, 0, 2)

    def test_invalid_sequence_options_fail_early(self):
        for kwargs in ({'batch_games': True}, {'session_steps': 0}, {'block_steps': 513},
                       {'loss_start_weight': 0}, {'loss_start_weight': float('nan')},
                       {'representation_updates': -1}, {'game_schedule': 'shuffle_steps'}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SequenceConfig(**kwargs)

    def test_native_level_boundary_can_be_context_but_never_delta_target(self):
        row = session((1, 2, 3))
        row.transitions[0] = replace(row.transitions[0], after=obs(12, level=1))
        row.transitions[1] = replace(row.transitions[1], before=obs(12, level=1),
                                    after=obs(13, level=1), reset_before=True)
        self.assertFalse(row.transitions[0].supervised)
        self.assertTrue(row.transitions[1].supervised)
        self.assertEqual(loss_coefficients([row], 0, 2, SequenceConfig(block_steps=2)), [[0., 1.]])

    def test_empty_sdk_response_preserves_visual_context_without_becoming_a_target(self):
        empty = obs(empty=True)
        row = GameSession('example', 0, [Transition(obs(1), Action(1), empty),
            Transition(empty, Action(1), obs(2)), Transition(obs(2), Action(1), obs(3))])
        engine = scalar_engine()
        learner = ScalarLearner(engine, SequenceConfig())
        encoded = EncodedSessions(engine, [row])
        self.assertEqual([step.supervised for step in row.transitions], [False, False, True])
        self.assertAlmostEqual(float(learner.prefix(encoded, 2)[0]), 1.5)

    def test_full_training_resume_preserves_weights_optimizer_and_next_session_schedule(self):
        from model import checkpoint
        from model.sequence_training import run
        from model.tests.test_training_config import config
        from model.training_config import load_plan
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            path, values = config(root, [dict(source='synthetic', game='sc01-v1', levels=[1, 2]),
                                        dict(source='synthetic', game='si01-v1', levels=[1, 2])],
                sequence=dict(batch_games=4, session_steps=4, block_steps=2, level_steps=2,
                              representation_updates=1))
            values['settings']['actions'] = 32
            path.write_text(json.dumps(values))
            def args(out, resume=False, maximum=None):
                return SimpleNamespace(training_config=path, output=root/out, source=None,
                                       resume=resume, device='cpu', compile=False, max_batches=maximum)
            from model.sequence_training import fit_batch
            def interrupted(*a, **kw):
                os.kill(os.getpid(), signal.SIGTERM)
                return fit_batch(*a, **kw)
            with patch('model.sequence_training.fit_batch', side_effect=interrupted):
                run(args('resumed'))
            first = checkpoint.load(root/'resumed/latest.pt')
            self.assertEqual(json.loads((root/'resumed/report.json').read_text())['status'], 'paused')
            self.assertEqual(first['progress']['actions'], 16)
            self.assertEqual(first['progress']['batches'], 1)
            self.assertTrue(first['predictor_optimizer']['state'])
            run(args('resumed', resume=True))
            run(args('continuous'))
            resumed, continuous = [checkpoint.load(root/name/'latest.pt') for name in ('resumed', 'continuous')]
            self.assertEqual(resumed['progress'], continuous['progress'])
            for k in resumed['weights']:
                torch.testing.assert_close(resumed['weights'][k], continuous['weights'][k], rtol=0, atol=0)
            for state, expected in zip(resumed['predictor_optimizer']['state'].values(),
                                       continuous['predictor_optimizer']['state'].values()):
                for k in state:
                    torch.testing.assert_close(state[k], expected[k], rtol=0, atol=0)
            report = json.loads((root/'resumed/report.json').read_text())
            self.assertEqual(report['status'], 'budget_complete')
            self.assertEqual([r['batch'] for r in report['history']], [1, 2])
            self.assertEqual([s['game'] for s in report['history'][-1]['sessions']],
                             ['sc01-v1', 'si01-v1', 'sc01-v1', 'si01-v1'])
            values['sequence']['loss_start_weight'] = .25
            path.write_text(json.dumps(values))
            with self.assertRaisesRegex(ValueError, 'same sequence'):
                run(args('resumed', resume=True))

    def test_unlimited_resume_passes_budget_and_keeps_manual_stop_and_optimizer_state(self):
        from model import checkpoint
        from model.sequence_training import run, fit_batch
        from model.tests.test_training_config import config
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            path, values = config(root, [dict(source='synthetic', game='sc01-v1', levels=[1, 2])],
                sequence=dict(batch_games=4, session_steps=4, block_steps=2, context_steps=1,
                              level_steps=2, representation_updates=1))
            values['settings']['actions'] = 16
            path.write_text(json.dumps(values))
            def args(name, *, resume=False, unlimited=False, maximum=None):
                return SimpleNamespace(training_config=path, output=root/name, source=None,
                    resume=resume, unlimited=unlimited, device='cpu', compile=False, max_batches=maximum)
            def report():
                return json.loads((root/'resumed/report.json').read_text())
            run(args('resumed'))
            first = checkpoint.load(root/'resumed/latest.pt')
            self.assertEqual(first['progress']['actions'], 16)
            self.assertEqual(report()['status'], 'budget_complete')
            self.assertEqual(report()['budget'], 16)

            run(args('resumed', resume=True, unlimited=True, maximum=1))
            self.assertEqual(report()['progress']['actions'], 32)
            self.assertIsNone(report()['budget'])
            self.assertEqual(report()['status'], 'paused')

            def interrupted(*a, **kw):
                os.kill(os.getpid(), signal.SIGINT)
                return fit_batch(*a, **kw)
            with patch('model.sequence_training.fit_batch', side_effect=interrupted):
                run(args('resumed', resume=True, unlimited=True))
            self.assertEqual(report()['progress']['actions'], 48)
            self.assertEqual(report()['status'], 'paused')
            self.assertIsNone(report()['budget'])
            self.assertEqual([r['batch'] for r in report()['history']], [1, 2, 3])

            run(args('continuous', unlimited=True, maximum=3))
            resumed, continuous = [checkpoint.load(root/name/'latest.pt') for name in ('resumed', 'continuous')]
            self.assertEqual(resumed['plan'], first['plan'])
            self.assertEqual(resumed['progress'], continuous['progress'])
            for name, expected in continuous['weights'].items():
                torch.testing.assert_close(resumed['weights'][name], expected, rtol=0, atol=0)
            for optimizer in ('predictor_optimizer', 'representation_optimizer'):
                self.assertEqual(resumed[optimizer]['param_groups'], continuous[optimizer]['param_groups'])
                for key, expected in continuous[optimizer]['state'].items():
                    for field, value in expected.items():
                        torch.testing.assert_close(resumed[optimizer]['state'][key][field], value, rtol=0, atol=0)
            torch.testing.assert_close(resumed['torch_rng'], continuous['torch_rng'], rtol=0, atol=0)

    def test_sequence_dashboard_serves_selected_run_and_escapes_no_values_into_html(self):
        from model.dashboard import make_server
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'report.json'
            path.write_text(json.dumps(dict(kind='game_sequences', status='paused', history=[],
                                            games=['<not-html>'], progress=dict(actions=16))))
            server = make_server(('127.0.0.1', 0), None, sequence_report=path)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                url = f'http://127.0.0.1:{server.server_port}'
                with urlopen(url+'/api/state') as response:
                    data = json.load(response)
                self.assertEqual(data['sequences']['progress']['actions'], 16)
                self.assertNotIn('subgoals', data)
                with urlopen(url) as response:
                    html = response.read().decode()
                self.assertIn('sequences.js', html)
                self.assertNotIn('<not-html>', html)
                with self.assertRaises(ValueError):
                    make_server(('127.0.0.1', 0), None, subgoal_report=path, sequence_report=path)
            finally:
                server.shutdown(); server.server_close(); thread.join()


if __name__ == '__main__':
    unittest.main()
