"""Real-prefix ownership, current recurrent context and one averaged batch step."""
from contextlib import nullcontext, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from model import agent_checkpoint, checkpoint
from model.config import TrainConfig
from model.data import Action, Episode, Observation
from model.learning import State
from model.online_learning import OnlinePredictor
from model.online_replay import PredictorReplay
from model.planning import fingerprint
from model.subgoal_training import main, play
from model.tests.test_subgoals import path_agent


def observation(color, level=0):
    return Observation(np.full((1, 1, 1), color, dtype=np.uint8), (1, 2), 'NOT_FINISHED', level, 5)


class ReplayTests(unittest.TestCase):
    def test_observed_prefixes_distinct_episodes_capacity_and_boundary_exclusion(self):
        engine = SimpleNamespace(config=TrainConfig(replay_bytes=100))
        replay = PredictorReplay(engine, episodes=3, batch_size=3, seed=9)
        sources = []
        for index in range(4):
            episode = Episode([observation(index)])
            for step in range(1, 4):
                episode.append(Action(1), observation(index + step))
                self.assertTrue(replay.record(episode))
            sources.append(episode)
        self.assertEqual(replay.status(), dict(capacity=3, episodes=3, transitions=9, batch_size=3))
        prefixes = replay.sample()
        self.assertEqual(len(prefixes), 3)
        self.assertEqual([o.final.item() for o in prefixes[0].observations], [3, 4, 5, 6])
        self.assertEqual({e.observations[0].final.item() for e in prefixes}, {1, 2, 3})
        self.assertTrue(all(1 <= len(e.actions) <= 3 for e in prefixes))
        # Appending an unseen level must not mutate the saved observed prefix.
        sources[-1].append(Action(1), observation(15, level=1))
        self.assertFalse(replay.record(sources[-1]))
        self.assertEqual(replay.status()['transitions'], 9)
        self.assertEqual([o.final.item() for o in replay.sample()[0].observations], [3, 4, 5, 6])
        # Raw byte budget evicts older episodes independently of capacity.
        replay.buffer.byte_limit = 4
        replay.buffer.trim()
        self.assertEqual(replay.status()['episodes'], 1)
        with self.assertRaisesRegex(ValueError, 'must not update weights'):
            play(None, None, 1, 1, 1, quality_horizon=3, predictor_replay=replay)

    def test_replayed_memory_is_rebuilt_with_current_dynamics_and_real_targets(self):
        model = nn.Module()
        model.config = SimpleNamespace(memory=1)
        model.weight = nn.Parameter(torch.tensor(1.))
        engine = SimpleNamespace(model=model, config=TrainConfig(), device=torch.device('cpu'), autocast=nullcontext,
            encode_detached_episode=lambda ep: [torch.tensor([[float(o.final.item())]]) for o in ep.observations],
            actions=lambda acts: torch.tensor([[a.id] for a in acts]),
            advance=lambda z, h, a: (h + z.mean(1) + model.weight * a, None))
        replay = PredictorReplay(engine, episodes=2, batch_size=2, horizon=2)
        episode = Episode([observation(1), observation(2), observation(3)], [Action(1), Action(2)])
        replay.record(episode)
        first = replay.examples(replay.sample())[0]
        self.assertEqual(first['state'].h.item(), 2.)
        with torch.no_grad():
            model.weight.fill_(2.)
        second = replay.examples(replay.sample())[0]
        self.assertEqual(second['state'].h.item(), 3.)
        self.assertEqual(second['target_z'].item(), 3.)
        self.assertFalse(second['state'].h.requires_grad)
        self.assertEqual([a.id for a in second['rollout'][1]], [1, 2])
        self.assertEqual(second['rollout'][0].h.item(), 0.)

    def test_minibatch_averages_objective_and_gradients_before_one_optimizer_step(self):
        class Dynamics(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(0.))

            def forward(self, z, h, action, height, width):
                return z + self.weight, h, torch.ones_like(z) * self.weight

        class Decoder(nn.Module):
            def forward(self, z, height, width, **kwargs):
                return z.sum() * 0 + torch.zeros(1, height, width, 16)

        model = nn.Module()
        model.encoder, model.decoder, model.dynamics = nn.Linear(1, 1), Decoder(), Dynamics()
        engine = SimpleNamespace(model=model, config=TrainConfig(prediction_delta_loss='union', delta_overlap_weight=2.),
            device=torch.device('cpu'), optimizer_bits=32, autocast=nullcontext,
            predict=model.dynamics, decode=model.decoder, actions=lambda a: torch.zeros(len(a), 12))
        learner = OnlinePredictor(engine, threshold=None)
        state = State(torch.zeros(1, 1), torch.zeros(1, 1), (1, 1))
        before = np.zeros((1, 1), dtype=np.uint8)
        samples = [dict(state=state, action=Action(1), reference=before,
                        after=observation(0), target_z=torch.tensor([[float(target)]])) for target in (1, 3)]
        samples[0]['rollout'] = (state, [Action(1)] * 3, before)
        gradients = []
        model.dynamics.weight.register_hook(lambda g: gradients.append(float(g)))
        result = learner.update_batch(samples)
        # At w=0: MSE=(1+9)/2; rollout=(1-3w)^2/2; gradient=-1-3-3=-7.
        self.assertEqual(result['latent'], 5.)
        self.assertEqual(result['rollout'], .5)
        self.assertEqual(result['total'], 5.5)
        self.assertEqual(set(result['terms']), {'latent', 'delta_nll', 'rollout'})
        self.assertEqual(sum(gradients), -7.)
        self.assertEqual(result['batch_size'], 2)
        self.assertEqual(learner.updates, 1)
        self.assertTrue(all(float(s['step']) == 1 for s in learner.optimizer.state.values()))
        self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))

    def test_resume_enables_replay_preserves_weights_and_inherits_settings_with_empty_buffer(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            agent_checkpoint.save(root / 'source.pt', path_agent())
            args = ['--source', str(root / 'source.pt'), '--output', str(root / 'run'), '--device', 'cpu',
                    '--steps', '2', '--evaluate-every', '4', '--evaluation-episodes', '1', '--freeze-encoder']
            main([*args, '--actions', '4'])
            filename = root / 'run/latest.pt'
            saved = checkpoint.load(filename)
            saved['training_progress'].pop('predictor_replay')
            saved['training_progress']['curriculum']['levels'][0].update(
                predictor=True, representation_passes=3, prediction_passes=2, promotion_passes=1)
            checkpoint.write(filename, saved)
            expected = fingerprint(agent_checkpoint.load(filename).engine.model)
            original_save = agent_checkpoint.save
            snapshots = []

            def capture(path, agent, **kwargs):
                progress = kwargs['training_progress']
                if progress['actions'] == 4:
                    self.assertEqual(fingerprint(agent.engine.model), expected)
                    self.assertEqual(progress['updates'], 0)
                    self.assertEqual(progress['representation_updates'], 4)
                    self.assertEqual(progress['predictor_replay'], dict(episodes=3, batch_size=3))
                    snapshots.append(progress)
                return original_save(path, agent, **kwargs)

            with patch('model.subgoal_training.agent_checkpoint.save', side_effect=capture):
                main([*args, '--actions', '12', '--resume', '--predictor-replay-episodes', '3', '--predictor-batch-size', '3'])
            self.assertEqual(len(snapshots), 1)
            readiness = snapshots[0]['curriculum']['levels'][0]
            self.assertTrue(readiness['predictor'])
            self.assertFalse(readiness['planner'])
            self.assertEqual([readiness[k] for k in ('representation_passes', 'prediction_passes', 'promotion_passes')], [0, 0, 0])
            report = json.loads((root / 'run/report.json').read_text())
            self.assertEqual(report['progress']['predictor_updates'], 8)
            self.assertEqual(report['progress']['predictor_replay'],
                dict(capacity=3, episodes=3, transitions=6, batch_size=3, actual_batch_size=3))
            self.assertEqual(report['resumed_from']['predictor_replay'],
                dict(old=dict(episodes=0, batch_size=1), new=dict(episodes=3, batch_size=3)))
            main([*args, '--actions', '14', '--resume'])
            report = json.loads((root / 'run/report.json').read_text())
            self.assertEqual(report['config']['predictor_replay_episodes'], 3)
            self.assertEqual(report['config']['predictor_batch_size'], 3)
            self.assertEqual(report['resumed_from']['predictor_replay_buffer'], 'empty')
            self.assertNotIn('predictor_replay', report['resumed_from'])
            self.assertEqual(report['progress']['predictor_updates'], 10)
            self.assertEqual(report['progress']['predictor_replay'],
                dict(capacity=3, episodes=1, transitions=2, batch_size=3, actual_batch_size=1))
            main([*args, '--actions', '16', '--resume', '--predictor-replay-episodes', '0', '--predictor-batch-size', '1'])
            report = json.loads((root / 'run/report.json').read_text())
            self.assertIsNone(report['progress']['predictor_replay'])
            self.assertEqual(report['progress']['predictor_updates'], 12)
