"""Ordinary predictor steps respect the inactive object branch's Adam history."""
from copy import deepcopy
from dataclasses import asdict
import unittest

import torch

from model.config import SequenceConfig, TrainConfig
from model.game_sequences import EncodedSessions, SequenceLearner
from model.learning import Engine
from model.networks import WorldModel
from model.sequence_upgrade import normalized_plan
from model.tests.test_objects import network, session


class ObjectOptimizerTests(unittest.TestCase):
    def test_legacy_object_plan_gains_only_observer_defaults(self):
        old = dict(initialization=dict(network=asdict(network())),
                   sequence_objects=dict(checkpoint='/offline/sam.pt', learning_rate=.001))
        current = deepcopy(old)
        current['sequence_objects'].update(observer_steps=32, observer_replay_frames=64)
        self.assertEqual(normalized_plan(old), normalized_plan(current))
        self.assertNotIn('observer_steps', old['sequence_objects'])
        for name, value in [('observer_steps', 16), ('observer_replay_frames', 128),
                            ('learning_rate', .01), ('checkpoint', '/other.pt')]:
            changed = deepcopy(current)
            changed['sequence_objects'][name] = value
            self.assertNotEqual(normalized_plan(old), normalized_plan(changed), name)

    def test_predictor_preserves_inactive_objects_but_trains_active_residual(self):
        for gain in (0., 1.):
            with self.subTest(gain=gain):
                torch.manual_seed(41)
                engine = Engine(WorldModel(network()), TrainConfig())
                learner = SequenceLearner(engine, SequenceConfig(
                    batch_games=1, session_steps=4, block_steps=2))
                head = engine.model.dynamics.objects
                # A real Adam step supplies nonzero momentum; fresh Adam would
                # miss the defect when the subsequent gradient is exactly zero.
                sum(p.square().sum() + p.sum() for p in head.parameters()).backward()
                learner.optimizer.step()
                learner.optimizer.zero_grad(set_to_none=True)
                head.gain.fill_(gain)
                weights = {name: p.detach().clone() for name, p in head.named_parameters()}
                states = {name: deepcopy(learner.optimizer.state[p])
                          for name, p in head.named_parameters()}
                predictor_before = engine.model.dynamics.delta.weight.detach().clone()
                result = learner.update(EncodedSessions(engine, [session()]), 0, 2)
                self.assertIsNotNone(result)
                self.assertFalse(torch.equal(predictor_before, engine.model.dynamics.delta.weight))
                if gain:
                    self.assertFalse(torch.equal(weights['output.weight'], head.output.weight))
                    self.assertEqual(learner.optimizer.state[head.output.weight]['step'].item(),
                                     states['output.weight']['step'].item() + 1)
                else:
                    for name, parameter in head.named_parameters():
                        torch.testing.assert_close(parameter, weights[name], rtol=0, atol=0)
                        for key, value in states[name].items():
                            torch.testing.assert_close(learner.optimizer.state[parameter][key],
                                                       value, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
