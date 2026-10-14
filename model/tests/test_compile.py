"""Compilation guards across joint/frozen learning and single/beam inference."""
import copy
from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from model.config import NetworkConfig
from model.learning import _CompiledDynamics, Engine
from model.config import TrainConfig
from model.networks import WorldModel


class CompileTests(unittest.TestCase):
    def test_decoder_joint_frozen_and_both_readouts_reuse_caches_and_gradients(self):
        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)
        torch.set_num_threads(2)
        model = WorldModel(replace(NetworkConfig.small(), latent=32, pixel_latent=True))
        reference = copy.deepcopy(model.decoder)
        graphs = []
        def backend(graph, inputs):
            graphs.append(graph)
            return graph.forward
        compiler = torch.compile
        with patch('torch.compile', side_effect=lambda f, **kw: compiler(f, backend=backend, **kw)):
            engine = Engine(model, TrainConfig(), compile_model=True)
        count = None
        for repeat in range(2):
            for learning in (True, False):
                for joint in (True, False):
                    for local in (False, True):
                        for batch in (1, 4, 16):
                            model.train(learning).requires_grad_(joint)
                            reference.train(learning).requires_grad_(joint)
                            model.zero_grad(set_to_none=True)
                            reference.zero_grad(set_to_none=True)
                            z = torch.randn(batch, 4, 32)
                            if repeat:
                                z = z.transpose(1, 2).contiguous().transpose(1, 2)
                            z.requires_grad_(learning)
                            other = z.detach().clone().requires_grad_(learning)
                            with torch.set_grad_enabled(learning):
                                actual = engine.decode(z, 8, 8, local=local, checkpoint_blocks=learning)
                                expected = reference(other, 8, 8, local=local, checkpoint_blocks=learning)
                                torch.testing.assert_close(actual, expected)
                                if learning:
                                    actual.square().mean().backward()
                                    expected.square().mean().backward()
                                    torch.testing.assert_close(z.grad, other.grad)
                                    for p, q in zip(model.decoder.parameters(), reference.parameters()):
                                        self.assertEqual(p.grad is None, q.grad is None)
                                        if p.grad is not None:
                                            torch.testing.assert_close(p.grad, q.grad)
            if count is None:
                count = len(graphs)
                self.assertGreater(count, 0)
            else:
                self.assertEqual(count, len(graphs))
        self.assertEqual(torch._dynamo.config.cache_size_limit, 8)

    def test_learning_and_planning_reuse_bounded_caches_and_preserve_gradients(self):
        # Exercise Dynamo's actual guard/cache machinery on CPU. Returning the
        # captured FX graph avoids expensive Inductor/CUDA compilation in CI;
        # fullgraph remains required, including checkpointed transformer blocks.
        torch.compiler.reset()
        self.addCleanup(torch.compiler.reset)
        torch.set_num_threads(2)
        config = replace(NetworkConfig.small(), latent=32, pixel_latent=True,
                         local_dynamics=True, broadcast_conditioning=True)
        model = WorldModel(config).dynamics
        reference = copy.deepcopy(model)
        graphs = []

        def backend(graph, inputs):
            graphs.append(graph)
            return graph.forward

        compiler = torch.compile
        with patch('torch.compile', side_effect=lambda f, **kw: compiler(f, backend=backend, **kw)):
            predict = _CompiledDynamics(model)
        self.assertEqual(torch._dynamo.config.cache_size_limit, 8)
        cases = [(True, joint, batch) for joint in (True, False) for batch in (1, 4, 16)]
        cases += [(False, False, batch) for batch in (1, 4, 16, 64)]
        count = None
        for repeat in range(2):
            for learning, joint, batch in cases:
                model.train(learning)
                reference.train(learning)
                # Freeze all world weights only in deployment, while a frozen
                # representation leaves dynamics trainable during learning.
                model.requires_grad_(learning)
                reference.requires_grad_(learning)
                model.zero_grad(set_to_none=True)
                reference.zero_grad(set_to_none=True)
                z = torch.randn(batch, 4, 32)
                h = torch.randn(batch, 32)
                action = torch.randn(batch, 12)
                if repeat:
                    # Same values/shapes, different strides, as with expanded
                    # MPC states and slices from replay/decoder minibatches.
                    z = z.transpose(1, 2).contiguous().transpose(1, 2)
                    h = h.T.contiguous().T
                    action = action.T.contiguous().T
                z.requires_grad_(learning and joint)
                h.requires_grad_(learning)
                real_z = z.detach().clone().requires_grad_(learning and joint)
                real_h = h.detach().clone().requires_grad_(learning)
                with torch.set_grad_enabled(learning):
                    next_z, next_h, _ = predict(z, h, action, 8, 8)
                    actual = predict(next_z, next_h, action, 8, 8)
                    next_z, next_h, _ = reference(real_z, real_h, action, 8, 8)
                    expected = reference(next_z, next_h, action, 8, 8)
                    for observed, wanted in zip(actual, expected):
                        torch.testing.assert_close(observed, wanted)
                    if learning:
                        sum(value.square().mean() for value in actual).backward()
                        sum(value.square().mean() for value in expected).backward()
                        torch.testing.assert_close(h.grad, real_h.grad)
                        if joint:
                            torch.testing.assert_close(z.grad, real_z.grad)
                        for p, expected_p in zip(model.parameters(), reference.parameters()):
                            self.assertEqual(p.grad is None, expected_p.grad is None)
                            if p.grad is not None:
                                torch.testing.assert_close(p.grad, expected_p.grad)
            if count is None:
                count = len(graphs)
                self.assertGreater(count, 0)
            else:
                self.assertEqual(len(graphs), count, 'Repeated phases/layouts must reuse compilation')


if __name__ == '__main__':
    unittest.main()
