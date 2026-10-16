"""Incremental learners on real actions, with explicit gradient ownership."""
import numpy as np
import torch
from torch.nn import functional as F

from .learning import optimizer, visual_delta_loss, soft_delta_error
from .networks import sigreg, variance_floor


def trainable(engine, *names):
    for name in ('encoder', 'decoder', 'dynamics'):
        module = getattr(engine.model, name)
        module.requires_grad_(name in names).train(name in names)
        # Do not leave gradients owned by a different optimizer around.
        if name not in names:
            module.zero_grad(set_to_none=True)


class OnlineRepresentation:
    def __init__(self, engine, learning_rate=3e-6, *, freeze_encoder=False,
                 encoder_learning_rate_after_predictor=None):
        if encoder_learning_rate_after_predictor is not None and not 0 < encoder_learning_rate_after_predictor <= learning_rate:
            raise ValueError('Post-predictor encoder LR must be positive and no greater than representation LR')
        self.engine = engine
        self.freeze_encoder = freeze_encoder
        self.encoder_frozen_by_option = freeze_encoder
        self.base_learning_rate = learning_rate
        self.frozen = False
        self.encoder_learning_rate_after_predictor = encoder_learning_rate_after_predictor
        groups = [dict(name='decoder', params=list(engine.model.decoder.parameters()))]
        if not freeze_encoder:
            groups.insert(0, dict(name='encoder', params=list(engine.model.encoder.parameters())))
        else:
            engine.model.encoder.requires_grad_(False).eval()
            engine.model.encoder.zero_grad(set_to_none=True)
        self.parameters = [p for group in groups for p in group['params']]
        self.optimizer = optimizer(groups, learning_rate,
                                   engine.device.type == 'cuda', engine.optimizer_bits)
        self.updates = 0
        self.sigreg_enabled = bool(engine.config.sigreg_weight) and not freeze_encoder

    @property
    def learning_rates(self):
        if self.frozen:
            return dict(encoder=None, decoder=None)
        return {'encoder': None, **{g['name']: g['lr'] for g in self.optimizer.param_groups}}

    def freeze(self):
        """Stop representation learning without changing weights or cached z."""
        self.frozen = self.freeze_encoder = True
        self.sigreg_enabled = False
        for module in (self.engine.model.encoder, self.engine.model.decoder):
            module.requires_grad_(False).eval()
            module.zero_grad(set_to_none=True)

    def activate_predictor(self):
        """Latch the shared representation phase without rebuilding optimizer state."""
        self.sigreg_enabled = False
        if self.encoder_learning_rate_after_predictor is not None:
            for group in self.optimizer.param_groups:
                if group['name'] == 'encoder':
                    group['lr'] = self.encoder_learning_rate_after_predictor

    def select_level(self, *, predictor, planner):
        """Select a level's phase, retaining weights and optimizer moments."""
        if planner:
            self.freeze()
            return
        self.frozen = False
        self.freeze_encoder = self.encoder_frozen_by_option
        for group in self.optimizer.param_groups:
            group['lr'] = self.base_learning_rate
        if predictor:
            self.activate_predictor()
        # SIGReg stays disabled once any predictor has been admitted. update()
        # sets gradient ownership when this level actually trains again.

    def update(self, observations, *, preview=None):
        """Reconstruct the last two real observations on EVERY learning action.

        Targets include unchanged pixels. The auxiliary local readout must also
        reconstruct from z without the decoder's contextual correction. No
        next-frame prediction or imagined target enters this optimizer.
        """
        if self.frozen:
            return None
        observations = [o for o in observations[-2:] if o.final is not None]
        if not observations or len({o.final.shape for o in observations}) != 1:
            return None
        e, c = self.engine, self.engine.config
        trainable(e, *(['decoder'] if self.freeze_encoder else ['encoder', 'decoder']))
        self.optimizer.zero_grad(set_to_none=True)
        with e.autocast():
            with torch.set_grad_enabled(not self.freeze_encoder):
                zs = torch.stack(e.encode(observations, training=not self.freeze_encoder))
            logits = e.decode(zs, *observations[-1].final.shape, checkpoint_blocks=True)
        if preview is not None:
            # Reuse the pre-update reconstruction, without another model pass or
            # retaining its graph/tensors in the dashboard's sample buffer.
            preview(logits[0].detach().argmax(-1).to(torch.uint8).cpu().numpy())
        actual = torch.tensor(np.stack([o.final for o in observations]), device=e.device, dtype=torch.long)
        # The initial frame uses an absolute CE target, also on stationary games.
        ce = F.cross_entropy(logits.float().movedim(-1, 1), actual)
        delta = visual_delta_loss(logits[-1:], actual[:1], actual[-1:]).mean()
        overlap = soft_delta_error(logits[-1:], actual[:1], actual[-1:]).mean()
        terms = dict(color_ce=ce, delta_nll=delta, delta_overlap=c.delta_overlap_weight * overlap)
        if c.local_reconstruction_weight and c.decoder_readout != 'local':
            with e.autocast():
                local_logits = e.decode(zs, *observations[-1].final.shape, local=True)
            local_loss = (F.cross_entropy(local_logits.float().movedim(-1, 1), actual)
                + visual_delta_loss(local_logits[-1:], actual[:1], actual[-1:]).mean()
                + c.delta_overlap_weight * soft_delta_error(local_logits[-1:], actual[:1], actual[-1:]).mean())
            terms['local_reconstruction'] = c.local_reconstruction_weight * local_loss
        # The curriculum disables SIGReg globally after the first predictor
        # admission. New levels must not restart it on the shared encoder.
        # Disabled regularizers do no work and consume no random draws.
        if self.sigreg_enabled and c.sigreg_weight and len(zs) >= c.min_observations:
            terms['sigreg'] = c.sigreg_weight * sigreg(zs, e.model.config.sigreg_slices)
        if not self.freeze_encoder and c.variance_weight and len(zs) >= c.min_observations:
            terms['variance_floor'] = c.variance_weight * variance_floor(zs)
        loss = sum(terms.values())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters, 1., error_if_nonfinite=True)
        self.optimizer.step()
        self.updates += 1
        if not self.freeze_encoder:
            e.invalidate_latent_cache()
        trainable(e)  # Detached acting between optimizer steps.
        return dict(total=float(loss.detach()), reconstruction=float(ce.detach()),
                    terms={name: float(value.detach()) for name, value in terms.items()})


class OnlinePredictor:
    def __init__(self, engine, threshold=.10, learning_rate=3e-5, horizon=3):
        self.engine, self.threshold, self.horizon = engine, threshold, horizon
        self.optimizer = optimizer(engine.model.dynamics.parameters(), learning_rate,
                                   engine.device.type == 'cuda', engine.optimizer_bits)
        self.updates = 0

    def update(self, state, action, reference, after, target_z, error, *, rollout=None):
        # The legacy error trigger remains available to callers. The cumulative
        # trainer uses threshold=None: once admitted, learn on every real action.
        if (error is None or (self.threshold is not None and error <= self.threshold)
                or after.final is None or target_z is None):
            return None
        return self.update_batch([dict(state=state, action=action, reference=reference,
                                       after=after, target_z=target_z, rollout=rollout)])

    def update_batch(self, samples):
        """Mean per-transition objective, one clipped optimizer step per batch.

        Backward each example separately to bound activation memory. All examples
        see the same parameters; missing rollout supervision contributes zero to
        the batch mean, and is omitted from logging when absent for the whole batch.
        """
        if not samples:
            return None
        e = self.engine
        trainable(e, 'dynamics')
        self.optimizer.zero_grad(set_to_none=True)
        logged = []
        for sample in samples:
            terms = self._loss(**sample)
            (sum(terms.values()) / len(samples)).backward()
            logged.extend((name, value.detach()) for name, value in terms.items())
        torch.nn.utils.clip_grad_norm_(e.model.dynamics.parameters(), 1., error_if_nonfinite=True)
        self.optimizer.step()
        self.updates += 1
        trainable(e)
        # Copy only detached scalars, once per batch instead of synchronizing
        # CUDA for every term of every example. Keep Python's original ordered
        # accumulation and missing-rollout denominator exactly as before.
        values = torch.stack([value for _, value in logged]).cpu().tolist()
        means = {}
        for (name, _), value in zip(logged, values):
            means[name] = means.get(name, 0.) + value / len(samples)
        return dict(total=sum(means.values()), latent=means['latent'], visual=means['delta_nll'],
                    rollout=means.get('rollout'), terms=means, batch_size=len(samples))

    def _loss(self, state, action, reference, after, target_z, *, rollout=None):
        e = self.engine
        before = torch.tensor(np.array(reference, copy=True), device=e.device, dtype=torch.long)[None]
        actual = torch.tensor(np.array(after.final, copy=True), device=e.device, dtype=torch.long)[None]
        with e.autocast():
            predicted, _, delta = e.predict(state.z[None].detach(), state.h.detach(), e.actions([action]), *state.geometry)
            logits = e.decode(predicted, *state.geometry, checkpoint_blocks=True)
        latent = F.mse_loss(delta.float(), (target_z - state.z).detach()[None].float())
        visual = visual_delta_loss(logits, before, actual, e.config.false_change_weight,
                                   mode=e.config.prediction_delta_loss).mean()
        # Predictive supervision is MSE + visual NLL. The shared overlap weight
        # still belongs to real reconstruction, not either predictive branch.
        terms = dict(latent=latent, delta_nll=visual)
        rollout_loss = None
        if rollout is not None:
            anchor, actions, reference = rollout
            z, h = anchor.z[None].detach(), anchor.h.detach()
            with e.autocast():
                for command in actions:
                    z, h, _ = e.predict(z, h, e.actions([command]), *anchor.geometry)
                logits = e.decode(z, *anchor.geometry, checkpoint_blocks=True)
            before = torch.tensor(np.array(reference, copy=True), device=e.device, dtype=torch.long)[None]
            rollout_loss = (F.mse_loss(z.float(), target_z.detach()[None].float())
                + visual_delta_loss(logits, before, actual, e.config.false_change_weight,
                                    mode=e.config.prediction_delta_loss).mean())
            terms['rollout'] = rollout_loss
        return terms
