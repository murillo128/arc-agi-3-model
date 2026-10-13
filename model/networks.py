"""Shared visual encoder, recurrent residual simulator, and visual decoder."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .config import NetworkConfig
from .data import frame_features


def positions(rows, columns, width, device, dtype):
    frequencies = torch.exp(-math.log(10000) * torch.arange(width // 4, device=device).float() / (width // 4))
    row, col = torch.meshgrid(torch.arange(rows, device=device), torch.arange(columns, device=device), indexing='ij')
    r, c = row.flatten()[:, None] * frequencies, col.flatten()[:, None] * frequencies
    return torch.cat((r.sin(), r.cos(), c.sin(), c.cos()), -1).to(dtype)


class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.q, self.k, self.v, self.out = [nn.Linear(width, width) for _ in range(4)]

    def forward(self, x, mask):
        b, t, d = x.shape
        q, k, v = [projection(x).reshape(b, t, self.heads, d // self.heads).transpose(1, 2)
                   for projection in (self.q, self.k, self.v)]
        result = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None, None, :], dropout_p=0.)
        result = result.transpose(1, 2).reshape(b, t, d)
        return self.out(result) * mask[:, :, None]


class Block(nn.Module):
    def __init__(self, width, heads, scale=1.):
        super().__init__()
        self.scale = scale
        self.attention_norm = nn.LayerNorm(width, eps=1e-6)
        self.attention = Attention(width, heads)
        self.mlp_norm = nn.LayerNorm(width, eps=1e-6)
        self.up, self.down = nn.Linear(width, 4 * width), nn.Linear(4 * width, width)

    def forward(self, x, mask):
        x = x + self.scale * self.attention(self.attention_norm(x), mask)
        return x + self.scale * self.down(F.gelu(self.up(self.mlp_norm(x))))


class RecurrentCell(nn.Module):
    def __init__(self, inputs, hidden):
        super().__init__()
        self.input = nn.Linear(inputs + hidden, hidden)
        self.norm = nn.LayerNorm(hidden, eps=1e-6)
        self.gates = nn.Linear(hidden, 3 * hidden)

    def forward(self, x, h):
        r, candidate, u = self.gates(F.gelu(self.norm(self.input(torch.cat((x, h), -1))))).chunk(3, -1)
        candidate = torch.tanh(torch.sigmoid(r) * candidate)
        update = torch.sigmoid(u - 1)
        return (1 - update) * h + update * candidate


class VisualEncoder(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.patch = nn.Linear(c.patch * c.patch * 35, c.width)
        if c.pixel_latent:
            self.pixel_embedding = nn.Linear(35, c.latent // c.patch ** 2)
        self.blocks = nn.ModuleList(Block(c.width, c.heads, c.residual_scale) for _ in range(c.encoder_depth))
        self.spatial_norm = nn.LayerNorm(c.width, eps=1e-6)
        self.temporal = RecurrentCell(c.width, c.animation_memory)
        self.latent = nn.Linear(c.animation_memory, c.latent)
        self.latent_norm = nn.LayerNorm(c.latent, eps=1e-6)

    def spatial(self, features, checkpoint_blocks=False):
        c = self.config
        b, _, h, w = features.shape
        p = c.patch
        # Flatten (local row, local column, channel), matching the architecture.
        patches = features.reshape(b, 35, h // p, p, w // p, p).permute(0, 2, 4, 3, 5, 1).reshape(b, -1, p * p * 35)
        valid = F.avg_pool2d(features[:, -1:], p).flatten(1) > 0
        tokens = self.patch(patches)
        tokens = tokens + c.position_scale * positions(h // p, w // p, c.width, tokens.device, tokens.dtype)
        for block in self.blocks:
            tokens = checkpoint(block, tokens, valid, use_reentrant=False) if checkpoint_blocks and self.training else block(tokens, valid)
        # Retain every patch and its position. No spatial averaging before z.
        context = self.spatial_norm(tokens) * valid[:, :, None]
        if c.pixel_latent:
            local = self.pixel_embedding(patches.reshape(b, -1, p * p, 35)).flatten(-2)
            return torch.cat((context, local), -1)
        return context

    def aggregate(self, frames):
        # Leading axes are independent observations/patches, never time. Only
        # the first axis advances the animation recurrence.
        c = self.config
        local = frames[-1, ..., c.width:] if c.pixel_latent else None
        frames = frames[..., :c.width]
        h = frames.new_zeros((*frames.shape[1:-1], c.animation_memory))
        for frame in frames:
            h = self.temporal(frame, h)
        z = self.latent(h)
        if c.pixel_latent:
            z = local + c.residual_scale * z
        return self.latent_norm(z).float()


class SpatialBlock(nn.Module):
    def __init__(self, width, scale):
        super().__init__()
        self.scale = scale
        self.norm1, self.norm2 = [nn.GroupNorm(math.gcd(8, width), width, eps=1e-6) for _ in range(2)]
        self.conv1, self.conv2 = [nn.Conv2d(width, width, 3, padding=1) for _ in range(2)]

    def forward(self, x):
        y = self.conv1(F.silu(self.norm1(x)))
        return x + self.scale * self.conv2(F.silu(self.norm2(y)))


class SpatialDynamics(nn.Module):
    """Generic local latent dynamics, with learned action/history conditioning."""
    def __init__(self, c):
        super().__init__()
        self.config = c
        channels, width = c.latent // c.patch ** 2, min(128, c.width)
        self.input = nn.Conv2d(channels, width, 3, padding=1)
        self.condition = nn.Linear(c.memory + c.action_embedding, width)
        self.blocks = nn.ModuleList(SpatialBlock(width, c.residual_scale) for _ in range(3))
        self.output = nn.Conv2d(width, channels, 1)

    def forward(self, z, h, action_embedding, height, width):
        c, b = self.config, z.shape[0]
        p = c.patch
        rows, columns = (height + p - 1) // p, (width + p - 1) // p
        grid = z.reshape(b, rows, columns, p, p, -1).permute(0, 5, 1, 3, 2, 4)
        grid = grid.reshape(b, -1, rows * p, columns * p)
        x = self.input(grid) + self.condition(torch.cat((h, action_embedding), -1))[:, :, None, None]
        for block in self.blocks:
            x = checkpoint(block, x, use_reentrant=False) if self.training else block(x)
        delta = self.output(x).reshape(b, -1, rows, p, columns, p)
        return delta.permute(0, 2, 4, 3, 5, 1).reshape_as(z).float()


class Dynamics(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.action_embedding = nn.Linear(12, c.action_embedding)
        self.action_norm = nn.LayerNorm(c.action_embedding, eps=1e-6)
        self.cell = RecurrentCell(c.latent + c.action_embedding, c.memory)
        self.state_token = nn.Linear(c.latent, c.width)
        self.memory_token = nn.Linear(c.memory, c.width)
        self.action_token = nn.Linear(c.action_embedding, c.width)
        self.roles = nn.Parameter(torch.empty(3, c.width))
        self.blocks = nn.ModuleList(Block(c.width, c.heads, c.residual_scale) for _ in range(c.predictor_depth))
        self.norm = nn.LayerNorm(c.width, eps=1e-6)
        self.delta = nn.Linear(c.width, c.latent)
        if c.local_dynamics:
            self.local = SpatialDynamics(c)

    def advance(self, z, h, action):
        a = F.gelu(self.action_norm(self.action_embedding(action)))
        # Only the global recurrent context is pooled. The prediction/residual
        # path retains all spatial tokens unchanged.
        return self.cell(torch.cat((z.mean(1), a), -1), h), a

    def forward(self, z, h, action, height=None, width=None):
        h_next, a = self.advance(z, h, action)
        memory, action_token = self.memory_token(h_next), self.action_token(a)
        spatial = self.state_token(z) + self.roles[0]
        if self.config.local_dynamics:
            if height is None or width is None:
                raise ValueError('Spatial dynamics requires the native geometry of the current state')
            p = self.config.patch
            spatial = spatial + self.config.position_scale * positions(
                (height + p - 1) // p, (width + p - 1) // p,
                self.config.width, z.device, spatial.dtype)
        if self.config.broadcast_conditioning:
            spatial = spatial + memory[:, None] + action_token[:, None]
        x = torch.cat((spatial, (memory + self.roles[1])[:, None],
                       (action_token + self.roles[2])[:, None]), 1)
        mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        for block in self.blocks:
            x = checkpoint(block, x, mask, use_reentrant=False) if self.training else block(x, mask)
        delta = self.delta(self.norm(x[:, :z.shape[1]])).float()
        if self.config.local_dynamics:
            delta = self.local(z, h_next, a, height, width) + self.config.residual_scale * delta
        return z.float() + delta, h_next.float(), delta


class VisualDecoder(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.latent_projection = nn.Linear(c.latent, c.width)
        self.blocks = nn.ModuleList(Block(c.width, c.heads, c.residual_scale) for _ in range(c.decoder_depth))
        self.norm = nn.LayerNorm(c.width, eps=1e-6)
        if c.pixel_latent:
            self.context = nn.Linear(c.width, c.latent)
            channels = c.latent // c.patch ** 2
            self.color = nn.Linear(channels, 16)
        else:
            self.color = nn.Linear(c.width, c.patch * c.patch * 16)

    def forward(self, z, height, width, checkpoint_blocks=False, local=False):
        c, b = self.config, z.shape[0]
        p = c.patch
        r, col = (height + p - 1) // p, (width + p - 1) // p
        condition = self.latent_projection(z)
        tokens = condition
        # Auxiliary real reconstruction must read each patch from its own z.
        # Shared projection/norm/classifier; no new parameters or pixel bypass.
        if not local:
            tokens = tokens + c.position_scale * positions(r, col, c.width, z.device, condition.dtype)
            mask = torch.ones((b, r * col), dtype=torch.bool, device=z.device)
            for block in self.blocks:
                tokens = checkpoint(block, tokens, mask, use_reentrant=False) if checkpoint_blocks and self.training else block(tokens, mask)
        if c.pixel_latent:
            pixels = z if local else z + c.residual_scale * self.context(self.norm(tokens))
            colors = self.color(pixels.reshape(b, r, col, p, p, -1))
        else:
            colors = self.color(self.norm(tokens)).reshape(b, r, col, p, p, 16)
        return colors.permute(0, 1, 3, 2, 4, 5).reshape(b, r * p, col * p, 16)[:, :height, :width]


class WorldModel(nn.Module):
    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.config = config
        self.encoder = VisualEncoder(config)
        self.dynamics = Dynamics(config)
        self.decoder = VisualDecoder(config)
        self.apply(self._init)
        nn.init.trunc_normal_(self.dynamics.roles, std=.02, a=-.04, b=.04)
        nn.init.trunc_normal_(self.dynamics.delta.weight, std=.001, a=-.002, b=.002)
        if config.local_dynamics:
            nn.init.zeros_(self.dynamics.local.output.weight)
            nn.init.zeros_(self.dynamics.local.output.bias)

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, std=.02, a=-.04, b=.04)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def encode(self, observations, spatial, chunk=8, checkpoint_blocks=False, prepared=None, aggregate=None):
        device = next(self.parameters()).device
        features, owners, result = {}, {}, [None] * len(observations)
        # Bucket native geometries; the batch axis contains independent frames.
        for index, observation in enumerate(observations):
            if not len(observation.frames):
                continue
            x = prepared[index][1] if prepared else frame_features(observation, device, self.config.patch)
            key = tuple(x.shape[-2:])
            features.setdefault(key, []).append(x)
            owners.setdefault(key, []).append((index, len(x)))
        for key, chunks in features.items():
            x = torch.cat(chunks)
            # Independent storage keeps compile guards from specializing on the
            # changing size of the full prefix behind each split view.
            encoded = torch.cat([spatial(part.clone(), checkpoint_blocks) for part in x.split(chunk)])
            start, animations = 0, {}
            for index, count in owners[key]:
                animations.setdefault(count, []).append((index, encoded[start:start + count]))
                start += count
            # Batch independent animations of equal length. This removes one
            # Python/kernel launch sequence per observation without mixing histories.
            for parts in animations.values():
                zs = (aggregate or self.encoder.aggregate)(torch.stack([x for _, x in parts], dim=1))
                for (index, _), z in zip(parts, zs.unbind()):
                    result[index] = z
        return result


def sigreg(z, slices=1024):
    """ECF over real observations at matching positions, never mixing positions.

    Uniformly sample at most 16 spatial positions to bound sketch memory. This
    prevents positional codes alone from masquerading as observation diversity.
    The 2-D vector case is useful for the independent numerical oracle.
    """
    z = z.float()
    if z.ndim == 2:
        z = z[:, None, :]
    selected = torch.randperm(z.shape[1], device=z.device)[:16]
    z = z[:, selected]
    directions = F.normalize(torch.randn(z.shape[-1], slices, device=z.device), dim=0)
    t = torch.linspace(0, 3, 17, device=z.device)
    projected = (z @ directions)[..., None] * t
    discrepancy = (projected.cos().mean(0) - torch.exp(-t.square() / 2)).square() + projected.sin().mean(0).square()
    # Integrand is even: the positive half represents both [-3, 0] and [0, 3].
    return (2 * torch.trapezoid(discrepancy * torch.exp(-t.square() / 2), t, dim=-1) * z.shape[0]).mean()


def variance_floor(z):
    """VICReg-style variance hinge across observations, separately per position.

    Supplies a stronger restoring gradient near small but nonzero variance than
    the ECF discrepancy alone. It does not whiten or rescale reported latents.
    """
    return F.relu(1 - torch.sqrt(z.float().var(0, correction=0) + 1e-8)).mean()
