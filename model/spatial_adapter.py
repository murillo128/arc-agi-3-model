"""Optional action-localized residual with shared quarter-turn spatial filters.

Orientation channels remain separate. Learned mixing, directional actions and
pre-existing latents do not imply equivariance of the complete world model.
"""
import torch
from torch import nn
from torch.nn import functional as F


def click_maps(action, height, width, padded_height, padded_width):
    """Native SDK coordinates, with zero maps for absent/out-of-frame clicks."""
    # Recover integer SDK coordinates before bounds checks: float32 inversion
    # can otherwise turn x == width into a seemingly valid x just below width.
    x = ((action[:, 8:9] + 1) * 31.5).round()
    y = ((action[:, 9:10] + 1) * 31.5).round()
    valid = ((action[:, 6:7] > 0) & (action[:, 10:11] > 0) & (action[:, 11:12] > 0)
             & (x >= 0) & (x < min(width, 64)) & (y >= 0) & (y < min(height, 64)))
    xx = torch.arange(padded_width, device=action.device, dtype=action.dtype)[None, None, :]
    yy = torch.arange(padded_height, device=action.device, dtype=action.dtype)[None, :, None]
    dx = (xx - x[:, :, None]).expand(-1, padded_height, -1)
    dy = (yy - y[:, :, None]).expand(-1, -1, padded_width)
    heat = torch.exp(-(dx.square() + dy.square()) / 4.5)
    mask = valid[:, :, None] & (xx < width) & (yy < height)
    return torch.stack((heat, dx / 63, dy / 63), 1) * mask[:, None]


class RotationAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch = config.patch
        channels, width = config.latent // config.patch ** 2, config.rotation_adapter_width
        self.filters = nn.Conv2d(channels, width, 5, padding=2, bias=False)
        self.click = nn.Conv2d(3, 4 * width, 1, bias=False)
        self.condition = nn.Linear(config.memory + config.action_embedding, 4 * width)
        self.spatial = nn.Conv2d(4 * width, 4 * width, 3, padding=4, dilation=4, groups=4 * width)
        self.mix = nn.Conv2d(4 * width, 4 * width, 1)
        self.output = nn.Conv2d(4 * width, channels, 1)
        # WorldModel initializes the inner layers, then zeros this projection again.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def oriented_features(self, grid):
        kernels = torch.cat([torch.rot90(self.filters.weight, k, (-2, -1)) for k in range(4)])
        return F.conv2d(grid, kernels, padding=2)

    def forward(self, z, memory, embedded_action, action, height, width):
        p, batch = self.patch, z.shape[0]
        rows, columns = (height + p - 1) // p, (width + p - 1) // p
        grid = z.reshape(batch, rows, columns, p, p, -1).permute(0, 5, 1, 3, 2, 4)
        grid = grid.reshape(batch, -1, rows * p, columns * p)
        maps = click_maps(action, height, width, rows * p, columns * p)
        condition = self.condition(torch.cat((memory, embedded_action), -1))[:, :, None, None]
        x = F.silu(self.oriented_features(grid) + self.click(maps) + condition)
        x = F.silu(self.mix(x + self.spatial(x)))
        delta = self.output(x).reshape(batch, -1, rows, p, columns, p)
        return delta.permute(0, 2, 4, 3, 5, 1).reshape_as(z).float()
