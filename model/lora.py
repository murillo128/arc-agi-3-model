"""Private low-rank parameters routed through one immutable shared base.

Forward hooks preserve the original modules/state_dict. The router lock spans
forward, backward (including checkpoint recomputation), and optimizer updates.
Only active adapter tensors and Adam state occupy the accelerator; all banks are
stored on CPU outside an activation. Callers must consume graphs inside it.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import math
import threading

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Target:
    name: str
    group: str
    module: nn.Module


class AdapterBank:
    def __init__(self, router, rank, alpha, seed, learning_rate):
        if type(rank) is not int or not 1 <= rank <= 64:
            raise ValueError('LoRA rank must be an integer in 1..64')
        if type(seed) is not int:
            raise ValueError('LoRA seed must be an integer')
        if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
               for v in (alpha, learning_rate)):
            raise ValueError('LoRA alpha and learning rate must be finite and positive')
        self.router, self.rank, self.scale = router, rank, alpha / rank
        self.tensors, self.groups = {}, {group: [] for group in router.modules}
        generator = torch.Generator(device='cpu').manual_seed(seed)
        for target in router.targets:
            module = target.module
            shape = ((rank, module.in_features) if isinstance(module, nn.Linear)
                     else (rank, module.in_channels, *module.kernel_size))
            a = torch.empty(shape, dtype=torch.float32)
            nn.init.kaiming_uniform_(a, a=math.sqrt(5), generator=generator)
            b = torch.zeros((module.weight.shape[0], rank) if a.ndim == 2
                            else (module.out_channels, rank, 1, 1))
            pair = (nn.Parameter(a, requires_grad=False), nn.Parameter(b, requires_grad=False))
            self.tensors[target.name] = pair
            self.groups[target.group].extend(pair)
        self.optimizers = {group: torch.optim.Adam(parameters, lr=learning_rate)
                           for group, parameters in self.groups.items() if parameters}

    def parameters(self, groups=None):
        names = tuple(self.groups) if groups is None else tuple(groups)
        if set(names) - self.groups.keys():
            raise ValueError('Unknown adapter group')
        return [parameter for group in names for parameter in self.groups[group]]

    @property
    def bytes(self):
        return sum(p.numel() * p.element_size() for p in self.parameters())

    def _move(self, device):
        for parameter in self.parameters():
            parameter.grad = None
            parameter.data = parameter.data.to(device)
        for optimizer in self.optimizers.values():
            for state in optimizer.state.values():
                for key, value in state.items():
                    # Adam's non-capturable step is intentionally a CPU scalar.
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to('cpu' if key == 'step' else device)

    def _trainable(self, groups):
        for group, parameters in self.groups.items():
            for parameter in parameters:
                parameter.requires_grad_(group in groups)


class AdapterRouter:
    def __init__(self, modules, device=None):
        if not modules or any(not isinstance(module, nn.Module) for module in modules.values()):
            raise ValueError('Adapter router requires named torch modules')
        self.modules = dict(modules)
        self.device = torch.device(device or next(next(iter(modules.values())).parameters()).device)
        self.lock = threading.RLock()
        self.active = None
        self.active_groups = frozenset()
        self.targets, self.excluded, self._handles = [], [], []
        seen = set()
        for group, root in self.modules.items():
            root.requires_grad_(False)
            for name, module in root.named_modules():
                if id(module) in seen:
                    raise ValueError('Adapter module groups must not overlap')
                seen.add(id(module))
                qualified = f'{group}.{name}' if name else group
                if isinstance(module, nn.Conv2d) and module.groups != 1:
                    self.excluded.append(dict(name=qualified, reason='grouped/depthwise Conv2d', groups=module.groups))
                elif isinstance(module, (nn.Linear, nn.Conv2d)):
                    target = Target(qualified, group, module)
                    self.targets.append(target)
                    self._handles.append(module.register_forward_hook(self._hook(target)))
        self.report = dict(version=1, supported=['Linear', 'Conv2d(groups=1)'],
                           targets=[dict(name=t.name, group=t.group, kind=type(t.module).__name__)
                                    for t in self.targets], excluded=self.excluded)

    def _hook(self, target):
        def apply(module, inputs, output):
            if self.active is None:
                return output
            a, b = self.active.tensors[target.name]
            x = inputs[0]
            if isinstance(module, nn.Linear):
                delta = F.linear(F.linear(x, a), b)
            else:
                padding = module.padding
                if module.padding_mode != 'zeros':
                    x = F.pad(x, module._reversed_padding_repeated_twice, mode=module.padding_mode)
                    padding = 0
                delta = F.conv2d(F.conv2d(x, a, stride=module.stride, padding=padding,
                                         dilation=module.dilation), b)
            return output + delta * self.active.scale
        return apply

    def new_bank(self, *, rank=4, alpha=4, seed=0, learning_rate=1e-4):
        return AdapterBank(self, rank, alpha, seed, learning_rate)

    @contextmanager
    def activate(self, bank, groups=()):
        if bank.router is not self:
            raise ValueError('Adapter bank belongs to another shared base')
        selected = frozenset(groups)
        if selected - self.modules.keys():
            raise ValueError('Unknown adapter group')
        with self.lock:
            previous, previous_groups = self.active, self.active_groups
            if previous is not None and previous is not bank:
                previous._trainable(())
                previous._move('cpu')
            if previous is not bank:
                bank._move(self.device)
            bank._trainable(selected)
            self.active, self.active_groups = bank, selected
            try:
                yield bank
            finally:
                bank._trainable(())
                if previous is not bank:
                    bank._move('cpu')
                if previous is not None:
                    if previous is not bank:
                        previous._move(self.device)
                    previous._trainable(previous_groups)
                self.active, self.active_groups = previous, previous_groups
