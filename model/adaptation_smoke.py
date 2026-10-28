"""Bounded adaptation probes; fixture self-check and public SDK smoke are distinct.

The packaged self-check has no game dependency. It reports engineering evidence
on explicitly simulated palette frames, never a game score. The local smoke only
uses public SDK observations from the two training-authorized synthetic games at
native level 1. Neither path writes replay or modifies the mounted base file.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from .data import Observation


FORMAT = 'arc3-adaptation-smoke'
VERSION = 1
TRAINING_GAMES = ('sc01-v1', 'si01-v1')


def _tensor_digest(named_tensors):
    """Exact content digest, transferring at most one tensor to CPU at a time."""
    digest = hashlib.sha256()
    size = 0
    for name, tensor in named_tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode('utf-8'))
        digest.update(str((value.dtype, tuple(value.shape))).encode('ascii'))
        # The uint8 view supports BF16 as well as NumPy-supported float types.
        raw = value.reshape(-1).view(torch.uint8).numpy()
        digest.update(memoryview(raw))
        size += tensor.numel() * tensor.element_size()
    return digest.hexdigest(), size


def base_digest(runtime):
    def tensors():
        for group, module in (('world', runtime.engine.model), ('goal', runtime.goal),
                              ('path', runtime.path), ('change', getattr(runtime, 'change', None))):
            if module is not None:
                for name, value in module.state_dict().items():
                    yield f'{group}.{name}', value
    return _tensor_digest(tensors())


def bank_digest(bank):
    return _tensor_digest((str(i), value) for i, value in enumerate(bank.parameters()))[0]


def _fixture():
    # A tiny palette fixture, not an SDK response or a training game recording.
    first = np.zeros((8, 8), dtype=np.uint8)
    first[1:3, 1:3] = 3
    first[5:7, 5:7] = 9
    second = first.copy()
    second[1:3, 1:3], second[1:3, 3:5] = 0, 3
    third = second.copy()
    third[5:7, 5:7], third[3:5, 5:7] = 0, 9
    return [Observation(frame[None], (1, 2, 3, 4), 'NOT_FINISHED', 0, 5)
            for frame in (first, second, third)]


def _probe(policy, observation):
    runtime = policy.runtime
    with runtime.router.activate(policy.bank), torch.no_grad(), runtime.engine.autocast():
        runtime.engine.model.eval()
        return runtime.engine.encode([observation])[0].detach().cpu().clone()


def _finite(value):
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_finite(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    return True


def _config(config):
    from .adaptation import AdaptationConfig

    if config is None:
        return AdaptationConfig()
    if isinstance(config, dict):
        return AdaptationConfig(**config)
    if not isinstance(config, AdaptationConfig):
        raise TypeError('config must be an AdaptationConfig or configuration dictionary')
    return config


def _report(kind, started, config, policies, before, after, isolation, observations):
    rows = []
    for game_id, policy, extra in policies:
        rows.append(dict(game=game_id, adapter_bytes=policy.bank.bytes,
                         diagnostics=policy.diagnostics, **extra))
    updates = sum(sum(row['diagnostics']['updates'].values()) for row in rows)
    finite = _finite(rows)
    result = dict(format=FORMAT, version=VERSION, kind=kind,
                  purpose='engineering smoke; no competition score or final Phase 2 benchmark',
                  elapsed_seconds=time.perf_counter() - started,
                  config=asdict(config), base_bytes=before[1],
                  base_sha256_before=before[0], base_sha256_after=after[0],
                  base_immutable=before == after, isolation=isolation,
                  confirmed_observations=observations, finite_diagnostics=finite,
                  optimizer_steps=updates, games=rows)
    result['passed'] = bool(result['base_immutable'] and all(isolation.values())
                            and finite and updates > 0 and all(
                                row['diagnostics']['numerical_failures'] == 0 for row in rows))
    return result


def self_check(checkpoint, *, device='cpu', config=None):
    """Exercise the mounted model and optimizer on tiny simulated observations.

    Confirmation calls below emulate the SDK boundary for an engineering test.
    They do not establish that a real environment transition occurred. The JSON
    records zero real SDK transitions and identifies the fixture explicitly.
    """
    from .adaptation import load

    started, config = time.perf_counter(), _config(config)
    active = load(checkpoint, device=device, config=config, max_actions=2, seed=145)
    inactive = load(checkpoint, device=device, config=config, max_actions=2, seed=146)
    fixture = _fixture()
    before = base_digest(active.runtime)
    active_before, inactive_before = bank_digest(active.bank), bank_digest(inactive.bank)
    probe_before = _probe(inactive, fixture[0])
    for prior, result in zip(fixture, fixture[1:]):
        action = active.choose_action(prior)
        active.confirm_result(result, action=action)
    after = base_digest(active.runtime)
    isolation = dict(shared_runtime=active.runtime is inactive.runtime,
                     inactive_bank_unchanged=inactive_before == bank_digest(inactive.bank),
                     inactive_inference_unchanged=torch.equal(probe_before, _probe(inactive, fixture[0])),
                     active_bank_updated=active_before != bank_digest(active.bank))
    report = _report('offline-fixture-self-check', started, config,
                     [('simulated-palette-fixture', active, dict(real_sdk_transitions=0,
                         simulated_confirmations=2))], before, after, isolation, 3)
    report['real_sdk_transitions'] = 0
    report['targets'] = active.runtime.router.report
    return report


def _training_level_one(observation):
    # Stop BEFORE passing a newly entered scene to the policy. The public level
    # counter is sufficient; no private game implementation/teacher is queried.
    return observation.levels_completed == 0 and observation.win_levels >= 2


def local_smoke(checkpoint, *, device='cpu', config=None, environments_dir='games',
                actions=4, seed=0):
    """Run public native level-1 SDK observations with no oracle or recorder."""
    from arc3.envs.sdk import SDKEnvironment, open_arcade
    from .adaptation import load

    if type(actions) is not int or not 1 <= actions <= 32:
        raise ValueError('Smoke action budget must be an integer in 1..32 per game')
    started, config = time.perf_counter(), _config(config)
    arcade = open_arcade('offline', str(environments_dir))
    sentinel = load(checkpoint, device=device, config=config, max_actions=actions, seed=seed + 100)
    before = base_digest(sentinel.runtime)
    inactive_before = bank_digest(sentinel.bank)
    probe = _fixture()[0]
    probe_before = _probe(sentinel, probe)
    rows, observations, real_transitions = [], 0, 0
    for index, game_id in enumerate(TRAINING_GAMES):
        environment = SDKEnvironment(arcade, game_id, seed)
        initial = environment.observation
        if not _training_level_one(initial):
            raise ValueError(f'{game_id}: public initialization must be training level 1 of at least two levels')
        policy = load(checkpoint, device=device, config=config, max_actions=actions,
                      seed=seed + index)
        raw_before = bank_digest(policy.bank)
        current = initial
        committed, executed, excluded, terminal = 0, 0, 0, None
        observations += 1
        for _ in range(actions):
            if current.state in {'WIN', 'GAME_OVER'}:
                terminal = current.state
                break
            action = policy.choose_action(current)
            try:
                result = environment.step(action)
            except BaseException:
                policy.cancel_pending()
                raise
            executed += 1
            if not _training_level_one(result):
                policy.cancel_pending()
                excluded += 1
                break
            policy.confirm_result(result, action=action)
            committed += 1
            observations += 1
            current = result
            if current.state in {'WIN', 'GAME_OVER'}:
                terminal = current.state
                break
        real_transitions += committed
        rows.append((game_id, policy, dict(real_sdk_transitions=committed,
                    executed_sdk_actions=executed, excluded_level_boundary_results=excluded,
                    native_terminal=terminal, active_bank_updated=raw_before != bank_digest(policy.bank))))
    after = base_digest(sentinel.runtime)
    isolation = dict(shared_runtime=all(policy.runtime is sentinel.runtime for _, policy, _ in rows),
                     inactive_bank_unchanged=inactive_before == bank_digest(sentinel.bank),
                     inactive_inference_unchanged=torch.equal(probe_before, _probe(sentinel, probe)))
    report = _report('public-sdk-level1-smoke', started, config, rows,
                     before, after, isolation, observations)
    report['real_sdk_transitions'] = real_transitions
    report['training_boundary'] = 'Only sc01-v1/si01-v1 native level 1; every level-crossing result excluded before adaptation'
    report['targets'] = sentinel.runtime.router.report
    report['passed'] &= real_transitions > 0 and all(row[2]['active_bank_updated'] for row in rows)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--actions', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--environments-dir', default='games')
    parser.add_argument('--config-json', default='{}')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.threads <= 16:
        parser.error('--threads must be in 1..16')
    if Path(args.output).resolve() == Path(args.checkpoint).resolve():
        parser.error('Smoke output must differ from the read-only checkpoint')
    try:
        config = json.loads(args.config_json)
        if not isinstance(config, dict):
            raise ValueError('--config-json must contain a JSON object')
        config = _config(config)
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    torch.set_num_threads(args.threads)
    if args.self_check:
        report = self_check(args.checkpoint, device=args.device, config=config)
    else:
        report = local_smoke(args.checkpoint, device=args.device, config=config,
                             environments_dir=args.environments_dir,
                             actions=args.actions, seed=args.seed)
    report['threads'] = args.threads
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'], output=str(output),
                         kind=report['kind'], elapsed_seconds=report['elapsed_seconds'])))
    if not report['passed']:
        raise SystemExit('Adaptation smoke failed; inspect the JSON evidence')


if __name__ == '__main__':
    main()
