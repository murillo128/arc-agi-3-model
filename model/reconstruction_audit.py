"""Frozen reconstruction on public held-out observations; never a training corpus.

Run capture, collect, then evaluate in a fresh ignored output directory. Only the
encoder/decoder enter CUDA. Native level access uses public SDK actions only.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
import hashlib
import json
import multiprocessing
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn

from arc3.envs.splits import load_split
from .config import NetworkConfig
from .data import Action, Observation
from .environment import Factory
from .networks import VisualEncoder, VisualDecoder, WorldModel
from .sequence_exploration import (ChangeHead, ChangeExplorer, ExplorationSession,
                                   SequenceExplorationConfig)


def weights_digest(weights):
    digest = hashlib.sha256()
    for name, value in sorted(weights.items()):
        value = value.detach().cpu().contiguous()
        digest.update(f'{name}:{value.dtype}:{tuple(value.shape)}'.encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def capture(source, output):
    output.mkdir(parents=True, exist_ok=False)
    # The trainer atomically replaces latest.pt. One open descriptor pins one
    # completed checkpoint, even if the pathname changes while reading it.
    with source.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        stream.seek(0)
        saved = torch.load(stream, map_location='cpu', weights_only=True)
    if saved.get('kind') != 'game_sequences':
        raise ValueError('Expected a sequence checkpoint')
    weights = {k: v for k, v in saved['weights'].items()
               if k.startswith(('encoder.', 'decoder.'))}
    metadata = dict(source_sha256=digest, weights_sha256=weights_digest(weights),
                    progress=saved['progress'], network=saved['network'],
                    decoder_readout=saved['training'].get('decoder_readout', 'contextual'),
                    captured_at=time.time())
    torch.save(dict(**metadata, weights=weights), output / 'representation.pt')
    exploration = saved.get('exploration')
    if exploration:
        torch.save(dict(config=exploration['config'], weights=exploration['weights']),
                   output / 'explorer.pt')
    write_json(output / 'snapshot.json', metadata)
    print(json.dumps(metadata), flush=True)


class FrozenReconstructor(nn.Module):
    def __init__(self, config, weights, readout='contextual'):
        super().__init__()
        if readout not in ('contextual', 'local'):
            raise ValueError('Unknown decoder readout')
        self.config, self.readout = config, readout
        # Allocate no random replacement parameters before loading exact weights.
        with torch.device('meta'):
            self.encoder, self.decoder = VisualEncoder(config), VisualDecoder(config)
        self.load_state_dict(weights, strict=True, assign=True)
        self.requires_grad_(False)
        self.eval()

    @torch.inference_mode()
    def reconstruct(self, observation, *, amp=False):
        device = next(self.parameters()).device
        context = torch.autocast('cuda', dtype=torch.bfloat16) if amp else nullcontext()
        with context:
            z = WorldModel.encode(self, [observation], self.encoder.spatial, chunk=1)[0]
            if z is None:
                raise ValueError('Cannot reconstruct an empty observation')
            logits = self.decoder(z[None], *observation.final.shape, local=self.readout == 'local')
        return logits.argmax(-1)[0].cpu().numpy()


class FrozenExplorer:
    probabilities = ChangeExplorer.probabilities

    def __init__(self, saved):
        self.head = ChangeHead().cpu().eval().requires_grad_(False)
        self.head.load_state_dict(saved['weights'], strict=True)


class UniqueReservoir:
    """Uniform reservoir over distinct (level, settled image) encounters."""
    def __init__(self, capacity, seed):
        self.capacity, self.rng = capacity, random.Random(seed)
        self.seen, self.samples = set(), []

    def add(self, observation, previous, *, seed, action_index):
        if observation.final is None or observation.terminal:
            return
        level = observation.levels_completed + 1
        key = (level, observation.final.shape,
               hashlib.sha256(observation.final.tobytes()).digest())
        if key in self.seen:
            return
        self.seen.add(key)
        index = len(self.samples) if len(self.samples) < self.capacity else self.rng.randrange(len(self.seen))
        if index >= self.capacity:
            return
        row = dict(frames=observation.frames.copy(), previous=None if previous is None else previous.copy(),
                   level=level, seed=seed, action_index=action_index)
        if index == len(self.samples):
            self.samples.append(row)
        else:
            self.samples[index] = row


def collect_target(factory, target_level, seeds, actions, capacity, timeout, explorer, config):
    pool = UniqueReservoir(capacity, f'{factory.game_id}:reconstruction')
    levels, attempts, calls = set(), [], 0
    started = time.monotonic()
    expected_levels = None
    for seed_index, seed in enumerate(seeds):
        limit = actions // len(seeds) + int(seed_index < actions % len(seeds))
        if time.monotonic() - started >= timeout:
            break
        env = factory.make(seed)
        obs = env.observation
        expected_levels = obs.win_levels
        if target_level is not None and expected_levels != target_level:
            raise ValueError(f'Split final level {target_level} differs from SDK win_levels {expected_levels}')
        policy = ExplorationSession(config, seed, explorer)
        rng = random.Random(seed)
        used, reached, previous = 0, 0, None
        while True:
            if not obs.terminal and obs.final is not None:
                level = obs.levels_completed + 1
                levels.add(level)
                reached = max(reached, level)
                if target_level is None or level == target_level:
                    pool.add(obs, previous, seed=seed, action_index=used)
            if used >= limit or time.monotonic() - started >= timeout:
                break
            before = obs
            action = Action(0) if before.terminal else policy.action(before, rng)
            obs = env.step(action)
            calls += 1
            used += 1
            policy.observe(before, action, obs)
            boundary = action.id == 0 or obs.starts_new_level(before)
            previous = (before.final if not boundary and not before.terminal and
                        before.final is not None and obs.final is not None and
                        before.final.shape == obs.final.shape else None)
            if boundary:
                # Intermediate frames can belong to the preceding level.
                obs = obs.level_initial()
                policy.reset()
        attempts.append(dict(seed=seed, actions=used, highest_level=reached))
    coverage = 'sampled' if pool.samples else 'unreached'
    return dict(game=factory.game_id, target_level=target_level, win_levels=expected_levels,
                cohort='held_out_games' if target_level is None else 'reserved_final_levels',
                status=coverage, reached_levels=sorted(levels), unique_seen=len(pool.seen),
                samples=pool.samples, actions=calls, attempts=attempts,
                timed_out=time.monotonic()-started >= timeout, seconds=time.monotonic()-started)


def _collect_job(job):
    torch.set_num_threads(1)
    output, environments, game, target, seeds, actions, capacity, timeout = job
    head_file = output / 'explorer.pt'
    saved = torch.load(head_file, map_location='cpu', weights_only=True) if head_file.exists() else None
    explorer = FrozenExplorer(saved) if saved else None
    config = (SequenceExplorationConfig(**saved['config']) if saved else
              SequenceExplorationConfig(policy='mixed', allow_undo=False))
    factory = Factory(game, environments, 'offline')
    try:
        result = collect_target(factory, target, seeds, actions, capacity, timeout, explorer, config)
        # Evaluation-only observations, never wired to collection/replay/training.
        tensors = []
        for row in result.pop('samples'):
            tensors.append({k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in row.items()})
        torch.save(tensors, output / f'{game}.observations.pt')
        result['sample_count'] = len(tensors)
        result['policy'] = config.policy
        write_json(output / f'{game}.coverage.json', result)
        return result
    finally:
        factory.arcade.close_scorecard()


def collect(args):
    split = load_split(args.split)
    targets = [(g, None) for g in split.evaluation] + [(g, cap+1) for g, cap in split.training.items()]
    if args.games:
        if set(args.games) - {g for g, _ in targets}:
            raise ValueError('Requested games must belong to the supplied split')
        targets = [(g, level) for g, level in targets if g in args.games]
    if not (args.output / 'representation.pt').exists():
        raise ValueError('Capture the immutable representation before collection')
    jobs = [(args.output, args.environments_dir, g, level, args.seeds, args.actions,
             args.samples, args.timeout) for g, level in targets]
    for g, _ in targets:
        if (args.output / f'{g}.coverage.json').exists():
            raise ValueError(f'Coverage already exists for {g}; use a separate output for another attempt')
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = {pool.submit(_collect_job, job): job[2] for job in jobs}
        for future in as_completed(futures):
            game = futures[future]
            try:
                print(json.dumps(future.result()), flush=True)
            except Exception as error:
                target = dict(targets)[game]
                failure = dict(game=game, target_level=target,
                               cohort='held_out_games' if target is None else 'reserved_final_levels',
                               status='error', error=str(error))
                write_json(args.output / f'{game}.coverage.json', failure)
                print(json.dumps(failure), flush=True)


class ReconstructionMetrics:
    def __init__(self):
        self.frames = self.perfect = self.wrong = self.pixels = 0
        self.color_pixels = np.zeros(16, dtype=np.int64)
        self.color_wrong = np.zeros(16, dtype=np.int64)
        self.foreground_wrong = self.foreground_pixels = 0
        self.changed_wrong = self.changed_pixels = 0

    def add(self, actual, predicted, previous=None):
        if actual.shape != predicted.shape or not actual.size:
            raise ValueError('Expected matching nonempty images')
        if any(np.any((v < 0) | (v > 15)) for v in (actual, predicted)):
            raise ValueError('Invalid palette index')
        bad = actual != predicted
        counts = np.bincount(actual.ravel(), minlength=16)
        self.color_pixels += counts
        self.color_wrong += np.bincount(actual[bad], minlength=16)
        # All tied dominant colors are excluded; no hardcoded background color.
        foreground = counts[actual] < counts.max()
        self.foreground_pixels += int(foreground.sum())
        self.foreground_wrong += int((foreground & bad).sum())
        if previous is not None:
            if previous.shape != actual.shape:
                raise ValueError('Previous frame geometry differs')
            changed = actual != previous
            self.changed_pixels += int(changed.sum())
            self.changed_wrong += int((changed & bad).sum())
        self.frames += 1
        self.perfect += int(not bad.any())
        self.wrong += int(bad.sum())
        self.pixels += actual.size

    def report(self):
        def ratio(n, d):
            return float(n / d) if d else None
        present = self.color_pixels > 0
        return dict(frames=self.frames, wrong=self.wrong, pixels=self.pixels,
                    pixel_error=ratio(self.wrong, self.pixels), perfect_frames=self.perfect,
                    perfect_fraction=ratio(self.perfect, self.frames),
                    balanced_color_error=float(np.mean(self.color_wrong[present] / self.color_pixels[present])) if present.any() else None,
                    non_dominant_error=ratio(self.foreground_wrong, self.foreground_pixels),
                    non_dominant_pixels=self.foreground_pixels,
                    changed_error=ratio(self.changed_wrong, self.changed_pixels), changed_pixels=self.changed_pixels,
                    colors=[dict(color=int(i), wrong=int(self.color_wrong[i]), pixels=int(self.color_pixels[i]))
                            for i in np.flatnonzero(present)])


def evaluate(args):
    torch.set_num_threads(args.cpu_threads)
    output = args.output
    destination = output / f'results-{args.device}{"-pilot" if args.pilot else ""}.json'
    if destination.exists():
        raise ValueError('Result already exists; preserve it and choose a separate output')
    saved = torch.load(output / 'representation.pt', map_location='cpu', weights_only=True)
    expected = saved['weights_sha256']
    if weights_digest(saved['weights']) != expected:
        raise ValueError('Snapshot weights changed')
    device = torch.device(args.device)
    amp = device.type == 'cuda' and torch.cuda.is_bf16_supported()
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction)
        torch.cuda.reset_peak_memory_stats()
    model = FrozenReconstructor(NetworkConfig(**saved['network']), saved['weights'], saved['decoder_readout']).to(device)
    del saved
    started = time.monotonic()
    targets, total_frames = [], 0
    coverage = sorted(output.glob('*.coverage.json'))
    if not coverage:
        raise ValueError('No observations collected')
    for path in coverage:
        target = json.loads(path.read_text())
        if target['status'] == 'error':
            targets.append(target)
            continue
        metrics = ReconstructionMetrics()
        samples = torch.load(output / f'{target["game"]}.observations.pt', weights_only=True)
        if args.pilot:
            samples = samples[:1] if not total_frames else []
        worst = []
        target_start = time.monotonic()
        for row in samples:
            obs = Observation(row['frames'].numpy(), (), 'NOT_FINISHED', row['level']-1, target['win_levels'])
            predicted = model.reconstruct(obs, amp=amp)
            previous = row['previous'].numpy() if row['previous'] is not None else None
            metrics.add(obs.final, predicted, previous)
            wrong = int(np.count_nonzero(obs.final != predicted))
            worst.append(dict(wrong=wrong, pixel_error=wrong/obs.final.size,
                              actual=obs.final.copy(), predicted=predicted,
                              level=row['level'], action_index=row['action_index'], seed=row['seed']))
            worst.sort(key=lambda x: x['pixel_error'], reverse=True)
            worst = worst[:3]
            total_frames += 1
        target['metrics'] = metrics.report()
        target['worst_pixel_error'] = worst[0]['pixel_error'] if worst else None
        target['inference_seconds'] = time.monotonic()-target_start
        if worst and not args.pilot:
            torch.save([{k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in row.items()}
                        for row in worst], output / f'{target["game"]}.worst-{args.device}.pt')
        targets.append(target)
        print(json.dumps(dict(game=target['game'], status=target['status'], **target['metrics'])), flush=True)
        if args.pilot and total_frames:
            break
    measured_seconds = time.monotonic()-started
    after = weights_digest(model.state_dict())
    if after != expected or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError('Frozen weight integrity failed')
    result = dict(snapshot=json.loads((output / 'snapshot.json').read_text()),
                  device=str(device), autocast_bf16=amp, weights_unchanged=True,
                  evaluated_frames=total_frames, evaluation_seconds=measured_seconds,
                  allocated_peak_mib=torch.cuda.max_memory_allocated()/2**20 if device.type == 'cuda' else None,
                  reserved_peak_mib=torch.cuda.max_memory_reserved()/2**20 if device.type == 'cuda' else None,
                  targets=targets)
    write_json(destination, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('targets', 'snapshot')}), flush=True)
    if args.hold_seconds:
        time.sleep(args.hold_seconds)  # optional bounded process-memory inspection


def gallery(output, device):
    """Render measured arrays, with errors at native pixel resolution."""
    from PIL import Image, ImageDraw, ImageFont

    result = json.loads((output / f'results-{device}.json').read_text())
    rows = [r for r in result['targets'] if r.get('metrics', {}).get('frames', 0)]
    if not rows:
        raise ValueError('No measured images for a gallery')
    colors = ['#ffffff', '#cccccc', '#999999', '#666666', '#333333', '#000000',
              '#e53aa3', '#ff7bcc', '#f93c31', '#1e93ff', '#88d8f1', '#ffdc00',
              '#ff851b', '#921231', '#4fcc30', '#a356d6']
    palette = np.array([tuple(bytes.fromhex(c[1:])) for c in colors], dtype=np.uint8)
    canvas = Image.new('RGB', (970, 100+320*len(rows)), '#101918')
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 16)
        title = ImageFont.truetype('DejaVuSans.ttf', 22)
    except OSError:
        font = title = ImageFont.load_default()
    draw.text((24, 14), 'Frozen reconstruction | held-out observations', font=title, fill='#b5e7d6')
    for x, label in zip([24, 344, 664], ['Actual image', 'Reconstructed image', 'Error mask (pink = wrong)']):
        draw.text((x, 55), label, font=font, fill='#b4c3bd')
    for index, row in enumerate(rows):
        example = torch.load(output / f'{row["game"]}.worst-{device}.pt', weights_only=True)[0]
        actual, predicted = example['actual'].numpy(), example['predicted'].numpy()
        error = np.full((*actual.shape, 3), [28, 43, 40], dtype=np.uint8)
        error[actual != predicted] = [255, 123, 192]
        top = 90+320*index
        draw.text((24, top), f'{row["game"]} | level {example["level"]} | '
                  f'{row["metrics"]["frames"]} images | worst: {example["wrong"]} wrong pixels',
                  font=font, fill='#e1ebe7')
        for x, pixels in zip([24, 344, 664], [palette[actual], palette[predicted], error]):
            tile = Image.fromarray(pixels)
            tile.thumbnail((256, 256), Image.Resampling.NEAREST)
            scale = max(1, min(256//tile.width, 256//tile.height))
            tile = tile.resize((tile.width*scale, tile.height*scale), Image.Resampling.NEAREST)
            canvas.paste(tile, (x, top+30))
    destination = output / f'gallery-{device}.png'
    canvas.save(destination)
    print(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('capture', 'collect', 'evaluate', 'gallery'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--split', type=Path, default=Path('configs/splits.public-development.json'))
    parser.add_argument('--environments-dir', type=Path, default=Path('environment_files'))
    parser.add_argument('--games', nargs='+')
    parser.add_argument('--seeds', type=int, nargs='+', default=[41001, 41003])
    parser.add_argument('--actions', type=int, default=4096)
    parser.add_argument('--samples', type=int, default=64)
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--cpu-threads', type=int, default=1)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--cuda-memory-fraction', type=float, default=.08)
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--hold-seconds', type=int, default=0)
    args = parser.parse_args()
    if not (1 <= args.actions <= 16384 and 1 <= args.samples <= 256 and 1 <= args.workers <= 4
            and 1 <= args.cpu_threads <= 8 and 0 < args.timeout <= 1800
            and 0 < args.cuda_memory_fraction <= 1 and 0 <= args.hold_seconds <= 30):
        parser.error('Invalid resource limits')
    if args.stage == 'capture':
        if args.checkpoint is None:
            parser.error('capture requires --checkpoint')
        capture(args.checkpoint, args.output)
    elif args.stage == 'collect':
        collect(args)
    elif args.stage == 'evaluate':
        evaluate(args)
    else:
        gallery(args.output, args.device)


if __name__ == '__main__':
    main()
