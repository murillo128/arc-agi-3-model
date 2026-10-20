"""Paired predictor LR experiment on one shared corpus of real training games.

Probe sessions use separate seeds and never update weights. This experiment
does not write to the source checkpoint, the live report, or curriculum gates.
Large local artifacts belong in ignored run storage, never in Git.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import pickle
import random
import time

import numpy as np
import torch

from . import checkpoint
from .config import SequenceConfig
from .game_sequences import SequenceLearner, collect_session, evaluate_sessions, merge_evaluations
from .sequence_control import SequenceControlConfig, SequencePhaseControl
from .sequence_training import (close_factories, factories_for, fit_batch, load_engine,
    predictor_learning_rate, representation_for, restore_run, synchronize, write_report)
from .training_config import load_plan


def emit(event, **values):
    print(json.dumps(dict(event=event, **values), allow_nan=False), flush=True)


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def summarize(quality, sources):
    """Report actual count ratios; games with no union have no defined error."""
    result = {}
    for name in ('all', 'synthetic', 'sdk'):
        games = [g for g, source in sources.items()
                 if name == 'all' or source == name]
        rows = [quality[g] for g in games if g in quality]
        counts = {k: sum(r[k] for r in rows)
                  for k in ('wrong', 'union', 'transitions', 'invented', 'missed', 'color', 'correct')}
        errors = [r['delta_error'] for r in rows if r['delta_error'] is not None]
        result[name] = dict(**counts,
            pooled_error=counts['wrong']/counts['union'] if counts['union'] else None,
            macro_error=sum(errors)/len(errors) if errors else None, measured_games=len(errors))
    return result


def compare(left, right, sources):
    """Negative differences favor the candidate; uncertainty resamples whole games."""
    paired = [g for g in sources if left[g]['delta_error'] is not None
              and right[g]['delta_error'] is not None]
    if not paired:
        return dict(candidate_better=False, paired_games=0)
    a = np.array([[left[g]['wrong'], left[g]['union']] for g in paired], dtype=float)
    b = np.array([[right[g]['wrong'], right[g]['union']] for g in paired], dtype=float)
    differences = b[:, 0]/b[:, 1] - a[:, 0]/a[:, 1]
    samples = np.random.default_rng(106).integers(0, len(paired), (2000, len(paired)))
    boot_a, boot_b = a[samples].sum(1), b[samples].sum(1)
    pooled = boot_b[:, 0]/boot_b[:, 1] - boot_a[:, 0]/boot_a[:, 1]
    groups = {label: summarize(q, sources) for label, q in (('baseline', left), ('candidate', right))}
    all_a, all_b = groups['baseline']['all'], groups['candidate']['all']
    # A lower loss alone is insufficient; both pixel- and game-weighted errors
    # must improve, with no reversal in either source group.
    source_ok = all(groups['candidate'][s]['pooled_error'] <= groups['baseline'][s]['pooled_error']
                    for s in ('synthetic', 'sdk') if groups['baseline'][s]['pooled_error'] is not None
                    and groups['candidate'][s]['pooled_error'] is not None)
    better = (all_b['pooled_error'] < all_a['pooled_error']
              and all_b['macro_error'] < all_a['macro_error'] and source_ok)
    return dict(candidate_better=better, paired_games=len(paired),
        improved_games=int((differences < 0).sum()), worsened_games=int((differences > 0).sum()),
        tied_games=int((differences == 0).sum()),
        macro_difference=float(differences.mean()),
        macro_ci95=np.quantile(differences[samples].mean(1), [.025, .975]).tolist(),
        pooled_difference=all_b['pooled_error']-all_a['pooled_error'],
        pooled_ci95=np.quantile(pooled, [.025, .975]).tolist(), groups=groups)


def run(args):
    if args.probe_steps < 1:
        raise ValueError('Probe steps must be positive')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    plan = load_plan(args.training_config)
    config = SequenceConfig(**plan.payload['sequence'])
    initial = checkpoint.load(args.checkpoint)
    if initial.get('kind') != 'game_sequences' or initial['plan'] != plan.payload:
        raise ValueError('Experiment requires the exact saved sequence training plan')
    sources = {s['game']: s['source'] for s in plan.stages}
    games = list(sources)
    random.Random(initial['progress']['batches']).shuffle(games)
    training, probes = [], []
    factories = {}
    try:
        factories = factories_for(plan)
        for i, game in enumerate(games):
            factory, levels = factories[game]
            seed = 4_000_000 + 2 * (plan.settings['seed'] * 10_000
                + initial['progress']['batches'] * config.batch_games + i)
            training.append(collect_session(factory, levels, seed, config))
            probes.append(collect_session(factory, levels, seed+1,
                replace(config, session_steps=args.probe_steps)))
            emit('collected', game=game, index=i+1, total=len(games),
                 train_steps=len(training[-1].transitions), probe_steps=len(probes[-1].transitions))
    finally:
        close_factories(factories)
    corpus = output/'corpus.pkl'
    with corpus.open('wb') as stream:
        pickle.dump(dict(training=training, probes=probes), stream)
    # The artifact is generated here and is never loaded from an untrusted source.
    report = dict(checkpoint_sha256=file_hash(args.checkpoint),
        source_progress=initial['progress'], corpus_sha256=file_hash(corpus),
        sequence=plan.payload['sequence'], probe_steps=args.probe_steps, sources=sources,
        training_seeds={s.game: s.seed for s in training},
        probe_seeds={s.game: s.seed for s in probes}, trials=[])
    del initial
    report_path = output/'comparison.json'
    write_report(report_path, report)
    size = config.batch_games
    train_batches = [training[i:i+size] for i in range(0, len(training), size)]
    probe_batches = [probes[i:i+size] for i in range(0, len(probes), size)]

    for rate in (3e-5, 1e-4):
        engine = load_engine(plan, args.checkpoint, device=args.device)
        learner = SequenceLearner(engine, config)
        representation = representation_for(engine, plan)
        control = (SequencePhaseControl(plan.stages, plan.settings,
            SequenceControlConfig(**plan.payload['sequence_control']))
            if 'sequence_control' in plan.payload else None)
        saved = checkpoint.load(args.checkpoint)
        restore_run(saved, plan, learner, representation, control)
        if control is not None and control.phase != 'predictor':
            raise ValueError('The paired experiment requires predictor phase')
        predictor_learning_rate(learner, rate)
        del saved
        trial = dict(learning_rate=rate, representation_rates=representation.learning_rates,
                     batches=[], probe_batches=[])
        if not report['trials']:
            initial_reports = []
            for i, batch in enumerate(probe_batches):
                quality = evaluate_sessions(engine, batch, reference_loss=True)
                initial_reports.append(quality)
                emit('initial_probe', batch=i+1, total=len(probe_batches))
            report['initial_probe'] = merge_evaluations(initial_reports)
            report['initial_summary'] = summarize(report['initial_probe'], sources)
            write_report(report_path, report)
            # Evaluation is read-only but explicitly reset the RNG before training.
            saved = checkpoint.load(args.checkpoint)
            torch.set_rng_state(saved['torch_rng'])
            if engine.device.type == 'cuda':
                torch.cuda.set_rng_state_all(saved['cuda_rng'])
            del saved
        report['trials'].append(trial)
        for i, batch in enumerate(train_batches):
            measured = fit_batch(learner, batch, representation)
            trial['batches'].append(dict(index=i+1, games=[s.game for s in batch], **measured))
            write_report(report_path, report)
            emit('trained', learning_rate=rate, batch=i+1, total=len(train_batches),
                 loss=measured['loss'], seconds=measured['training_seconds'])
        for i, batch in enumerate(probe_batches):
            quality = evaluate_sessions(engine, batch, reference_loss=True)
            trial['probe_batches'].append(quality)
            write_report(report_path, report)
            emit('final_probe', learning_rate=rate, batch=i+1, total=len(probe_batches))
        trial['quality'] = merge_evaluations(trial['probe_batches'])
        trial['summary'] = summarize(trial['quality'], sources)
        trial['final_representation_rates'] = representation.learning_rates
        trial['updates'] = learner.updates
        write_report(report_path, report)
        emit('trial_complete', learning_rate=rate, summary=trial['summary'])
        del learner, representation, engine, control
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()
    report['comparison'] = compare(*(t['quality'] for t in report['trials']), sources)
    report['source_checkpoint_unchanged'] = file_hash(args.checkpoint) == report['checkpoint_sha256']
    write_report(report_path, report)
    emit('complete', comparison=report['comparison'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--probe-steps', type=int, default=128)
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_threads)
    run(args)


if __name__ == '__main__':
    main()
