"""Prepare verified, data-free weight assets; never uploads or submits them."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import tempfile

import torch

from . import agent_checkpoint, checkpoint
from .curriculum import CurriculumConfig, measured
from .training_config import digest, read_json, run_owner


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def final_quality_passes(quality, success, gates, world_only=False):
    """Frozen final seeds test quality, not drift between training updates.

    Drift/patience remain required in each stage's development qualification;
    a single frozen checkpoint cannot supply that temporal measurement.
    """
    if not isinstance(quality, dict):
        return False
    limits = {'reconstruction_error': gates.representation_open,
              'prediction_error': gates.prediction_open, 'rollout_error': gates.prediction_open}
    if any(not measured(quality.get(k)) or quality[k] > v for k, v in limits.items()):
        return False
    if min(quality.get('samples', 0), quality.get('rollout_samples', 0)) < gates.minimum_samples:
        return False
    stationary = (quality.get('changed') == 0 and quality.get('rollout_changed') == 0
                  and quality.get('completed_episodes', 0) > 0)
    if not stationary and min(quality.get('changed', 0), quality.get('rollout_changed', 0)) < gates.minimum_changed:
        return False
    return world_only or (measured(success) and gates.success <= success <= 1)


def _finite(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(v) for v in value)
    return True


def prepare(run, output):
    run = Path(run)
    if not run.is_dir():
        raise ValueError('Training run directory does not exist')
    # Keep report, hash checks, load and export bound to one immutable snapshot.
    # A simultaneously resumed configured trainer must not replace latest.pt.
    with run_owner(run, True):
        return _prepare(run, output)


def _prepare(run, output):
    run, output = Path(run), Path(output)
    if output.exists():
        raise ValueError('Release output must not already exist; never overwrite assets')
    report = read_json(run / 'report.json')
    saved = checkpoint.load(run / 'latest.pt')
    config = report.get('config', {})
    plan = config.get('training_plan')
    progress = saved.get('training_progress', {})
    if not plan or config.get('training_plan_id') != digest(plan) or progress.get('training_plan_id') != digest(plan):
        raise ValueError('Report/checkpoint training plan identity mismatch')
    if report.get('checkpoint_sha256') != file_sha256(run / 'latest.pt'):
        raise ValueError('Final evidence does not belong to this exact checkpoint')
    if (report.get('status') != 'converged' or not progress.get('curriculum', {}).get('completed')
            or report.get('manual_controls') or progress.get('manual_controls')
            or report.get('checkpoint_action') != progress.get('actions')
            or report.get('curriculum') != progress.get('curriculum')):
        raise ValueError('Publication requires a converged run with matching state and no manual overrides')
    provenance = report.get('provenance', {})
    if provenance.get('dirty') is not False or not re.fullmatch(r'[0-9a-f]{40}', provenance.get('revision', '')):
        raise ValueError('Publication requires a recorded clean source revision')
    stages = plan['stages']
    evidence = report.get('final_validation', [])
    rows = progress['curriculum']['levels']
    pairs = [(s['game'], s['level']) for s in stages]
    if ([(r['game'], r['level']) for r in evidence] != pairs or
            [(r['game'], r['level']) for r in rows] != pairs):
        raise ValueError('Final validation must cover every requested stage exactly once in order')
    for stage, final, row in zip(stages, evidence, rows, strict=True):
        gates = CurriculumConfig(**stage['thresholds'])
        if (final.get('thresholds') != asdict(gates) or final.get('source') != stage['source']
                or row.get('promotion_passes', 0) < gates.patience
                or final.get('objective') != stage['objective']
                or not final_quality_passes(final.get('quality'), final.get('success'), gates, stage['objective'] == 'world_model')):
            raise ValueError(f'Quality/success gates did not pass for {stage["game"]}/{stage["level"]}')
    if not _finite(saved):
        raise ValueError('Checkpoint contains nonfinite weights')
    agent = agent_checkpoint.load(run / 'latest.pt', device='cpu', compile_model=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.training-release-', dir=output.parent) as temporary:
        staged = Path(temporary) / 'assets'
        staged.mkdir()
        agent_checkpoint.save(staged / 'agent.pt', agent, include_runtime=False)
        # No local directory, runtime carry, observations, replay or cached
        # trajectories are copied. Only allowlisted reproducibility metadata.
        public_plan = {k: plan[k] for k in ('version', 'phase', 'initialization', 'settings', 'stages')}
        if plan.get('sdk'):
            public_plan['sdk'] = {k: v for k, v in plan['sdk'].items() if k != 'environments_dir'}
        manifest = dict(version=1, training_plan_id=digest(plan), source_revision=provenance['revision'],
            training_actions=progress['actions'], action_budget=config.get('actions'), plan=public_plan,
            validation=[{k: r[k] for k in ('game', 'level', 'source', 'objective', 'thresholds', 'quality', 'success')} for r in evidence],
            agent_sha256=file_sha256(staged / 'agent.pt'),
            scope='Local training validation; subgoal planner goals are teacher-provided. World-model stages validate predictive quality, not autonomous planning.')
        (staged / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
        (staged / 'SHA256SUMS').write_text(''.join(f'{file_sha256(staged / name)}  {name}\n'
                                               for name in ('agent.pt', 'manifest.json')))
        staged.rename(output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = prepare(args.run, args.output)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(dict(output=str(args.output), agent_sha256=result['agent_sha256'])))


if __name__ == '__main__':
    main()
