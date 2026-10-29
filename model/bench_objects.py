"""Offline CPU-capable object audit on copied weights and real synthetic play.

This is a training-side probe, not a held-out evaluation or a success benchmark.
Never writes the source checkpoint or starts/stops the live trainer.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import time
from types import SimpleNamespace

import torch

from .config import NetworkConfig, SequenceConfig, TrainConfig
from .environment import Factory
from .game_sequences import SequenceLearner, collect_session
from .learning import Engine
from .networks import WorldModel, Dynamics
from .object_training import ObjectTrainer, ObjectTrainingConfig
from .object_upgrade import migrate
from .segmentation import SAM2Segmenter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--sam-checkpoint', required=True)
    parser.add_argument('--output', required=True, help='New JSON evidence file, outside Git')
    parser.add_argument('--game', default='mc01-v1', choices=['mc01-v1', 'sc01-v1', 'si01-v1', 'bf01-v1', 'bf02-v1'])
    parser.add_argument('--passes', type=int, default=8)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if Path(args.output).exists() or not 1 <= args.passes <= 64 or args.threads < 1:
        parser.error('Use a new output file, 1..64 passes and positive threads')
    torch.set_num_threads(args.threads)
    saved = torch.load(args.source, map_location='cpu', weights_only=True, mmap=True)
    previous = NetworkConfig(**saved['network'])
    target = replace(previous, object_slots=32, object_width=16)
    configuration = ObjectTrainingConfig(args.sam_checkpoint, samples=2, passes=1, teacher_frames=8, click_frames=0)
    payload = deepcopy(saved['plan']); payload['initialization']['network'] = asdict(target)
    payload['sequence_objects'] = asdict(configuration)
    updated = migrate(saved, SimpleNamespace(payload=payload))
    config = SequenceConfig(batch_games=1, session_steps=6, level_steps=6, block_steps=3)
    factory = Factory(args.game, Path(__file__).resolve().parents[1]/'games')
    try:
        real = collect_session(factory, [1], 178, config)
    finally:
        factory.arcade.close_scorecard()
    model = WorldModel(previous).eval(); model.load_state_dict(saved['weights'])
    engine = Engine(model, TrainConfig(**saved['training']))
    with torch.no_grad():
        z = engine.encode([real.transitions[0].before])[0][None]
        h = torch.zeros(1, previous.memory); action = engine.actions([real.transitions[0].action])
        original = engine.predict(z, h, action, 64, 64)
    model.dynamics = Dynamics(target).eval()
    model.config = target
    model.load_state_dict(updated['weights'], strict=True)
    engine = Engine(model, engine.config)
    with torch.no_grad():
        predicted = engine.predict(z, h, action, 64, 64)
        for a, b in zip(original, predicted):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    learner = SequenceLearner(engine, config)
    learner.optimizer.load_state_dict(updated['predictor_optimizer'])
    teacher = SAM2Segmenter(args.sam_checkpoint)
    trainer = ObjectTrainer(engine, configuration, teacher)
    records = []; start = time.perf_counter()
    for index in range(args.passes):
        measured, sample = trainer.fit(learner, [real])
        records.append(measured)
        print(json.dumps(dict(pass_index=index+1, **measured)), flush=True)
    unchanged = all(torch.equal(tensor, model.state_dict()[name]) for name, tensor in saved['weights'].items())
    if not unchanged:
        raise AssertionError('Auxiliary learning changed existing weights')
    report = dict(source_batch=saved['progress']['batches'], game=args.game, seed=178, real_actions=len(real.transitions),
                  exact_initial_prediction=True, old_weights_unchanged=unchanged, residual_gain=float(model.dynamics.objects.gain),
                  seconds=time.perf_counter()-start, records=records, sample=sample)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
