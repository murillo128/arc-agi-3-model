"""Resume one practice curriculum, then transfer learned weights to the next game.

The checkpoint owns stage and plan together. Only an explicitly converged trainer
can advance it. Plans/checkpoints stay outside Git; no saved replay is needed.
"""
import argparse
from contextlib import contextmanager
import fcntl
import gc
import json
from pathlib import Path
import random
import signal
import subprocess
import sys

from . import checkpoint
from .environment import SYNTHETIC_LEVELS


def emit(event, **data):
    print(json.dumps({'event': event, **data}), flush=True)


def public_split(game_ids):
    """Local development policy: stable ordering, seed 42, five game holdouts."""
    games = sorted(game_ids, key=lambda game: game.split('-')[0])
    bases = [game.split('-')[0] for game in games]
    if len(games) != 25 or len(set(bases)) != 25:
        raise ValueError('Expected the 25 distinct public SDK games; review catalog before training')
    held_out = set(random.Random(42).sample(bases, 5))
    return ([game for game in games if game.split('-')[0] not in held_out],
            [game for game in games if game.split('-')[0] in held_out])


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def refresh_checkout(plan):
    """Rebase a private, unpublished execution branch, never a review target."""
    source, target = Path(plan['source_checkout']), Path(plan['refresh_checkout'])
    branch = plan['refresh_branch']
    if target == source or not branch.startswith('training/'):
        raise ValueError('Refresh requires a separate private training worktree')
    if not target.exists():
        git(source, 'worktree', 'add', '-b', branch, str(target), plan['revision'])
    if (git(target, 'branch', '--show-current') != branch or
            git(target, 'status', '--porcelain') or
            git(target, 'rev-parse', '--path-format=absolute', '--git-common-dir') !=
            git(source, 'rev-parse', '--path-format=absolute', '--git-common-dir')):
        raise ValueError('Refresh worktree ownership/cleanliness check failed')
    git(target, 'fetch', 'origin', 'main')
    try:
        git(target, 'rebase', 'origin/main')
    except subprocess.CalledProcessError:
        # Leave the private branch and source checkpoint usable after conflicts.
        git(target, 'rebase', '--abort')
        raise
    emit('sequence_rebased', checkout=str(target), head=git(target, 'rev-parse', 'HEAD'))
    return str(target)


def stage_run(plan, stage, checkout):
    game = plan['games'][stage]
    synthetic = game in SYNTHETIC_LEVELS
    return {'game': game,
            'environments_dir': str(Path(checkout) / 'games') if synthetic else plan['sdk_environments_dir'],
            'mode': 'offline' if synthetic else 'normal',
            'levels': list(range(1, SYNTHETIC_LEVELS[game] + 1)) if synthetic else [None],
            'reserve_final_level': not synthetic}


def enroll(payload, plan):
    """Extend only the saved curriculum's tail; learned state is unchanged."""
    if 'sequence' in payload['run']:
        check_stage(payload, plan)
        return False
    expected = stage_run(plan, 0, plan['source_checkout'])
    run, progress = payload['run'], payload['progress']
    if (run['game'] != expected['game'] or run['mode'] != expected['mode'] or
            Path(run['environments_dir']).resolve() != Path(expected['environments_dir']).resolve() or
            not run['levels'] or run['levels'] != expected['levels'][:len(run['levels'])] or
            not 0 <= progress['level_index'] <= len(run['levels'])):
        raise ValueError('Checkpoint is not a compatible initial synthetic curriculum')
    extended = run['levels'] != expected['levels']
    run.update(expected)
    run['sequence'] = {'plan': plan, 'stage': 0, 'checkout': plan['source_checkout']}
    if extended:
        # Completion of an earlier short curriculum is not full-game convergence.
        progress.pop('stop_reason', None)
    return True


def check_stage(payload, plan):
    state = payload['run']['sequence']
    if state['plan'] != plan:
        raise ValueError('Sequence plan differs from the enrolled checkpoint')
    index = state['stage']
    if type(index) is not int or not 0 <= index < len(plan['games']):
        raise ValueError('Invalid sequence stage')
    expected_checkout = plan['source_checkout'] if index == 0 else plan['refresh_checkout']
    if state['checkout'] != expected_checkout:
        raise ValueError('Checkpoint execution checkout does not match sequence stage')
    expected = stage_run(plan, index, state['checkout'])
    if any(payload['run'].get(k) != v for k, v in expected.items()):
        raise ValueError('Checkpoint game/curriculum does not match sequence stage')


def converged(payload):
    return (payload['progress'].get('stop_reason') == 'curriculum_converged' and
            payload['progress']['level_index'] == len(payload['run']['levels']))


def advance(payload, plan, refresh=refresh_checkout):
    check_stage(payload, plan)
    if not converged(payload):
        raise ValueError('Only a fully converged curriculum may advance')
    previous = payload['run']['sequence']
    stage = previous['stage'] + 1
    if stage == len(plan['games']):
        return False
    checkout = refresh(plan) if stage == 1 else previous['checkout']
    payload['run'] = stage_run(plan, stage, checkout) | {
        'seed': payload['run']['seed'],
        'sequence': {'plan': plan, 'stage': stage, 'checkout': checkout}}
    payload['progress'] = dict(actions=0, updates=0, episodes=0, level_index=0,
                               level_actions=0, passes=0, best_error=1.)
    # Weights, optimizer moments, policy, current gate mode and RNG remain intact.
    return True


@contextmanager
def owner(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'sequence.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another supervisor owns this training output') from exc
        yield


def launch(checkout, output, device):
    command = [sys.executable, '-u', '-m', 'model', 'train',
               '--checkpoint', str(output / 'latest.pt'), '--output', str(output),
               '--device', device, '--max-steps', '0', '--max-level-steps', '0']
    # CUDA compilation, precision and optimizer mode are restored by the trainer.
    process = subprocess.Popen(command, cwd=checkout, start_new_session=True)
    try:
        return process.wait()
    except KeyboardInterrupt:
        process.send_signal(signal.SIGINT)
        process.wait()  # Trainer atomically saves latest.pt before exiting.
        return None


def supervise(output, plan, device):
    output = Path(output).resolve()
    with owner(output):
        payload = checkpoint.load(output / 'latest.pt')
        if enroll(payload, plan):
            checkpoint.write(output / 'latest.pt', payload)
        while True:
            check_stage(payload, plan)
            if converged(payload):
                if not advance(payload, plan):
                    emit('sequence_completed', games=plan['games'])
                    return
                # One atomic commit contains both transfer and stage. Restarting
                # between this write and launch cannot transfer a second time.
                checkpoint.write(output / 'latest.pt', payload)
                (output / 'best.pt').unlink(missing_ok=True)
            state = payload['run']['sequence']
            emit('sequence_stage', stage=state['stage'] + 1, total=len(plan['games']),
                 game=payload['run']['game'], levels=payload['run']['levels'],
                 next_games=plan['games'][state['stage'] + 1:])
            checkout = state['checkout']
            del payload
            gc.collect()
            code = launch(checkout, output, device)
            if code is None:
                emit('sequence_stopped', reason='interrupted')
                return
            if code:
                raise RuntimeError(f'Trainer exited with status {code}; sequence did not advance')
            payload = checkpoint.load(output / 'latest.pt')
            check_stage(payload, plan)
            if not converged(payload):
                emit('sequence_stopped', reason=payload['progress'].get('stop_reason', 'unknown'))
                return


def make_plan(args):
    from arc3.envs.sdk import open_arcade
    source = Path(__file__).resolve().parents[1]
    if git(source, 'status', '--porcelain'):
        raise ValueError('Commit implementation before freezing the execution revision')
    arcade = open_arcade('normal', str(args.sdk_environments_dir.resolve()))
    try:
        training, evaluation = public_split([env.game_id for env in arcade.get_environments()])
    finally:
        arcade.close_scorecard()
    plan = {'games': ['sc01-v1', 'si01-v1', *training], 'evaluation_games': evaluation,
            'sdk_environments_dir': str(args.sdk_environments_dir.resolve()),
            'source_checkout': str(source), 'revision': git(source, 'rev-parse', 'HEAD'),
            'refresh_checkout': str(args.refresh_checkout.resolve()),
            'refresh_branch': args.refresh_branch}
    args.plan.parent.mkdir(parents=True, exist_ok=True)
    with args.plan.open('x') as stream:
        json.dump(plan, stream, indent=2)
        stream.write('\n')
    emit('sequence_plan', path=str(args.plan), **plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plan', 'run'))
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--sdk-environments-dir', type=Path)
    parser.add_argument('--refresh-checkout', type=Path)
    parser.add_argument('--refresh-branch', default='training/practice-sequence')
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    if args.command == 'plan':
        if args.sdk_environments_dir is None or args.refresh_checkout is None:
            parser.error('plan requires --sdk-environments-dir and --refresh-checkout')
        make_plan(args)
    else:
        if args.output is None:
            parser.error('run requires --output')
        supervise(args.output, json.loads(args.plan.read_text()), args.device)


if __name__ == '__main__':
    main()
