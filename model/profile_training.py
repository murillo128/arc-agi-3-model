"""Bounded CPU/CUDA profiling of a copied curriculum run; never updates the source.

Pass the original model.subgoal_training arguments after --, without --output or
--actions. Pause competing GPU workloads externally before comparing runs. The
copy starts with fresh optimizer/replay, following normal resume semantics.
Warmup and optional kernel tracing precede the measured window. The window includes
periodic evaluation and report writes; final evaluation/checkpoint after the action
budget is excluded. cProfile, tracing and synchronization are optional and
perturb throughput: use the same options on both
sides and use a separate unprofiled comparison for speed claims. Category timings
are exclusive host elapsed times (including waits), not additive CPU+GPU compute.
Kernel activity is available in the optional Chrome trace; annotation totals can
overlap, so do not sum them as GPU utilization. Check for late compilation.
"""
import argparse
import cProfile
from collections import defaultdict
from contextlib import contextmanager, nullcontext
import functools
import json
from pathlib import Path
import shutil
import time

import torch
from model import subgoal_training as st, online_replay, path_planning
from model.environment import Environment, Factory
from model.learning import Engine
from model.training_telemetry import PlannerTrajectory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--actions', type=int, default=384)
    parser.add_argument('--warmup', type=int, default=64)
    parser.add_argument('--trace-actions', type=int, default=0)
    parser.add_argument('--cpu-profile', action='store_true')
    parser.add_argument('--synchronize', action='store_true')
    parser.add_argument('trainer_args', nargs=argparse.REMAINDER)
    options = parser.parse_args(argv)
    if options.warmup < 1 or options.trace_actions < 0 or options.actions <= options.warmup + options.trace_actions:
        parser.error('Require actions > warmup + trace-actions, warmup >= 1 and trace-actions >= 0')
    if options.output.exists():
        parser.error('Output must not exist; the source run is read-only')
    if options.output.resolve().is_relative_to(options.source_run.resolve()):
        parser.error('Output must be outside the source run')
    args = options.trainer_args
    if args[:1] == ['--']:
        args = args[1:]
    if any(a.split('=')[0] in ('--output', '--actions') for a in args):
        parser.error('Supply output/actions to this profiler, not to the trainer')
    if not args:
        parser.error('Supply the original trainer arguments after --, excluding output/actions')
    out = options.output
    out.mkdir(parents=True)
    for filename in ('latest.pt', 'report.json'):
        shutil.copy2(options.source_run / filename, out / filename)
    if (options.source_run / 'evaluation').exists():
        shutil.copytree(options.source_run / 'evaluation', out / 'evaluation')
    saved = torch.load(out / 'latest.pt', map_location='cpu', weights_only=True)
    initial = saved['training_progress']['actions']
    del saved
    args += ['--output', str(out), '--actions', str(initial + options.actions)]
    if '--resume' not in args:
        args += ['--resume']
    (out / 'profile-command.json').write_text(json.dumps(dict(trainer_args=args,
        options={k: str(v) if isinstance(v, Path) else v for k,v in vars(options).items()}), indent=2))
    counts = defaultdict(lambda: dict(calls=0, seconds=0., cpu_seconds=0.))
    stack = []
    enabled = False
    mode = 'between'
    completed = 0
    cuda = torch.cuda.is_available() and not (
        "--device=cpu" in args or "--device" in args and args[args.index("--device")+1] == "cpu")

    def synchronize():
        if cuda:
            torch.cuda.synchronize()
    profile = cProfile.Profile()
    tracer = None
    window_begin = None
    cpu_begin = None
    window_end = cpu_end = None
    patched = []

    @contextmanager
    def span(label):
        active = enabled
        label = mode + '/' + label
        if not active:
            yield
            return
        # Synchronize only outer CPU/GPU categories. Nested categories are attributed
        # by host elapsed time; the separate kernel trace reports actual GPU activity.
        outer = not stack
        if outer and options.synchronize:
            synchronize()
        t, cpu = time.perf_counter(), time.process_time()
        entry = [0., 0.]
        stack.append(entry)
        try:
            with torch.profiler.record_function(label) if tracer is not None else nullcontext():
                yield
        finally:
            if outer and options.synchronize:
                synchronize()
            dt, dc = time.perf_counter() - t, time.process_time() - cpu
            stack.pop()
            item = counts[label]
            item['calls'] += 1
            item['seconds'] += dt - entry[0]
            item['cpu_seconds'] += dc - entry[1]
            if stack:
                stack[-1][0] += dt
                stack[-1][1] += dc

    def wrap(owner, attr, label):
        original = getattr(owner, attr)
        patched.append((owner, attr, original))
        @functools.wraps(original)
        def call(*a, **kw):
            with span(label):
                return original(*a, **kw)
        setattr(owner, attr, call)

    for owner, attr, label in [
        (st.Agent, 'act', 'planner'), (st.Agent, 'observe', 'perception'),
        (Engine, 'prediction', 'prediction_preview'),
        (st, 'real_states', 'memory'), (online_replay, 'real_states', 'memory'),
        (path_planning, 'real_states', 'memory'),
        (st.OnlinePredictor, 'update_batch', 'predictor_other'),
        (st.OnlinePredictor, '_loss', 'predictor_forward'),
        (torch.autograd, 'backward', 'backward'),
        (st.PathLearner, 'update', 'path_learning'),
        (st.OnlineRepresentation, 'update', 'representation'),
        (Environment, 'step', 'environment'), (Factory, 'make', 'environment'),
        (st, 'write_report', 'dashboard_write'),
        (PlannerTrajectory, 'capture_step', 'trajectory'),
        (PlannerTrajectory, 'add', 'trajectory'),
        (st.QualityProbe, 'observe', 'quality'),
    ]:
        wrap(owner, attr, label)

    original_play = st.play
    def play(*a, **kw):
        nonlocal mode, completed, enabled, tracer, window_begin, cpu_begin
        old_mode = mode
        mode = 'training' if any(kw.get(key) is not None for key in
            ('world', 'learner', 'representation')) else 'evaluation'
        progress = kw.get('progress')
        if mode == 'training':
            def tick(row):
                nonlocal completed, enabled, tracer, window_begin, cpu_begin, window_end, cpu_end
                completed += 1
                if progress:
                    progress(row)
                if completed == options.warmup and options.trace_actions:
                    synchronize()
                    tracer = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU] +
                        ([torch.profiler.ProfilerActivity.CUDA] if cuda else []))
                    tracer.__enter__()
                    enabled = True
                if completed == options.warmup + options.trace_actions:
                    if tracer is not None:
                        tracer.__exit__(None, None, None)
                        tracer.export_chrome_trace(str(out / 'kernels.json'))
                        (out / 'kernels.txt').write_text(tracer.key_averages().table(sort_by='self_cuda_time_total', row_limit=40))
                        del tracer
                        tracer = None
                    counts.clear()
                    synchronize()
                    enabled = True
                    window_begin, cpu_begin = time.perf_counter(), time.process_time()
                    if options.cpu_profile:
                        profile.enable()
                if completed == options.actions:
                    synchronize()
                    window_end, cpu_end = time.perf_counter(), time.process_time()
                    enabled = False
                    profile.disable()
            kw['progress'] = tick
        try:
            return original_play(*a, **kw)
        finally:
            mode = old_mode
    patched.append((st, 'play', original_play))
    st.play = play
    wrap(st, 'evaluate_recording', 'quality_recording')
    try:
        st.main(args)
    finally:
        profile.disable()
        for owner, attr, original in reversed(patched):
            setattr(owner, attr, original)
        if tracer is not None:
            tracer.__exit__(None, None, None)
        if window_begin is not None:
            synchronize()
            result = dict(completed=completed, measured_actions=max(0, completed-options.warmup-options.trace_actions),
                wall_seconds=(window_end or time.perf_counter())-window_begin,
                cpu_seconds=(cpu_end or time.process_time())-cpu_begin,
                categories=dict(counts), peak_gpu_allocated=torch.cuda.max_memory_allocated() if cuda else 0,
                peak_gpu_reserved=torch.cuda.max_memory_reserved() if cuda else 0)
            (out / 'timing.json').write_text(json.dumps(result, indent=2))
            if options.cpu_profile:
                profile.dump_stats(out / 'cpu.prof')
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
