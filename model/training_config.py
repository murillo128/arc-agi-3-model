"""Strict, reusable configuration for the whole-agent training entry point."""
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re

from arc3.envs.splits import load_split
from .config import NetworkConfig, TrainConfig, SequenceConfig
from .curriculum import CurriculumConfig
from .environment import SYNTHETIC_LEVELS, synthetic_levels


SETTINGS = {
    'actions': 262144, 'steps': 128, 'seed': 0, 'evaluate_every': 256,
    'evaluation_episodes': 8, 'checkpoint_every': 2048,
    'learning_rate': 3e-5, 'representation_learning_rate': 3e-6,
    'encoder_learning_rate_after_predictor': 3e-7,
    'freeze_encoder': False, 'freeze_representation_after_planner': True,
    'prediction_delta_loss': 'recall_focal_capped',
    'predictor_replay_episodes': 0, 'predictor_batch_size': 1,
    'path_hindsight': False, 'path_predicted_inputs': False,
    'path_exploration_warmup': 8192, 'path_horizon': 3, 'path_noop_cooldown': 8,
}


def object_keys(value, allowed, name):
    if not isinstance(value, dict) or set(value) - set(allowed):
        raise ValueError(f'{name}: expected an object with keys {sorted(allowed)}')
    return value


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Duplicate JSON key: {key}')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f'Nonfinite value: {value}')))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def thresholds(value, defaults=None):
    values = asdict(defaults or CurriculumConfig()) | object_keys(value, CurriculumConfig.__dataclass_fields__, 'thresholds')
    for key in ('representation_open', 'prediction_open', 'latent_drift', 'success'):
        if type(values[key]) not in (float, int) or not math.isfinite(values[key]):
            raise ValueError(f'Invalid threshold: {key}')
    return CurriculumConfig(**values)


@dataclass(frozen=True)
class TrainingPlan:
    init_config: Path
    settings: dict
    stages: tuple
    payload: dict

    @property
    def identity(self):
        return digest(self.payload)

    def arguments(self):
        result = ['--init-config', str(self.init_config), '--game', self.stages[0]['game']]
        for key, value in self.settings.items():
            flag = key.replace('_', '-')
            result.extend([f'--{flag}' if value else f'--no-{flag}'] if type(value) is bool
                          else [f'--{flag}', str(value)])
        return result


def load_plan(path):
    path = Path(path).resolve()
    value = object_keys(read_json(path), {'version', 'phase', 'init_config', 'settings', 'thresholds', 'sdk', 'curriculum', 'sequence', 'sequence_control', 'sequence_exploration'}, 'training config')
    if type(value.get('version')) is not int or value['version'] != 1:
        raise ValueError('Training configuration version must be 1')
    phase = value.get('phase', 'development')
    if phase not in ('development', 'final'):
        raise ValueError('phase must be development or final')
    if 'sequence' in value and phase != 'development':
        raise ValueError('Sequence experiments require phase development')
    def relative(raw):
        if not isinstance(raw, str) or not raw:
            raise ValueError('Expected a nonempty file/directory path')
        return (path.parent / raw).resolve()
    initial_path = relative(value.get('init_config'))
    initial = object_keys(read_json(initial_path), {'network', 'training', 'outcome_width'}, 'init_config')
    NetworkConfig(**initial['network'])
    TrainConfig(**initial.get('training', {}))
    if type(initial.get('outcome_width', 64)) is not int or initial.get('outcome_width', 64) < 1:
        raise ValueError('outcome_width must be positive')
    settings = SETTINGS | object_keys(value.get('settings', {}), SETTINGS, 'settings')
    for key, default in SETTINGS.items():
        actual = settings[key]
        if type(default) is bool:
            valid = type(actual) is bool
        elif type(default) is int:
            valid = type(actual) is int and actual >= (0 if key in ('seed', 'predictor_replay_episodes', 'path_noop_cooldown') else 1)
        elif type(default) is float:
            valid = type(actual) in (int, float) and math.isfinite(actual) and 0 < actual < 1
        else:
            valid = actual in ('balanced', 'union', 'focal', 'recall_focal', 'recall_focal_capped')
        if not valid:
            raise ValueError(f'Invalid setting: {key}')
    if settings['steps'] > 128 or settings['seed'] >= 100:
        raise ValueError('steps must be <= 128 and seed < 100')
    if settings['encoder_learning_rate_after_predictor'] > settings['representation_learning_rate']:
        raise ValueError('Post-predictor encoder LR must not exceed representation LR')
    if settings['predictor_batch_size'] > max(1, settings['predictor_replay_episodes']):
        raise ValueError('Batch size exceeds replay capacity')
    from .path_planning import PathConfig
    PathConfig(horizon=settings['path_horizon'], noop_cooldown=settings['path_noop_cooldown'])
    default_gates = thresholds(value.get('thresholds', {}))
    entries = value.get('curriculum')
    if not isinstance(entries, list) or not entries:
        raise ValueError('curriculum must be a nonempty ordered list')
    sdk = object_keys(value.get('sdk', {}), {'environments_dir', 'mode', 'split', 'access_steps'}, 'sdk')
    sdk_config, split = {}, None
    if sdk:
        sdk_config = dict(environments_dir=str(relative(sdk.get('environments_dir'))),
                          mode=sdk.get('mode', 'offline'), access_steps=sdk.get('access_steps', 512))
        if sdk_config['mode'] not in ('offline', 'normal'):
            raise ValueError('SDK mode must be offline or normal')
        if type(sdk_config['access_steps']) is not int or sdk_config['access_steps'] < 1:
            raise ValueError('SDK access_steps must be positive')
        if phase == 'development':
            split = load_split(relative(sdk.get('split')))
            sdk_config['split'] = dict(training_games=list(split.training),
                                      training_level_caps=split.training, evaluation_games=list(split.evaluation))
    stages, seen = [], set()
    bundled = synthetic_levels()
    for entry in entries:
        object_keys(entry, {'source', 'game', 'levels', 'thresholds', 'objective'}, 'curriculum entry')
        source, game, levels = entry.get('source'), entry.get('game'), entry.get('levels')
        if source not in ('synthetic', 'sdk') or not isinstance(game, str) or not re.fullmatch(r'[a-z0-9]{4}(?:-[a-z0-9]+)?', game):
            raise ValueError('Every entry needs a source and valid game ID')
        if source == 'synthetic' and game not in bundled:
            raise ValueError(f'Unknown bundled synthetic game: {game}')
        if source == 'sdk' and (not sdk_config or game.split('-')[0] in {g.split('-')[0] for g in bundled}):
            raise ValueError('SDK entries need sdk settings and a public SDK game')
        objective = entry.get('objective', 'subgoals' if 'sequence' not in value and source == 'synthetic' and game in SYNTHETIC_LEVELS else 'world_model')
        if 'sequence' in value and objective != 'world_model':
            raise ValueError('Sequence mode trains world_model; it does not train subgoal/planner heads')
        if objective not in ('subgoals', 'world_model') or (objective == 'subgoals' and
                (source != 'synthetic' or game not in SYNTHETIC_LEVELS)):
            raise ValueError('Subgoal training is only supported by sc01-v1 and si01-v1')
        if not isinstance(levels, list) or not levels:
            raise ValueError('Every entry needs a nonempty ordered levels list')
        cap = None
        if source == 'sdk' and phase == 'development':
            matches = [g for g in split.training if g.split('-')[0] == game.split('-')[0]]
            if len(matches) != 1:
                raise ValueError(f'{game} is not in the development training split')
            cap = split.training[matches[0]]
        gates = thresholds(entry.get('thresholds', {}), default_gates)
        for level in levels:
            if type(level) is not int or level < 1 or (source == 'synthetic' and level > bundled[game]):
                raise ValueError(f'Invalid level {level!r} for {game}')
            key = game.split('-')[0], level
            if key in seen:
                raise ValueError(f'Duplicate game/level: {game}/{level}')
            if cap is not None and level > cap:
                raise ValueError(f'{game}/{level} exceeds its training level cap')
            seen.add(key)
            stages.append(dict(game=game, level=level, source=source, objective=objective, thresholds=asdict(gates)))
    payload = dict(version=1, phase=phase, initialization=initial, settings=settings,
                   stages=stages, sdk=sdk_config)
    if 'sequence' in value:
        payload['sequence'] = asdict(SequenceConfig(**object_keys(
            value['sequence'], SequenceConfig.__dataclass_fields__, 'sequence')))
    if 'sequence_exploration' in value:
        from .sequence_exploration import SequenceExplorationConfig
        if 'sequence' not in value:
            raise ValueError('sequence_exploration requires sequence training')
        payload['sequence_exploration'] = asdict(SequenceExplorationConfig(**object_keys(
            value['sequence_exploration'], SequenceExplorationConfig.__dataclass_fields__, 'sequence_exploration')))
    if 'sequence_control' in value:
        from .sequence_control import SequenceControlConfig
        if 'sequence' not in value:
            raise ValueError('sequence_control requires sequence training')
        control = SequenceControlConfig(**object_keys(value['sequence_control'],
            SequenceControlConfig.__dataclass_fields__, 'sequence_control'))
        if control.decoder_learning_rate_after_predictor > settings['representation_learning_rate']:
            raise ValueError('Post-predictor decoder LR must not exceed representation LR')
        payload['sequence_control'] = asdict(control)
    return TrainingPlan(initial_path, settings, tuple(stages), payload)


@contextmanager
def run_owner(output, resume):
    """Hold one process lease across report/checkpoint creation and resume."""
    output = Path(output)
    if not resume and output.exists() and any(output.iterdir()):
        raise ValueError('A fresh training run requires an empty output directory')
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'training.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('Another trainer owns this output directory') from error
        # Recheck after taking the lease, including simultaneous fresh launches.
        if not resume and any(p.name != 'training.lock' for p in output.iterdir()):
            raise ValueError('A fresh training run requires an empty output directory')
        yield
