"""Validation-boundary hysteresis for joint versus predictor-only learning."""
import math


def representation_gate(config, reports, levels, previous=None):
    """Return checkpointable evidence; never mutate weights or use prediction error."""
    scope = [str(level) for level in levels]
    state = dict(levels=scope, auto_representation=config.auto_representation,
                 freeze_representation=config.freeze_representation,
                 unfreeze_error=config.representation_unfreeze_error,
                 freeze_error=config.representation_freeze_error,
                 patience=config.representation_gate_patience,
                 open_streak=0, close_streak=0, reconstruction_error=None,
                 eligible=False, changed=False, reason='disabled')
    if not config.auto_representation:
        return state
    previous = previous or {}
    if all(previous.get(key) == state[key] for key in (
            'levels', 'auto_representation', 'freeze_representation',
            'unfreeze_error', 'freeze_error', 'patience')):
        for key in ('open_streak', 'close_streak'):
            value = previous.get(key, 0)
            if type(value) is int and 0 <= value < state['patience']:
                state[key] = value

    def finite(value):
        return type(value) in (int, float) and math.isfinite(value)

    selected = [reports.get(level) for level in scope]
    eligible = bool(selected) and all(
        isinstance(report, dict)
        and finite(report.get('decoder_delta_error'))
        and 0 <= report['decoder_delta_error'] <= 1
        and finite(report.get('latent_std')) and report['latent_std'] >= 0
        and all(finite(report.get(key)) and report[key] >= minimum for key, minimum in (
            ('decoder_transitions', config.min_validation_transitions),
            ('transitions', config.min_validation_transitions),
            ('changed_transitions', config.min_changed_transitions)))
        for report in selected)
    if not eligible:
        state.update(open_streak=0, close_streak=0, reason='insufficient_evidence')
        return state
    error = max(report['decoder_delta_error'] for report in selected)
    state.update(eligible=True, reconstruction_error=error, reason='hold')
    frozen = config.freeze_representation
    high = error > config.representation_unfreeze_error
    low = error <= config.representation_freeze_error and all(
        report['latent_std'] >= config.min_latent_std for report in selected)
    state['open_streak'] = state['open_streak'] + 1 if frozen and high else 0
    state['close_streak'] = state['close_streak'] + 1 if not frozen and low else 0
    if max(state['open_streak'], state['close_streak']) >= state['patience']:
        state.update(freeze_representation=not frozen, changed=True,
                     reason='reconstruction_high' if frozen else 'reconstruction_low',
                     open_streak=0, close_streak=0)
    return state
