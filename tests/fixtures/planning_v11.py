"""Literal pre-action plan; probabilities and values are synthetic, not learned."""
from copy import deepcopy


def with_plan(base):
    attempt = deepcopy(base)
    attempt['steps'][0]['planning'] = {
        'objective': 'sdk_level_progress', 'mode': 'mpc', 'horizon': 2, 'beam': 4,
        'discount': .9, 'state_value': .4, 'selected_score': .65,
        'steps': [
            {'action': deepcopy(attempt['steps'][0]['action']), 'continue': .6, 'progress': .3,
             'game_over': .1, 'successor_value': .5},
            {'action': {'id': 6, 'data': {'x': 3, 'y': 5}}, 'continue': .2, 'progress': .8,
             'game_over': 0., 'successor_value': .7},
        ],
    }
    return attempt


def invalid_plans():
    p = ('steps', 0, 'planning')
    return [
        ('null', p, None, 'map'),
        ('mode', (*p, 'mode'), 'invented', 'expected'),
        ('score-range', (*p, 'selected_score'), 1.5, 'range'),
        ('negative-probability', (*p, 'steps', 0, 'game_over'), -.1, 'probability'),
        ('probability-sum', (*p, 'steps', 0, 'continue'), .8, 'sum'),
        ('nonfinite', (*p, 'state_value'), float('nan'), 'finite'),
        ('length', (*p, 'horizon'), 1, 'length'),
        ('too-long', (*p, 'horizon'), 65, 'integer'),
        ('different-action', (*p, 'steps', 0, 'action'), {'id': 6, 'data': {'x': 1, 'y': 2}}, 'differs'),
        ('zero-discount', (*p, 'discount'), 0, 'range'),
        ('bad-click', (*p, 'steps', 1, 'action', 'data', 'x'), 64, 'integer'),
    ]
