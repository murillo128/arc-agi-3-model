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


def with_goal(base):
    """Literal absolute goal differs from both endpoint and observed result."""
    attempt = with_plan(base)
    p = attempt['steps'][0]['planning']
    p.update(objective='latent_goal', mode='goal_mpc', selected_score=0.6065306597126334)
    p['goal'] = {'source': 'imagined_proposal', 'created_position': 0, 'proposal_score': .8,
                 'distance': 2., 'frame': {'dtype': 'u8', 'shape': [2, 2], 'data': bytes([4, 4, 4, 4])}}
    for step, frame, distance in zip(p['steps'], (bytes([1, 1, 1, 1]), bytes([2, 2, 2, 2])), (1., .5)):
        step.update(frame={'dtype': 'u8', 'shape': [2, 2], 'data': frame}, goal_distance=distance)
    return attempt


def invalid_goals():
    p = ('steps', 0, 'planning')
    g = (*p, 'goal')
    return [
        ('future', (*g, 'created_position'), 1, 'integer'),
        ('mode-pair', (*p, 'objective'), 'sdk_level_progress', 'incompatible'),
        ('missing-goal', g, None, 'map'),
        ('source', (*g, 'source'), 'real_future', 'expected'),
        ('goal-geometry', (*g, 'frame'), {'dtype': 'u8', 'shape': [1, 4], 'data': bytes(4)}, 'geometry'),
        ('step-geometry', (*p, 'steps', 0, 'frame'), {'dtype': 'u8', 'shape': [1, 4], 'data': bytes(4)}, 'geometry'),
        ('palette', (*g, 'frame', 'data'), bytes([16, 0, 0, 0]), 'palette'),
        ('negative', (*g, 'distance'), -1., 'range|>='),
        ('nonfinite', (*p, 'steps', 0, 'goal_distance'), float('inf'), 'finite'),
        ('negative-step', (*p, 'steps', 0, 'goal_distance'), -.1, 'range|>='),
        ('missing-image', (*p, 'steps', 0, 'frame'), None, 'missing'),
        ('contradictory-score', (*p, 'selected_score'), .9, 'contradicts'),
        ('proposal-range', (*g, 'proposal_score'), 1.1, 'outside'),
    ]
