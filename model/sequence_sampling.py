"""Explicit session frequencies; omitted games keep one visit per round."""


def game_weights(value, games):
    if not isinstance(value, dict) or set(value) - set(games):
        raise ValueError('sequence_game_weights must map configured game IDs to integer weights')
    if any(type(weight) is not int or not 1 <= weight <= 16 for weight in value.values()):
        raise ValueError('sequence_game_weights must contain integer weights in 1..16')
    # Canonical defaults preserve the identity and ordering of old plans.
    return {game: value[game] for game in games if value.get(game, 1) != 1}


def reweighted_plan_matches(previous, current):
    return ({k: v for k, v in previous.items() if k != 'sequence_game_weights'} ==
            {k: v for k, v in current.items() if k != 'sequence_game_weights'})
