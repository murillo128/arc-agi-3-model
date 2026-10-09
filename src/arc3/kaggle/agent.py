"""Official framework adapter for the *same* policy used locally."""

from __future__ import annotations

from arcengine import GameAction
from agents.agent import Agent

from arc3.core.policy import RandomPolicy, state_name


class MyAgent(Agent):
    MAX_ACTIONS = 80

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._policy = RandomPolicy(self.game_id)

    def is_done(self, frames, latest_frame) -> bool:
        return state_name(latest_frame) == "WIN"

    def choose_action(self, frames, latest_frame) -> GameAction:
        # On competition runs, allowed actions may also appear in latest_frame;
        # the baseline follows the official Starter's all-actions strategy.
        decision = self._policy.choose(
            latest_frame, tuple(a for a in GameAction if a is not GameAction.RESET), GameAction.RESET
        )
        if decision.data is not None:
            decision.action.set_data(decision.data)
        return decision.action
