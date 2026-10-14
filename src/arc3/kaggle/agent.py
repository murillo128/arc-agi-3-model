"""Official adapter for the exact shared policy; evaluation never updates weights."""

from agents.agent import Agent
from arcengine import GameAction

from arc3.core.policy import Observation, RandomPolicy


class MyAgent(Agent):
    MAX_ACTIONS = 80
    SEED = 0
    AGENT_CHECKPOINT = None
    DEVICE = "cpu"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.AGENT_CHECKPOINT is None:
            self.policy = RandomPolicy(self.SEED, self.game_id)
        else:
            from model.agent_checkpoint import load

            self.policy = load(self.AGENT_CHECKPOINT, device=self.DEVICE,
                               max_actions=self.MAX_ACTIONS, seed=f"{self.SEED}:{self.game_id}")

    def is_done(self, frames, latest_frame):
        # The upstream loop uses <= MAX_ACTIONS; stop at the exact budget here.
        return latest_frame.state.name == "WIN" or self.action_counter >= self.MAX_ACTIONS

    def choose_action(self, frames, latest_frame):
        self.selected = self.policy.choose_action(Observation.from_public(latest_frame))
        return GameAction.from_id(self.selected.id)

    def do_action_request(self, action):
        # Upstream Swarm uses threads, but GameAction payloads live on shared
        # enum members. Pass this game's data directly to the public SDK step.
        raw = self.arc_env.step(action, data=self.selected.data,
                                reasoning={"policy": self.policy.config.mode if self.AGENT_CHECKPOINT else "seeded random baseline"})
        return self._convert_raw_frame_data(raw)
