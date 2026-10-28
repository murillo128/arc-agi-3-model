"""Official transport adapter with explicitly opt-in per-game adaptation."""

import logging
import time

from agents.agent import Agent
from arcengine import GameAction

from arc3.core.policy import Observation, RandomPolicy


class MyAgent(Agent):
    MAX_ACTIONS = None
    SEED = 0
    AGENT_CHECKPOINT = None
    DEVICE = "cpu"
    ADAPTIVE = False
    ADAPTATION_CONFIG = None

    def __init__(self, *args, **kwargs):
        if not self.ADAPTIVE and self.MAX_ACTIONS is None:
            self.MAX_ACTIONS = 80
        super().__init__(*args, **kwargs)
        if self.ADAPTIVE:
            if self.AGENT_CHECKPOINT is None:
                raise ValueError("Adaptive mode requires a portable adaptation base")
            from model.adaptation import AdaptationConfig, load

            self.policy = load(self.AGENT_CHECKPOINT, device=self.DEVICE,
                               config=AdaptationConfig(**(self.ADAPTATION_CONFIG or {})),
                               max_actions=self.MAX_ACTIONS, seed=f"{self.SEED}:{self.game_id}")
        elif self.AGENT_CHECKPOINT is None:
            self.policy = RandomPolicy(self.SEED, self.game_id)
        else:
            from model.agent_checkpoint import load

            self.policy = load(self.AGENT_CHECKPOINT, device=self.DEVICE,
                               max_actions=self.MAX_ACTIONS, seed=f"{self.SEED}:{self.game_id}")

    def is_done(self, frames, latest_frame):
        # The upstream loop uses <= MAX_ACTIONS; stop at the exact budget here.
        return latest_frame.state.name == "WIN" or (
            self.MAX_ACTIONS is not None and self.action_counter >= self.MAX_ACTIONS)

    def main(self):
        if self.MAX_ACTIONS is not None:
            return super().main()
        # The pinned official loop compares its counter to MAX_ACTIONS and its
        # cleanup does the same. Own just the unlimited lifecycle; keep SDK,
        # conversion, recording and real-result confirmation in their owners.
        self.timer = time.time()
        try:
            while not self.is_done(self.frames, self.frames[-1]):
                latest = self._convert_raw_frame_data(self.arc_env.observation_space)
                if self.is_done(self.frames, latest):
                    break
                action = self.choose_action(self.frames, latest)
                frame = self.take_action(action)
                if frame is not None:
                    self.append_frame(frame)
                self.action_counter += 1
                # The policy retains its own causal replay; the transport only
                # needs the current public frame, even on a long-lived game.
                self.frames[:] = self.frames[-1:]
        finally:
            self.policy.cancel_pending()
            self.cleanup()

    def cleanup(self, scorecard=None):
        if self.MAX_ACTIONS is not None:
            return super().cleanup(scorecard)
        if self._cleanup:
            self._cleanup = False
            if hasattr(self, 'recorder') and not self.is_playback and scorecard:
                self.recorder.record(scorecard.get(self.game_id))
            logging.getLogger().info('Finishing: %s took %s actions in %s seconds',
                                     self.game_id, self.action_counter, self.seconds)

    def choose_action(self, frames, latest_frame):
        self.selected = self.policy.choose_action(Observation.from_public(latest_frame))
        return GameAction.from_id(self.selected.id)

    def do_action_request(self, action):
        # Upstream Swarm uses threads, but GameAction payloads live on shared
        # enum members. Pass this game's data directly to the public SDK step.
        if self.ADAPTIVE and action.value != self.selected.id:
            raise ValueError("Executed action must match the selected action")
        mode = ("per-game LoRA adaptation" if self.ADAPTIVE else
                self.policy.config.mode if self.AGENT_CHECKPOINT else "seeded random baseline")
        try:
            raw = self.arc_env.step(action, data=dict(self.selected.data), reasoning={"policy": mode})
            result = self._convert_raw_frame_data(raw)
        except Exception:
            if self.ADAPTIVE:
                self.policy.cancel_pending()
            raise
        if self.ADAPTIVE:
            # Commit at the SDK return boundary: the framework may never call
            # choose_action again after a terminal frame or the action budget.
            self.policy.confirm_result(Observation.from_public(result), action=self.selected)
        return result
