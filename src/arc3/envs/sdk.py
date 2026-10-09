"""Only public SDK methods and observation fields cross this boundary."""

import logging
import os
from importlib.metadata import version
from typing import Any

from arc3.core.policy import Action, Observation

SDK_VERSION = "0.9.9"


def open_arcade(mode: str, environments_dir: str) -> Any:
    if mode not in {"offline", "normal"}:
        raise ValueError("Local mode must be offline or normal")
    if os.getenv("OPERATION_MODE", "").lower() in {"online", "competition"}:
        raise RuntimeError("Local lifecycle cannot use a competition/remote override")
    if version("arc-agi") != SDK_VERSION:
        raise RuntimeError(f"Install the pinned arc-agi=={SDK_VERSION}")
    from arc_agi import Arcade, OperationMode

    logger = logging.getLogger("arc3.sdk")  # Keep CLI JSON stdout free of SDK logs.
    arcade = Arcade(
        operation_mode=OperationMode[mode.upper()],
        environments_dir=environments_dir,
        logger=logger,
    )
    if arcade.operation_mode not in {OperationMode.OFFLINE, OperationMode.NORMAL}:
        raise RuntimeError("Local lifecycle cannot use a competition/remote override")
    return arcade


class SDKEnvironment:
    def __init__(self, arcade: Any, game_id: str, seed: int) -> None:
        # make initializes the game. Never make again to peek or reset it.
        self.wrapper = arcade.make(game_id, seed=seed, save_recording=False)
        if self.wrapper is None:
            raise RuntimeError(f"Could not make {game_id}; check local public game setup")
        self.observation = Observation.from_public(self.wrapper.observation_space)

    def step(self, action: Action) -> Observation:
        from arcengine import GameAction

        raw = self.wrapper.step(GameAction.from_id(action.id), data=action.data)
        self.observation = Observation.from_public(raw)
        return self.observation
