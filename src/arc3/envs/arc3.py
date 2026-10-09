"""Lazy SDK boundary; importing core does not require the game engine."""

from __future__ import annotations


def make_arcade(*, offline: bool = False):
    from arc_agi import Arcade, OperationMode

    return Arcade(operation_mode=OperationMode.OFFLINE if offline else OperationMode.NORMAL)


def reset_action():
    from arcengine import GameAction

    return GameAction.RESET
