"""Opt-in offline public-training capture of structure, with fresh synthetic pixels.

Never reads game source or exports game frames. The existing runtime discards
transitions entering the configured/reserved level before this callback sees them.
No download, credentials, evaluation games or training/weight writes are involved.
"""

import argparse
import json
from pathlib import Path

from arc3.core.policy import Action
from arc3.core.runtime import run_game
from arc3.envs.sdk import SDKEnvironment, SDK_VERSION, open_arcade
from arc3.envs.splits import load_split
from arc3.traces import write_attempt


def structure(observation):
    return {"state": observation.state, "levels_completed": observation.levels_completed,
            "win_levels": observation.win_levels, "available_actions": list(observation.available_actions),
            "source_shape": [len(observation.frame),
                             len(observation.frame[0]) if observation.frame else 0,
                             len(observation.frame[0][0]) if observation.frame and observation.frame[0] else 0]}


def synthetic_observation(public, phase):
    # Keep only sequence length; replace dimensions and every pixel, without
    # sampling, downscaling, recoloring or consulting any original game bitmap.
    count = public["source_shape"][0]
    if count > 16:
        raise ValueError("Short fixture extraction refuses more than 16 frames")
    return {key: value for key, value in public.items() if key != "source_shape"} | {
        "frames": {"dtype": "u8", "shape": [count, 2, 2] if count else [0, 0, 0],
                   "data": bytes((phase + pixel) % 16 for pixel in range(count * 4))},
    }


class FixedActions:
    def __init__(self):
        self.actions = iter((1, 2, 3))

    def choose_action(self, observation):
        if observation.state == "NOT_PLAYED":
            return Action(0, {})
        if observation.state == "GAME_OVER":
            raise ValueError("Capture ended at GAME_OVER; no further reset is sampled")
        action = next(self.actions)
        if action not in observation.available_actions:
            raise ValueError(f"ACTION{action} is unavailable; this command needs public actions 1,2,3")
        return Action(action, {})


def capture(arcade, game_id, seed, cap):
    env = SDKEnvironment(arcade, game_id, seed)
    if env.observation.win_levels < 2 or env.observation.levels_completed >= min(cap, env.observation.win_levels - 1):
        raise ValueError("Initial observation is inside a reserved/unknown level; nothing extracted")
    initial = None if env.observation.state == "NOT_PLAYED" else structure(env.observation)
    steps = []

    def collect(transition):
        nonlocal initial
        public = structure(transition.next_observation)
        if transition.action.id == 0:
            initial = public  # Bootstrap result, outside attempt accounting.
        else:
            steps.append({"index": len(steps), "action": {
                "id": transition.action.id, "data": dict(transition.action.data),
            }, "observation": public})

    # Three ordinary calls plus a possible bootstrap RESET. FixedActions refuses
    # a fourth ordinary call; the SDK normally initializes make with a real frame.
    budget = 4 if initial is None else 3
    metrics = run_game(env, FixedActions(), budget, on_transition=collect, training_level_cap=cap)
    if initial is None:
        raise ValueError("No permitted initial observation available; nothing extracted")
    report = {"game_id": env.observation.game_id, "sdk_version": SDK_VERSION, "seed": seed,
              "initial_observation": initial, "steps": steps,
              "stop_reason": metrics["stop_reason"], "session_actions": metrics["actions"],
              "final_state": metrics["state"], "levels_completed": metrics["levels_completed"],
              "win_levels": metrics["win_levels"]}
    last = steps[-1]["observation"] if steps else initial
    boundary = metrics["stop_reason"] == "training_level_boundary"
    reason = {"WIN": "win", "GAME_OVER": "game_over"}.get(last["state"], metrics["stop_reason"])
    if boundary:
        reason = "training_level_boundary"
    attempt = {
        "metadata": {"game_id": "synthetic-public-structure-v1", "run_id": "public-structure-v1",
                     "session_id": "public-structure-v1", "attempt_index": 0, "seed": seed,
                     "sdk_version": SDK_VERSION, "source_split": "unspecified",
                     "started_at": "2026-10-09T12:00:00Z", "config": {
                         "source_game": report["game_id"], "source_seed": seed,
                         "transformation": "Preserve public structure/sequence count; replace all pixels with original 2x2 ramps",
                     }},
        "initial_observation": synthetic_observation(initial, 0),
        "steps": [{**step, "observation": synthetic_observation(step["observation"], step["index"] + 1)} for step in steps],
        "termination": {"reason": reason, "final_state": metrics["state"]},
        "summary": {"real_actions": len(steps) + int(boundary), "levels_completed": metrics["levels_completed"],
                    "win_levels": metrics["win_levels"], "wall_seconds": 0.0},
    }
    return report, attempt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environments-dir", required=True)
    parser.add_argument("--split", default="configs/splits.example.json")
    parser.add_argument("--game", default="ls20-9607627b")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True, help="New ignored local directory")
    args = parser.parse_args()
    training = load_split(args.split).training
    matches = [cap for game, cap in training.items() if game.split("-", 1)[0] == args.game.split("-", 1)[0]]
    if len(matches) != 1:
        parser.error("Game must belong to the approved training split")
    if args.output.exists():
        parser.error("Output already exists; choose a new directory")
    arcade = open_arcade("offline", args.environments_dir)
    try:
        report, attempt = capture(arcade, args.game, args.seed, matches[0])
    finally:
        arcade.close_scorecard()
    args.output.mkdir(parents=True)
    (args.output / "structure.json").write_text(json.dumps(report, indent=2) + "\n")
    write_attempt(args.output / "synthetic-public-structure.arc3", attempt)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
