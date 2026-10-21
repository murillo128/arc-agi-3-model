#!/usr/bin/env python3
"""List original synthetic curricula or inspect an explicitly requested teacher view."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from games.skill_core import SPECS, LATER_PRACTICE, make_puzzle

ACTION_LABELS = {1:'↑ / A1', 2:'↓ / A2', 3:'← / A3', 4:'→ / A4', 5:'Space / A5'}


def control_targets(puzzle):
    """Human-only direct actions or one actual-object hitbox per command."""
    targets = {command: dict(command=command, effect=puzzle.commands[command], action=action)
               for action,command in puzzle.action_commands().items()}
    for command,rect in puzzle.click_commands():
        targets.setdefault(command, dict(command=command, effect=puzzle.commands[command], action=6,
                                        rect=list(rect), center=[rect[0]+rect[2]//2,rect[1]+rect[3]//2]))
    return [targets[command] for command in sorted(targets)]


def control_descriptions(puzzle):
    descriptions = []
    for item in control_targets(puzzle):
        if item['action'] != 6:
            description = f'{ACTION_LABELS[item["action"]]}: {item["effect"]}'
        elif getattr(puzzle,'direct_controls',False):
            description = item['effect']
        else:
            description = f'{item["effect"]} · tap object ({item["center"][0]}, {item["center"][1]})'
        descriptions.append(description)
    return list(dict.fromkeys(descriptions))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--game',help='bf01-v1..bf08-v1, sy01-v1..sy60-v1, short ID, or skill slug')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--level',type=int,choices=range(1,7),default=1)
    parser.add_argument('--appearance-seed',type=int)
    parser.add_argument('--teacher-info',action='store_true',
                        help='Explicitly print privileged synthetic-only subgoal labels')
    args=parser.parse_args(argv)
    if args.teacher_info and not args.game:
        parser.error('--teacher-info requires --game')
    try:
        if args.game:
            puzzle=make_puzzle(args.game,args.seed,args.level,appearance_seed=args.appearance_seed)
            result=puzzle.get_training_info() if args.teacher_info else {
                **asdict(puzzle.spec),'level':puzzle.level,'seed':puzzle.seed,
                'controls':control_targets(puzzle),
            }
        else:
            result=[{**asdict(spec),'game_id':spec.game_id+'-v1','levels':6, 'later_practice':LATER_PRACTICE.get(spec.game_id, ())} for spec in SPECS]
        print(json.dumps(result,indent=2))
    except ValueError as error:
        parser.error(str(error))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
