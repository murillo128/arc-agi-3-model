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


def control_targets(puzzle):
    """Human-only addressing from the renderer's actual semantic hitboxes."""
    return [dict(command=i, effect=effect, rect=list(rect),
                 center=[rect[0]+rect[2]//2,rect[1]+rect[3]//2])
            for i,(effect,rect) in enumerate(zip(puzzle.commands,puzzle.button_rects()))]


def control_descriptions(puzzle):
    if getattr(puzzle,'direct_controls',False): return list(dict.fromkeys(puzzle.commands))
    return [f'{item["effect"]} · tap ({item["center"][0]}, {item["center"][1]})'
            for item in control_targets(puzzle)]


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
