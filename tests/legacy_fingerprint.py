"""Reproducible legacy behavior digest; can run against an isolated base archive.

Usage: python tests/legacy_fingerprint.py ROOT OUTPUT.json
All 62 old IDs, seeds 0/42: all pure sy levels plus real SDK initial/action/reset
frames and detached old/new-level labels. No privileged original games are read.
"""
import hashlib
import json
from pathlib import Path
import sys


def fingerprints(root):
    sys.path.insert(0,str(root))
    from arc_agi import Arcade, OperationMode
    from arcengine import GameAction
    from games.skill_core import make_puzzle
    result={}
    arcade=Arcade(operation_mode=OperationMode.OFFLINE,environments_dir=str(root/'games'))
    card=arcade.open_scorecard()
    try:
        for game in ['sc01','si01']+[f'sy{i:02d}' for i in range(1,61)]:
            digest=hashlib.sha256()
            def record(value):
                digest.update(json.dumps(value,sort_keys=True,separators=(',',':')).encode())
            for seed in (0,42):
                if game.startswith('sy'):
                    for level in range(1,7):
                        p=make_puzzle(game,seed,level)
                        record([p.render().tolist(),p.get_training_info()])
                        for i in range(12):
                            if i==5: p.undo()
                            elif i==8: p.miss()
                            else: p.act(i%len(p.commands))
                            record([p.render().tolist(),p.get_training_info(),p.state,p.won,p.lost,
                                    [f.tolist() for f in p.transient_frames]])
                env=arcade.make(game+'-v1',seed=seed,scorecard_id=card)
                def frame(f):
                    record([f.state.name,f.levels_completed,f.win_levels,list(f.available_actions),[a.tolist() for a in f.frame]])
                    if hasattr(env._game,'get_training_info'): record(env._game.get_training_info())
                frame(env.observation_space)
                for i in range(8):
                    if i==6:
                        frame(env.reset()); continue
                    if hasattr(env._game,'puzzle'):
                        p=env._game.puzzle
                        x,y,w,h=p.button_rects()[i%len(p.commands)]
                        action,data=6,{'x':x+w//2,'y':y+h//2}
                    elif game=='sc01': action,data=(i%4)+1,None
                    else: action,data=6,{'x':16+i*4,'y':32}
                    frame(env.step(GameAction.from_id(action),data=data))
            result[game+'-v1']=digest.hexdigest()
    finally: arcade.close_scorecard(card)
    return result


if __name__=='__main__':
    root=Path(sys.argv[1]).resolve()
    Path(sys.argv[2]).write_text(json.dumps(fingerprints(root),indent=2)+'\n')
