"""On-demand baseline/candidate semantic replay and native visual evidence.

Run this file against an isolated git archive with --root. The compact fixture
must be captured from the accepted old revision, never refreshed from a candidate.
All generated images, native animation arrays and detailed digests stay ignored.
Test-only witnesses never enter the SDK's actor observations or production code.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

TESTS=Path(__file__).resolve().parent
ROOT=TESTS.parent
UNCHANGED={'sc01-v1','si01-v1','sy10-v1','sy47-v1','sy48-v1','sy55-v1','sy56-v1','sy57-v1'} | {f'bf{i:02}-v1' for i in range(1,9)}


def replay(root, output, images=False):
    sys.path.insert(0,str(root))
    from arc_agi import Arcade, OperationMode
    from arc_agi.rendering import COLOR_MAP
    from arcengine import GameAction, GameState
    from games.skill_core import SPECS, make_puzzle
    import numpy as np
    from foundation_reference import plan as basic_plan
    from skill_reference import plan as skill_plan
    from square_cross_reference import reference_actions, reference_path
    from PIL import Image, ImageDraw
    output.mkdir(parents=True,exist_ok=True)
    palette=np.array([[int(COLOR_MAP[i][j:j+2],16) for j in (1,3,5)] for i in range(16)],dtype=np.uint8)
    def sheet(frames,path,cols=6):
        image=Image.new('RGB',(cols*132,((len(frames)+cols-1)//cols)*148),'#333333')
        pen=ImageDraw.Draw(image)
        for i,(label,frame) in enumerate(frames):
            x,y=(i%cols)*132,(i//cols)*148
            pen.text((x,y),label,fill='white')
            image.paste(Image.fromarray(palette[frame]).resize((128,128),Image.Resampling.NEAREST),(x,y+16))
        image.save(path)
    specs={s.game_id+'-v1':s for s in SPECS}
    arcade=Arcade(operation_mode=OperationMode.OFFLINE,environments_dir=str(root/'games'))
    card=arcade.open_scorecard()
    records={}; completed=0; observed_frames=0
    try:
        ids=sorted(info.game_id for info in arcade.get_environments())
        assert set(ids)==set(specs)|{'sc01-v1','si01-v1'}
        for game in ids:
            logical=hashlib.sha256(); pixels=hashlib.sha256(); native=[]; gallery=[]
            def record(value): logical.update(json.dumps(value,sort_keys=True,separators=(',',':')).encode())
            for seed in (0,1,42,123):
                env=arcade.make(game,seed=seed,scorecard_id=card)
                def observe(frame):
                    nonlocal observed_frames
                    g=env._game
                    record([frame.state.name,frame.levels_completed,frame.win_levels,frame.available_actions])
                    if hasattr(g,'puzzle'): record([g.puzzle.state,g.puzzle.target,g.get_training_info()])
                    else: record([g.scene,g.remaining,[(p.x,p.y) for p in getattr(g,'players',[g.player])],
                                  sorted(getattr(g,'delivered',())),getattr(g,'selected',None),sorted(getattr(g,'click_selected',()))])
                    for f in frame.frame:
                        pixels.update(f.tobytes()); observed_frames+=1
                        if images: native.append(f.copy())
                    if hasattr(g,'puzzle'): np.testing.assert_array_equal(frame.frame[-1],g.puzzle.render())
                observe(env.observation_space)
                levels=6 if game in specs else 5
                for level in range(1,levels+1):
                    g=env._game
                    if game in specs:
                        p=g.puzzle
                        commands=(basic_plan if game.startswith('bf') else skill_plan)(p)
                        actions=[]
                        for command in commands:
                            if command in p.arrow_commands: actions.append((p.arrow_commands.index(command)+1,None,command))
                            else:
                                x,y,w,h=p.button_rects()[command]
                                actions.append((6,{'x':x+w//2,'y':y+h//2},command))
                    elif game.startswith('sc'):
                        actions=[(a,None,None) for a in reference_path(g.scene)]
                    else: actions=[(a,d,None) for a,d in reference_actions(g.scene,level)]
                    if images: gallery.append((f'{seed} L{level} initial',env.observation_space.frame[-1].copy()))
                    for index,(action,data,command) in enumerate(actions):
                        assert action in env.observation_space.available_actions
                        if game in specs:
                            # Every semantic alternative at every witnessed state,
                            # including blocked/fatal commands, not just the solve.
                            record([p.transition(p.state,i) for i in range(len(p.commands))])
                        result=env.step(GameAction.from_id(action),data=data)
                        assert result.state!=GameState.GAME_OVER,(game,seed,level)
                        observe(result)
                        if images and index==len(actions)//2:
                            gallery.append((f'{seed} L{level} progressed',(p.render() if game in specs else result.frame[-1]).copy()))
                    assert result.levels_completed==level,(game,seed,level)
                    completed+=1
                    if images: gallery.append((f'{seed} L{level} complete',(p.render() if game in specs else result.frame[-1]).copy()))
                assert result.state==GameState.WIN
                observe(env.reset())
                if game in specs:
                    p=env._game.puzzle
                    for i in range(16):
                        if i in (4,12): p.undo()
                        elif i==8: p.miss()
                        else: p.act(i%len(p.commands))
                        record([p.state,p.steps,p.history,p.won,p.lost,p.get_training_info()])
                        pixels.update(p.render().tobytes())
            # Cross appearance bounds at every level, without changing logic.
            if images and game in specs:
                for size,offset,reverse in ((3,0,False),(5,2,False),(3,2,True),(5,0,True)):
                    for level in range(1,7):
                        p=make_puzzle(game,42,level)
                        p.size=size; p.offset=(offset,offset)
                        if reverse: p.palette=p.palette[::-1]
                        gallery.append((f'L{level} size{size} off{offset}',p.render()))
                        for command in (basic_plan if game.startswith('bf') else skill_plan)(p):
                            p.act(command)
                            native.extend(p.transient_frames)
                        gallery.append((f'L{level} complete s{size}',p.render()))
            records[game]={'logic':logical.hexdigest(),'pixels':pixels.hexdigest()}
            if images:
                sheet(gallery,output/f'{game}.png')
                np.savez_compressed(output/f'{game}-native.npz',frames=np.stack(native))
            print(game,flush=True)
    finally: arcade.close_scorecard(card)
    result={'seeds':[0,1,42,123],'games':records,'completed_levels':completed,'native_frames':observed_frames}
    (output/'replay.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/symbols-v2/current')
    parser.add_argument('--images',action='store_true')
    parser.add_argument('--capture-baseline',type=Path)
    args=parser.parse_args()
    result=replay(args.root.resolve(),args.output.resolve(),args.images)
    if args.capture_baseline:
        args.capture_baseline.write_text(json.dumps(result,indent=2)+'\n')
    else:
        baseline=json.loads((TESTS/'fixtures/symbols-v1-behavior.json').read_text())
        assert baseline['seeds']==result['seeds']
        assert baseline['games'].keys()==result['games'].keys()
        for game,values in result['games'].items():
            assert values['logic']==baseline['games'][game]['logic'],f'Logical replay changed: {game}'
            if game in UNCHANGED: assert values['pixels']==baseline['games'][game]['pixels'],f'Unaffected pixels changed: {game}'
        print(f"PASS: {len(result['games'])} games, {result['completed_levels']} sequential SDK levels, {len(UNCHANGED)} unchanged pixel paths")


if __name__=='__main__': main()
