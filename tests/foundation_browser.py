"""Required desktop/touch acceptance for every foundation, with real native frames."""
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

from games.skill_core import FOUNDATION_SPECS, LATER_PRACTICE
from foundation_reference import plan
from skill_browser import server, settled, load, click_command
from play_game_browser import assert_native_pixels, assert_display_pixels

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'artifacts/synthetic-validation/foundations'


def command(page,session,index,mobile):
    p=session.env._game.puzzle
    if p.arrow_commands:
        action=p.arrow_commands.index(index)+1
        if mobile:
            page.locator(f'[data-action="ACTION{action}"]').tap()
        else:
            page.locator(f'[data-action="ACTION{action}"]').focus()
            page.keyboard.press(('ArrowUp','ArrowDown','ArrowLeft','ArrowRight')[action-1])
        settled(page)
        assert_native_pixels(page,session.payload())
    else:
        click_command(page,session,index,touch=mobile)


def check(page,url,session,mobile):
    page.goto(url); settled(page)
    assert session.selected['id']=='sc01-v1'
    catalog=page.request.get(url+'/play/games').json()
    assert catalog['groups'][0]['label']=='Synthetic · Foundations / Basics'
    assert len(catalog['groups'][0]['games'])==8
    assert len(catalog['groups'][1]['games'])==63
    debug_requests=[]
    page.on('request',lambda r: debug_requests.append(r.url) if '/play/debug/' in r.url else None)
    for spec in FOUNDATION_SPECS:
        game=spec.game_id+'-v1'
        load(page,'synthetic',game)
        assert session.frame.win_levels==6
        assert page.locator('#debug-enabled').is_checked() is False
        assert 'subgoals' not in page.request.get(url+'/play/state').json()
        assert 'numbers below' not in page.locator('#control-addressing').inner_text()
        assert not debug_requests
        page.locator('#help').click()
        expect(page.locator('#help-title')).to_contain_text(spec.title)
        assert 'A:' not in page.locator('#control-legend').inner_text()
        for later in LATER_PRACTICE[spec.game_id]:
            expect(page.locator('#later-practice')).to_contain_text(later)
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        page.locator('#close-help').click()
        page.locator('#debug-enabled').check(); page.evaluate('() => detailsQueue')
        for level in range(1,7):
            p=session.env._game.puzzle
            assert p.level==level
            expect(page.locator('#subgoals')).to_contain_text(f'{game} · Level {level}')
            if level in (1,6):
                assert_display_pixels(page,session.payload())
                page.locator('#game').screenshot(path=str(OUTPUT/f'{spec.game_id}-L{level}-{"mobile" if mobile else "desktop"}.png'))
            if level==6:
                # Actual Undo followed by same-seed current-level RESET.
                initial=session.payload()['frame']
                choices=[i for i in range(len(p.commands)) if not p.solved(p.transition(p.state,i))]
                command(page,session,choices[0],mobile)
                page.locator('[data-action="ACTION7"]').click(); settled(page)
                assert p.state==p.initial_state and p.steps==2
                page.locator('#restart').click(); settled(page)
                assert session.payload()['frame']==initial
                assert session.env._game.puzzle.steps==0
            for index in plan(session.env._game.puzzle):
                command(page,session,index,mobile)
            assert session.frame.levels_completed==level, (game,level,session.env._game.puzzle.state)
        assert session.frame.state.name=='WIN'
        page.locator('#restart').click(); settled(page)
        assert session.env._game.puzzle.level==1
        page.locator('#debug-enabled').uncheck(); page.evaluate('() => detailsQueue')
        debug_requests.clear()
    # A mismatch exposes both cards for >= half a second, then hides; every
    # painted frame must be a native SDK frame. Timing applies without reduced motion.
    load(page,'synthetic','bf08-v1')
    for _ in range(2):
        for index in plan(session.env._game.puzzle): command(page,session,index,mobile)
    p=session.env._game.puzzle
    command(page,session,0,mobile)
    second=next(i for i,v in enumerate(p.deck) if v!=p.deck[0])
    page.evaluate('() => {window.frameLog=[]; window.savedPaint=paint; paint=(frame,palette)=>{frameLog.push({frame,time:performance.now()}); savedPaint(frame,palette)}}')
    command(page,session,second,mobile)
    frames=page.evaluate('frameLog'); native=[f.tolist() for f in session.frame.frame]
    assert [f['frame'] for f in frames[:len(native)]]==native
    assert frames[8]['time']-frames[0]['time']>=500
    assert p.state==(0,-1,-1) and p.steps==2
    page.evaluate('() => {paint=savedPaint}')
    # Cancel pending reveal with RESET, using the existing queue/epoch guard.
    command(page,session,0,mobile)
    x,y,w,h=p.button_rects()[second]
    page.evaluate('() => {window.frameLog=[]; window.savedPaint=paint; paint=(frame,palette)=>{frameLog.push(frame); savedPaint(frame,palette)}}')
    page.locator('#game').scroll_into_view_if_needed()
    box=page.locator('#game').bounding_box()
    page.mouse.click(box['x']+(x+w/2)*box['width']/64,box['y']+(y+h/2)*box['height']/64)
    page.wait_for_function('frameLog.length > 0')
    page.locator('#restart').click(); settled(page)
    page.evaluate('() => {paint=savedPaint}')
    assert session.env._game.puzzle.steps==0
    assert_native_pixels(page,session.payload())
    for game in ('sy08-v1','sy26-v1','sc01-v1','si01-v1'):
        load(page,'synthetic',game)
        assert_native_pixels(page,session.payload())
        assert 'subgoals' not in session.payload()
        page.locator('#restart').click(); settled(page)
    print(f'Foundations: 8 sequential six-level WINs, controls/HELP/Undo/reset/privacy/memory/cancellation ({"mobile" if mobile else "desktop"})',flush=True)



def check_reduced_motion_pairs(page,url,session,mobile):
    page.emulate_media(reduced_motion='reduce')
    page.goto(url); settled(page)
    load(page,'synthetic','bf08-v1')
    for _ in range(2):
        for index in plan(session.env._game.puzzle): command(page,session,index,mobile)
    p=session.env._game.puzzle
    first=0; second=next(i for i,v in enumerate(p.deck) if v!=p.deck[first])
    command(page,session,first,mobile)
    page.evaluate('() => {window.frameLog=[]; window.savedPaint=paint; paint=(frame,palette)=>{frameLog.push({frame,time:performance.now()}); savedPaint(frame,palette)}}')
    command(page,session,second,mobile)
    frames=page.evaluate('frameLog')
    # Independent visible-card expectation: both centres contain their face
    # colours, not the common back. Native frames alone aren't the oracle.
    centers=[(x+w//2,y+h//2) for x,y,w,h in p.button_rects()]
    def both_faces(frame):
        return all(frame[centers[i][1]][centers[i][0]]==p.palette[p.deck[i]] for i in (first,second))
    reveals=[i for i,item in enumerate(frames) if both_faces(item['frame'])]
    assert reveals,'Reduced motion discarded the second card identity on a mismatch'
    start=reveals[0]
    hidden=next(i for i in range(start+1,len(frames)) if not both_faces(frames[i]['frame']))
    assert frames[hidden]['time']-frames[start]['time']>=500
    assert all(item['frame']==frames[start]['frame'] for item in frames[start:hidden]),'Reveal must be still'
    native=[f.tolist() for f in session.frame.frame]
    assert all(item['frame'] in native for item in frames),'Presentation must use only native actor frames'
    assert p.state==(0,-1,-1) and p.steps==2
    assert_native_pixels(page,session.payload())
    assert not {'deck','target','subgoals','training_info'} & session.payload().keys()
    page.evaluate('() => {paint=savedPaint}')

    # Cancel the still hold through actual inputs; queued actions keep their
    # existing epoch/animation guards, and no delayed reveal may repaint later.
    for cancel in ('reset','switch'):
        if cancel=='reset':
            page.locator('#restart').click(); settled(page)
        p=session.env._game.puzzle
        command(page,session,first,mobile)
        page.evaluate('() => {window.frameLog=[]; window.savedPaint=paint; paint=(frame,palette)=>{frameLog.push({frame,time:performance.now()}); savedPaint(frame,palette)}}')
        x,y,w,h=p.button_rects()[second]
        page.locator('#game').scroll_into_view_if_needed()
        box=page.locator('#game').bounding_box()
        (page.touchscreen.tap if mobile else page.mouse.click)(box['x']+(x+w/2)*box['width']/64,box['y']+(y+h/2)*box['height']/64)
        page.wait_for_function('frameLog.length > 0')
        if cancel=='reset':
            page.locator('#restart').click(); settled(page)
            assert session.env._game.puzzle.steps==0
            assert session.env._game.puzzle.level==3
        else:
            load(page,'synthetic','sy08-v1')
            assert session.selected['id']=='sy08-v1'
        count=page.evaluate('frameLog.length')
        page.wait_for_timeout(700)
        assert page.evaluate('frameLog.length')==count,'Cancelled reveal repainted a newer session'
        assert_native_pixels(page,session.payload())
        page.evaluate('() => {paint=savedPaint}')
    page.emulate_media(reduced_motion='no-preference')
    print(f'PASS reduced-motion pairs: still mismatch reveal, one-action cost, privacy, RESET/switch cancellation ({"mobile" if mobile else "desktop"})',flush=True)


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--originals-dir',type=Path)
    args=parser.parse_args()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    with sync_playwright() as pw:
        browser=pw.chromium.launch()
        try:
            for mobile in (False,True):
                with server(args.originals_dir) as (url,session):
                    page=browser.new_page(viewport={'width':390,'height':844} if mobile else {'width':1280,'height':900},has_touch=True)
                    errors=[]
                    page.on('pageerror',lambda e: errors.append(str(e)))
                    check(page,url,session,mobile)
                    check_reduced_motion_pairs(page,url,session,mobile)
                    assert not errors,errors
                    page.close()
        finally: browser.close()


if __name__=='__main__': main()
