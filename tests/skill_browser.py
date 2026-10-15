"""Required real Chromium catalog/context acceptance; no mocked game responses.

Test-only witnesses inspect synthetic state; every interaction enters the real
play protocol and SDK. Never import this module into an agent or production.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
from threading import Thread

from flask import Flask
from playwright.sync_api import expect, sync_playwright
from werkzeug.serving import make_server

from arc3.envs.sdk import open_arcade
from games.skill_core import SPECS
from scripts.play_game import PlaySession, web_routes
from play_game_browser import assert_native_pixels, assert_display_pixels, selected, capture
from skill_reference import plan

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def server(originals=None):
    with tempfile.TemporaryDirectory() as empty:
        sources = {}
        for name,path in (('synthetic',ROOT/'games'),('original', originals or empty)):
            arcade = open_arcade('offline',str(path))
            sources[name] = arcade,arcade.open_scorecard()
        session = PlaySession(sources,'sc01-v1',42,debug_subgoals=True)
        app = Flask(__name__)
        web_routes(session)(sources['synthetic'][0],app)
        http = make_server('127.0.0.1',0,app,threaded=True)
        thread = Thread(target=http.serve_forever,daemon=True); thread.start()
        try:
            yield f'http://127.0.0.1:{http.server_port}',session
        finally:
            http.shutdown(); thread.join(); http.server_close()
            for arcade,card in sources.values(): arcade.close_scorecard(card)


def settled(page):
    page.evaluate('() => queue')
    expect(page.locator('#error')).to_be_empty()


def load(page, source, game):
    page.locator('#game-id').select_option(selected(source,game))
    settled(page)


def click_command(page,session,command,touch=False):
    x,y,w,h = session.env._game.puzzle.button_rects()[command]
    page.locator('#game').scroll_into_view_if_needed()
    box = page.locator('#game').bounding_box()
    # Native pointer/touch conversion, including noninteger CSS scaling.
    px=box['x']+(x+w/2)*box['width']/64
    py=box['y']+(y+h/2)*box['height']/64
    (page.touchscreen.tap if touch else page.mouse.click)(px,py)
    settled(page)
    assert_native_pixels(page,session.payload())


def check_catalog(page,url,session,mobile):
    page.goto(url); settled(page)
    catalog = page.request.get(url+'/play/games').json()
    games = catalog['groups'][0]['games']
    expected = {'sc01-v1','si01-v1'} | {s.game_id+'-v1' for s in SPECS}
    assert {g['id'] for g in games} == expected and len(games)==len(expected)
    debug_requests=[]
    page.on('request',lambda r: debug_requests.append(r.url) if '/play/debug/' in r.url else None)
    for item in games:
        load(page,'synthetic',item['id'])
        initial=session.payload()
        assert_native_pixels(page,initial)
        assert not page.locator('#debug-enabled').is_checked()
        if item['id'].startswith('sy'):
            help_text=page.locator('#game-help').inner_text()
            assert 'Explore the game' not in help_text and len(help_text)>50
            assert page.locator('#control-legend li').count()==len(session.env._game.puzzle.commands)
            click_command(page,session,0,touch=mobile)
        elif item['id']=='sc01-v1':
            page.keyboard.press('ArrowRight'); settled(page)
        else:
            x,y=session.env._game.scene.players[0][0]
            box=page.locator('#game').bounding_box()
            page.mouse.click(box['x']+(x+.5)*box['width']/64,box['y']+(y+.5)*box['height']/64)
            settled(page)
        page.locator('#restart').click(); settled(page)
        assert_native_pixels(page,session.payload())
    assert not debug_requests,'Teacher labels fetched without opt-in'
    print(f'Catalog: {len(games)} real loads/actions/resets ({"mobile" if mobile else "desktop"})')


def check_contexts(page,url,session,mobile):
    representatives={}
    for spec in SPECS: representatives.setdefault(spec.context,spec.game_id+'-v1')
    representatives['linkage']='sy26-v1'
    representatives['flow']='sy43-v1'
    representatives['program']='sy54-v1'
    for context,game in representatives.items():
        load(page,'synthetic',game)
        page.locator('#debug-enabled').check()
        page.evaluate('() => detailsQueue')
        expect(page.locator('#subgoals')).to_contain_text(f'{game} · Level 1')
        page.locator('#help').click()
        expect(page.locator('#help-title')).to_be_in_viewport()
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        assert 'Space:' in page.locator('#keyboard-help').inner_text() if session.env._game.puzzle.run_command is not None else 'Space is unavailable' in page.locator('#keyboard-help').inner_text()
        page.locator('#close-help').click()
        page.evaluate('scrollTo(0,0)')
        for level in range(1,7):
            p=session.env._game.puzzle
            assert p.level==level
            if level==6:
                if context=='ghost':
                    expect(page.locator('[data-goal=exit]')).to_contain_text('Dependencies: plate-0, plate-1')
                assert_display_pixels(page,session.payload())
                capture(page,f'skill-{context}-advanced-{"mobile" if mobile else "desktop"}')
            commands=plan(p)
            for command in commands:
                click_command(page,session,command,touch=mobile and level in (1,6))
            assert session.frame.levels_completed==level
            expect(page.locator('#subgoals')).to_contain_text(f'Level {min(level+1,6)}')
        assert session.frame.state.name=='WIN'
        page.locator('#restart').click(); settled(page)
        expect(page.locator('#subgoals')).to_contain_text('Level 1')
        assert session.env._game.puzzle.steps==0
        page.locator('#debug-enabled').uncheck(); page.evaluate('() => detailsQueue')
    print(f'Contexts: {len(representatives)} sequential browser WINs, levels 1–6 ({"mobile" if mobile else "desktop"})')


def check_reversible_and_animation(page,url,session):
    load(page,'synthetic','sy26-v1')
    for command in plan(session.env._game.puzzle): click_command(page,session,command)
    page.locator('#debug-enabled').check(); page.evaluate('() => detailsQueue')
    goal=page.locator('[data-goal="target-0"]')
    expect(goal).to_have_attribute('data-achieved','true')
    click_command(page,session,0)
    expect(goal).to_have_attribute('data-achieved','false')
    page.locator('#game').click(position={'x':1,'y':1})  # Miss costs, no state change.
    settled(page)
    page.locator('#game').evaluate('e=>e.focus()')
    page.keyboard.press('z'); settled(page)
    expect(goal).to_have_attribute('data-achieved','true')
    page.locator('#restart').click(); settled(page)
    expect(page.locator('#subgoals')).to_contain_text('0 /')
    for game in ('sy55-v1','sy51-v1','sy43-v1'):
        load(page,'synthetic',game)
        for _ in range(2):
            for command in plan(session.env._game.puzzle): click_command(page,session,command)
        p=session.env._game.puzzle
        if game=='sy55-v1':
            click_command(page,session,0)
            command=next(i for i,v in enumerate(p.deck) if v!=p.deck[0])
        else:
            for command in plan(p)[:-1]: click_command(page,session,command)
            command=p.run_command
        page.evaluate('() => {window.frameLog=[]; window.savedPaint=paint; paint=(frame,palette)=>{frameLog.push(frame); savedPaint(frame,palette)}}')
        old_steps=p.steps
        click_command(page,session,command)
        expected=[f.tolist() for f in session.frame.frame]
        frames=page.evaluate('frameLog')
        assert len(expected)>1 and frames[:len(expected)]==expected
        assert p.steps==old_steps+1
        page.evaluate('() => {paint=savedPaint}')
        # Repeat a real multi-frame action with reduced motion; RESET alone
        # would not test suppression of transient animation frames.
        page.emulate_media(reduced_motion='reduce')
        page.locator('#restart').click(); settled(page)
        p=session.env._game.puzzle
        if game=='sy55-v1':
            click_command(page,session,0)
            command=next(i for i,v in enumerate(p.deck) if v!=p.deck[0])
        else:
            for command in plan(p)[:-1]: click_command(page,session,command)
            command=p.run_command
        page.evaluate('() => {window.frameLog=[]; window.savedPaint=paint; paint=(frame,palette)=>{frameLog.push(frame); savedPaint(frame,palette)}}')
        click_command(page,session,command)
        assert len(session.frame.frame)>1
        assert all(f==session.payload()['frame'] for f in page.evaluate('frameLog'))
        page.evaluate('() => {paint=savedPaint}')
        page.emulate_media(reduced_motion='no-preference')
    # Cancel a real program animation by queuing reset; labels and canvas must
    # refer only to the retry when the queue drains.
    load(page,'synthetic','sy54-v1')
    for _ in range(5):
        for command in plan(session.env._game.puzzle): click_command(page,session,command)
    for command in plan(session.env._game.puzzle)[:-1]: click_command(page,session,command)
    page.locator('#debug-enabled').check(); page.evaluate('() => detailsQueue')
    page.evaluate('() => {window.frameLog=[]; window.savedPaint=paint; paint=(frame,palette)=>{frameLog.push(frame); savedPaint(frame,palette)}}')
    page.locator('[data-action="ACTION5"]').click()
    page.wait_for_function('frameLog.length > 0')
    page.locator('#restart').click(); settled(page)
    page.evaluate('() => {paint=savedPaint}')
    assert_native_pixels(page,session.payload())
    expect(page.locator('#subgoals')).to_contain_text('Level 1')  # WIN reset
    load(page,'synthetic','sc01-v1')
    expect(page.locator('#debug-enabled')).not_to_be_checked()
    expect(page.locator('#debug-enabled')).to_be_disabled()
    expect(page.locator('#subgoals')).to_be_hidden()
    print('Real memory/program/flow frames, reversible labels, reduced motion and cancellation verified')


def check_originals(page,url,session):
    originals=[g for g in session.games if g['source']=='original']
    if not originals:
        print('UNAVAILABLE optional original-cache lane: no authorized games installed')
        return
    for game in originals:
        load(page,'original',game['id'])
        assert_native_pixels(page,session.payload())
        expect(page.locator('#debug-enabled')).to_be_disabled()
        data=session.payload()
        assert 'subgoals' not in data
        action=next(a for a in data['available_actions'] if 1<=a<=7)
        if action==6:
            page.locator('#game').click(position={'x':10,'y':10})
        else: page.locator(f'[data-action="ACTION{action}"]').click()
        settled(page)
        page.locator('#restart').click(); settled(page)
        assert_native_pixels(page,session.payload())
        load(page,'synthetic','sy01-v1')
    print(f'Original cache: {len(originals)} discovery/load/legal-action/reset/source-switch smokes')


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--originals-dir',type=Path)
    args=parser.parse_args()
    with sync_playwright() as pw:
        browser=pw.chromium.launch()
        try:
            for mobile in (False,True):
                with server(args.originals_dir) as (url,session):
                    page=browser.new_page(viewport={'width':390,'height':844} if mobile else {'width':1280,'height':900},has_touch=True)
                    errors=[]
                    page.on('pageerror',lambda error: errors.append(str(error)))
                    page.on('console',lambda msg: errors.append(msg.text) if msg.type=='error' else None)
                    page.on('requestfailed',lambda r: errors.append(r.url))
                    page.on('response',lambda r: errors.append(f'{r.status} {r.url}') if r.status>=400 else None)
                    check_catalog(page,url,session,mobile)
                    check_contexts(page,url,session,mobile)
                    check_reversible_and_animation(page,url,session)
                    check_originals(page,url,session)
                    assert not errors,errors
                    page.close()
        finally: browser.close()


if __name__=='__main__': main()
