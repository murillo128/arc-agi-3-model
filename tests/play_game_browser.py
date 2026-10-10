"""Optional real Chromium acceptance. Run with the repository's Python 3.12 venv."""

import argparse
import base64
from contextlib import contextmanager
from io import BytesIO
import json
from pathlib import Path

import socket
import subprocess
import sys
import tempfile
from threading import Thread
import time
from urllib.request import urlopen

from PIL import Image
from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
# The isolated SDK fixture imports the real launcher; no test endpoints or
# geometry are added to the product protocol.
sys.path.insert(0, str(ROOT))

from arcengine import GameAction
from flask import Flask
from werkzeug.serving import make_server

from arc3.envs.sdk import open_arcade
from scripts.play_game import PlaySession, web_routes
from square_cross_reference import reference_actions


def assert_native_pixels(page, data):
    pixels = page.evaluate("Array.from(document.querySelector('canvas').getContext('2d').getImageData(0,0,64,64).data)")
    palette = [[int(color[i:i+2], 16) for i in (1, 3, 5, 7)] for color in data['palette']]
    expected = [channel for row in data['frame'] for index in row for channel in palette[index]]
    assert pixels == expected, 'Canvas must render the actual SDK observation with official palette'


def assert_display_pixels(page, data):
    rendered = Image.open(BytesIO(page.locator('#game').screenshot())).convert('RGB')
    for y, row in enumerate(data['frame']):
        for x, index in enumerate(row):
            expected = tuple(int(data['palette'][index][i:i+2], 16) for i in (1, 3, 5))
            assert rendered.getpixel((int((x+.5)*rendered.width/64), int((y+.5)*rendered.height/64))) == expected


def selected(source, game):
    return json.dumps([source, game], separators=(',', ':'))


def capture(page, name):
    output = ROOT / 'artifacts' / 'square-cross' / f'{name}.png'
    output.parent.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(output), full_page=True)
    print(f'Screenshot: {output}')


def check_viewport(page, url, name):
    errors = []
    pending = set()
    max_pending = 0
    actions = []

    def started(request):
        nonlocal max_pending
        if request.url == url + "/play/action":
            actions.append(request.post_data_json["action"])
            pending.add(request)
            max_pending = max(max_pending, len(pending))
    page.on("request", started)
    page.on("response", lambda response: pending.discard(response.request))
    page.on("requestfailed", lambda request: pending.discard(request))
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
    # Hold the initial request to inspect the real connecting UI before a response.
    connecting = []
    page.route("**/play/state", lambda route: connecting.append(route))
    page.goto(url)
    expect(page.locator("#status")).to_have_text("Connecting…")
    expect(page.locator("#level")).to_have_text("LEVEL —")
    expect(page.locator("#up")).to_be_disabled()
    assert len(connecting) == 1
    connecting[0].continue_()
    page.unroute("**/play/state")
    page.wait_for_function("document.querySelector('#status').textContent.includes('NOT_FINISHED')")
    initial = page.request.get(url + "/play/state").json()
    assert initial["seed"] == 42

    def pixels():
        return page.evaluate("Array.from(document.querySelector('canvas').getContext('2d').getImageData(0,0,64,64).data)")

    def assert_canvas(data):
        assert_native_pixels(page, data)

    def bar_pixels(data):
        return sum(row.count(0) for row in data["frame"])

    def square(data):
        # Seed-42 level 1: find the 25-pixel solid object from observation only.
        counts = {i: sum(row.count(i) for row in data["frame"]) for i in range(6, 16)}
        color = next(i for i, count in counts.items() if count == 25)
        return {(x, y) for y, row in enumerate(data["frame"]) for x, value in enumerate(row) if value == color}

    def command(operation):
        with page.expect_response(url + "/play/action") as response:
            operation()
        data = response.value.json()
        page.wait_for_function("() => document.querySelector('#error').textContent === ''")
        page.evaluate("() => queue")
        assert_canvas(data)
        assert_badges(data)
        return data

    def assert_badges(data):
        expect(page.locator("#game-id")).to_have_value(json.dumps([data["source"], data["game"]], separators=(",", ":")))
        level = min(data["levels_completed"] + 1, data["win_levels"])
        expected = f"LEVEL {level} / {data['win_levels']}" if data["win_levels"] else "LEVEL —"
        expect(page.locator("#level")).to_have_text(expected)
        expect(page.locator("#status")).to_have_text(f"{data['game']} · Seed {data['seed']} · {data['state']}")

    def assert_layout():
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), "Horizontal overflow"
        geometry = page.locator("#game").evaluate("""canvas => {
            const r = canvas.getBoundingClientRect();
            return {width: r.width, height: r.height, native: [canvas.width, canvas.height],
                visible: r.left >= 0 && r.right <= innerWidth && r.top >= 0 && r.bottom <= innerHeight,
                corners: [[r.left+.5,r.top+.5], [r.right-.5,r.top+.5],
                    [r.left+.5,r.bottom-.5], [r.right-.5,r.bottom-.5]]
                    .every(([x,y]) => document.elementFromPoint(x,y) === canvas)};
        }""")
        assert geometry["native"] == [64, 64]
        assert abs(geometry["width"] - geometry["height"]) < .1
        assert geometry["visible"] and geometry["corners"], geometry
        for button in page.get_by_role("button", disabled=False).all():
            box = button.bounding_box()
            assert box and box["width"] >= 44 and box["height"] >= 44, button.inner_text()
            assert button.get_attribute("aria-label") or button.inner_text().strip()

    def screenshot(suffix):
        output = ROOT / "artifacts" / "square-cross" / f"{suffix}-{name}.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(output), full_page=True)
        print(f"Screenshot: {output}")

    assert_canvas(initial)
    expect(page.locator('#catalog-hint')).to_contain_text('No originals discovered')
    expect(page.locator('optgroup[label="Original ARC-AGI-3"] option')).to_be_disabled()
    assert bar_pixels(initial) == 180
    expect(page.locator('#game')).to_have_attribute('aria-label', '64 by 64 game observation')
    assert_badges(initial)
    assert_layout()
    # Check displayed pixels too: CSS clipping, overlays or tint must not change the observation.
    assert_display_pixels(page, initial)
    screenshot("after")
    action_count = len(actions)
    page.locator("#restart").focus()
    page.keyboard.press("Tab")
    expect(page.locator("#help")).to_be_focused()
    assert page.locator("#help").evaluate("e => getComputedStyle(e).outlineStyle") != "none"
    page.keyboard.press("Enter")
    expect(page.locator("#help-panel")).to_be_visible()
    expect(page.locator("#help")).to_have_attribute("aria-expanded", "true")
    expect(page.locator("#help-panel")).to_be_focused()
    expect(page.locator("#help-title")).to_be_in_viewport()
    page.keyboard.press("ArrowDown")  # Help content can scroll without moving the square.
    page.keyboard.press("Tab")
    expect(page.get_by_role("button", name="Close help")).to_be_focused()
    box = page.locator("#close-help").bounding_box()
    assert box["width"] >= 44 and box["height"] >= 44
    page.keyboard.press("Enter")
    expect(page.locator("#help-panel")).to_be_hidden()
    expect(page.locator("#help")).to_be_focused()
    page.locator("#help").tap()
    page.keyboard.press("Escape")
    expect(page.locator("#help-panel")).to_be_hidden()
    expect(page.locator("#click-control")).to_have_attribute("aria-disabled", "true")
    for label in ("SPACEBAR", "UNDO (Z)", "SELECT"):
        button = page.get_by_role("button", name=label, exact=True)
        expect(button).to_be_disabled()
        box = button.bounding_box()
        page.touchscreen.tap(box["x"] + box["width"]/2, box["y"] + box["height"]/2)
    page.locator("#help").evaluate("e => e.blur()")
    page.keyboard.press("z")
    page.keyboard.press("Space")
    page.evaluate("() => queue")
    assert len(actions) == action_count, "Help and unsupported controls must not submit gameplay"
    assert page.request.get(url + "/play/state").json() == initial
    page.evaluate("scrollTo(0,0)")
    original_pixels = pixels()
    scroll_before = page.evaluate("scrollY")
    moved = command(lambda: page.keyboard.press("ArrowRight"))
    assert page.evaluate("scrollY") == scroll_before
    assert actions[-1:] == ["ACTION4"]
    assert square(moved) == {(x + 5, y) for x, y in square(initial)}
    assert 0 < bar_pixels(moved) < bar_pixels(initial)
    before_tap = len(actions)
    touched = command(lambda: page.get_by_role("button", name="Down", exact=True).tap())
    assert square(touched) == {(x, y + 5) for x, y in square(moved)}
    assert 0 < bar_pixels(touched) < bar_pixels(moved)
    page.evaluate("() => queue")
    assert actions[before_tap:] == ["ACTION2"], "One tap must dispatch exactly once"
    # Burst input must preserve all commands in order, with no simultaneous requests.
    page.evaluate("() => { for (let i=0;i<3;i++) document.querySelector('#left').click(); }")
    page.evaluate("() => queue")
    burst = page.request.get(url + "/play/state").json()
    assert square(burst) == {(x - 15, y) for x, y in square(touched)}
    assert 0 < bar_pixels(burst) < bar_pixels(touched)
    assert actions[-3:] == ["ACTION3"] * 3
    assert max_pending == 1, "Browser action requests must be serialized"
    assert_canvas(burst)
    reset = command(lambda: page.get_by_role("button", name="Restart", exact=True).click())
    assert reset["frame"] == initial["frame"]
    assert pixels() == original_pixels
    # A reset in a mixed burst must stay between its neighbouring moves.
    action_count = len(actions)
    page.evaluate("() => { for (const id of ['left', 'restart', 'down']) document.getElementById(id).click(); }")
    page.evaluate("() => queue")
    mixed = page.request.get(url + "/play/state").json()
    assert actions[action_count:] == ["ACTION3", "RESET", "ACTION2"]
    assert square(mixed) == {(x, y + 5) for x, y in square(initial)}
    assert bar_pixels(mixed) == bar_pixels(moved)
    assert_canvas(mixed)
    command(lambda: page.get_by_role("button", name="Restart", exact=True).click())
    assert not errors, errors
    # Explicit simulated transport outage: only the error/recovery UI is intercepted.
    page.route("**/play/action", lambda route: route.abort())
    page.get_by_role("button", name="Up", exact=True).click()
    page.wait_for_function("document.querySelector('#error').textContent.includes('Restart')")
    expect(page.locator("#error")).to_be_visible()
    screenshot("error")
    page.unroute("**/play/action")
    errors.clear()  # Chromium's expected failed-fetch console line belongs to the injected outage.
    command(lambda: page.get_by_role("button", name="Restart", exact=True).click())
    assert not errors, errors
    # Real SDK depletion, terminal controls and reset; no game internals are inspected.
    previous = bar_pixels(initial)
    for _ in range(128):  # Contract cap, not a privileged budget in browser/API state.
        depleted = command(lambda: page.keyboard.press("ArrowDown"))  # Away from seed-42's cross.
        assert bar_pixels(depleted) < previous
        previous = bar_pixels(depleted)
        if depleted['state'] == 'GAME_OVER':
            break
    assert depleted["state"] == "GAME_OVER"
    assert bar_pixels(depleted) == 0
    expect(page.locator("#message")).to_contain_text("GAME_OVER")
    expect(page.locator("#message")).to_be_visible()
    for direction in ("Up", "Down", "Left", "Right"):
        expect(page.get_by_role("button", name=direction, exact=True)).to_be_disabled()
    screenshot("game-over")
    action_count = len(actions)
    page.keyboard.press("ArrowRight")
    page.evaluate("() => queue")
    assert len(actions) == action_count
    reset = command(lambda: page.get_by_role("button", name="Restart", exact=True).tap())
    assert reset["frame"] == initial["frame"]
    # Presentation-only response fixtures cover badge updates and WIN without adding a UI solver.
    # Real multi-level/win SDK lifecycle proof remains in test_square_cross.py.
    for completed, total, state in ((1, 8, "NOT_FINISHED"), (8, 8, "WIN"), (0, 0, "NOT_FINISHED")):
        fixture = dict(initial, levels_completed=completed, win_levels=total, state=state)
        page.route("**/play/state", lambda route: route.fulfill(json=fixture))
        page.reload()
        page.evaluate("() => queue")
        assert_badges(fixture)
        assert_canvas(fixture)
        if state == "WIN":
            expect(page.locator("#message")).to_contain_text("WIN")
            expect(page.locator("#message")).to_be_visible()
            expect(page.locator("#up")).to_be_disabled()
            screenshot("win")
        page.unroute("**/play/state")
    page.reload()
    page.evaluate("() => queue")
    # A smaller phone still has usable controls, even when vertical scrolling is needed.
    if name == "mobile":
        page.set_viewport_size({"width": 320, "height": 568})
        page.evaluate("scrollTo(0,0)")
        assert_layout()
        page.emulate_media(reduced_motion="reduce")
        scroll_before = page.evaluate("scrollY")
        with page.expect_response(url + "/play/action"):
            page.keyboard.down("ArrowDown")
        page.evaluate("() => queue")
        assert page.locator("#down").evaluate("e => getComputedStyle(e, '::before').transform") != "none"
        page.keyboard.up("ArrowDown")
        assert page.locator("#down").evaluate("e => getComputedStyle(e, '::before').transform") == "none"
        assert page.evaluate("scrollY") == scroll_before, "Gameplay arrows must not scroll a short page"
        command(lambda: page.get_by_role("button", name="Restart", exact=True).tap())
        command(lambda: page.get_by_role("button", name="Down", exact=True).tap())
        command(lambda: page.get_by_role("button", name="Restart", exact=True).tap())
    assert not errors, errors
    print(f"PASS {name}: exact SDK/display pixels, responsive geometry, focus, help, disabled controls, "
          "keyboard/touch, ordered bursts, decrements, reset, terminal/status and error recovery")


def check_switching(page, url, name):
    errors, writes, pending = [], [], set()
    max_pending = 0
    def started(request):
        nonlocal max_pending
        if request.url in (url + '/play/action', url + '/play/select'):
            writes.append((request.url.rsplit('/', 1)[1], request.post_data_json))
            pending.add(request)
            max_pending = max(max_pending, len(pending))
    page.on('request', started)
    page.on('response', lambda response: pending.discard(response.request))
    page.on('requestfailed', lambda request: pending.discard(request))
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.goto(url)
    page.evaluate('() => queue')
    picker = page.get_by_role('combobox', name='Choose game')
    expect(picker).to_be_enabled()
    assert picker.bounding_box()['height'] >= 44
    expect(page.locator('optgroup[label="Synthetic"] option')).to_have_text('sc01-v1 · Square Cross')
    expect(page.locator('optgroup[label="Original ARC-AGI-3"] option')).to_have_text('tp01-v1 · Input test fixture (not official)')
    # Native popup, opened by touch on mobile and keyboard on desktop.
    if name == 'mobile':
        picker.tap()
    else:
        picker.focus()
        page.keyboard.press('Alt+ArrowDown')
    assert picker.evaluate("element => element.matches(':open')"), 'Native game picker should open'
    page.keyboard.press('Escape')
    count = len(writes)
    page.keyboard.press('Space')
    page.keyboard.press('Escape')
    page.keyboard.press('z')
    page.evaluate('() => queue')
    assert len(writes) == count, 'Selector shortcuts must not reach the game'
    picker.evaluate('element => element.blur()')
    page.keyboard.press('ArrowRight')
    page.evaluate('() => queue')
    moved = page.request.get(url + '/play/state').json()
    picker.dispatch_event('change')  # Choosing current ID must not reset progress.
    page.evaluate('() => queue')
    assert page.request.get(url + '/play/state').json() == moved
    assert_native_pixels(page, moved)

    # A delayed action and a selection share one queue. Badge stays on the live board.
    held = []
    page.route('**/play/action', lambda route: held.append(route))
    with page.expect_request(url + '/play/action'):
        page.locator('#left').click()
    assert len(held) == 1
    picker.select_option(selected('original', 'tp01-v1'))
    expect(picker).to_have_value(selected('synthetic', 'sc01-v1'))
    expect(page.locator('#up')).to_be_disabled()
    held.pop().continue_()
    page.unroute('**/play/action')
    page.evaluate('() => queue')
    alternate = page.request.get(url + '/play/state').json()
    assert alternate['source'] == 'original' and alternate['game'] == 'tp01-v1'
    assert alternate['seed'] == 42 and alternate['available_actions'] == [5, 6, 7]
    assert_native_pixels(page, alternate)
    expect(picker).to_have_value(selected('original', 'tp01-v1'))
    expect(page.locator('#up')).to_be_disabled()
    expect(page.get_by_role('button', name='SPACEBAR', exact=True)).to_be_enabled()
    expect(page.get_by_role('button', name='UNDO (Z)', exact=True)).to_be_enabled()
    expect(page.locator('#click-control')).to_have_attribute('aria-disabled', 'false')
    expect(page.locator('#game-help')).not_to_contain_text('cross')
    capture(page, f'fixture-{name}')

    # Exact scaled edge cells and a single real SDK action per touch.
    canvas = page.locator('#game')
    box = canvas.bounding_box()
    before = len(writes)
    for x, y in ((0, 0), (63, 63)):
        px = box['x'] + (1 if x == 0 else box['width'] - 1)
        py = box['y'] + (1 if y == 0 else box['height'] - 1)
        if name == 'mobile':
            page.touchscreen.tap(px, py)
        else:
            page.mouse.click(px, py)
        page.evaluate('() => queue')
        data = page.request.get(url + '/play/state').json()
        assert writes[-1][1]['data'] == {'x': x, 'y': y}
        assert data['frame'][y][x] == 9
        assert_native_pixels(page, data)
    assert len(writes) == before + 2
    assert data['frame'][2][2] == 2, 'One tap must cause one SDK action'
    picker.evaluate('element => element.blur()')
    # A focused native RESET button uses Space to reset, not to send ACTION5.
    page.locator('#restart').focus()
    page.keyboard.press('Space')
    page.evaluate('() => queue')
    assert writes[-1][1]['action'] == 'RESET'
    page.locator('#restart').evaluate('element => element.blur()')
    # Restore the edge mark for the independent animation colour oracle.
    box = canvas.bounding_box()
    page.mouse.click(box['x'] + box['width'] - 1, box['y'] + box['height'] - 1)
    page.evaluate('() => queue')
    # Observe actual canvas paints through the real SDK two-frame action.
    page.evaluate("""() => {
        window.paintedColors = [];
        const context = document.querySelector('canvas').getContext('2d');
        const fill = context.fillRect.bind(context);
        context.fillRect = function(x,y,w,h) {
            if ((x === 63 && y === 63) || (x === 32 && y === 32)) window.paintedColors.push(this.fillStyle);
            return fill(x,y,w,h);
        };
    }""")
    before = len(writes)
    page.keyboard.press('Space')
    page.keyboard.press('z')
    page.evaluate('() => queue')
    assert [body['action'] for _, body in writes[before:]] == ['ACTION5', 'ACTION7']
    colors = page.evaluate('paintedColors')
    assert '#f93c31' in colors and '#4fcc30' in colors, colors
    assert colors.index('#f93c31') < colors.index('#4fcc30'), colors
    assert_native_pixels(page, page.request.get(url + '/play/state').json())

    # Cancel presentation on a queued reset or switch; no stale frames may repaint afterward.
    for target in ('reset', 'switch'):
        page.evaluate('paintedColors.length = 0')
        page.evaluate("document.querySelector('[data-action=ACTION5]').click()")
        page.wait_for_function("paintedColors.includes('#f93c31')")
        if target == 'reset':
            page.locator('#restart').click()
        else:
            picker.select_option(selected('synthetic', 'sc01-v1'))
        page.evaluate('() => queue')
        data = page.request.get(url + '/play/state').json()
        page.wait_for_timeout(150)
        assert_native_pixels(page, data)
        if target == 'switch':
            picker.select_option(selected('original', 'tp01-v1'))
            page.evaluate('() => queue')
        page.evaluate('paintedColors.length = 0')

    # Available actions can change within a game, and HELP consumes no actions.
    box = canvas.bounding_box()
    page.mouse.click(box['x'] + 1.5 * box['width']/64, box['y'] + 1.5 * box['height']/64)
    page.evaluate('() => queue')
    assert page.request.get(url + '/play/state').json()['available_actions'] == [6, 7]
    expect(page.get_by_role('button', name='SPACEBAR', exact=True)).to_be_disabled()
    count = len(writes)
    page.keyboard.press('Space')
    page.keyboard.press('ArrowUp')
    page.locator('#help').click()
    page.keyboard.press('z')
    page.keyboard.press('Space')
    page.keyboard.press('ArrowDown')
    page.keyboard.press('Escape')
    page.evaluate('() => queue')
    assert len(writes) == count

    # Real second tab changes the shared game; old queued actions and reset must not leak.
    other = page.context.browser.new_page()
    try:
        other.goto(url)
        other.evaluate('() => queue')
        other.locator('#game-id').select_option(selected('synthetic', 'sc01-v1'))
        other.evaluate('() => queue')
        shared = page.request.get(url + '/play/state').json()
        count = len(writes)
        page.evaluate("() => { document.querySelector('[data-action=ACTION7]').click(); document.querySelector('#restart').click(); }")
        page.evaluate('() => queue')
        assert len(writes) == count + 1, 'Conflict must discard pending reset'
        expect(page.locator('#error')).to_contain_text('Session changed')
        expect(picker).to_have_value(selected('synthetic', 'sc01-v1'))
        assert page.request.get(url + '/play/state').json() == shared
        assert_native_pixels(page, shared)
        page.locator('#down').click()
        page.evaluate('() => queue')
        expect(page.locator('#error')).to_be_empty()
    finally:
        other.close()

    # Selection transport failure preserves the live badge/board and recovers.
    previous = page.request.get(url + '/play/state').json()
    page.route('**/play/select', lambda route: route.fulfill(status=500, json={'error': 'Selection unavailable'}))
    picker.select_option(selected('original', 'tp01-v1'))
    page.evaluate('() => queue')
    expect(page.locator('#error')).to_contain_text('Selection unavailable')
    expect(picker).to_have_value(selected('synthetic', 'sc01-v1'))
    assert_native_pixels(page, previous)
    page.unroute('**/play/select')
    picker.select_option(selected('original', 'tp01-v1'))
    page.evaluate('() => queue')
    picker.select_option(selected('synthetic', 'sc01-v1'))
    page.evaluate('() => queue')
    assert max_pending == 1, 'Selection and actions must never be simultaneous'
    assert not errors, errors
    print(f'PASS {name}: native selector, source switching, action gating, scaled clicks/taps, animation, shared-tab conflict and load recovery')


def check_original(page, url, name):
    page.goto(url)
    page.evaluate('() => queue')
    catalog = page.request.get(url + '/play/games').json()
    originals = catalog['groups'][1]['games']
    ls20 = next((game for game in originals if game['id'].split('-')[0] == 'ls20'), None)
    assert ls20, 'Requested official smoke cache does not discover ls20'
    page.locator('#game-id').select_option(selected('original', ls20['id']))
    page.evaluate('() => queue')
    initial = page.request.get(url + '/play/state').json()
    assert initial['game'] == ls20['id'] and initial['source'] == 'original'
    assert initial['state'] == 'NOT_FINISHED' and initial['levels_completed'] == 0
    assert initial['available_actions']
    assert_native_pixels(page, initial)
    expect(page.locator('#game-help')).not_to_contain_text('square')
    capture(page, f'ls20-{name}')
    action = initial['available_actions'][0]
    with page.expect_response(url + '/play/action') as response:
        if action == 6:
            page.locator('#game').click(position={'x': 1, 'y': 1})
        else:
            page.locator(f'[data-action=ACTION{action}]').click()
    assert response.value.status == 200
    page.evaluate('() => queue')
    assert_native_pixels(page, response.value.json())
    page.locator('#game-id').select_option(selected('synthetic', 'sc01-v1'))
    page.evaluate('() => queue')
    returned = page.request.get(url + '/play/state').json()
    assert returned['source'] == 'synthetic' and returned['game'] == 'sc01-v1'
    assert_native_pixels(page, returned)
    print(f"PASS {name}: genuine offline {ls20['id']}, level 1/{initial['win_levels']}, actions {initial['available_actions']}, ACTION{action}, return to sc01-v1")


@contextmanager
def running_selection_server():
    """One real SDK seed, advanced through level 6 before starting the test UI."""
    arcade = open_arcade('offline', str(ROOT / 'games'))
    server = thread = None
    try:
        session = PlaySession({'synthetic': (arcade, arcade.open_scorecard())}, 'sc01-v1', 42)
        for level in range(1, 7):
            for action, data in reference_actions(session.env._game.scene, level):
                session.frame = session.env.step(GameAction.from_id(action), data=data)
        assert session.frame.levels_completed == 6
        app = Flask('square-cross-selection-test')
        web_routes(session)(arcade, app)
        server = make_server('127.0.0.1', 0, app)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        yield f'http://127.0.0.1:{server.server_port}', session
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=5)
        arcade.close_scorecard()


def check_selection(page, url, name, session):
    errors, writes = [], []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.on('console', lambda message: errors.append(message.text) if message.type == 'error' else None)
    page.on('request', lambda request: writes.append(request.post_data_json)
            if request.url == url + '/play/action' else None)
    page.goto(url)
    page.evaluate('() => queue')
    directions = {1: 'Up', 2: 'Down', 3: 'Left', 4: 'Right'}
    for level in (7, 8):
        expect(page.locator('#level')).to_have_text(f'LEVEL {level} / 8')
        expect(page.locator('#click-control')).to_have_attribute('aria-disabled', 'false')
        scene = session.env._game.scene
        for action, coordinates in reference_actions(scene, level):
            before = page.request.get(url + '/play/state').json()
            count = len(writes)
            with page.expect_response(url + '/play/action') as response:
                if action == 6:
                    box = page.locator('#game').bounding_box()
                    x, y = coordinates['x'], coordinates['y']
                    px, py = box['x'] + (x+.5)*box['width']/64, box['y'] + (y+.5)*box['height']/64
                    if name == 'mobile':
                        page.touchscreen.tap(px, py)
                    else:
                        page.mouse.click(px, py)
                else:
                    page.get_by_role('button', name=directions[action], exact=True).tap()
            assert response.value.status == 200
            data = response.value.json()
            page.evaluate('() => queue')
            assert len(writes) == count + 1
            assert writes[-1]['action'] == f'ACTION{action}'
            assert_native_pixels(page, data)
            if action == 6:
                assert writes[-1]['data'] == coordinates
                changed = {(x, y) for y in range(64) for x in range(64)
                           if before['frame'][y][x] != data['frame'][y][x]}
                x, y = coordinates['x'], coordinates['y']
                assert changed == {(x-1, y-1), (x, y-1), (x-1, y), (x, y)}
                assert all(data['frame'][y][x] == 0 for x, y in changed)
                assert data['palette'][0] == '#FFFFFFFF'
                assert_display_pixels(page, data)
                capture(page, f'selection-level-{level}-{name}')
                native = page.locator('#game').evaluate('canvas => canvas.toDataURL()')
                output = ROOT / 'artifacts/square-cross' / f'selection-level-{level}-{name}-native.png'
                output.write_bytes(base64.b64decode(native.split(',', 1)[1]))
    assert data['state'] == 'WIN' and data['levels_completed'] == data['win_levels'] == 8
    expect(page.locator('#message')).to_contain_text('WIN')
    assert not errors, errors
    print(f'PASS {name}: real synthetic levels 7–8, scaled selection, four white pixels, exact native/display frames, D-pad delivery and WIN')


@contextmanager
def running_server(*args):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    with tempfile.TemporaryFile(mode="w+") as log:
        server = subprocess.Popen([sys.executable, str(ROOT / "scripts/play_game.py"), "--mode", "web",
                                   "--seed", "42", "--port", str(port), *args], stdout=log, stderr=log)
        try:
            for _ in range(100):
                try:
                    with urlopen(url + "/api/healthcheck", timeout=1):
                        break
                except OSError:
                    if server.poll() is not None:
                        raise AssertionError("Web launcher exited before serving")
                    time.sleep(.1)
            else:
                raise AssertionError("Web launcher did not start")
            yield url
        except Exception:
            log.seek(0)
            print(log.read(), file=sys.stderr)
            raise
        finally:
            # SIGINT lets the launcher's finally block close its scorecard.
            import signal
            server.send_signal(signal.SIGINT)
            server.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--originals-dir', type=Path, help='Optional already provisioned public SDK cache for genuine ls20 smoke')
    args = parser.parse_args()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            with tempfile.TemporaryDirectory() as empty:
                lanes = [(empty, check_viewport), (str(ROOT / 'tests/fixtures/play_game'), check_switching)]
                if args.originals_dir:
                    lanes.append((str(args.originals_dir.resolve()), check_original))
                else:
                    print('NOT RUN: genuine ls20 smoke; supply --originals-dir with an authorized public SDK cache')
                for directory, check in lanes:
                    with running_server('--originals-dir', directory) as url:
                        for name, viewport in (("mobile", {"width": 390, "height": 844}),
                                               ("desktop", {"width": 1280, "height": 900})):
                            page = browser.new_page(viewport=viewport, has_touch=True)
                            check(page, url, name)
                            page.close()
                for name, viewport in (("mobile", {"width": 390, "height": 844}),
                                       ("desktop", {"width": 1280, "height": 900})):
                    with running_selection_server() as (url, session):
                        page = browser.new_page(viewport=viewport, has_touch=True)
                        check_selection(page, url, name, session)
                        page.close()
        finally:
            browser.close()


if __name__ == "__main__":
    main()
