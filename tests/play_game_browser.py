"""Optional real Chromium acceptance. Run with the repository's Python 3.12 venv."""

from io import BytesIO
from pathlib import Path

import socket
import subprocess
import sys
import tempfile
import time
from urllib.request import urlopen

from PIL import Image
from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]


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
        palette = [[int(color[i:i+2], 16) for i in (1, 3, 5, 7)] for color in data["palette"]]
        expected = [channel for row in data["frame"] for index in row for channel in palette[index]]
        assert pixels() == expected, "Canvas must render the actual SDK observation with official palette"

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
        expect(page.locator("#game-id")).to_have_text(data["game"])
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
    assert bar_pixels(initial) == 180
    expect(page.locator('#game')).to_have_attribute('aria-label', 'Game observation, including remaining-moves bar')
    assert_badges(initial)
    assert_layout()
    # Check displayed pixels too: CSS clipping, overlays or tint must not change the observation.
    rendered = Image.open(BytesIO(page.locator("#game").screenshot())).convert("RGB")
    for y, row in enumerate(initial["frame"]):
        for x, index in enumerate(row):
            expected = tuple(int(initial["palette"][index][i:i+2], 16) for i in (1, 3, 5))
            assert rendered.getpixel((int((x+.5)*rendered.width/64), int((y+.5)*rendered.height/64))) == expected
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
    for label in ("SPACEBAR", "CLICK", "UNDO (Z)", "SELECT"):
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
    for completed, total, state in ((1, 5, "NOT_FINISHED"), (5, 5, "WIN"), (0, 0, "NOT_FINISHED")):
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


def main():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    with tempfile.TemporaryFile(mode="w+") as log:
        server = subprocess.Popen([sys.executable, str(ROOT / "scripts/play_game.py"), "--mode", "web",
                                   "--seed", "42", "--port", str(port)], stdout=log, stderr=log)
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
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch()
                try:
                    for name, viewport in (("mobile", {"width": 390, "height": 844}),
                                           ("desktop", {"width": 1280, "height": 900})):
                        page = browser.new_page(viewport=viewport, has_touch=True)
                        check_viewport(page, url, name)
                        page.close()
                finally:
                    browser.close()
        except Exception:
            log.seek(0)
            print(log.read(), file=sys.stderr)
            raise
        finally:
            # SIGINT lets the launcher's finally block close its scorecard.
            import signal
            server.send_signal(signal.SIGINT)
            server.wait(timeout=10)


if __name__ == "__main__":
    main()
