"""Optional real Chromium acceptance. Run with the repository's Python 3.12 venv."""

from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.request import urlopen

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


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
                    page = browser.new_page(viewport={"width": 720, "height": 1000}, has_touch=True)
                    errors = []
                    pending = set()
                    max_pending = 0
                    def started(request):
                        nonlocal max_pending
                        if request.url == url + "/play/action":
                            pending.add(request)
                            max_pending = max(max_pending, len(pending))
                    page.on("request", started)
                    page.on("response", lambda response: pending.discard(response.request))
                    page.on("pageerror", lambda error: errors.append(str(error)))
                    page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
                    page.goto(url)
                    page.wait_for_function("document.querySelector('#status').textContent.includes('NOT_FINISHED')")
                    initial = page.request.get(url + "/play/state").json()
                    assert initial["seed"] == 42

                    def pixels():
                        return page.evaluate("Array.from(document.querySelector('canvas').getContext('2d').getImageData(0,0,64,64).data)")

                    def assert_canvas(data):
                        palette = [[int(color[i:i+2], 16) for i in (1, 3, 5, 7)] for color in data["palette"]]
                        expected = [channel for row in data["frame"] for index in row for channel in palette[index]]
                        assert pixels() == expected, "Canvas must render the actual SDK observation with official palette"

                    def dots(data):
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
                        # Response dispatch and painting finish on the next browser task.
                        page.evaluate("() => new Promise(resolve => requestAnimationFrame(resolve))")
                        assert_canvas(data)
                        return data

                    assert_canvas(initial)
                    original_pixels = pixels()
                    moved = command(lambda: page.keyboard.press("ArrowRight"))
                    assert square(moved) == {(x + 1, y) for x, y in square(initial)}
                    assert dots(moved) == dots(initial) - 1
                    touched = command(lambda: page.get_by_role("button", name="Down", exact=True).tap())
                    assert square(touched) == {(x, y + 1) for x, y in square(moved)}
                    assert dots(touched) == dots(initial) - 2
                    # Burst input must preserve all commands in order, with no simultaneous requests.
                    page.evaluate("() => { for (let i=0;i<3;i++) document.querySelector('#left').click(); }")
                    page.evaluate("() => queue")
                    burst = page.request.get(url + "/play/state").json()
                    assert square(burst) == {(x - 3, y) for x, y in square(touched)}
                    assert dots(burst) == dots(initial) - 5
                    assert max_pending == 1, "Browser action requests must be serialized"
                    assert_canvas(burst)
                    reset = command(lambda: page.get_by_role("button", name="Restart", exact=True).click())
                    assert reset["frame"] == initial["frame"]
                    assert pixels() == original_pixels
                    assert not errors, errors
                    # Explicit simulated transport outage: only the error/recovery UI is intercepted.
                    page.route("**/play/action", lambda route: route.abort())
                    page.get_by_role("button", name="Up", exact=True).click()
                    page.wait_for_function("document.querySelector('#error').textContent.includes('Restart')")
                    page.unroute("**/play/action")
                    errors.clear()  # Chromium's expected failed-fetch console line belongs to the injected outage.
                    command(lambda: page.get_by_role("button", name="Restart", exact=True).click())
                    assert not errors, errors
                    output = ROOT / "artifacts" / "square-cross-browser.png"
                    output.parent.mkdir(exist_ok=True)
                    page.screenshot(path=str(output), full_page=True)
                    print("PASS: real SDK canvas, keyboard, touch, burst inputs, dot decrements, reset, error recovery; no unexpected JS errors")
                    print(f"Screenshot: {output}")
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
