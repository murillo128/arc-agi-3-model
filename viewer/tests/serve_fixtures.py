"""Local-only static fixture server for reproducible cross-origin acceptance."""

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default=".fixtures")
    parser.add_argument("--port", type=int, default=4178)
    parser.add_argument("--cors-origin", default="http://127.0.0.1:4177")
    args = parser.parse_args()

    class Handler(SimpleHTTPRequestHandler):
        def end_headers(self):
            self.send_header("Access-Control-Allow-Origin", args.cors_origin)
            super().end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, directory=args.directory))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
