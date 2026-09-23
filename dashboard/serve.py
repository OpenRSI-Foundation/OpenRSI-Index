#!/usr/bin/env python3
"""Build the local RSI archive dashboard and serve it on localhost."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "tools"))
from build import build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--skip-build", action="store_true", help="Reuse the last generated archive snapshot")
    args = parser.parse_args()
    data = ROOT / "assets/logs-data"
    if not args.skip_build or not (data / "index.json").exists():
        build(ROOT.parent, data)
    handler = partial(SimpleHTTPRequestHandler, directory=str(ROOT))
    with ThreadingHTTPServer((args.host, args.port), handler) as server:
        print(f"\nRSI Logs dashboard: http://{args.host}:{server.server_port}/", flush=True)
        print("Press Ctrl+C to stop. Restart to refresh the archive snapshot.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
