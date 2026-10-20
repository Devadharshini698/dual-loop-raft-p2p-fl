"""Launch the full Option B Dual-Loop Judge Observatory."""

from __future__ import annotations

import argparse
import threading
import webbrowser
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if not Path("frontend/dist/index.html").exists():
        raise SystemExit("Build the UI first: cd frontend && npm install && npm run build")
    try:
        import uvicorn
    except ImportError as error:
        raise SystemExit("Install the adapter: pip install -r requirements-showcase.txt") from error
    from src.fl.p2p_fl_task import flower_runtime_status

    flower = flower_runtime_status()
    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    print(f"Dual-Loop Judge Observatory ready at {url}")
    if flower["fallback"]:
        print(f"WARNING: {flower['message']}")
    else:
        print(f"Flower SDK {flower['version']} active with in-process FedAvg")
    uvicorn.run("server.app:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()

