"""Run a real-socket smoke test before deploying to separate lab machines."""

from __future__ import annotations

import json

from physical_testbed.demo import run_local_socket_smoke


def main() -> None:
    result = run_local_socket_smoke("results")
    print("Physical adapter socket smoke test")
    print("----------------------------------")
    print(json.dumps(result, indent=2))
    print("\nThis proves the real UDP adapter, not a multi-machine physical result.")


if __name__ == "__main__":
    main()
