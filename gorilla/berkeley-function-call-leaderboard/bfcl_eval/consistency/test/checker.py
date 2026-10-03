"""One subprocess per solver check: no shared Z3 global context across threads."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from .common import load_callable, read_json, write_json


def check_isolated(steps, plugin, options, directory, timeout=120):
    directory = Path(directory)
    request, response = directory / "input.json", directory / "result.json"
    write_json(request, {"steps": steps, "plugin": plugin, "options": options})
    start = time.monotonic()
    try:
        process = subprocess.run(
            [sys.executable, "-m", "bfcl_eval.consistency.test.checker", str(request), str(response)],
            cwd=Path(__file__).resolve().parents[3], capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if process.returncode or not response.exists():
            result = {"status": "checker_error", "error": process.stderr[-3000:]}
        else:
            result = read_json(response)
    except subprocess.TimeoutExpired:
        result = {"status": "unknown", "reason": "checker_wall_timeout", "timeout_seconds": timeout}
    except Exception as exc:
        result = {"status": "checker_error", "error": f"{type(exc).__name__}: {exc}"}
    result["seconds"] = time.monotonic() - start
    write_json(response, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("response", type=Path)
    args = parser.parse_args()
    request = read_json(args.request)
    try:
        result = load_callable(request["plugin"])(request["steps"], **request["options"])
        if not isinstance(result, dict) or "status" not in result:
            raise ValueError("Checker must return a dictionary containing status")
    except Exception as exc:
        result = {"status": "checker_error", "error": f"{type(exc).__name__}: {exc}"}
    write_json(args.response, result)


if __name__ == "__main__":
    main()
