"""Check grounded reference traces using observations alone and save results."""

import argparse
import hashlib
import json
import platform
import time
from collections import Counter
from pathlib import Path

import z3

from bfcl_eval.consistency.trading_solver_hard import check_trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grounded_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"python": platform.python_version(), "z3": z3.get_version_string(),
              "input": str(args.grounded_dir.resolve()), "cases": [], "source_sha256": {}}
    for path in Path(__file__).parent.glob("trading_*.py"):
        report["source_sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(args.grounded_dir.rglob("*.json")):
        if "_run_history" in path.parts or path.name == "manifest.json":
            continue
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if data.get("status") != "accepted" or "trace" not in data:
            continue
        # Never feed initial_state, branch labels or state snapshots to the solver.
        steps = [{"tool": step["tool"], "args": step["args"], "observation": step["result"]}
                 for step in data["trace"]]
        start = time.perf_counter()
        result = check_trace(steps)
        report["cases"].append({"source": str(path.relative_to(args.grounded_dir)),
                                "seconds": time.perf_counter() - start, **result})
        print(path.name, result["status"], flush=True)
    report["counts"] = dict(Counter(case["status"] for case in report["cases"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["counts"]))
    return 0 if report["cases"] and all(case["status"] == "sat" for case in report["cases"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
