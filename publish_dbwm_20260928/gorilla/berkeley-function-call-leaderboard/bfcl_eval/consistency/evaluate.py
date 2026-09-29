"""Evaluate TradingBot consistency detection on manually labeled JSONL traces.

Each line is {"steps": [...], "inconsistent": true/false}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bfcl_eval.consistency.trading_solver import check_trace


def evaluate_corpus(records: list[dict]) -> dict:
    counts = {
        "total": 0,
        "inconsistent": 0,
        "consistent": 0,
        "true_positive": 0,
        "false_negative": 0,
        "false_positive": 0,
        "true_negative": 0,
        "unsupported": 0,
        "unknown": 0,
    }
    details = []
    for index, record in enumerate(records, start=1):
        if not isinstance(record.get("inconsistent"), bool):
            raise ValueError(f"Record {index} needs a boolean inconsistent label")
        result = check_trace(record["steps"])
        label = record["inconsistent"]
        status = result["status"]
        counts["total"] += 1
        counts["inconsistent" if label else "consistent"] += 1
        if status in {"unsupported", "unknown"}:
            counts[status] += 1
        if label:
            counts["true_positive" if status == "unsat" else "false_negative"] += 1
        else:
            if status == "unsat":
                counts["false_positive"] += 1
            elif status == "sat":
                counts["true_negative"] += 1
        if status != ("unsat" if label else "sat"):
            details.append({"record": index, "inconsistent": label, **result})
    counts["detection_rate"] = (
        counts["true_positive"] / counts["inconsistent"]
        if counts["inconsistent"] else None
    )
    counts["false_positive_rate"] = (
        counts["false_positive"] / counts["consistent"]
        if counts["consistent"] else None
    )
    return {"metrics": counts, "nonmatching_results": details}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path, help="JSONL file of labeled traces")
    args = parser.parse_args()
    records = [json.loads(line) for line in args.corpus.read_text(encoding="utf-8-sig").splitlines()
               if line.strip()]
    print(json.dumps(evaluate_corpus(records), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
