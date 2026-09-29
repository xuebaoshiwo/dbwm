"""CLI for generating grounded consistency trajectories.

Run: python -m bfcl_eval.consistency.data_generator.generate --domain trading_bot
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from bfcl_eval.consistency.data_generator.core import Job, generate_many
from bfcl_eval.consistency.data_generator.trading import (
    REPO_ROOT,
    STATE_POOL,
    TradingAdapter,
    load_initial_states,
)
from bfcl_eval.consistency.initial_states_pools.trading_bot.trading_scenarios import SCENARIOS


DEFAULT_OUTPUT = REPO_ROOT / "bfcl_eval/consistency/data/trading_bot"
DEFAULT_CLIENT = REPO_ROOT.parent.parent / "utils/model_client.py"


def make_trading_jobs(
    lengths: list[int],
    read_gaps: list[int],
    write_counts: list[int],
    *,
    state_pool: Path = STATE_POOL,
    seed: int = 42,
) -> list[Job]:
    if not (len(lengths) == len(read_gaps) == len(write_counts)):
        raise ValueError("lengths, read-gaps, and write-counts need the same number of entries")
    states = load_initial_states(state_pool)
    if len(states) != len(SCENARIOS):
        raise ValueError(f"Expected {len(SCENARIOS)} initial states, found {len(states)}")
    jobs: list[Job] = []
    for tier, (length, read_gap, write_count) in enumerate(zip(lengths, read_gaps, write_counts)):
        for index, state in enumerate(states):
            jobs.append(
                Job(
                    scenario_name=SCENARIOS[index]["name"],
                    initial_state=state,
                    initial_state_source=f"{state_pool.as_posix()}#object-{index + 1}",
                    length=length,
                    read_gap=read_gap,
                    write_count=write_count,
                    target="account_info.balance" if (tier + index) % 2 == 0 else "watch_list",
                    seed=seed + tier * len(states) + index,
                )
            )
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=["trading_bot"], default="trading_bot")
    parser.add_argument("--lengths", type=int, nargs="+", default=[5, 8, 12])
    parser.add_argument("--read-gaps", type=int, nargs="+", default=[2, 5, 9])
    parser.add_argument("--write-counts", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--state-pool", type=Path, default=STATE_POOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-client", type=Path, default=DEFAULT_CLIENT)
    parser.add_argument("--model", default=None, help="Override model_client.DEFAULT_MODEL")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-fallback", action="store_true", help="Regenerate only missing or deterministic-fallback files")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.workers < 1 or args.retries < 1:
        parser.error("workers and retries must be positive")
    jobs = make_trading_jobs(
        args.lengths, args.read_gaps, args.write_counts,
        state_pool=args.state_pool, seed=args.seed,
    )
    if args.retry_fallback:
        def needs_retry(job: Job) -> bool:
            path = args.output_dir / f"trading_bot_{job.scenario_name}_l{job.length}.json"
            if not path.exists():
                return True
            existing = json.loads(path.read_text(encoding="utf-8"))
            return existing["generation"]["model_proposal"]["source"] != "model_client.chat_text"

        jobs = [job for job in jobs if needs_retry(job)]
        print(f"retrying {len(jobs)} missing/fallback trajectories", flush=True)
        if not jobs:
            return
    paths = generate_many(
        TradingAdapter(), jobs, args.output_dir, args.model_client,
        model=args.model, workers=args.workers, retries=args.retries,
    )
    sources = Counter(
        json.loads(path.read_text(encoding="utf-8"))["generation"]["model_proposal"]["source"]
        for path in paths
    )
    lengths = Counter(job.length for job in jobs)
    print(json.dumps({"generated": len(paths), "lengths": lengths, "proposal_sources": sources, "output_dir": str(args.output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
