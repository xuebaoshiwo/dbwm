"""Generate a reproducible grid with independently sampled targets per case."""

from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path

from bfcl_eval.consistency.data_generator_v2.generate_backward_trajectories import (
    BackwardSampler, DEFAULT_CATALOG, SamplingError, load_inputs, load_lifecycle_rules,
)
from bfcl_eval.consistency.data_generator_v2.generate_tool_state_specs import _write_json_atomic


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count-per-group", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--max-length", type=int, default=60)
    parser.add_argument("--attempts", type=int, default=200)
    parser.add_argument("--linked-distractors", action="store_true",
                        help="Fill shortfalls with independent backward-linked distractor chains")
    args = parser.parse_args(argv)
    if args.count_per_group < 1 or args.attempts < 1 or args.max_length < 21:
        parser.error("Require positive count/attempts and max-length >= 21")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory must be empty")
    specs, targets, refinements, specs_path = load_inputs(args.target_catalog)
    if len(targets) < 3:
        parser.error("The grid requires at least three candidate targets")
    rules, lifecycle_path = load_lifecycle_rules(args.target_catalog, specs, targets)
    chooser = random.Random(args.seed)
    manifest = {
        "status": "running", "seed": args.seed,
        "count_per_group": args.count_per_group, "requested": 9 * args.count_per_group,
        "accepted": 0, "failures": [], "candidate_targets": targets,
        "target_selection": "uniform_without_replacement_per_case; fixed_during_retries",
        "length_target_count_tiers": [{"min_length": n * 7, "target_count": n} for n in (1, 2, 3)],
        "min_write_tiers": [2, 3, 4], "max_length": args.max_length,
        "dependency_max_writes": 1, "attempts_per_case": args.attempts,
        "linked_distractors": args.linked_distractors,
        "filler_strategy": (["support_anchor", "linked_distractor", "distractor"]
                            if args.linked_distractors else ["support_anchor", "distractor"]),
        "write_count_policy": "minimum; shared writes may exceed; lifecycle shortfalls retained",
        "planning_only": True, "backend_execution": False,
        "input_sha256": {
            "catalog": hashlib.sha256(args.target_catalog.read_bytes()).hexdigest(),
            "specs": hashlib.sha256(specs_path.read_bytes()).hexdigest(),
            "lifecycle_rules": hashlib.sha256(lifecycle_path.read_bytes()).hexdigest() if lifecycle_path else None,
        },
        "groups": [],
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output / "manifest.json", manifest)
    for target_count in (1, 2, 3):
        min_length = 7 * target_count
        for min_writes in (2, 3, 4):
            group_path = Path(f"min_length_{min_length}") / f"min_writes_{min_writes}"
            group = {"directory": group_path.as_posix(), "min_length": min_length,
                     "target_count": target_count, "min_writes": min_writes, "cases": []}
            manifest["groups"].append(group)
            for index in range(1, args.count_per_group + 1):
                names = chooser.sample(list(targets), target_count)
                case_seed = chooser.randrange(2**63)
                case_id = f"l{min_length}_w{min_writes}_{index:03d}"
                sampler = BackwardSampler(specs, targets, reader_refinements=refinements,
                                          lifecycle_rules=rules, seed=case_seed)
                try:
                    result = sampler.build(names, max_writes=min_writes,
                                           min_length=min_length, max_length=args.max_length,
                                           attempts=args.attempts,
                                           linked_distractors=args.linked_distractors)
                except SamplingError as exc:
                    manifest["failures"].append({"id": case_id, "targets": names,
                                                 "seed": case_seed, "reason": str(exc)})
                    _write_json_atomic(output / "manifest.json", manifest)
                    continue
                result["id"] = case_id
                result["planning"]["seed"] = case_seed
                result["sampling_group"] = {"min_length": min_length, "target_count": target_count,
                                             "min_writes": min_writes, "batch_seed": args.seed}
                filename = group_path / f"{case_id}.json"
                _write_json_atomic(output / filename, result)
                chains = result["planning"]["target_chains"]
                group["cases"].append({
                    "id": case_id, "file": filename.as_posix(), "seed": case_seed,
                    "targets": names, "length": len(result["steps"]),
                    "distractor_count": result["planning"]["distractor_count"],
                    "distractor_chain_lengths": result["planning"]["distractor_chain_lengths"],
                    "write_counts": {name: chain["write_count"] for name, chain in chains.items()},
                    "write_shortfalls": {name: chain["write_shortfall"] for name, chain in chains.items()},
                })
                manifest["accepted"] += 1
                _write_json_atomic(output / "manifest.json", manifest)
                print(f"{case_id}: {len(result['steps'])} steps; {group['cases'][-1]['write_counts']}", flush=True)
    manifest["status"] = "partial" if manifest["failures"] else "complete"
    _write_json_atomic(output / "manifest.json", manifest)
    lines = [
        "# TradingBotHard raw tool chains", "",
        f"Generated {manifest['accepted']} / {manifest['requested']} symbolic plans. Batch seed: {args.seed}.", "",
        "Each case samples distinct targets uniformly from the catalog; targets stay fixed during retries.",
        "`holdings` monitors the complete map. Each group has its stated minimum length and write goal.",
        f"Maximum length: {args.max_length}. Dependency write maximum: 1.",
        f"Linked distractors enabled: {args.linked_distractors}. Chains fill available padding slots when feasible.",
        "Shared effects can exceed minimum writes. Order lifecycles may fall short of the goal; actual counts are below.",
        "These plans have no concrete arguments, initial states or backend execution.", "",
        "Regenerate from the repository root into an empty output directory:", "", "```powershell",
        "python -m bfcl_eval.consistency.data_generator_v2.generate_trajectory_grid "
        f"--count-per-group {args.count_per_group} --seed {args.seed} --max-length {args.max_length} "
        f"--attempts {args.attempts} " + ("--linked-distractors " if args.linked_distractors else "")
        + "--output-dir <empty-output-directory>", "```", "",
        "| Case | Targets | Actual length | Actual writes | Write shortfalls | Distractor chain lengths |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for group in manifest["groups"]:
        for case in group["cases"]:
            writes = ", ".join(f"{k}={v}" for k, v in case["write_counts"].items())
            shortfalls = ", ".join(f"{k}={v}" for k, v in case["write_shortfalls"].items() if v) or "none"
            lines.append(f"| [{case['id']}]({case['file']}) | {', '.join(case['targets'])} | "
                         f"{case['length']} | {writes} | {shortfalls} | {case['distractor_chain_lengths']} |")
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if manifest["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
