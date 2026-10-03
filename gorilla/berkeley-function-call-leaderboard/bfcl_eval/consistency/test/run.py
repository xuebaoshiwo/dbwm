"""Run endpoint (teacher-forced) or complete autoregressive WM experiments."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from .checker import check_isolated
from .client import ChatClient
from .common import digest, load_tools, read_json, write_json
from .dataset import load_jobs, observation_step
from .prompts import build_messages


def run_job(job, *, mode, client, task, tools, examples, checker, checker_options,
            checker_timeout, directory, resume=False):
    directory = Path(directory)
    output = directory / "case.json"
    if resume and output.exists():
        record = read_json(output)
        if record.get("complete"):
            return record
    else:
        record = {"source": job["source"], "target": job["target"], "group": job["group"],
                  "agent_query": job["agent_query"],
                  "mode": mode, "endpoint_step_id": job["step_ids"][-1],
                  "prefix_length": len(job["truth"]), "monitor": job.get("monitor"),
                  "predictions": [], "complete": False}
    if "reference_check" not in record:
        record["reference_check"] = check_isolated(
            job["truth"], checker, checker_options, directory / "reference", checker_timeout)
        write_json(output, record)
    if record["reference_check"]["status"] == "unsat":
        record.update(status="invalid_reference", complete=True)
        write_json(output, record)
        return record
    history = job["truth"][:-1] if mode == "endpoint" else []
    calls = job["truth"][-1:] if mode == "endpoint" else job["truth"]
    for index, current in enumerate(calls):
        messages = build_messages(task, tools, examples, history, current,
                                  agent_query=job["agent_query"])
        if index < len(record["predictions"]):
            prediction = record["predictions"][index]
            if prediction["messages"] != messages:
                raise ValueError("Checkpoint prompt differs from current experiment")
        else:
            prediction = client.predict(messages)
            prediction["messages"] = messages
            prediction["step_id"] = job["step_ids"][-1] if mode == "endpoint" else job["step_ids"][index]
            record["predictions"].append(prediction)
            write_json(output, record)
        if prediction["status"] != "ok":
            record.update(status=prediction["status"], complete=True, evaluated_steps=history)
            write_json(output, record)
            return record
        history = [*history, observation_step(current, prediction["observation"])]
    record["evaluated_steps"] = history
    record["prediction_check"] = check_isolated(
        history, checker, checker_options, directory / "prediction", checker_timeout)
    record.update(status=record["prediction_check"]["status"], complete=True)
    # Exact equality is descriptive only: existential consistency is the score.
    record["exact_reference_match"] = history == job["truth"]
    write_json(output, record)
    return record


def summarize(records, requested):
    counts = Counter(r["status"] for r in records)
    decided = counts["sat"] + counts["unsat"]
    groups = defaultdict(Counter)
    usage = Counter()
    for record in records:
        labels = {"target": record.get("target") or "all",
                  "min_length": str(record.get("group", {}).get("min_length", "unknown")),
                  "min_writes": str(record.get("group", {}).get("min_writes", "unknown"))}
        for key, value in labels.items():
            groups[f"{key}={value}"][record["status"]] += 1
        for prediction in record.get("predictions", []):
            for key, value in prediction.get("response", {}).get("usage", {}).items():
                if isinstance(value, (int, float)):
                    usage[key] += value
    by_group = {}
    for group, statuses in sorted(groups.items()):
        denominator = statuses["sat"] + statuses["unsat"]
        by_group[group] = {"counts": dict(statuses), "decided": denominator,
                           "consistency_rate": statuses["sat"] / denominator if denominator else None}
    return {"requested": requested, "completed": len(records), "counts": dict(counts),
            "decided": decided, "decision_coverage": decided / requested if requested else None,
            "consistency_rate": counts["sat"] / decided if decided else None,
            "reference_statuses": dict(Counter(r.get("reference_check", {}).get("status", "missing") for r in records)),
            "exact_reference_matches": sum(r.get("exact_reference_match", False) for r in records),
            "usage": dict(usage), "by_group": by_group}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Domain configuration JSON")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--symbolic-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("endpoint", "rollout"), default="endpoint")
    parser.add_argument("--model", required=True)
    parser.add_argument("--thinking", choices=("enabled", "disabled", "omit"), default="enabled")
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--request-timeout", type=int, default=300)
    parser.add_argument("--checker-timeout", type=int, default=120)
    parser.add_argument("--transport-retries", type=int, default=1)
    parser.add_argument("--base-url", help="Optional OpenAI-compatible base URL, e.g. http://localhost:8000/v1")
    parser.add_argument("--api-key-env", default="WM_API_KEY")
    parser.add_argument("--api-format", choices=("openai", "anthropic"), default="openai",
                        help="Native wire protocol; Messages can preserve controls lost by gateway translation")
    parser.add_argument("--request-options", type=Path, help="JSON of additional provider controls (saved in manifest)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, help="Optional limit on independent tests, not source sequences")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and enumerate tests without model requests")
    args = parser.parse_args(argv)
    if min(args.workers, args.max_tokens, args.request_timeout, args.checker_timeout) < 1 or args.transport_retries < 0:
        parser.error("Workers, budgets and timeouts must be positive; retries must be nonnegative")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    config = read_json(args.config)
    request_options = read_json(args.request_options) if args.request_options else {}
    if not isinstance(request_options, dict):
        raise ValueError("Request options must be a JSON object")
    config_root = args.config.resolve().parent
    tools = load_tools(config_root / config["tool_schema"])
    examples = read_json(config_root / config["examples"])["tools"]
    task = (config_root / config["task_description"]).read_text(encoding="utf-8")
    names = {tool["name"] for tool in tools}
    if set(examples) != names or any(len(examples[name]) != 3 for name in names):
        raise ValueError("Examples must cover every schema tool exactly three times")
    if any(e["tool_call"]["tool"] != name for name in names for e in examples[name]):
        raise ValueError("Example tool does not match its group")
    jobs, sources = load_jobs(args.input_dir, args.symbolic_dir, args.mode)
    for job in jobs:
        if any(step["tool"] not in names for step in job["truth"]):
            raise ValueError("Dataset contains a tool absent from the full schema")
    if args.limit:
        jobs = jobs[:args.limit]
    code = {p.name: p.read_text(encoding="utf-8") for p in Path(__file__).parent.glob("*.py")}
    settings = {"mode": args.mode, "model": args.model, "thinking": args.thinking,
                "workers": args.workers, "max_tokens": args.max_tokens,
                "request_timeout": args.request_timeout, "checker_timeout": args.checker_timeout,
                "transport_retries": args.transport_retries, "base_url": args.base_url,
                "api_key_env": args.api_key_env, "api_format": args.api_format,
                "request_options": request_options, "config": config,
                "input_dir": str(args.input_dir.resolve()),
                "symbolic_dir": str(args.symbolic_dir.resolve()) if args.symbolic_dir else None,
                "schema_sha256": digest(tools), "examples_sha256": digest(examples),
                "task_sha256": digest(task), "jobs_sha256": digest(jobs), "runner_sha256": digest(code)}
    fingerprint = digest(settings)
    if args.dry_run:
        print(f"Validated {len(sources)} source sequences, {len(jobs)} tests, {len(names) * 3} examples; mode={args.mode}")
        return 0
    output = args.output_dir.resolve()
    manifest_path = output / "manifest.json"
    if args.resume:
        manifest = read_json(manifest_path)
        if manifest["fingerprint"] != fingerprint:
            raise ValueError("Resume configuration/data/code changed; use a new output directory")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Output directory is not empty; use --resume or a new directory")
        manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "fingerprint": fingerprint,
                    "settings": settings, "source_count": len(sources), "requested": len(jobs)}
    manifest["status"] = "running"
    write_json(manifest_path, manifest)
    write_json(output / "runner_source_snapshot.json", code)
    # Freeze all non-secret prompt inputs once for exact experiment reproduction.
    write_json(output / "prompt_inputs.json", {"task_description": task, "tool_schema": tools, "examples": examples})
    client = ChatClient(model=args.model, thinking=args.thinking, max_tokens=args.max_tokens,
                        timeout=args.request_timeout, base_url=args.base_url,
                        api_key_env=args.api_key_env, retries=args.transport_retries, api_format=args.api_format,
                        request_options=request_options)
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for index, job in enumerate(jobs, 1):
            job_id = f"{index:04d}_{Path(job['source']).stem}_{job['target'] or 'rollout'}"
            future = pool.submit(run_job, job, mode=args.mode, client=client, task=task, tools=tools,
                                 examples=examples, checker=config["checker"],
                                 checker_options=config.get("checker_options", {}),
                                 checker_timeout=args.checker_timeout, directory=output / "cases" / job_id,
                                 resume=args.resume)
            futures[future] = (job_id, job)
        for future in as_completed(futures):
            job_id, job = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"source": job["source"], "target": job["target"], "group": job["group"],
                          "status": "runner_error", "error": f"{type(exc).__name__}: {exc}", "complete": False}
                write_json(output / "cases" / job_id / "runner_error.json", result)
            records.append(result)
            summary = summarize(records, len(jobs))
            manifest["summary"] = summary
            manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
            write_json(manifest_path, manifest)
            write_json(output / "summary.json", summary)
            print(f"[{len(records)}/{len(jobs)}] {job_id}: {result['status']}", flush=True)
    manifest["status"] = "complete"
    write_json(manifest_path, manifest)
    print(f"Results: {output / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
