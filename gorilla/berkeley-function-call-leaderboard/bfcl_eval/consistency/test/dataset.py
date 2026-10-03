"""Resolve monitored endpoints by stable step IDs, never by repaired positions."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from .common import read_json


def observation_step(step, observation):
    return {"tool": step["tool"], "args": deepcopy(step.get("args", {})),
            "observation": deepcopy(observation)}


def make_jobs(case, symbolic, source, mode):
    query = case.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError(f"{source}: query must be a non-empty string")
    chain, trace = case["tool_chain"], case["trace"]
    ids = [step["step_id"] for step in chain]
    if len(set(ids)) != len(ids) or len(chain) != len(trace):
        raise ValueError(f"{source}: duplicate IDs or incomplete replay")
    truth = []
    for call, record in zip(chain, trace):
        if (call["step_id"], call["tool"], call.get("args", {})) != (
                record["step_id"], record["tool"], record.get("args", {})):
            raise ValueError(f"{source}: replay does not match tool chain")
        truth.append(observation_step(call, record["result"]))
    common = {"source": source, "agent_query": query,
              "group": (symbolic or {}).get("sampling_group", {}),
              "truth": truth, "step_ids": ids}
    if mode == "rollout":
        return [{**common, "target": None, "endpoint": len(chain) - 1}]
    if not symbolic:
        raise ValueError("Endpoint mode requires symbolic planning metadata")
    jobs = []
    for name, monitor in symbolic["planning"]["target_chains"].items():
        required = [monitor["anchor_step"], *monitor["write_steps"], monitor["read_step"]]
        if any(step_id not in ids for step_id in required):
            raise ValueError(f"{source}/{name}: grounding deleted a monitored step")
        anchor, endpoint = ids.index(required[0]), ids.index(required[-1])
        lower = anchor + (monitor.get("anchor_phase", "before") == "after")
        upper = endpoint + (monitor.get("read_phase", "before") == "after")
        if anchor > endpoint or any(not lower <= ids.index(w) < upper for w in monitor["write_steps"]):
            raise ValueError(f"{source}/{name}: grounding broke the observation interval")
        jobs.append({**common, "target": name, "monitor": deepcopy(monitor),
                     "endpoint": endpoint, "truth": truth[:endpoint + 1],
                     "step_ids": ids[:endpoint + 1]})
    return jobs


def load_jobs(input_dir, symbolic_dir, mode):
    root = Path(input_dir).resolve()
    jobs, sources = [], []
    for path in sorted(root.rglob("*.json")):
        relative = path.relative_to(root)
        if any(part.startswith("_") for part in relative.parts) or path.name == "manifest.json":
            continue
        case = read_json(path)
        if not isinstance(case, dict) or case.get("status") != "accepted":
            continue
        if "tool_chain" not in case:
            continue
        symbolic = None
        if symbolic_dir:
            source = Path(case.get("source", relative.as_posix()).replace("\\", "/"))
            symbolic_root = Path(symbolic_dir).resolve()
            symbolic_path = (symbolic_root / source).resolve()
            if not symbolic_path.is_relative_to(symbolic_root):
                raise ValueError("Symbolic source escapes the configured directory")
            symbolic = read_json(symbolic_path)
        jobs.extend(make_jobs(case, symbolic, relative.as_posix(), mode))
        sources.append(relative.as_posix())
    if not jobs:
        raise ValueError("No accepted test cases found")
    return jobs, sources
