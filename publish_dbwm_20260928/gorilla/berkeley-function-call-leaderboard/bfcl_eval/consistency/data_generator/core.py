"""Domain-independent scheduling, model proposals, rollout, and validation."""

from __future__ import annotations

import importlib.util
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol


@dataclass(frozen=True)
class Call:
    tool: str
    args: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "args": deepcopy(self.args)}


@dataclass(frozen=True)
class Job:
    scenario_name: str
    initial_state: dict[str, Any]
    initial_state_source: str
    length: int
    read_gap: int
    write_count: int
    target: str
    seed: int


class DomainAdapter(Protocol):
    name: str

    def target_read_call(self, job: Job) -> Call: ...

    def make_skeleton(self, job: Job) -> list[Call | None]: ...

    def distractor_candidates(self, job: Job) -> list[Call]: ...

    def rollout(self, initial_state: Mapping[str, Any], calls: list[Call]) -> tuple[list[dict[str, Any]], list[Any]]: ...

    def target_value(self, state: Any, target: str) -> Any: ...

    def describe_call(self, call: Call) -> str: ...

    def tool_schema(self) -> list[dict[str, Any]]: ...


def _load_chat_text(path: Path) -> Callable[..., str]:
    spec = importlib.util.spec_from_file_location("consistency_external_model_client", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load model client at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "chat_text"):
        raise RuntimeError("Model client must expose chat_text")
    chat_text = module.chat_text
    setattr(chat_text, "_consistency_default_model", getattr(module, "DEFAULT_MODEL", "unspecified"))
    return chat_text


def _parse_proposal(raw: str, slot_count: int, candidate_count: int) -> tuple[list[int], str]:
    text = raw.strip()
    if not text:
        raise ValueError("Empty model response")
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start < 0:
            raise
        value, _ = json.JSONDecoder().raw_decode(text, start)
    choices = value["distractor_ids"]
    if (
        not isinstance(choices, list)
        or len(choices) != slot_count
        or any(type(index) is not int or not 0 <= index < candidate_count for index in choices)
    ):
        raise ValueError("Invalid distractor choices")
    introduction = value.get("query_introduction", "")
    if not isinstance(introduction, str) or len(introduction) > 400:
        raise ValueError("Invalid query introduction")
    return choices, introduction.strip()


def _model_proposal(
    chat_text: Callable[..., str],
    adapter: DomainAdapter,
    job: Job,
    skeleton: list[Call | None],
    candidates: list[Call],
    model: str | None,
    retries: int,
) -> tuple[list[int], str, dict[str, Any]]:
    prompt = json.dumps(
        {
            "domain": adapter.name,
            "target_field": job.target,
            "trajectory_length": job.length,
            "read_gap_intervening_calls": job.read_gap,
            "target_write_count": job.write_count,
            "fixed_skeleton": [call.as_dict() if call else "DISTRACTOR" for call in skeleton],
            "distractor_candidates": [
                {"id": index, **call.as_dict(), "intent": adapter.describe_call(call)}
                for index, call in enumerate(candidates)
            ],
            "required_distractor_count": skeleton.count(None),
        },
        ensure_ascii=False,
    )
    system = (
        "Design a challenging long-horizon tool-use task. Return only JSON with "
        "distractor_ids (one candidate ID per DISTRACTOR slot, in order). "
        "Choose diverse read-only distractors that do not modify the target field. "
        "Do not invent tool calls, observations, balances, holdings, or outcomes. "
        "Do not reveal the initial backend state. The exact step-by-step task "
        "will be constructed programmatically from the selected calls."
    )
    error_types: list[str] = []
    for attempt in range(1, retries + 1):
        try:
            kwargs: dict[str, Any] = {
                "system": system,
                "thinking": False,
                "temperature": 0.35,
                "max_tokens": 4000,
                "timeout": 90,
            }
            if model:
                kwargs["model"] = model
            raw = chat_text(prompt, **kwargs)
            choices, introduction = _parse_proposal(raw, skeleton.count(None), len(candidates))
            introduction_rejected = "\ufffd" in introduction or any(ord(character) > 127 for character in introduction)
            if introduction_rejected:
                introduction = ""
            return choices, introduction, {
                "source": "model_client.chat_text",
                "model": model or getattr(chat_text, "_consistency_default_model", "unspecified"),
                "attempts": attempt,
                "proposed_distractor_ids": choices,
                "query_introduction": introduction,
                "introduction_rejected": introduction_rejected,
                "validation": "accepted",
            }
        except Exception as exc:  # Retries also cover transient provider failures.
            error_types.append(type(exc).__name__)
            if attempt < retries:
                time.sleep(min(2 ** (attempt - 1), 8))
    rng = random.Random(job.seed)
    choices = [rng.randrange(len(candidates)) for _ in range(skeleton.count(None))]
    return choices, "", {
        "source": "deterministic_fallback",
        "model": model or getattr(chat_text, "_consistency_default_model", "unspecified"),
        "attempts": retries,
        "proposed_distractor_ids": choices,
        "validation": "fallback_after_model_failure",
        "failure_types": error_types,
    }


def format_query(adapter: DomainAdapter, calls: list[Call]) -> str:
    """Render an executable user request without leaking backend observations."""
    instructions = "\n".join(
        f"{index}. {adapter.describe_call(call)}" for index, call in enumerate(calls, start=1)
    )
    return "请按以下顺序完成操作，并在每一步之后观察工具返回，再继续下一步：\n" + instructions


def build_trajectory(
    adapter: DomainAdapter,
    job: Job,
    chat_text: Callable[..., str],
    *,
    model: str | None = None,
    retries: int = 2,
) -> dict[str, Any]:
    skeleton = adapter.make_skeleton(job)
    candidates = adapter.distractor_candidates(job)
    if len(skeleton) != job.length or not candidates:
        raise ValueError("Adapter returned an invalid skeleton or no distractors")
    choices, _introduction, proposal = _model_proposal(
        chat_text, adapter, job, skeleton, candidates, model, retries
    )
    iterator = iter(choices)
    calls = [call if call is not None else candidates[next(iterator)] for call in skeleton]
    observations, states = adapter.rollout(job.initial_state, calls)
    if len(observations) != job.length or len(states) != job.length + 1:
        raise ValueError("Backend rollout length does not match the requested length")

    target_read = adapter.target_read_call(job)
    fixed_reads = [index for index, call in enumerate(skeleton) if call == target_read]
    if len(fixed_reads) != 2:
        raise ValueError(f"Expected exactly two target reads, got {fixed_reads}")
    first_read, second_read = fixed_reads
    actual_gap = second_read - first_read - 1
    if actual_gap != job.read_gap:
        raise ValueError("Read gap mismatch")
    target_values = [adapter.target_value(state, job.target) for state in states]
    write_steps = [index for index, (before, after) in enumerate(zip(target_values, target_values[1:])) if before != after]
    if len(write_steps) != job.write_count or any(not first_read < step < second_read for step in write_steps):
        raise ValueError(f"Target field changed at unexpected steps: {write_steps}")
    if any("error" in observation for observation in observations if isinstance(observation, dict)):
        raise ValueError("A planned tool returned a backend error")

    query = format_query(adapter, calls)
    steps = [{**call.as_dict(), "observation": deepcopy(observation)} for call, observation in zip(calls, observations)]
    return {
        "id": f"{adapter.name}_{job.scenario_name}_l{job.length}",
        "domain": adapter.name,
        "query": query,
        "tool_schema": adapter.tool_schema(),
        "initial_state_ref": job.initial_state_source,
        "scenario_name": job.scenario_name,
        "initial_state": deepcopy(job.initial_state),
        "controls": {
            "length": job.length,
            "target_field": job.target,
            "read_gap": job.read_gap,
            "read_gap_definition": "number of tool calls strictly between target reads",
            "write_count": job.write_count,
            "write_frequency": job.write_count / job.read_gap,
            "seed": job.seed,
        },
        "validation": {
            "backend_replay": "passed",
            "target_read_steps": [first_read + 1, second_read + 1],
            "target_write_steps": [step + 1 for step in write_steps],
            "actual_length": len(steps),
            "actual_read_gap": actual_gap,
            "actual_write_count": len(write_steps),
        },
        "generation": {"model_proposal": proposal, "observation_source": "real_backend"},
        "steps": steps,
    }


def generate_many(
    adapter: DomainAdapter,
    jobs: list[Job],
    output_dir: Path,
    model_client_path: Path,
    *,
    model: str | None = None,
    workers: int = 4,
    retries: int = 2,
) -> list[Path]:
    chat_text = _load_chat_text(model_client_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(build_trajectory, adapter, job, chat_text, model=model, retries=retries): job
            for job in jobs
        }
        for future in as_completed(futures):
            trajectory = future.result()
            generated[trajectory["id"]] = trajectory
            print(f"validated {len(generated)}/{len(jobs)} {trajectory['id']}", flush=True)
    paths: list[Path] = []
    for identifier, trajectory in sorted(generated.items()):
        validator = getattr(adapter, "validate_trace", None)
        if validator is not None:
            verdict = validator(trajectory["steps"])
            trajectory["validation"]["solver_status"] = verdict["status"]
            if verdict["status"] != "sat":
                raise ValueError(f"Consistency solver rejected generated trajectory {identifier}: {verdict['status']}")
        path = output_dir / f"{identifier}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(trajectory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
        paths.append(path)
    return paths
