"""Ground symbolic tool chains with an LLM, then replay them on a backend.

The sampler deliberately emits symbolic plans.  This module is the generic
bridge from those plans to executable cases:

1. compress a symbolic plan into the information needed by a grounding model;
2. assign concrete values to numbered placeholders;
3. fill bound arguments and identity requirements deterministically;
4. ask the model for the remaining arguments and an initial state;
5. replay every call on the supplied backend;
6. ask the model to audit the observed branches and optionally repair the case.

The backend is loaded from a Python file.  A backend class should expose the
tool methods named in the tool document and one of ``_load_scenario``,
``load_scenario`` or ``set_state``.  The TradingBotHard implementation already
matches this convention through ``_load_scenario``.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import inspect
import json
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, time
from functools import partial
from pathlib import Path
from time import sleep
from typing import Any, Iterable, Mapping, Sequence

from bfcl_eval.consistency.data_generator_v2.grounding.diversity_memory import (
    DiversityMemory,
    MODEL_TOKENS,
    memory_scope,
)

try:
    from utils.model_client import chat
except ModuleNotFoundError:  # Running from the nested BFCL checkout.
    _workspace_root = Path(__file__).resolve().parents[6]
    if str(_workspace_root) not in sys.path:
        sys.path.insert(0, str(_workspace_root))
    from utils.model_client import chat


PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_.]*)\}")
RESULT_BINDING = re.compile(r"^result\.(.+)$")
VALUE_REFERENCE = re.compile(r"^\$binding:([A-Za-z_][A-Za-z0-9_.]*)$")


class GroundingError(RuntimeError):
    """A candidate could not be grounded, replayed or audited."""


def _jsonable(value: Any) -> Any:
    """Convert backend values into deterministic JSON data for prompts/output."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return repr(value)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_or_jsonl(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        values = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                values.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise GroundingError(f"Invalid JSON/JSONL at {path}:{line_number}") from exc
        return values


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        for attempt in range(6):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                # Windows readers or file scanners can briefly hold the target.
                # Preserve atomic replacement and bound retries to 3.1 seconds.
                if attempt == 5:
                    raise
                sleep(0.1 * 2 ** attempt)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _strings(item)


def _placeholder_tokens(value: Any) -> set[str]:
    return {token for text in _strings(value) for token in PLACEHOLDER.findall(text)}


def _substitute(value: Any, bindings: Mapping[str, Any]) -> Any:
    """Substitute ``{placeholder}`` and ``$binding:placeholder`` references."""

    if isinstance(value, str):
        reference = VALUE_REFERENCE.fullmatch(value)
        if reference:
            return copy.deepcopy(bindings[reference.group(1)])

        def replace(match: re.Match[str]) -> str:
            token = match.group(1)
            if token not in bindings:
                return match.group(0)
            replacement = bindings[token]
            return str(replacement) if not isinstance(replacement, (dict, list)) else match.group(0)

        return PLACEHOLDER.sub(replace, value)
    if isinstance(value, Mapping):
        return {key: _substitute(item, bindings) for key, item in value.items()}
    if isinstance(value, list):
        return [_substitute(item, bindings) for item in value]
    return value


def load_tool_document(path: Path) -> dict[str, dict[str, Any]]:
    """Load common OpenAI-style and BFCL list-style tool documents."""

    raw = _read_json_or_jsonl(path)
    if isinstance(raw, Mapping):
        if "tools" in raw:
            raw = raw["tools"]
        elif "function" in raw:
            raw = raw["function"]
        elif "name" in raw:
            raw = [raw]
    if not isinstance(raw, list):
        raise GroundingError(f"Tool document must contain a list of tools: {path}")
    result: dict[str, dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        if "function" in item and isinstance(item["function"], Mapping):
            item = item["function"]
        name = item.get("name")
        if not name:
            continue
        result[str(name)] = dict(item)
    if not result:
        raise GroundingError(f"No tools found in {path}")
    return result


def _tool_parameters(tool: Mapping[str, Any]) -> tuple[dict[str, Any], set[str]]:
    parameters = tool.get("parameters", {})
    if not isinstance(parameters, Mapping):
        return {}, set()
    properties = parameters.get("properties", {})
    if not isinstance(properties, Mapping):
        properties = {}
    required = parameters.get("required", [])
    return dict(properties), {str(item) for item in required}


def _identity_requirements(step: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Collect result aliases and target identity obligations in one field."""

    requirements: list[dict[str, Any]] = []
    for alias, symbolic in sorted((step.get("placeholder_bindings") or {}).items()):
        match = RESULT_BINDING.fullmatch(str(alias))
        if match:
            requirements.append({
                "kind": "result_binding",
                "result": f"$.result.{match.group(1)}",
                "placeholder": symbolic,
                "meaning": "The backend result must supply the concrete value used by later calls.",
            })
    for mutation in step.get("mutations", []) or []:
        for relation in mutation.get("target_identity_sources", []) or []:
            aliases = [
                alias
                for alias, symbolic in (step.get("placeholder_bindings") or {}).items()
                if symbolic == relation.get("placeholder")
            ]
            requirements.append({
                "kind": "target_identity",
                "placeholder": relation.get("placeholder"),
                "binding_aliases": aliases,
                "path": relation.get("path"),
                "logic": relation.get("logic"),
                "meaning": "At this write, the addressed state key must equal the bound identity.",
            })
    return requirements


def compact_symbolic_trajectory(
    trajectory: Mapping[str, Any],
    tools: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Keep only the grounding contract while preserving original step IDs."""

    compact_steps: list[dict[str, Any]] = []
    placeholders: set[str] = set()
    for step in trajectory.get("steps", []):
        tool_name = step["tool"]
        tool = tools.get(tool_name, {})
        properties, required = _tool_parameters(tool)
        original_bindings = dict(step.get("placeholder_bindings") or {})
        placeholders.update(str(value) for value in original_bindings.values())
        parameters = []
        for name, schema in properties.items():
            alias = f"args.{name}"
            symbolic = original_bindings.get(alias)
            entry: dict[str, Any] = {
                "name": name,
                "type": schema.get("type") if isinstance(schema, Mapping) else None,
                "required": name in required,
            }
            if symbolic is None:
                entry["value"] = "unrestricted"
            else:
                entry["placeholder"] = symbolic
                placeholders.add(str(symbolic))
            parameters.append(entry)

        requirements = _identity_requirements(step)
        placeholders.update(
            str(item["placeholder"])
            for item in requirements
            if item.get("placeholder")
        )
        compact_steps.append({
            "step_id": step.get("index"),
            "tool": tool_name,
            "branch": step.get("branch"),
            "parameters": parameters,
            "branch_conditions": {
                "selected": step.get("branch_condition"),
                "avoid": step.get("earlier_branch_conditions_to_avoid", []),
                "additional": step.get("additional_conditions", []),
            },
            "requirements": requirements,
            "is_distractor": step.get("role") == "distractor" or "distractor" in step.get("roles", []),
        })
    return {
        "trajectory_id": trajectory.get("id"),
        "state_conditions": trajectory.get("state_conditions", {}),
        "targets": trajectory.get("targets", []),
        "steps": compact_steps,
        "placeholder_names": sorted(placeholders),
        "planning": {
            "target_chains": trajectory.get("planning", {}).get("target_chains", {}),
            "lifecycle_instances": trajectory.get("planning", {}).get("lifecycle_instances", {}),
            "grounding_obligations": trajectory.get("planning", {}).get("grounding_obligations", []),
        },
    }


def _json_from_text(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise GroundingError(f"Model did not return valid JSON: {text[:500]}") from exc
    if not isinstance(value, dict):
        raise GroundingError("Model JSON response must be an object")
    return value


def _model_json(messages: Sequence[Mapping[str, Any]], *, model: str, max_tokens: int,
                timeout: int = 300, temperature: float = 0.0, thinking: bool = False) -> dict[str, Any]:
    response = chat(
        messages,
        model=model,
        temperature=temperature,
        thinking=thinking,
        max_tokens=max_tokens,
        timeout=timeout,
        response_format={"type": "json_object"},
    )
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise GroundingError("Model response is missing choices[0].message.content") from exc
    if not isinstance(content, str):
        raise GroundingError("Model response content must be text")
    if not content.strip():
        reason = response["choices"][0].get("finish_reason", "unknown")
        raise GroundingError(f"Empty model response (finish_reason={reason}, max_tokens={max_tokens})")
    return _json_from_text(content)


def _binding_answer(answer: Mapping[str, Any], fallback: Mapping[str, Any] | None = None) -> Any:
    if "placeholder_values" in answer:
        return answer["placeholder_values"]
    if "placeholder_bindings" in answer:
        return answer["placeholder_bindings"]
    return fallback


def _common_system_prompt(avoidance_guide: str = "") -> str:
    prompt = """You ground symbolic tool trajectories into executable backend cases. Return JSON only.

The symbolic trajectory is authoritative for tool order, branch names, roles and
placeholder aliases. Never invent a new tool or branch. A distractor is still a
real call and must satisfy its conditions. Preserve identity: one symbolic
placeholder always has one concrete value, and different numbered identities
must remain distinct unless the trajectory explicitly permits equality.

The backend replay is the final authority. Do not fabricate observations. Use
the supplied backend source to choose values that make every selected branch
and every identity requirement true at the moment of the call. Prefer a
realistic, varied environment and meaningful operations; avoid immediately
undoing an earlier change merely to create noise or satisfy a call count. If the
backend source contains DEFAULT_STATE or a fallback scenario, treat it only as
a schema hint: construct a new realistic state with different identifiers,
numeric scales, entity populations, relationships and histories where the
backend supports them, instead of copying the default state."""
    if avoidance_guide:
        prompt += (
            "\n\nDIVERSITY AVOIDANCE GUIDE (soft preference, never overrides branch conditions, "
            "identity bindings, execution validity or repair priority):\n" + avoidance_guide
        )
    return prompt


def _placeholder_prompt(compact: Mapping[str, Any], tool_doc: Any, backend: str,
                        avoidance_guide: str = "") -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _common_system_prompt(avoidance_guide)},
        {"role": "user", "content": f"""Assign concrete values to every placeholder in this trajectory.

Field meanings:
- placeholder_names: every symbolic identity that must receive a concrete value.
- parameters.placeholder: an argument must later be filled from that identity.
- requirements.result_binding: a backend result field must equal the identity used later.
- requirements.target_identity: the state key addressed by a write must equal the identity.
- branch_conditions: the selected branch must be true and every avoid condition false.
- is_distractor: this call is filler, but its conditions and side effects remain mandatory.

Return exactly:
{{"placeholder_values": {{"placeholder_name": "concrete value"}}}}

Use numbers for numeric identities, strings for symbols, and keep all values
stable and distinct where numbered identities require it. Do not return args or
initial_state in this phase.

SYMBOLIC TRAJECTORY:
{json.dumps(compact, ensure_ascii=False, indent=2)}

TOOL DOCUMENT:
{json.dumps(tool_doc, ensure_ascii=False, indent=2)}

BACKEND SOURCE:
{backend}
"""},
    ]


def _grounding_prompt(
    hydrated: Mapping[str, Any],
    tool_doc: Any,
    backend: str,
    avoidance_guide: str = "",
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _common_system_prompt(avoidance_guide)},
        {"role": "user", "content": f"""Complete the already-bound trajectory.

The script has already filled every argument marked with a placeholder and has
substituted all result/identity requirements. Those values are fixed. Fill only
the remaining parameters marked value=unrestricted and create one complete
realistic initial_state. The state must include every field needed by the
backend, satisfy all selected branches and avoid conditions, and support every
side effect throughout the whole sequence. Do not return observations.

Return exactly:
{{
  "initial_state": {{...}},
  "steps": [
    {{"step_id": 1, "args": {{...}}}}
  ],
  "replay_config": {{"long_context": false, "random_seed": 12345}}
}}

Include all required arguments. You may include no-argument tools with args={{}}.
Do not alter step_id, tool, branch, order, or any fixed argument.

HYDRATED TRAJECTORY:
{json.dumps(hydrated, ensure_ascii=False, indent=2)}

TOOL DOCUMENT:
{json.dumps(tool_doc, ensure_ascii=False, indent=2)}

BACKEND SOURCE:
{backend}
"""},
    ]


def _validate_placeholder_values(compact: Mapping[str, Any], values: Mapping[str, Any]) -> None:
    expected = set(compact.get("placeholder_names", []))
    missing = sorted(expected - set(values))
    if missing:
        raise GroundingError(f"Model omitted placeholder values: {missing}")
    extra = sorted(set(values) - expected)
    if extra:
        raise GroundingError(f"Model introduced unknown placeholders: {extra}")


def _coerce_placeholder_values(compact: Mapping[str, Any], values: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize obvious numeric aliases before they reach backend arguments."""

    expected_types: dict[str, str] = {}
    for step in compact.get("steps", []):
        for parameter in step.get("parameters", []):
            placeholder = parameter.get("placeholder")
            if placeholder and parameter.get("type"):
                expected_types.setdefault(str(placeholder), str(parameter["type"]))
    normalized = dict(values)
    for placeholder, expected in expected_types.items():
        if placeholder not in normalized or not isinstance(normalized[placeholder], str):
            continue
        value = normalized[placeholder].strip()
        try:
            if expected in {"integer", "int"}:
                normalized[placeholder] = int(value)
            elif expected in {"number", "float"}:
                normalized[placeholder] = float(value)
        except ValueError as exc:
            raise GroundingError(
                f"Placeholder {placeholder} must be a {expected}, got {normalized[placeholder]!r}"
            ) from exc
    return normalized


def hydrate_compact(compact: Mapping[str, Any], values: Mapping[str, Any]) -> dict[str, Any]:
    """Fill bound argument slots and requirement values after phase one."""

    values = _coerce_placeholder_values(compact, dict(values))
    _validate_placeholder_values(compact, values)
    hydrated = copy.deepcopy(compact)
    hydrated["placeholder_values"] = dict(values)
    for step in hydrated["steps"]:
        for parameter in step["parameters"]:
            if "placeholder" in parameter:
                parameter["value"] = copy.deepcopy(values[parameter["placeholder"]])
                parameter["fixed"] = True
        step["requirements"] = _substitute(step.get("requirements", []), values)
    return hydrated


def _step_map(steps: Sequence[Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
    result = {}
    for step in steps:
        try:
            result[int(step["step_id"])] = step
        except (KeyError, TypeError, ValueError) as exc:
            raise GroundingError(f"Invalid step_id in model output: {step}") from exc
    if len(result) != len(steps):
        raise GroundingError("Model returned duplicate step_id values")
    return result


def materialize_candidate(
    compact: Mapping[str, Any],
    hydrated: Mapping[str, Any],
    answer: Mapping[str, Any],
    values: Mapping[str, Any],
    *,
    allow_sequence_changes: bool = False,
) -> dict[str, Any]:
    """Merge model values with fixed aliases and retain symbolic bindings."""

    state = answer.get("initial_state")
    if not isinstance(state, Mapping):
        raise GroundingError("Grounding response has no initial_state object")
    proposed_steps = answer.get("steps")
    if not isinstance(proposed_steps, list):
        raise GroundingError("Grounding response has no steps list")
    by_id = _step_map(proposed_steps)
    expected = _step_map(compact["steps"])
    if (not allow_sequence_changes and set(by_id) != set(expected)) or (
        allow_sequence_changes and (not set(by_id) or not set(by_id) <= set(expected))
    ):
        raise GroundingError("Grounding response changed the symbolic step set illegally")
    order = list(by_id) if allow_sequence_changes else [int(step["step_id"]) for step in compact["steps"]]
    hydrated_by_id = _step_map(hydrated["steps"])

    grounded_steps = []
    for step_id in order:
        original = expected[step_id]
        hydrated_step = hydrated_by_id[step_id]
        answer_step = by_id[step_id]
        if answer_step.get("tool") not in (None, original["tool"]):
            raise GroundingError(f"Model changed tool for step {step_id}")
        if answer_step.get("branch") not in (None, original.get("branch")):
            raise GroundingError(f"Model changed branch for step {step_id}")
        args = answer_step.get("args", {})
        if not isinstance(args, Mapping):
            raise GroundingError(f"args must be an object at step {step_id}")
        fixed = {
            p["name"]: copy.deepcopy(p["value"])
            for p in hydrated_step["parameters"]
            if "placeholder" in p
        }
        merged = dict(args)
        for name, value in fixed.items():
            if name in merged and merged[name] != value:
                raise GroundingError(f"Model changed fixed argument {name} at step {step_id}")
            merged[name] = value
        grounded_steps.append({
            "step_id": step_id,
            "tool": original["tool"],
            "branch": original.get("branch"),
            "args": _substitute(merged, values),
            "placeholder_bindings": {},
            "branch_conditions": copy.deepcopy(original.get("branch_conditions", {})),
            "is_distractor": original.get("is_distractor", False),
            "requirements": copy.deepcopy(hydrated_step.get("requirements", [])),
        })
    for grounded in grounded_steps:
        original = expected[grounded["step_id"]]
        # Keep the original aliases for downstream identity auditing.
        grounded["placeholder_bindings"] = _aliases_for_step(
            original, compact.get("steps", []), values
        )
    replay_config = dict(answer.get("replay_config") or {})
    replay_config.setdefault("long_context", False)
    return {
        "placeholder_values": dict(values),
        "initial_state": _jsonable(state),
        "replay_config": replay_config,
        "steps": grounded_steps,
    }


def _aliases_for_step(
    step: Mapping[str, Any], all_steps: Sequence[Mapping[str, Any]], values: Mapping[str, Any]
) -> dict[str, Any]:
    # Compact steps intentionally retain only parameter-level aliases.  The
    # requirement values carry result and target-identity aliases.
    aliases: dict[str, Any] = {}
    for parameter in step.get("parameters", []):
        if "placeholder" in parameter:
            aliases[f"args.{parameter['name']}"] = parameter["placeholder"]
    for requirement in step.get("requirements", []):
        if requirement.get("kind") == "result_binding":
            aliases[requirement["result"].replace("$.result.", "result.")] = requirement["placeholder"]
    return aliases


def load_backend(path: Path, class_name: str | None = None) -> Any:
    # Backend modules commonly import their repository package.  Add every
    # ancestor so this loader works whether invoked from the workspace root or
    # from the nested BFCL checkout.
    for ancestor in path.resolve().parents:
        if str(ancestor) not in sys.path:
            sys.path.insert(0, str(ancestor))
    module_name = f"grounding_backend_{abs(hash(path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise GroundingError(f"Could not load backend module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if class_name:
        backend_class = getattr(module, class_name)
    else:
        candidates = []
        for _, candidate in inspect.getmembers(module, inspect.isclass):
            if candidate.__module__ != module.__name__:
                continue
            methods = set(dir(candidate))
            if {"_load_scenario", "load_scenario", "set_state"} & methods:
                candidates.append(candidate)
        if len(candidates) != 1:
            raise GroundingError(
                f"Specify --backend-class; discovered backend classes: {[c.__name__ for c in candidates]}"
            )
        backend_class = candidates[0]
    return backend_class()


def _load_state(instance: Any, state: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    loader = next((getattr(instance, name, None) for name in ("_load_scenario", "load_scenario", "set_state")
                   if callable(getattr(instance, name, None))), None)
    if loader is None:
        raise GroundingError("Backend must expose _load_scenario, load_scenario or set_state")
    kwargs = {}
    try:
        signature = inspect.signature(loader)
        if "long_context" in signature.parameters:
            kwargs["long_context"] = bool(config.get("long_context", False))
    except (TypeError, ValueError):
        pass
    loader(copy.deepcopy(dict(state)), **kwargs)


def _snapshot(instance: Any) -> dict[str, Any]:
    hook = getattr(instance, "state_snapshot", None)
    if callable(hook):
        return _jsonable(hook())
    hook = getattr(instance, "get_state", None)
    if callable(hook):
        return _jsonable(hook())
    data = {}
    for key, value in vars(instance).items():
        if key.startswith("_"):
            continue
        data[key] = copy.deepcopy(value)
    return _jsonable(data)


@dataclass
class ReplayResult:
    ok: bool
    trace: list[dict[str, Any]]
    errors: list[dict[str, Any]]


def replay(candidate: Mapping[str, Any], backend_path: Path, backend_class: str | None) -> ReplayResult:
    instance = load_backend(backend_path, backend_class)
    config = candidate.get("replay_config", {})
    _load_state(instance, candidate["initial_state"], config)
    trace = []
    errors = []
    for step in candidate["steps"]:
        before = _snapshot(instance)
        args = copy.deepcopy(step.get("args", {}))
        record = {
            "step_id": step["step_id"],
            "tool": step["tool"],
            "expected_branch": step.get("branch"),
            "is_distractor": step.get("is_distractor", False),
            "args": _jsonable(args),
            "state_before": before,
        }
        try:
            method = getattr(instance, step["tool"])
            result = method(**args)
            record["result"] = _jsonable(result)
            if isinstance(result, Mapping) and "error" in result:
                record["error"] = result.get("error")
                errors.append({"step_id": step["step_id"], "tool": step["tool"], "error": result.get("error")})
        except Exception as exc:  # backend errors are part of the repair signal
            record["exception"] = f"{type(exc).__name__}: {exc}"
            errors.append({"step_id": step["step_id"], "tool": step["tool"], "exception": record["exception"]})
        record["state_after"] = _snapshot(instance)
        trace.append(record)
        if errors:
            break
    return ReplayResult(not errors and len(trace) == len(candidate["steps"]), trace, errors)


def _audit_prompt(
    candidate: Mapping[str, Any],
    replay_result: ReplayResult,
    compact: Mapping[str, Any],
    tool_doc: Any,
    backend: str,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _common_system_prompt()},
        {"role": "user", "content": f"""Audit this replay. Determine whether every executed call entered its expected branch.

Use state_before, args, result and state_after for each step. A call is invalid
if it returned an error, raised an exception, violated a selected condition, or
violated a result/target identity requirement. Do not repair the case here.

Return exactly:
{{
  "valid": true,
  "steps": [{{"step_id": 1, "branch_match": true, "observed_branch": "...", "reason": "..."}}],
  "issues": []
}}

EXPECTED GROUNDED CASE:
{json.dumps(candidate, ensure_ascii=False, indent=2)}

SYMBOLIC CONTRACT:
{json.dumps(compact, ensure_ascii=False, indent=2)}

REPLAY TRACE:
{json.dumps(replay_result.trace, ensure_ascii=False, indent=2)}

TOOL DOCUMENT:
{json.dumps(tool_doc, ensure_ascii=False, indent=2)}

BACKEND SOURCE:
{backend}
"""},
    ]


def _debug_prompt(
    candidate: Mapping[str, Any],
    replay_result: ReplayResult,
    audit: Mapping[str, Any] | None,
    compact: Mapping[str, Any],
    tool_doc: Any,
    backend: str,
    avoidance_guide: str = "",
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _common_system_prompt(avoidance_guide)},
        {"role": "user", "content": f"""Repair this grounded trajectory. Return a complete replacement JSON object.

Debug priority is mandatory:
1. First try changing only initial_state.
2. If that cannot work, change unrestricted args or concrete placeholder values,
   while preserving every symbolic alias and all result/identity bindings.
3. If the sequence itself is invalid, first remove or reorder distractor steps.
4. Then consider reordering non-distractor steps.
5. Finally consider deleting non-distractor steps.

Make the smallest repair. You may delete/reorder steps only in stages 3-5.
Never invent a tool or branch. Keep step_id, tool, branch, is_distractor and
placeholder_bindings for retained steps. Return:
{{"placeholder_values": {{...}}, "initial_state": {{...}},
  "replay_config": {{...}}, "steps": [{{"step_id": 1, "args": {{...}}}}]}}

SYMBOLIC CONTRACT:
{json.dumps(compact, ensure_ascii=False, indent=2)}

CURRENT CANDIDATE:
{json.dumps(candidate, ensure_ascii=False, indent=2)}

REPLAY:
{json.dumps({"ok": replay_result.ok, "trace": replay_result.trace, "errors": replay_result.errors}, ensure_ascii=False, indent=2)}

BRANCH AUDIT:
{json.dumps(audit or {}, ensure_ascii=False, indent=2)}

TOOL DOCUMENT:
{json.dumps(tool_doc, ensure_ascii=False, indent=2)}

BACKEND SOURCE:
{backend}
"""},
    ]


def ground_one(
    trajectory: Mapping[str, Any],
    *,
    tool_doc: Any,
    tool_map: Mapping[str, Mapping[str, Any]],
    backend_path: Path,
    backend_class: str | None,
    model: str,
    max_debug: int,
    max_tokens: int,
    memory: DiversityMemory | None = None,
    request_timeout: int = 300,
) -> dict[str, Any]:
    ask = partial(_model_json, timeout=request_timeout)
    compact = compact_symbolic_trajectory(trajectory, tool_map)
    backend_source = backend_path.read_text(encoding="utf-8")
    scope = memory_scope(backend_source, tool_doc, backend_class) if memory else ""
    diversity: dict[str, Any] = {"enabled": memory is not None, "guide": ""}
    if memory:
        try:
            diversity = memory.guide(compact, scope, ask, model)
        except Exception as exc:
            diversity["warning"] = f"Memory retrieval/guide skipped: {type(exc).__name__}: {str(exc)[:200]}"
    guide = diversity["guide"]
    values_answer = ask(_placeholder_prompt(compact, tool_doc, backend_source, guide), model=model, max_tokens=max_tokens)
    values = _binding_answer(values_answer)
    if not isinstance(values, Mapping):
        raise GroundingError("Placeholder phase must return placeholder_values")
    values = _coerce_placeholder_values(compact, dict(values))
    hydrated = hydrate_compact(compact, values)
    answer = ask(_grounding_prompt(hydrated, tool_doc, backend_source, guide), model=model, max_tokens=max_tokens)
    candidate = materialize_candidate(compact, hydrated, answer, values)

    audit: dict[str, Any] | None = None
    for debug_round in range(max_debug + 1):
        replay_result = replay(candidate, backend_path, backend_class)
        if replay_result.ok:
            audit = ask(
                _audit_prompt(candidate, replay_result, compact, tool_doc, backend_source),
                model=model,
                max_tokens=max_tokens,
            )
            if audit.get("valid") is True:
                result = {
                    "status": "accepted",
                    "candidate": candidate,
                    "trace": replay_result.trace,
                    "branch_audit": audit,
                    "debug_rounds": debug_round,
                    "diversity": diversity,
                }
                if memory:
                    try:
                        diversity["update"] = memory.remember(
                            result, scope, str(trajectory.get("id", "")), ask, model
                        )
                    except Exception as exc:
                        diversity["update"] = {"stored": False, "warning": f"{type(exc).__name__}: {str(exc)[:200]}"}
                return result
        if debug_round >= max_debug:
            return {
                "status": "rejected",
                "candidate": candidate,
                "trace": replay_result.trace,
                "errors": replay_result.errors,
                "branch_audit": audit,
                "debug_rounds": debug_round,
                "diversity": diversity,
            }
        repair = ask(
            _debug_prompt(candidate, replay_result, audit, compact, tool_doc, backend_source, guide),
            model=model,
            max_tokens=max_tokens,
        )
        repaired_values = _binding_answer(repair, values)
        if not isinstance(repaired_values, Mapping):
            raise GroundingError("Repair response has invalid placeholder_values")
        repaired_values = dict(repaired_values)
        hydrated = hydrate_compact(compact, repaired_values)
        candidate = materialize_candidate(
            compact, hydrated, repair, repaired_values, allow_sequence_changes=True
        )
        values = repaired_values
    raise AssertionError("unreachable")


def _iter_trajectories(input_dir: Path, trajectory: Path | None) -> Iterable[tuple[Path, dict[str, Any]]]:
    if trajectory:
        yield trajectory, _read_json(trajectory)
        return
    for path in sorted(input_dir.rglob("*.json")):
        if path.name == "manifest.json":
            continue
        yield path, _read_json(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--trajectory", type=Path)
    source.add_argument("--input-dir", type=Path)
    parser.add_argument("--tool-doc", type=Path, required=True)
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--backend-class")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--max-debug", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=12000)
    parser.add_argument("--request-timeout", type=int, default=300,
                        help="HTTP connect/read timeout per model request in seconds (default: 300)")
    parser.add_argument("--resume", action="store_true",
                        help="Keep accepted cases in output-dir; retry failures and generate unfinished cases")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--memory", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable persistent diversity memory (default: enabled; --no-memory disables it)")
    parser.add_argument("--memory-path", type=Path,
                        default=Path(__file__).resolve().parents[2] / "diversity_memory.json",
                        help="Shared pool, containing one global summary and structured accepted-case features")
    parser.add_argument("--memory-summary-chars", type=int, default=100,
                        help="Global summary character limit, including punctuation (1-100; default: 100)")
    parser.add_argument("--memory-seed", type=int, default=42,
                        help="Seed for reproducible retrieval sampling")
    parser.add_argument("--memory-max-tokens", type=int, default=MODEL_TOKENS,
                        help="Token cap per short memory call, including provider reasoning (default: 2048)")
    parser.add_argument("--memory-model", help="Optional model for short memory calls (default: same as --model)")
    args = parser.parse_args(argv)
    if args.max_debug < 0 or args.max_tokens < 1:
        parser.error("max-debug must be nonnegative and max-tokens must be positive")
    if args.request_timeout < 1:
        parser.error("request-timeout must be positive")
    if not 1 <= args.memory_summary_chars <= 100:
        parser.error("memory-summary-chars must be between 1 and 100")
    if args.memory_max_tokens < 1:
        parser.error("memory-max-tokens must be positive")
    memory = (DiversityMemory(args.memory_path.resolve(), summary_chars=args.memory_summary_chars,
                              seed=args.memory_seed, max_tokens=args.memory_max_tokens,
                              model=args.memory_model) if args.memory else None)
    tool_map = load_tool_document(args.tool_doc)
    tool_doc = _read_json_or_jsonl(args.tool_doc)
    input_dir = args.input_dir.resolve() if args.input_dir else None
    output_dir = args.output_dir.resolve()
    previous = None
    if args.resume:
        if not (output_dir / "manifest.json").is_file():
            parser.error("--resume requires an existing manifest.json in output-dir")
        previous = _read_json(output_dir / "manifest.json")
        for field in ("tool_doc", "backend"):
            if Path(previous[field]).resolve() != getattr(args, field).resolve():
                parser.error(f"--resume requires the same {field} path")
        for field in ("input_dir", "trajectory", "backend_class"):
            current = getattr(args, field)
            current = str(current.resolve()) if isinstance(current, Path) else current
            if field in previous and previous[field] != current:
                parser.error(f"--resume requires the same {field}")
    elif output_dir.exists() and any(output_dir.iterdir()):
        parser.error("Output directory must be empty (or use --resume)")
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = list(_iter_trajectories(input_dir, args.trajectory))
    if args.limit is not None:
        entries = entries[:args.limit]
    entries = [(str(path.name if args.trajectory else path.relative_to(input_dir)), trajectory)
               for path, trajectory in entries]
    retained = {}
    retry_sources = set()
    snapshot = None
    if previous is not None:
        snapshot = output_dir / "_run_history" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        _write_json(snapshot / "manifest.json", previous)
        retry_sources.update(item["source"] for item in previous.get("failures", []))
        retry_sources.update(item["source"] for item in previous.get("cases", [])
                             if item.get("status") != "accepted")
        for relative, _ in entries:
            destination = output_dir / relative
            if not destination.is_file():
                continue
            try:
                saved = _read_json(destination)
            except (ValueError, OSError):
                saved = {}
            if saved.get("status") == "accepted" and saved.get("source") == relative:
                retained[relative] = {"source": relative, "status": "accepted",
                                      "debug_rounds": saved.get("debug_rounds", 0)}
            else:
                retry_sources.add(relative)
                archived = snapshot / relative
                archived.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(destination, archived)
        retry_sources.intersection_update(relative for relative, _ in entries if relative not in retained)
    pending = [(relative, trajectory) for relative, trajectory in entries if relative not in retained]
    pending.sort(key=lambda entry: entry[0] not in retry_sources)
    manifest: dict[str, Any] = {
        "status": "running",
        "requested": len(entries),
        "accepted": len(retained),
        "rejected": 0,
        "failures": [],
        "model": args.model,
        "max_debug": args.max_debug,
        "max_tokens": args.max_tokens,
        "request_timeout": args.request_timeout,
        "resume": {"enabled": args.resume, "retained_accepted": len(retained),
                   "scheduled": len(pending), "retry_failed_sources": sorted(retry_sources),
                   "previous_manifest": str((snapshot / "manifest.json").relative_to(output_dir)) if snapshot else None},
        "diversity_memory": {"enabled": args.memory,
                             "path": str(args.memory_path.resolve()) if args.memory else None,
                             "summary_chars": args.memory_summary_chars,
                             "guide_chars": 50, "seed": args.memory_seed,
                             "max_tokens": args.memory_max_tokens,
                             "model": args.memory_model or args.model},
        "tool_doc": str(args.tool_doc.resolve()),
        "backend": str(args.backend.resolve()),
        "backend_class": args.backend_class,
        "input_dir": str(input_dir) if input_dir else None,
        "trajectory": str(args.trajectory.resolve()) if args.trajectory else None,
        "general_backend_protocol": ["tool methods", "scenario loader", "public state snapshot"],
        "cases": list(retained.values()),
    }
    _write_json(output_dir / "manifest.json", manifest)
    for relative, trajectory in pending:
        manifest["current_source"] = relative
        _write_json(output_dir / "manifest.json", manifest)
        print(f"Generating {relative} (accepted={manifest['accepted']}, rejected={manifest['rejected']})", flush=True)
        try:
            result = ground_one(
                trajectory,
                tool_doc=tool_doc,
                tool_map=tool_map,
                backend_path=args.backend.resolve(),
                backend_class=args.backend_class,
                model=args.model,
                max_debug=args.max_debug,
                max_tokens=args.max_tokens,
                memory=memory,
                request_timeout=args.request_timeout,
            )
            destination = output_dir / relative
            candidate = result.get("candidate", {})
            output_case = {
                "source": str(relative),
                "status": result["status"],
                "placeholder_values": candidate.get("placeholder_values", {}),
                "replay_config": candidate.get("replay_config", {}),
                "initial_state": candidate.get("initial_state", {}),
                "tool_chain": candidate.get("steps", []),
                "trace": result.get("trace", []),
                "branch_audit": result.get("branch_audit"),
                "debug_rounds": result.get("debug_rounds", 0),
                "diversity": result.get("diversity", {"enabled": False}),
            }
            if result.get("errors"):
                output_case["errors"] = result["errors"]
            _write_json(destination, output_case)
            manifest["accepted" if result["status"] == "accepted" else "rejected"] += 1
            manifest["cases"].append({"source": str(relative), "status": result["status"], "debug_rounds": result.get("debug_rounds", 0)})
        except Exception as exc:
            manifest["rejected"] += 1
            failure = {"source": str(relative), "error": f"{type(exc).__name__}: {exc}"}
            manifest["failures"].append(failure)
            _write_json(output_dir / relative, {"status": "failed", **failure})
        _write_json(output_dir / "manifest.json", manifest)
    manifest.pop("current_source", None)
    manifest["status"] = "complete" if not manifest["failures"] and not manifest["rejected"] else "partial"
    _write_json(output_dir / "manifest.json", manifest)
    return 0 if manifest["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
