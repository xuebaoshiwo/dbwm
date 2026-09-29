"""Generate normalized state-mutation and return specifications per tool."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


BFCL_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BACKEND = (
    BFCL_ROOT
    / "bfcl_eval/eval_checker/multi_turn_eval/func_source_code/trading_bot.py"
)
DEFAULT_TOOL_SCHEMA = BFCL_ROOT / "bfcl_eval/data/multi_turn_func_doc/trading_bot.json"
DEFAULT_OUTPUT_DIR = BFCL_ROOT / "bfcl_eval/consistency/data_v2/trading_bot"
DEFAULT_MODEL_CLIENT = BFCL_ROOT.parent.parent / "utils/model_client.py"

TOP_LEVEL_KEYS = {
    "schema_version",
    "tool",
    "branches",
    "mutations",
    "returns",
    "warnings",
}
OPERATIONS = {"set", "increment", "decrement", "append", "insert", "update", "delete"}
SOURCE_KINDS = {"state_direct", "arg_direct", "derived", "constant", "external"}
TRANSFORMS = {"copy", "arithmetic", "format", "filter", "comparison", "selection", "aggregation", "collection", "unknown"}
RECOVERABILITY = {"exact", "conditional", "partial", "none", "unknown"}
REFERENCE_PREFIXES = ("$.args", "$.state_before", "$.state_after", "$.result")
CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


SYSTEM_PROMPT = r"""
You are a strict static-analysis engine for backend tool implementations. Given
backend source code, the full tool-schema inventory, and one target tool,
produce a normalized description of that tool's state dependencies.

Return exactly one JSON object. Do not emit Markdown, code fences, commentary,
or hidden reasoning. Every natural-language string in the output must be in
English.

ANALYSIS GOALS

For the target tool, identify:
1. Every ordered execution branch, including success, business errors,
   authorization failures, validation failures, and early returns.
2. The backend state fields and arguments read to select each branch.
3. Every persistent state field modified in each branch, including indirect
   helper effects, counters, collection mutations, and history records.
4. Every actual return field and whether its value is a direct state read, a
   direct argument, a derived value, a constant, or an external value.
5. Any mismatch between the tool schema and the backend implementation.

SOURCE-OF-TRUTH RULES

1. Backend source determines actual branches, mutations, and returned fields.
2. Tool schemas describe the intended contract and help interpret parameters.
3. If schema and code disagree, preserve backend behavior and add a warning.
4. Never invent state fields, branches, mutations, or outputs from domain
   knowledge.
5. Trace helpers called by the target method. Local variables are not
   persistent state, but the state fields used to compute them are dependencies.

THE ONLY ALLOWED TOP-LEVEL SHAPE

{
  "schema_version": "1.1",
  "tool": "exact target tool name",
  "branches": [],
  "mutations": [],
  "returns": [],
  "warnings": []
}

Do not add other top-level keys.

JSONPATH CONVENTION

Use only these roots:
- $.args for tool arguments
- $.state_before for backend state before the call
- $.state_after for backend state after the call
- $.result for the returned value

Examples:
- $.state_before.account.balance
- $.state_after.account.balance
- $.args.amount
- $.result.new_balance

Use JSONPath templates for dynamic keys:
- $.state_before.orders['{args.order_id}'].status
- $.state_before.resources['{args.resource_id}'].owner
- $.state_before.inventory['{item.sku}'].quantity

Do not emit Python expressions such as self.account, raw database aliases, or
unqualified field names in place of normalized paths.

BRANCHES

List branches in the backend's real evaluation and early-return order. Every
branch object must contain exactly:

{
  "id": "success_name|fallback_name|error_name",
  "if": "concise condition using normalized paths",
  "uses": ["state or argument paths read to choose this branch"],
  "description": "English explanation"
}

Branch rules:
- Every branch id must start with exactly one of these prefixes:
  - success_: the tool performs an intended normal business function. A tool
    may have multiple success_ branches when valid inputs produce materially
    different state transitions or return behavior.
  - fallback_: the call completes without an explicit backend error, but it is
    a no-op, miss, empty/neutral outcome, already-satisfied request, or response
    to input outside the tool's primary intended function.
  - error_: the backend explicitly rejects the operation, returns an error
    response, or reports a failed execution condition.
- Classify branches by backend semantics, not merely by whether the returned
  object contains a field literally named error.
- Branch ids must be lowercase snake_case after the prefix, for example
  success_created, success_sell_position_closed, fallback_already_exists, or
  error_not_authenticated.
- uses contains condition dependencies only, not fields used solely to compute
  a returned or written value.
- Use "otherwise" for an unconditional final branch; its uses may be empty.
- Preserve priority expressed by sequential if/return statements.
- Conditions may use ==, !=, >, >=, <, <=, in, not in, exists, and, or, not.
- If a branch condition uses a computed value, list the original fields and
  arguments read by that computation.

MUTATIONS

Create one mutation entry per persistent target path:

{
  "branch": "branch id",
  "target": "$.state_after path",
  "operation": "set|increment|decrement|append|insert|update|delete",
  "value_from": ["$.state_before.x", "$.args.delta"],
  "state_source_relations": [{"path": "$.state_before.x", "transform": "arithmetic", "recoverability": "conditional", "given": [], "reason": "Subtract the known delta from the observed new value on the exact numeric domain."}],
  "external_from": ["optional time, randomness, or external service source"],
  "logic": "English explanation"
}

Mutation rules:
- Use [] when the tool does not modify persistent state.
- target must start with $.state_after.
- Old state references must use $.state_before; arguments use $.args.
- append adds to a collection, insert creates a keyed or positioned object,
  update changes part of an existing object, and delete removes an item or key.
- Do not treat local-variable assignments as mutations.
- Include helper-driven changes, counters, generated records, and side effects
  that occur before returning.
- If branches modify the same target differently, create separate entries.
- For append/delete/update of a collection, include its old state in value_from
  when computing the resulting collection depends on existing contents. The
  operation name alone does not replace that state dependency.

RETURNS

Create exactly one returns entry for every branch:

{
  "branch": "branch id",
  "fields": []
}

Every field path must start with $.result. Use one of these source forms.
Every return field, including constants and argument-only fields, must also have
"state_source_relations": [] (or one entry per backend state path in its
source/value_from). Never classify branch.uses as return dependencies.

Unchanged backend state value:
{
  "path": "$.result.field",
  "source_kind": "state_direct",
  "source": "$.state_before.x or $.state_after.x",
  "state_source_relations": [{"path": "$.state_before.x", "transform": "copy", "recoverability": "exact", "given": [], "reason": "Returned unchanged."}],
  "logic": "English explanation"
}

Unchanged input argument:
{
  "path": "$.result.field",
  "source_kind": "arg_direct",
  "source": "$.args.x",
  "state_source_relations": [],
  "logic": "English explanation"
}

Computed, formatted, filtered, normalized, selected, or aggregated value:
{
  "path": "$.result.field",
  "source_kind": "derived",
  "value_from": ["$.state_before.account.balance", "$.state_before.account.fee"],
  "state_source_relations": [
    {"path": "$.state_before.account.balance", "transform": "arithmetic", "recoverability": "conditional", "given": ["$.state_before.account.fee"], "reason": "For a net=balance-fee result, add the known fee to recover balance on the exact numeric domain."},
    {"path": "$.state_before.account.fee", "transform": "arithmetic", "recoverability": "conditional", "given": ["$.state_before.account.balance"], "reason": "For a net=balance-fee result, subtract net from the known balance on the exact numeric domain."}
  ],
  "external_from": ["optional external sources"],
  "logic": "English explanation"
}

Literal value from source code:
{
  "path": "$.result.field",
  "source_kind": "constant",
  "value": "the actual literal value",
  "state_source_relations": [],
  "logic": "English explanation"
}

Value produced entirely by time, randomness, or an external service:
{
  "path": "$.result.field",
  "source_kind": "external",
  "external_from": ["specific external source"],
  "state_source_relations": [],
  "logic": "English explanation"
}

Return rules:
- source_kind is the direct-read marker. Use state_direct only for an unchanged
  backend-state value. copy or deepcopy and unchanged local forwarding still
  count as direct. Arithmetic, formatting, type conversion, filtering, sorting,
  aggregation, normalization, and conditional selection are derived.
- A value read after a successful mutation uses $.state_after. A value read
  from pre-call state uses $.state_before.
- Literal success and error strings are constant. Their condition fields belong
  in branch.uses.
- If the backend returns an entire mapping or list unchanged, one field may
  describe that object. If code constructs explicit fields, describe each.
- Record actual backend field names. Do not rename them to match the schema.
- A branch returning a scalar or list still needs a field; use $.result for the
  whole returned value.

STATE SOURCE RELATIONS (REQUIRED FOR BOTH RETURNS AND MUTATIONS)

For each state_before/state_after path actually listed in that item's source
or value_from, add exactly one object to state_source_relations:
{"path": "same normalized state path", "transform": "copy|arithmetic|format|filter|comparison|selection|aggregation|collection|unknown", "recoverability": "exact|conditional|partial|none|unknown", "given": ["other state paths that must be known"], "reason": "English explanation"}

Interpret recoverability with the produced return field observed, or for a
mutation with its state_after target observed. Tool arguments are known; list
only OTHER backend state paths in given, not args or the source itself.
- exact: the full source value is returned unchanged (including unchanged
  copies). Only use with transform=copy; a direct parent object also exposes
  its descendants, but a partial projection does not expose its parent.
- conditional: the source can be uniquely recovered given listed co-sources,
  known arguments, and any explicitly stated type/domain assumptions. Explain
  the inverse in reason. A derived value with one source is NOT automatically
  conditional: int conversion, rounding, truncation and normalization may
  discard information.
- partial: the result constrains the source (bounds, membership, subsets,
  rounded values, or branch-specific selection) but does not determine it.
- none: the result conveys no usable information about the full source.
- unknown: the backend does not justify a stronger claim. Prefer unknown to
  an unjustified inverse.
Use transform=comparison for comparisons/thresholds, filter for subset/window
selection, arithmetic for numeric formulas, format for string formatting,
selection for conditional choice or lookup, aggregation for summaries, and
collection for appends, inserts and deletes. Explain non-injective behavior.
For writes, recoverability is about reconstructing PRE-WRITE source from the
observed POST-WRITE target, not whether forward computation is deterministic.
An external random/time source prevents a conditional inverse unless its
effect can be separated or is independently observed; explain any assumption.
Do not mark a field exactly recoverable just because it is the only source.

WARNINGS

Use [] when there is no issue. Each warning must contain at least:

{
  "type": "schema_backend_mismatch|ambiguous_state|dynamic_behavior|unsupported_construct",
  "description": "English explanation"
}

For schema/backend field conflicts, also include schema_path and backend_path
when possible.

FINAL SELF-CHECK

- The top-level object has exactly the six required keys.
- schema_version is 1.1. Every return field and mutation has the exact state
  source_relation coverage required above, with no branch-only dependencies.
- tool exactly matches the requested tool name.
- Every mutation and returns entry references an existing branch.
- Every branch has exactly one returns entry.
- Every branch id starts with success_, fallback_, or error_, and its prefix
  matches the backend semantics of that path.
- All persistent mutations and actual return paths are covered.
- No local variable is mislabeled as persistent state.
- state_direct is never used for a transformed value.
- Every state, argument, and result reference follows the normalized roots.
- All descriptions, logic text, conditions, warning text, and branch ids are
  English-only.
- The response is one valid JSON object parseable by a standard JSON parser.
""".strip()


def load_model_client(path: Path) -> tuple[Callable[..., str], str]:
    spec = importlib.util.spec_from_file_location("tool_state_spec_model_client", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load model client at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "chat_text"):
        raise RuntimeError("Model client must expose chat_text")
    return module.chat_text, getattr(module, "DEFAULT_MODEL", "unspecified")


def load_tool_schemas(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        position = 0
        value = []
        while position < len(text):
            while position < len(text) and text[position].isspace():
                position += 1
            if position == len(text):
                break
            item, position = decoder.raw_decode(text, position)
            value.append(item)
    if isinstance(value, dict):
        value = value["tools"] if isinstance(value.get("tools"), list) else [value]
    if not isinstance(value, list) or not value:
        raise ValueError("Tool schema must contain at least one tool object")
    schemas = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise ValueError(f"Tool schema entry {index} needs a string name")
        schemas.append(item)
    names = [item["name"] for item in schemas]
    if len(names) != len(set(names)):
        raise ValueError("Tool schema contains duplicate tool names")
    return schemas


def build_prompt(
    backend_source: str,
    schemas: list[dict[str, Any]],
    target_schema: dict[str, Any],
    previous_error: str | None = None,
) -> str:
    correction = ""
    if previous_error:
        correction = (
            "\nThe previous response failed structural validation. Reanalyze the "
            f"tool and correct this error:\n{previous_error}\n"
        )
    return f"""
Analyze the tool named {target_schema['name']!r}. Return only the required JSON object.
{correction}
<target_tool_schema>
{json.dumps(target_schema, ensure_ascii=False, indent=2)}
</target_tool_schema>

<all_tool_schemas>
{json.dumps(schemas, ensure_ascii=False, indent=2)}
</all_tool_schemas>

<backend_source>
{backend_source}
</backend_source>

Analyze the backend method matching the target tool, every helper it calls,
instance or class state it reads, indirect side effects, and every return path.
The remaining schemas provide contract context only; do not invent cross-tool
behavior for the target tool.
""".strip()


def parse_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    fence = chr(96) * 3
    if text.startswith(fence):
        first_newline = text.find("\n")
        text = text[first_newline + 1 :] if first_newline >= 0 else text
        if text.rstrip().endswith(fence):
            text = text.rstrip()[:-3]
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start < 0:
            raise ValueError("Model response does not contain a JSON object")
        try:
            value, end = json.JSONDecoder().raw_decode(text, start)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Model response is not valid JSON: {exc}") from exc
        if text[end:].strip().strip(chr(96)):
            raise ValueError("Model response contains trailing non-JSON content")
    if not isinstance(value, dict):
        raise ValueError("Model response must be a JSON object")
    return value


def _require_string(value: Any, location: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty string")
    if CJK_PATTERN.search(value):
        raise ValueError(f"{location} must be English-only")


def _require_string_list(value: Any, location: str, *, english_only: bool = False) -> None:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{location} must be an array of strings")
    if english_only and any(CJK_PATTERN.search(item) for item in value):
        raise ValueError(f"{location} must be English-only")


def _validate_reference(
    path: str,
    location: str,
    prefixes: tuple[str, ...] = REFERENCE_PREFIXES,
) -> None:
    if not path.startswith(prefixes):
        raise ValueError(f"{location} has invalid path root: {path}")


def _state_references(paths: list[str]) -> set[str]:
    return {path for path in paths if path.startswith(("$.state_before.", "$.state_before[", "$.state_after.", "$.state_after["))}


def _validate_source_relations(item: dict[str, Any], paths: list[str], location: str,
                               *, direct: bool = False, required: bool = True) -> None:
    relations = item.get("state_source_relations")
    if not required and relations is None:
        return
    if not isinstance(relations, list):
        raise ValueError(f"{location}.state_source_relations must be an array")
    expected = _state_references(paths)
    found: set[str] = set()
    for index, relation in enumerate(relations):
        label = f"{location}.state_source_relations[{index}]"
        if not isinstance(relation, dict) or set(relation) != {"path", "transform", "recoverability", "given", "reason"}:
            raise ValueError(f"{label} must have exactly path, transform, recoverability, given, reason")
        path = relation["path"]
        _require_string(path, f"{label}.path")
        if path not in expected or path in found:
            raise ValueError(f"{label}.path must uniquely match a state source in this item")
        found.add(path)
        if relation["transform"] not in TRANSFORMS:
            raise ValueError(f"{label}.transform must be a supported transform")
        if relation["recoverability"] not in RECOVERABILITY:
            raise ValueError(f"{label}.recoverability must be a supported level")
        if (relation["recoverability"] == "exact") != (relation["transform"] == "copy"):
            raise ValueError(f"{label}: exact recoverability requires copy, and copy requires exact")
        if direct and (relation["transform"], relation["recoverability"]) != ("copy", "exact"):
            raise ValueError(f"{label}: state_direct must be an exact copy")
        _require_string_list(relation["given"], f"{label}.given")
        if len(relation["given"]) != len(set(relation["given"])) or any(
            other not in expected - {path} for other in relation["given"]
        ):
            raise ValueError(f"{label}.given must list distinct other state sources from this item")
        _require_string(relation["reason"], f"{label}.reason")
    if found != expected:
        raise ValueError(f"{location}.state_source_relations coverage mismatch: missing={sorted(expected - found)}")


def validate_spec(spec: dict[str, Any], tool_name: str) -> None:
    if set(spec) != TOP_LEVEL_KEYS:
        missing = sorted(TOP_LEVEL_KEYS - set(spec))
        extra = sorted(set(spec) - TOP_LEVEL_KEYS)
        raise ValueError(f"Top-level keys mismatch; missing={missing}, extra={extra}")
    if spec["schema_version"] not in {"1.0", "1.1"}:
        raise ValueError("schema_version must be '1.0' or '1.1'")
    current = spec["schema_version"] == "1.1"
    if spec["tool"] != tool_name:
        raise ValueError(f"tool must be {tool_name!r}")
    for key in ("branches", "mutations", "returns", "warnings"):
        if not isinstance(spec[key], list):
            raise ValueError(f"{key} must be an array")

    branch_ids: list[str] = []
    for index, branch in enumerate(spec["branches"]):
        location = f"branches[{index}]"
        expected_keys = {"id", "if", "uses", "description"}
        if not isinstance(branch, dict) or set(branch) != expected_keys:
            raise ValueError(f"{location} must contain exactly {sorted(expected_keys)}")
        for key in ("id", "if", "description"):
            _require_string(branch[key], f"{location}.{key}")
        if not re.fullmatch(r"(?:success|fallback|error)_[a-z0-9_]+", branch["id"]):
            raise ValueError(
                f"{location}.id must use a success_, fallback_, or error_ prefix"
            )
        _require_string_list(branch["uses"], f"{location}.uses")
        for use_index, path in enumerate(branch["uses"]):
            _validate_reference(
                path,
                f"{location}.uses[{use_index}]",
                ("$.args", "$.state_before"),
            )
        branch_ids.append(branch["id"])
    if not branch_ids:
        raise ValueError("branches must not be empty")
    if len(branch_ids) != len(set(branch_ids)):
        raise ValueError("branch ids must be unique")
    branch_id_set = set(branch_ids)

    for index, mutation in enumerate(spec["mutations"]):
        location = f"mutations[{index}]"
        if not isinstance(mutation, dict):
            raise ValueError(f"{location} must be an object")
        required = {"branch", "target", "operation", "value_from", "logic"}
        allowed = required | {"external_from"} | ({"state_source_relations"} if current else set())
        if not required <= set(mutation) or not set(mutation) <= allowed:
            raise ValueError(f"{location} has invalid keys")
        if mutation["branch"] not in branch_id_set:
            raise ValueError(f"{location}.branch references an unknown branch")
        _require_string(mutation["target"], f"{location}.target")
        _validate_reference(mutation["target"], f"{location}.target", ("$.state_after",))
        if mutation["operation"] not in OPERATIONS:
            raise ValueError(f"{location}.operation must be one of {sorted(OPERATIONS)}")
        _require_string_list(mutation["value_from"], f"{location}.value_from")
        for source_index, path in enumerate(mutation["value_from"]):
            _validate_reference(
                path,
                f"{location}.value_from[{source_index}]",
                ("$.args", "$.state_before"),
            )
        if "external_from" in mutation:
            _require_string_list(
                mutation["external_from"],
                f"{location}.external_from",
                english_only=True,
            )
        _require_string(mutation["logic"], f"{location}.logic")
        _validate_source_relations(mutation, mutation["value_from"], location, required=current)

    return_branches = set()
    for index, result in enumerate(spec["returns"]):
        location = f"returns[{index}]"
        if not isinstance(result, dict) or set(result) != {"branch", "fields"}:
            raise ValueError(f"{location} must contain exactly branch and fields")
        if result["branch"] not in branch_id_set:
            raise ValueError(f"{location}.branch references an unknown branch")
        if result["branch"] in return_branches:
            raise ValueError(f"{location}.branch is duplicated")
        return_branches.add(result["branch"])
        if not isinstance(result["fields"], list) or not result["fields"]:
            raise ValueError(f"{location}.fields must be a non-empty array")
        for field_index, field in enumerate(result["fields"]):
            field_location = f"{location}.fields[{field_index}]"
            if not isinstance(field, dict):
                raise ValueError(f"{field_location} must be an object")
            required = {"path", "source_kind", "logic"}
            if not required <= set(field):
                raise ValueError(f"{field_location} is missing required keys")
            _require_string(field["path"], f"{field_location}.path")
            _validate_reference(field["path"], f"{field_location}.path", ("$.result",))
            kind = field["source_kind"]
            if kind not in SOURCE_KINDS:
                raise ValueError(
                    f"{field_location}.source_kind must be one of {sorted(SOURCE_KINDS)}"
                )
            _require_string(field["logic"], f"{field_location}.logic")
            if kind == "state_direct":
                _require_string(field.get("source"), f"{field_location}.source")
                _validate_reference(
                    field["source"],
                    f"{field_location}.source",
                    ("$.state_before", "$.state_after"),
                )
            elif kind == "arg_direct":
                _require_string(field.get("source"), f"{field_location}.source")
                _validate_reference(
                    field["source"],
                    f"{field_location}.source",
                    ("$.args",),
                )
            elif kind == "constant":
                if "value" not in field:
                    raise ValueError(f"{field_location}.value is required for constant")
                if isinstance(field["value"], str) and CJK_PATTERN.search(field["value"]):
                    raise ValueError(f"{field_location}.value must be English-only")
            elif kind == "derived":
                _require_string_list(
                    field.get("value_from"),
                    f"{field_location}.value_from",
                )
                for source_index, path in enumerate(field["value_from"]):
                    _validate_reference(
                        path,
                        f"{field_location}.value_from[{source_index}]",
                    )
                if "external_from" in field:
                    _require_string_list(
                        field["external_from"],
                        f"{field_location}.external_from",
                        english_only=True,
                    )
            elif kind == "external":
                _require_string_list(
                    field.get("external_from"),
                    f"{field_location}.external_from",
                    english_only=True,
                )
            if current:
                dependencies = ([field["source"]] if kind in {"state_direct", "arg_direct"}
                                else field.get("value_from", []))
                _validate_source_relations(field, dependencies, field_location,
                                           direct=kind == "state_direct")
            elif "state_source_relations" in field:
                raise ValueError(f"{field_location}.state_source_relations requires schema_version 1.1")

    if return_branches != branch_id_set:
        missing = sorted(branch_id_set - return_branches)
        raise ValueError(f"Every branch must have one returns entry; missing={missing}")

    for index, warning in enumerate(spec["warnings"]):
        location = f"warnings[{index}]"
        if not isinstance(warning, dict):
            raise ValueError(f"{location} must be an object")
        _require_string(warning.get("type"), f"{location}.type")
        _require_string(warning.get("description"), f"{location}.description")


def generate_spec(
    chat_text: Callable[..., str],
    backend_source: str,
    schemas: list[dict[str, Any]],
    target_schema: dict[str, Any],
    *,
    model: str | None,
    retries: int,
    max_tokens: int,
) -> tuple[dict[str, Any], int]:
    previous_error = None
    errors = []
    for attempt in range(1, retries + 1):
        prompt = build_prompt(backend_source, schemas, target_schema, previous_error)
        try:
            kwargs: dict[str, Any] = {
                "system": SYSTEM_PROMPT,
                "thinking": True,
                "temperature": 0.0,
                "max_tokens": max_tokens,
                "timeout": 240,
                "response_format": {"type": "json_object"},
            }
            if model:
                kwargs["model"] = model
            raw = chat_text(prompt, **kwargs)
            result = parse_json_object(raw)
            validate_spec(result, target_schema["name"])
            return result, attempt
        except Exception as exc:
            previous_error = f"{type(exc).__name__}: {exc}"
            errors.append(previous_error)
            if attempt < retries:
                time.sleep(min(2 ** (attempt - 1), 8))
    raise RuntimeError(f"Failed after {retries} attempts: {' | '.join(errors)}")


def _safe_filename(tool_name: str) -> str:
    filename = re.sub(r"[^A-Za-z0-9_.-]+", "_", tool_name).strip("._")
    if not filename:
        raise ValueError(f"Cannot create a filename for tool {tool_name!r}")
    return filename + ".json"


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_generation(
    backend_path: Path,
    tool_schema_path: Path,
    output_dir: Path,
    model_client_path: Path,
    *,
    workers: int,
    retries: int,
    model: str | None,
    selected_tools: set[str] | None,
    overwrite: bool,
    max_tokens: int,
) -> dict[str, Any]:
    backend_source = backend_path.read_text(encoding="utf-8-sig")
    schemas = load_tool_schemas(tool_schema_path)
    if selected_tools is not None:
        available = {schema["name"] for schema in schemas}
        missing = sorted(selected_tools - available)
        if missing:
            raise ValueError(f"Unknown tools requested: {missing}")
        schemas_to_generate = [
            schema for schema in schemas if schema["name"] in selected_tools
        ]
    else:
        schemas_to_generate = schemas

    chat_text, client_default_model = load_model_client(model_client_path)
    effective_model = model or client_default_model
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: dict[str, dict[str, Any]] = {}
    attempts: dict[str, int] = {}
    skipped = []
    pending = []
    for schema in schemas_to_generate:
        destination = output_dir / _safe_filename(schema["name"])
        if destination.exists() and not overwrite:
            existing = json.loads(destination.read_text(encoding="utf-8"))
            validate_spec(existing, schema["name"])
            if existing["schema_version"] == "1.1":
                generated[schema["name"]] = existing
                skipped.append(schema["name"])
            else:
                pending.append(schema)
        else:
            pending.append(schema)

    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                generate_spec,
                chat_text,
                backend_source,
                schemas,
                schema,
                model=model,
                retries=retries,
                max_tokens=max_tokens,
            ): schema
            for schema in pending
        }
        completed = 0
        for future in as_completed(futures):
            schema = futures[future]
            tool_name = schema["name"]
            completed += 1
            try:
                result, attempt_count = future.result()
                generated[tool_name] = result
                attempts[tool_name] = attempt_count
                _write_json_atomic(output_dir / _safe_filename(tool_name), result)
                print(
                    f"validated {completed}/{len(pending)} {tool_name} "
                    f"(attempts={attempt_count})",
                    flush=True,
                )
            except Exception as exc:
                failures[tool_name] = f"{type(exc).__name__}: {exc}"
                print(
                    f"failed {completed}/{len(pending)} {tool_name}: "
                    f"{failures[tool_name]}",
                    flush=True,
                )

    ordered_specs = [
        generated[schema["name"]]
        for schema in schemas_to_generate
        if schema["name"] in generated
    ]
    _write_json_atomic(output_dir / "all_tools.json", ordered_specs)
    manifest = {
        "schema_version": "1.1" if all(spec["schema_version"] == "1.1" for spec in ordered_specs) else "mixed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "backend": str(backend_path.resolve()),
        "tool_schema": str(tool_schema_path.resolve()),
        "model_client": str(model_client_path.resolve()),
        "model": effective_model,
        "thinking": True,
        "workers": workers,
        "requested_tools": len(schemas_to_generate),
        "generated_tools": len(ordered_specs),
        "skipped_existing": skipped,
        "attempts": attempts,
        "failures": failures,
    }
    _write_json_atomic(output_dir / "manifest.json", manifest)
    if failures:
        raise RuntimeError(
            f"Generation failed for {len(failures)} tools; see manifest.json"
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", type=Path, default=DEFAULT_BACKEND)
    parser.add_argument("--tool-schema", type=Path, default=DEFAULT_TOOL_SCHEMA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-client", type=Path, default=DEFAULT_MODEL_CLIENT)
    parser.add_argument("--model", default=None, help="Override model_client.DEFAULT_MODEL")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=8000)
    parser.add_argument("--tools", nargs="+", help="Generate only the named tools")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.retries < 1 or args.max_tokens < 1:
        parser.error("workers, retries, and max-tokens must be positive")
    manifest = run_generation(
        args.backend,
        args.tool_schema,
        args.output_dir,
        args.model_client,
        workers=args.workers,
        retries=args.retries,
        model=args.model,
        selected_tools=set(args.tools) if args.tools else None,
        overwrite=args.overwrite,
        max_tokens=args.max_tokens,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
