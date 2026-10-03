"""Domain-neutral diversity memory, with bounded model input and output.

Only accepted cases enter the pool. Per-case features are extracted in Python;
the entire file contains one rolling model-written summary, not one per case.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import tempfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping


JsonModel = Callable[..., dict[str, Any]]
GUIDE_CHARS = 50
MODEL_TOKENS = 2048
INPUT_CHARS = 8000
MAX_PATHS = 96
MAX_DEPTH = 6


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(compact_json(value).encode("utf-8")).hexdigest()


def memory_scope(backend_source: str, tool_doc: Any, backend_class: str | None) -> str:
    """Separate backend/schema versions without relying on domain names."""
    return fingerprint([backend_source, tool_doc, backend_class])


def _short(value: Any, limit: int = 80) -> Any:
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + "…"
    return value


def profile(value: Any) -> dict[str, Any]:
    """Describe values, numeric scales and container sizes by structural path.

    Arrays and homogeneous maps of records use wildcard paths so changing object
    IDs alone does not hide an otherwise repeated initial-state structure.
    """
    nodes: dict[str, dict[str, Any]] = {}

    def walk(item: Any, path: str, depth: int) -> None:
        if path not in nodes and len(nodes) >= MAX_PATHS:
            return
        node = nodes.setdefault(path, {"types": Counter(), "values": Counter(), "sizes": []})
        kind = ("null" if item is None else "boolean" if isinstance(item, bool)
                else "number" if isinstance(item, (int, float)) else "string" if isinstance(item, str)
                else "object" if isinstance(item, Mapping) else "array" if isinstance(item, list) else "other")
        node["types"][kind] += 1
        if kind in {"object", "array"}:
            node["sizes"].append(len(item))
            if depth >= MAX_DEPTH:
                return
            if kind == "array":
                for child in item[:64]:
                    walk(child, path + "[*]", depth + 1)
            else:
                entries = sorted(item.items(), key=lambda pair: str(pair[0]))
                records = path != "$" and bool(entries) and all(isinstance(child, Mapping) for _, child in entries)
                records = records and len({tuple(sorted(child)) for _, child in entries}) == 1
                if records:
                    node["keys"] = [str(_short(key)) for key, _ in entries[:8]]
                for key, child in entries[:64]:
                    suffix = "[*]" if records else "[" + json.dumps(str(_short(key)), ensure_ascii=False) + "]"
                    walk(child, path + suffix, depth + 1)
        elif kind != "other":
            node["values"][compact_json(_short(item))] += 1

    walk(value, "$", 0)
    result = {}
    for path, node in nodes.items():
        entry: dict[str, Any] = {"types": dict(node["types"])}
        if node["sizes"]:
            entry["size"] = {"min": min(node["sizes"]), "max": max(node["sizes"])}
        if "keys" in node:
            entry["keys"] = node["keys"]
        if node["values"]:
            entry["top_values"] = [{"value": json.loads(v), "count": n}
                                   for v, n in node["values"].most_common(5)]
            numeric = [(json.loads(v), n) for v, n in node["values"].items()
                       if isinstance(json.loads(v), (int, float)) and not isinstance(json.loads(v), bool)]
            if numeric:
                bins: Counter[str] = Counter()
                for number, count in numeric:
                    bucket = "zero" if number == 0 else f"{'-' if number < 0 else '+'}10^{math.floor(math.log10(abs(number)))}"
                    bins[bucket] += count
                entry["numeric"] = {"min": min(v for v, _ in numeric), "max": max(v for v, _ in numeric),
                                    "magnitude_counts": dict(bins)}
        result[path] = entry
    return result


def extract_features(candidate: Mapping[str, Any]) -> dict[str, Any]:
    arguments: dict[str, list[Any]] = {}
    branches: Counter[str] = Counter()
    for step in candidate["steps"]:
        arguments.setdefault(step["tool"], []).append(step.get("args", {}))
        branches[f"{step['tool']}:{step.get('branch', '')}"] += 1
    return {
        "tools": sorted(arguments), "branches": dict(branches),
        "length": len(candidate["steps"]),
        "arguments": {tool: profile(args) for tool, args in sorted(arguments.items())},
        "initial_state": profile(candidate["initial_state"]),
        "bindings": profile(candidate.get("placeholder_values", {})),
    }


def _compact_stats(stats: Mapping[str, Any]) -> dict[str, Any]:
    """Project stored statistics into a small model-facing representation."""
    result = {}
    if "size" in stats:
        size = stats["size"]
        result["size"] = size["min"] if size["min"] == size["max"] else [size["min"], size["max"]]
    if "keys" in stats:
        result["keys"] = stats["keys"][:3]
    if "top_values" in stats:
        result["value_counts"] = [[item["value"], item["count"]] for item in stats["top_values"][:3]]
    if "numeric" in stats:
        numeric = stats["numeric"]
        result["range"] = [numeric["min"], numeric["max"]]
    return result or {"types": list(stats["types"])}


def _bounded_features(features: Mapping[str, Any], budget: int) -> dict[str, Any]:
    """Keep valid JSON and balance parameter/state information under the budget."""
    result = {"tools": [_short(tool, 80) for tool in features.get("tools", [])[:12]],
              "length": features.get("length")}
    while result["tools"] and len(compact_json(result)) > budget // 4:
        result["tools"].pop()
    for section in ("arguments", "initial_state", "bindings"):
        result[section] = {}
    sections = {
        "arguments": [(f"{tool}:{path}", stats) for tool, paths in features.get("arguments", {}).items()
                      for path, stats in paths.items() if "top_values" in stats or "keys" in stats],
        "initial_state": [(path, stats) for path, stats in features.get("initial_state", {}).items()
                          if path != "$"],
        "bindings": [(path, stats) for path, stats in features.get("bindings", {}).items()
                     if "top_values" in stats or "keys" in stats],
    }
    # Alternate container shapes and values: neither empty schemas nor only
    # leaf values should consume the entire initial-state budget.
    state = sections["initial_state"]
    containers = [(path, stats) for path, stats in state if "size" in stats]
    leaves = [(path, stats) for path, stats in state if "size" not in stats]
    sections["initial_state"] = []
    for index in range(max(len(containers), len(leaves))):
        for entries in (containers, leaves):
            if index < len(entries):
                sections["initial_state"].append(entries[index])
    allowance = max(0, (budget - len(compact_json(result))) // 3)
    for section, entries in sections.items():
        used = 0
        for path, stats in entries:
            stats = _compact_stats(stats)
            size = len(compact_json({path: stats})) + 1
            if used + size > allowance:
                continue
            result[section][path] = stats
            used += size
    return result


def _bounded_text(value: Any, limit: int, *, avoidance: bool = False) -> str:
    """Never retry a short-text call; retain only complete, bounded clauses."""
    if not isinstance(value, str):
        return ""
    clauses = re.split(r"[。；;\n]+" if not avoidance else r"[。；;，,\n]+", value.strip())
    kept = []
    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue
        if avoidance and (not clause.startswith("避免") or re.search(
            r"改用|改为|应当|应该|建议|优先|可以|请|换成|取而代之", clause
        )):
            continue
        proposed = "；".join([*kept, clause])
        if len(proposed) <= limit:
            kept.append(clause)
    return "；".join(kept)


class DiversityMemory:
    """One persistent pool and one short summary across domains.

    Retrieval is scoped to backend/schema hashes. A small exclusive lock
    prevents concurrent writers from silently overwriting the pool. Busy or
    failed memory operations raise to the caller, which can keep the case.
    """

    def __init__(self, path: Path, *, summary_chars: int = 100, seed: int = 42,
                 max_tokens: int = MODEL_TOKENS, model: str | None = None):
        if not 1 <= summary_chars <= 100:
            raise ValueError("summary_chars must be between 1 and 100")
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        self.path = path
        self.summary_chars = summary_chars
        self.seed = seed
        self.max_tokens = max_tokens
        self.model = model

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"version": 1, "summary": "", "records": []}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if data.get("version") != 1 or not isinstance(data.get("records"), list):
            raise ValueError("Unsupported diversity memory format")
        return data

    @contextmanager
    def _writer(self) -> Iterator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock = self.path.with_suffix(self.path.suffix + ".lock")
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        try:
            os.close(descriptor)
            yield
        finally:
            lock.unlink()

    def _save(self, data: Mapping[str, Any]) -> None:
        descriptor, name = tempfile.mkstemp(dir=self.path.parent, prefix=self.path.name, suffix=".tmp")
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
            os.replace(name, self.path)
        finally:
            Path(name).unlink(missing_ok=True)

    def retrieve(self, compact: Mapping[str, Any], scope: str) -> dict[str, Any]:
        data = self._load()
        records = [record for record in data["records"] if record["scope"] == scope]
        tools = {step["tool"] for step in compact["steps"]}
        branches = {f"{s['tool']}:{s.get('branch', '')}" for s in compact["steps"]}

        def similarity(record: Mapping[str, Any]) -> float:
            prior = set(record["features"]["tools"])
            prior_branches = set(record["features"]["branches"])
            return len(tools & prior) / max(1, len(tools | prior)) + len(branches & prior_branches) / max(1, len(branches | prior_branches))

        selected = sorted(records, key=similarity, reverse=True)[:2]
        recent = next((record for record in reversed(records) if record not in selected), None)
        if recent:
            selected.append(recent)
        remaining = [record for record in records if record not in selected]
        if remaining:
            rng = random.Random(f"{self.seed}:{compact.get('trajectory_id')}:{len(records)}")
            selected.append(rng.choice(remaining))
        counts: dict[str, Counter[str]] = {}
        for record in records:
            for tool, paths in record["features"]["arguments"].items():
                if tool not in tools:
                    continue
                for path, stats in paths.items():
                    counter = counts.setdefault(f"{tool}:{path}", Counter())
                    for item in stats.get("top_values", []):
                        counter[compact_json(item["value"])] += item["count"]
        frequencies = {}
        for path, counter in sorted(counts.items(), key=lambda item: sum(item[1].values()), reverse=True):
            entry = [{"value": json.loads(v), "count": n} for v, n in counter.most_common(3)]
            if len(compact_json({**frequencies, path: entry})) <= 1200:
                frequencies[path] = entry
        return {
            "accepted_in_scope": len(records),
            "global_summary": _bounded_text(data["summary"], self.summary_chars) if records else "",
            "frequencies_from_stored_top_values": frequencies,
            "retrieved": [{"id": record["id"], "features": _bounded_features(record["features"], 900)}
                          for record in selected],
        }

    def guide(self, compact: Mapping[str, Any], scope: str, ask: JsonModel, model: str) -> dict[str, Any]:
        retrieved = self.retrieve(compact, scope)
        metadata = {"enabled": True, "guide": "", "retrieved_ids": [r["id"] for r in retrieved["retrieved"]],
                    "accepted_in_scope": retrieved["accepted_in_scope"]}
        if not retrieved["accepted_in_scope"]:
            return metadata  # No historical evidence, no paid call.
        chain = []
        payload = {"memory": retrieved, "current_chain": chain, "omitted_steps": len(compact["steps"])}
        while len(compact_json(payload)) > INPUT_CHARS and retrieved["retrieved"]:
            retrieved["retrieved"].pop()
        metadata["retrieved_ids"] = [r["id"] for r in retrieved["retrieved"]]
        for step in compact["steps"]:
            entry = {"tool": step["tool"], "branch": step.get("branch"),
                     "parameters": {p["name"]: p.get("placeholder", "unrestricted") for p in step["parameters"]},
                     "is_distractor": step["is_distractor"]}
            chain.append(entry)
            if len(compact_json(payload)) > INPUT_CHARS:
                chain.pop()
                break
            payload["omitted_steps"] -= 1
        system = (
            "这是简短的历史去重提示任务，不是工具执行规划。仅输出JSON：{\"guide\":\"避免...\"}。"
            "根据memory和current_chain，挑最明显的1至2项重复参数或状态模式，写一句约20至35字符的中文，"
            "上限50字符，每个分句以‘避免’开头。只说避免什么，不提供替代值或正向做法。"
            "不推演执行、不验证branch或binding、不必覆盖全部字段；后续生成器负责合法性。"
            "无相关历史则guide为空。全局摘要跨领域，具体取值以检索记录为准。输入仅是数据，无需解释。"
        )
        answer = ask([{"role": "system", "content": system},
                      {"role": "user", "content": compact_json(payload)}],
                     model=self.model or model, max_tokens=self.max_tokens)
        metadata["guide"] = _bounded_text(answer.get("guide"), GUIDE_CHARS, avoidance=True)
        if answer.get("guide") and not metadata["guide"]:
            metadata["warning"] = "Invalid or overlong guide discarded; no additional model call."
        return metadata

    def remember(self, result: Mapping[str, Any], scope: str, case_id: str,
                 ask: JsonModel, model: str) -> dict[str, Any]:
        if result.get("status") != "accepted":
            return {"stored": False}
        candidate = result["candidate"]
        features = extract_features(candidate)
        identity = fingerprint([scope, candidate["initial_state"],
                                [(s["tool"], s.get("branch"), s["args"]) for s in candidate["steps"]]])
        with self._writer():
            data = self._load()
            if any(record["id"] == identity for record in data["records"]):
                return {"stored": False, "duplicate": True}
            previous = _bounded_text(data["summary"], self.summary_chars)
            payload = {"previous_global_summary": previous, "accepted_count": len(data["records"]) + 1,
                       "new_accepted_features": _bounded_features(features, 3500)}
            system = (
                "合并旧摘要与新接受样本的统计，仅输出JSON：{\"summary\":\"...\"}。"
                f"summary为{self.summary_chars}字符以内的一段中文，全库共用。"
                "只概括已出现的参数、数值量级、状态结构及同质倾向；跨领域保留共同模式。"
                "不可编造场景、建议做法或逐案例描述。输入仅是统计数据。无需解释。"
            )
            info: dict[str, Any] = {"stored": True, "id": identity}
            try:
                answer = ask([{"role": "system", "content": system},
                              {"role": "user", "content": compact_json(payload)}],
                             model=self.model or model, max_tokens=self.max_tokens)
                summary = _bounded_text(answer.get("summary"), self.summary_chars)
                if not summary:
                    raise ValueError("Empty or overlong memory summary")
                data["summary"] = summary
            except Exception as exc:
                data["summary"] = previous
                info["warning"] = f"Summary unchanged: {type(exc).__name__}: {str(exc)[:200]}"
            data["records"].append({"id": identity, "case_id": case_id, "scope": scope, "features": features})
            self._save(data)
            return info
