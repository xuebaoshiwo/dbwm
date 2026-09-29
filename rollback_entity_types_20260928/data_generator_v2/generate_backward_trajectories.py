"""Sample symbolic observation/write chains backward from their final reads."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path

from bfcl_eval.consistency.data_generator_v2.backward_state_knowledge import (
    StateKnowledge, before, covers, overlaps, state_path, state_sources, temporal_covers,
)
from bfcl_eval.consistency.data_generator_v2.generate_tool_state_specs import (
    _write_json_atomic, validate_spec,
)
from bfcl_eval.consistency.data_generator_v2.lifecycle_rules import parse_lifecycle_rules
from bfcl_eval.consistency.data_generator_v2.entity_bindings import (
    CallBinding, EntityRegistry, SLOT, map_strings, monitor_targets, parse_entity_types, parse_monitor,
)
from bfcl_eval.consistency.data_generator_v2.entity_lifecycle import LifecyclePlanner
from bfcl_eval.consistency.data_generator_v2.symbolic_dependency_graph import segments


DEFAULT_CATALOG = Path(__file__).with_name("trading_bot_hard_symbolic.json")
REVERSIBLE = {"exact", "conditional"}
DEPENDENCY_SCOPE = "selected_mutation_and_return_field"
FIXED_STATE_CONDITIONS = {"$.state_before.long_context": False}
LONG_CONTEXT_DISABLED = "$.state_before.long_context == false"
LONG_CONTEXT_ENABLED = re.compile(r"\$\.state_(?:before|after)\.long_context\s*==\s*true\b", re.IGNORECASE)


@dataclass(frozen=True)
class Rule:
    source: str
    result_field: str
    requires: tuple[str, ...]
    singleton: bool
    condition: str | None = None
    reason: str = ""


@dataclass
class Branch:
    tool: str
    branch: dict
    earlier: list[dict]
    returns: list[dict]
    mutations: list[dict]
    rules: list[Rule] = field(default_factory=list)
    entity_binding: CallBinding | None = None

    def bind(self, binding):
        def transform(text):
            return SLOT.sub(lambda match: binding.slots.get(match[1], match[0]), text)

        return replace(self, branch=map_strings(self.branch, transform), earlier=map_strings(self.earlier, transform),
                       returns=map_strings(self.returns, transform), mutations=map_strings(self.mutations, transform),
                       rules=[replace(rule, source=transform(rule.source), requires=tuple(map(transform, rule.requires)),
                                      condition=transform(rule.condition) if rule.condition else None) for rule in self.rules],
                       entity_binding=binding)

    def writes(self, path):
        # A descendant write is not a whole-field write of its parent.
        return bool(self.matching_mutation_indices(path))

    def matching_mutation_indices(self, path):
        return [index for index, mutation in enumerate(self.mutations)
                if covers(mutation["target"], path)]

    def referenced_paths(self):
        return {path for rule in self.rules for path in (rule.source, *rule.requires)} | {
            source for mutation in self.mutations for source in state_sources(mutation)
        }

    def touches(self, path):
        return any(overlaps(mutation["target"], path) for mutation in self.mutations)


@dataclass
class Need:
    path: str
    depth: int
    remaining: int
    required_by: set[int] = field(default_factory=set)


class SamplingError(ValueError):
    pass


class BackwardSampler:
    """Keep one reverse-time frontier so interleaved chains share state versions."""

    def __init__(self, specs, targets, *, reader_refinements=(), lifecycle_rules=None, entity_types=None, seed=42):
        self.targets = targets
        self.lifecycle_rules = lifecycle_rules or {}
        self.entity_types = parse_entity_types(entity_types or {})
        if any(rule.entity_scoped for rule in self.lifecycle_rules.values()) and not self.entity_types:
            raise ValueError("Entity lifecycle rules require the catalog's entity_types")
        self._registry = None
        self.rng = random.Random(seed)
        refinements = {
            (item["tool"], item["branch"], item["field"], item["source"]): item
            for item in reader_refinements
        }
        used_refinements = set()
        self.nodes = []
        for spec in specs.values():
            if spec["schema_version"] not in {"1.1", "1.2"}:
                raise ValueError("Backward sampling requires schema version 1.1 or 1.2")
            for index, branch in enumerate(spec["branches"]):
                if not branch["id"].startswith("success_"):
                    continue
                node = Branch(
                    spec["tool"], branch, spec["branches"][:index],
                    next(result["fields"] for result in spec["returns"] if result["branch"] == branch["id"]),
                    [mutation for mutation in spec["mutations"] if mutation["branch"] == branch["id"]],
                )
                for returned in node.returns:
                    sources = returned["state_source_relations"]
                    observations = returned.get("state_observation_relations", [])
                    relations = [(item, True) for item in sources] + [(item, False) for item in observations]
                    for relation, is_source in relations:
                        key = (node.tool, branch["id"], returned["path"], relation["path"])
                        override = refinements.get(key)
                        if override:
                            used_refinements.add(key)
                        if relation["recoverability"] not in REVERSIBLE and not (
                            override and override["recoverability"] in REVERSIBLE
                        ):
                            continue
                        requirements = tuple(dict.fromkeys([
                            *relation["given"],
                            *(other["path"] for other in sources
                              if is_source and other["path"] != relation["path"]),
                        ]))
                        node.rules.append(Rule(
                            relation["path"], returned["path"], requirements, not requirements,
                            override.get("condition") if override else None, relation["reason"],
                        ))
                # Current specifications express enabled-mode guards as explicit conjuncts.
                if LONG_CONTEXT_ENABLED.search(branch["if"]) or any(
                    overlaps(mutation["target"], "$.state_before.long_context")
                    for mutation in node.mutations
                ):
                    continue
                self.nodes.append(node)
        if set(refinements) != used_refinements:
            raise ValueError("A reader refinement does not match the supplied specifications")
        self._option_cache = {}
        self._nodes = {(node.tool, node.branch["id"]): node for node in self.nodes}

    def candidate_nodes(self, path=None):
        for node in self.nodes:
            if self._registry is None:
                yield node
                continue
            binding = self._registry.bindings_for({"tool": node.tool, "branch": node.branch,
                "earlier": node.earlier, "returns": node.returns, "mutations": node.mutations}, path)
            if binding is not None:
                yield node.bind(binding)

    def node_for_step(self, step):
        node = self._nodes[step["tool"], step["branch"]]
        if "entity_bindings" in step:
            node = node.bind(CallBinding(step["entity_bindings"], step["symbolic_slots"]))
        return node

    @staticmethod
    def binding_metadata(node):
        if node.entity_binding is None:
            return {}
        return {"entity_bindings": dict(node.entity_binding.entities),
                "symbolic_slots": dict(node.entity_binding.slots),
                "key_bindings": copy.deepcopy(node.entity_binding.keys)}

    def closure(self, node, known=()):
        """Resolve same-call inverses to a fixed point without confusing pre/post state."""
        proofs = {path: {"kind": "earlier_state", "source": path} for path in known}
        changed = True
        while changed:
            changed = False
            for rule in node.rules:
                if (rule.source in proofs and proofs[rule.source]["kind"] != "earlier_state") or not all(
                    any(temporal_covers(path, requirement) for path in proofs)
                    for requirement in rule.requires
                ):
                    continue
                proofs[rule.source] = {
                    "kind": "return", "result_field": rule.result_field,
                    "source": rule.source, "requires": list(rule.requires),
                    "condition": rule.condition, "reason": rule.reason,
                }
                changed = True
            for path, proof in list(proofs.items()):
                prior = before(path)
                if path.startswith("$.state_after") and prior not in proofs and not node.touches(path):
                    proofs[prior] = {**proof, "kind": "unchanged_post_state", "source": prior}
                    changed = True
                after_path = path.replace("$.state_before", "$.state_after", 1)
                if path.startswith("$.state_before") and after_path not in proofs and not node.touches(path):
                    proofs[after_path] = {**proof, "source": after_path}
                    changed = True
            # An observed post-write value can fix an old source only via a declared inverse.
            for mutation in node.mutations:
                if not any(temporal_covers(path, mutation["target"]) for path in proofs):
                    continue
                relations = mutation["state_source_relations"]
                for relation in relations:
                    source = relation["path"]
                    required = [*relation["given"], *(r["path"] for r in relations if r["path"] != source)]
                    if source in proofs or relation["recoverability"] not in REVERSIBLE or not all(
                        any(temporal_covers(path, other) for path in proofs) for other in required
                    ):
                        continue
                    proofs[source] = {
                        "kind": "mutation_inverse", "source": source,
                        "observed_target": mutation["target"], "requires": required,
                        "reason": relation["reason"],
                    }
                    changed = True
        return proofs

    @staticmethod
    def proof(proofs, path):
        candidates = [(source, value) for source, value in proofs.items() if temporal_covers(source, path)]
        return min(candidates, key=lambda pair: (pair[1]["kind"] == "earlier_state",
                                                len(segments(path)) - len(segments(pair[0]))))[1] if candidates else None

    def observation_proof(self, node, proofs, path, depth, *, phase="before"):
        path = before(path) if phase == "before" else before(path).replace("$.state_before", "$.state_after", 1)
        proof = self.proof(proofs, path)
        if not proof or proof["kind"] == "earlier_state":
            return None
        if depth < 2:
            return proof
        for rule in node.rules:
            source = rule.source
            if phase == "before" and source.startswith("$.state_after"):
                if node.touches(source):
                    continue
                source = before(source)
            if rule.singleton and temporal_covers(source, path):
                return {"kind": "return", "result_field": rule.result_field, "source": source,
                        "requires": [], "condition": rule.condition, "reason": rule.reason}
        return None

    @staticmethod
    def anchor_phase(node, rule, path):
        return "after" if rule.source.startswith("$.state_after") and node.touches(path) else "before"

    def fix_options(self, path, depth):
        key = (path, depth >= 2)
        if key in self._option_cache:
            return self._option_cache[key]
        options = []
        seen = set()
        for node in self.candidate_nodes(path):
            intrinsic = self.closure(node)
            for rule in node.rules:
                source = before(rule.source)
                if not covers(source, path) or (depth >= 2 and not rule.singleton):
                    continue
                unresolved = [requirement for requirement in rule.requires
                              if not self.proof(intrinsic, requirement)]
                if any(requirement.startswith("$.state_after") and node.touches(requirement)
                       for requirement in unresolved):
                    continue
                missing = tuple(dict.fromkeys(before(requirement) for requirement in unresolved))
                candidate = (node.tool, node.branch["id"], rule.result_field, source,
                             self.anchor_phase(node, rule, path), missing, rule.condition)
                if candidate in seen:
                    continue
                seen.add(candidate)
                options.append((node, rule, missing))
        self._option_cache[key] = options
        return options

    def _can_fix(self, path, depth, trail=()):
        if any(covers(ancestor, path) and covers(path, ancestor) for ancestor in trail):
            return False
        return any(not missing or (depth < 2 and all(
            self._can_fix(source, depth + 1, (*trail, path)) for source in missing
        )) for _, _, missing in self.fix_options(path, depth))

    def _pick(self, options):
        # Sample tools, then success branches, then return fields; retain every branch.
        grouped = defaultdict(lambda: defaultdict(list))
        for option in options:
            node = option[0]
            grouped[node.tool][node.branch["id"]].append(option)
        tool = self.rng.choice(sorted(grouped))
        branch = self.rng.choice(sorted(grouped[tool]))
        return self.rng.choice(grouped[tool][branch])

    def _reuse_anchors(self, steps, chain, paths):
        """Carry observed state through source-fixed writes before retaining an anchor."""
        knowledge = StateKnowledge()
        kept = []
        by_id = {}
        removed = []

        def reuse_fixed(fixed, previous):
            origin = by_id[previous.anchor_id]
            transferred = copy.deepcopy(fixed)
            transferred["evidence"] = copy.deepcopy(previous.anchor_evidence)
            transferred["phase"] = previous.anchor_phase
            matching = next((entry for entry in origin["fixes"]
                             if entry["path"] == fixed["path"] and entry["kind"] == fixed["kind"]), None)
            if matching:
                matching["required_by"] = sorted(set(matching.get("required_by", [])) |
                                                 set(fixed.get("required_by", [])))
            else:
                origin["fixes"].append(transferred)
            if "anchor" not in origin["roles"]:
                origin["roles"].append("anchor")
            if fixed["kind"] == "initial_anchor":
                for name, path in paths.items():
                    if fixed["path"] == path:
                        chain[name]["initial_id"] = origin["_id"]
                        chain[name]["initial_phase"] = previous.anchor_phase

        for step in steps:
            node = self.node_for_step(step)
            reusable = []
            for fixed in step["fixes"]:
                previous = knowledge.get(fixed["path"])
                post_write = fixed.get("phase") == "after" and node.touches(fixed["path"])
                if not post_write and fixed["kind"] != "final_read" and previous and previous.anchor_id is not None:
                    reusable.append((fixed, previous))
            for fixed, previous in reusable:
                reuse_fixed(fixed, previous)
                step["fixes"].remove(fixed)
            if (step["role"] in {"anchor", "support_anchor"} and not node.mutations
                    and "read" not in step["roles"]
                    and not step["fixes"] and reusable):
                removed.append({"original_step_id": step["_id"], "tool": step["tool"],
                                "fields": [fixed["path"] for fixed, _ in reusable],
                                "reused_from_ids": sorted({previous.anchor_id for _, previous in reusable})})
                continue

            referenced = node.referenced_paths() | {fixed["path"] for fixed in step["fixes"]}
            proofs = self.closure(node, knowledge.known_paths(referenced))
            for fixed in step["fixes"]:
                path = fixed["path"]
                if fixed.get("phase") == "after":
                    path = before(path).replace("$.state_before", "$.state_after", 1)
                if not self.proof(proofs, path):
                    raise SamplingError(f"Unfixed observation after anchor reuse: {fixed['path']}")
            for requirement in step["write_source_requirements"]:
                source = requirement["path"]
                if not self.proof(proofs, source):
                    raise SamplingError(f"Unfixed source after anchor reuse: {source}")
                previous = knowledge.get(source)
                if previous and requirement["resolution"] == "earlier_chain":
                    requirement["evidence"] = previous.evidence(source)
            knowledge.observe(proofs, step["_id"])
            step["mutation_knowledge"] = knowledge.apply_mutations(node.mutations, proofs, step["_id"])
            if not any(fixed["kind"] != "final_read" for fixed in step["fixes"]):
                step["roles"] = [role for role in step["roles"] if role != "anchor"]
            kept.append(step)
            by_id[step["_id"]] = step

        kept_ids = set(by_id)
        for step in kept:
            step["write_targets"] = []
            for fixed in step["fixes"]:
                if "required_by" in fixed:
                    fixed["required_by"] = [item for item in fixed["required_by"] if item in kept_ids]
        for name, entry in chain.items():
            start = next(index for index, step in enumerate(kept) if step["_id"] == entry["initial_id"])
            end = next(index for index, step in enumerate(kept) if step["_id"] == entry["final_id"])
            entry["write_ids"] = []
            entry["source_fixed_write_ids"] = []
            if entry.get("initial_phase") == "after":
                start += 1
            if entry.get("final_phase") == "after":
                end += 1
            for step in kept[start:end]:
                if self.node_for_step(step).writes(paths[name]):
                    entry["write_ids"].append(step["_id"])
                    if all(status["sources_fixed"] for status in step["mutation_knowledge"]
                           if covers(status["target"], paths[name])):
                        entry["source_fixed_write_ids"].append(step["_id"])
                    step["write_targets"].append(name)
                    if "write" not in step["roles"]:
                        step["roles"].append("write")
        return kept, removed

    def _support_anchor_candidates(self, steps, paths):
        """Find finite, read-only observations suitable for pre-distractor padding."""
        core_branches = {(step["tool"], step["branch"]) for step in steps
                         if step["role"] != "distractor"}
        selected_returns = {(step["tool"], step["branch"], step.get("selected_return_field"),
                             step.get("selected_return_source")) for step in steps}
        target_roots = {segments(path)[0] for path in paths.values() if segments(path)}
        candidates = []
        seen = set()
        candidates_by_path = [node for path in paths.values() for node in self.candidate_nodes(path)]
        for node in candidates_by_path:
            branch_key = (node.tool, node.branch["id"])
            if node.mutations:
                continue
            intrinsic = self.closure(node)
            for rule in node.rules:
                if not state_path(rule.source) or (node.tool, node.branch["id"], rule.result_field,
                                                   rule.source) in selected_returns:
                    continue
                if self.proof(intrinsic, rule.source) is None:
                    continue
                if any(self.proof(intrinsic, requirement) is None for requirement in rule.requires):
                    continue
                source_root = segments(rule.source)
                same_target_dict = bool(source_root and source_root[0] in target_roots)
                if (node.tool, node.branch["id"], rule.result_field, rule.source) in seen:
                    continue
                # A same-branch return is useful even outside a target dictionary;
                # other readers are limited to source fields in selected dictionaries.
                same_branch = branch_key in core_branches
                if not same_branch and not same_target_dict:
                    continue
                seen.add((node.tool, node.branch["id"], rule.result_field, rule.source))
                candidates.append((node, rule, "same_branch_return" if same_branch else "target_dict_source"))
        self.rng.shuffle(candidates)
        return candidates

    @staticmethod
    def _support_anchor_step(node, rule, reason, step_id):
        conditions = [LONG_CONTEXT_DISABLED]
        if rule.condition:
            conditions.append(rule.condition)
        return {
            "_id": step_id, "tool": node.tool, "branch": node.branch["id"],
            "branch_condition": node.branch["if"], "branch_uses": node.branch["uses"],
            "earlier_branch_conditions_to_avoid": [branch["if"] for branch in node.earlier],
            "earlier_error_conditions_to_avoid": [branch["if"] for branch in node.earlier
                                                   if branch["id"].startswith("error_")],
            "role": "support_anchor", "roles": ["support_anchor", "anchor"],
            "field": before(rule.source), "dependency_depth": 0,
            "additional_conditions": sorted(set(conditions)),
            "return_fields": copy.deepcopy(node.returns), "mutations": [],
            "selected_return_field": rule.result_field, "selected_return_source": rule.source,
            "fix_dependencies": [],
            "fixes": [{"path": before(rule.source), "depth": 0, "kind": "support_anchor",
                       "reason": reason, "evidence": {"kind": "return", "result_field": rule.result_field,
                                                         "source": rule.source, "requires": [],
                                                         "condition": rule.condition, "reason": rule.reason}}],
            "write_targets": [], "write_source_requirements": [],
            "selected_mutation_index": None, "mutation_knowledge": [],
            **BackwardSampler.binding_metadata(node),
        }

    def build(self, names=None, *, max_writes=3, target_max_writes=None, dependency_max_writes=1,
              min_length=20, max_length=60, attempts=200, monitors=None):
        """Legacy max_writes keywords now specify minimum target write counts."""
        targets = self.targets
        if monitors is not None:
            if names is not None or not self.entity_types:
                raise ValueError("Entity monitors require entity types and cannot be combined with target names")
            targets = monitor_targets(monitors, self.entity_types)
        names = [name for name, target in targets.items() if target.get("default", True)] if names is None else list(names)
        if not names or len(set(names)) != len(names) or set(names) - set(targets):
            raise ValueError("Choose distinct names from the target catalog")
        if max_writes < 1 or dependency_max_writes < 0 or attempts < 1:
            raise ValueError("max-writes and attempts must be positive; dependency-max-writes must be nonnegative")
        if not 1 <= min_length <= max_length:
            raise ValueError("Require 1 <= min-length <= max-length")
        overrides = target_max_writes or {}
        if set(overrides) - set(names) or any(not isinstance(value, int) or value < 1 for value in overrides.values()):
            raise ValueError("Per-target write counts must be positive integers for selected targets")
        limits = {name: overrides.get(name, max_writes) for name in names}
        targets = {name: dict(targets[name]) for name in names}
        if self.entity_types:
            for name, target in targets.items():
                kind = self.entity_types[target["entity_type"]]
                target.setdefault("entity", name)
                target["path"] = kind.field_path(target["entity"], target["field"])
        failures = Counter()
        for attempt in range(1, attempts + 1):
            try:
                result = self._build_once(targets, limits, dependency_max_writes, min_length, max_length)
                result["planning"]["attempts"] = attempt
                result["planning"]["rejected_attempts"] = dict(failures)
                return result
            except SamplingError as exc:
                failures[str(exc)] += 1
        raise SamplingError(f"No chain after {attempts} attempts: {dict(failures)}")

    def _build_once(self, targets, limits, dependency_max_writes, min_length, max_length):
        names = list(targets)
        paths = {name: before(target["path"]) for name, target in targets.items()}
        self._registry = EntityRegistry(self.entity_types, targets) if self.entity_types else None
        self._option_cache.clear()
        lifecycle = LifecyclePlanner(self.lifecycle_rules, targets, self._registry, max_length)
        lifecycles = lifecycle.constrained_fields
        todo = set(names)
        active = {}
        dependencies = {}
        reverse_steps = []
        chain = {name: {"path": path, "write_ids": [], "initial_id": None, "final_id": None}
                 for name, path in paths.items()}

        def is_pending(path):
            return any(covers(need.path, path) for need in [*active.values(), *dependencies.values()])

        def add_need(path, depth, consumer):
            path = before(path)
            for need in [*active.values(), *dependencies.values()]:
                if covers(need.path, path):
                    need.required_by.add(consumer)
                    need.depth = min(need.depth, depth)
                    return
            if not self._can_fix(path, depth):
                raise SamplingError(f"No reversible fixer at depth {depth}: {path}")
            budget = self.rng.randint(0, dependency_max_writes) if depth == 1 else 0
            dependencies[path] = Need(path, depth, budget, {consumer})

        def feasible(node, depth, missing=(), mutation_index=None):
            intrinsic = self.closure(node)
            if not all(self.proof(intrinsic, source) or is_pending(source) or self._can_fix(
                before(source), depth + 1
            ) for source in missing):
                return False
            known = self.closure(node, missing)
            sources = state_sources(node.mutations[mutation_index]) if mutation_index is not None else ()
            return all(self.proof(known, source) or is_pending(source) or self._can_fix(
                before(source), depth + 1
            ) for source in sources)

        def emit(node, role, path, depth, *, rule=None, missing=(), mutation_index=None):
            # Provisional anchors can be removed later; writers and final reads cannot.
            mandatory = sum(bool(step["mutations"]) or "read" in step["roles"] for step in reverse_steps)
            if mandatory >= max_length and (node.mutations or role == "read"):
                raise SamplingError("Necessary anchors and writes exceed max-length")
            step_id = len(reverse_steps)
            resolved = self.closure(node, missing)
            conditions = sorted({LONG_CONTEXT_DISABLED, *(proof["condition"] for proof in resolved.values()
                                                         if proof.get("condition"))})
            step = {
                "_id": step_id, "tool": node.tool, "branch": node.branch["id"],
                "branch_condition": node.branch["if"], "branch_uses": node.branch["uses"],
                "earlier_branch_conditions_to_avoid": [b["if"] for b in node.earlier],
                "earlier_error_conditions_to_avoid": [b["if"] for b in node.earlier if b["id"].startswith("error_")],
                "role": role, "roles": [role], "field": path, "dependency_depth": depth,
                "additional_conditions": conditions,
                "return_fields": copy.deepcopy(node.returns), "mutations": copy.deepcopy(node.mutations),
                "selected_mutation_index": mutation_index,
                "fixes": [], "write_targets": [], "write_source_requirements": [],
                **self.binding_metadata(node),
            }
            if rule:
                step["selected_return_field"] = rule.result_field
                step["selected_return_source"] = rule.source
                step["fix_dependencies"] = list(missing)
            reverse_steps.append(step)
            if self._registry:
                self._registry.commit(node.entity_binding)
                self._option_cache.clear()
            applied = lifecycle.apply(node)
            if applied:
                step["lifecycle_transitions"] = applied
            for name, need in list(active.items()):
                phase = self.anchor_phase(node, rule, need.path) if rule and path == need.path else "before"
                proof = self.observation_proof(node, resolved, need.path, 0, phase=phase)
                if phase == "before" and node.writes(need.path):
                    need.remaining = max(0, need.remaining - 1)
                    chain[name]["write_ids"].append(step_id)
                    step["write_targets"].append(name)
                    if "write" not in step["roles"]:
                        step["roles"].append("write")
                if proof and need.remaining == 0:
                    chain[name]["initial_id"] = step_id
                    chain[name]["initial_phase"] = phase
                    step["fixes"].append({"path": need.path, "depth": 0, "kind": "initial_anchor",
                                          "phase": phase, "evidence": proof})
                    if "anchor" not in step["roles"]:
                        step["roles"].append("anchor")
                    del active[name]
            for key, need in list(dependencies.items()):
                phase = self.anchor_phase(node, rule, need.path) if rule and path == need.path else "before"
                proof = self.observation_proof(node, resolved, need.path, need.depth, phase=phase)
                if proof and need.remaining == 0:
                    step["fixes"].append({"path": need.path, "depth": need.depth,
                                          "kind": "dependency_anchor", "phase": phase,
                                          "required_by": sorted(need.required_by),
                                          "evidence": proof})
                    if "anchor" not in step["roles"]:
                        step["roles"].append("anchor")
                    del dependencies[key]
            for source in missing:
                add_need(source, depth + 1, step_id)
            if mutation_index is not None:
                mutation = node.mutations[mutation_index]
                for source in state_sources(mutation):
                    proof = self.proof(resolved, source)
                    if proof and proof["kind"] == "earlier_state":
                        proof = None
                    step["write_source_requirements"].append({
                        "path": source, "mutation_target": mutation["target"],
                        "resolution": "same_call_return" if proof else "earlier_chain",
                        "evidence": proof,
                    })
                    if not proof:
                        add_need(source, depth + 1, step_id)
            return step_id

        while todo or active or dependencies or lifecycle.pending(active):
            # Deep dependencies are fixed immediately, without further writer sampling.
            deep = [need for need in dependencies.values() if need.depth >= 2]
            tasks = [("dependency", need) for need in deep] if deep else (
                [("final", name) for name in sorted(todo)] +
                [("target", name) for name in active] +
                [("dependency", need) for need in dependencies.values()] +
                [("lifecycle", entity) for entity in lifecycle.pending(active)]
            )
            self.rng.shuffle(tasks)
            advanced = False
            for kind, subject in tasks:
                if kind == "lifecycle":
                    lifecycle_path = lifecycle.state_path(subject, lifecycle.entity_rule(subject))
                    options = [(node, index) for node in self.candidate_nodes(lifecycle_path)
                               for index in node.matching_mutation_indices(lifecycle_path)
                               if lifecycle.eligible(node) and feasible(node, 0, mutation_index=index)]
                    if not options:
                        continue
                    node, index = self._pick(options)
                    emit(node, "lifecycle_write", lifecycle_path, 0, mutation_index=index)
                elif kind == "final":
                    name = subject
                    path = paths[name]
                    options = [option for option in self.fix_options(path, 0)
                               if feasible(option[0], 0, option[2]) and lifecycle.eligible(option[0])]
                    if name in lifecycles and options:
                        scores = [lifecycle.capacity(name, limits[name], node)
                                  if node.writes(path) and self.anchor_phase(node, rule, path) == "after"
                                  else lifecycle.capacity(name, limits[name])
                                  for node, rule, _ in options]
                        best = max(scores)
                        options = [option for option, score in zip(options, scores) if score >= 0 and score == best]
                    if not options:
                        continue
                    node, rule, missing = self._pick(options)
                    step_id = emit(node, "read", path, 0, rule=rule, missing=missing)
                    chain[name]["final_id"] = step_id
                    phase = self.anchor_phase(node, rule, path)
                    chain[name]["final_phase"] = phase
                    reverse_steps[-1]["fixes"].append({
                        "path": path, "depth": 0, "kind": "final_read",
                        "phase": phase,
                        "evidence": self.observation_proof(node, self.closure(node, missing), path, 0, phase=phase),
                    })
                    todo.remove(name)
                    required = max(0, limits[name] - int(phase == "after" and node.writes(path)))
                    if name in lifecycles:
                        possible = lifecycle.capacity(name, required)
                        if possible < 0:
                            raise SamplingError(f"No lifecycle path to an initial state: {name}")
                        active[name] = Need(path, 0, possible)
                    else:
                        active[name] = Need(path, 0, required)
                else:
                    need = active[subject] if kind == "target" else subject
                    if need.remaining:
                        options = [(node, mutation_index) for node in self.candidate_nodes(need.path)
                                   for mutation_index in node.matching_mutation_indices(need.path)
                                   if feasible(node, need.depth, mutation_index=mutation_index)
                                   and lifecycle.eligible(node)]
                        if kind == "target" and subject in lifecycles and options:
                            scores = {id(node): lifecycle.capacity(subject, need.remaining, node)
                                      for node, _ in options}
                            best = max(scores.values())
                            options = [(node, index) for node, index in options
                                       if best > 0 and scores[id(node)] == best]
                            need.remaining = min(need.remaining, best)
                        if not options:
                            if kind == "dependency":
                                need.remaining = 0
                                advanced = True
                                break
                            if kind == "target" and (subject in lifecycles or self._registry):
                                need.remaining = 0
                                advanced = True
                                break
                            continue
                        node, mutation_index = self._pick(options)
                        if kind == "dependency":
                            need.remaining -= 1
                        emit(node, "write", need.path, need.depth, mutation_index=mutation_index)
                    else:
                        options = [option for option in self.fix_options(need.path, need.depth)
                                   if feasible(option[0], need.depth, option[2])
                                   and lifecycle.eligible(option[0])]
                        if not options:
                            continue
                        if kind == "target" and not lifecycle.at_initial(subject):
                            continue
                        node, rule, missing = self._pick(options)
                        emit(node, "anchor", need.path, need.depth, rule=rule, missing=missing)
                advanced = True
                break
            if not advanced:
                raise SamplingError("No next backward step satisfies source fixing and minimum target writes")

        steps, removed_anchors = self._reuse_anchors(list(reversed(reverse_steps)), chain, paths)
        core_length = len(steps)
        if core_length > max_length:
            raise SamplingError("Necessary anchors and writes exceed max-length")
        filler_remaining = max(0, min_length - core_length)
        support_candidates = self._support_anchor_candidates(steps, paths)
        support_count = min(filler_remaining, len(support_candidates))
        next_id = max((step["_id"] for step in steps), default=-1) + 1
        for node, rule, reason in support_candidates[:support_count]:
            support = self._support_anchor_step(node, rule, reason, next_id)
            next_id += 1
            steps.insert(self.rng.randrange(1, len(steps)), support)
            if self._registry:
                self._registry.commit(node.entity_binding)
        steps, support_removed = self._reuse_anchors(steps, chain, paths)
        removed_anchors.extend(support_removed)
        support_count = sum(step["role"] == "support_anchor" for step in steps)
        necessary_length = len(steps)
        if necessary_length > max_length:
            raise SamplingError("Support anchors exceed max-length")
        protected = {path for step in steps for path in [
            step["field"], *step.get("fix_dependencies", []),
            *(entry["path"] for entry in step["write_source_requirements"]),
        ]}
        distractors = []
        for node in self.candidate_nodes():
            if node.mutations:
                continue
            sources = [*node.branch["uses"], *(source for branch in node.earlier for source in branch["uses"]),
                       *(relation["path"] for returned in node.returns
                         for relation in (*returned["state_source_relations"],
                                          *returned.get("state_observation_relations", [])))]
            if not any(overlaps(source, path) for source in sources if state_path(source) for path in protected):
                distractors.append((node,))
        filler_count = max(0, min_length - necessary_length)
        if filler_count and not distractors:
            raise SamplingError("No unrelated read-only branch can fill min-length")
        for _ in range(filler_count):
            node = self._pick(distractors)[0]
            steps.insert(self.rng.randrange(1, len(steps)), {
                "tool": node.tool, "branch": node.branch["id"], "branch_condition": node.branch["if"],
                "branch_uses": node.branch["uses"],
                "earlier_branch_conditions_to_avoid": [branch["if"] for branch in node.earlier],
                "earlier_error_conditions_to_avoid": [branch["if"] for branch in node.earlier if branch["id"].startswith("error_")],
                "role": "distractor", "roles": ["distractor"], "dependency_depth": None,
                "additional_conditions": [LONG_CONTEXT_DISABLED], "return_fields": copy.deepcopy(node.returns),
                "mutations": [], "fixes": [], "write_targets": [], "write_source_requirements": [],
                "selected_mutation_index": None, "mutation_knowledge": [],
                **self.binding_metadata(node),
            })
            if self._registry:
                self._registry.commit(node.entity_binding)
        positions = {step["_id"]: index for index, step in enumerate(steps, 1) if "_id" in step}
        removed_origins = {item["original_step_id"]: item["reused_from_ids"] for item in removed_anchors}

        def retained_origins(step_id):
            if step_id in positions:
                return {positions[step_id]}
            return {origin for previous in removed_origins[step_id] for origin in retained_origins(previous)}

        for removed in removed_anchors:
            removed["reused_from_steps"] = sorted({origin for item in removed.pop("reused_from_ids")
                                                   for origin in retained_origins(item)})
        chains = {}
        for name, entry in chain.items():
            write_steps = sorted(positions[index] for index in entry["write_ids"])
            source_fixed_steps = sorted(positions[index] for index in entry["source_fixed_write_ids"])
            initial, final = positions[entry["initial_id"]], positions[entry["final_id"]]
            end = final + int(entry.get("final_phase") == "after")
            if (not self._registry and name not in lifecycles and len(write_steps) < limits[name]) or not all(
                initial <= index < end for index in write_steps
            ):
                raise SamplingError("A target's anchor/write/read interval is invalid")
            chains[name] = {"path": entry["path"], "anchor_step": initial, "read_step": final,
                            "anchor_phase": entry.get("initial_phase", "before"),
                            "read_phase": entry.get("final_phase", "before"),
                            "write_steps": write_steps, "write_count": len(write_steps),
                            "source_fixed_write_steps": source_fixed_steps,
                            "fully_source_fixed": len(source_fixed_steps) == len(write_steps),
                            "min_write_count": limits[name],
                            "write_shortfall": max(0, limits[name] - len(write_steps)),
                            "shortfall_reason": ("lifecycle_writers_exhausted" if name in lifecycles else "eligible_writers_exhausted")
                            if len(write_steps) < limits[name] else None,
                            "writer_tools": sorted({steps[index - 1]["tool"] for index in write_steps})}
        for index, step in enumerate(steps, 1):
            step.pop("_id", None)
            step["index"] = index
            for fixed in step["fixes"]:
                if "required_by" in fixed:
                    fixed["required_by"] = sorted(positions[item] for item in fixed["required_by"])
            for requirement in step["write_source_requirements"]:
                evidence = requirement["evidence"]
                if evidence and evidence["kind"] == "propagated_state":
                    anchor_id = evidence.pop("anchor_id")
                    evidence["anchor_step"] = positions[anchor_id] if anchor_id is not None else None
                    evidence["mutation_steps"] = sorted({positions[item] for item in evidence.pop("mutation_ids")})
                    evidence["observation_steps"] = sorted({positions[item] for item in evidence.pop("observation_ids")})
        writer_candidates = {}
        for name, path in paths.items():
            writer_candidates[name] = []
            for node in self.candidate_nodes(path):
                intrinsic = self.closure(node)
                for mutation_index in node.matching_mutation_indices(path):
                    mutation = node.mutations[mutation_index]
                    unfixed = sorted({source for source in state_sources(mutation)
                                      if not self.proof(intrinsic, source)
                                      and not any(covers(target, source) for target in paths.values())
                                      and not self._can_fix(before(source), 1)})
                    writer_candidates[name].append({"tool": node.tool, "branch": node.branch["id"],
                                                    "mutation_index": mutation_index,
                                                    "mutation_target": mutation["target"],
                                                    "unfixable_write_sources": unfixed})
        try:
            entity_lifecycles = lifecycle.audit(steps)
        except ValueError as exc:
            raise SamplingError(str(exc)) from exc
        entities = self._registry.describe(targets) if self._registry else {}
        return {
            "targets": [{"name": name, "path": paths[name]} for name in names],
            "monitored_entities": [{"id": entity, "type": entry["type"], "fields": sorted(entry["monitored_fields"])}
                                   for entity, entry in entities.items() if entry["monitored_fields"]],
            "state_conditions": dict(FIXED_STATE_CONDITIONS),
            "steps": steps,
            "planning": {
                "direction": "backward", "target_chains": chains,
                "dependency_scope": DEPENDENCY_SCOPE,
                "target_min_writes": limits, "write_count_mode": "best_effort_entities" if self._registry else
                "best_effort_lifecycle" if lifecycles else "at_least_minimum",
                "lifecycle_targets": sorted(lifecycles),
                "entities": entities,
                "entity_lifecycles": entity_lifecycles,
                "dependency_max_writes": dependency_max_writes,
                "min_length": min_length, "max_length": max_length,
                "core_length": core_length, "support_anchor_count": support_count,
                "necessary_length": necessary_length, "distractor_count": filler_count,
                "redundant_anchors_removed": removed_anchors,
                "writer_tools_by_target": {name: value["writer_tools"] for name, value in chains.items()},
                "writer_candidates_by_target": writer_candidates,
                "unresolved_structural_sources": [],
                "unfixed_side_effects": [
                    {"step": step["index"], **status}
                    for step in steps for status in step["mutation_knowledge"] if not status["sources_fixed"]
                ],
                "execution": "not_run", "arguments": "not_instantiated", "initial_state": "not_generated",
                "grounding_obligations": [
                    "Load the backend with long_context=False and keep that environment mode fixed throughout the trajectory.",
                    "Satisfy each selected branch condition and avoid every earlier branch.",
                    "Bind dynamic paths consistently within each target chain and its source dependencies.",
                    "Assign distinct concrete keys to distinct keyed entity instances; @TYPE:INSTANCE paths are symbolic keys.",
                    "Honor every step's key_bindings and every entity relation, including unmonitored entities.",
                    "Ground every retained side effect; unfixed_side_effects does not imply the affected state is derivable from observations.",
                    "Satisfy reversible-domain assumptions and additional reader conditions.",
                    "Instantiate time/random sources consistently with mutations and observations.",
                    "Reject and resample chains whose conditions cannot be jointly instantiated; no backend feasibility is claimed.",
                ],
            },
        }


def load_inputs(catalog_path, specs_path=None):
    catalog_path = catalog_path.resolve()
    config = json.loads(catalog_path.read_text(encoding="utf-8"))
    specs_path = specs_path or (catalog_path.parent / config["state_specs"] / "all_tools.json")
    specs_path = specs_path.resolve()
    entries = json.loads(specs_path.read_text(encoding="utf-8"))
    specs = {}
    for spec in entries:
        validate_spec(spec, spec["tool"])
        if spec["tool"] in specs:
            raise ValueError(f"Duplicate tool specification: {spec['tool']}")
        specs[spec["tool"]] = spec
    # The old writer templates, state assumptions and Pending-order condition are not used.
    return specs, config["targets"], config.get("reader_refinements", []), specs_path


def load_lifecycle_rules(catalog_path, specs, targets, override=None, disabled=False):
    if disabled:
        return {}, None
    config = json.loads(catalog_path.read_text(encoding="utf-8"))
    configured = override or config.get("lifecycle_rules")
    if not configured:
        return {}, None
    path = (Path(configured) if override else catalog_path.parent / configured).resolve()
    types = parse_entity_types(config.get("entity_types", {}))
    return parse_lifecycle_rules(json.loads(path.read_text(encoding="utf-8")), specs, targets, types), path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--specs", type=Path)
    rules = parser.add_mutually_exclusive_group()
    rules.add_argument("--lifecycle-rules", type=Path, help="Override the catalog's lifecycle rules")
    rules.add_argument("--no-lifecycle-rules", action="store_true", help="Use the original unconstrained sampler")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--targets", nargs="+")
    selection.add_argument("--monitor", action="append", metavar="INSTANCE=TYPE:FIELD[,FIELD...]",
                           help="Monitor distinct entity instances; repeat for multiple entities")
    parser.add_argument("--min-writes", "--max-writes", dest="max_writes", type=int, default=3,
                        help="Requested writes per field; best effort in entity mode (legacy alias: --max-writes)")
    parser.add_argument("--target-min-writes", "--target-max-writes", dest="target_max_writes",
                        nargs="*", default=[], metavar="NAME=N",
                        help="Override requested field writes, e.g. order_A.status=2 (legacy alias: --target-max-writes)")
    parser.add_argument("--dependency-max-writes", type=int, default=1)
    parser.add_argument("--min-length", type=int, default=20)
    parser.add_argument("--max-length", type=int, default=60)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attempts", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if min(args.max_writes, args.count, args.attempts) < 1 or args.dependency_max_writes < 0:
        parser.error("min-writes, count and attempts must be positive; dependency-max-writes must be nonnegative")
    if not 1 <= args.min_length <= args.max_length:
        parser.error("Require 1 <= min-length <= max-length")
    specs, targets, refinements, specs_path = load_inputs(args.target_catalog, args.specs)
    config = json.loads(args.target_catalog.read_text(encoding="utf-8"))
    entity_types = config.get("entity_types", {})
    lifecycle_rules, lifecycle_path = load_lifecycle_rules(
        args.target_catalog, specs, targets, args.lifecycle_rules, args.no_lifecycle_rules,
    )
    monitors = None
    if args.monitor:
        try:
            monitors = [parse_monitor(text) for text in args.monitor]
            selected = monitor_targets(monitors, parse_entity_types(entity_types))
        except ValueError as exc:
            parser.error(str(exc))
    else:
        args.targets = args.targets or ["balance", "transaction_history", "orders"]
        selected = {name: targets[name] for name in args.targets if name in targets}
    names = list(selected)
    if not names or (args.targets and (len(set(args.targets)) != len(args.targets) or set(args.targets) - set(targets))):
        parser.error("Choose distinct target names from the target catalog")
    overrides = {}
    for entry in args.target_max_writes:
        name, separator, value = entry.partition("=")
        if not separator or name not in names or not value.isdigit() or int(value) < 1 or name in overrides:
            parser.error("target-min-writes entries must be unique selected NAME=positive_integer")
        overrides[name] = int(value)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory must be empty")
    sampler = BackwardSampler(specs, targets, reader_refinements=refinements,
                              lifecycle_rules=lifecycle_rules, entity_types=entity_types, seed=args.seed)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "running", "strategy": "backward_state_source_sampling", "requested": args.count,
        "accepted": 0, "failures": [], "planning_only": True, "backend_execution": False,
        "state_conditions": dict(FIXED_STATE_CONDITIONS),
        "dependency_scope": DEPENDENCY_SCOPE,
        "seed": args.seed, "targets": names, "monitors": monitors, "min_writes": args.max_writes,
        "target_min_writes": {name: overrides.get(name, args.max_writes) for name in names},
        "write_count_mode": "best_effort_entities" if entity_types else
        "best_effort_lifecycle" if set(names) & lifecycle_rules.keys() else "at_least_minimum",
        "lifecycle_entity_types": sorted(lifecycle_rules) if entity_types else [],
        "lifecycle_rules_sha256": hashlib.sha256(lifecycle_path.read_bytes()).hexdigest() if lifecycle_path else None,
        "dependency_max_writes": args.dependency_max_writes,
        "min_length": args.min_length, "max_length": args.max_length,
        "specs_sha256": hashlib.sha256(specs_path.read_bytes()).hexdigest(),
        "catalog_sha256": hashlib.sha256(args.target_catalog.read_bytes()).hexdigest(),
        "catalog_fields_used": ["targets", "state_specs", "reader_refinements", "entity_types"] + (
            ["lifecycle_rules"] if lifecycle_path and not args.lifecycle_rules else []),
        "writer_sequences_used": False,
        "filler_strategy": ["support_anchor", "distractor"],
        "support_anchor_policy": "sample_then_forward_anchor_reuse",
    }
    _write_json_atomic(output / "manifest.json", manifest)
    for index in range(args.count):
        trajectory_id = f"backward_{args.seed}_{index:04d}"
        try:
            result = sampler.build(None if monitors else names, monitors=monitors, max_writes=args.max_writes,
                                   target_max_writes=overrides,
                                   dependency_max_writes=args.dependency_max_writes,
                                   min_length=args.min_length, max_length=args.max_length,
                                   attempts=args.attempts)
            result["id"] = trajectory_id
            result["planning"]["seed"] = args.seed
            _write_json_atomic(output / f"{trajectory_id}.json", result)
            manifest["accepted"] += 1
            print(f"{trajectory_id}: {len(result['steps'])} steps, {result['planning']['distractor_count']} distractors", flush=True)
        except SamplingError as exc:
            manifest["failures"].append({"id": trajectory_id, "reason": str(exc)})
            print(f"{trajectory_id}: failed: {exc}", flush=True)
        _write_json_atomic(output / "manifest.json", manifest)
    manifest["status"] = "partial" if manifest["failures"] else "complete"
    _write_json_atomic(output / "manifest.json", manifest)
    if manifest["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
