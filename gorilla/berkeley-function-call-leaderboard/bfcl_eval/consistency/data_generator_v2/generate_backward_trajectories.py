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
    StateKnowledge, before, covers, mutation_sources, mutation_writes_path, overlaps,
    state_path, temporal_covers,
)
from bfcl_eval.consistency.data_generator_v2.generate_tool_state_specs import (
    _write_json_atomic, validate_spec,
)
from bfcl_eval.consistency.data_generator_v2.lifecycle_rules import parse_lifecycle_rules
from bfcl_eval.consistency.data_generator_v2.placeholder_bindings import (
    PLACEHOLDER, PlaceholderPool, path_bindings, substitute,
)
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
    bindings: dict[str, str] = field(default_factory=dict)

    def writes(self, path):
        return bool(self.matching_mutation_indices(path))

    def matching_mutation_indices(self, path):
        return [index for index, mutation in enumerate(self.mutations)
                if mutation_writes_path(mutation, path)]

    def referenced_paths(self):
        return {path for rule in self.rules for path in (rule.source, *rule.requires)} | {
            source for mutation in self.mutations for source in mutation_sources(mutation)
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


def select_targets(targets, names):
    counts = Counter(names)
    if not names or set(names) - set(targets):
        raise ValueError("Choose names from the target catalog")
    if any(count > 1 and not PLACEHOLDER.search(targets[name]["path"])
           for name, count in counts.items()):
        raise ValueError("Repeated targets require a placeholder path")
    used = Counter()
    selected = {}
    for name in names:
        used[name] += 1
        label = name if counts[name] == 1 else f"{name}_{used[name]}"
        if label in selected:
            raise ValueError(f"Conflicting target label: {label}")
        selected[label] = name
    return selected


class BackwardSampler:
    """Keep one reverse-time frontier so interleaved chains share state versions."""

    def __init__(self, specs, targets, *, reader_refinements=(), lifecycle_rules=None, seed=42):
        self.targets = targets
        self.lifecycle_rules = lifecycle_rules or {}
        self.rng = random.Random(seed)
        refinements = {
            (item["tool"], item["branch"], item["field"], item["source"]): item
            for item in reader_refinements
        }
        used_refinements = set()
        self.nodes = []
        for spec in specs.values():
            if spec["schema_version"] not in {"1.1", "1.2", "1.3"}:
                raise ValueError("Backward sampling requires schema version 1.1, 1.2 or 1.3")
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
        self._pool = None

    def bind_node(self, node, source=None, path=None, *, bindings=None, pool=None):
        if bindings is None:
            pool = pool or self._pool
            if pool is None:
                return node
            bindings = pool.bindings(
                [node.branch, node.earlier, node.returns, node.mutations,
                 [vars(rule) for rule in node.rules]], source, path,
            )
        mutations = substitute(node.mutations, bindings)
        for mutation in mutations:
            for item in mutation.get("target_identity_sources", []):
                item["placeholder"] = bindings.get(item["placeholder"], item["placeholder"])
        return replace(
            node, branch=substitute(node.branch, bindings), earlier=substitute(node.earlier, bindings),
            returns=substitute(node.returns, bindings), mutations=mutations,
            rules=[replace(rule, source=substitute(rule.source, bindings),
                           result_field=substitute(rule.result_field, bindings),
                           requires=substitute(rule.requires, bindings),
                           condition=substitute(rule.condition, bindings), reason=substitute(rule.reason, bindings))
                   for rule in node.rules], bindings=bindings,
        )

    def step_node(self, step):
        template = next(node for node in self.nodes
                        if (node.tool, node.branch["id"]) == (step["tool"], step["branch"]))
        return self.bind_node(template, bindings=step.get("placeholder_bindings", {}))

    def writer_options(self, path):
        for template in self.nodes:
            for index in template.matching_mutation_indices(path):
                node = self.bind_node(template, template.mutations[index]["target"], path)
                if node.writes(path):
                    yield node, index

    def lifecycle_for(self, path):
        for rule in self.lifecycle_rules.values():
            template = rule.path or self.targets[rule.target]["path"]
            if covers(template, path) and covers(path, template):
                return rule
        return None

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
        for template in self.nodes:
            for index, template_rule in enumerate(template.rules):
                if not covers(before(template_rule.source), path):
                    continue
                node = self.bind_node(template, template_rule.source, path)
                rule = node.rules[index]
                intrinsic = self.closure(node)
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
            node = self.step_node(step)
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
                node = self.step_node(step)
                if node.writes(paths[name]):
                    entry["write_ids"].append(step["_id"])
                    if all(status["sources_fixed"]
                           for mutation, status in zip(node.mutations, step["mutation_knowledge"])
                           if mutation_writes_path(mutation, paths[name])):
                        entry["source_fixed_write_ids"].append(step["_id"])
                    step["write_targets"].append(name)
                    if "write" not in step["roles"]:
                        step["roles"].append("write")
        return kept, removed

    def _support_anchor_candidates(self, steps, paths, chain):
        """Find finite, read-only observations suitable for pre-distractor padding."""
        core_branches = defaultdict(list)
        for step in steps:
            if step["role"] != "distractor":
                core_branches[(step["tool"], step["branch"])].append(step)
        selected_returns = {
            (step["tool"], step["branch"], rule.result_field, rule.source)
            for step in steps for rule in self.step_node(step).rules
        }
        candidates = []
        seen = set()
        for template in self.nodes:
            branch_key = (template.tool, template.branch["id"])
            if template.mutations:
                continue
            for index, template_rule in enumerate(template.rules):
                source_root = segments(template_rule.source)
                same_branch = core_branches[branch_key]
                contexts = ([(self.bind_node(template, bindings=step["placeholder_bindings"]),
                              "same_branch_return", step["_id"], False)
                             for step in same_branch] if same_branch else [
                    (self.bind_node(template, template_rule.source, path), "target_dict_source",
                     chain[name]["final_id"], chain[name]["final_phase"] == "after")
                    for name, path in paths.items() if source_root and segments(path)[0] == source_root[0]
                ])
                for node, reason, reference_id, insert_after in contexts:
                    rule = node.rules[index]
                    key = (node.tool, node.branch["id"], rule.result_field, rule.source)
                    if key in seen or key in selected_returns or not state_path(rule.source):
                        continue
                    intrinsic = self.closure(node)
                    if self.proof(intrinsic, rule.source) is None or any(
                        self.proof(intrinsic, requirement) is None for requirement in rule.requires
                    ):
                        continue
                    seen.add(key)
                    candidates.append((node, rule, reason, reference_id, insert_after))
        self.rng.shuffle(candidates)
        return candidates

    @staticmethod
    def _support_anchor_step(node, rule, reason, step_id):
        conditions = [LONG_CONTEXT_DISABLED]
        if rule.condition:
            conditions.append(rule.condition)
        return {
            "_id": step_id, "tool": node.tool, "branch": node.branch["id"],
            "placeholder_bindings": dict(node.bindings),
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
        }

    @staticmethod
    def _distractor_sources(node):
        branch_sources = (source for branch in (node.branch, *node.earlier)
                          for source in branch["uses"])
        return_sources = (relation["path"] for returned in node.returns
                          for relation in (*returned["state_source_relations"],
                                           *returned.get("state_observation_relations", [])))
        mutation_inputs = (source for mutation in node.mutations
                           for source in mutation_sources(mutation))
        return tuple(dict.fromkeys((*branch_sources, *return_sources, *mutation_inputs)))

    def _distractor_safe(self, node, protected, reserved_entities):
        if any(overlaps(source, path) for source in self._distractor_sources(node)
               if state_path(source) for path in protected):
            return False
        if any(overlaps(mutation["target"], path)
               for mutation in node.mutations for path in protected):
            return False
        for mutation in node.mutations:
            target = mutation["target"]
            for identity in mutation.get("target_identity_sources", []):
                position = target.find("{" + identity["placeholder"] + "}")
                if position >= 0:
                    parent = target[:target.rfind("[", 0, position)]
                    if any(overlaps(parent, path) for path in protected):
                        return False
        return not any(reserved_entities.intersection(PLACEHOLDER.findall(mutation["target"]))
                       for mutation in node.mutations)

    def _distractor_link_sources(self, node):
        sources = [relation["path"] for returned in node.returns
                   for relation in (*returned["state_source_relations"],
                                    *returned.get("state_observation_relations", []))]
        sources.extend(source for mutation in node.mutations for source in mutation_sources(mutation))
        effects = self._distractor_lifecycle_effects(node)
        if effects:
            sources.extend(source for branch in (node.branch, *node.earlier)
                           for source in branch["uses"]
                           if source in effects)
        return tuple(dict.fromkeys(source for source in sources
                                   if source.startswith("$.state_before")))

    def _distractor_lifecycle_effects(self, node):
        effects = {}
        for rule in self.lifecycle_rules.values():
            template = before(rule.path or self.targets[rule.target]["path"])
            for mutation in node.mutations:
                if not mutation_writes_path(mutation, template):
                    continue
                path = substitute(template, path_bindings(template, mutation["target"]))
                if all(re.fullmatch(r".+_[1-9][0-9]*", token)
                       for token in PLACEHOLDER.findall(path)):
                    transition = rule.transition(node.tool, node.branch["id"])
                    if transition is None:
                        return None
                    effects[path] = (rule, transition)
        return effects

    def _valid_distractor_chain(self, newest_first):
        states = {}
        calls = Counter()
        for node in reversed(newest_first):
            effects = self._distractor_lifecycle_effects(node)
            if effects is None:
                return False
            for path, (_, transition) in effects.items():
                key = (path, node.tool, node.branch["id"])
                if (path in states and states[path] != transition.before) or (
                    transition.max_calls is not None and calls[key] >= transition.max_calls
                ):
                    return False
                states[path] = transition.after
                calls[key] += 1
        return True

    def _linked_distractor_chain(self, limit, protected, reserved_entities):
        """Find a long independent write-to-read chain within a bounded search."""
        if limit < 2:
            return (), ()
        templates = list(self.nodes)
        self.rng.shuffle(templates)
        best_nodes, best_links = (), ()
        for template in templates:
            for _ in range(2):
                pool = PlaceholderPool()
                pool.counts = self._pool.counts.copy()
                sink = self.bind_node(template, pool=pool)
                if (not self._distractor_safe(sink, protected, reserved_entities)
                        or reserved_entities.intersection(sink.bindings.values())
                        or not self._valid_distractor_chain((sink,))):
                    continue
                pool.commit(sink.bindings)
                nodes, links = [sink], []
                while len(nodes) < limit:
                    options = []
                    used_tools = {node.tool for node in nodes}
                    for source in self._distractor_link_sources(nodes[-1]):
                        for writer in templates:
                            for index in writer.matching_mutation_indices(source):
                                predecessor = self.bind_node(writer, writer.mutations[index]["target"],
                                                             source, pool=pool)
                                if (not mutation_writes_path(predecessor.mutations[index], source)
                                        or not self._distractor_safe(predecessor, protected, reserved_entities)
                                        or reserved_entities.intersection(predecessor.bindings.values())
                                        or not self._valid_distractor_chain((*nodes, predecessor))):
                                    continue
                                options.append((predecessor, source))
                    if not options:
                        break
                    self.rng.shuffle(options)
                    options.sort(key=lambda option: (
                        bool(self._distractor_link_sources(option[0])),
                        option[0].tool not in used_tools,
                    ), reverse=True)
                    predecessor, source = options[0]
                    pool.commit(predecessor.bindings)
                    nodes.append(predecessor)
                    links.append(source)
                if len(nodes) > len(best_nodes):
                    best_nodes, best_links = tuple(nodes), tuple(links)
                if len(best_nodes) == limit:
                    return tuple(reversed(best_nodes)), (None, *reversed(best_links))
        if not best_nodes:
            return (), ()
        return tuple(reversed(best_nodes)), (None, *reversed(best_links))

    def build(self, names=None, *, max_writes=3, target_max_writes=None, dependency_max_writes=1,
              min_length=20, max_length=60, attempts=200, linked_distractors=False):
        """Legacy max_writes keywords now specify minimum target write counts."""
        selected = select_targets(self.targets, list(self.targets) if names is None else list(names))
        names = list(selected)
        if max_writes < 1 or dependency_max_writes < 0 or attempts < 1:
            raise ValueError("max-writes and attempts must be positive; dependency-max-writes must be nonnegative")
        if not 1 <= min_length <= max_length:
            raise ValueError("Require 1 <= min-length <= max-length")
        overrides = target_max_writes or {}
        if set(overrides) - set(names) or any(not isinstance(value, int) or value < 1 for value in overrides.values()):
            raise ValueError("Per-target write counts must be positive integers for selected targets")
        limits = {name: overrides.get(name, max_writes) for name in names}
        failures = Counter()
        for attempt in range(1, attempts + 1):
            try:
                result = self._build_once(selected, limits, dependency_max_writes, min_length, max_length,
                                          linked_distractors)
                result["planning"]["attempts"] = attempt
                result["planning"]["rejected_attempts"] = dict(failures)
                return result
            except SamplingError as exc:
                failures[str(exc)] += 1
        raise SamplingError(f"No chain after {attempts} attempts: {dict(failures)}")

    def _build_once(self, selected, limits, dependency_max_writes, min_length, max_length,
                    linked_distractors):
        self._pool = PlaceholderPool()
        self._option_cache.clear()
        names = list(selected)
        paths = {name: self._pool.allocate(before(self.targets[target]["path"]))
                 for name, target in selected.items()}
        lifecycles = {name: rule for name, path in paths.items()
                      if (rule := self.lifecycle_for(path)) is not None}
        lifecycle_paths = {paths[name]: rule for name, rule in lifecycles.items()}
        monitored_lifecycle_paths = set(lifecycle_paths)
        frontiers = {}
        lifecycle_counts = defaultdict(Counter)

        def states(path, rule):
            return frontiers.get(path, set(rule.initial_states) | {item.after for item in rule.transitions})

        def effects(node):
            affected = {path: rule for path, rule in lifecycle_paths.items() if node.writes(path)}
            for rule in self.lifecycle_rules.values():
                template = before(rule.path or self.targets[rule.target]["path"])
                for mutation in node.mutations:
                    if mutation_writes_path(mutation, template):
                        path = substitute(template, path_bindings(template, mutation["target"]))
                        if all(re.fullmatch(r".+_[1-9][0-9]*", token)
                               for token in PLACEHOLDER.findall(path)):
                            affected[path] = rule
            return affected

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
            sources = mutation_sources(node.mutations[mutation_index]) if mutation_index is not None else ()
            return all(self.proof(known, source) or is_pending(source) or self._can_fix(
                before(source), depth + 1
            ) for source in sources)

        def lifecycle_eligible(node):
            for path, rule in effects(node).items():
                transition = rule.transition(node.tool, node.branch["id"])
                if transition is None or transition.after not in states(path, rule) or (
                    transition.max_calls is not None and
                    lifecycle_counts[path][(node.tool, node.branch["id"])] >= transition.max_calls
                ):
                    return False
            return True

        def lifecycle_capacity(node, name, limit):
            lifecycle = lifecycles[name]
            transition = lifecycle.transition(node.tool, node.branch["id"])
            counts = lifecycle_counts[paths[name]].copy()
            counts[(node.tool, node.branch["id"])] += 1
            earlier = lifecycle.maximum_writes({transition.before}, max(0, limit - 1), counts)
            return earlier + 1 if earlier >= 0 else -1

        def emit(node, role, path, depth, *, rule=None, missing=(), mutation_index=None):
            # Provisional anchors can be removed later; writers and final reads cannot.
            mandatory = sum(bool(step["mutations"]) or "read" in step["roles"] for step in reverse_steps)
            if mandatory >= max_length and (node.mutations or role == "read"):
                raise SamplingError("Necessary anchors and writes exceed max-length")
            step_id = len(reverse_steps)
            self._pool.commit(node.bindings)
            self._option_cache.clear()
            resolved = self.closure(node, missing)
            conditions = sorted({LONG_CONTEXT_DISABLED, *(proof["condition"] for proof in resolved.values()
                                                         if proof.get("condition"))})
            step = {
                "_id": step_id, "tool": node.tool, "branch": node.branch["id"],
                "placeholder_bindings": dict(node.bindings),
                "branch_condition": node.branch["if"], "branch_uses": node.branch["uses"],
                "earlier_branch_conditions_to_avoid": [b["if"] for b in node.earlier],
                "earlier_error_conditions_to_avoid": [b["if"] for b in node.earlier if b["id"].startswith("error_")],
                "role": role, "roles": [role], "field": path, "dependency_depth": depth,
                "additional_conditions": conditions,
                "return_fields": copy.deepcopy(node.returns), "mutations": copy.deepcopy(node.mutations),
                "selected_mutation_index": mutation_index,
                "fixes": [], "write_targets": [], "write_source_requirements": [],
            }
            if rule:
                step["selected_return_field"] = rule.result_field
                step["selected_return_source"] = rule.source
                step["fix_dependencies"] = list(missing)
            reverse_steps.append(step)
            if node.mutations:
                applied = {}
                for entity_path, lifecycle in effects(node).items():
                    lifecycle_paths[entity_path] = lifecycle
                    transition = lifecycle.transition(node.tool, node.branch["id"])
                    frontiers[entity_path] = {transition.before}
                    lifecycle_counts[entity_path][(node.tool, node.branch["id"])] += 1
                    applied[entity_path] = {"from": transition.before, "to": transition.after}
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
                for source in mutation_sources(mutation):
                    proof = self.proof(resolved, source)
                    if proof and proof["kind"] == "earlier_state":
                        proof = None
                    step["write_source_requirements"].append({
                        "path": source, "mutation_target": mutation["target"],
                        "resolution": "same_call_return" if proof else "earlier_chain",
                        "evidence": proof,
                        "target_identity_relations": [copy.deepcopy(item)
                            for item in mutation.get("target_identity_sources", []) if item["path"] == source],
                    })
                    if not proof:
                        add_need(source, depth + 1, step_id)
            return step_id

        while todo or active or dependencies:
            # Deep dependencies are fixed immediately, without further writer sampling.
            deep = [need for need in dependencies.values() if need.depth >= 2]
            tasks = [("dependency", need) for need in deep] if deep else (
                [("final", name) for name in sorted(todo)] +
                [("target", name) for name in active] +
                [("dependency", need) for need in dependencies.values()]
            )
            self.rng.shuffle(tasks)
            advanced = False
            for kind, subject in tasks:
                if kind == "final":
                    name = subject
                    path = paths[name]
                    options = [option for option in self.fix_options(path, 0)
                               if feasible(option[0], 0, option[2]) and lifecycle_eligible(option[0])]
                    if name in lifecycles and options:
                        scores = [lifecycle_capacity(node, name, limits[name])
                                  if node.writes(path) and self.anchor_phase(node, rule, path) == "after"
                                  else lifecycles[name].maximum_writes(states(paths[name], lifecycles[name]), limits[name], lifecycle_counts[paths[name]])
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
                        possible = lifecycles[name].maximum_writes(
                            states(path, lifecycles[name]), required, lifecycle_counts[path])
                        if possible < 0:
                            raise SamplingError(f"No lifecycle path to an initial state: {name}")
                        active[name] = Need(path, 0, possible)
                    else:
                        active[name] = Need(path, 0, required)
                else:
                    need = active[subject] if kind == "target" else subject
                    if need.remaining:
                        options = [(node, mutation_index) for node, mutation_index in self.writer_options(need.path)
                                   if feasible(node, need.depth, mutation_index=mutation_index)
                                   and lifecycle_eligible(node)]
                        if kind == "target" and subject in lifecycles and options:
                            scores = {id(node): lifecycle_capacity(node, subject, need.remaining)
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
                            if kind == "target" and subject in lifecycles:
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
                                   and lifecycle_eligible(option[0])]
                        if not options:
                            continue
                        if kind == "target" and subject in lifecycles and not (
                            states(need.path, lifecycles[subject]) & lifecycles[subject].initial_states
                        ):
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
        support_candidates = self._support_anchor_candidates(steps, paths, chain)
        support_count = min(filler_remaining, len(support_candidates))
        next_id = max((step["_id"] for step in steps), default=-1) + 1
        for node, rule, reason, reference_id, insert_after in support_candidates[:support_count]:
            self._pool.commit(node.bindings)
            support = self._support_anchor_step(node, rule, reason, next_id)
            next_id += 1
            if PLACEHOLDER.search(rule.source):
                reference_index = next(index for index, step in enumerate(steps)
                                       if step["_id"] == reference_id)
                steps.insert(reference_index + int(insert_after), support)
            else:
                steps.insert(self.rng.randrange(1, len(steps)), support)
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
        for step in steps:
            node = self.step_node(step)
            protected.update(source for branch in (node.branch, *node.earlier)
                             for source in branch["uses"] if state_path(source))
        used_entities = {identity for step in steps
                         for identity in step["placeholder_bindings"].values()}
        used_entities.update(identity for path in protected for identity in PLACEHOLDER.findall(path))

        def distractor_options():
            for template in self.nodes:
                node = self.bind_node(template)
                if (self._distractor_safe(node, protected, used_entities)
                        and self._valid_distractor_chain((node,))):
                    yield (node,)

        def distractor_step(node, chain_id=None, link_source=None):
            step = {
                "tool": node.tool, "branch": node.branch["id"], "branch_condition": node.branch["if"],
                "placeholder_bindings": dict(node.bindings),
                "branch_uses": node.branch["uses"],
                "earlier_branch_conditions_to_avoid": [branch["if"] for branch in node.earlier],
                "earlier_error_conditions_to_avoid": [branch["if"] for branch in node.earlier if branch["id"].startswith("error_")],
                "role": "distractor", "roles": ["distractor"], "dependency_depth": None,
                "additional_conditions": [LONG_CONTEXT_DISABLED], "return_fields": copy.deepcopy(node.returns),
                "mutations": copy.deepcopy(node.mutations), "fixes": [], "write_targets": [], "write_source_requirements": [],
                "selected_mutation_index": None, "mutation_knowledge": [],
            }
            if chain_id is not None:
                step["distractor_chain_id"] = chain_id
                if link_source is not None:
                    step["distractor_link_source"] = link_source
            for path, (rule, transition) in self._distractor_lifecycle_effects(node).items():
                lifecycle_paths[path] = rule
                step.setdefault("lifecycle_transitions", {})[path] = {
                    "from": transition.before, "to": transition.after,
                }
            return step

        filler_count = max(0, min_length - necessary_length)
        chain_lengths = []
        if linked_distractors:
            remaining = filler_count
            while remaining >= 2:
                chain_nodes, link_sources = self._linked_distractor_chain(
                    remaining, protected, used_entities,
                )
                if len(chain_nodes) < 2:
                    break
                chain_lengths.append(len(chain_nodes))
                block = []
                for node, source in zip(chain_nodes, link_sources):
                    self._pool.commit(node.bindings)
                    used_entities.update(node.bindings.values())
                    block.append(distractor_step(node, len(chain_lengths), source))
                insert_at = self.rng.randrange(1, len(steps))
                steps[insert_at:insert_at] = block
                protected.update(source for node in chain_nodes
                                 for source in self._distractor_sources(node) if state_path(source))
                protected.update(mutation["target"] for node in chain_nodes
                                 for mutation in node.mutations)
                remaining -= len(block)
        for _ in range(filler_count - sum(chain_lengths)):
            options = list(distractor_options())
            if not options:
                raise SamplingError("No unrelated branch can fill min-length")
            node = self._pick(options)[0]
            self._pool.commit(node.bindings)
            used_entities.update(node.bindings.values())
            steps.insert(self.rng.randrange(1, len(steps)), distractor_step(node))
        if any(step["role"] == "distractor" and step["mutations"] for step in steps):
            knowledge = StateKnowledge()
            for index, step in enumerate(steps):
                node = self.step_node(step)
                referenced = node.referenced_paths() | {fixed["path"] for fixed in step["fixes"]}
                proofs = self.closure(node, knowledge.known_paths(referenced))
                knowledge.observe(proofs, index)
                statuses = knowledge.apply_mutations(node.mutations, proofs, index)
                if step["role"] == "distractor":
                    step["mutation_knowledge"] = statuses
        positions = {step["_id"]: index for index, step in enumerate(steps, 1) if "_id" in step}
        # The second knowledge pass can remove an origin retained by the first.
        redirects = {item["original_step_id"]: item["reused_from_ids"] for item in removed_anchors}

        def retained_origins(step_id):
            if step_id in positions:
                return {positions[step_id]}
            return {origin for previous in redirects[step_id] for origin in retained_origins(previous)}

        for removed in removed_anchors:
            removed["reused_from_steps"] = sorted({origin for item in removed.pop("reused_from_ids")
                                                   for origin in retained_origins(item)})
        chains = {}
        for name, entry in chain.items():
            write_steps = sorted(positions[index] for index in entry["write_ids"])
            source_fixed_steps = sorted(positions[index] for index in entry["source_fixed_write_ids"])
            initial, final = positions[entry["initial_id"]], positions[entry["final_id"]]
            end = final + int(entry.get("final_phase") == "after")
            if (name not in lifecycles and len(write_steps) < limits[name]) or not all(
                initial <= index < end for index in write_steps
            ):
                raise SamplingError("A target's anchor/write/read interval is invalid")
            if name in lifecycles:
                rule = lifecycles[name]
                possible = set(rule.initial_states)
                for index in write_steps:
                    step = steps[index - 1]
                    transition = rule.transition(step["tool"], step["branch"])
                    if paths[name] not in step.get("lifecycle_transitions", {}) or transition is None or transition.before not in possible:
                        raise SamplingError(f"Invalid forward lifecycle for {name}")
                    possible = {transition.after}
            chains[name] = {"path": entry["path"], "anchor_step": initial, "read_step": final,
                            "anchor_phase": entry.get("initial_phase", "before"),
                            "read_phase": entry.get("final_phase", "before"),
                            "write_steps": write_steps, "write_count": len(write_steps),
                            "source_fixed_write_steps": source_fixed_steps,
                            "fully_source_fixed": len(source_fixed_steps) == len(write_steps),
                            "min_write_count": limits[name],
                            "write_shortfall": max(0, limits[name] - len(write_steps)),
                            "shortfall_reason": "lifecycle_writers_exhausted" if name in lifecycles and len(write_steps) < limits[name] else None,
                            "writer_tools": sorted({steps[index - 1]["tool"] for index in write_steps})}
        lifecycle_instances = {}
        for path, rule in lifecycle_paths.items():
            transitions = [(index, rule.transition(step["tool"], step["branch"]))
                           for index, step in enumerate(steps, 1)
                           if path in step.get("lifecycle_transitions", {})]
            possible = set(rule.initial_states)
            if transitions and path not in monitored_lifecycle_paths:
                possible = {transitions[0][1].before}
            if transitions and transitions[0][1].before not in possible:
                first = transitions[0][1]
                if first.after not in possible or any(item.after == first.before for item in rule.transitions):
                    raise SamplingError(f"Invalid initial lifecycle for {path}")
                possible = {first.before}
            initial_states = sorted(possible)
            for _, transition in transitions:
                if transition.before not in possible:
                    raise SamplingError(f"Invalid forward lifecycle for {path}")
                possible = {transition.after}
            lifecycle_instances[path] = {"rule": rule.target, "initial_states": initial_states,
                                         "transition_steps": [index for index, _ in transitions]}
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
            for node in self.nodes:
                intrinsic = self.closure(node)
                for mutation_index in node.matching_mutation_indices(path):
                    mutation = node.mutations[mutation_index]
                    unfixed = sorted({source for source in mutation_sources(mutation)
                                      if not self.proof(intrinsic, source)
                                      and not any(covers(target, source) for target in paths.values())
                                      and not self._can_fix(before(source), 1)})
                    writer_candidates[name].append({"tool": node.tool, "branch": node.branch["id"],
                                                    "mutation_index": mutation_index,
                                                    "mutation_target": mutation["target"],
                                                    "unfixable_write_sources": unfixed})
        return {
            "targets": [{"name": name, "path": paths[name]} for name in names],
            "state_conditions": dict(FIXED_STATE_CONDITIONS),
            "steps": steps,
            "planning": {
                "direction": "backward", "target_chains": chains,
                "dependency_scope": DEPENDENCY_SCOPE,
                "target_min_writes": limits, "write_count_mode": "best_effort_lifecycle" if lifecycles else "at_least_minimum",
                "lifecycle_targets": sorted(lifecycles),
                "lifecycle_instances": lifecycle_instances,
                "dependency_max_writes": dependency_max_writes,
                "min_length": min_length, "max_length": max_length,
                "core_length": core_length, "support_anchor_count": support_count,
                "necessary_length": necessary_length, "distractor_count": filler_count,
                "linked_distractors": linked_distractors, "distractor_chain_lengths": chain_lengths,
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
                    "Assign concrete keys to numbered placeholders, preserving their identities and each lifecycle instance's initial states.",
                    "At each selected write, require every target_identity_relations path's pre-call value to equal the concrete key bound to its placeholder; verify this against observations, not only initial state.",
                    "Ground every retained side effect; unfixed_side_effects does not imply the affected state is derivable from observations.",
                    *(["Ground linked distractor entity placeholders to concrete keys distinct from the monitored chains and other distractor chains."]
                      if linked_distractors else []),
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
    return parse_lifecycle_rules(json.loads(path.read_text(encoding="utf-8")), specs, targets), path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--specs", type=Path)
    rules = parser.add_mutually_exclusive_group()
    rules.add_argument("--lifecycle-rules", type=Path, help="Override the catalog's lifecycle rules")
    rules.add_argument("--no-lifecycle-rules", action="store_true", help="Use the original unconstrained sampler")
    parser.add_argument("--targets", nargs="+", default=["balance", "transaction_history", "orders"])
    parser.add_argument("--min-writes", "--max-writes", dest="max_writes", type=int, default=3,
                        help="Minimum write-call count for each target; shared writes may exceed it (legacy alias: --max-writes)")
    parser.add_argument("--target-min-writes", "--target-max-writes", dest="target_max_writes",
                        nargs="*", default=[], metavar="NAME=N",
                        help="Override the minimum write count for individual targets (legacy alias: --target-max-writes)")
    parser.add_argument("--dependency-max-writes", type=int, default=1)
    parser.add_argument("--min-length", type=int, default=20)
    parser.add_argument("--max-length", type=int, default=60)
    parser.add_argument("--linked-distractors", action="store_true",
                        help="Fill shortfalls with independent backward-linked distractor chains")
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
    lifecycle_rules, lifecycle_path = load_lifecycle_rules(
        args.target_catalog, specs, targets, args.lifecycle_rules, args.no_lifecycle_rules,
    )
    try:
        selected = select_targets(targets, args.targets)
    except ValueError as exc:
        parser.error(str(exc))
    overrides = {}
    for entry in args.target_max_writes:
        name, separator, value = entry.partition("=")
        if not separator or name not in selected or not value.isdigit() or int(value) < 1 or name in overrides:
            parser.error("target-min-writes entries must be unique selected NAME=positive_integer")
        overrides[name] = int(value)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory must be empty")
    sampler = BackwardSampler(specs, targets, reader_refinements=refinements,
                              lifecycle_rules=lifecycle_rules, seed=args.seed)
    constrained = sorted(name for name, target in selected.items()
                         if sampler.lifecycle_for(targets[target]["path"]))
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "running", "strategy": "backward_state_source_sampling", "requested": args.count,
        "accepted": 0, "failures": [], "planning_only": True, "backend_execution": False,
        "state_conditions": dict(FIXED_STATE_CONDITIONS),
        "dependency_scope": DEPENDENCY_SCOPE,
        "seed": args.seed, "targets": args.targets, "min_writes": args.max_writes,
        "target_min_writes": {name: overrides.get(name, args.max_writes) for name in selected},
        "write_count_mode": "best_effort_lifecycle" if constrained else "at_least_minimum",
        "lifecycle_targets": constrained,
        "lifecycle_rules_sha256": hashlib.sha256(lifecycle_path.read_bytes()).hexdigest() if lifecycle_path else None,
        "dependency_max_writes": args.dependency_max_writes,
        "min_length": args.min_length, "max_length": args.max_length,
        "specs_sha256": hashlib.sha256(specs_path.read_bytes()).hexdigest(),
        "catalog_sha256": hashlib.sha256(args.target_catalog.read_bytes()).hexdigest(),
        "catalog_fields_used": ["targets", "state_specs", "reader_refinements"] + (
            ["lifecycle_rules"] if lifecycle_path and not args.lifecycle_rules else []),
        "writer_sequences_used": False,
        "filler_strategy": (["support_anchor", "linked_distractor", "distractor"]
                            if args.linked_distractors else ["support_anchor", "distractor"]),
        "linked_distractors": args.linked_distractors,
        "support_anchor_policy": "sample_then_forward_anchor_reuse",
    }
    _write_json_atomic(output / "manifest.json", manifest)
    for index in range(args.count):
        trajectory_id = f"backward_{args.seed}_{index:04d}"
        try:
            result = sampler.build(args.targets, max_writes=args.max_writes,
                                   target_max_writes=overrides,
                                   dependency_max_writes=args.dependency_max_writes,
                                   min_length=args.min_length, max_length=args.max_length,
                                   attempts=args.attempts,
                                   linked_distractors=args.linked_distractors)
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
