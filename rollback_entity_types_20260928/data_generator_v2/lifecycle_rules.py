"""Declarative state-machine constraints for entities and legacy targets."""

from dataclasses import dataclass
from functools import lru_cache

from bfcl_eval.consistency.data_generator_v2.backward_state_knowledge import covers


@dataclass(frozen=True)
class Transition:
    tool: str
    branch: str
    before: str
    after: str
    max_calls: int | None = None
    creates: bool = False


@dataclass(frozen=True)
class LifecycleRule:
    target: str
    initial_states: frozenset[str]
    transitions: tuple[Transition, ...]
    state_path: str | None = None
    entity_scoped: bool = False

    def transition(self, tool, branch):
        return next((item for item in self.transitions
                     if (item.tool, item.branch) == (tool, branch)), None)

    def maximum_writes(self, states, limit, counts=None):
        """Maximum reverse path ending at an allowed initial state, capped by limit."""
        counts = counts or {}
        used = tuple(counts.get((item.tool, item.branch), 0) for item in self.transitions)

        @lru_cache(None)
        def visit(state, remaining, calls):
            best = 0 if state in self.initial_states else -1
            if not remaining:
                return best
            for index, item in enumerate(self.transitions):
                if item.after != state or (item.max_calls is not None and calls[index] >= item.max_calls):
                    continue
                next_calls = (*calls[:index], calls[index] + 1, *calls[index + 1:])
                earlier = visit(item.before, remaining - 1, next_calls)
                if earlier >= 0:
                    best = max(best, earlier + 1)
            return best

        return max((visit(state, limit, used) for state in states), default=-1)


def parse_lifecycle_rules(data, specs, targets, entity_types=None):
    version = data.get("version")
    section = "targets" if version == 1 else "entity_types"
    if version not in {1, 2} or set(data) != {"version", section} or not isinstance(data[section], dict):
        raise ValueError("Lifecycle rules require version 1/targets or version 2/entity_types")
    rules = {}
    for name, entry in data[section].items():
        if version == 2:
            kind = (entity_types or {}).get(name)
            if kind is None or entry.get("state_field") not in kind.fields:
                raise ValueError(f"Invalid lifecycle entity type or state field: {name}")
            path = kind.fields[entry["state_field"]]
        else:
            path = targets.get(name, {}).get("path")
        expected = {"initial_states", "transitions"} | ({"state_field"} if version == 2 else set())
        if path is None or set(entry) != expected:
            raise ValueError(f"Invalid lifecycle target: {name}")
        initial = entry["initial_states"]
        if not isinstance(initial, list) or not initial or any(not isinstance(x, str) for x in initial):
            raise ValueError(f"Invalid initial states for {name}")
        transitions = []
        seen = set()
        for item in entry["transitions"]:
            if not {"tool", "branch", "from", "to"} <= set(item) <= {
                "tool", "branch", "from", "to", "max_calls", "creates"
            } or any(not isinstance(item[key], str) or not item[key]
                     for key in ("tool", "branch", "from", "to")) or (
                "max_calls" in item and (type(item["max_calls"]) is not int or item["max_calls"] < 1)
            ) or (
                "creates" in item and type(item["creates"]) is not bool
            ):
                raise ValueError(f"Invalid transition for {name}")
            key = (item["tool"], item["branch"])
            if key in seen:
                raise ValueError(f"Duplicate lifecycle branch for {name}: {key}")
            seen.add(key)
            spec = specs.get(item["tool"])
            if spec is None or not any(branch["id"] == item["branch"] for branch in spec["branches"]) or not any(
                mutation["branch"] == item["branch"] and covers(mutation["target"], path)
                for mutation in spec["mutations"]
            ):
                raise ValueError(f"Lifecycle branch does not write {name}: {key}")
            transitions.append(Transition(item["tool"], item["branch"], item["from"], item["to"],
                                          item.get("max_calls"), item.get("creates", False)))
        if not transitions:
            raise ValueError(f"Lifecycle target {name} has no transitions")
        entity_scoped = version == 2
        if version == 1 and entity_types and targets[name].get("entity_type") in entity_types:
            kind = entity_types[targets[name]["entity_type"]]
            path = kind.fields[targets[name]["field"]]
            name = kind.name
            entity_scoped = True
        rule = LifecycleRule(name, frozenset(initial), tuple(transitions), path, entity_scoped)
        if name in rules and rules[name] != rule:
            raise ValueError(f"Conflicting lifecycle rules for entity type: {name}")
        rules[name] = rule
    return rules
