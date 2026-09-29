"""Reverse lifecycle frontiers and chronological validation per entity instance."""

from collections import Counter

from bfcl_eval.consistency.data_generator_v2.backward_state_knowledge import covers


class LifecyclePlanner:
    def __init__(self, rules, targets, registry, limit):
        self.rules = rules
        self.targets = targets
        self.registry = registry
        self.limit = limit
        self.frontiers = {}
        self.counts = {}
        self.created = set()

    def entity_rule(self, entity, kind=None):
        if self.registry:
            kind = kind or self.registry.instances.get(entity)
            return self.rules.get(kind)
        return self.rules.get(entity)

    def state_path(self, entity, rule):
        if self.registry:
            return rule.state_path.replace("{id}", self.registry.types[rule.target].marker(entity))
        return self.targets[entity]["path"]

    def field_entity(self, name):
        target = self.targets[name]
        entity = target.get("entity", name)
        rule = self.entity_rule(entity, target.get("entity_type"))
        return entity if rule and covers(self.state_path(entity, rule), target["path"]) else None

    @property
    def constrained_fields(self):
        return {name for name in self.targets if self.field_entity(name) is not None}

    def states(self, entity, rule):
        return self.frontiers.get(entity, set(rule.initial_states) | {item.after for item in rule.transitions})

    def affected(self, node):
        bindings = node.entity_binding.entities if self.registry else {
            name: name for name in self.rules if name in self.targets}
        for kind, entity in bindings.items():
            rule = self.entity_rule(entity, kind)
            if rule and node.touches(self.state_path(entity, rule)):
                yield entity, rule

    def eligible(self, node):
        if self.registry and any(entity in self.created for entity in node.entity_binding.entities.values()):
            return False
        for entity, rule in self.affected(node):
            transition = rule.transition(node.tool, node.branch["id"])
            counts = self.counts.get(entity, Counter())
            if transition is None or transition.after not in self.states(entity, rule) or (
                transition.max_calls is not None and counts[node.tool, node.branch["id"]] >= transition.max_calls
            ):
                return False
            if rule.entity_scoped and not transition.creates:
                earlier_counts = counts.copy()
                earlier_counts[node.tool, node.branch["id"]] += 1
                if rule.maximum_writes({transition.before}, self.limit, earlier_counts) < 0:
                    return False
        return True

    def capacity(self, name, limit, node=None):
        entity = self.field_entity(name)
        rule = self.entity_rule(entity)
        counts = self.counts.get(entity, Counter()).copy()
        states = self.states(entity, rule)
        if node is None or not node.writes(self.targets[name]["path"]):
            return rule.maximum_writes(states, limit, counts)
        transition = rule.transition(node.tool, node.branch["id"])
        if transition is None:
            return -1
        counts[node.tool, node.branch["id"]] += 1
        earlier = rule.maximum_writes({transition.before}, max(0, limit - 1), counts)
        return earlier + 1 if earlier >= 0 else -1

    def at_initial(self, name):
        entity = self.field_entity(name)
        if entity is None:
            return True
        rule = self.entity_rule(entity)
        return bool(self.states(entity, rule) & rule.initial_states)

    def apply(self, node):
        transitions = {}
        for entity, rule in self.affected(node):
            item = rule.transition(node.tool, node.branch["id"])
            self.frontiers[entity] = {item.before}
            self.counts.setdefault(entity, Counter())[node.tool, node.branch["id"]] += 1
            # Creation is the earliest call on an entity; no pre-existing value may be reused.
            if item.creates or (not rule.entity_scoped and item.after in rule.initial_states
                               and not any(t.after == item.before for t in rule.transitions)):
                self.created.add(entity)
            transitions[entity] = {"from": item.before, "to": item.after}
        return transitions

    def pending(self, active):
        controlled = {self.field_entity(name) for name in active}
        return [entity for entity, states in self.frontiers.items()
                if entity not in controlled and entity not in self.created
                and not states & self.entity_rule(entity).initial_states]

    def audit(self, steps):
        traces = {}
        for step in steps:
            for entity, transition in step.get("lifecycle_transitions", {}).items():
                traces.setdefault(entity, []).append({"step": step["index"], "tool": step["tool"],
                    "branch": step["branch"], **transition})
        result = {}
        entities = set(traces)
        if self.registry:
            entities.update(entity for entity in self.registry.instances if self.entity_rule(entity))
        for entity in sorted(entities):
            trace = traces.get(entity, [])
            rule = self.entity_rule(entity)
            first = rule.transition(trace[0]["tool"], trace[0]["branch"]) if trace else None
            initial = first.before if entity in self.created else None
            possible = {initial} if initial is not None else set(rule.initial_states)
            counts = Counter()
            for call in trace:
                item = rule.transition(call["tool"], call["branch"])
                counts[item.tool, item.branch] += 1
                if item.before not in possible or (item.max_calls is not None and counts[item.tool, item.branch] > item.max_calls):
                    raise ValueError(f"Invalid forward lifecycle for {entity}")
                possible = {item.after}
            if initial is not None and self.registry:
                marker = self.registry.types[rule.target].marker(entity)
                first_step = trace[0]["step"]
                if any(marker in step.get("symbolic_slots", {}).values() for step in steps[:first_step - 1]):
                    raise ValueError(f"Entity referenced before creation: {entity}")
            result[entity] = {"initial_states": sorted({initial} if initial is not None else rule.initial_states),
                              "final_states": sorted(possible), "transitions": trace}
        return result
