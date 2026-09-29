"""Track symbolic state knowledge through observations and partial side effects."""

from dataclasses import dataclass, replace

from bfcl_eval.consistency.data_generator_v2.symbolic_dependency_graph import covers, segments


def before(path):
    return path.replace("$.state_after", "$.state_before", 1)


def state_path(path):
    return path in {"$.state_before", "$.state_after"} or path.startswith((
        "$.state_before.", "$.state_after.", "$.state_before[", "$.state_after[",
    ))


def overlaps(left, right):
    return covers(left, right) or covers(right, left)


def mutation_writes_path(mutation, monitored_path):
    """Return whether a mutation is guaranteed to write ``monitored_path``.

    A mutation of the monitored field or one of its descendants writes the
    monitored region.  An ancestor target is only treated as writing the
    descendant when the operation inserts that ancestor object, since an
    ordinary replacement of a parent does not guarantee the descendant.
    """
    target = mutation["target"]
    return covers(monitored_path, target) or (
        mutation.get("operation") == "insert" and covers(target, monitored_path)
    )


def temporal_covers(parent, child):
    return parent.startswith("$.state_after") == child.startswith("$.state_after") and covers(parent, child)


def state_sources(mutation):
    return tuple(dict.fromkeys(source for source in mutation["value_from"] if state_path(source)))


def target_identity_sources(mutation):
    return tuple(dict.fromkeys(item["path"] for item in mutation.get(
        "target_identity_sources", []
    ) if state_path(item["path"])))


def mutation_sources(mutation):
    """Value inputs plus target-address inputs; never expand source-path keys."""
    return tuple(dict.fromkeys((*state_sources(mutation), *target_identity_sources(mutation))))


@dataclass(frozen=True)
class KnownValue:
    anchor_id: int | None
    anchor_evidence: dict | None
    mutation_ids: tuple[int, ...] = ()
    observation_ids: tuple[int, ...] = ()
    anchor_phase: str = "before"

    def evidence(self, source):
        return {"kind": "propagated_state", "source": source, "anchor_id": self.anchor_id,
                "anchor_phase": self.anchor_phase, "mutation_ids": list(self.mutation_ids),
                "observation_ids": list(self.observation_ids)}


class StateKnowledge:
    """More-specific facts override ancestors; None marks an unknown state region."""

    def __init__(self):
        self.facts: dict[str, KnownValue | None] = {}

    def get(self, path):
        path = before(path)
        matches = [(source, value) for source, value in self.facts.items() if covers(source, path)]
        if not matches:
            return None
        value = max(matches, key=lambda pair: len(segments(pair[0])))[1]
        children = [child for source, child in self.facts.items() if covers(path, source)]
        if value is None or any(child is None for child in children):
            return None
        mutations = dict.fromkeys(value.mutation_ids)
        observations = dict.fromkeys(value.observation_ids)
        for child in children:
            mutations.update(dict.fromkeys(child.mutation_ids))
            observations.update(dict.fromkeys(child.observation_ids))
            if child.anchor_id is not None and child.anchor_id != value.anchor_id:
                observations[child.anchor_id] = None
        return replace(value, mutation_ids=tuple(mutations), observation_ids=tuple(observations))

    def known_paths(self, referenced=()):
        return {path for path in (*self.facts, *referenced)
                if path.startswith("$.state_before") and self.get(path) is not None}

    def _set(self, path, value, *, preserve_children=False):
        children = {child: self.get(child) for child in self.facts if child != path and covers(path, child)}
        self.facts = {source: fact for source, fact in self.facts.items() if not covers(path, source)}
        self.facts[path] = value
        if preserve_children:
            self.facts.update({child: prior for child, prior in children.items()
                               if prior is not None and prior.anchor_id is not None})

    def observe(self, proofs, step_id, *, phase="before"):
        prefix = "$.state_before" if phase == "before" else "$.state_after"
        for source, evidence in proofs.items():
            if not source.startswith(prefix) or evidence["kind"] == "earlier_state":
                continue
            path = before(source)
            previous = self.get(path)
            if previous is None or previous.anchor_id is None:
                value = KnownValue(step_id, evidence, anchor_phase=phase)
                self._set(path, value, preserve_children=True)

    def apply_mutations(self, mutations, proofs, step_id):
        """Apply all effects, but never infer an effect with unknown state inputs."""
        statuses = []
        for index, mutation in enumerate(mutations):
            missing = [source for source in mutation_sources(mutation)
                       if not any(temporal_covers(path, source) for path in proofs)]
            missing_identities = [item for item in mutation.get("target_identity_sources", [])
                                  if item["path"] in missing]
            statuses.append({"mutation_index": index, "target": mutation["target"],
                             "sources_fixed": not missing, "missing_sources": missing,
                             "missing_target_identities": missing_identities})
        for mutation, status in zip(mutations, statuses):
            target = before(mutation["target"])
            if not status["sources_fixed"]:
                # An unresolved address can affect any child of its containing map.
                # Audit side effects conservatively without scheduling more anchors.
                for item in status["missing_target_identities"]:
                    token = "{" + item["placeholder"] + "}"
                    position = target.find(token)
                    if position >= 0:
                        target = target[:target.rfind("[", 0, position)]
                self._set(target, None)
                continue
            prior = self.get(target) or KnownValue(None, None)
            children = {child: self.get(child) for child in self.facts if child != target and covers(target, child)}
            self._set(target, replace(prior, mutation_ids=(*prior.mutation_ids, step_id)))
            for child, value in children.items():
                if value is not None:
                    self.facts[child] = replace(value, mutation_ids=(*value.mutation_ids, step_id))
        self.observe(proofs, step_id, phase="after")
        for status in statuses:
            status["post_state_known"] = self.get(status["target"]) is not None
        return statuses
