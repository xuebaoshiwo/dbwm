"""Allocate symbolic identities directly from path placeholders."""

import re
from collections import Counter

from bfcl_eval.consistency.data_generator_v2.symbolic_dependency_graph import segments


PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_.]*)\}")
INSTANCE = re.compile(r"(.+)_([1-9][0-9]*)$")


def placeholder_name(name):
    name = re.sub(r"^(?:args\.|result\.|target_)", "", name)
    return name


def substitute(value, bindings):
    if isinstance(value, str):
        return PLACEHOLDER.sub(lambda match: "{" + bindings.get(match[1], match[1]) + "}", value)
    if isinstance(value, dict):
        return {key: substitute(item, bindings) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(substitute(item, bindings) for item in value)
    return value


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from strings(item)


def path_bindings(source, path):
    """Align index slots structurally, including differently named tool aliases."""
    bindings = {}
    for left, right in zip(segments(source), segments(path)):
        if left.startswith("{") and right.startswith("{"):
            name, identity = left[1:-1], right[1:-1]
            if name in bindings and bindings[name] != identity:
                raise ValueError("A repeated placeholder cannot bind to different identities")
            bindings[name] = identity
    return bindings


class PlaceholderPool:
    def __init__(self):
        self.counts = Counter()

    def bindings(self, value, source=None, path=None):
        aliases = path_bindings(source, path) if source is not None else {}
        groups = {placeholder_name(name): identity for name, identity in aliases.items()}
        counts = self.counts.copy()
        result = {}
        for text in strings(value):
            for name in PLACEHOLDER.findall(text):
                if name in result:
                    continue
                if INSTANCE.fullmatch(name):
                    result[name] = name
                    continue
                group = placeholder_name(name)
                if group not in groups:
                    counts[group] += 1
                    groups[group] = f"{group}_{counts[group]}"
                result[name] = groups[group]
        return result

    def commit(self, bindings):
        for identity in bindings.values():
            match = INSTANCE.fullmatch(identity)
            if match:
                self.counts[match[1]] = max(self.counts[match[1]], int(match[2]))

    def allocate(self, path):
        bindings = self.bindings(path)
        self.commit(bindings)
        return substitute(path, bindings)
