"""Entity instances, monitored fields, and per-call symbolic key bindings."""

import re
from dataclasses import dataclass, field

from bfcl_eval.consistency.data_generator_v2.symbolic_dependency_graph import segments


IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
SLOT = re.compile(r"\{([^{}]+)\}")


def map_strings(value, transform):
    if isinstance(value, str):
        return transform(value)
    if isinstance(value, list):
        return [map_strings(item, transform) for item in value]
    if isinstance(value, dict):
        return {key: map_strings(item, transform) for key, item in value.items()}
    return value


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)


@dataclass(frozen=True)
class EntityType:
    name: str
    paths: tuple[str, ...]
    fields: dict[str, str]
    relations: dict

    @property
    def keyed(self):
        return "{id}" in self.paths[0]

    def marker(self, entity):
        return f"@{self.name}:{entity}"

    def field_path(self, entity, field):
        return self.fields[field].replace("{id}", self.marker(entity))

    def keys_in(self, text):
        for path in self.paths:
            pattern = re.escape(path).replace("state_before", "state_(?:before|after)")
            if self.keyed:
                pattern = pattern.replace(re.escape("{id}"), "([^'\\]]+)")
                for match in re.finditer(pattern, text):
                    yield match.group(1)


@dataclass(frozen=True)
class CallBinding:
    entities: dict[str, str]
    slots: dict[str, str]
    keys: dict = field(default_factory=dict)
    links: dict[tuple[str, str], str] = field(default_factory=dict)


def parse_entity_types(data):
    if not isinstance(data, dict):
        raise ValueError("Entity types must be an object")
    types = {}
    for name, entry in data.items():
        if not isinstance(entry, dict) or not IDENTIFIER.fullmatch(name) or not {"paths", "fields"} <= set(entry) <= {
            "paths", "fields", "relations"
        }:
            raise ValueError(f"Invalid entity type: {name}")
        paths, fields = entry["paths"], entry["fields"]
        if not isinstance(paths, list) or not paths or any(
            not isinstance(path, str) or not path.startswith("$.state_before.") for path in paths
        ) or len({"{id}" in path for path in paths}) != 1 or any(
            SLOT.findall(path) not in ([], ["id"]) or ("{id}" in path and "['{id}']" not in path) for path in paths
        ):
            raise ValueError(f"Invalid entity paths: {name}")
        if not isinstance(fields, dict) or not fields or any(
            not IDENTIFIER.fullmatch(field) or not isinstance(path, str) or not any(
                segments(path)[:len(segments(root))] == segments(root) for root in paths
            ) for field, path in fields.items()
        ):
            raise ValueError(f"Invalid entity fields: {name}")
        relations = entry.get("relations", {})
        if not isinstance(relations, dict):
            raise ValueError(f"Entity relations must be an object: {name}")
        types[name] = EntityType(name, tuple(paths), dict(fields), relations)
    for kind in types.values():
        for field, relation in kind.relations.items():
            if not isinstance(relation, dict) or field not in kind.fields or set(relation) != {"type", "slots", "arguments"} or (
                relation["type"] not in types or not types[relation["type"]].keyed
            ) or not isinstance(relation["slots"], list) or any(not isinstance(slot, str) for slot in relation["slots"]) or (
                not isinstance(relation["arguments"], dict) or any(not isinstance(path, str) or not path.startswith("$.args.")
                                                               for path in relation["arguments"].values())
            ):
                raise ValueError(f"Invalid entity relation: {kind.name}.{field}")
    return types


def monitor_targets(monitors, types):
    """Expand instance-level monitoring into field intervals for the sampler."""
    targets = {}
    seen = set()
    singletons = set()
    for monitor in monitors:
        if not isinstance(monitor, dict) or set(monitor) != {"id", "type", "fields"}:
            raise ValueError("A monitor requires id, type, and fields")
        entity, name, fields = monitor["id"], monitor["type"], monitor["fields"]
        if not isinstance(entity, str) or not IDENTIFIER.fullmatch(entity) or entity in seen or name not in types:
            raise ValueError(f"Invalid or duplicate monitored entity: {entity}")
        kind = types[name]
        if not isinstance(fields, list) or not fields or len(set(fields)) != len(fields) or set(fields) - set(kind.fields):
            raise ValueError(f"Choose distinct declared fields for {entity}")
        if not kind.keyed and name in singletons:
            raise ValueError(f"Singleton entity type cannot have multiple instances: {name}")
        seen.add(entity)
        if not kind.keyed:
            singletons.add(name)
        for field in fields:
            targets[f"{entity}.{field}"] = {"path": kind.field_path(entity, field),
                "entity": entity, "entity_type": name, "field": field}
    if not targets:
        raise ValueError("Choose at least one monitored entity")
    return targets


def parse_monitor(text):
    entity, equals, selection = text.partition("=")
    kind, colon, fields = selection.partition(":")
    if not equals or not colon:
        raise ValueError("Monitor syntax is INSTANCE=TYPE:FIELD[,FIELD...]")
    return {"id": entity, "type": kind, "fields": fields.split(",")}


class EntityRegistry:
    """Preview bindings without allocation; commit only selected calls."""

    def __init__(self, types, targets):
        self.types = types
        self.instances = {}
        self.links = {}
        for target in targets.values():
            entity, name = target["entity"], target["entity_type"]
            if entity in self.instances and self.instances[entity] != name:
                raise ValueError(f"Entity ID used by different types: {entity}")
            if not types[name].keyed and any(kind == name and other != entity for other, kind in self.instances.items()):
                raise ValueError(f"Singleton entity type cannot have multiple instances: {name}")
            self.instances[entity] = name

    def next_id(self, name):
        if not self.types[name].keyed:
            existing = next((entity for entity, kind in self.instances.items() if kind == name), None)
            if existing is not None:
                return existing
            if name not in self.instances:
                return name
        index = 1
        while f"{name}_{index}" in self.instances:
            index += 1
        return f"{name}_{index}"

    def bindings_for(self, document, desired=None):
        texts = list(strings(document))
        slots = {}
        used = set()
        for name, kind in self.types.items():
            argument_slots = set()
            for text in texts:
                if kind.keyed:
                    keys = list(kind.keys_in(text))
                    if keys:
                        used.add(name)
                    for key in keys:
                        if SLOT.fullmatch(key):
                            slot = key[1:-1]
                            if slot in slots and slots[slot] != name:
                                raise ValueError(f"Entity key slot belongs to multiple types: {slot}")
                            slots[slot] = name
                            if slot.startswith("args."):
                                argument_slots.add(slot)
                        elif key != "*":
                            raise ValueError(f"Concrete entity keys in tool specifications are unsupported: {key}")
                elif any(path.replace("state_before", "state_after") in text or path in text for path in kind.paths):
                    used.add(name)
            if len(argument_slots) > 1:
                raise ValueError(f"Multiple independent {name} IDs in one tool require role bindings")
        bindings = {}
        if desired:
            for name, kind in self.types.items():
                for key in kind.keys_in(desired):
                    if key.startswith(f"@{name}:"):
                        bindings[name] = key.split(":", 1)[1]
                if not kind.keyed and any(desired.startswith(path) for path in kind.paths):
                    bindings[name] = self.next_id(name)
        desired_bindings = dict(bindings)
        for name in sorted(used):
            if name not in bindings:
                bindings[name] = self.next_id(name)

        # A property used as another entity's key keeps its identity across calls.
        links = {}
        for owner_type, kind in self.types.items():
            if owner_type not in bindings:
                continue
            owner = bindings[owner_type]
            for field, relation in kind.relations.items():
                related_type = relation["type"]
                previous = self.links.get((owner, field))
                relevant = any(slot in slots for slot in relation["slots"]) or document["tool"] in relation["arguments"]
                if not relevant:
                    continue
                if previous and related_type in bindings and desired and any(
                    key == self.types[related_type].marker(bindings[related_type])
                    for key in self.types[related_type].keys_in(desired)
                ) and bindings[related_type] != previous:
                    return None
                existing = [entity for entity, name in self.instances.items() if name == related_type]
                preferred = existing[0] if len(existing) == 1 else None
                related = previous or desired_bindings.get(related_type) or preferred or bindings.get(related_type) or self.next_id(related_type)
                bindings[related_type] = related
                links[owner, field] = related
                for slot in relation["slots"]:
                    if slot in slots:
                        slots[slot] = related_type
        replacements = {slot: self.types[name].marker(bindings[name]) for slot, name in slots.items()}
        key_bindings = {"$." + slot: {"entity": bindings[name], "field": "key"}
                        for slot, name in slots.items() if slot.startswith(("args.", "result."))}
        for (owner, field), related in links.items():
            owner_type = next(name for name, entity in bindings.items() if entity == owner)
            relation = self.types[owner_type].relations[field]
            argument = relation["arguments"].get(document["tool"])
            if argument:
                key_bindings[argument] = {"entity": related, "field": "key"}
                replacements[argument.removeprefix("$.")] = self.types[relation["type"]].marker(related)
        return CallBinding(bindings, replacements, key_bindings, links)

    def commit(self, binding):
        for name, entity in binding.entities.items():
            if entity in self.instances and self.instances[entity] != name:
                raise ValueError(f"Entity ID used by different types: {entity}")
            self.instances[entity] = name
        self.links.update(binding.links)

    def describe(self, targets):
        monitored = {}
        for name, target in targets.items():
            monitored.setdefault(target["entity"], {})[target["field"]] = name
        return {entity: {"type": kind, "key": self.types[kind].marker(entity) if self.types[kind].keyed else None,
                         "monitored_fields": monitored.get(entity, {}),
                         "relations": {field: related for (owner, field), related in self.links.items() if owner == entity}}
                for entity, kind in self.instances.items()}
