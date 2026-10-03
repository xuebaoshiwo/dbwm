"""Execute independent domain fixtures to produce real backend few-shot outputs."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from .common import digest, load_callable, load_tools, read_json, write_json


def generate(config, root):
    tools = load_tools(root / config["tool_schema"])
    fixtures = load_callable(config["example_fixtures"])()
    backend = load_callable(config["backend"])
    names = {tool["name"] for tool in tools}
    if set(fixtures) != names or any(len(fixtures[name]) != 3 for name in names):
        raise ValueError("Fixtures must cover every tool exactly three times")
    examples = {}
    for name in sorted(names):
        examples[name] = []
        for fixture in fixtures[name]:
            instance = backend()
            getattr(instance, config["state_loader"])(deepcopy(fixture["initial_state"]),
                                                       **config.get("loader_options", {}))
            call = {"tool": name, "args": deepcopy(fixture["args"])}
            observation = deepcopy(getattr(instance, name)(**call["args"]))
            examples[name].append({"scenario": fixture["label"], "tool_call": call,
                                   "observation": observation,
                                   "provenance": {"initial_state": fixture["initial_state"],
                                                  "backend": config["backend"]}})
    return {"selection_policy": "Three representative common scenarios per tool, curated independently of test cases. "
            "Tools with fewer than three distinct behaviors repeat a behavior with a different scenario; "
            "the fixed clock necessarily returns the same observation. No empirical frequency ranking is claimed.",
            "schema_sha256": digest(tools), "tools": examples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    root = args.config.resolve().parent
    examples = generate(config, root)
    destination = root / config["examples"]
    write_json(destination, examples)
    print(f"Generated {len(examples['tools']) * 3} backend examples: {destination}")


if __name__ == "__main__":
    main()
