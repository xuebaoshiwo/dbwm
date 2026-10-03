"""Small serialization and plugin helpers shared by experiment commands."""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import tempfile
import time
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        # Windows readers/indexers can briefly hold a handle without delete
        # sharing. Preserve the atomic replacement and retry only that step.
        for attempt in range(10):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def load_callable(reference):
    module, name = reference.split(":", 1)
    return getattr(importlib.import_module(module), name)


def load_tools(path):
    text = Path(path).read_text(encoding="utf-8-sig")
    try:
        tools = json.loads(text)
    except json.JSONDecodeError:
        tools = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(tools, dict):
        tools = tools["tools"]
    if not isinstance(tools, list) or not tools:
        raise ValueError("Tool document must contain a nonempty list of tools")
    return tools
