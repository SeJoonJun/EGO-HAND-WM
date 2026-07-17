"""Small dependency-free YAML configuration loader with dotted CLI overrides."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import yaml


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping at the root of {path}")
    config = deepcopy(config)
    for override in overrides:
        apply_override(config, override)
    return config


def apply_override(config: dict[str, Any], expression: str) -> None:
    if "=" not in expression:
        raise ValueError(f"Override must be KEY=VALUE, got {expression!r}")
    dotted_key, raw_value = expression.split("=", 1)
    keys = dotted_key.split(".")
    if any(not key for key in keys):
        raise ValueError(f"Invalid dotted key: {dotted_key!r}")
    value = yaml.safe_load(raw_value)
    node: dict[str, Any] = config
    for key in keys[:-1]:
        child = node.get(key)
        if child is None:
            child = {}
            node[key] = child
        if not isinstance(child, dict):
            raise ValueError(f"Cannot descend through non-mapping config key {key!r}")
        node = child
    if keys[-1] not in node:
        raise KeyError(
            f"Unknown config key {dotted_key!r}; add it to the YAML before overriding it"
        )
    node[keys[-1]] = value


def require(config: dict[str, Any], dotted_key: str) -> Any:
    node: Any = config
    for key in dotted_key.split("."):
        if not isinstance(node, dict) or key not in node:
            raise KeyError(f"Missing required config key: {dotted_key}")
        node = node[key]
    return node
