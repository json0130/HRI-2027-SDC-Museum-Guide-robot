"""Tiny YAML config loader shared by the museum modules.

Code holds the defaults; ``config/<name>.yaml`` overrides them one level deep (same merge
rule as Library-Book-Robot ``Config.section``), so a partial file is fine and a missing
file means "defaults".
"""

from __future__ import annotations

import os
from typing import Optional

import yaml

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "config")


def load_config(name: str, defaults: dict, path: Optional[str] = None) -> dict:
    path = path or os.path.join(CONFIG_DIR, f"{name}.yaml")
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in defaults.items()}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    for k, v in data.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    return out
