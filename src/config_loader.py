import copy
import json
import os

try:
    from config import CONFIG
except ImportError:
    from .config import CONFIG


def load_config():
    cfg = copy.deepcopy(CONFIG)
    override_path = os.path.join(os.path.dirname(__file__), "config_override.json")

    if os.path.exists(override_path):
        with open(override_path, "r", encoding="utf-8") as f:
            override = json.load(f) or {}
        if not isinstance(override, dict):
            raise ValueError("config_override.json must contain a JSON object.")
        cfg.update(override)

    return cfg
