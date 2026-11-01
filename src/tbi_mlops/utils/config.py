"""YAML configuration loading with profile overrides (dev / research)."""
from __future__ import annotations
import copy, os, pathlib
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[3]
CONFIG_DIR = ROOT / "configs"


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load_yaml(name: str, profile: str | None = None) -> dict:
    with open(CONFIG_DIR / f"{name}.yaml") as f:
        cfg = yaml.safe_load(f) or {}
    profiles = cfg.pop("profiles", {})
    if profile and profile in profiles:
        cfg = _merge(cfg, profiles[profile])
    return cfg


def load_config(profile: str | None = None) -> dict:
    profile = profile or os.environ.get("TBI_PROFILE", "dev")
    cfg = {n: load_yaml(n, profile) for n in ["data", "model", "training", "monitoring", "simulation"]}
    cfg["profile"] = profile
    return cfg


def path(*parts: str) -> pathlib.Path:
    p = ROOT.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
