from __future__ import annotations
import copy
import os
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parents[1]
CONFIGS = PROJECT / "configs"

ENV_CANDIDATES = [PROJECT / ".env", PROJECT / ".env.deepseek",
                  CONFIGS / "secrets" / ".edl_env_deepseek"]

def _providers() -> dict:
    p = CONFIGS / "providers.yaml"
    return (yaml.safe_load(p.read_text(encoding="utf-8")) or {}) if p.exists() else {}

def _resolve_profile(profile=None) -> str:
    return profile or os.environ.get("EDL_PROFILE") or _providers().get("default") or "deepseek"

def _profile_env_file(profile: str):
    ef = ((_providers().get("profiles") or {}).get(profile, {}) or {}).get("env_file")
    return (CONFIGS / ef) if ef else None

def load_env(path=None, profile=None) -> None:
    if path is None:
        path = _profile_env_file(_resolve_profile(profile))
    files = [Path(path)] if path else ENV_CANDIDATES
    for f in files:
        if not f or not f.exists():
            continue
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            if "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip().strip('"').strip("'")
        break

def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _deep_merge(out[k], v) if (
            isinstance(v, dict) and isinstance(out.get(k), dict)) else v
    return out

def load_config(experiment=None, *, profile=None, memory_dir=None, output_dir=None,
                require_api_key=True) -> dict:
    prof = _resolve_profile(profile)
    load_env(profile=prof)
    cfg = yaml.safe_load((CONFIGS / "default.yaml").read_text("utf-8")) or {}
    if experiment:
        exp = CONFIGS / "experiments" / f"{experiment}.yaml"
        if not exp.exists():
            avail = sorted(p.stem for p in (CONFIGS / "experiments").glob("*.yaml"))\
                if (CONFIGS / "experiments").exists() else []
            raise FileNotFoundError(
                "No experiment override '%s' (expected configs/experiments/%s.yaml). "
                "Available: %s" % (experiment, experiment, avail or "(none)"))
        cfg = _deep_merge(cfg, yaml.safe_load(exp.read_text("utf-8")) or {})
    cfg["provider"] = prof
    cfg["api_key"] = os.environ.get("EDL_API_KEY")
    cfg["base_url"] = os.environ.get("EDL_BASE_URL", cfg.get("base_url"))
    cfg["model"] = os.environ.get("EDL_MODEL", cfg.get("model"))
    if memory_dir:
        cfg["memory_dir"] = str(memory_dir)
    if output_dir:
        cfg["output_dir"] = str(output_dir)
    if require_api_key and not cfg.get("api_key"):
        raise RuntimeError(
            "No EDL_API_KEY for profile '%s'. `cp .env.example .env` and fill it, or "
            "`set -a; source configs/secrets/.edl_env_deepseek; set +a`." % prof)
    return cfg
