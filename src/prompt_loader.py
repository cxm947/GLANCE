from __future__ import annotations
from pathlib import Path
import yaml as _yaml

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

def load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / (name + ".txt")).read_text(encoding="utf-8").rstrip("\n")

def load_template(name: str) -> dict[str, str]:
    data = _yaml.safe_load((_PROMPTS_DIR / (name + ".yaml")).read_text(encoding="utf-8")) or {}
    out = {k: (v.rstrip("\n") if isinstance(v, str) else v) for k, v in data.items()}
    out.setdefault("system", "")
    out.setdefault("user", "")
    return out
