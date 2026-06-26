from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Local helpers — experience.py deliberately has no project-internal imports so
# that memory_engine can depend on it without any circular-import risk.

def parent_of(technique: Any) -> str:
    """Parent ATT&CK technique id (T####) for a technique / subtechnique string."""
    m = re.search(r"T\d{4}", str(technique or ""))
    return m.group(0) if m else ""

def as_list(x: Any) -> list:
    return x if isinstance(x, list) else ([x] if x else [])

# Canonical vocabularies. Extend these (and add a seed record) to introduce a new
# kind of experience — no new file format, no new matcher.
SCOPES = ("node", "edge", "structural")
ACTIONS = (
    "add_edge", "remove_edge", "reverse_edge", "reconnect", "reconnect_or_drop",
    "reconnect_or_remap", "drop_node", "insert_node", "remap", "flag",
)
# Structural anomaly kinds emitted by graph_structure.analyze_structure(); a
# structural experience advertises which it speaks to via trigger.kinds.
STRUCTURAL_KINDS = ("isolated", "illegitimate_root", "disconnected_component")

# polarity -> default repair action, for the (homogeneous) edge stores.
_EDGE_POLARITY_ACTION = {
    "neg": "remove_edge", "wrong": "remove_edge",
    "pos": "add_edge", "missing": "add_edge",
    "review": "flag",
}

@dataclass
class Experience:
    """One uniformly-typed piece of cross-report experience.

    A single schema spans the three previously-separate seed stores —
    node memory (``nodes.json``), edge rules (``rules.json``), edge cases
    (``cases.json``) — plus the new ``structural`` scope. Adding a new kind of
    experience becomes: append one record in this shape. The adapters below read
    the legacy formats verbatim (they never mutate the on-disk seed data), so the
    abstraction is fully backward compatible.
    """
    id: str
    scope: str                                       # one of SCOPES
    polarity: str = "flag"                            # neg|pos|wrong|missing|review|find|correct|flag
    trigger: dict = field(default_factory=dict)       # techniques_src/dst, keywords_src/dst, kw_mode,
    #                                                   exclude_kw, structural_condition, kinds
    action: str = "flag"                              # one of ACTIONS
    hint: str = ""
    confidence: float = 1.0
    provenance: dict = field(default_factory=dict)    # source, doc_id, count, w_pos, w_neg, ts, corrector_stage
    tier: str | None = None                           # explicit_rule|implicit_rule (edge rules only)
    hit: dict = field(default_factory=dict)           # runtime: concrete pairs/edges/nodes that matched

    def to_dict(self) -> dict:
        d = {
            "id": self.id, "scope": self.scope, "polarity": self.polarity,
            "trigger": self.trigger, "action": self.action, "hint": self.hint,
            "confidence": self.confidence, "provenance": self.provenance,
        }
        if self.tier is not None:
            d["tier"] = self.tier
        if self.hit:
            d["hit"] = self.hit
        return d

    def index_entry(self) -> dict:
        """Compact, scannable row for the experience index (the MEMORY.md analogue)."""
        return {
            "id": self.id, "scope": self.scope, "polarity": self.polarity,
            "action": self.action,
            "confidence": round(float(self.confidence or 0.0), 3),
            "count": int((self.provenance or {}).get("count", 0) or 0),
            "source": (self.provenance or {}).get("source", "seed"),
            "hint": (self.hint or "")[:120],
        }

    # ------------------------------------------------------------------
    # Adapters — read the existing seed formats verbatim (no on-disk change).
    # ------------------------------------------------------------------
    @staticmethod
    def _edge_trigger(d: dict) -> dict:
        return {
            "techniques_src": as_list(d.get("src")),
            "techniques_dst": as_list(d.get("dst")),
            "keywords_src": as_list(d.get("src_kw")),
            "keywords_dst": as_list(d.get("dst_kw")),
            "kw_mode": d.get("kw_mode", "either"),
        }

    @staticmethod
    def from_rule(d: dict) -> "Experience":
        """``rules.json`` entry: {id, polarity(neg/pos/review), tier, mode, src, dst, src_kw, dst_kw, kw_mode, hint}."""
        pol = str(d.get("polarity") or "")
        exp = Experience(
            id=str(d.get("id", "")), scope="edge", polarity=pol,
            trigger=Experience._edge_trigger(d),
            action=_EDGE_POLARITY_ACTION.get(pol, "flag"),
            hint=str(d.get("hint", "")), tier=d.get("tier"),
            provenance={"source": "seed"},
        )
        if d.get("hit_pairs"):
            exp.hit = {"hit_pairs": d.get("hit_pairs")}
        return exp

    @staticmethod
    def from_case(d: dict) -> "Experience":
        """``cases.json`` entry: same five-tuple as a rule; polarity wrong/missing/review."""
        pol = str(d.get("polarity") or "")
        exp = Experience(
            id=str(d.get("id", "")), scope="edge", polarity=pol,
            trigger=Experience._edge_trigger(d),
            action=_EDGE_POLARITY_ACTION.get(pol, "flag"),
            hint=str(d.get("hint", "")),
            provenance={"source": "seed", "kind": "case"},
        )
        if d.get("hit_edges"):
            exp.hit = {"hit_edges": d.get("hit_edges")}
        return exp

    @staticmethod
    def from_node_memory(d: dict) -> "Experience":
        """``nodes.json`` ``memory`` entry: {id, tech, when_kw, exclude_kw, hint, find_node}."""
        find = bool(d.get("find_node"))
        return Experience(
            id=str(d.get("id", "")), scope="node",
            polarity="find" if find else "correct",
            trigger={
                "techniques_src": as_list(d.get("tech")),
                "keywords_src": as_list(d.get("when_kw")),
                "exclude_kw": as_list(d.get("exclude_kw")),
            },
            action="insert_node" if find else "flag",
            hint=str(d.get("hint", "")), provenance={"source": "seed"},
        )

    @staticmethod
    def from_structural(d: dict) -> "Experience":
        """``structural.json`` entry: {id, polarity, trigger:{kinds:[...], structural_condition:{...}}, action, hint, confidence}."""
        return Experience(
            id=str(d.get("id", "")), scope="structural",
            polarity=str(d.get("polarity", "flag")),
            trigger=dict(d.get("trigger") or {}),
            action=str(d.get("action", "flag")),
            hint=str(d.get("hint", "")),
            confidence=float(d.get("confidence", 1.0) or 1.0),
            provenance={"source": "seed"},
        )

    @staticmethod
    def from_dict(d: dict) -> "Experience":
        """Inverse of ``to_dict`` — load a learned (non-seed) experience back."""
        return Experience(
            id=str(d.get("id", "")), scope=str(d.get("scope", "")),
            polarity=str(d.get("polarity", "flag")),
            trigger=dict(d.get("trigger") or {}),
            action=str(d.get("action", "flag")),
            hint=str(d.get("hint", "")),
            confidence=float(d.get("confidence", 1.0) or 1.0),
            provenance=dict(d.get("provenance") or {}),
            tier=d.get("tier"), hit=dict(d.get("hit") or {}),
        )

    def dedup_key(self) -> tuple:
        """Content key for de-duplication — the 'is there already a record covering
        this?' check, mirroring the harness memory discipline."""
        t = self.trigger or {}
        src = tuple(sorted(parent_of(x) for x in as_list(t.get("techniques_src"))))
        dst = tuple(sorted(parent_of(x) for x in as_list(t.get("techniques_dst"))))
        kinds = tuple(sorted(str(k) for k in as_list(t.get("kinds"))))
        hint = re.sub(r"\s+", " ", (self.hint or "").strip().lower())[:80]
        return (self.scope, self.polarity, self.action, src, dst, kinds, hint)
