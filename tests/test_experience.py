"""Unit tests for the unified experience layer (Phase 1).

Zero extra dependencies: runnable directly (`.venv/bin/python tests/test_experience.py`)
and also discoverable by pytest (functions are named test_*).
"""
from __future__ import annotations
import json
import os
import sys

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT, "src"))

from experience import Experience, parent_of, as_list, SCOPES, STRUCTURAL_KINDS  # noqa: E402
from memory_engine import MemoryEngine  # noqa: E402

SEED = os.path.join(_PROJECT, "data", "memory_seed")

def _mem() -> MemoryEngine:
    return MemoryEngine(SEED)

def test_parent_of_and_aslist():
    assert parent_of("T1059.001") == "T1059"
    assert parent_of("step_3_add_T1027") == "T1027"
    assert parent_of("") == ""
    assert as_list(None) == []
    assert as_list("x") == ["x"]
    assert as_list(["a", "b"]) == ["a", "b"]

def test_adapters_preserve_polarity_and_action():
    r = Experience.from_rule({"id": "r1", "polarity": "neg", "tier": "explicit_rule",
                              "src": ["T1071"], "dst": ["T1041"], "src_kw": ["c2"], "kw_mode": "either"})
    assert r.scope == "edge" and r.polarity == "neg" and r.action == "remove_edge"
    assert r.trigger["techniques_src"] == ["T1071"] and r.tier == "explicit_rule"

    c = Experience.from_case({"id": "c1", "polarity": "missing", "src": ["T1204"], "dst": ["T1203"]})
    assert c.scope == "edge" and c.action == "add_edge"

    n = Experience.from_node_memory({"id": "n1", "tech": "T1071.001", "when_kw": ["http"], "find_node": True})
    assert n.scope == "node" and n.action == "insert_node" and n.polarity == "find"

    s = Experience.from_structural({"id": "s1", "trigger": {"kinds": ["isolated"]},
                                    "action": "reconnect_or_drop", "hint": "x", "confidence": 0.9})
    assert s.scope == "structural" and s.action == "reconnect_or_drop" and s.confidence == 0.9

def test_index_covers_all_seed_stores():
    rows = _mem().experience_index()
    scopes = {r["scope"] for r in rows}
    assert scopes == set(SCOPES), scopes
    # each row is compact + scannable
    assert all(set(r) >= {"id", "scope", "polarity", "action", "hint", "confidence"} for r in rows)
    assert len(rows) > 100  # the shipped seed has hundreds of experiences

def test_match_experiences_on_output_graph():
    g = json.load(open(os.path.join(_PROJECT, "data", "casestudy", "output",
                                    "Cobalt_Kitty_GLANCE_output.json")))
    nodes = [{"node_id": n["node_id"], "technique_id": n.get("attack_id") or n.get("technique_id"),
              "procedure": (n.get("metadata") or {}).get("procedure", {})} for n in g["nodes"]]
    edges = [{"src": e["src"], "dst": e["dst"]} for e in g["edges"]]
    exps = _mem().match_experiences(nodes, edges)
    assert exps, "expected at least some experiences to fire on the shipped output"
    assert all(e.scope in ("node", "edge") for e in exps)

def test_get_structural_experiences_by_kind():
    mem = _mem()
    report = {"findings": [{"kind": "isolated", "node_id": "step_3_add_T1027", "technique_id": "T1027"}]}
    se = mem.get_structural_experiences([], [], report)
    ids = {x["id"] for x in se}
    assert "struct_isolated_node" in ids
    assert "struct_no_root_late_tactic" not in ids  # only the matching kind fires
    # empty report -> nothing
    assert mem.get_structural_experiences([], [], {"findings": []}) == []
    # every advertised kind is known to the schema
    for d in mem._load_structural():
        for k in (d.get("trigger") or {}).get("kinds", []):
            assert k in STRUCTURAL_KINDS, k

def test_dedup_key_is_content_addressed():
    a = Experience.from_structural({"id": "a", "trigger": {"kinds": ["isolated"]},
                                    "action": "reconnect_or_drop", "hint": "  Hello  World "})
    b = Experience.from_structural({"id": "b", "trigger": {"kinds": ["isolated"]},
                                    "action": "reconnect_or_drop", "hint": "hello world"})
    assert a.dedup_key() == b.dedup_key()
    c = Experience.from_structural({"id": "c", "trigger": {"kinds": ["illegitimate_root"]},
                                    "action": "reconnect_or_drop", "hint": "hello world"})
    assert a.dedup_key() != c.dedup_key()

def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print("PASS", fn.__name__)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print("FAIL", fn.__name__, "->", repr(exc))
    print("\n%d/%d passed" % (len(fns) - failed, len(fns)))
    return 1 if failed else 0

if __name__ == "__main__":
    raise SystemExit(_run())
