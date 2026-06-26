"""Unit tests for the structural evaluation metrics (Phase 4): they must expose the
failures edge_parent_f1 cannot see (isolated nodes, lost/spurious starts, fragmentation)
without disturbing the existing metrics. Zero extra deps.
"""
from __future__ import annotations
import json
import os
import sys

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT, "src"))

from evaluation.metrics import (  # noqa: E402
    evaluate, structural_metrics, isolated_node_count, weak_component_count)

GOLD = os.path.join(_PROJECT, "data", "casestudy", "gold", "Cobalt_Kitty_FlowCloud.json")
OUT = os.path.join(_PROJECT, "data", "casestudy", "output", "Cobalt_Kitty_GLANCE_output.json")

def _g(p):
    return json.load(open(p))

def test_gold_is_structurally_perfect_against_itself():
    m = structural_metrics(_g(GOLD), _g(GOLD))
    assert m["isolated_node_count"] == 0
    assert m["connected_components"] == 1
    assert m["structure_valid"] is True
    assert m["root_set_f1"] == 1.0 and m["reachability_parent_f1"] == 1.0

def test_shipped_output_structural_failures_are_visible():
    pred, gold = _g(OUT), _g(GOLD)
    m = structural_metrics(pred, gold)
    # the exact symptoms the user reported, now measured
    assert m["isolated_node_count"] == 1
    assert m["connected_components"] == 2
    assert m["gold_connected_components"] == 1
    assert m["structure_valid"] is False

def test_deterministic_fix_makes_structure_valid():
    pred, gold = _g(OUT), _g(GOLD)
    # drop the isolated node (what the safety net does deterministically)
    ids, indeg, outdeg = __import__("evaluation.metrics", fromlist=["_degrees"])._degrees(pred)
    iso = {nid for nid in ids if indeg[nid] == 0 and outdeg[nid] == 0}
    fixed = dict(pred)
    fixed["nodes"] = [n for i, n in enumerate(pred["nodes"]) if ids[i] not in iso]
    keep = {ids[i] for i in range(len(ids)) if ids[i] not in iso}
    fixed["edges"] = [e for e in pred["edges"] if e["src"] in keep and e["dst"] in keep]
    m = structural_metrics(fixed, gold)
    assert m["isolated_node_count"] == 0
    assert m["connected_components"] == 1
    assert m["structure_valid"] is True

def test_evaluate_merges_structural_keys_without_regressing_edge_parent_f1():
    res = evaluate(_g(OUT), _g(GOLD))
    # structural keys are present in the standard evaluation output
    for k in ("isolated_node_count", "connected_components", "root_set_f1",
              "reachability_parent_f1", "structure_valid", "subtechnique_fraction"):
        assert k in res, k
    # the headline metric is unchanged (deterministic; README reports 0.88)
    assert abs(res["edge_parent_f1"] - 0.88) < 0.03, res["edge_parent_f1"]
    assert res["isolated_node_count"] == 1

def test_subtechnique_fraction_quantifies_coarsening():
    # gold uses many subtechniques (T####.###); the coarsened output uses fewer
    g = structural_metrics(_g(OUT), _g(GOLD))
    assert g["gold_subtechnique_fraction"] > g["subtechnique_fraction"]

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
