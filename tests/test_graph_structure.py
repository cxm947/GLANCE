"""Unit tests for global structure analysis + deterministic pipeline repair (Phase 2).

Covers the parts that must hold *by construction* (not by LLM luck): anomaly
detection, root selection, the structure applier, the isolated-node safety net,
and the extended schema validator. Zero extra deps.
"""
from __future__ import annotations
import json
import os
import sys
from types import SimpleNamespace

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT, "src"))

from graph_structure import analyze_structure, weakly_connected_components  # noqa: E402
from pipeline_clean import CleanPipeline  # noqa: E402
from schema import GraphDocument, GraphNode, GraphEdge  # noqa: E402

def _n(nid, tech, sid=None):
    return {"node_id": nid, "technique_id": tech,
            "evidence_sentence_ids": [sid] if sid is not None else []}

def _e(s, d):
    return {"src": s, "dst": d}

def test_detect_isolated_node():
    nodes = [_n("a", "T1566.002"), _n("b", "T1059.001"), _n("c", "T1027")]
    rep = analyze_structure(nodes, [_e("a", "b")])
    assert rep.isolated == ["c"]
    kinds = [f["kind"] for f in rep.findings]
    assert kinds.count("isolated") == 1
    # an isolated singleton is NOT also reported as a disconnected component
    assert "disconnected_component" not in kinds

def test_detect_illegitimate_root():
    # T1027 (Defense Evasion, phase 6) as a proper root (in=0, out>0) is illegitimate
    nodes = [_n("a", "T1027"), _n("b", "T1041")]
    rep = analyze_structure(nodes, [_e("a", "b")])
    assert "a" in rep.roots
    illeg = [f for f in rep.findings if f["kind"] == "illegitimate_root"]
    assert len(illeg) == 1 and illeg[0]["node_id"] == "a"

def test_two_legit_phishing_roots_are_clean():
    # two independent Initial-Access roots converging — legitimate, no findings
    nodes = [_n("r1", "T1566.002"), _n("r2", "T1566.001"), _n("m", "T1059.001")]
    rep = analyze_structure(nodes, [_e("r1", "m"), _e("r2", "m")])
    assert set(rep.roots) == {"r1", "r2"}
    assert rep.findings == []

def test_detect_disconnected_components():
    nodes = [_n("a", "T1566.002"), _n("b", "T1059.001"),
             _n("c", "T1566.001"), _n("d", "T1059.005")]
    rep = analyze_structure(nodes, [_e("a", "b"), _e("c", "d")])  # two separate chains
    assert len(rep.components) == 2
    dc = [f for f in rep.findings if f["kind"] == "disconnected_component"]
    assert len(dc) == 1

def test_main_path_excludes_isolated_and_starts_early():
    nodes = [_n("a", "T1566.002", 0), _n("b", "T1059.001", 3), _n("c", "T1027", 9)]
    mp = CleanPipeline._main_path(nodes, [_e("a", "b")])
    assert mp[0] == "a"            # early Initial-Access root, not the isolated T1027
    assert "c" not in mp          # isolated node never on the main path

def test_main_path_orders_roots_by_tactic_phase():
    # late-tactic root (C2) and early root both present; main path must start early
    nodes = [_n("late", "T1071", 0), _n("early", "T1566.002", 5), _n("sink", "T1041", 9)]
    mp = CleanPipeline._main_path(nodes, [_e("late", "sink"), _e("early", "sink")])
    assert mp[0] == "early"

def test_apply_structure_stage_add_remove_replace():
    graph = {"nodes": [_n("a", "T1566.002"), _n("b", "T1059.001"), _n("c", "T1027")],
             "edges": [_e("a", "b")]}
    # add_edge reconnects the isolated node
    g2 = CleanPipeline._apply_structure_stage(graph, [
        {"action": "add_edge", "target": "b->c", "confidence": 0.7}])
    assert ("b", "c") in {(e["src"], e["dst"]) for e in g2["edges"]}
    # remove drops the node and its incident edges
    g3 = CleanPipeline._apply_structure_stage(g2, [{"action": "remove", "target": "c"}])
    assert "c" not in {n["node_id"] for n in g3["nodes"]}
    assert all(e["src"] != "c" and e["dst"] != "c" for e in g3["edges"])
    # replace retags using a technique id in the reason
    g4 = CleanPipeline._apply_structure_stage(
        graph, [{"action": "replace", "target": "c", "reason": "其实是 T1059"}])
    tech_c = {n["node_id"]: n["technique_id"] for n in g4["nodes"]}["c"]
    assert tech_c == "T1059"

def test_drop_residual_isolated_safety_net():
    dummy = SimpleNamespace(structure_drop_isolated=True)
    nodes = [_n("a", "T1566.002"), _n("b", "T1059.001"), _n("c", "T1027")]
    edges = [_e("a", "b")]
    kn, ke = CleanPipeline._drop_residual_isolated(dummy, nodes, edges)
    assert {n["node_id"] for n in kn} == {"a", "b"}     # isolated c dropped
    # never drops the sole node of a single-node graph
    kn1, _ = CleanPipeline._drop_residual_isolated(dummy, [_n("only", "T1027")], [])
    assert len(kn1) == 1
    # respects the off switch
    off = SimpleNamespace(structure_drop_isolated=False)
    kn2, _ = CleanPipeline._drop_residual_isolated(off, nodes, edges)
    assert len(kn2) == 3

def test_schema_validate_flags_global_structure():
    nodes = [GraphNode(node_id="a", mention="x", node_type="action", attack_id="T1566.002"),
             GraphNode(node_id="b", mention="y", node_type="action", attack_id="T1059.001"),
             GraphNode(node_id="c", mention="z", node_type="action", attack_id="T1027")]
    edges = [GraphEdge(src="a", dst="b", relation="enables")]
    doc = GraphDocument(doc_id="d", source_dataset="s", source_path="p", nodes=nodes, edges=edges)
    errs = " | ".join(doc.validate())
    assert "Isolated" in errs and "disconnected" in errs
    # a clean, connected graph yields no structural errors
    doc2 = GraphDocument(doc_id="d", source_dataset="s", source_path="p",
                         nodes=nodes[:2], edges=edges)
    assert not [x for x in doc2.validate() if "Isolated" in x or "disconnected" in x]

def test_on_shipped_output_graph():
    g = json.load(open(os.path.join(_PROJECT, "data", "casestudy", "output",
                                    "Cobalt_Kitty_GLANCE_output.json")))
    nodes = [{"node_id": n["node_id"], "technique_id": n.get("attack_id")} for n in g["nodes"]]
    edges = [{"src": e["src"], "dst": e["dst"]} for e in g["edges"]]
    rep = analyze_structure(nodes, edges)
    assert "step_3_add_T1027" in rep.isolated     # the real-world symptom is detected
    assert any(f["kind"] == "isolated" for f in rep.findings)

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
