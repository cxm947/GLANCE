"""Unit tests for the memory lifecycle (Phase 3): de-dup, confidence, provenance,
Memory Gate, bounded growth, and index/learned persistence. Runs against an isolated
temp copy of the seed so it never touches data/memory_seed. Zero extra deps.
"""
from __future__ import annotations
import json
import os
import shutil
import sys
import tempfile

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT, "src"))

from experience import Experience  # noqa: E402
from memory_engine import MemoryEngine, MAX_BLACKBOARD  # noqa: E402

def _fresh_mem():
    d = tempfile.mkdtemp(prefix="glance_mem_")
    shutil.copytree(os.path.join(_PROJECT, "data", "memory_seed"), d, dirs_exist_ok=True)
    return MemoryEngine(d), d

def _struct_exp(kind="isolated", tech="T1027", hint="x"):
    return Experience(id="e", scope="structural", polarity="flag",
                      trigger={"kinds": [kind], "techniques_src": [tech]},
                      action="drop_node", hint=hint, confidence=0.6)

def test_learn_experience_dedups_and_counts():
    mem, _ = _fresh_mem()
    e1 = mem.learn_experience(_struct_exp())
    assert e1.provenance["source"] == "learned" and e1.provenance["count"] == 1
    assert "ts" in e1.provenance
    e2 = mem.learn_experience(_struct_exp())  # same content -> dedup
    assert len(mem.learned_experiences) == 1
    assert e2.provenance["count"] == 2 and e2.confidence > 0.6  # confidence nudged up
    mem.learn_experience(_struct_exp(kind="illegitimate_root"))  # different -> new
    assert len(mem.learned_experiences) == 2

def test_memory_gate_promotion():
    mem, _ = _fresh_mem()
    mem.gate_min_count, mem.gate_min_confidence = 2, 0.6
    mem.learn_experience(_struct_exp())
    assert mem.promote_experiences() == []          # count 1 < gate
    mem.learn_experience(_struct_exp())              # -> count 2, conf bumped
    promoted = mem.promote_experiences()
    assert len(promoted) == 1 and promoted[0].provenance["count"] >= 2
    assert mem.promote_experiences(scope="edge") == []  # scope filter works

def test_error_record_dedup_and_count():
    mem, _ = _fresh_mem()
    n0 = len(mem.error_memory)
    for _ in range(3):
        mem.add_error_record("mismap", "T1059", ["powershell"], "T1071", "T1059.001",
                             "evidence says ps", "docA")
    assert len(mem.error_memory) == n0 + 1
    rec = mem.error_memory[-1]
    assert rec.count == 3 and rec.to_dict()["count"] == 3

def test_blackboard_is_bounded():
    mem, _ = _fresh_mem()
    for i in range(MAX_BLACKBOARD + 25):
        mem.append_blackboard("k", {"i": i}, source="t")
    assert len(mem.blackboard) == MAX_BLACKBOARD
    assert mem.blackboard[-1]["payload"]["i"] == MAX_BLACKBOARD + 24  # newest kept

def test_save_writes_index_and_learned_and_reloads():
    mem, d = _fresh_mem()
    mem.learn_experience(_struct_exp())
    mem.learn_experience(_struct_exp())  # count 2
    mem.save()
    idx_path = os.path.join(d, "experience_index.json")
    learned_path = os.path.join(d, "learned_experiences.json")
    assert os.path.exists(idx_path) and os.path.exists(learned_path)
    idx = json.load(open(idx_path))
    assert idx["n"] == len(idx["experiences"]) and idx["n"] > 100
    assert any(r.get("source") == "learned" for r in idx["experiences"])
    # reload: learned store + counts survive a fresh engine on the same dir
    mem2 = MemoryEngine(d)
    assert len(mem2.learned_experiences) == 1
    assert mem2.learned_experiences[0].provenance["count"] == 2

def test_promoted_learned_structural_feeds_retrieval():
    mem, _ = _fresh_mem()
    report = {"findings": [{"kind": "isolated", "node_id": "n", "technique_id": "T1027"}]}
    base = {x["id"] for x in mem.get_structural_experiences([], [], report)}
    mem.learn_experience(_struct_exp())
    mem.learn_experience(_struct_exp())  # promoted (count 2, conf>0.6)
    got = mem.get_structural_experiences([], [], report)
    assert any(x.get("source") == "learned" for x in got)
    assert {x["id"] for x in got} >= base  # seed experiences still present

def test_fresh_seed_has_no_learned_so_behaviour_unchanged():
    mem, _ = _fresh_mem()
    assert mem.learned_experiences == []
    report = {"findings": [{"kind": "isolated", "node_id": "n", "technique_id": "T1027"}]}
    got = mem.get_structural_experiences([], [], report)
    assert all(x.get("source") != "learned" for x in got)  # only seed on a fresh run

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
