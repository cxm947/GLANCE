from __future__ import annotations
import sys, shutil
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "configs"))
import logging
logging.basicConfig(level=logging.WARNING, format="%(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
from pipeline_clean import CleanPipeline
from report_loader import load_case
from schema import StandardizedInputDocument
from evaluation import evaluate, load_graph
from loader import load_config

CASESTUDY = PROJECT / "data" / "casestudy"

GROUPS = {
    "c0_main": None,
    "c1_no_identifier": "c1_no_identifier",
    "c2_no_reasoner": "c2_no_reasoner",
    "c3_no_short_memory": "c3_no_short_memory",
    "c4_no_long_memory": "c4_no_long_memory",
    "c5_no_verify": "c5_no_verify",
    "c6_no_route_back": "c6_no_route_back",
    "c7_naive": "c7_naive",
    "c8_no_reasoner_memory": "c8_no_reasoner_memory",
    "c9_no_TI_VR": "c9_no_TI_VR",
    "c10_no_RR_VR": "c10_no_RR_VR",
    "c11_no_STM_LTM": "c11_no_STM_LTM",
}

class _MockLLM:
    def __init__(self):
        self.usage_stats = {"total_input_tokens": 0, "total_output_tokens": 0, "total_calls": 0}

    def chat(self, *a, **k):
        return {"nodes": [], "edges": [], "split": [], "attack_sentence_ids": [], "attack_sentences": [],
                "findings": [], "challenges": [], "verdicts": [], "decisions": [], "missing": [],
                "missing_edges": [], "sentences": [], "rewritten_sentences": [], "rewritten_report": "",
                "keep": [], "remove": [], "add": [], "retag": [], "result": [], "passed": True}

def _load_docs():
    docs = []
    for gf in sorted((CASESTUDY / "gold").glob("*.json")):
        case = load_case(gf.stem, base=str(CASESTUDY))
        gold = case["gold"]
        safe = case["doc_id"].replace(" ", "_").replace("/", "_")
        docs.append((StandardizedInputDocument(
            doc_id=case["doc_id"], source_dataset=gold.get("source_dataset"), source_path=gf.stem,
            text=case["text"], sentences=case["sentences"], ttp_candidates=[]), gold, safe))
    return docs

def run_group(name, experiment, docs, dry):
    import os as _os
    tag = _os.environ.get("ABL_TAG", "")
    base = PROJECT / "outputs" / "ablation" / tag if tag else PROJECT / "outputs" / "ablation"
    out = base / name
    out.mkdir(parents=True, exist_ok=True)
    mem = base / ("_mem_" + name)
    if mem.exists():
        shutil.rmtree(mem)
    shutil.copytree(PROJECT / "data" / "memory_seed", mem)
    cfg = load_config(experiment, memory_dir=str(mem), output_dir=str(out))
    pipe = CleanPipeline(cfg)
    if dry:
        mock = _MockLLM()
        pipe.llm = mock
        for ag in (pipe.identifier, pipe.reasoner, pipe.verifier):
            ag.llm = mock
    edge = []
    for doc, gold, safe in docs:
        pipe.run(doc)
        m = evaluate(load_graph(str(out / "graphs" / (safe + ".json"))), gold)
        edge.append(m.get("edge_parent_f1") or 0)
    return sum(edge) / len(edge) if edge else 0.0

def main():
    args = sys.argv[1:]
    dry = "--dry-run" in args
    names = [a for a in args if not a.startswith("--")] or list(GROUPS)
    unknown = [n for n in names if n not in GROUPS]
    if unknown:
        print("未知组 %s; 可用: %s" % (unknown, list(GROUPS)))
        sys.exit(1)
    docs = _load_docs()
    mode = "DRY-RUN (mock LLM, 不调 API, 只验证能跑通)" if dry else "真跑 (调 API)"
    print("消融 [%s]: %d 组 × %d 篇 (data/casestudy) -> outputs/ablation/" % (mode, len(names), len(docs)))
    print("=" * 68)
    nfail = 0
    for n in names:
        try:
            avg = run_group(n, GROUPS[n], docs, dry)
            print("  [OK]   %-24s %s" % (n, "能跑通" if dry else "edge_parent=%.4f" % avg))
        except Exception as e:
            nfail += 1
            print("  [FAIL] %-24s %r" % (n, e))
    print("-" * 68)
    print("%s: %d/%d 组通过%s" % ("DRY-RUN" if dry else "真跑", len(names) - nfail, len(names),
                                 "" if nfail == 0 else "  !! %d 组 FAIL" % nfail))

if __name__ == "__main__":
    main()
