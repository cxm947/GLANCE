from __future__ import annotations
import sys, shutil
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "configs"))
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
from pipeline_clean import CleanPipeline
from report_loader import load_case
from schema import StandardizedInputDocument
from evaluation import evaluate, load_graph
from loader import load_config

def main():
    if len(sys.argv) < 2:
        print("用法: run_one.py <gold_stem> [--keep-mem]"); sys.exit(1)
    stem = sys.argv[1]
    keep_mem = "--keep-mem" in sys.argv
    out = PROJECT / "outputs"; out.mkdir(parents=True, exist_ok=True)
    mem = out / "_mem"
    if not keep_mem or not mem.exists():
        if mem.exists():
            shutil.rmtree(mem)
        shutil.copytree(PROJECT / "data" / "memory_seed", mem)
    cfg = load_config(memory_dir=str(mem), output_dir=str(out))
    pipe = CleanPipeline(cfg)
    casestudy = PROJECT / "data" / "casestudy"
    gf = casestudy / "gold" / (stem + ".json")
    case = load_case(gf.stem, base=str(casestudy))
    gold = case["gold"]
    safe = case["doc_id"].replace(" ", "_").replace("/", "_")
    doc = StandardizedInputDocument(doc_id=case["doc_id"], source_dataset=gold.get("source_dataset"),
                                    source_path=gf.stem, text=case["text"], sentences=case["sentences"], ttp_candidates=[])
    print("单篇真跑: %s (doc_id=%s)" % (stem, case["doc_id"]), flush=True)
    print("=" * 72, flush=True)
    pipe.run(doc)
    m = evaluate(load_graph(str(out / "graphs" / (safe + ".json"))), gold)
    print("-" * 72, flush=True)
    print("RESULT %s" % case["doc_id"], flush=True)
    print("  edge_parent_f1=%.4f  P=%.4f  R=%.4f  | TP=%d FP=%d FN=%d" % (
        m["edge_parent_f1"], m["edge_parent_precision"], m["edge_parent_recall"],
        m["edge_parent_tp"], m["edge_parent_fp"], m["edge_parent_fn"]), flush=True)
    print("  ttp_f1=%.4f  node_f1=%.4f  edge_f1=%.4f" % (
        m["ttp_f1"], m["node_f1"], m.get("edge_f1", 0.0)), flush=True)
    print("  tokens=%s" % (pipe.llm.usage_stats,), flush=True)

if __name__ == "__main__":
    main()
