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
    out = PROJECT / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    mem = PROJECT / "outputs" / "_mem"
    if mem.exists():
        shutil.rmtree(mem)
    shutil.copytree(PROJECT / "data" / "memory_seed", mem)

    cfg = load_config(memory_dir=str(mem), output_dir=str(out))
    pipe = CleanPipeline(cfg)

    casestudy = PROJECT / "data" / "casestudy"
    gold_files = sorted((casestudy / "gold").glob("*.json"))
    print("端到端 %d 篇 (data/casestudy: inputs/*.txt + gold/*.json) -> outputs/" % len(gold_files), flush=True)
    print("=" * 72, flush=True)
    rows = []
    for gf in gold_files:
        case = load_case(gf.stem, base=str(casestudy))
        gold = case["gold"]
        safe = case["doc_id"].replace(" ", "_").replace("/", "_")
        doc = StandardizedInputDocument(doc_id=case["doc_id"], source_dataset=gold.get("source_dataset"),
                                        source_path=gf.stem, text=case["text"], sentences=case["sentences"], ttp_candidates=[])
        pipe.run(doc)
        m = evaluate(load_graph(str(out / "graphs" / (safe + ".json"))), gold)
        row = {"doc": gold["doc_id"], "ttp": m.get("ttp_f1") or 0, "attack_step": m.get("attack_step_f1") or 0,
               "node_chain": m.get("node_chain_score") or 0, "workflow_graph": m.get("workflow_graph_score") or 0,
               "graph_triple": m.get("graph_triple_f1") or 0, "edge_parent": m.get("edge_parent_f1") or 0}
        rows.append(row)
        print("● %-36s ttp=%.3f astep=%.3f nchain=%.3f wgraph=%.3f gtriple=%.3f edge_par=%.3f" % (
            gold["doc_id"][:36], row["ttp"], row["attack_step"], row["node_chain"],
            row["workflow_graph"], row["graph_triple"], row["edge_parent"]), flush=True)
    ks = ["ttp", "attack_step", "node_chain", "workflow_graph", "graph_triple", "edge_parent"]
    avg = {k: sum(r[k] for r in rows) / len(rows) for k in ks}
    print("-" * 68, flush=True)
    print("均值(%d 篇): ttp=%.4f astep=%.4f nchain=%.4f wgraph=%.4f gtriple=%.4f edge_parent=%.4f" % (
        len(rows), avg["ttp"], avg["attack_step"], avg["node_chain"], avg["workflow_graph"],
        avg["graph_triple"], avg["edge_parent"]), flush=True)
    print("tokens=%s" % (pipe.llm.usage_stats,), flush=True)

if __name__ == "__main__":
    main()
