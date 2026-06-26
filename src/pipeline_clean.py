from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

from schema import StandardizedInputDocument, GraphDocument, GraphNode, GraphEdge, GraphPath
from llm_client import LLMClient
from memory_engine import MemoryEngine
from knowledge import get_tactic_id, get_tactic_name, get_phase_index
from graph_structure import analyze_structure
from prompt_loader import load_template
from agents.identifier_clean import CleanIdentifier
from agents.reasoner_clean import CleanReasoner
from agents.verifier_clean import CleanVerifier

logger = logging.getLogger(__name__)

VALID_RELATIONS = {"precedes", "enables", "causes", "uses", "targets", "phase_transition", "related"}

PAPER_ACTIONS = {"remove", "replace", "demote"}

class CleanPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.llm = LLMClient(
            api_key=config["api_key"],
            base_url=config.get("base_url", "https://api.deepseek.com/v1"),
            model=config.get("model", "deepseek-chat"),
            max_retries=config.get("max_retries", 3),
            retry_delay=config.get("retry_delay", 2.0),
            response_format_enabled=config.get("response_format_enabled", True),
            seed=config.get("seed"),
        )
        self.memory = MemoryEngine(config.get("memory_dir", "data/memory"))

        _m = (config.get("model") or "").lower()
        _send_think = config.get("send_thinking_param", ("qwen" in _m or ("glm" in _m and "0414" not in _m)))
        eb = {"enable_thinking": bool(config.get("thinking", False))} if _send_think else None

        _eff = config.get("reasoning_effort") or os.environ.get("EDL_REASONING_EFFORT")
        if _eff:
            eb = {**(eb or {}), "reasoning_effort": _eff}
        self._extra_body = eb
        self.identifier = CleanIdentifier(self.llm, temperature=config.get("identifier_temperature", 0.0), extra_body=eb,
                                          mode=config.get("identifier_mode", "full"),
                                          self_consistency=config.get("identifier_self_consistency", 1))
        self.reasoner = CleanReasoner(
            self.llm, temperature=config.get("reasoner_temperature", 0.0), extra_body=eb,
            self_consistency=config.get("reasoner_self_consistency", 3),
            require_anchor=config.get("reasoner_require_anchor", True),
            hub_priority=config.get("reasoner_hub_priority", True),
            reattach_hub=config.get("reasoner_reattach_hub", False),
            mode=config.get("reasoner_mode", "full"),
        )
        self.verifier = CleanVerifier(self.llm, temperature=config.get("verifier_temperature", 0.0), extra_body=eb,
                                      missing_k=config.get("verifier_missing_k", 3),
                                      missing_thresh=config.get("verifier_missing_thresh", 2))

        self.enable_verify = config.get("enable_verify", True)
        self.enable_short_memory = config.get("enable_short_memory", True)
        self.pipeline_mode = config.get("pipeline_mode", "full")
        self.enable_route_back = config.get("enable_route_back", True)

        self.route_back_mode = config.get("route_back_mode", "deterministic")

        self.route_back_conf_floor = float(config.get("route_back_conf_floor", 0.5))

        self.apply_missing_edges = bool(config.get("apply_missing_edges", True))
        self.missing_edge_conf_floor = float(config.get("missing_edge_conf_floor", 0.0))

        self.transitive_reduction = bool(config.get("transitive_reduction", True))
        self.enable_memory = config.get("enable_memory", True)

        # ④ graph-level structure repair (global perspective): detect + repair
        # isolated nodes / illegitimate roots / disconnected components.
        self.enable_structure_repair = bool(config.get("enable_structure_repair", True))
        _cutoff = config.get("root_late_tactic_cutoff", "TA0005")
        self.root_late_phase = (get_phase_index(_cutoff)
                                if isinstance(_cutoff, str) and _cutoff.upper().startswith("TA")
                                else int(_cutoff))
        self.structure_drop_isolated = bool(config.get("structure_drop_unsupported_isolated", True))

        # memory lifecycle: cross-report learning + Memory Gate thresholds
        self.enable_learning = bool(config.get("enable_learning", True))
        self.memory.gate_min_count = int(config.get("memory_gate_min_count", self.memory.gate_min_count))
        self.memory.gate_min_confidence = float(
            config.get("memory_gate_min_confidence", self.memory.gate_min_confidence))

        self.pos_examples = self._load_positive_examples(config.get("memory_dir", "data/memory"))

        self.output_dir = config.get("output_dir", "outputs")
        for sub in ["graphs", "agent_a_results", "agent_b_results", "verify_results", "traces",
                    "routeback_identifier", "routeback_reasoner"]:
            os.makedirs(os.path.join(self.output_dir, sub), exist_ok=True)

    @staticmethod
    def _load_positive_examples(memory_dir: str) -> list[dict]:

        nodes_path = os.path.join(memory_dir, "nodes.json")
        if os.path.exists(nodes_path):
            try:
                return (json.load(open(nodes_path, encoding="utf-8")) or {}).get("positive", []) or []
            except Exception as exc:
                logger.warning("nodes.json positive load failed: %s", exc)
                return []
        mem_path = os.path.join(memory_dir, "memory.json")
        if os.path.exists(mem_path):
            try:
                m = json.load(open(mem_path, encoding="utf-8")) or {}
                return (m.get("node") or {}).get("positive", []) or []
            except Exception as exc:
                logger.warning("memory.json positive load failed: %s", exc)
                return []
        path = os.path.join(memory_dir, "positive_examples.json")
        if not os.path.exists(path):
            return []
        try:
            data = json.load(open(path, encoding="utf-8"))
            return data.get("examples", []) if isinstance(data, dict) else (data or [])
        except Exception as exc:
            logger.warning("positive_examples load failed: %s", exc)
            return []

    def _neg_examples(self, evidence_text: str) -> list[dict]:
        if not self.enable_memory:
            return []
        try:
            return self.memory.get_generic_error_warnings(top_n=12, evidence_text=evidence_text)
        except TypeError:
            return self.memory.get_generic_error_warnings(top_n=12)

    def _transition_priors(self, technique_ids: list[str], evidence_text: str) -> dict:
        if not self.enable_memory:
            return {}
        return self.memory.get_all_priors(technique_ids, evidence_text=evidence_text)

    def _missing_patterns(self, technique_ids: list[str], action_text: str = "") -> list[dict]:
        if not self.enable_memory:
            return []
        try:
            return self.memory.get_missing_edge_patterns(technique_ids, action_text)
        except Exception:
            return []

    def run(self, doc: StandardizedInputDocument) -> GraphDocument:
        t0 = time.time()
        doc_id = doc.doc_id
        safe = doc_id.replace(" ", "_").replace("/", "_")
        sentences = list(doc.sentences or [])
        full_text = doc.text or " ".join(sentences)
        clear = getattr(self.llm, "clear_agent_call_log", None)
        if callable(clear):
            clear()
        logger.info("=" * 60)
        logger.info("CLEAN Pipeline START: %s (%d sentences)", doc_id, len(sentences))

        neg = self._neg_examples(full_text)

        if self.pipeline_mode == "naive":
            return self._run_naive(doc, doc_id, safe, sentences, full_text, t0)

        find_hints = [h for h in self.memory.get_node_memory([], "", full_text) if h.get("find_node")]\
            if self.enable_memory else []
        nodes = self.identifier.identify(doc_id, sentences, full_text, neg, self.pos_examples, find_hints=find_hints)
        nodes = self._backfill_nodes(nodes, sentences)
        if not self.enable_short_memory:

            for n in nodes:
                n["procedure"] = {"actor": "", "action": "", "object": "", "purpose": ""}
                n["evidence_sentence_ids"] = []
                n["evidence_text"] = ""
        self._save(os.path.join(self.output_dir, "agent_a_results", f"{safe}.json"),
                   {"doc_id": doc_id, "nodes": nodes})
        logger.info("  ① identify: %d explicit nodes", len(nodes))

        tech_ids = sorted({n["technique_id"] for n in nodes if n.get("technique_id")})
        if self.config.get("reasoner_no_memory"):
            priors, exp_rules, imp_rules, r_neg = {}, [], [], []
        else:
            priors = self._transition_priors(tech_ids, full_text)

            exp_rules = (self.memory.get_explicit_rules(nodes)
                         if (self.enable_memory and self.config.get("reasoner_inject_pos_rules", False)) else [])
            imp_rules = self.memory.get_implicit_rules(nodes) if self.enable_memory else []
            r_neg = neg
        graph = self.reasoner.reason(doc_id, sentences, nodes, priors, r_neg,
                                     explicit_rules=exp_rules, implicit_rules=imp_rules)
        graph = self._backfill_graph(graph, sentences)
        self._save(os.path.join(self.output_dir, "agent_b_results", f"{safe}.json"),
                   {"doc_id": doc_id, "num_nodes": len(graph.get("nodes", [])),
                    "num_edges": len(graph.get("edges", [])), "graph": graph})
        logger.info("  ② reason: %d nodes, %d edges (%d explicit / %d implicit)",
                    len(graph.get("nodes", [])), len(graph.get("edges", [])),
                    sum(1 for e in graph.get("edges", []) if e.get("evidence_type") == "explicit"),
                    sum(1 for e in graph.get("edges", []) if e.get("evidence_type") == "implicit"))

        verify_record: dict[str, Any] = {"enabled": self.enable_verify}
        if self.enable_verify:
            def _act_text(gn):
                return " ".join("%s %s" % ((n.get("procedure") or {}).get("action", ""),
                                           (n.get("procedure") or {}).get("object", "")) for n in gn)

            rep, rwsents = self.verifier.rewrite(
                doc_id, graph.get("nodes", []), graph.get("edges", []), graph.get("main_path", []))
            all_findings: list[dict] = []

            gnodes, gedges = graph.get("nodes", []), graph.get("edges", [])
            node_mem = self.memory.get_node_memory(
                [n.get("technique_id") for n in gnodes], _act_text(gnodes), full_text) if self.enable_memory else []
            node_findings = self.verifier.check_nodes(doc_id, sentences, gnodes, gedges, rwsents, node_mem)
            logger.info("  ③a 节点核对: %d finding(s)", len(node_findings))
            all_findings += node_findings
            if self.enable_route_back:
                graph = self._adjudicate_nodes_stage(doc_id, sentences, graph, node_findings, neg, safe)
                graph = self._backfill_graph(graph, sentences)

            gnodes, gedges = graph.get("nodes", []), graph.get("edges", [])
            em_b = self.memory.get_edge_memory(gnodes, gedges) if self.enable_memory else {"wrong": [], "missing": []}
            wrong_mem = em_b.get("wrong", [])
            edge_sigs = self.memory.get_edge_signals(gnodes, gedges) if self.enable_memory else {}
            fanout_alerts = self.memory.detect_fanout_chains(gnodes, gedges) if self.enable_memory else []
            edge_findings = self.verifier.check_edges(doc_id, sentences, gnodes, gedges, rwsents, wrong_mem, edge_sigs, fanout_alerts, em_b.get("review", []))
            logger.info("  ③b 边去伪: %d finding(s) (扇出%d+冲突提醒%d)", len(edge_findings), len(fanout_alerts), len(em_b.get("review", [])))
            all_findings += edge_findings
            if self.enable_route_back:
                graph = self._adjudicate_edges_stage(doc_id, sentences, graph, edge_findings, neg, priors, safe)
                graph = self._backfill_graph(graph, sentences)

            gnodes, gedges = graph.get("nodes", []), graph.get("edges", [])
            em_c = self.memory.get_edge_memory(gnodes, gedges) if self.enable_memory else {"wrong": [], "missing": []}
            missing_mem = em_c.get("missing", [])
            missing_findings = self.verifier.find_missing(
                doc_id, sentences, gnodes, gedges, missing_mem, [], rwsents, edge_rules=[])
            logger.info("  ③c 找漏: %d finding(s)", len(missing_findings))
            all_findings += missing_findings
            if self.enable_route_back and self.apply_missing_edges:
                graph = self._apply_missing_stage(graph, missing_findings, sentences, edge_rules=missing_mem)
                graph = self._backfill_graph(graph, sentences)

            if self.enable_route_back and self.enable_structure_repair:
                graph = self._structure_repair_stage(doc_id, sentences, graph, rwsents, safe)
                graph = self._backfill_graph(graph, sentences)

            verify_record.update({
                "passed": len(all_findings) == 0, "findings": all_findings,
                "rewritten_report": rep, "rewritten_sentences": rwsents,
                "route_back_applied": bool(all_findings), "route_back_mode": "serial",
            })
        self._save(os.path.join(self.output_dir, "verify_results", f"{safe}.json"),
                   {"doc_id": doc_id, **verify_record})

        graph = self._correct_misattribution(graph)

        graph = self._clean_normalize(graph)
        graph_doc = self._assemble(doc, graph)
        graph_doc.to_json(os.path.join(self.output_dir, "graphs", f"{safe}.json"))

        if self.enable_memory:
            self.memory.save()

        trace = {
            "doc_id": doc_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": {k: v for k, v in self.config.items() if k != "api_key"},
            "tokens": self.llm.usage_stats,
            "stats": {"nodes": len(graph_doc.nodes), "edges": len(graph_doc.edges), "paths": len(graph_doc.paths)},
            "duration_s": round(time.time() - t0, 2),
        }
        self._save(os.path.join(self.output_dir, "traces", f"{safe}_trace.json"), trace)
        logger.info("CLEAN Pipeline DONE: %s (%.1fs, %d nodes, %d edges)",
                    doc_id, time.time() - t0, len(graph_doc.nodes), len(graph_doc.edges))
        logger.info("=" * 60)
        return graph_doc

    def verify_only(self, doc_id, safe, sentences, full_text, graph):
        neg = self._neg_examples(full_text)
        tech_ids = sorted({par for n in graph.get("nodes", []) for par in [n.get("technique_id")] if par})
        priors = self._transition_priors(tech_ids, full_text) if self.enable_memory else {}

        def _act_text(gn):
            return " ".join("%s %s" % ((n.get("procedure") or {}).get("action", ""),
                                       (n.get("procedure") or {}).get("object", "")) for n in gn)
        rep, rwsents = self.verifier.rewrite(doc_id, graph.get("nodes", []), graph.get("edges", []),
                                             graph.get("main_path", []))
        all_findings = []

        gnodes, gedges = graph.get("nodes", []), graph.get("edges", [])
        node_mem = self.memory.get_node_memory([n.get("technique_id") for n in gnodes], _act_text(gnodes), full_text) if self.enable_memory else []
        nf = self.verifier.check_nodes(doc_id, sentences, gnodes, gedges, rwsents, node_mem)
        all_findings += nf
        logger.info("  ③a 节点核对: %d finding(s)", len(nf))
        graph = self._adjudicate_nodes_stage(doc_id, sentences, graph, nf, neg, safe)
        graph = self._backfill_graph(graph, sentences)

        gnodes, gedges = graph.get("nodes", []), graph.get("edges", [])
        em_b = self.memory.get_edge_memory(gnodes, gedges) if self.enable_memory else {"wrong": [], "missing": []}
        edge_sigs = self.memory.get_edge_signals(gnodes, gedges) if self.enable_memory else {}
        fanout = self.memory.detect_fanout_chains(gnodes, gedges) if self.enable_memory else []
        ef = self.verifier.check_edges(doc_id, sentences, gnodes, gedges, rwsents, em_b.get("wrong", []), edge_sigs, fanout, em_b.get("review", []))
        all_findings += ef
        logger.info("  ③b 边去伪: %d finding(s) (扇出提醒%d)", len(ef), len(fanout))
        graph = self._adjudicate_edges_stage(doc_id, sentences, graph, ef, neg, priors, safe)
        graph = self._backfill_graph(graph, sentences)

        gnodes, gedges = graph.get("nodes", []), graph.get("edges", [])
        em_c = self.memory.get_edge_memory(gnodes, gedges) if self.enable_memory else {"wrong": [], "missing": []}
        mf = self.verifier.find_missing(doc_id, sentences, gnodes, gedges, em_c.get("missing", []), [], rwsents, edge_rules=[])
        all_findings += mf
        logger.info("  ③c 找漏: %d finding(s)", len(mf))
        if self.apply_missing_edges:
            graph = self._apply_missing_stage(graph, mf, sentences, edge_rules=em_c.get("missing", []))
            graph = self._backfill_graph(graph, sentences)
        self._save(os.path.join(self.output_dir, "verify_results", f"{safe}.json"),
                   {"doc_id": doc_id, "findings": all_findings})
        graph = self._clean_normalize(graph)
        self._save(os.path.join(self.output_dir, "graphs", f"{safe}.json"), graph)
        return graph

    def _route_back(self, doc_id, sentences, full_text, nodes, graph, findings, neg, priors) -> dict:
        return self._apply_deletions(graph, findings)

    @staticmethod
    def _apply_deletions(graph: dict, findings: list[dict]) -> dict:
        nodes = list(graph.get("nodes", []) or [])
        edges = list(graph.get("edges", []) or [])
        node_by_id = {str(n.get("node_id")): n for n in nodes}
        edge_set = {(str(e.get("src")), str(e.get("dst"))) for e in edges}

        def parse_edge(tgt):
            s = str(tgt or "")
            if "->" not in s:
                return None
            a, b = s.split("->", 1)
            return a.strip(), b.strip()

        del_edges: set[tuple[str, str]] = set()
        rev_edges: set[tuple[str, str]] = set()
        cand_del_nodes: set[str] = set()
        for f in findings:
            action = str(f.get("action") or "").strip().lower()
            tgt = f.get("target")
            if action == "delete_edge":
                pe = parse_edge(tgt)
                if pe and pe in edge_set:
                    del_edges.add(pe)
            elif action == "reverse_edge":
                pe = parse_edge(tgt)
                if pe and pe in edge_set:
                    rev_edges.add(pe)
            elif action == "delete_node":
                nid = str(tgt or "").strip()
                if nid in node_by_id:
                    cand_del_nodes.add(nid)

        del_nodes: set[str] = set()
        for nid in cand_del_nodes:
            n = node_by_id.get(nid) or {}
            sids = list(n.get("evidence_sentence_ids") or [])
            reused = bool(sids) and all(
                any(str(o.get("node_id")) != nid and s in (o.get("evidence_sentence_ids") or [])
                    for o in nodes)
                for s in sids
            )
            proc = n.get("procedure") or {}
            act = str((proc.get("action") if isinstance(proc, dict) else "") or "").strip().lower()
            dup_or_empty = (not act) or any(
                str(o.get("node_id")) != nid
                and isinstance(o.get("procedure"), dict)
                and str((o.get("procedure") or {}).get("action") or "").strip().lower() == act
                for o in nodes
            )
            if reused and dup_or_empty:
                del_nodes.add(nid)

        new_edges: list[dict] = []
        for e in edges:
            key = (str(e.get("src")), str(e.get("dst")))
            if key in del_edges:
                continue
            if key[0] in del_nodes or key[1] in del_nodes:
                continue
            if key in rev_edges:
                e = dict(e)
                e["src"], e["dst"] = key[1], key[0]
            new_edges.append(e)
        new_nodes = [n for n in nodes if str(n.get("node_id")) not in del_nodes]

        if del_edges or rev_edges or del_nodes:
            logger.info("    route-back/incremental: -%d edge, ~%d reverse, -%d node",
                        len(del_edges), len(rev_edges), len(del_nodes))
        graph = dict(graph)
        graph["nodes"] = new_nodes
        graph["edges"] = new_edges
        graph.setdefault("main_path", graph.get("main_path", []))
        return graph

    def _route_back_agents(self, doc_id, sentences, full_text, nodes, graph,
                           findings, neg, priors, safe) -> dict:
        paper = [f for f in findings
                 if str(f.get("action") or "").strip().lower() in PAPER_ACTIONS]
        id_findings = [f for f in paper if f.get("route_to") == "identifier"]
        re_findings = [f for f in paper if f.get("route_to") == "reasoner"]
        id_fb = self._findings_to_identifier_feedback(id_findings)
        re_fb = self._findings_to_reasoner_feedback(re_findings)
        logger.info("    route-back/agents: %d->identifier (%d fb), %d->reasoner (%d fb)",
                    len(id_findings), len(id_fb), len(re_findings), len(re_fb))

        id_record: dict[str, Any] = {
            "doc_id": doc_id, "routed_to": "identifier", "actions": "remove/replace/demote",
            "n_findings": len(id_findings), "findings": id_findings, "feedback": id_fb,
            "ran": False, "nodes_before": len(nodes),
        }
        nodes2 = nodes
        if id_fb:
            try:
                relabel = self.identifier.identify(
                    doc_id, sentences, full_text, neg, self.pos_examples, feedback=id_fb)
                nodes2 = self._backfill_nodes(relabel, sentences)
                id_record.update({"ran": True, "nodes_after": len(nodes2), "nodes": nodes2})
            except Exception as exc:
                logger.warning("[route-back] identifier re-run failed for %s: %s", doc_id, exc)
                id_record["error"] = str(exc)
        self._save(os.path.join(self.output_dir, "routeback_identifier", f"{safe}.json"), id_record)

        nodes_changed = nodes2 is not nodes
        re_record: dict[str, Any] = {
            "doc_id": doc_id, "routed_to": "reasoner", "actions": "remove/replace/demote",
            "n_findings": len(re_findings), "findings": re_findings, "feedback": re_fb,
            "nodes_changed_by_identifier": nodes_changed, "ran": False,
            "edges_before": len(graph.get("edges", []) or []),
        }
        graph2 = graph
        if re_fb or nodes_changed:
            try:
                tech_ids = sorted({n["technique_id"] for n in nodes2 if n.get("technique_id")})
                priors2 = self._transition_priors(tech_ids, full_text) if nodes_changed else priors
                exp_r2 = self.memory.get_explicit_rules(nodes2) if self.enable_memory else []
                imp_r2 = self.memory.get_implicit_rules(nodes2) if self.enable_memory else []
                reasoned = self.reasoner.reason(doc_id, sentences, nodes2, priors2, neg,
                                                feedback=re_fb, explicit_rules=exp_r2, implicit_rules=imp_r2)
                graph2 = self._backfill_graph(reasoned, sentences)
                re_record.update({
                    "ran": True, "edges_after": len(graph2.get("edges", []) or []),
                    "num_nodes": len(graph2.get("nodes", [])), "graph": graph2,
                })
            except Exception as exc:
                logger.warning("[route-back] reasoner re-run failed for %s: %s", doc_id, exc)
                re_record["error"] = str(exc)
                graph2 = graph
        self._save(os.path.join(self.output_dir, "routeback_reasoner", f"{safe}.json"), re_record)
        return graph2

    @staticmethod
    def _findings_to_identifier_feedback(findings: list[dict]) -> list[dict]:
        verb = {"remove": "删除该节点(原文无支持/幻觉, 重标时不要再产出它)",
                "replace": "改正该节点的技术标注(证据句对应的其实是别的技术)",
                "demote": "该节点缺乏原文支持, 重标时谨慎或移除"}
        out: list[dict] = []
        for f in findings:
            span = f.get("evidence_span") or {}
            sid = span.get("sentence_id")
            if not isinstance(sid, int):
                try:
                    sid = int(sid)
                except (TypeError, ValueError):
                    continue
            action = str(f.get("action") or "").strip().lower()
            tgt = str(f.get("target") or "")
            reason = str(f.get("reason") or "")
            out.append({"sentence_id": sid,
                        "issue": f"[{action}] {verb.get(action, action)} (target={tgt}); 依据: {reason}"})
        return out

    @staticmethod
    def _findings_to_reasoner_feedback(findings: list[dict]) -> list[dict]:
        verb = {"remove": "删除该边(原文无此先后/因果依据, 乱连)",
                "replace": "改正该边方向(原文方向与之相反)",
                "demote": "降级/移除该隐式边或隐式节点(攻击逻辑上讲不通或缺前置)"}
        out: list[dict] = []
        for f in findings:
            action = str(f.get("action") or "").strip().lower()
            tgt = str(f.get("target") or "")
            reason = str(f.get("reason") or "")
            out.append({"target": tgt,
                        "issue": f"[{action}] {verb.get(action, action)}; 依据: {reason}"})
        return out

    def _route_back_adjudicate(self, doc_id, sentences, full_text, nodes, graph,
                               findings, neg, priors, safe) -> dict:
        floor = self.route_back_conf_floor
        paper = [f for f in findings
                 if str(f.get("action") or "").strip().lower() in PAPER_ACTIONS]
        kept = [f for f in paper if self._conf(f) >= floor]
        dropped = len(paper) - len(kept)
        id_ch = [f for f in kept if f.get("route_to") == "identifier"]
        edge_ch = [f for f in kept if f.get("route_to") == "reasoner"]
        logger.info("    route-back/adjudicate: %d node-challenge, %d edge-challenge "
                    "(conf>=%.2f, %d paper-finding(s) below floor dropped)",
                    len(id_ch), len(edge_ch), floor, dropped)

        node_verdicts: list[dict] = []
        if id_ch:
            node_verdicts = self.identifier.adjudicate_nodes(
                doc_id, sentences, nodes, id_ch, neg, self.pos_examples)
        self._save(os.path.join(self.output_dir, "routeback_identifier", f"{safe}.json"), {
            "doc_id": doc_id, "routed_to": "identifier", "mode": "adjudicate",
            "conf_floor": floor, "n_challenges": len(id_ch), "challenges": id_ch,
            "verdicts": node_verdicts, "verdict_counts": self._vcount(node_verdicts),
            "nodes_before": len(nodes),
        })

        gnodes = graph.get("nodes", []) or []
        gedges = graph.get("edges", []) or []
        edge_sigs = self.memory.get_edge_signals(gnodes, gedges) if self.enable_memory else {}
        auto_remove, to_adjudicate = [], []
        for f in edge_ch:
            tgt = str(f.get("target") or "")
            sig = edge_sigs.get(tgt)
            if sig and sig.get("tier") == "特别错误" and str(f.get("action")).lower() in ("remove", "demote"):
                auto_remove.append(tgt)
            else:
                to_adjudicate.append(f)
        edge_verdicts: list[dict] = []
        if to_adjudicate:
            edge_verdicts = self.reasoner.adjudicate_edges(
                doc_id, sentences, graph, to_adjudicate, priors, neg, edge_signals=edge_sigs)
        for tgt in auto_remove:
            edge_verdicts.append({"edge": tgt, "verdict": "remove", "reason": "特别错误档(记忆∩矩阵低)确定性删"})
        if auto_remove or to_adjudicate:
            logger.info("    route-back/闸: %d 特别错误确定删 + %d 送裁决(4信号)", len(auto_remove), len(to_adjudicate))
        graph2 = self._apply_adjudication(graph, node_verdicts, edge_verdicts,
                                          protect_main=getattr(self.reasoner, "recall_pass", False))
        self._save(os.path.join(self.output_dir, "routeback_reasoner", f"{safe}.json"), {
            "doc_id": doc_id, "routed_to": "reasoner", "mode": "adjudicate",
            "conf_floor": floor, "n_challenges": len(edge_ch), "challenges": edge_ch,
            "verdicts": edge_verdicts, "verdict_counts": self._vcount(edge_verdicts),
            "edges_before": len(graph.get("edges", []) or []),
            "edges_after": len(graph2.get("edges", []) or []),
            "nodes_after": len(graph2.get("nodes", []) or []),
            "graph": graph2,
        })
        return graph2

    def _adjudicate_nodes_stage(self, doc_id, sentences, graph, node_findings, neg, safe) -> dict:
        floor = self.route_back_conf_floor
        kept = [f for f in node_findings if self._conf(f) >= floor]
        nodes = graph.get("nodes", []) or []
        node_verdicts: list[dict] = []
        if kept:
            node_verdicts = self.identifier.adjudicate_nodes(
                doc_id, sentences, nodes, kept, neg, self.pos_examples)
        graph2 = self._apply_adjudication(graph, node_verdicts, [], sentences=sentences)
        n_add = sum(1 for v in node_verdicts if v.get("verdict") == "add")
        if kept:
            logger.info("    节点裁决: %d challenge -> %s%s", len(kept), self._vcount(node_verdicts),
                        (" 新增%d" % n_add) if n_add else "")
        self._save(os.path.join(self.output_dir, "routeback_identifier", f"{safe}.json"), {
            "doc_id": doc_id, "routed_to": "identifier", "mode": "node_stage",
            "conf_floor": floor, "n_challenges": len(kept), "challenges": kept,
            "verdicts": node_verdicts, "verdict_counts": self._vcount(node_verdicts),
            "nodes_before": len(nodes), "nodes_after": len(graph2.get("nodes", []) or []),
        })
        return graph2

    def _adjudicate_edges_stage(self, doc_id, sentences, graph, edge_findings, neg, priors, safe) -> dict:
        floor = self.route_back_conf_floor
        kept = [f for f in edge_findings if self._conf(f) >= floor
                and str(f.get("action") or "").lower() in PAPER_ACTIONS]
        gnodes, gedges = graph.get("nodes", []) or [], graph.get("edges", []) or []
        edge_sigs = self.memory.get_edge_signals(gnodes, gedges) if self.enable_memory else {}
        auto_remove, to_adjudicate = [], []
        force_adj = os.environ.get("EDGE_FORCE_ADJ") == "1"
        for f in kept:
            tgt = str(f.get("target") or "")
            sig = edge_sigs.get(tgt)
            if (not force_adj) and sig and sig.get("tier") == "特别错误" and str(f.get("action")).lower() in ("remove", "demote"):
                auto_remove.append(tgt)
            else:
                to_adjudicate.append(f)
        edge_verdicts: list[dict] = []
        if to_adjudicate:
            edge_verdicts = self.reasoner.adjudicate_edges(
                doc_id, sentences, graph, to_adjudicate, priors, neg, edge_signals=edge_sigs)
        for tgt in auto_remove:
            edge_verdicts.append({"edge": tgt, "verdict": "remove", "reason": "特别错误档(记忆∩矩阵低)确定删"})
        if auto_remove or to_adjudicate:
            logger.info("    边裁决/闸: %d 特别错误确定删 + %d 送裁决(4信号)", len(auto_remove), len(to_adjudicate))
        graph2 = self._apply_adjudication(graph, [], edge_verdicts,
                                          protect_main=getattr(self.reasoner, "recall_pass", False))
        self._save(os.path.join(self.output_dir, "routeback_reasoner", f"{safe}.json"), {
            "doc_id": doc_id, "routed_to": "reasoner", "mode": "edge_stage",
            "conf_floor": floor, "n_challenges": len(kept), "challenges": kept,
            "verdicts": edge_verdicts, "verdict_counts": self._vcount(edge_verdicts),
            "edges_before": len(gedges), "edges_after": len(graph2.get("edges", []) or []),
            "graph": graph2,
        })
        return graph2

    def _apply_missing_stage(self, graph: dict, missing_findings: list[dict], sentences: list[str],
                             edge_rules: list[dict] | None = None) -> dict:
        added = []
        for r in (edge_rules or []):
            for (a, b) in r.get("hit_edges", []):
                added.append({"action": "add_edge", "target": "%s->%s" % (a, b),
                              "confidence": 0.9, "from_missing": True})
        if not added:
            return graph
        logger.info("    找漏/边记忆补: %d 条(obj_link承接命中)", len(added))
        return self._apply_add_edges(graph, added, self.missing_edge_conf_floor)

    def _structure_repair_stage(self, doc_id, sentences, graph, rwsents, safe) -> dict:
        """③d Agent C graph-level repair: detect global structural anomalies and
        repair each in place (reconnect / drop / retag) under memory guidance,
        freezing the rest of the graph."""
        gnodes = graph.get("nodes", []) or []
        gedges = graph.get("edges", []) or []
        report = analyze_structure(gnodes, gedges, root_late_phase=self.root_late_phase)
        if not report.has_anomalies():
            return graph
        kc = {k: sum(1 for f in report.findings if f["kind"] == k)
              for k in ("isolated", "illegitimate_root", "disconnected_component")}
        logger.info("  ③d 图级结构: %d 异常 (孤立%d/非法起点%d/断裂%d)",
                    len(report.findings), kc["isolated"], kc["illegitimate_root"],
                    kc["disconnected_component"])
        struct_exps = (self.memory.get_structural_experiences(gnodes, gedges, report)
                       if self.enable_memory else [])
        findings = self.verifier.check_structure(doc_id, sentences, gnodes, gedges,
                                                 rwsents, report, struct_exps)
        graph2 = self._apply_structure_stage(graph, findings)
        if self.enable_learning and self.enable_memory:
            self._learn_from_structure(doc_id, report, findings)
        self._save(os.path.join(self.output_dir, "verify_results", f"{safe}_structure.json"), {
            "doc_id": doc_id, "report": report.to_dict(),
            "experiences": struct_exps, "findings": findings,
            "nodes_before": len(gnodes), "edges_before": len(gedges),
            "nodes_after": len(graph2.get("nodes", []) or []),
            "edges_after": len(graph2.get("edges", []) or []),
        })
        return graph2

    def _learn_from_structure(self, doc_id: str, report, findings: list[dict]) -> None:
        """Sediment each structural correction into cross-report learned memory
        (deduped + gated). Write-only — never changes the current run's output; it
        surfaces as a soft prior on later --keep-mem runs once the Memory Gate clears."""
        import re
        from experience import Experience
        by_node = {f.get("node_id"): f for f in report.findings}
        for f in findings or []:
            action = str(f.get("action") or "")
            if action not in ("remove", "replace"):
                continue
            rep_f = by_node.get(str(f.get("target") or "")) or {}
            tech = str(rep_f.get("technique_id") or "")
            kind = str(rep_f.get("kind") or "")
            m = re.search(r"T\d{4}", tech)
            par = m.group(0) if m else "x"
            try:
                conf = float(f.get("confidence"))
            except (TypeError, ValueError):
                conf = 0.6
            self.memory.learn_experience(Experience(
                id="learned_struct_%s_%s" % (kind or "x", par),
                scope="structural", polarity="flag",
                trigger={"kinds": [kind] if kind else [], "techniques_src": [tech] if tech else []},
                action="drop_node" if action == "remove" else "remap",
                hint="结构修复经验: 技术 %s 在结构异常(%s)下被%s" % (
                    tech or "?", kind or "?", "删除" if action == "remove" else "重映射"),
                confidence=conf,
                provenance={"source": "learned", "doc_id": doc_id},
            ))

    @staticmethod
    def _apply_structure_stage(graph: dict, findings: list[dict]) -> dict:
        import re
        nodes = list(graph.get("nodes") or [])
        edges = list(graph.get("edges") or [])
        node_by_id = {str(n.get("node_id")): n for n in nodes}
        valid = set(node_by_id)
        present = {(str(e.get("src")), str(e.get("dst"))) for e in edges}
        del_nodes: set[str] = set()
        retag: dict[str, str] = {}
        added = 0
        for f in findings or []:
            action = str(f.get("action") or "")
            tgt = str(f.get("target") or "")
            if action == "add_edge" and "->" in tgt:
                s, d = (x.strip() for x in tgt.split("->", 1))
                if s in valid and d in valid and s != d and (s, d) not in present:
                    try:
                        conf = float(f.get("confidence"))
                    except (TypeError, ValueError):
                        conf = 0.6
                    edges.append({"src": s, "dst": d, "relation": str(f.get("relation") or "enables"),
                                  "evidence_type": "implicit", "from_structure": True, "confidence": conf})
                    present.add((s, d))
                    added += 1
            elif action == "remove" and tgt in node_by_id:
                del_nodes.add(tgt)
            elif action == "replace" and tgt in node_by_id:
                m = re.search(r"T\d{4}(?:\.\d{3})?", str(f.get("reason") or ""))
                if m:
                    retag[tgt] = m.group(0)
        for nid, tid in retag.items():
            node_by_id[nid]["technique_id"] = tid
            node_by_id[nid]["technique_name"] = ""
        new_nodes = [n for n in nodes if str(n.get("node_id")) not in del_nodes]
        new_edges = [e for e in edges
                     if str(e.get("src")) not in del_nodes and str(e.get("dst")) not in del_nodes]
        if added or del_nodes or retag:
            logger.info("    结构修复应用: +%d 边, -%d 节点, ~%d 重映射",
                        added, len(del_nodes), len(retag))
        g = dict(graph)
        g["nodes"] = new_nodes
        g["edges"] = new_edges
        g.setdefault("main_path", graph.get("main_path", []))
        return g

    @staticmethod
    def _apply_adjudication(graph: dict, node_verdicts: list[dict], edge_verdicts: list[dict],
                            protect_main: bool = False, sentences: list[str] | None = None) -> dict:
        import re

        _OBJ_STOP = {"the", "a", "an", "of", "to", "and", "with", "its", "this", "that", "for", "on", "in",
                     "using", "used", "via", "from", "as", "it", "was", "were", "be", "by",
                     "powershell", "script", "scripts", "command", "commands", "file", "files", "code",
                     "payload", "payloads", "data", "program", "process", "tool", "component", "components",
                     "脚本", "命令", "文件", "载荷", "程序", "进程", "工具", "组件", "使用", "一个", "这个", "该"}

        def _obj_core(obj):
            ws = re.findall(r"[a-z0-9][a-z0-9.\-]*|[一-鿿]+", str(obj or "").lower())
            return {w for w in ws if w not in _OBJ_STOP and len(w) > 1}
        nodes = list(graph.get("nodes") or [])
        edges = list(graph.get("edges") or [])
        sentences = sentences or []
        node_by_id = {str(n.get("node_id")): n for n in nodes}
        nid2sid = {str(n.get("node_id")): (n.get("evidence_sentence_ids") or [None])[0] for n in nodes}
        nid2obj = {str(n.get("node_id")): ((n.get("procedure") or {}).get("object", "")) for n in nodes}

        sids_sorted = sorted({(n.get("evidence_sentence_ids") or [10**9])[0] for n in nodes
                              if isinstance((n.get("evidence_sentence_ids") or [None])[0], int)})

        def _step_for(sid):
            if not isinstance(sid, int):
                return len(nodes)
            k = 0
            for s in sids_sorted:
                if s <= sid:
                    k += 1
            return k
        del_nodes: set[str] = set()
        add_nodes: list[dict] = []
        for v in node_verdicts or []:
            verdict = v.get("verdict")
            if verdict == "add":
                tid = str(v.get("technique_id") or "")
                sid = v.get("sentence_id")
                if not tid:
                    continue
                ev_txt = sentences[sid] if isinstance(sid, int) and 0 <= sid < len(sentences) else ""
                add_nodes.append({
                    "node_id": "step_%d_add_%s" % (_step_for(sid), tid), "technique_id": tid, "technique_name": "",
                    "node_type": "obs", "explicit": True,
                    "evidence_sentence_ids": [sid] if isinstance(sid, int) else [],
                    "evidence_text": ev_txt,
                    "procedure": {"actor": "", "action": "", "object": "", "purpose": ""},
                    "confidence": 0.7,
                })
                continue
            nid = str(v.get("node_id"))
            if verdict == "remove":

                reason = str(v.get("reason") or "")
                obj = str(nid2obj.get(nid) or "").lower()
                pself = (re.search(r"T1059", nid) and
                         any(w in obj for w in (".exe", ".dll", "dropper", "downloaded pe", "malicious pe",
                                                "组件", "可执行", "beacon", "implant", "木马")) and
                         not any(w in obj for w in ("powershell", "cmd", "script", "脚本", "命令", ".sh", "shell", "code", "wmi")))
                prog_self = bool(pself) or ("程序自执行" in reason)
                if not prog_self:
                    m = re.search(r"step_\d+_[A-Za-z0-9_]*T\d{4}", reason)
                    anchor = m.group(0) if m else None
                    pnid = re.search(r"T\d{4}", nid)
                    panc = re.search(r"T\d{4}", anchor) if anchor else None

                    cross_tech = (panc and pnid and panc.group(0) != pnid.group(0))
                    same_sid = (anchor and anchor in node_by_id and not cross_tech
                                and nid2sid.get(nid) is not None
                                and nid2sid.get(nid) == nid2sid.get(anchor)
                                and (_obj_core(nid2obj.get(nid)) & _obj_core(nid2obj.get(anchor))))
                    if not same_sid:
                        logger.info("    P1守门: 降级误删 %s(非程序自执行/跨技术/非同句铁重复)→keep", nid)
                        continue
                del_nodes.add(nid)
            elif verdict == "retag" and v.get("new_technique_id") and nid in node_by_id:
                node_by_id[nid]["technique_id"] = v["new_technique_id"]
                node_by_id[nid]["technique_name"] = ""
        del_edges: set[tuple] = set()
        rev_edges: set[tuple] = set()
        for v in edge_verdicts or []:
            edge, verdict = str(v.get("edge") or ""), v.get("verdict")
            if "->" not in edge:
                continue
            s, d = (x.strip() for x in edge.split("->", 1))
            if verdict == "remove":
                del_edges.add((s, d))
            elif verdict == "reverse":
                rev_edges.add((s, d))
        new_edges = []
        for e in edges:
            key = (str(e.get("src")), str(e.get("dst")))
            if key[0] in del_nodes or key[1] in del_nodes:
                continue
            if key in del_edges and (not protect_main or e.get("recall_pass")):
                continue
            if key in rev_edges:
                e = dict(e)
                e["src"], e["dst"] = key[1], key[0]
            new_edges.append(e)
        g = dict(graph)
        g["nodes"] = [n for n in nodes if str(n.get("node_id")) not in del_nodes] + add_nodes
        g["edges"] = new_edges
        g.setdefault("main_path", graph.get("main_path", []))
        return g

    @staticmethod
    def _apply_add_edges(graph: dict, findings: list[dict], conf_floor: float = 0.0) -> dict:
        graph = graph or {}
        nodes = graph.get("nodes", []) or []
        edges = list(graph.get("edges", []) or [])
        valid = {str(n.get("node_id")) for n in nodes if n.get("node_id")}
        present = {(str(e.get("src")), str(e.get("dst"))) for e in edges}
        added = 0
        for f in findings or []:
            if str(f.get("action") or "") != "add_edge":
                continue
            try:
                conf = float(f.get("confidence"))
            except (TypeError, ValueError):
                conf = 0.5
            if conf < conf_floor:
                continue
            t = str(f.get("target") or "")
            if "->" not in t:
                continue
            s, d = (x.strip() for x in t.split("->", 1))
            if s not in valid or d not in valid or s == d or (s, d) in present:
                continue
            edges.append({"src": s, "dst": d, "evidence_type": "implicit",
                          "relation": str(f.get("relation") or "enables"),
                          "from_missing": True, "confidence": conf})
            present.add((s, d))
            added += 1
        if added:
            graph = dict(graph)
            graph["edges"] = edges
            logger.info("    route-back/找漏: +%d 漏边已补", added)
        return graph

    @staticmethod
    def _conf(f: dict) -> float:
        try:
            return float(f.get("confidence"))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _vcount(verdicts: list[dict]) -> dict:
        out: dict = {}
        for v in verdicts or []:
            out[str(v.get("verdict"))] = out.get(str(v.get("verdict")), 0) + 1
        return out

    @staticmethod
    def _backfill_one(node: dict) -> dict:
        tid = str(node.get("technique_id") or "")
        tac_id = node.get("tactic_id") or (get_tactic_id(tid) or "")
        node["tactic_id"] = tac_id
        if not node.get("tactic"):
            node["tactic"] = get_tactic_name(tac_id) if tac_id else ""
        node.setdefault("node_type", "obs")
        node.setdefault("explicit", node.get("node_type") == "obs")
        node.setdefault("confidence", 0.9)
        node.setdefault("procedure", {})
        node.setdefault("evidence_sentence_ids", [])
        node.setdefault("evidence_text", "")
        return node

    def _naive_pipeline(self, doc_id: str, sentences: list[str], full_text: str) -> dict:
        lines = [f"[{i}] {s}" for i, s in enumerate(sentences)]
        _naive = load_template("pipeline/naive")
        result = self.llm.chat(
            _naive["system"], _naive["user"].replace("{{SENTENCES}}", "\n".join(lines)),
            temperature=0.0, max_tokens=4096, extra_body=self._extra_body, agent="naive_pipeline")
        raw_nodes = result.get("nodes", []) if isinstance(result, dict) else []
        raw_edges = result.get("edges", []) if isinstance(result, dict) else []
        nodes, idx2id, n_sent = [], {}, len(sentences)
        for rn in (raw_nodes or []):
            tid = str(rn.get("technique_id") or "").strip()
            if not tid:
                continue
            nid = "step_%d_%s" % (len(nodes), tid)
            if rn.get("index") is not None:
                idx2id[str(rn.get("index"))] = nid
            sid = rn.get("sentence_id")
            sid = int(sid) if (isinstance(sid, int) or (isinstance(sid, str) and str(sid).isdigit())) else None
            ev = [sid] if (sid is not None and 0 <= sid < n_sent) else []
            nodes.append({"node_id": nid, "technique_id": tid, "mention": rn.get("mention", ""),
                          "evidence_sentence_ids": ev})
        nodes = self._backfill_nodes(nodes, sentences)
        valid = {n["node_id"] for n in nodes}
        edges, seen = [], set()
        for e in (raw_edges or []):
            s, d = idx2id.get(str(e.get("src"))), idx2id.get(str(e.get("dst")))
            if s and d and s in valid and d in valid and s != d and (s, d) not in seen:
                seen.add((s, d))
                edges.append({"src": s, "dst": d, "relation": "enables", "evidence_type": "explicit"})
        return {"nodes": nodes, "edges": edges, "main_path": [n["node_id"] for n in nodes]}

    def _run_naive(self, doc, doc_id, safe, sentences, full_text, t0):
        graph = self._naive_pipeline(doc_id, sentences, full_text)
        self._save(os.path.join(self.output_dir, "agent_a_results", f"{safe}.json"),
                   {"doc_id": doc_id, "nodes": graph["nodes"]})
        self._save(os.path.join(self.output_dir, "agent_b_results", f"{safe}.json"),
                   {"doc_id": doc_id, "num_nodes": len(graph["nodes"]), "num_edges": len(graph["edges"]), "graph": graph})
        self._save(os.path.join(self.output_dir, "verify_results", f"{safe}.json"), {"doc_id": doc_id, "enabled": False})
        logger.info("  naive(纯最简): %d nodes, %d edges", len(graph["nodes"]), len(graph["edges"]))
        graph = self._clean_normalize(graph)
        graph_doc = self._assemble(doc, graph)
        graph_doc.to_json(os.path.join(self.output_dir, "graphs", f"{safe}.json"))
        if self.enable_memory:
            self.memory.save()
        trace = {
            "doc_id": doc_id, "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": {k: v for k, v in self.config.items() if k != "api_key"},
            "tokens": self.llm.usage_stats,
            "stats": {"nodes": len(graph_doc.nodes), "edges": len(graph_doc.edges), "paths": len(graph_doc.paths)},
            "duration_s": round(time.time() - t0, 2),
        }
        self._save(os.path.join(self.output_dir, "traces", f"{safe}_trace.json"), trace)
        logger.info("CLEAN Pipeline DONE (naive): %s (%d nodes, %d edges)", doc_id, len(graph_doc.nodes), len(graph_doc.edges))
        return graph_doc

    def _backfill_nodes(self, nodes: list[dict], sentences: list[str]) -> list[dict]:
        out = []
        seen = set()
        for n in nodes or []:
            if not n or not n.get("node_id") or not n.get("technique_id"):
                continue
            if n["node_id"] in seen:
                continue
            seen.add(n["node_id"])
            if not n.get("evidence_text") and n.get("evidence_sentence_ids"):
                sid = n["evidence_sentence_ids"][0]
                if isinstance(sid, int) and 0 <= sid < len(sentences):
                    n["evidence_text"] = sentences[sid]
            out.append(self._backfill_one(n))
        return out

    def _backfill_graph(self, graph: dict, sentences: list[str]) -> dict:
        graph = dict(graph or {})
        graph["nodes"] = self._backfill_nodes(graph.get("nodes", []), sentences)
        graph["edges"] = list(graph.get("edges", []) or [])
        graph.setdefault("main_path", [])
        return graph

    _MISATTR_ALLOWLIST = {
        ("T1566", "T1203"), ("T1059", "T1105"), ("T1140", "T1071"), ("T1566", "T1053"),
        ("T1021", "T1047"), ("T1070", "T1105"), ("T1566", "T1059"), ("T1547", "T1105"),
        ("T1566", "T1221"), ("T1110", "T1219"), ("T1204", "T1140"), ("T1027", "T1083"),
        ("T1204", "T1203"), ("T1552", "T1041"),
    }

    @staticmethod
    def _correct_misattribution(graph: dict) -> dict:
        import re
        def par(t):
            m = re.search(r"T\d{4}", str(t or "")); return m.group(0) if m else str(t or "")
        nodes = list(graph.get("nodes") or [])
        edges = list(graph.get("edges") or [])
        nid2par = {str(n.get("node_id")): par(n.get("technique_id") or n.get("attack_id")) for n in nodes}
        src_by_dstpar: dict[str, set] = {}
        for e in edges:
            ps, pd = nid2par.get(str(e.get("src"))), nid2par.get(str(e.get("dst")))
            if ps and pd:
                src_by_dstpar.setdefault(pd, set()).add(ps)
        kept, removed = [], 0
        for e in edges:
            ps, pd = nid2par.get(str(e.get("src"))), nid2par.get(str(e.get("dst")))
            if (ps, pd) in CleanPipeline._MISATTR_ALLOWLIST and len(src_by_dstpar.get(pd, set()) - {ps}) >= 1:
                removed += 1
                continue
            kept.append(e)
        if removed:
            logger.info("  [L2] 源错挂纠正: 删 %d 条错源边(竞争门控)", removed)
            graph = dict(graph); graph["edges"] = kept
        return graph

    def _clean_normalize(self, graph: dict) -> dict:
        nodes = list(graph.get("nodes", []) or [])
        node_ids = {n["node_id"] for n in nodes if n.get("node_id")}
        parent_of = {}
        for n in nodes:
            tid = str(n.get("technique_id") or n.get("attack_id") or "")
            parent_of[n.get("node_id")] = tid.split(".")[0]
        seen = set()
        edges = []
        dropped_same_parent = 0
        for e in graph.get("edges", []) or []:
            src, dst = str(e.get("src") or ""), str(e.get("dst") or "")
            if not src or not dst or src == dst:
                continue
            if src not in node_ids or dst not in node_ids:
                continue

            ps, pd = parent_of.get(src), parent_of.get(dst)
            if ps and pd and ps == pd:
                dropped_same_parent += 1
                continue
            if (src, dst) in seen:
                continue
            rel = e.get("relation") or "enables"
            if rel not in VALID_RELATIONS:
                rel = "enables"
            e["relation"] = rel
            seen.add((src, dst))
            edges.append(e)
        if dropped_same_parent:
            logger.info("  [phase1] dropped %d same-parent self-loop edge(s)", dropped_same_parent)
        edges = self._break_cycles(edges)
        if self.transitive_reduction:
            edges = self._transitive_reduction(edges)
        nodes, edges = self._drop_residual_isolated(nodes, edges)
        graph["nodes"] = nodes
        graph["edges"] = edges
        graph["main_path"] = self._main_path(nodes, edges)
        return graph

    def _drop_residual_isolated(self, nodes: list[dict], edges: list[dict]):
        """Deterministic safety net: after structure repair + normalization, any node
        still isolated (in=out=0) is dropped — an isolated node cannot belong to an
        attack process and Agent C already had its chance to reconnect it. Never drops
        the sole node of a single-node graph."""
        if not self.structure_drop_isolated or len(nodes) <= 1:
            return nodes, edges
        node_ids = {n["node_id"] for n in nodes if n.get("node_id")}
        indeg = {nid: 0 for nid in node_ids}
        outdeg = {nid: 0 for nid in node_ids}
        for e in edges:
            s, d = str(e.get("src")), str(e.get("dst"))
            if s in outdeg:
                outdeg[s] += 1
            if d in indeg:
                indeg[d] += 1
        iso = [nid for nid in node_ids if indeg[nid] == 0 and outdeg[nid] == 0]
        if not iso:
            return nodes, edges
        iso_set = set(iso)
        kept_nodes = [n for n in nodes if n.get("node_id") not in iso_set]
        kept_ids = {n["node_id"] for n in kept_nodes if n.get("node_id")}
        kept_edges = [e for e in edges
                      if str(e.get("src")) in kept_ids and str(e.get("dst")) in kept_ids]
        logger.info("  [structure] 安全网: 删 %d 个残留孤立节点 %s", len(iso), iso)
        return kept_nodes, kept_edges

    @staticmethod
    def _transitive_reduction(edges: list[dict]) -> list[dict]:

        if any(e.get("recall_pass") for e in edges):
            logger.info("  [transitive-reduction] 召回趟模式: 跳过(精度交验证器)")
            return edges
        adj: dict[str, set] = {}
        for e in edges:
            adj.setdefault(str(e.get("src")), set()).add(str(e.get("dst")))

        def reachable_via_other(u: str, v: str) -> bool:
            stack = [w for w in adj.get(u, set()) if w != v]
            seen: set[str] = set()
            while stack:
                c = stack.pop()
                if c == v:
                    return True
                if c in seen:
                    continue
                seen.add(c)
                stack.extend(adj.get(c, set()))
            return False

        kept, dropped = [], 0
        for e in edges:
            if e.get("from_missing"):
                kept.append(e)
                continue
            if reachable_via_other(str(e.get("src")), str(e.get("dst"))):
                dropped += 1
                continue
            kept.append(e)
        if dropped:
            logger.info("  [transitive-reduction] 删 %d 条传递跨越边(最直接归因)", dropped)
        return kept

    @staticmethod
    def _break_cycles(edges: list[dict]) -> list[dict]:
        kept_pairs: set[tuple[str, str]] = set()
        adj: dict[str, list[str]] = {}

        def reachable(s, d):
            stack, seen = [s], set()
            while stack:
                c = stack.pop()
                if c == d:
                    return True
                if c in seen:
                    continue
                seen.add(c)
                stack.extend(adj.get(c, []))
            return False

        kept = []
        for e in edges:
            s, d = e["src"], e["dst"]
            if reachable(d, s):
                continue
            kept.append(e)
            kept_pairs.add((s, d))
            adj.setdefault(s, []).append(d)
        return kept

    @staticmethod
    def _main_path(nodes: list[dict], edges: list[dict]) -> list[str]:
        adj: dict[str, list[str]] = {}
        indeg: dict[str, int] = {n["node_id"]: 0 for n in nodes}
        outdeg: dict[str, int] = {n["node_id"]: 0 for n in nodes}
        sid_of: dict[str, int] = {}
        tech_of: dict[str, str] = {}
        for n in nodes:
            nid = n["node_id"]
            sids = [int(s) for s in (n.get("evidence_sentence_ids") or []) if isinstance(s, int)]
            sid_of[nid] = min(sids) if sids else 10 ** 9
            tech_of[nid] = str(n.get("technique_id") or n.get("attack_id") or "")
        for e in edges:
            adj.setdefault(e["src"], []).append(e["dst"])
            indeg[e["dst"]] = indeg.get(e["dst"], 0) + 1
            outdeg[e["src"]] = outdeg.get(e["src"], 0) + 1
        # roots = in==0 AND out>0 (exclude isolated nodes — a degree-0 node is not a
        # legitimate start), ordered by (tactic phase, evidence sentence) so the main
        # path begins at a semantically-early entry rather than an arbitrary node.
        roots = [nid for nid, d in indeg.items() if d == 0 and outdeg.get(nid, 0) > 0]
        if not roots:
            roots = ([nid for nid in indeg if outdeg.get(nid, 0) > 0][:1]
                     or [n["node_id"] for n in nodes[:1]])
        roots.sort(key=lambda nid: (get_phase_index(get_tactic_id(tech_of.get(nid, ""))),
                                    sid_of.get(nid, 10 ** 9)))
        best: list[str] = []
        for r in roots:
            stack = [(r, [r])]
            while stack:
                cur, path = stack.pop()
                if len(path) > len(best):
                    best = path
                for nxt in adj.get(cur, []):
                    if nxt not in path:
                        stack.append((nxt, path + [nxt]))
        return best

    def _assemble(self, doc: StandardizedInputDocument, graph: dict) -> GraphDocument:
        nodes = []
        for n in graph.get("nodes", []):
            tid = n.get("technique_id") or n.get("attack_id") or ""
            meta = {
                "tactic_id": n.get("tactic_id") or get_tactic_id(tid) or "",
                "node_type_edl": n.get("node_type", "obs"),
                "evidence_text": n.get("evidence_text", ""),
                "evidence_sentence_ids": n.get("evidence_sentence_ids", []),
                "explicit": n.get("explicit", n.get("node_type") == "obs"),
            }
            if n.get("procedure"):
                meta["procedure"] = n["procedure"]
            sids = n.get("evidence_sentence_ids") or []
            nodes.append(GraphNode(
                node_id=n.get("node_id", ""),
                mention=n.get("technique_name") or tid,
                node_type="action",
                attack_id=tid or None,
                sentence_id=sids[0] if sids else None,
                confidence=n.get("confidence"),
                metadata=meta,
            ))
        node_ids = {n.node_id for n in nodes}
        edges = []
        for e in graph.get("edges", []):
            rel = e.get("relation", "related")
            if rel not in VALID_RELATIONS:
                rel = "related"
            if e.get("src") not in node_ids or e.get("dst") not in node_ids or e.get("src") == e.get("dst"):
                continue
            edges.append(GraphEdge(
                src=e["src"], dst=e["dst"], relation=rel,
                confidence=e.get("confidence"),
                metadata={
                    "evidence_type": e.get("evidence_type", ""),
                    "evidence_text": e.get("evidence_text", ""),
                    "evidence_sentence_ids": e.get("evidence_sentence_ids", []),
                },
            ))
        paths = []
        mp = [nid for nid in graph.get("main_path", []) if nid in node_ids]
        if len(mp) >= 2:
            paths.append(GraphPath(path_id=f"{doc.doc_id}:main_execution", node_ids=mp, confidence=0.8))
        return GraphDocument(
            doc_id=doc.doc_id, source_dataset=doc.source_dataset, source_path=doc.source_path,
            text=doc.text, nodes=nodes, edges=edges, paths=paths,
            metadata={"pipeline": "EDL-CLEAN", "model": self.config.get("model", "deepseek-chat"),
                      "timestamp": datetime.now(timezone.utc).isoformat()},
        )

    @staticmethod
    def _save(path: str, data: dict):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
