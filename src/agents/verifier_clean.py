from __future__ import annotations
from prompt_loader import load_template

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

def _step_idx(node_id: str):
    m = re.match(r"step_(\d+)_", str(node_id or ""))
    return int(m.group(1)) if m else None

REWRITE_SYSTEM_PROMPT = load_template('verifier/rewrite')['system']

NODE_CHECK_SYSTEM_PROMPT = load_template('verifier/node_check')['system']
EDGE_CHECK_SYSTEM_PROMPT = load_template('verifier/edge_check')['system']
STRUCTURE_CHECK_SYSTEM_PROMPT = load_template('verifier/structure_check')['system']

class CleanVerifier:

    def __init__(self, llm, temperature: float = 0.0, extra_body: dict | None = None,
                 missing_k: int = 3, missing_thresh: int = 2):
        self.llm = llm
        self.temperature = temperature
        self.extra_body = extra_body

        self.missing_k = int(missing_k)
        self.missing_thresh = int(missing_thresh)

        self.missing_use_rewrite = True

    def rewrite(self, doc_id: str, nodes: list[dict], edges: list[dict],
                main_path: list[str] | None = None) -> tuple[str, list[dict]]:
        prompt = self._build_rewrite_prompt(doc_id, nodes, edges, list(main_path or []))
        try:
            r1 = self.llm.chat(REWRITE_SYSTEM_PROMPT, prompt, temperature=self.temperature,
                               max_tokens=8192, extra_body=self.extra_body, agent="verifier_rewrite")
            return (str(r1.get("rewritten_report") or ""),
                    self._normalize_rewritten_sentences(r1.get("rewritten_sentences")))
        except Exception as exc:
            logger.warning("[CleanVerifier] rewrite failed for %s: %s", doc_id, exc)
            return "", []

    def check_nodes(self, doc_id: str, sentences: list[str], nodes: list[dict], edges: list[dict],
                    rewritten_sentences: list[dict], node_mem: list[dict] | None = None) -> list[dict]:
        sentences = list(sentences or [])
        prompt = self._build_node_check_prompt(doc_id, sentences, nodes, edges,
                                               rewritten_sentences, node_mem or [])
        valid_ids = {str(n.get("node_id")) for n in nodes if n.get("node_id")}
        valid_parents = {self._parent_of(n.get("technique_id") or n.get("node_id")) for n in nodes}
        valid_parents.discard(None)
        try:
            r = self.llm.chat(NODE_CHECK_SYSTEM_PROMPT, prompt, temperature=self.temperature,
                              max_tokens=8192, extra_body=self.extra_body, agent="verifier_node_check")
        except Exception as exc:
            logger.warning("[CleanVerifier] node_check failed for %s: %s", doc_id, exc)
            return []
        f = self._normalize_findings(r.get("findings") if isinstance(r, dict) else None,
                                     valid_ids, set(), valid_parents, sentences)
        return [x for x in f if x["type"] == "OBD"]

    def check_edges(self, doc_id: str, sentences: list[str], nodes: list[dict], edges: list[dict],
                    rewritten_sentences: list[dict], wrong_edge_mem: list[dict] | None = None,
                    edge_signals: dict | None = None, fanout_alerts: list[dict] | None = None,
                    review_mem: list[dict] | None = None) -> list[dict]:
        sentences = list(sentences or [])

        alerts = list(fanout_alerts or []) + [{"hint": r.get("hint", "")} for r in (review_mem or [])]
        prompt = self._build_edge_check_prompt(doc_id, sentences, nodes, edges, rewritten_sentences,
                                               wrong_edge_mem or [], edge_signals or {}, alerts)
        valid_ids = {str(n.get("node_id")) for n in nodes if n.get("node_id")}
        valid_edges = {(str(e.get("src")), str(e.get("dst"))) for e in edges}
        valid_parents = {self._parent_of(n.get("technique_id") or n.get("node_id")) for n in nodes}
        valid_parents.discard(None)
        try:
            r = self.llm.chat(EDGE_CHECK_SYSTEM_PROMPT, prompt, temperature=self.temperature,
                              max_tokens=8192, extra_body=self.extra_body, agent="verifier_edge_check")
        except Exception as exc:
            logger.warning("[CleanVerifier] edge_check failed for %s: %s", doc_id, exc)
            return []
        f = self._normalize_findings(r.get("findings") if isinstance(r, dict) else None,
                                     valid_ids, valid_edges, valid_parents, sentences)
        f = [x for x in f if x["action"] in ("remove", "replace", "demote")
             and x["type"] in ("ODD", "IDD", "IND")]

        edge2hint = {}
        for r2 in (wrong_edge_mem or []):
            for (a, b) in r2.get("hit_edges", []):
                edge2hint.setdefault("%s->%s" % (a, b), []).append("[记忆·该删] " + str(r2.get("hint", ""))[:120])
        for r2 in (review_mem or []):
            for (a, b) in r2.get("hit_edges", []):
                edge2hint.setdefault("%s->%s" % (a, b), []).append("[记忆·核对] " + str(r2.get("hint", ""))[:120])
        for x in f:
            mh = edge2hint.get(str(x.get("target", "")))
            if mh:
                x["mem_hint"] = " ".join(mh)
        return f

    def find_missing(self, doc_id: str, sentences: list[str], nodes: list[dict], edges: list[dict],
                     missing_edge_mem: list[dict] | None = None,
                     missing_candidates: list[dict] | None = None,
                     rewritten_sentences: list[dict] | None = None,
                     edge_rules: list[dict] | None = None) -> list[dict]:
        sentences = list(sentences or [])
        valid_ids = {str(n.get("node_id")) for n in nodes if n.get("node_id")}
        valid_edges = {(str(e.get("src")), str(e.get("dst"))) for e in edges}
        valid_parents = {self._parent_of(n.get("technique_id") or n.get("node_id")) for n in nodes}
        valid_parents.discard(None)
        prompt = self._build_missing_prompt(doc_id, sentences, nodes, edges, missing_edge_mem or [],
                                            rewritten_sentences if self.missing_use_rewrite else None,
                                            missing_candidates or [], edge_rules or [])
        try:
            r = self.llm.chat(self._missing_system(), prompt, temperature=self.temperature,
                              max_tokens=4096, extra_body=self.extra_body, agent="verifier_missing")
        except Exception as exc:
            logger.warning("[CleanVerifier] find_missing failed for %s: %s", doc_id, exc)
            return []
        raw = (r if isinstance(r, dict) else {}).get("findings")
        out = self._normalize_missing(raw, valid_ids, valid_edges, sentences)
        ind = [x for x in self._normalize_findings(raw, valid_ids, valid_edges, valid_parents, sentences)
               if x["type"] == "IND" and x["action"] == "insert_node"]
        return out + ind

    def check_structure(self, doc_id: str, sentences: list[str], nodes: list[dict], edges: list[dict],
                        rewritten_sentences: list[dict] | None,
                        structural_report: Any, structural_experiences: list[dict] | None = None) -> list[dict]:
        """Graph-level repair stage (Agent C ④): replay the *whole* graph and repair
        each detected structural anomaly in place — reconnect an isolated / illegitimate
        root to an evidence-supported neighbour, drop it if unsupported, or retag it.
        Findings reuse the standard finding schema and flow through the normal apply path.
        """
        sentences = list(sentences or [])
        prompt = self._build_structure_check_prompt(doc_id, sentences, nodes, edges,
                                                    rewritten_sentences or [], structural_report,
                                                    structural_experiences or [])
        valid_ids = {str(n.get("node_id")) for n in nodes if n.get("node_id")}
        valid_edges = {(str(e.get("src")), str(e.get("dst"))) for e in edges}
        valid_parents = {self._parent_of(n.get("technique_id") or n.get("node_id")) for n in nodes}
        valid_parents.discard(None)
        try:
            r = self.llm.chat(STRUCTURE_CHECK_SYSTEM_PROMPT, prompt, temperature=self.temperature,
                              max_tokens=4096, extra_body=self.extra_body, agent="verifier_structure_check")
        except Exception as exc:
            logger.warning("[CleanVerifier] structure_check failed for %s: %s", doc_id, exc)
            return []
        f = self._normalize_findings(r.get("findings") if isinstance(r, dict) else None,
                                     valid_ids, valid_edges, valid_parents, sentences)
        return [x for x in f if x["action"] in ("add_edge", "remove", "replace")]

    def _build_structure_check_prompt(self, doc_id, sentences, nodes, edges,
                                      rewritten_sentences, structural_report, structural_experiences):
        t = load_template('verifier/structure_check')
        rep = (structural_report.to_dict() if hasattr(structural_report, "to_dict")
               else dict(structural_report or {}))
        in_deg = rep.get("in_degree") or {}
        out_deg = rep.get("out_degree") or {}
        tech_by_id = {str(n.get("node_id")): str(n.get("technique_id") or "") for n in nodes}
        parts = [t['head'].replace("{{DOC_ID}}", doc_id), ""]
        parts.append(t['sec_1'])
        parts.extend([f"[{i}] {s}" for i, s in enumerate(sentences)] or ["(无原文句)"])
        parts.append("")
        parts.append(t['sec_2'])
        if rewritten_sentences:
            for rs in rewritten_sentences:
                parts.append("[rid=%s] %s  (from_node_ids=%s)" % (
                    rs.get("rid"), str(rs.get("text", ""))[:120], rs.get("from_node_ids", [])))
        else:
            parts.append("(还原报告为空)")
        parts.append("")
        parts.append(t['sec_3'])
        for n in nodes:
            nid = str(n.get("node_id"))
            proc = n.get("procedure") or {}
            act = ("action=%s object=%s" % (proc.get("action", ""), proc.get("object", ""))
                   if isinstance(proc, dict) else "")
            ev_ids = n.get("evidence_sentence_ids") or []
            parts.append("- %s [in=%d,out=%d] | %s | %s | 证据句%s" % (
                nid, int(in_deg.get(nid, 0)), int(out_deg.get(nid, 0)),
                tech_by_id.get(nid, ""), act, ev_ids))
        parts.append("  边:")
        for e in (edges or []):
            parts.append("    %s -> %s" % (e.get("src"), e.get("dst")))
        parts.append("")
        parts.append(t['sec_struct'])
        for f in rep.get("findings", []) or []:
            parts.append("- [%s] node=%s tech=%s tactic=%s | %s" % (
                f.get("kind"), f.get("node_id"), f.get("technique_id"),
                f.get("tactic_name") or f.get("tactic") or "?", f.get("detail", "")))
        parts.append("")
        if structural_experiences:
            parts.append(t['sec_exp'])
            for x in structural_experiences:
                parts.append("- [%s] %s" % ("/".join(x.get("kinds", [])), str(x.get("hint", ""))[:240]))
            parts.append("")
        parts.append(t['tail'])
        return "\n".join(parts)

    def _build_node_check_prompt(self, doc_id, sentences, nodes, edges, rewritten_sentences, node_mem):
        t = load_template('verifier/node_check')
        corr_mem = [w for w in (node_mem or []) if not w.get("find_node")]
        find_mem = [w for w in (node_mem or []) if w.get("find_node")]
        parts = [t['head'].replace("{{DOC_ID}}", doc_id), ""]
        parts.append(t['sec_1'])
        parts.extend([f"[{i}] {s}" for i, s in enumerate(sentences)] or ["(无原文句)"])
        parts.append("")

        if find_mem:
            parts.append("## ★高度疑似漏抽节点 (检索命中, 【逐条必答·补或不补】, 这是本步重点不要只删不补):")
            for w in find_mem:
                parts.append("- [核对是否补 %s] %s" % ("/".join(w.get("tech", [])), str(w.get("hint", ""))))
            parts.append("")
        parts.append(t['sec_2'])
        if rewritten_sentences:
            for rs in rewritten_sentences:
                parts.append(f"[rid={rs.get('rid')}] {rs.get('text', '')}"
                             f"  (from_node_ids={rs.get('from_node_ids', [])})")
        else:
            parts.append("(还原报告为空)")
        parts.append("")
        parts.append(t['sec_3'])
        if nodes:
            for i, n in enumerate(nodes):
                node_type = self._node_type_label(n.get("node_type"))
                proc = n.get("procedure") or {}
                act = ((f"actor={proc.get('actor','')} action={proc.get('action','')} "
                        f"object={proc.get('object','')}").strip() if isinstance(proc, dict) else "")
                ev_ids = n.get("evidence_sentence_ids") or []
                ev_txt = ""
                for sid in ev_ids:
                    try:
                        si = int(sid)
                        if 0 <= si < len(sentences):
                            ev_txt = sentences[si]; break
                    except (TypeError, ValueError):
                        pass
                parts.append(f"N{i}: {n.get('node_id','')} | {n.get('technique_id','')} | {node_type} | {act}")
                parts.append(f"     证据句[{ev_ids}]: {ev_txt}")
        else:
            parts.append("(无节点)")
        parts.append("")

        import re as _re
        groups: dict = {}
        for n in nodes:
            if self._node_type_label(n.get("node_type")) != "explicit":
                continue
            m = _re.search(r"T\d{4}", str(n.get("technique_id") or ""))
            if m:
                groups.setdefault(m.group(0), []).append(n)
        multi = {p: ns for p, ns in groups.items() if len(ns) >= 2}
        if multi:
            parts.append(t['sec_multi'])
            for p, ns in multi.items():
                parts.append("### 技术 %s (%d 个节点):" % (p, len(ns)))
                for n in ns:
                    pr = n.get("procedure") or {}
                    sid = (n.get("evidence_sentence_ids") or [None])[0]
                    parts.append("  - %s: s%s | action=%s | object=%s" % (
                        n.get("node_id"), sid, (pr or {}).get("action", ""), (pr or {}).get("object", "")))
            parts.append("")

        if corr_mem:
            parts.append(t['sec_node_mem'])
            parts.extend(f"- {str(w.get('hint', ''))}" for w in corr_mem)
            parts.append("")
        prof = self._structural_profile(nodes, edges)
        parts.append(t['sec_5'])
        parts.append(t['prof_density'].replace("{{DENSITY}}", str(prof['density'])))
        parts.append(t['prof_sparse'] if prof["sparse"] else t['prof_normal'])
        if prof["hub_suspects"]:
            parts.append(t['prof_hub'].replace("{{HUBS}}", str(prof['hub_suspects'])))
        if prof["weak_edges"]:
            parts.append(t['prof_weak'].replace("{{WEAK}}", str(prof['weak_edges'])))
        parts.append("")
        parts.append(t['tail'])
        return "\n".join(parts)

    def _build_edge_check_prompt(self, doc_id, sentences, nodes, edges, rewritten_sentences,
                                 wrong_edge_mem, edge_signals, fanout_alerts=None):
        t = load_template('verifier/edge_check')
        tech_by_id = {str(n.get("node_id")): str(n.get("technique_id") or "") for n in nodes}
        parts = [t['head'].replace("{{DOC_ID}}", doc_id), ""]
        parts.append(t['sec_1'])
        parts.extend([f"[{i}] {s}" for i, s in enumerate(sentences)] or ["(无原文句)"])
        parts.append("")
        parts.append(t['sec_2'])
        if rewritten_sentences:
            for rs in rewritten_sentences:
                parts.append(f"[rid={rs.get('rid')}] {rs.get('text', '')}"
                             f"  (from_node_ids={rs.get('from_node_ids', [])}, from_edges={rs.get('from_edges', [])})")
        else:
            parts.append("(还原报告为空)")
        parts.append("")
        parts.append(t['sec_35'])
        if edges:
            for i, e in enumerate(edges):
                s, d = str(e.get("src")), str(e.get("dst"))
                et = str(e.get("evidence_type") or "")
                lab = "隐式implicit" if et.lower().startswith("impl") else "显式explicit"
                sig = edge_signals.get("%s->%s" % (s, d))
                tag = (" | 可疑度=%s(矩阵ratio=%.2f%s)" % (
                    sig["tier"], sig["ratio"], ",记忆命中" if sig["wrong"] else "")) if sig else ""
                parts.append(f"E{i}: {s}({tech_by_id.get(s,'')}) -> {d}({tech_by_id.get(d,'')}) "
                             f"| {lab} | relation={e.get('relation','')}{tag}")
        else:
            parts.append("(无边)")
        parts.append("")
        inf_nodes = [n for n in nodes if self._node_type_label(n.get("node_type")) == "implicit"]
        if inf_nodes:
            parts.append(t['sec_node_mem'])
            parts.extend(f"- {n.get('node_id')} | {n.get('technique_id','')}" for n in inf_nodes)
            parts.append("")
        if wrong_edge_mem:
            parts.append(t['sec_wrong_edge'])
            parts.extend(f"- {str(w.get('hint', ''))}" for w in wrong_edge_mem)
            parts.append("")

        if fanout_alerts:
            parts.append("## 扇出/扇入结构核对 (下列枢纽链请逐条判: 分支该并行挂枢纽 还是 真链挂中间步; 据原文产物承接定, 别默认串成链):")
            parts.extend(f"- {str(a.get('hint', ''))}" for a in fanout_alerts)
            parts.append("")
        prof = self._structural_profile(nodes, edges)
        parts.append(t['sec_5'])
        parts.append(t['prof_density'].replace("{{DENSITY}}", str(prof['density'])))
        parts.append(t['prof_sparse'] if prof["sparse"] else t['prof_normal'])
        if prof["hub_suspects"]:
            parts.append(t['prof_hub'].replace("{{HUBS}}", str(prof['hub_suspects'])))
        if prof["weak_edges"]:
            parts.append(t['prof_weak'].replace("{{WEAK}}", str(prof['weak_edges'])))
        parts.append("")
        parts.append(t['tail'])
        return "\n".join(parts)

    def _missing_system(self) -> str:
        return load_template('verifier/missing')['system']

    def _build_missing_prompt(self, doc_id: str, sentences: list[str], nodes: list[dict],
                              edges: list[dict], missing_edge_mem: list[dict] | None = None,
                              rewritten_sentences: list[dict] | None = None,
                              missing_candidates: list[dict] | None = None,
                              edge_rules: list[dict] | None = None) -> str:
        sentences = list(sentences or [])
        t = load_template('verifier/missing')
        parts = [t['head'].replace("{{DOC_ID}}", doc_id), ""]

        parts.append(t['sec_1'])
        if sentences:
            for i, s in enumerate(sentences):
                parts.append(f"[{i}] {s}")
        else:
            parts.append("(无原文句)")
        parts.append("")

        rewritten_sentences = rewritten_sentences or []
        if rewritten_sentences:
            parts.append(t['sec_15'])
            for rs in rewritten_sentences:
                parts.append("[rid=%s] %s (来自节点=%s, 边=%s)" % (
                    rs.get("rid"), str(rs.get("text", ""))[:90],
                    rs.get("from_node_ids", []), rs.get("from_edges", [])))
            parts.append("")

        parts.append(t['sec_2'])
        for nd in nodes:
            nid = str(nd.get("node_id", ""))
            proc = nd.get("procedure") or {}
            act = ("action=%s object=%s" % (proc.get("action", ""), proc.get("object", ""))) if isinstance(proc, dict) else ""
            ev_ids = nd.get("evidence_sentence_ids") or []
            ev_txt = ""
            for sid in ev_ids:
                try:
                    si = int(sid)
                    if 0 <= si < len(sentences):
                        ev_txt = sentences[si]
                        break
                except (TypeError, ValueError):
                    pass
            parts.append("- %s | %s | %s" % (nid, nd.get("technique_id", ""), act))
            parts.append("    原句[%s]: %s" % (ev_ids, ev_txt))
        parts.append("")

        parts.append(t['sec_3'])
        if edges:
            for e in edges:
                parts.append("- %s -> %s" % (e.get("src"), e.get("dst")))
        else:
            parts.append("(当前无边)")
        parts.append("")

        missing_edge_mem = missing_edge_mem or []
        parts.append(t['sec_4'])
        if missing_edge_mem:
            parts.append(t['sec_4_hit'])
            for m in missing_edge_mem:
                parts.append("- %s" % str(m.get("hint", ""))[:240])
        else:
            parts.append("(本报告无检索到的漏边提醒)")
        parts.append("")

        missing_candidates = missing_candidates or []
        if missing_candidates:
            parts.append(t['sec_5'])
            parts.append(t['sec_5_sub'])
            for c in sorted(missing_candidates, key=lambda x: (x.get("level") != "strong",)):
                parts.append("- [%s] %s -> %s (%s) | 转移矩阵:%s 角色记忆:%s" % (
                    "强" if c.get("level") == "strong" else "中", c.get("src"), c.get("dst"),
                    c.get("par"), "命中" if c.get("matrix") else "—", "命中" if c.get("mem") else "—"))
            parts.append("")

        pos_er = [r for r in (edge_rules or []) if r.get("dep")]
        if pos_er:
            parts.append("## (6) 强绑定该连规则 (产物天然绑定·已按叙述邻接筛过, 命中且原文支持就补; 仍以原文为准):")
            for r in pos_er:
                parts.append("- %s" % str(r.get("hint", ""))[:200])
            parts.append("")
        parts.append(t['tail'])
        return "\n".join(parts)

    @staticmethod
    def _normalize_missing(raw, valid_node_ids, valid_edges, sentences) -> list:
        out: list[dict] = []
        if not isinstance(raw, list):
            return out
        seen = set()
        for f in raw:
            if not isinstance(f, dict) or str(f.get("action") or "") != "add_edge":
                continue
            t = str(f.get("target") or "")
            if "->" not in t:
                continue
            s, d = (x.strip() for x in t.split("->", 1))
            if s not in valid_node_ids or d not in valid_node_ids or s == d:
                continue
            if (s, d) in valid_edges or (s, d) in seen:
                continue

            si, di = _step_idx(s), _step_idx(d)
            if si is not None and di is not None and si > di:
                continue
            seen.add((s, d))
            try:
                conf = float(f.get("confidence"))
            except (TypeError, ValueError):
                conf = 0.5
            out.append({
                "type": "ODD", "action": "add_edge", "target": "%s->%s" % (s, d),
                "route_to": "reasoner", "confidence": conf,
                "reason": str(f.get("reason") or "")[:40],
                "evidence_span": f.get("evidence_span") or {},
            })
        return out

    def _build_rewrite_prompt(
        self, doc_id: str, nodes: list[dict], edges: list[dict], main_path: list[str]
    ) -> str:
        node_lines: list[str] = []
        if nodes:
            for n in nodes:
                node_type = self._node_type_label(n.get("node_type"))
                proc = n.get("procedure") or {}
                proc_str = (
                    f"actor={proc.get('actor', '')} | action={proc.get('action', '')} | "
                    f"object={proc.get('object', '')} | purpose={proc.get('purpose', '')}"
                )
                node_lines.append(
                    f"- node_id={n.get('node_id', '')} | "
                    f"technique_id={n.get('technique_id', '')} | "
                    f"technique_name={n.get('technique_name', '')} | "
                    f"node_type={node_type}"
                )
                node_lines.append(f"    五元组: {proc_str}")
                node_lines.append(
                    f"    evidence_sentence_ids={n.get('evidence_sentence_ids', [])}"
                )
                node_lines.append(f"    evidence_text={n.get('evidence_text', '')}")
        else:
            node_lines.append("(无节点)")

        edge_lines: list[str] = []
        if edges:
            for e in edges:
                edge_lines.append(
                    f"- {e.get('src', '')}->{e.get('dst', '')} | "
                    f"relation={e.get('relation', 'enables')} | "
                    f"evidence_type={e.get('evidence_type', '')}"
                )
        else:
            edge_lines.append("(无边)")

        main_path_str = " -> ".join(main_path) if main_path else "(无主路径)"
        tpl = load_template('verifier/rewrite')
        return (tpl['user'].replace("{{DOC_ID}}", doc_id)
                .replace("{{NODES}}", "\n".join(node_lines))
                .replace("{{EDGES}}", "\n".join(edge_lines))
                .replace("{{MAIN_PATH}}", main_path_str))

    @staticmethod
    def _structural_profile(nodes: list[dict], edges: list[dict]) -> dict:
        n = len(nodes)
        m = len(edges)
        density = m / max(n, 1)
        rho: dict[str, int] = {}
        for nd in nodes:
            sids = [int(s) for s in (nd.get("evidence_sentence_ids") or [])
                    if str(s).lstrip("-").isdigit()]
            rho[str(nd.get("node_id"))] = min(sids) if sids else 10 ** 9
        out_deg: dict[str, int] = {str(nd.get("node_id")): 0 for nd in nodes}
        in_deg: dict[str, int] = {str(nd.get("node_id")): 0 for nd in nodes}
        for e in edges:
            s, d = str(e.get("src")), str(e.get("dst"))
            out_deg[s] = out_deg.get(s, 0) + 1
            in_deg[d] = in_deg.get(d, 0) + 1
        isolated = sorted(nid for nid in out_deg
                          if out_deg.get(nid, 0) == 0 and in_deg.get(nid, 0) == 0)

        degs = sorted(out_deg.values())
        if degs:
            mid = len(degs) // 2
            median = degs[mid] if len(degs) % 2 == 1 else (degs[mid - 1] + degs[mid]) / 2
        else:
            median = 0
        threshold = max(3, median + 2)
        hub_suspects = sorted(nid for nid, d in out_deg.items() if d >= threshold)
        weak_edges: list[str] = []
        for e in edges:
            if str(e.get("evidence_text") or "").strip():
                continue
            rs = rho.get(str(e.get("src")), 10 ** 9)
            rd = rho.get(str(e.get("dst")), 10 ** 9)
            if abs(rd - rs) >= 5:
                weak_edges.append(f"{e.get('src')}->{e.get('dst')}")
        return {
            "density": round(density, 3),
            "out_degree": out_deg,
            "in_degree": in_deg,
            "isolated": isolated,
            "hub_suspects": hub_suspects,
            "weak_edges": weak_edges,
            "sparse": density < 0.8,
        }

    @staticmethod
    def _node_type_label(node_type: Any) -> str:
        nt = str(node_type or "").strip().lower()
        if nt in ("inf", "implicit", "inferred"):
            return "implicit"

        return "explicit"

    @staticmethod
    def _normalize_rewritten_sentences(raw: Any) -> list[dict]:
        out: list[dict] = []
        if not isinstance(raw, list):
            return out
        for i, item in enumerate(raw):
            if not isinstance(item, dict):
                continue
            rid = item.get("rid", i)
            try:
                rid = int(rid)
            except (TypeError, ValueError):
                rid = i
            from_nodes = item.get("from_node_ids") or []
            if not isinstance(from_nodes, list):
                from_nodes = [from_nodes]
            from_edges = item.get("from_edges") or []
            if not isinstance(from_edges, list):
                from_edges = [from_edges]
            out.append({
                "rid": rid,
                "text": str(item.get("text") or ""),
                "from_node_ids": [str(x) for x in from_nodes],
                "from_edges": [str(x) for x in from_edges],
            })
        return out

    @staticmethod
    def _parent_of(x: Any) -> str | None:
        m = re.search(r"T\d{4}", str(x or ""))
        return m.group(0) if m else None

    @staticmethod
    def _normalize_findings(
        raw: Any,
        valid_node_ids: set | None = None,
        valid_edges: set | None = None,
        valid_parents: set | None = None,
        sentences: list | None = None,
    ) -> list[dict]:
        out: list[dict] = []
        if not isinstance(raw, list):
            return out
        valid_node_ids = valid_node_ids or set()
        valid_edges = valid_edges or set()
        valid_parents = valid_parents or set()
        actions = {"insert_node", "add_edge", "replace", "remove", "demote"}
        tech_re = re.compile(r"^[Tt]\d{4}(\.\d{3})?$")

        def parse_edge(t: str):
            if "->" not in t:
                return None
            a, b = (x.strip() for x in t.split("->", 1))
            return (a, b) if a and b else None

        for item in raw:
            if not isinstance(item, dict):
                continue
            action = str(item.get("action") or "").strip().lower()
            if action not in actions:
                continue

            dtype = str(item.get("type") or "").strip().upper()
            if dtype not in ("OBD", "ODD", "IND", "IDD"):
                obs = str(item.get("layer") or "").lower().startswith("obs")
                node = str(item.get("target_kind") or "").lower() == "node"
                dtype = ("OBD" if (obs and node) else "ODD" if (obs and not node)
                         else "IND" if (not obs and node) else "IDD")

            span = item.get("evidence_span")
            if not isinstance(span, dict):
                continue
            try:
                sid = int(span.get("sentence_id"))
            except (TypeError, ValueError):
                continue
            text = str(span.get("text") or "").strip()
            if not text and isinstance(sentences, list) and 0 <= sid < len(sentences):
                text = str(sentences[sid]).strip()
            if not text:
                continue
            target = str(item.get("target") or "").strip()

            if action == "insert_node":
                if not tech_re.match(target):
                    continue
                target = target.upper()

                if CleanVerifier._parent_of(target) in valid_parents:
                    logger.warning("[CleanVerifier] drop insert_node %r: parent already in graph", target)
                    continue
            elif action == "add_edge":
                pe = parse_edge(target)
                if not pe or pe[0] not in valid_node_ids or pe[1] not in valid_node_ids:
                    continue
                if pe in valid_edges:
                    continue
            elif action in ("remove", "replace"):
                if dtype in ("ODD", "IDD"):
                    pe = parse_edge(target)
                    if not pe or pe not in valid_edges:
                        logger.warning("[CleanVerifier] drop %s: edge %r not in graph", action, target)
                        continue
                else:
                    if target not in valid_node_ids:
                        logger.warning("[CleanVerifier] drop %s: node %r not in graph", action, target)
                        continue
            elif action == "demote":
                pe = parse_edge(target)
                if pe is not None:
                    if pe not in valid_edges:
                        continue
                elif target not in valid_node_ids:
                    continue
            route = "identifier" if dtype == "OBD" else "reasoner"
            try:
                conf = float(item.get("confidence"))
            except (TypeError, ValueError):
                conf = 0.5
            out.append({
                "type": dtype,
                "action": action,
                "target": target,
                "route_to": route,
                "evidence_span": {"sentence_id": sid, "text": text},
                "confidence": conf,
                "reason": str(item.get("reason") or ""),
            })
        return out
