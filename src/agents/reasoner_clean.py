from __future__ import annotations
from prompt_loader import load_template

import logging
from typing import Any

from knowledge import get_tactic_id, get_tactic_name

logger = logging.getLogger(__name__)

_LEGAL_RELATIONS = {"precedes", "enables", "causes", "uses"}

_BIG = 10 ** 9

_HUB_TECH_PREFIXES = ("T1055",)
_MEM_VERBS = (
    "inject", "injected into", "loaded into memory", "load into memory",
    "mapped into", "reflectively loaded", "reflective load", "reflectively load",
    "process hollow", "hollowing", "payload launch", "launched in memory",
    "注入", "装入内存", "载入内存", "加载到内存", "反射加载", "反射式加载",
    "进程镂空", "进程空洞", "内存执行",
)

EXPLICIT_HUB_PREFIXES = ("T1059", "T1105", "T1055", "T1021", "T1219")

_EXPLICIT_CHUNK = 60

def _parent_tech(t: str) -> str:
    import re
    m = re.search(r"T\d{4}", str(t or ""))
    return m.group(0) if m else ""

def _is_hub(tech: str) -> bool:
    return _parent_tech(tech) in EXPLICIT_HUB_PREFIXES

def _is_exfil(tech: str) -> bool:
    return get_tactic_id(_parent_tech(tech)) == "TA0010"

def _is_collection_source(tech: str) -> bool:
    return get_tactic_id(_parent_tech(tech)) in ("TA0009", "TA0007")

REASONER_ADJUDICATE_SYSTEM = load_template('reasoner/adjudicate')['system']

class CleanReasoner:

    def __init__(self, llm, temperature: float = 0.0, extra_body: dict | None = None,
                 self_consistency: int = 1, require_anchor: bool = True,
                 hub_priority: bool = False, reattach_hub: bool = False,
                 mode: str = "full"):
        self.llm = llm
        self.temperature = temperature
        self.extra_body = extra_body

        self.self_consistency = max(1, int(self_consistency))

        self.require_anchor = bool(require_anchor)

        self.hub_priority = bool(hub_priority)

        self.reattach_hub = bool(reattach_hub)
        self.mode = mode
        self._feedback: list[dict] = []

    def reason(
        self,
        doc_id: str,
        sentences: list[str],
        nodes: list[dict],
        transition_priors: dict,
        neg_examples: list[dict],
        feedback: list[dict] | None = None,
        explicit_rules: list[dict] | None = None,
        implicit_rules: list[dict] | None = None,
    ) -> dict:
        if self.mode == "simple":
            return self._simple_reason(doc_id, sentences, nodes)
        nodes = list(nodes or [])
        sentences = list(sentences or [])
        transition_priors = transition_priors or {}
        neg_examples = neg_examples or []
        self._feedback = feedback or []
        self._explicit_rules = explicit_rules or []
        self._implicit_rules = implicit_rules or []

        rho_by_id: dict[str, int] = {}
        order_by_id: dict[str, int] = {}
        node_by_id: dict[str, dict] = {}
        for idx, n in enumerate(nodes):
            nid = n.get("node_id")
            if not nid:
                continue
            sids = n.get("evidence_sentence_ids") or []
            ints = [int(s) for s in sids if _is_int(s)]
            rho_by_id[nid] = min(ints) if ints else idx
            order_by_id[nid] = idx
            node_by_id[nid] = n

        ordered = sorted(
            [nid for nid in node_by_id],
            key=lambda nid: (rho_by_id[nid], order_by_id[nid]),
        )

        k = self.self_consistency
        runs: list[tuple[list, list]] = []
        for _ in range(k):

            explicit_edges = self._call_explicit(
                doc_id, sentences, ordered, node_by_id, rho_by_id, neg_examples
            )

            hidden_deps = self._call_implicit_existence(
                doc_id, sentences, ordered, node_by_id, rho_by_id, explicit_edges, neg_examples
            )

            new_inf_nodes, implicit_edges = self._call_implicit_intermediate(
                doc_id, sentences, node_by_id, rho_by_id, hidden_deps, transition_priors
            )
            runs.append((list(new_inf_nodes), list(explicit_edges) + list(implicit_edges)))

        if k == 1:
            new_inf_nodes, all_edges = runs[0]
        else:
            new_inf_nodes, all_edges = self._majority_vote(runs, k)
            logger.info(
                "[Reasoner] self-consistency k=%d -> %d edge(s) kept by majority", k, len(all_edges)
            )

        if self.reattach_hub:
            all_inf_by_id = {str(n.get("node_id")): n for n in new_inf_nodes}
            hub_lookup = dict(node_by_id)
            hub_lookup.update(all_inf_by_id)
            hub_rho = dict(rho_by_id)
            for nid, n in all_inf_by_id.items():
                ints = [int(s) for s in (n.get("evidence_sentence_ids") or []) if _is_int(s)]
                hub_rho[nid] = min(ints) if ints else _BIG
            all_edges = self._reattach_to_hub(all_edges, hub_lookup, hub_rho, ordered)

        all_nodes = list(nodes) + list(new_inf_nodes)

        # W4: final holistic completion/repair pass over the WHOLE graph. Additive —
        # it starts from the existing all_edges and returns a revised COMPLETE edge set
        # that REPLACES them, unless the call looks like a failure (guarded below).
        all_edges = self._holistic_complete(
            doc_id, sentences, all_nodes, all_edges, transition_priors
        )
        # The holistic pass REPLACES the edge set, so an inferred bridge node it chose
        # not to re-link would be left orphaned. Drop only such orphaned INFERRED nodes
        # (observed/real-technique nodes are always kept) to honor the "no orphans" goal.
        all_nodes = self._prune_orphan_inferred(all_nodes, all_edges)

        main_path = [
            nid
            for nid in ordered
            if (node_by_id[nid].get("node_type") or "obs") == "obs"
        ]
        return {"nodes": all_nodes, "edges": all_edges, "main_path": main_path}

    @staticmethod
    def _prune_orphan_inferred(all_nodes: list[dict], all_edges: list[dict]) -> list[dict]:
        """Drop INFERRED nodes (node_type != 'obs') left with degree 0 after the holistic
        edge replacement. An inferred node exists solely to bridge a dependency; an
        unconnected one is redundant and would otherwise surface as an isolated node. If
        the holistic guard kept the original edges, inferred bridges keep their s->inf->d
        edges and nothing is pruned. Observed nodes are never touched here."""
        deg: dict[str, int] = {}
        for e in all_edges:
            deg[str(e.get("src"))] = deg.get(str(e.get("src")), 0) + 1
            deg[str(e.get("dst"))] = deg.get(str(e.get("dst")), 0) + 1
        kept, dropped = [], 0
        for n in all_nodes:
            inferred = (n.get("node_type") or "obs") != "obs"
            if inferred and deg.get(str(n.get("node_id")), 0) == 0:
                dropped += 1
                continue
            kept.append(n)
        if dropped:
            logger.info("[Reasoner] holistic: 丢弃 %d 个全图修订后未被连接的孤立推断桥接节点", dropped)
        return kept

    def _holistic_complete(
        self,
        doc_id: str,
        sentences: list[str],
        all_nodes: list[dict],
        all_edges: list[dict],
        transition_priors: dict,
    ) -> list[dict]:
        """W4 holistic completion: one whole-graph DeepSeek pass that COMPLETES and
        REPAIRS the current edge set into a single coherent connected DAG. Returns the
        revised COMPLETE edge set (it REPLACES all_edges in reason()). Run exactly ONCE
        regardless of self_consistency k — a single whole-graph revision over the
        already-(vote-)settled base edges keeps cost bounded and is deterministic at
        temperature 0.0. Guard: if the validated result has <50% as many edges as the
        input (a likely truncation / parse failure), the ORIGINAL edges are kept."""
        node_by_id = {str(n.get("node_id")): n for n in all_nodes if n.get("node_id")}
        if len(node_by_id) < 2:
            return all_edges

        def _rho(n: dict) -> int:
            ints = [int(s) for s in (n.get("evidence_sentence_ids") or []) if _is_int(s)]
            return min(ints) if ints else _BIG

        rho_by_id = {nid: _rho(n) for nid, n in node_by_id.items()}
        ordered = sorted(node_by_id, key=lambda nid: (rho_by_id[nid], nid))

        try:
            t = load_template('reasoner/holistic')
        except Exception as exc:
            logger.warning("[Reasoner] holistic template load failed: %s; keep original edges", exc)
            return all_edges

        node_lines: list[str] = []
        for nid in ordered:
            n = node_by_id[nid]
            proc = n.get("procedure") if isinstance(n.get("procedure"), dict) else {}
            ao = ("%s %s" % (proc.get("action", ""), proc.get("object", ""))).strip()
            rho = rho_by_id[nid]
            sid_str = str(rho) if rho < _BIG else "?"
            node_lines.append("%s | %s | %s | sent[%s] | %s" % (
                nid, n.get("technique_id", ""), n.get("tactic", ""), sid_str, ao))
        sent_lines = ["[%d] %s" % (i, s) for i, s in enumerate(sentences)]
        prior_lines = self._render_priors_top3(ordered, node_by_id, transition_priors or {})
        edge_lines = ["%s -> %s" % (str(e.get("src")), str(e.get("dst"))) for e in all_edges]

        user_prompt = (t["user"]
                       .replace("{{DOC_ID}}", doc_id)
                       .replace("{{NODES}}", "\n".join(node_lines))
                       .replace("{{SENTENCES}}", "\n".join(sent_lines))
                       .replace("{{PRIORS}}", "\n".join(prior_lines) or "(无)")
                       .replace("{{CURRENT_EDGES}}", "\n".join(edge_lines) or "(空)"))
        try:
            result = self.llm.chat(
                t["system"], user_prompt, temperature=self.temperature,
                max_tokens=8192, extra_body=self.extra_body, agent="reasoner_holistic")
        except Exception as exc:
            logger.warning("[Reasoner] holistic completion call failed: %s; keep original edges", exc)
            return all_edges

        n_sent = len(sentences)
        validated: list[dict] = []
        seen: set[tuple[str, str]] = set()
        for e in self._rget(result, "edges"):
            if not isinstance(e, dict):
                continue
            s, d = str(e.get("src") or ""), str(e.get("dst") or "")
            if s not in node_by_id or d not in node_by_id or s == d:
                continue
            # prefer src sentence <= dst sentence: orient each edge along the narrative
            # (92% of gold deps are forward, local), consistent with the base stages.
            if rho_by_id.get(s, _BIG) > rho_by_id.get(d, _BIG):
                s, d = d, s
            if (s, d) in seen:
                continue
            seen.add((s, d))
            relation = e.get("relation")
            if relation not in _LEGAL_RELATIONS:
                relation = "enables"
            sid = e.get("evidence_sentence_id")
            if _is_int(sid) and 0 <= int(sid) < n_sent:
                sids = [int(sid)]
            else:
                sids = self._merge_sids(node_by_id[s].get("evidence_sentence_ids"),
                                        node_by_id[d].get("evidence_sentence_ids"))
            validated.append({
                "src": s,
                "dst": d,
                "relation": relation,
                "evidence_type": "implicit",
                "confidence": 0.8,
                "evidence_sentence_ids": sids,
                "evidence_text": self._join_evidence(
                    node_by_id[s].get("evidence_text"), node_by_id[d].get("evidence_text")),
                "reason": str(e.get("reason") or ""),
            })

        if len(validated) < 0.5 * len(all_edges):
            logger.info(
                "[Reasoner] holistic: 返回 %d 条 < 输入 %d 的 50%%, 视为失败, 保留原边集",
                len(validated), len(all_edges))
            return all_edges
        logger.info(
            "[Reasoner] holistic completion: 输入 %d 边 -> 输出 %d 边 (全图修订+补全)",
            len(all_edges), len(validated))
        return validated

    def _simple_reason(self, doc_id: str, sentences: list[str], nodes: list[dict]) -> dict:
        nodes = list(nodes or [])
        valid = {str(n.get("node_id")) for n in nodes}
        lines = []
        for n in nodes:
            proc = n.get("procedure") or {}
            ao = ("%s %s" % (proc.get("action", ""), proc.get("object", ""))).strip()
            sent = ""
            for sid in (n.get("evidence_sentence_ids") or []):
                if _is_int(sid) and 0 <= int(sid) < len(sentences):
                    sent = sentences[int(sid)]
                    break
            lines.append("%s | %s | %s | %s" % (n.get("node_id"), n.get("technique_id", ""), ao, sent[:90]))
        tpl = load_template('reasoner/simple')
        user_prompt = tpl['user'].replace("{{NODES}}", "\n".join(lines))
        result = self.llm.chat(
            tpl['system'], user_prompt,
            temperature=self.temperature, max_tokens=4096,
            extra_body=self.extra_body, agent="reasoner_simple",
        )
        raw_edges = result.get("edges", []) if isinstance(result, dict) else []
        edges, seen = [], set()
        for e in (raw_edges or []):
            s, d = str(e.get("src", "")), str(e.get("dst", ""))
            if s in valid and d in valid and s != d and (s, d) not in seen:
                seen.add((s, d))
                edges.append({"src": s, "dst": d, "relation": e.get("relation", "enables"),
                              "evidence_type": "explicit"})
        main_path = [str(n.get("node_id")) for n in nodes if (n.get("node_type") or "obs") == "obs"]
        return {"nodes": nodes, "edges": edges, "main_path": main_path}

    @staticmethod
    def _majority_vote(runs: list[tuple[list, list]], k: int) -> tuple[list, list]:
        threshold = (k // 2) + 1
        edge_count: dict[tuple[str, str], int] = {}
        edge_repr: dict[tuple[str, str], dict] = {}
        inf_by_id: dict[str, dict] = {}
        for inf_nodes, edges in runs:
            for n in inf_nodes:
                inf_by_id[str(n.get("node_id"))] = n
            seen_in_run: set[tuple[str, str]] = set()
            for e in edges:
                key = (str(e.get("src")), str(e.get("dst")))
                if key in seen_in_run:
                    continue
                seen_in_run.add(key)
                edge_count[key] = edge_count.get(key, 0) + 1
                edge_repr.setdefault(key, e)
        kept_edges = [edge_repr[key] for key, c in edge_count.items() if c >= threshold]
        ref_ids: set[str] = set()
        for e in kept_edges:
            ref_ids.add(str(e.get("src")))
            ref_ids.add(str(e.get("dst")))
        kept_inf = [n for nid, n in inf_by_id.items() if nid in ref_ids]
        return kept_inf, kept_edges

    @staticmethod
    def _is_hub(n: dict) -> bool:
        if not isinstance(n, dict):
            return False
        tid = str(n.get("technique_id") or "")
        parent = tid.split(".")[0]
        if parent in _HUB_TECH_PREFIXES or any(tid.startswith(p) for p in _HUB_TECH_PREFIXES):
            return True
        proc = n.get("procedure") or {}
        if isinstance(proc, dict):
            blob = f"{proc.get('action', '')} {proc.get('object', '')}".lower()
        else:
            blob = str(proc).lower()
        return any(v in blob for v in _MEM_VERBS)

    def _reattach_to_hub(
        self,
        edges: list[dict],
        node_by_id: dict[str, dict],
        rho_by_id: dict[str, int],
        ordered: list[str],
    ) -> list[dict]:
        hubs = [nid for nid in ordered if self._is_hub(node_by_id.get(nid, {}))]
        if not hubs:
            return edges
        out: list[dict] = []
        seen: set[tuple[str, str]] = set()
        reattached = 0
        for e in edges:
            src, dst = e.get("src"), e.get("dst")
            rs = rho_by_id.get(src, _BIG)
            rd = rho_by_id.get(dst, _BIG)
            new_src = src
            if src in node_by_id and not self._is_hub(node_by_id[src]):
                best_h = None
                best_rh = -1
                for h in hubs:
                    if h == src or h == dst:
                        continue
                    rh = rho_by_id.get(h, _BIG)
                    if rs < rh < rd and 0 <= rd - rh <= 8:
                        if rh > best_rh:
                            best_h, best_rh = h, rh
                if best_h is not None:
                    new_src = best_h
            if new_src != src:
                e = dict(e)
                e["src"] = new_src
                if not e.get("anchor_type"):
                    e["anchor_type"] = "actor"
                reattached += 1
            key = (str(e.get("src")), str(e.get("dst")))
            if e.get("src") == e.get("dst") or key in seen:
                continue
            seen.add(key)
            out.append(e)
        if reattached:
            logger.info("[Reasoner] R2-2 reattach_hub: re-attached %d edge(s) to exec hub", reattached)
        return out

    @staticmethod
    def _rget(result, key):
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return result.get(key) or []
        return []

    def _gen_candidates(self, ordered, node_by_id, rho_by_id):
        def _tech(nid):
            return str(node_by_id[nid].get("technique_id") or "")
        cands: list[tuple[str, str]] = []
        for i, a in enumerate(ordered):
            ra = rho_by_id[a]
            a_hub = _is_hub(_tech(a))
            a_src = _is_collection_source(_tech(a))
            for b in ordered[i + 1:]:
                rb = rho_by_id[b]
                if a == b or ra > rb:
                    continue
                if (rb - ra) <= 3 or a_hub or (a_src and _is_exfil(_tech(b))):
                    cands.append((a, b))
        return cands

    def _call_explicit(
        self,
        doc_id: str,
        sentences: list[str],
        ordered: list[str],
        node_by_id: dict[str, dict],
        rho_by_id: dict[str, int],
        neg_examples: list[dict],
        only_pairs: set | None = None,
    ) -> list[dict]:

        candidates = self._gen_candidates(ordered, node_by_id, rho_by_id)

        if only_pairs is not None:
            candidates = [c for c in candidates if c in only_pairs]

        if not candidates:
            return []

        system_prompt = self._explicit_system(neg_examples)

        raw: list = []
        for ci in range(0, len(candidates), _EXPLICIT_CHUNK):
            chunk = candidates[ci:ci + _EXPLICIT_CHUNK]
            user_prompt = self._explicit_user(doc_id, ordered, node_by_id, rho_by_id, chunk)
            try:
                result = self.llm.chat(
                    system_prompt,
                    user_prompt,
                    temperature=self.temperature,
                    max_tokens=8192,
                    extra_body=self.extra_body,
                    agent="reasoner_explicit",
                )
            except Exception as exc:
                logger.warning("[Reasoner] explicit-edge call (chunk %d) failed: %s", ci // _EXPLICIT_CHUNK, exc)
                continue
            raw.extend(self._rget(result, "explicit_edges"))
        if len(raw) < len(candidates):
            logger.info("[Reasoner] explicit: 模型返回 %d 条记录 < %d 候选 (疑似截断/漏判 %d 对)",
                        len(raw), len(candidates), len(candidates) - len(raw))
        edges: list[dict] = []
        seen: set[tuple[str, str]] = set()
        n_anchor_dropped = 0
        for e in raw:
            if not isinstance(e, dict):
                continue
            try:
                connected = int(e.get("connected", 0))
            except (TypeError, ValueError):
                connected = 0
            if connected != 1:
                continue

            anchor = str(e.get("anchor_type") or "").strip().lower()
            reason = str(e.get("reason") or "").strip()
            valid_anchors = ("artifact", "state", "actor", "chain")
            if self.require_anchor and (anchor not in valid_anchors or not reason):
                n_anchor_dropped += 1
                continue
            src = e.get("src")
            dst = e.get("dst")

            if src not in node_by_id or dst not in node_by_id:
                continue
            if src == dst:
                continue
            if (src, dst) in seen:
                continue
            seen.add((src, dst))
            relation = e.get("relation")
            if relation not in _LEGAL_RELATIONS:
                relation = "enables"
            sids = self._merge_sids(
                e.get("evidence_sentence_ids"),
                node_by_id[src].get("evidence_sentence_ids"),
                node_by_id[dst].get("evidence_sentence_ids"),
            )
            evidence_text = self._join_evidence(
                node_by_id[src].get("evidence_text"),
                node_by_id[dst].get("evidence_text"),
            )
            edge = {
                "src": src,
                "dst": dst,
                "relation": relation,
                "evidence_type": "explicit",
                "evidence_sentence_ids": sids,
                "evidence_text": evidence_text,
                "confidence": 0.85,
                "anchor_type": anchor,
                "reason": reason,
            }
            edges.append(edge)
        logger.info(
            "[Reasoner] explicit: %d candidate pair(s) -> %d explicit edge(s) (%d dropped by anchor gate)",
            len(candidates), len(edges), n_anchor_dropped,
        )
        return edges

    def _explicit_system(self, neg_examples: list[dict]) -> str:
        t = load_template('reasoner/explicit')
        parts = [t['head']]
        if self.require_anchor:
            parts.append(t['anchor'])
        parts.append(t['mid'])
        if self.hub_priority:
            parts.append(t['hub'])
        parts.extend(self._render_neg(neg_examples))
        parts.extend(self._render_rules(getattr(self, "_explicit_rules", None), "显式边连接规则"))
        parts.append(t['output_anchor'] if self.require_anchor else t['output_plain'])
        return "\n".join(parts)

    def _explicit_user(
        self,
        doc_id: str,
        ordered: list[str],
        node_by_id: dict[str, dict],
        rho_by_id: dict[str, int],
        candidates: list[tuple[str, str]],
    ) -> str:
        t = load_template('reasoner/explicit')
        nodes = "\n".join(self._render_nodes(ordered, node_by_id, rho_by_id))
        cands = "\n".join(
            "[%d] src=%s (%s)  ->  dst=%s (%s)" % (
                k, a, node_by_id[a].get("technique_id", ""), b, node_by_id[b].get("technique_id", ""))
            for k, (a, b) in enumerate(candidates))
        fb = self._render_feedback()
        fb_block = ("\n".join(fb) + "\n\n") if fb else ""
        return (t["user"].replace("{{DOC_ID}}", doc_id).replace("{{NODES}}", nodes)
                .replace("{{CANDIDATES}}", cands).replace("{{FEEDBACK}}", fb_block))

    def _call_implicit_existence(self, doc_id, sentences, ordered, node_by_id, rho_by_id,
                                 explicit_edges, neg_examples):
        epairs = set()
        for e in explicit_edges:
            epairs.add((e.get("src"), e.get("dst")))
            epairs.add((e.get("dst"), e.get("src")))
        groups = []
        for a in ordered:
            cands = [b for b in ordered
                     if b != a and rho_by_id.get(b, _BIG) >= rho_by_id.get(a, _BIG)
                     and (a, b) not in epairs]
            if cands:
                groups.append((a, cands))
        if not groups:
            return []
        try:
            r = self.llm.chat(
                self._impl_exist_system(neg_examples),
                self._impl_exist_user(doc_id, sentences, groups, node_by_id),
                temperature=self.temperature, max_tokens=4096,
                extra_body=self.extra_body, agent="reasoner_impl_exist")
        except Exception as exc:
            logger.warning("[Reasoner] implicit-existence call failed: %s", exc)
            return []
        out, seen = [], set()
        for h in self._rget(r, "hidden_edges"):
            if not isinstance(h, dict):
                continue
            s, d = h.get("src"), h.get("dst")
            if (s in node_by_id and d in node_by_id and s != d
                    and rho_by_id.get(s, _BIG) <= rho_by_id.get(d, _BIG)
                    and (s, d) not in epairs and (s, d) not in seen):
                seen.add((s, d))
                out.append({"src": s, "dst": d, "reason": str(h.get("reason") or "")})
        logger.info("[Reasoner] implicit-existence: %d hidden dep(s) over %d source group(s)",
                    len(out), len(groups))
        return out

    def _call_implicit_intermediate(self, doc_id, sentences, node_by_id, rho_by_id,
                                    hidden_deps, transition_priors):
        if not hidden_deps:
            return [], []
        try:
            r = self.llm.chat(
                self._impl_mid_system(),
                self._impl_mid_user(doc_id, hidden_deps, node_by_id, transition_priors),
                temperature=self.temperature, max_tokens=3072,
                extra_body=self.extra_body, agent="reasoner_impl_mid")
        except Exception as exc:
            logger.warning("[Reasoner] implicit-intermediate call failed: %s", exc)
            r = {}
        res = {}
        for x in self._rget(r, "results"):
            if isinstance(x, dict) and x.get("src") and x.get("dst"):
                res[(x.get("src"), x.get("dst"))] = x
        new_nodes, new_edges, made_inf, added = [], [], set(), set()
        for dep in hidden_deps:
            s, d = dep["src"], dep["dst"]
            if s not in node_by_id or d not in node_by_id:
                continue
            x = res.get((s, d), {})
            sids = self._merge_sids(node_by_id[s].get("evidence_sentence_ids"),
                                    node_by_id[d].get("evidence_sentence_ids"))
            if str(x.get("kind") or "direct").lower() == "bridge":
                tid = x.get("intermediate_technique_id")
                ends = {node_by_id[s].get("technique_id"), node_by_id[d].get("technique_id")}
                if _is_legal_technique(tid) and str(tid).strip() not in ends:
                    tid = str(tid).strip()
                    inf_id = f"inf_{s}_{d}_{tid.replace('.', '')}"
                    if inf_id not in made_inf:
                        made_inf.add(inf_id)
                        tac = get_tactic_id(tid) or ""
                        new_nodes.append({
                            "node_id": inf_id, "technique_id": tid, "technique_name": "",
                            "tactic": get_tactic_name(tac) if tac else "Unknown", "tactic_id": tac,
                            "node_type": "inf", "explicit": False, "evidence_sentence_ids": sids,
                            "evidence_text": "", "procedure": {}, "confidence": 0.7})
                        new_edges.append(self._implicit_edge(s, inf_id, sids))
                        new_edges.append(self._implicit_edge(inf_id, d, sids))
                    continue
            if (s, d) not in added:
                added.add((s, d))
                new_edges.append(self._implicit_edge(s, d, sids))
        logger.info("[Reasoner] implicit-intermediate: %d dep -> %d inf node, %d implicit edge",
                    len(hidden_deps), len(new_nodes), len(new_edges))
        return new_nodes, new_edges

    def _impl_exist_system(self, neg_examples):
        t = load_template('reasoner/implicit_exist')
        parts = [t['head']]
        parts.extend(self._render_rules(getattr(self, "_implicit_rules", None), "隐式推理规则(该补的依赖/别脑补的乱连)"))
        parts.append(t['output'])
        return "\n".join(parts)

    def _impl_exist_user(self, doc_id, sentences, groups, node_by_id):
        t = load_template('reasoner/implicit_exist')
        glines = []
        for gi, (a, cands) in enumerate(groups):
            pa = node_by_id[a].get("procedure") if isinstance(node_by_id[a].get("procedure"), dict) else {}
            glines.append("## 子任务%d ── 源 %s (%s): actor=%s action=%s object=%s" % (
                gi, a, node_by_id[a].get('technique_id', ''), pa.get('actor', ''), pa.get('action', ''), pa.get('object', '')))
            glines.append("   源证据: %s" % self._sent_of(node_by_id[a], sentences))
            glines.append("   候选后继(逐个判与源有无隐式依赖):")
            for b in cands:
                pb = node_by_id[b].get("procedure") if isinstance(node_by_id[b].get("procedure"), dict) else {}
                glines.append("     - %s (%s): action=%s object=%s | %s" % (
                    b, node_by_id[b].get('technique_id', ''), pb.get('action', ''), pb.get('object', ''), self._sent_of(node_by_id[b], sentences)))
            glines.append("")
        return t["user"].replace("{{DOC_ID}}", doc_id).replace("{{GROUPS}}", "\n".join(glines))

    def _impl_mid_system(self):
        return load_template('reasoner/implicit_mid')['system']

    def _impl_mid_user(self, doc_id, hidden_deps, node_by_id, transition_priors):
        subparts = []
        for gi, dep in enumerate(hidden_deps):
            s, d = dep["src"], dep["dst"]
            ts, td = node_by_id[s].get("technique_id", ""), node_by_id[d].get("technique_id", "")
            succ = self._prior_succ_techs(transition_priors, ts, 3)
            subparts.append(f"## 子任务{gi} ── {s}({ts}) -> {d}({td})   依据: {dep.get('reason','')}")
            subparts.append(f"   矩阵候选(a 的常见后继 top3): {succ or '(无)'}")
            subparts.append("")
        tpl = load_template('reasoner/implicit_mid')
        return tpl['user'].replace("{{DOC_ID}}", doc_id).replace("{{SUBTASKS}}", "\n".join(subparts))

    @staticmethod
    def _sent_of(node, sentences):
        for sid in (node.get("evidence_sentence_ids") or []):
            if isinstance(sid, int) and 0 <= sid < len(sentences):
                return f"[{sid}] {sentences[sid][:90]}"
        return "(无句)"

    @staticmethod
    def _prior_succ_techs(transition_priors, tech, n=3):
        import re
        m = re.search(r"T\d{4}", str(tech or ""))
        parent = m.group(0) if m else str(tech or "")
        lst = (transition_priors or {}).get(tech) or (transition_priors or {}).get(parent) or []
        return [it.get("target") for it in lst[:n] if isinstance(it, dict)]

    def _render_feedback(self) -> list[str]:
        fb = getattr(self, "_feedback", None) or []
        if not fb:
            return []
        lines = ["## 上一轮验证发现的边问题 (请针对性修正: 补出漏掉的边 / 改正方向)"]
        for f in fb:
            if not isinstance(f, dict):
                continue
            tgt = f.get("target", "")
            issue = f.get("issue") or f.get("reason") or ""
            lines.append(f"- {tgt}: {issue}")
        return lines if len(lines) > 1 else []

    def adjudicate_edges(
        self,
        doc_id: str,
        sentences: list[str],
        graph: dict,
        challenges: list[dict],
        transition_priors: dict,
        neg_examples: list[dict] | None = None,
        edge_signals: dict | None = None,
    ) -> list[dict]:
        sentences = list(sentences or [])
        node_by_id = {str(n.get("node_id")): n for n in (graph.get("nodes") or [])}
        edge_by_key = {(str(e.get("src")), str(e.get("dst"))): e for e in (graph.get("edges") or [])}
        items: list[tuple] = []
        for c in challenges or []:
            tgt = str(c.get("target") or "")
            if "->" not in tgt:
                continue
            s, d = (x.strip() for x in tgt.split("->", 1))
            e = edge_by_key.get((s, d))
            if e is None:
                continue
            items.append((f"{s}->{d}", s, d, node_by_id.get(s, {}), node_by_id.get(d, {}), e, c))
        if not items:
            return []
        prompt = self._build_edge_adjudicate_prompt(doc_id, sentences, items, transition_priors, edge_signals)
        try:
            r = self.llm.chat(
                REASONER_ADJUDICATE_SYSTEM, prompt, temperature=self.temperature,
                max_tokens=2048, extra_body=self.extra_body, agent="reasoner_adjudicate",
            )
        except Exception as exc:
            logger.warning("[Reasoner.adjudicate] %s failed: %s", doc_id, exc)
            return []
        return self._normalize_edge_verdicts(self._rget(r, "verdicts"), {t[0] for t in items})

    def _build_edge_adjudicate_prompt(self, doc_id, sentences, items, transition_priors, edge_signals=None) -> str:
        def sent_of(n):
            for sid in (n.get("evidence_sentence_ids") or []):
                if isinstance(sid, int) and 0 <= sid < len(sentences):
                    return f"[{sid}] {sentences[sid]}"
            return "(无证据句)"

        sent_lines = [f"[{i}] {s}" for i, s in enumerate(sentences)]
        edge_lines: list[str] = []
        for tgt, s, d, sn, dn, e, c in items:
            et = str(e.get("evidence_type") or "")
            lab = "隐式implicit" if et.lower().startswith("impl") else "显式explicit"
            sp = sn.get("procedure") if isinstance(sn.get("procedure"), dict) else {}
            dp = dn.get("procedure") if isinstance(dn.get("procedure"), dict) else {}
            edge_lines.append(f"### 边 {tgt}  ({lab}, relation={e.get('relation','')})")
            edge_lines.append(f"- src {s}: {sn.get('technique_id','')} {sn.get('technique_name','')} | 动作={sp.get('action','')} | 证据句={sent_of(sn)}")
            edge_lines.append(f"- dst {d}: {dn.get('technique_id','')} {dn.get('technique_name','')} | 动作={dp.get('action','')} | 证据句={sent_of(dn)}")
            sig = (edge_signals or {}).get(tgt)
            sigtxt = ("档=%s (转移矩阵ratio=%.2f%s)" % (
                sig["tier"], sig["ratio"], ", 错边记忆命中" if sig["wrong"] else "")) if sig else "(无矩阵信号)"
            edge_lines.append(f"- 转移先验(src→dst): {self._prior_for(transition_priors, sn.get('technique_id'), dn.get('technique_id'))}")
            edge_lines.append(f"- 转移矩阵+记忆信号: {sigtxt}")
            edge_lines.append(f"- ⚠️ 验证器质疑: action={c.get('action')} conf={c.get('confidence')} 理由={str(c.get('reason') or '')[:60]}")
            if c.get("mem_hint"):
                edge_lines.append(f"- 📌 记忆判据(据此判, 别凭空裁回): {str(c.get('mem_hint'))[:200]}")
            edge_lines.append("")
        tpl = load_template('reasoner/adjudicate')
        return (tpl['user'].replace("{{DOC_ID}}", doc_id)
                .replace("{{SENTENCES}}", "\n".join(sent_lines))
                .replace("{{CHALLENGED_EDGES}}", "\n".join(edge_lines)))

    @staticmethod
    def _prior_for(transition_priors: dict, src_tid, dst_tid) -> str:
        import re

        def parent(t):
            m = re.search(r"T\d{4}", str(t or ""))
            return m.group(0) if m else str(t or "")

        sp, dp = parent(src_tid), parent(dst_tid)
        for key in (src_tid, sp):
            for it in (transition_priors or {}).get(key) or []:
                if it.get("target") == dst_tid or parent(it.get("target")) == dp:
                    return f"常见后继 prob={it.get('probability')} count={it.get('count')}"
        return "矩阵无此后继(矩阵稀疏, 仅参考, 不得单凭此删边)"

    @staticmethod
    def _normalize_edge_verdicts(raw, valid_targets: set) -> list[dict]:
        out, seen = [], set()
        if not isinstance(raw, list):
            return out
        for it in raw:
            if not isinstance(it, dict):
                continue
            edge = str(it.get("edge") or it.get("target") or "").strip()
            verdict = str(it.get("verdict") or "").strip().lower()
            if edge not in valid_targets or edge in seen:
                continue
            if verdict not in ("keep", "remove", "reverse"):
                continue
            seen.add(edge)
            out.append({"edge": edge, "verdict": verdict, "reason": str(it.get("reason") or "")})
        return out

    @staticmethod
    def _implicit_edge(src: str, dst: str, sids: list[int]) -> dict:
        return {
            "src": src,
            "dst": dst,
            "relation": "enables",
            "evidence_type": "implicit",
            "evidence_sentence_ids": list(sids),
            "evidence_text": "",
            "confidence": 0.7,
        }

    @staticmethod
    def _render_nodes(
        ordered: list[str],
        node_by_id: dict[str, dict],
        rho_by_id: dict[str, int],
    ) -> list[str]:
        lines: list[str] = []
        for nid in ordered:
            n = node_by_id[nid]
            proc = n.get("procedure") or {}
            if isinstance(proc, dict):
                tup = (
                    f"actor={proc.get('actor', '')} | action={proc.get('action', '')} | "
                    f"object={proc.get('object', '')} | purpose={proc.get('purpose', '')}"
                )
            else:
                tup = str(proc)
            rho = rho_by_id.get(nid)
            rho_str = "?" if rho is None or rho >= _BIG else str(rho)
            lines.append(
                f"- {nid} | {n.get('technique_id', '')} {n.get('technique_name', '')} "
                f"| 句号={rho_str} | {tup}"
            )
        return lines

    def _render_priors(
        self,
        explicit_edges: list[dict],
        node_by_id: dict[str, dict],
        transition_priors: dict,
    ) -> list[str]:
        keys: set[str] = set()
        for e in explicit_edges:
            for end in (e.get("src"), e.get("dst")):
                tid = node_by_id.get(end, {}).get("technique_id")
                if not tid:
                    continue
                keys.add(tid)
                parent = str(tid).split(".")[0]
                keys.add(parent)
        lines: list[str] = []
        seen: set[tuple[str, str]] = set()
        for k in keys:
            entries = transition_priors.get(k)
            if not entries:
                continue
            for ent in entries:
                if not isinstance(ent, dict):
                    continue
                tgt = ent.get("target")
                if not tgt:
                    continue
                if (k, tgt) in seen:
                    continue
                seen.add((k, tgt))
                prob = ent.get("probability")
                count = ent.get("count")
                lines.append(f"- {k} 常后继 {tgt}  prob={prob} count={count}")
        return lines

    def _render_priors_top3(
        self,
        ordered: list[str],
        node_by_id: dict[str, dict],
        transition_priors: dict,
    ) -> list[str]:
        lines: list[str] = []
        seen_keys: set[str] = set()
        for nid in ordered:
            tid = node_by_id.get(nid, {}).get("technique_id")
            if not tid:
                continue
            for key in (tid, str(tid).split(".")[0]):
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                entries = transition_priors.get(key)
                if not entries:
                    continue
                top = [e for e in entries if isinstance(e, dict) and e.get("target")][:3]
                if not top:
                    continue
                succ = ", ".join(
                    f"{e.get('target')}(p={e.get('probability')})" for e in top
                )
                lines.append(f"- {key} 最可能后继: {succ}")
        return lines

    @staticmethod
    def _render_neg(neg_examples: list[dict]) -> list[str]:
        if not neg_examples:
            return []
        lines = ["## 反例 (历史易犯错误, 避免重蹈)"]
        for w in neg_examples:
            if not isinstance(w, dict):
                continue
            lines.append(
                f"- 模式: {w.get('pattern', '')} | 错误: {w.get('wrong_mapping', '')} "
                f"-> 正确: {w.get('correct_mapping', '')} | 原因: {w.get('reason', '')}"
            )
        return lines

    @staticmethod
    def _render_rules(rules, title) -> list[str]:
        rules = rules or []
        if not rules:
            return []
        lines = ["## " + title + " (本报告命中的通用规则·软提示, 仍以原文为准):"]
        for r in rules:
            if not isinstance(r, dict):
                continue
            tag = "【该连·别漏】" if r.get("polarity") == "pos" else "【别连·常见乱连/别脑补】"
            src = "/".join(r.get("src", []) if isinstance(r.get("src"), list) else [str(r.get("src"))])
            dst = "/".join(r.get("dst", []) if isinstance(r.get("dst"), list) else [str(r.get("dst"))])
            lines.append(f"  - {src} → {dst} {tag} {r.get('hint', '')}")
        return lines if len(lines) > 1 else []

    @staticmethod
    def _merge_sids(*sid_lists: Any) -> list[int]:
        out: list[int] = []
        seen: set[int] = set()
        for lst in sid_lists:
            for s in (lst or []):
                if _is_int(s):
                    v = int(s)
                    if v not in seen:
                        seen.add(v)
                        out.append(v)
        out.sort()
        return out

    @staticmethod
    def _join_evidence(*texts: Any) -> str:
        parts = [str(t).strip() for t in texts if t and str(t).strip()]
        return " ".join(parts)

def _is_int(x: Any) -> bool:
    try:
        int(x)
        return True
    except (TypeError, ValueError):
        return False

def _is_legal_technique(tid: Any) -> bool:
    if not tid or not isinstance(tid, str):
        return False
    s = tid.strip()
    if not s or s[0] not in ("T", "t"):
        return False
    body = s[1:]
    if "." in body:
        base, sub = body.split(".", 1)
        return base.isdigit() and len(base) == 4 and sub.isdigit() and len(sub) >= 1
    return body.isdigit() and len(body) == 4
