from __future__ import annotations
from prompt_loader import load_prompt, load_template

import logging
import re
from typing import Any, Dict, List

from knowledge import get_tactic_id, get_tactic_name

logger = logging.getLogger(__name__)

_TPL_SPLIT = load_template('identifier/split')
_TPL_LABEL = load_template('identifier/label')
_TPL_SIMPLE = load_template('identifier/simple')
_TPL_ADJUDICATE = load_template('identifier/adjudicate')

SPLIT_SYSTEM_PROMPT = _TPL_SPLIT['system']
LABEL_SYSTEM_PROMPT = _TPL_LABEL['system']
ADJUDICATE_SYSTEM_PROMPT = _TPL_ADJUDICATE['system']

class CleanIdentifier:

    def __init__(self, llm, temperature: float = 0.0, extra_body=None, mode: str = "full",
                 self_consistency: int = 1):
        self.llm = llm
        self.temperature = temperature
        self.extra_body = extra_body
        self.mode = mode

        self.self_consistency = max(1, int(self_consistency))

    def identify(
        self,
        doc_id: str,
        sentences: List[str],
        full_text: str,
        neg_examples: List[Dict[str, Any]],
        pos_examples: List[Dict[str, Any]],
        feedback: List[Dict[str, Any]] | None = None,
        find_hints: List[Dict[str, Any]] | None = None,
    ) -> List[Dict[str, Any]]:
        n_sent = len(sentences)
        feedback = feedback or []
        attack_ids: List[int] = []

        if self.mode == "simple":

            raw_nodes = self._simple_identify(sentences, full_text)
        else:

            attack_ids = self._split_attack_sentences(sentences)

            fb_ids = []
            for fb in feedback:
                sid = fb.get("sentence_id")
                if isinstance(sid, int) and 0 <= sid < n_sent and sid not in attack_ids:
                    fb_ids.append(sid)
            if fb_ids:
                attack_ids = sorted(set(attack_ids) | set(fb_ids))
            if not attack_ids:
                logger.info("[%s] CleanIdentifier: no attack sentences found.", doc_id)
                return []

            if self.self_consistency > 1:
                raw_nodes = self._label_sc(
                    attack_ids, sentences, full_text, neg_examples, pos_examples, feedback, find_hints
                )
            else:
                raw_nodes = self._label_sentences(
                    attack_ids, sentences, full_text, neg_examples, pos_examples, feedback, find_hints
                )

        nodes: List[Dict[str, Any]] = []
        for item in raw_nodes:
            if not isinstance(item, dict):
                continue

            sid = item.get("sentence_id")
            if not isinstance(sid, bool) and isinstance(sid, int):
                pass
            else:

                try:
                    sid = int(sid)
                except (TypeError, ValueError):
                    continue

            if sid < 0 or sid >= n_sent:
                continue

            technique_id = item.get("technique_id")
            if not isinstance(technique_id, str) or not technique_id.strip():
                continue
            technique_id = technique_id.strip()

            technique_name = item.get("technique_name")
            if not isinstance(technique_name, str):
                technique_name = ""

            proc_in = item.get("procedure")
            if not isinstance(proc_in, dict):
                proc_in = {}
            procedure = {
                "actor": _as_str(proc_in.get("actor")),
                "action": _as_str(proc_in.get("action")),
                "object": _as_str(proc_in.get("object")),
                "purpose": _as_str(proc_in.get("purpose")),
            }

            i = len(nodes)
            tactic_id = get_tactic_id(technique_id) or ""
            tactic = get_tactic_name(tactic_id) if tactic_id else ""

            node = {
                "node_id": f"step_{i}_{technique_id}",
                "technique_id": technique_id,
                "technique_name": technique_name,
                "tactic": tactic,
                "tactic_id": tactic_id,
                "node_type": "obs",
                "explicit": True,
                "evidence_sentence_ids": [sid],
                "evidence_text": sentences[sid],
                "procedure": procedure,
                "confidence": 0.9,
            }
            nodes.append(node)

        n_before = len(nodes)
        nodes = self._dedup_same_sentence(nodes)
        logger.info(
            "[%s] CleanIdentifier: %d attack sentences -> %d nodes (同句保守去重 %d->%d).",
            doc_id, len(attack_ids), len(nodes), n_before, len(nodes),
        )
        return nodes

    _AUTO_EXEC_VERBS = (
        "启动", "自启", "释放并启动", "写入并启动", "写入并执行", "保存并执行", "落地后启动",
        "launched", "executed", "self-start", "self start", "auto-start",
    )

    _HUMAN_EXEC_HINTS = (
        "打开", "双击", "点击", "启用", "运行宏", "诱导", "收件", "受害", "用户",
        "opened", "double", "click", "enabled", "induced", "user", "victim", "recipient",
    )

    @classmethod
    def _filter_auto_user_exec(cls, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for n in nodes:
            tid = (n.get("technique_id") or "")
            if tid.startswith("T1204"):
                act = ((n.get("procedure") or {}).get("action") or "").lower()
                has_auto = any(v.lower() in act for v in cls._AUTO_EXEC_VERBS)
                has_human = any(h.lower() in act for h in cls._HUMAN_EXEC_HINTS)
                if has_auto and not has_human:
                    continue
            out.append(n)
        return out

    @staticmethod
    def _dedup_same_sentence(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        def _norm(v: Any) -> str:
            return re.sub(r"\s+", " ", str(v or "").strip().lower())

        groups: Dict[tuple, List[Dict[str, Any]]] = {}
        order: List[tuple] = []
        for n in nodes:
            sid = (n.get("evidence_sentence_ids") or [None])[0]
            parent = (n.get("technique_id") or "").strip().upper().split(".")[0]
            proc = n.get("procedure") or {}
            # Conservative: only merge nodes that share sentence + parent technique AND have
            # essentially identical action+object. Distinct steps (different action/object) are kept.
            key = (sid, parent, _norm(proc.get("action")), _norm(proc.get("object")))
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(n)
        out: List[Dict[str, Any]] = []
        for i, key in enumerate(order):
            grp = groups[key]
            if len(grp) == 1:
                rep = grp[0]
            else:
                rep = sorted(
                    grp,
                    key=lambda x: (
                        "." in (x.get("technique_id") or ""),
                        len(((x.get("procedure") or {}).get("action") or "")),
                    ),
                    reverse=True,
                )[0]
            node = dict(rep)
            node["node_id"] = f"step_{i}_{node.get('technique_id')}"
            out.append(node)
        return out

    def adjudicate_nodes(
        self,
        doc_id: str,
        sentences: List[str],
        nodes: List[Dict[str, Any]],
        challenges: List[Dict[str, Any]],
        neg_examples: List[Dict[str, Any]] | None = None,
        pos_examples: List[Dict[str, Any]] | None = None,
    ) -> List[Dict[str, Any]]:
        sentences = list(sentences or [])
        node_by_id = {str(n.get("node_id")): n for n in (nodes or [])}
        items, missing = [], []
        for c in challenges or []:
            if str(c.get("action") or "").lower() == "insert_node":
                tid = str(c.get("target") or "").strip().upper()
                sid = (c.get("evidence_span") or {}).get("sentence_id")
                if re.match(r"^T\d{4}(\.\d{3})?$", tid):
                    missing.append({"technique_id": tid, "sentence_id": sid,
                                    "reason": str(c.get("reason") or "")})
            else:
                nid = str(c.get("target") or "")
                n = node_by_id.get(nid)
                if n is not None:
                    items.append((nid, n, c))
        if not items and not missing:
            return []
        prompt = self._build_node_adjudicate_prompt(doc_id, sentences, items, missing, neg_examples)
        try:
            r = self.llm.chat(
                ADJUDICATE_SYSTEM_PROMPT, prompt, temperature=self.temperature,
                max_tokens=1536, extra_body=self.extra_body, agent="identifier_adjudicate",
            )
        except Exception as exc:
            logger.warning("[Identifier.adjudicate] %s failed: %s", doc_id, exc)
            return []

        verdicts = r if isinstance(r, list) else (r.get("verdicts") if isinstance(r, dict) else None)
        return self._normalize_node_verdicts(verdicts, {t[0] for t in items}, len(sentences))

    def _build_node_adjudicate_prompt(self, doc_id, sentences, items, missing, neg_examples) -> str:
        item_lines = []
        for nid, n, c in items:
            proc = n.get("procedure") if isinstance(n.get("procedure"), dict) else {}
            ev = "(无证据句)"
            for sid in (n.get("evidence_sentence_ids") or []):
                if isinstance(sid, int) and 0 <= sid < len(sentences):
                    ev = f"[{sid}] {sentences[sid]}"
                    break
            item_lines.append(f"### 节点 {nid}: {n.get('technique_id','')} {n.get('technique_name','')}")
            item_lines.append(f"- 五元组: actor={proc.get('actor','')} action={proc.get('action','')} "
                              f"object={proc.get('object','')} purpose={proc.get('purpose','')}")
            item_lines.append(f"- 原文证据句: {ev}")
            item_lines.append(f"- ⚠️ 验证器质疑: action={c.get('action')} conf={c.get('confidence')} 理由={str(c.get('reason') or '')[:60]}")
            item_lines.append("")
        items_block = "\n".join(item_lines) + "\n" if item_lines else "(无现有节点质疑)\n"

        miss_lines = []
        for m in missing:
            sid = m.get("sentence_id")
            sent = sentences[sid] if isinstance(sid, int) and 0 <= sid < len(sentences) else "(句号越界)"
            miss_lines.append(f"### 疑似漏抽: 技术 {m.get('technique_id')} @ 句[{sid}]: {sent}")
            miss_lines.append(f"- ⚠️ 验证器理由: {str(m.get('reason') or '')[:60]}")
            miss_lines.append("")
        miss_block = "\n".join(miss_lines) if miss_lines else "(无漏抽提议)"
        neg = _format_neg_examples(neg_examples or [])
        neg_seg = (neg + "\n\n") if neg else ""
        return (_TPL_ADJUDICATE["user"]
                .replace("{{DOC_ID}}", str(doc_id))
                .replace("{{ITEMS}}", items_block)
                .replace("{{MISSING}}", miss_block)
                .replace("{{NEG}}", neg_seg))

    @staticmethod
    def _normalize_node_verdicts(raw: Any, valid_ids: set, n_sent: int = 0) -> List[Dict[str, Any]]:
        import re
        out, seen = [], set()
        if not isinstance(raw, list):
            return out
        tech_re = re.compile(r"^[Tt]\d{4}(\.\d{3})?$")
        for it in raw:
            if not isinstance(it, dict):
                continue
            verdict = str(it.get("verdict") or "").strip().lower()

            if verdict == "add":
                tid = str(it.get("technique_id") or it.get("new_technique_id") or "").strip().upper()
                if not tech_re.match(tid):
                    continue
                try:
                    sid = int(it.get("sentence_id"))
                except (TypeError, ValueError):
                    continue
                if not (0 <= sid < n_sent):
                    continue
                key = ("add", tid, sid)
                if key in seen:
                    continue
                seen.add(key)
                out.append({"verdict": "add", "technique_id": tid, "sentence_id": sid,
                            "reason": str(it.get("reason") or "")})
                continue
            nid = str(it.get("node_id") or it.get("target") or "").strip()
            if nid not in valid_ids or nid in seen:
                continue
            if verdict not in ("keep", "remove", "retag"):
                continue
            v = {"node_id": nid, "verdict": verdict, "reason": str(it.get("reason") or "")}
            if verdict == "retag":
                new_tid = str(it.get("new_technique_id") or "").strip().upper()
                if not tech_re.match(new_tid):
                    continue
                v["new_technique_id"] = new_tid
            seen.add(nid)
            out.append(v)
        return out

    def _split_attack_sentences(self, sentences: List[str]) -> List[int]:
        lines = [f"[{i}] {s}" for i, s in enumerate(sentences)]
        user_prompt = _TPL_SPLIT["user"].replace("{{SENTENCES}}", "\n".join(lines))

        result = self.llm.chat(
            SPLIT_SYSTEM_PROMPT,
            user_prompt,
            temperature=self.temperature,
            max_tokens=4096,
            extra_body=self.extra_body,
            agent="identifier_split",
        )

        n_sent = len(sentences)
        seen = set()
        ordered_ids: List[int] = []
        if isinstance(result, dict):
            items = result.get("attack_sentences")
            if isinstance(items, list):
                for it in items:
                    if not isinstance(it, dict):
                        continue
                    sid = it.get("sentence_id")
                    if isinstance(sid, bool) or not isinstance(sid, int):
                        try:
                            sid = int(sid)
                        except (TypeError, ValueError):
                            continue
                    if sid < 0 or sid >= n_sent:
                        continue
                    if sid in seen:
                        continue
                    seen.add(sid)
                    ordered_ids.append(sid)
        return ordered_ids

    def _label_sentences(
        self,
        attack_ids: List[int],
        sentences: List[str],
        full_text: str,
        neg_examples: List[Dict[str, Any]],
        pos_examples: List[Dict[str, Any]],
        feedback: List[Dict[str, Any]] | None = None,
        find_hints: List[Dict[str, Any]] | None = None,
    ) -> List[Dict[str, Any]]:

        hit_lines = []
        for sid in attack_ids:
            hit_lines.append(f"[{sid}] {sentences[sid]}")
        hits_block = "\n".join(hit_lines)

        pos_block = _format_pos_examples(pos_examples)
        neg_block = _format_neg_examples(neg_examples)
        fb_block = _format_feedback(feedback, sentences)

        user_prompt = (
            _TPL_LABEL["user"]
            .replace("{{HITS}}", hits_block)
            .replace("{{FULL_TEXT}}", full_text)
            .replace("{{FEEDBACK}}", "\n\n" + fb_block if fb_block else "")
        )

        sys_prompt = LABEL_SYSTEM_PROMPT
        ex_blocks = [b for b in (pos_block, neg_block) if b]
        if ex_blocks:
            sys_prompt = (
                LABEL_SYSTEM_PROMPT
                + "\n\n## 针对本报告检索到的相关案例 (命中其场景的, 必须照它标注/改正, 不得忽略):\n"
                + "\n\n".join(ex_blocks)
            )

        if find_hints:
            hb = ["\n\n## 易漏抽技术·核对提醒 (原文疑似存在但常被漏抽; 逐条对照判据与原文: 符合才标注并给证据句, 不符合就忽略——不要凑数):"]
            for h in find_hints:
                hb.append("- [核对 %s] %s" % ("/".join(h.get("tech", []) or []), str(h.get("hint", ""))))
            sys_prompt += "\n".join(hb)

        result = self.llm.chat(
            sys_prompt,
            user_prompt,
            temperature=self.temperature,
            max_tokens=4096,
            extra_body=self.extra_body,
            agent="identifier_label",
        )

        if isinstance(result, dict):
            nodes = result.get("nodes")
            if isinstance(nodes, list):
                return nodes
        return []

    def _label_sc(self, attack_ids, sentences, full_text, neg_examples, pos_examples, feedback, find_hints=None):
        k = self.self_consistency
        runs = [self._label_sentences(attack_ids, sentences, full_text,
                                      neg_examples, pos_examples, feedback, find_hints) or [] for _ in range(k)]
        votes, rep = {}, {}
        for run in runs:
            seen = set()
            for n in run:
                if not isinstance(n, dict):
                    continue
                m = re.search(r"T\d{4}", str(n.get("technique_id") or ""))
                pt = m.group(0) if m else ""
                key = (n.get("sentence_id"), pt)
                if not pt or key in seen:
                    continue
                seen.add(key)
                votes[key] = votes.get(key, 0) + 1
                if key not in rep or len(str(n.get("procedure"))) > len(str(rep[key].get("procedure"))):
                    rep[key] = n
        thresh = (k + 1) // 2
        return [rep[key] for key, c in votes.items() if c >= thresh]

    def _simple_identify(self, sentences: List[str], full_text: str) -> List[Dict[str, Any]]:
        lines = [f"[{i}] {s}" for i, s in enumerate(sentences)]
        user_prompt = _TPL_SIMPLE["user"].replace("{{SENTENCES}}", "\n".join(lines))
        result = self.llm.chat(
            _TPL_SIMPLE["system"],
            user_prompt,
            temperature=self.temperature,
            max_tokens=4096,
            extra_body=self.extra_body,
            agent="identifier_simple",
        )
        if isinstance(result, dict):
            nodes = result.get("nodes")
            if isinstance(nodes, list):
                return nodes
        return []

def _as_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    return str(v)

def _format_feedback(feedback: List[Dict[str, Any]] | None, sentences: List[str]) -> str:
    if not feedback:
        return ""
    out = ["## 上一轮验证发现的问题 (请针对性修正: 补出漏掉的动作 / 改正选错的技术):"]
    for fb in feedback:
        if not isinstance(fb, dict):
            continue
        sid = fb.get("sentence_id")
        issue = _as_str(fb.get("issue") or fb.get("reason"))
        loc = ""
        if isinstance(sid, int) and 0 <= sid < len(sentences):
            loc = f"句[{sid}] \"{sentences[sid]}\": "
        elif sid is not None:
            loc = f"句[{sid}]: "
        out.append(f"- {loc}{issue}")
    if len(out) == 1:
        return ""
    return "\n".join(out)

def _format_pos_examples(pos_examples: List[Dict[str, Any]]) -> str:
    if not pos_examples:
        return ""
    out = ["## 正例 (句子 -> 正确技术与 5 元组, 用于校准):"]
    for ex in pos_examples:
        if not isinstance(ex, dict):
            continue
        sent = _as_str(ex.get("sentence"))
        tid = _as_str(ex.get("technique_id"))
        proc = ex.get("procedure")
        proc_str = ""
        if isinstance(proc, dict):
            proc_str = (
                f"actor={_as_str(proc.get('actor'))}, "
                f"action={_as_str(proc.get('action'))}, "
                f"object={_as_str(proc.get('object'))}, "
                f"purpose={_as_str(proc.get('purpose'))}"
            )
        note = _as_str(ex.get("note"))
        line = f"- 句子: {sent}\n  -> 技术: {tid}; 5元组: {{{proc_str}}}"
        if note:
            line += f"; 备注: {note}"
        out.append(line)
    if len(out) == 1:
        return ""
    return "\n".join(out)

def _format_neg_examples(neg_examples: List[Dict[str, Any]]) -> str:
    if not neg_examples:
        return ""
    out = ["## 反例 (避免重犯; 命中时按 correct_mapping 纠偏):"]
    for ex in neg_examples:
        if not isinstance(ex, dict):
            continue
        pattern = _as_str(ex.get("pattern"))
        wrong = _as_str(ex.get("wrong_mapping"))
        correct = _as_str(ex.get("correct_mapping"))
        reason = _as_str(ex.get("reason"))
        etype = _as_str(ex.get("error_type"))
        line = "- "
        if pattern:
            line += f"模式: {pattern}; "
        if wrong:
            line += f"错误映射: {wrong} -> "
        if correct:
            line += f"正确映射: {correct}; "
        if reason:
            line += f"原因: {reason}; "
        if etype:
            line += f"错误类型: {etype}"
        out.append(line.rstrip("; ").rstrip())
    if len(out) == 1:
        return ""
    return "\n".join(out)
