from __future__ import annotations
import json
import logging
import os
import re
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

logger = logging.getLogger(__name__)

LAMBDA_PENALTY = 1.0
DELTA_SMOOTH = 0.01
MAX_ERROR_ENTRIES = 500
MAX_KB_PAIRS = 10000

CONTEXT_REQUIRED_TARGET_CUES: dict[str, tuple[str, ...]] = {

    "T1204": (
        "open", "opened", "opening", "click", "clicked", "clicking",
        "launch", "launched", "run", "ran", "execute", "executed",
        "enable macros", "enabled macros", "enable content",
    ),
    "T1140": (
        "decode", "decoded", "decoding", "decrypt", "decrypted", "decrypting",
        "deobfuscate", "deobfuscated", "unpack", "unpacked", "extract",
        "extracted", "rc4", "base64-decoded",
    ),
    "T1041": (
        "exfiltrate", "exfiltrated", "exfiltration", "send", "sent",
        "upload", "uploaded", "transmit", "transmitted", "transfer out",
    ),
    "T1071": (
        "http", "https", "dns", "ftp", "smtp", "imap", "pop3",
        "websocket", "application layer", "application-layer",
    ),
    "T1105": (
        "download", "downloaded", "fetch", "fetched", "retrieve", "retrieved",
        "transfer", "transferred", "get malware updates", "additional files",
    ),
    "T1573": (
        "encrypt", "encrypted", "encryption", "aes", "rsa", "ssl", "tls",
        "cryptography", "encrypted channel",
    ),

    "T1059.001": (
        "powershell", "powershell.exe", "pwsh", "ps1", "powershell payload",
        "encodedcommand", "iex ", "invoke-expression",
    ),
    "T1059.003": (
        "cmd.exe", "cmd ", "command shell", "windows command shell",
        "command prompt", "batch script", ".bat",
    ),
    "T1059.005": (
        "vbscript", "vba", "visual basic", "wscript", "cscript",
    ),
    "T1547.001": (
        "run key", "registry run", "hkcu\\software\\microsoft\\windows\\currentversion\\run",
        "hklm\\software\\microsoft\\windows\\currentversion\\run", "startup folder",
    ),
    "T1546.003": (
        "wmi event subscription", "__eventfilter", "commandlineeventconsumer",
        "wmi event consumer", "permanent event subscription",
    ),
    "T1110.003": (
        "password spraying", "password spray", "spraying",
    ),
    "T1110.001": (
        "password guessing", "brute force passwords", "guessing common passwords",
    ),
    "T1110.002": (
        "password cracking", "crack hashes", "hashcat", "john the ripper",
    ),
    "T1003.001": (
        "lsass", "lsass.exe", "lsa memory", "mimikatz", "lsadump",
    ),
    "T1003.002": (
        "sam database", "sam hive", "security account manager",
    ),
    "T1003.003": (
        "ntds.dit", "ntds database", "active directory database",
    ),
    "T1027.013": (
        "encrypted/encoded payload", "encrypted/encoded file",
        "encrypted file or information", "encrypted code",
    ),
    "T1027.009": (
        "embedded payload", "embedded shellcode", "embedded binary",
    ),
    "T1195.002": (
        "compromise software supply chain", "compromised installer",
        "compromised the software supply chain",
    ),
    "T1573.001": (
        "aes", "rc4", "symmetric key", "symmetric encryption",
    ),
    "T1573.002": (
        "rsa", "asymmetric encryption", "public key", "asymmetric cryptography",
    ),
}

def _has_context_cue(text: str, cues: tuple[str, ...]) -> bool:
    lowered = str(text or "").lower()
    for cue in cues:
        cue_lower = cue.lower()
        if not cue_lower:
            continue
        if " " in cue_lower or "-" in cue_lower:
            if cue_lower in lowered:
                return True
            continue
        if re.search(rf"(?<![a-z0-9_]){re.escape(cue_lower)}(?![a-z0-9_])", lowered):
            return True
    return False

@dataclass
class TransitionEntry:
    w_pos: float = 0.0
    w_neg: float = 0.0
    count: int = 0

    def score(self) -> float:
        return max(0.0, self.w_pos - LAMBDA_PENALTY * self.w_neg) + DELTA_SMOOTH

    def to_dict(self) -> dict:
        return {"w_pos": self.w_pos, "w_neg": self.w_neg, "count": self.count}

    @staticmethod
    def from_dict(d: dict) -> "TransitionEntry":
        return TransitionEntry(w_pos=d.get("w_pos", 0), w_neg=d.get("w_neg", 0),
                               count=d.get("count", 0))

@dataclass
class ErrorRecord:
    error_id: str
    error_type: str
    technique_id: str
    action_verbs: list[str]
    original_value: str
    corrected_value: str
    reason: str
    doc_id: str
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "error_id": self.error_id,
            "error_type": self.error_type,
            "technique_id": self.technique_id,
            "action_verbs": self.action_verbs,
            "original_value": self.original_value,
            "corrected_value": self.corrected_value,
            "reason": self.reason,
            "doc_id": self.doc_id,
            "timestamp": self.timestamp,
        }

    @staticmethod
    def from_dict(d: dict) -> "ErrorRecord":
        return ErrorRecord(**{k: d[k] for k in ErrorRecord.__dataclass_fields__ if k in d})

class MemoryEngine:

    def __init__(self, memory_dir: str):
        self.memory_dir = memory_dir
        os.makedirs(memory_dir, exist_ok=True)

        self.kb_path = os.path.join(memory_dir, "transition_kb.json")
        self.error_path = os.path.join(memory_dir, "error_memory.json")

        self.transition_kb: dict[str, dict[str, TransitionEntry]] = {}

        self.error_memory: list[ErrorRecord] = []

        self.working: dict[str, Any] = {}

        self.blackboard: list[dict[str, Any]] = []

        self.excluded_doc_ids: set[str] = set()

        self._load()

    def _load(self):

        nodes_path = os.path.join(self.memory_dir, "nodes.json")
        mem_path = os.path.join(self.memory_dir, "memory.json")
        if os.path.exists(nodes_path):
            m = json.load(open(nodes_path, encoding="utf-8")) or {}
            self.error_memory = [ErrorRecord.from_dict(e) for e in m.get("error", [])]
            logger.info("Loaded error memory(nodes.json): %d records", len(self.error_memory))
        elif os.path.exists(mem_path):
            m = json.load(open(mem_path, encoding="utf-8")) or {}
            self.error_memory = [ErrorRecord.from_dict(e) for e in (m.get("node") or {}).get("error", [])]
            logger.info("Loaded error memory(memory.json node.error): %d records", len(self.error_memory))
        elif os.path.exists(self.error_path):
            with open(self.error_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.error_memory = [ErrorRecord.from_dict(e) for e in data.get("errors", [])]
            logger.info("Loaded error memory: %d records", len(self.error_memory))

    def save(self):
        mem_path = os.path.join(self.memory_dir, "memory.json")
        if not os.path.exists(mem_path):
            return
        m = json.load(open(mem_path, encoding="utf-8")) or {}
        m.setdefault("node", {})["error"] = [e.to_dict() for e in self.error_memory[-MAX_ERROR_ENTRIES:]]
        m["updated"] = datetime.now(timezone.utc).isoformat()
        with open(mem_path, "w", encoding="utf-8") as f:
            json.dump(m, f, indent=2, ensure_ascii=False)
        logger.info("Memory saved(memory.json): node.error=%d, KB=%d(csv只读)",
                     len(self.error_memory), len(self.transition_kb))

    def _load_confirmed(self) -> dict:
        if getattr(self, "_confirmed", None) is not None:
            return self._confirmed
        self._confirmed = {}
        path = os.path.join(self.memory_dir, "transition_matrix_meta.json")
        if os.path.exists(path):
            try:
                cm = (json.load(open(path, encoding="utf-8")) or {}).get("confirmed_map") or {}
                self._confirmed = {s: [(d, float(sc)) for d, sc in lst] for s, lst in cm.items()}
            except Exception as exc:
                logger.warning("confirmed_map load failed: %s", exc)
        return self._confirmed

    def get_transition_priors(
        self,
        technique_id: str,
        top_n: int = 5,
        evidence_text: str | None = None,
    ) -> list[dict]:
        cm = self._load_confirmed()
        parent = str(technique_id).split(".")[0]
        merged: dict[str, float] = {}
        for tid in (technique_id, parent):
            for dst, sc in cm.get(tid, []):
                if sc > merged.get(dst, 0.0):
                    merged[dst] = sc
        survivors = [(dst, sc) for dst, sc in merged.items()
                     if self._transition_prior_supported_by_context(dst, evidence_text)]
        if not survivors:
            return []
        total = sum(sc for _, sc in survivors) or 1.0
        results = [{"target": dst, "probability": round(sc / total, 3), "count": 0,
                    "w_pos": sc, "w_neg": 0.0} for dst, sc in survivors]
        results.sort(key=lambda x: x["probability"], reverse=True)
        return results[:top_n]

    def get_all_priors(
        self,
        technique_ids: list[str],
        top_n: int = 5,
        evidence_text: str | None = None,
    ) -> dict[str, list[dict]]:
        priors = {}
        for tid in technique_ids:
            p = self.get_transition_priors(tid, top_n, evidence_text=evidence_text)
            if p:
                priors[tid] = p
        return priors

    def get_missing_edge_patterns(self, technique_ids: list[str], action_text: str = "",
                                  top_n: int = 6) -> list[dict]:
        import re
        mem_path = os.path.join(self.memory_dir, "memory.json")
        path = os.path.join(self.memory_dir, "missing_edge_patterns.json")
        try:
            if os.path.exists(mem_path):
                m = json.load(open(mem_path, encoding="utf-8")) or {}
                pats = (m.get("missing_edge") or {}).get("patterns", []) or []
            elif os.path.exists(path):
                pats = (json.load(open(path, encoding="utf-8")) or {}).get("patterns", []) or []
            else:
                return []
        except Exception as exc:
            logger.warning("missing_edge_patterns load failed: %s", exc)
            return []

        def parent(t):
            m = re.search(r"T\d{4}", str(t or ""))
            return m.group(0) if m else ""

        present = {parent(t) for t in (technique_ids or [])}
        text = (action_text or "").lower()
        out = []
        for p in pats:
            if not isinstance(p, dict):
                continue
            tech_hit = any(parent(t) in present for t in (p.get("when_techs") or []))
            kw_hit = any(str(k).lower() in text for k in (p.get("when_kw") or []))
            if tech_hit or kw_hit:
                out.append({"id": p.get("id", ""), "hint": p.get("hint", "")})
            if len(out) >= top_n:
                break
        return out

    def _load_verifier_memory(self) -> dict:

        path = os.path.join(self.memory_dir, "nodes.json")
        if os.path.exists(path):
            try:
                m = json.load(open(path, encoding="utf-8")) or {}
                return {"node": m.get("memory", []), "wrong_edge": [], "missing_edge": []}
            except Exception as exc:
                logger.warning("nodes.json load failed: %s", exc)
        path = os.path.join(self.memory_dir, "memory.json")
        if os.path.exists(path):
            try:
                m = json.load(open(path, encoding="utf-8")) or {}
                return {"node": (m.get("node") or {}).get("memory", []),
                        "wrong_edge": [], "missing_edge": []}
            except Exception as exc:
                logger.warning("memory.json load failed: %s", exc)
        path2 = os.path.join(self.memory_dir, "verifier_memory.json")
        if os.path.exists(path2):
            try:
                return json.load(open(path2, encoding="utf-8")) or {}
            except Exception:
                pass
        return {}

    @staticmethod
    def _vparent(t) -> str:
        import re
        m = re.search(r"T\d{4}", str(t or ""))
        return m.group(0) if m else ""

    @staticmethod
    def _vstep(node_id) -> int | None:
        import re
        m = re.match(r"step_(\d+)_", str(node_id or ""))
        return int(m.group(1)) if m else None

    _FANOUT_HUB = {"T1059", "T1105", "T1055", "T1574", "T1620", "T1106"}
    _FANIN_SINK = {"T1041", "T1048", "T1567", "T1071", "T1573", "T1003"}
    _BRANCH = {"T1547", "T1543", "T1053", "T1546", "T1137", "T1071", "T1573", "T1140",
               "T1027", "T1070", "T1005", "T1082", "T1083", "T1016", "T1057", "T1218",
               "T1036", "T1564", "T1113", "T1056", "T1496"}

    def detect_fanout_chains(self, nodes, edges):
        pN = {str(n.get("node_id")): self._vparent(n.get("technique_id")) for n in (nodes or [])}
        out_adj = {}
        for e in (edges or []):
            out_adj.setdefault(str(e.get("src")), set()).add(str(e.get("dst")))
        alerts = []
        seen = set()
        for hub_nid, mids in out_adj.items():
            if pN.get(hub_nid) not in self._FANOUT_HUB:
                continue
            for mid in mids:
                for c in out_adj.get(mid, set()):
                    if pN.get(c) in self._BRANCH and pN.get(c) != pN.get(hub_nid):
                        key = (pN.get(hub_nid), pN.get(mid), pN.get(c))
                        if key in seen:
                            continue
                        seen.add(key)
                        alerts.append({
                            "type": "fanout_chain", "hub": pN.get(hub_nid), "mid": pN.get(mid), "branch": pN.get(c),
                            "hint": "结构提醒[枢纽扇出 vs 链]: 图中 %s→%s→%s。请回原文核对 %s 消费的是 %s 的产物"
                                    "(→保持链,挂%s)还是 %s 载荷的并行行为(与%s互不依赖→改挂枢纽%s)。执行/下载/侧加载"
                                    "枢纽常并行驱动多个后续(持久化/C2/采集各自挂枢纽); 但解密/清痕等若消费的是中间步产物则是真链。"
                                    % (key[0], key[1], key[2], key[2], key[1], key[1], key[0], key[1], key[0]),
                        })
        return alerts

    @staticmethod
    def _aslist(x) -> list:
        return x if isinstance(x, list) else ([x] if x else [])

    def _match_fingerprint(self, items, nodes):
        pN = {str(n.get("node_id")): self._vparent(n.get("technique_id")) for n in (nodes or [])}
        present = set(pN.values()); present.discard("")
        ntext = {str(n.get("node_id")): (str((n.get("procedure") or {}).get("action", "")) + " "
                 + str((n.get("procedure") or {}).get("object", ""))).lower() for n in (nodes or [])}
        byp: dict = {}
        for n in (nodes or []):
            p = self._vparent(n.get("technique_id"))
            if p:
                byp.setdefault(p, []).append(str(n.get("node_id")))

        def kw_hit(parent, kws):
            if not kws:
                return None
            for nid in byp.get(parent, []):
                t = ntext.get(nid, "")
                if any(str(k).lower() in t for k in kws):
                    return True
            return False
        out = []
        for r in items:
            srcs = {self._vparent(s) for s in self._aslist(r.get("src"))} & present
            dsts = {self._vparent(d) for d in self._aslist(r.get("dst"))} & present
            if not (srcs and dsts):
                continue
            mode = r.get("kw_mode", "either")
            hit_pairs = set()
            for s in srcs:
                for d in dsts:
                    if s == d:
                        continue
                    sh = kw_hit(s, r.get("src_kw")); dh = kw_hit(d, r.get("dst_kw"))
                    if mode == "role_only":
                        ok = True
                    elif mode == "both":
                        ok = (sh is not False) and (dh is not False) and (sh or dh)
                    else:
                        ok = (sh is True) or (dh is True) or (sh is None and dh is None)
                    if ok:
                        hit_pairs.add((s, d))
            if hit_pairs:
                rr = dict(r); rr["hit_pairs"] = sorted(hit_pairs); rr["srcs"] = sorted(srcs); rr["dsts"] = sorted(dsts)
                out.append(rr)
        return out

    def _load_rules(self, tier=None):
        path = os.path.join(self.memory_dir, "rules.json")
        try:
            rules = (json.load(open(path, encoding="utf-8")) or {}).get("rules", []) or []
        except Exception:
            return []
        return [r for r in rules if (tier is None or r.get("tier") == tier)]

    def get_explicit_rules(self, nodes: list) -> list[dict]:
        return self._match_fingerprint(self._load_rules("explicit_rule"), nodes)

    def get_implicit_rules(self, nodes: list) -> list[dict]:
        return self._match_fingerprint(self._load_rules("implicit_rule"), nodes)

    def get_node_memory(self, technique_ids: list[str], action_text: str = "",
                        full_text: str = "") -> list[dict]:
        items = self._load_verifier_memory().get("node", []) or []
        present = {self._vparent(t) for t in (technique_ids or [])}
        text = ((action_text or "") + " " + (full_text or "")).lower()
        out = []
        for it in items:
            techs = {self._vparent(t) for t in self._aslist(it.get("tech"))}
            if it.get("find_node"):
                if techs & present:
                    continue
                kw = it.get("when_kw") or []
                ex = it.get("exclude_kw") or []
                if kw and any(str(k).lower() in text for k in kw) and not any(str(e).lower() in text for e in ex):
                    out.append({"id": it.get("id", ""), "hint": it.get("hint", ""),
                                "find_node": True, "tech": sorted(techs)})
            elif techs & present:
                out.append({"id": it.get("id", ""), "hint": it.get("hint", "")})
        return out

    @staticmethod
    def _obj_tokens(s):
        import re as _re
        stop = {"the", "a", "an", "of", "to", "and", "with", "its", "this", "that", "for", "on", "in",
                "using", "used", "via", "from", "as", "it", "was", "were", "be", "by", "into", "then",
                "powershell", "script", "scripts", "command", "commands", "file", "files", "code",
                "payload", "payloads", "data", "program", "process", "tool", "component", "components",
                "脚本", "命令", "文件", "载荷", "程序", "进程", "工具", "组件", "使用", "一个", "这个", "该", "的"}
        ws = _re.findall(r"[a-z0-9][a-z0-9._\-]*|[一-鿿]{2,}", str(s or "").lower())
        return {w for w in ws if w not in stop and len(w) > 1}

    def get_edge_memory(self, nodes: list[dict], edges: list[dict]) -> dict:
        path = os.path.join(self.memory_dir, "cases.json")
        try:
            items = (json.load(open(path, encoding="utf-8")) or {}).get("cases", []) or []
        except Exception:
            return {"wrong": [], "missing": []}
        nodes = nodes or []
        pN = {str(n.get("node_id")): self._vparent(n.get("technique_id")) for n in nodes}
        present = set(pN.values()); present.discard("")
        edge_par = {(pN.get(str(e.get("src"))), pN.get(str(e.get("dst")))) for e in (edges or [])}
        edge_par.discard((None, None))

        ntext = {str(n.get("node_id")): (str((n.get("procedure") or {}).get("action", "")) + " "
                 + str((n.get("procedure") or {}).get("object", ""))).lower() for n in nodes}
        all_edges = [(str(e.get("src")), str(e.get("dst"))) for e in (edges or [])]
        byp: dict = {}
        for n in nodes:
            p = self._vparent(n.get("technique_id"))
            if p:
                byp.setdefault(p, []).append(str(n.get("node_id")))

        def kw1(nid, kws):
            if not kws:
                return None
            t = ntext.get(nid, "")
            return any(str(k).lower() in t for k in kws)

        def edge_kw_ok(a, b, skw, dkw, mode):
            sh, dh = kw1(a, skw), kw1(b, dkw)
            if mode == "role_only":
                return True
            if mode == "both":
                return (sh is not False) and (dh is not False) and (sh or dh)
            return (sh is True) or (dh is True) or (sh is None and dh is None)

        out = {"wrong": [], "missing": [], "review": []}
        for r in items:
            srcs = {self._vparent(s) for s in self._aslist(r.get("src"))} & present
            dsts = {self._vparent(d) for d in self._aslist(r.get("dst"))} & present
            if not (srcs and dsts):
                continue
            pol = str(r.get("polarity") or "")
            skw, dkw = r.get("src_kw"), r.get("dst_kw")
            mode = r.get("kw_mode", "either")
            hit_edges = []
            if pol in ("wrong", "review"):
                for (a, b) in all_edges:
                    if pN.get(a) in srcs and pN.get(b) in dsts and edge_kw_ok(a, b, skw, dkw, mode):
                        hit_edges.append((a, b))
            elif pol == "missing":
                for s in srcs:
                    for d in dsts:
                        if s == d:
                            continue
                        for a in byp.get(s, []):
                            for b in byp.get(d, []):
                                if (pN.get(a), pN.get(b)) in edge_par:
                                    continue
                                if edge_kw_ok(a, b, skw, dkw, mode):
                                    hit_edges.append((a, b))
            if not hit_edges:
                continue
            out[pol if pol in ("wrong", "missing", "review") else "wrong"].append({
                "id": r.get("id", ""), "hint": r.get("hint", ""), "mode": r.get("mode", "suspect"),
                "hit_edges": hit_edges, "srcs": sorted(srcs), "dsts": sorted(dsts),
                "pair": "%s->%s" % ("/".join(sorted(srcs)), "/".join(sorted(dsts)))})
        return out

    def get_wrong_edge_memory(self, nodes: list[dict], edges: list[dict]) -> list[dict]:
        items = self._load_verifier_memory().get("wrong_edge", []) or []
        pN = {str(n.get("node_id")): self._vparent(n.get("technique_id")) for n in (nodes or [])}
        edge_par = {(pN.get(str(e.get("src"))), pN.get(str(e.get("dst")))) for e in (edges or [])}
        edge_par.discard((None, None))
        out = []
        for it in items:
            srcs = {self._vparent(s) for s in self._aslist(it.get("src"))}
            dsts = {self._vparent(d) for d in self._aslist(it.get("dst"))}
            if any((a, b) in edge_par for a in srcs for b in dsts):
                out.append({"id": it.get("id", ""), "hint": it.get("hint", "")})
        return out

    def get_missing_edge_memory(self, nodes: list[dict], edges: list[dict],
                                action_text: str = "", window: int = 4) -> list[dict]:
        items = self._load_verifier_memory().get("missing_edge", []) or []
        nodes = nodes or []
        tech_steps: dict[str, list] = {}
        for n in nodes:
            p = self._vparent(n.get("technique_id"))
            if p:
                tech_steps.setdefault(p, []).append(self._vstep(n.get("node_id")))
        pN = {str(n.get("node_id")): self._vparent(n.get("technique_id")) for n in nodes}
        edge_par = {(pN.get(str(e.get("src"))), pN.get(str(e.get("dst")))) for e in (edges or [])}
        present = set(tech_steps.keys())
        text = (action_text or "").lower()
        out = []
        for it in items:
            if it.get("src") and it.get("dst"):
                srcs = {self._vparent(s) for s in self._aslist(it.get("src"))}
                dsts = {self._vparent(d) for d in self._aslist(it.get("dst"))}
                hit = False
                for s in srcs:
                    for d in dsts:
                        if not s or not d or (s, d) in edge_par:
                            continue
                        si = [x for x in tech_steps.get(s, []) if x is not None]
                        di = [x for x in tech_steps.get(d, []) if x is not None]
                        if any(0 < (dj - sj) <= window for sj in si for dj in di):
                            hit = True
                if hit:
                    out.append({"id": it.get("id", ""), "hint": it.get("hint", "")})
            else:
                if any(self._vparent(t) in present for t in (it.get("when_techs") or []))\
                        or any(str(k).lower() in text for k in (it.get("when_kw") or [])):
                    out.append({"id": it.get("id", ""), "hint": it.get("hint", "")})
        return out

    def _load_grid(self) -> dict:
        if getattr(self, "_grid", None) is not None:
            return self._grid
        self._grid = {}
        path = os.path.join(self.memory_dir, "transition_matrix_grid.csv")
        if not os.path.exists(path):
            logger.warning("transition_matrix_grid.csv 缺失, 转移先验将为空: %s", path)
            return self._grid
        import csv as _csv
        with open(path, encoding="utf-8") as f:
            rdr = _csv.reader(f)
            cols = next(rdr)[1:]
            for r in rdr:
                if not r:
                    continue
                d = {}
                for j, x in enumerate(r[1:]):
                    if x and x != "0":
                        v = float(x)
                        if v > 0:
                            d[cols[j]] = v
                self._grid[r[0]] = d
        return self._grid

    def _grid_row(self, tech) -> dict:
        return self._load_grid().get(self._vparent(tech), {})

    def _load_parent_matrix(self) -> dict:
        if getattr(self, "_pmatrix", None) is not None:
            return self._pmatrix
        self._pmatrix = dict(self._load_grid())
        return self._pmatrix

    def get_succ(self, tech: str, k: int = 3) -> list[str]:
        row = self._load_parent_matrix().get(self._vparent(tech), {})
        return [t for t, _ in sorted(row.items(), key=lambda kv: -kv[1])[:k]]

    def get_pred(self, tech: str, k: int = 3) -> list[str]:
        p = self._vparent(tech)
        col = [(s, d.get(p, 0.0)) for s, d in self._load_parent_matrix().items()]
        return [t for t, v in sorted(col, key=lambda kv: -kv[1])[:k] if v > 0]

    def get_missing_candidates(self, nodes: list[dict], edges: list[dict],
                               k: int = 3, window: int = 4) -> list[dict]:
        nodes = nodes or []
        pN = {str(n.get("node_id")): self._vparent(n.get("technique_id")) for n in nodes}
        step = {str(n.get("node_id")): self._vstep(n.get("node_id")) for n in nodes}
        edge_par = {(pN.get(str(e.get("src"))), pN.get(str(e.get("dst")))) for e in (edges or [])}
        dep_rules = []
        for it in self._load_verifier_memory().get("missing_edge", []):
            if it.get("src") and it.get("dst"):
                dep_rules.append(({self._vparent(s) for s in self._aslist(it["src"])},
                                  {self._vparent(d) for d in self._aslist(it["dst"])}))
        succ_cache, pred_cache = {}, {}
        ids = [str(n.get("node_id")) for n in nodes]
        out, seen = [], set()
        for a in ids:
            pa, sa = pN.get(a), step.get(a)
            if not pa or sa is None:
                continue
            succ_a = succ_cache.setdefault(pa, set(self.get_succ(pa, k)))
            for b in ids:
                pb, sb = pN.get(b), step.get(b)
                if a == b or not pb or sb is None or pa == pb:
                    continue
                if not (0 < sb - sa <= window) or (pa, pb) in edge_par or (a, b) in seen:
                    continue
                pred_b = pred_cache.setdefault(pb, set(self.get_pred(pb, k)))
                mat = (pb in succ_a) or (pa in pred_b)
                mem = any(pa in S and pb in D for S, D in dep_rules)
                if not mat and not mem:
                    continue

                if mat and not mem and (sb - sa) > 2:
                    continue
                seen.add((a, b))
                out.append({"src": a, "dst": b, "par": "%s->%s" % (pa, pb),
                            "matrix": bool(mat), "mem": bool(mem),
                            "level": "strong" if (mat and mem) else "mid"})
        return out

    def get_edge_score(self, src_tech: str, dst_tech: str) -> float | None:
        row = self._grid_row(src_tech)
        if not row:
            return None
        rm = max(row.values())
        return (row.get(self._vparent(dst_tech), 0.0) / rm) if rm else 0.0

    def get_edge_signals(self, nodes: list[dict], edges: list[dict],
                         low: float = 0.2) -> dict:
        pN = {str(n.get("node_id")): self._vparent(n.get("technique_id")) for n in (nodes or [])}

        em = self.get_edge_memory(nodes, edges)
        case_hit = {(a, b) for r in em.get("wrong", []) for a, b in r.get("hit_edges", [])}
        neg_pairs = set()
        for r in self.get_explicit_rules(nodes) + self.get_implicit_rules(nodes):
            if r.get("polarity") == "neg":
                neg_pairs |= set(r.get("hit_pairs", []))
        out = {}
        for e in (edges or []):
            s, d = str(e.get("src")), str(e.get("dst"))
            pa, pb = pN.get(s), pN.get(d)
            if not pa or not pb:
                continue
            r = self.get_edge_score(pa, pb)
            r = 0.0 if r is None else r
            case = (s, d) in case_hit
            neg = (pa, pb) in neg_pairs
            lo = r < low

            if case and lo:
                tier = "特别错误"
            elif case or neg:
                tier = "一般错误"
            elif lo:
                tier = "轻疑"
            else:
                tier = "真边"
            out["%s->%s" % (s, d)] = {"wrong": case or neg, "ratio": round(r, 3), "tier": tier}
        return out

    def add_edge_filter(self, nodes: list[dict], add_findings: list[dict], low: float = 0.2):
        pN = {str(n.get("node_id")): self._vparent(n.get("technique_id")) for n in (nodes or [])}
        dep_rules = []
        for it in self._load_verifier_memory().get("missing_edge", []):
            if it.get("src") and it.get("dst"):
                dep_rules.append(({self._vparent(s) for s in self._aslist(it["src"])},
                                  {self._vparent(d) for d in self._aslist(it["dst"])}))
        kept, dropped = [], []
        for f in (add_findings or []):
            t = str(f.get("target") or "")
            if "->" not in t:
                dropped.append(f); continue
            s, d = (x.strip() for x in t.split("->", 1))
            pa, pb = pN.get(s), pN.get(d)
            r = self.get_edge_score(pa, pb) if (pa and pb) else 0.0
            r = 0.0 if r is None else r
            mem = any(pa in S and pb in D for S, D in dep_rules) if (pa and pb) else False
            (kept if (r >= low or mem) else dropped).append(f)
        return kept, dropped

    @staticmethod
    def _transition_prior_supported_by_context(dst: str, evidence_text: str | None) -> bool:
        if not evidence_text:
            return True
        dst_parent = str(dst or "").split(".")[0]
        cues = CONTEXT_REQUIRED_TARGET_CUES.get(dst) or CONTEXT_REQUIRED_TARGET_CUES.get(dst_parent)
        if not cues:
            return True
        return _has_context_cue(evidence_text, cues)

    def get_negative_transition_boundaries(
        self,
        technique_ids: list[str],
        top_n: int = 8,
    ) -> list[dict]:
        requested = set(technique_ids)
        requested.update(tid.split(".")[0] for tid in technique_ids if tid)
        boundaries = []
        seen = set()
        for src, targets in self.transition_kb.items():
            if src not in requested:
                continue
            for dst, entry in targets.items():
                if entry.w_neg <= entry.w_pos:
                    continue
                key = (src, dst)
                if key in seen:
                    continue
                seen.add(key)
                boundaries.append({
                    "source": src,
                    "target": dst,
                    "w_pos": entry.w_pos,
                    "w_neg": entry.w_neg,
                    "count": entry.count,
                    "reason": "Rejected transition boundary from prior Critic/JudgeLoop audits.",
                })
        boundaries.sort(key=lambda item: (item["w_neg"] - item["w_pos"], item["w_neg"]), reverse=True)
        return boundaries[:top_n]

    def update_transition_kb(self, confirmed_edges: list[tuple[str, str]],
                             rejected_edges: list[tuple[str, str]],
                             confidence_scores: dict[tuple[str, str], float] | None = None):
        for src, dst in confirmed_edges:
            src_parent = src.split(".")[0]
            dst_parent = dst.split(".")[0]
            c = 1.0
            if confidence_scores and (src, dst) in confidence_scores:
                c = confidence_scores[(src, dst)]

            for s in [src, src_parent]:
                for d in [dst, dst_parent]:
                    if s not in self.transition_kb:
                        self.transition_kb[s] = {}
                    if d not in self.transition_kb[s]:
                        self.transition_kb[s][d] = TransitionEntry()
                    self.transition_kb[s][d].w_pos += c
                    self.transition_kb[s][d].count += 1

        for src, dst in rejected_edges:
            src_parent = src.split(".")[0]
            for s in [src, src_parent]:
                if s not in self.transition_kb:
                    self.transition_kb[s] = {}
                d_parent = dst.split(".")[0]
                for d in [dst, d_parent]:
                    if d not in self.transition_kb[s]:
                        self.transition_kb[s][d] = TransitionEntry()
                    self.transition_kb[s][d].w_neg += 1

    def set_excluded_doc_ids(self, doc_ids: Iterable[str] | None):
        self.excluded_doc_ids = {str(doc_id) for doc_id in (doc_ids or []) if str(doc_id)}

    def clear_excluded_doc_ids(self):
        self.excluded_doc_ids = set()

    def get_error_warnings(self, technique_ids: list[str] | None = None,
                           top_n: int = 8,
                           exclude_doc_ids: Iterable[str] | None = None) -> list[dict]:
        error_memory = self._filtered_error_memory(exclude_doc_ids)
        if not error_memory:
            return []

        if technique_ids is None:

            recent = error_memory[-top_n:]
        else:
            tid_set = set(technique_ids)
            parent_set = {t.split(".")[0] for t in tid_set}
            relevant = [e for e in error_memory
                        if e.technique_id in tid_set or e.technique_id.split(".")[0] in parent_set]
            recent = relevant[-top_n:] if relevant else []

        return [self._error_warning_to_dict(e) for e in recent]

    def _ensure_generic_index(self) -> list[ErrorRecord]:
        generic = [r for r in self._filtered_error_memory() if _is_generic_error_record(r)]
        sig = len(generic)
        if getattr(self, "_generic_sig", None) != sig:
            inv: dict[str, set[int]] = {}
            for i, rec in enumerate(generic):
                for cue in rec.action_verbs or []:
                    c = str(cue or "").strip().lower()
                    if c:
                        inv.setdefault(c, set()).add(i)
            self._generic_index = inv
            self._generic_list = generic
            self._generic_sig = sig
        return self._generic_list

    @staticmethod
    def _relevance_score(record: "ErrorRecord", evidence_text: str) -> float:
        text = str(evidence_text or "")
        if not text:
            return 0.0
        lowered = text.lower()
        score = 0.0
        for cue in record.action_verbs or []:
            cl = str(cue or "").strip().lower()
            if not cl:
                continue
            is_phrase = (" " in cl) or ("-" in cl) or ("\\" in cl) or (not cl.isascii())
            if is_phrase:
                if cl in lowered:
                    score += 2.0
            elif _has_context_cue(text, (cl,)):
                score += 1.0
            elif cl.isalpha() and len(cl) >= 5 and re.search(
                rf"(?<![a-z0-9_]){re.escape(cl)}(?:s|ed|ing)?(?![a-z0-9_])", lowered):
                score += 1.0
        return score

    def get_generic_error_warnings(
        self,
        top_n: int = 8,
        evidence_text: str | None = None,
    ) -> list[dict]:
        generic = self._ensure_generic_index()
        if not generic:
            return []
        if not evidence_text:
            return [self._error_warning_to_dict(e) for e in generic[-top_n:]]

        lowered = str(evidence_text).lower()

        cand: set[int] = set()
        for cue, ids in (self._generic_index or {}).items():
            if cue and cue in lowered:
                cand |= ids
        scored = [(self._relevance_score(generic[i], evidence_text), i) for i in cand]
        scored = [(s, i) for (s, i) in scored if s > 0]
        scored.sort(key=lambda t: (-t[0], t[1]))
        selected = [generic[i] for (_, i) in scored[:top_n]]
        return [self._error_warning_to_dict(e) for e in selected]

    @staticmethod
    def _error_record_supported_by_context(record: ErrorRecord, evidence_text: str) -> bool:
        cues = tuple(str(cue or "").strip() for cue in record.action_verbs if str(cue or "").strip())
        if not cues:
            return True
        if _has_context_cue(evidence_text, cues):
            return True
        lowered = str(evidence_text or "").lower()
        for cue in cues:
            cue_lower = cue.lower()
            if not cue_lower.isalpha() or len(cue_lower) < 5:
                continue
            if re.search(rf"(?<![a-z0-9_]){re.escape(cue_lower)}(?:s|ed|ing)?(?![a-z0-9_])", lowered):
                return True
        return False

    @staticmethod
    def _error_warning_to_dict(e: ErrorRecord) -> dict:
        return {
            "error_id": e.error_id,
            "error_type": e.error_type,
            "technique_id": e.technique_id,
            "pattern": f"{e.error_type} on {e.technique_id}",
            "wrong_mapping": e.original_value,
            "correct_mapping": e.corrected_value,
            "reason": e.reason,
            "source_doc_id": e.doc_id,
        }

    def _filtered_error_memory(self, exclude_doc_ids: Iterable[str] | None = None) -> list[ErrorRecord]:
        excluded = set(self.excluded_doc_ids)
        excluded.update(str(doc_id) for doc_id in (exclude_doc_ids or []) if str(doc_id))
        if not excluded:
            return self.error_memory
        return [record for record in self.error_memory if record.doc_id not in excluded]

    def add_error_record(self, error_type: str, technique_id: str,
                         action_verbs: list[str], original: str, corrected: str,
                         reason: str, doc_id: str):
        record = ErrorRecord(
            error_id=f"err_{len(self.error_memory):04d}",
            error_type=error_type,
            technique_id=technique_id,
            action_verbs=action_verbs,
            original_value=original,
            corrected_value=corrected,
            reason=reason,
            doc_id=doc_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self.error_memory.append(record)

    def reset_working(self):
        self.working = {}
        self.blackboard = []

    def append_blackboard(
        self,
        kind: str,
        payload: Any,
        source: str,
        phase: str | None = None,
        iteration: int | None = None,
    ) -> dict[str, Any]:
        record = {
            "event_id": f"bb_{len(self.blackboard):06d}",
            "kind": kind,
            "payload": deepcopy(payload),
            "source": source,
            "phase": phase,
            "iteration": iteration,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.blackboard.append(record)
        return deepcopy(record)

    def get_blackboard(self, kind: str | None = None) -> list[dict[str, Any]]:
        records = self.blackboard
        if kind is not None:
            records = [record for record in records if record["kind"] == kind]
        return deepcopy(records)

    def blackboard_to_dict(self) -> dict[str, list[dict[str, Any]]]:
        events = self.get_blackboard()
        return {"blackboard": events, "events": events}

    def set_working(self, key: str, value: Any):
        self.working[key] = value

    def get_working(self, key: str, default: Any = None) -> Any:
        return self.working.get(key, default)

def _is_generic_error_record(record: ErrorRecord) -> bool:
    doc_id = str(getattr(record, "doc_id", "") or "").lower()
    error_id = str(getattr(record, "error_id", "") or "").lower()
    return (
        doc_id in {"generic-seed-memory", "generic"}
        or doc_id.startswith("generic-")
        or error_id.startswith("seed_")
        or error_id.startswith("seederr")
    )
