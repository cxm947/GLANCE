"""Deterministic global-structure analysis for attack-process graphs.

Supplies the *global perspective* the per-node / per-edge agents lack: it computes
in/out degree, weakly-connected components, and flags the three structural
anomalies that should never silently survive — isolated nodes, illegitimate roots
(in-degree 0 but a late-tactic technique), and disconnected components. The report
feeds Agent C's graph-level repair stage and the offline structural metrics.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from knowledge import get_tactic_id, get_tactic_name, get_phase_index, technique_phase

# A root whose tactic phase index is >= this is "too late" to be an attack entry
# point. TA0005 (Defense Evasion) == phase 6 in knowledge.TACTIC_ORDER. Legit entry
# tactics are the earlier ones: Reconnaissance(0) … Privilege Escalation(5).
DEFAULT_ROOT_LATE_PHASE = 6  # == get_phase_index("TA0005")

STRUCTURAL_KINDS = ("isolated", "illegitimate_root", "disconnected_component")

def _parent(t: Any) -> str:
    m = re.search(r"T\d{4}", str(t or ""))
    return m.group(0) if m else str(t or "")

def _node_tech(n: dict) -> str:
    return str(n.get("technique_id") or n.get("attack_id") or "")

@dataclass
class StructuralReport:
    n_nodes: int
    n_edges: int
    in_degree: dict = field(default_factory=dict)
    out_degree: dict = field(default_factory=dict)
    roots: list = field(default_factory=list)          # in==0 & out>0
    sinks: list = field(default_factory=list)          # out==0 & in>0
    isolated: list = field(default_factory=list)       # in==0 & out==0
    components: list = field(default_factory=list)      # list[set[node_id]], largest first
    findings: list = field(default_factory=list)        # [{kind, node_id, technique_id, tactic, tactic_name, detail}]

    def has_anomalies(self) -> bool:
        return bool(self.findings)

    def kinds(self) -> set:
        return {str(f.get("kind")) for f in self.findings}

    def to_dict(self) -> dict:
        return {
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "n_components": len(self.components),
            "in_degree": self.in_degree,
            "out_degree": self.out_degree,
            "roots": list(self.roots),
            "sinks": list(self.sinks),
            "isolated": list(self.isolated),
            "components": [sorted(c) for c in self.components],
            "findings": list(self.findings),
        }

def weakly_connected_components(node_ids: list[str], edges: list[dict]) -> list[set]:
    """Weakly-connected components (edges treated as undirected), largest first."""
    parent = {nid: nid for nid in node_ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for e in (edges or []):
        s, d = str(e.get("src")), str(e.get("dst"))
        if s in parent and d in parent:
            rs, rd = find(s), find(d)
            if rs != rd:
                parent[rs] = rd
    comps: dict[str, set] = {}
    for nid in node_ids:
        comps.setdefault(find(nid), set()).add(nid)
    return sorted(comps.values(), key=lambda c: (-len(c), sorted(c)))

def analyze_structure(nodes: list[dict], edges: list[dict],
                      root_late_phase: int = DEFAULT_ROOT_LATE_PHASE) -> StructuralReport:
    nodes = nodes or []
    node_ids = [str(n.get("node_id")) for n in nodes if n.get("node_id")]
    tech_by_id = {str(n.get("node_id")): _node_tech(n) for n in nodes}

    in_deg = {nid: 0 for nid in node_ids}
    out_deg = {nid: 0 for nid in node_ids}
    for e in (edges or []):
        s, d = str(e.get("src")), str(e.get("dst"))
        if s in out_deg:
            out_deg[s] += 1
        if d in in_deg:
            in_deg[d] += 1

    roots = [nid for nid in node_ids if in_deg[nid] == 0 and out_deg[nid] > 0]
    sinks = [nid for nid in node_ids if out_deg[nid] == 0 and in_deg[nid] > 0]
    isolated = [nid for nid in node_ids if in_deg[nid] == 0 and out_deg[nid] == 0]
    components = weakly_connected_components(node_ids, edges or [])
    isolated_set = set(isolated)

    findings: list[dict] = []

    def _finding(kind: str, nid: str, detail: str) -> dict:
        tech = tech_by_id.get(nid, "")
        tac = get_tactic_id(_parent(tech)) if tech else None
        return {"kind": kind, "node_id": nid, "technique_id": tech,
                "tactic": tac, "tactic_name": get_tactic_name(tac) if tac else "",
                "detail": detail}

    # 1) isolated nodes (in==0 & out==0): definitionally broken in an attack DAG.
    for nid in isolated:
        findings.append(_finding("isolated", nid, "入度=出度=0, 与攻击链脱节"))

    # 2) illegitimate roots: a proper root (in==0, out>0) but a late-phase tactic.
    for nid in roots:
        ph = technique_phase(_parent(tech_by_id.get(nid, "")))
        if ph >= root_late_phase:
            findings.append(_finding(
                "illegitimate_root", nid,
                "起点战术阶段过晚(phase=%d≥%d), 不应作为攻击起点" % (ph, root_late_phase)))

    # 3) disconnected components: >1 component once isolated singletons (already
    #    reported above) are set aside.
    if len(node_ids) > 1:
        real_comps = [c for c in components
                      if not (len(c) == 1 and next(iter(c)) in isolated_set)]
        if len(real_comps) > 1:
            for comp in real_comps[1:]:  # real_comps[0] is the main (largest) component
                anchor = sorted(comp)[0]
                findings.append({
                    "kind": "disconnected_component", "node_id": anchor, "technique_id": "",
                    "tactic": None, "tactic_name": "",
                    "detail": "与主分量不连通的子图: %s" % sorted(comp)})

    return StructuralReport(
        n_nodes=len(node_ids), n_edges=len(edges or []),
        in_degree=in_deg, out_degree=out_deg,
        roots=roots, sinks=sinks, isolated=isolated,
        components=components, findings=findings,
    )
