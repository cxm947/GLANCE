import json
import sys
import textwrap
from collections import deque

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

TACTIC = [
    ("TA0043", "Reconnaissance",        "#D9D2C5"),
    ("TA0042", "Resource Development",  "#D9D2C5"),
    ("TA0001", "Initial Access",        "#E8B9B9"),
    ("TA0002", "Execution",             "#F6DCBE"),
    ("TA0003", "Persistence",           "#FBEFC1"),
    ("TA0004", "Privilege Escalation",  "#EEE2B0"),
    ("TA0005", "Defense Evasion",       "#CCE7CC"),
    ("TA0006", "Credential Access",     "#D7D1E9"),
    ("TA0007", "Discovery",             "#C7DAF0"),
    ("TA0008", "Lateral Movement",      "#E8CFDF"),
    ("TA0009", "Collection",            "#CDE6E0"),
    ("TA0011", "Command & Control",     "#CAD8DD"),
    ("TA0010", "Exfiltration",          "#D7C9BE"),
    ("TA0040", "Impact",                "#E6C3C0"),
]
FILL = {tid: col for tid, _, col in TACTIC}
TNAME = {tid: nm for tid, nm, _ in TACTIC}

C_FANOUT, C_FANIN, C_BOTH, C_NORMAL = "#E0A526", "#7A3FB0", "#E8731A", "#9AA0A8"
C_INF = "#B0744A"
C_EXPLICIT, C_IMPLICIT = "#3A3F47", "#CC4436"
C_TITLE, C_ID = "#1F2329", "#6B7178"

X_STEP, Y_STEP = 3.95, 2.45
HW, HH = 1.62, 0.95

def _tactic_id(n):
    return (n.get("metadata") or {}).get("tactic_id") or n.get("tactic_id") or ""

def _is_inf(n):
    return ((n.get("metadata") or {}).get("node_type_edl") or n.get("node_type_edl")) == "inf"

def _edge_implicit(e):
    et = (e.get("metadata") or {}).get("evidence_type") or e.get("evidence_type") or "explicit"
    return et == "implicit"

def layer_layout(node_ids, edges):
    succ = {n: [] for n in node_ids}
    pred = {n: [] for n in node_ids}
    for s, d in edges:
        if s in succ and d in succ:
            succ[s].append(d)
            pred[d].append(s)
    indeg = {n: len(pred[n]) for n in node_ids}
    q = deque([n for n in node_ids if indeg[n] == 0])
    topo, ind = [], dict(indeg)
    while q:
        n = q.popleft()
        topo.append(n)
        for d in succ[n]:
            ind[d] -= 1
            if ind[d] == 0:
                q.append(d)
    topo += [n for n in node_ids if n not in topo]
    layer = {n: 0 for n in node_ids}
    for n in topo:
        for d in succ[n]:
            layer[d] = max(layer[d], layer[n] + 1)

    cols = {}
    for n in node_ids:
        cols.setdefault(layer[n], []).append(n)
    order = {c: list(ns) for c, ns in cols.items()}

    for _ in range(4):
        for c in sorted(order):
            if c == 0:
                continue
            prev_pos = {n: i for i, n in enumerate(order[c - 1])}
            order[c].sort(key=lambda n: (
                sum(prev_pos.get(p, 0) for p in pred[n]) / len(pred[n]) if pred[n] else 0.0))
    pos = {}
    for c, ns in order.items():
        k = len(ns)
        for i, n in enumerate(ns):
            pos[n] = (c * X_STEP, ((k - 1) / 2.0 - i) * Y_STEP)
    return pos, succ, pred, layer

def render(graph_path, out_path, title=None):
    g = json.load(open(graph_path, encoding="utf-8"))
    nodes = {n["node_id"]: n for n in g["nodes"]}
    edges = [(e["src"], e["dst"], e) for e in g["edges"] if e["src"] in nodes and e["dst"] in nodes]
    node_ids = list(nodes)
    pos, succ, pred, layer = layer_layout(node_ids, [(s, d) for s, d, _ in edges])

    indeg = {n: 0 for n in node_ids}
    outdeg = {n: 0 for n in node_ids}
    for s, d, _ in edges:
        outdeg[s] += 1
        indeg[d] += 1

    def border(nid):
        if _is_inf(nodes[nid]):
            return C_INF, 2.4, (0, (4, 2))
        o, i = outdeg[nid] >= 2, indeg[nid] >= 2
        if o and i:
            return C_BOTH, 3.2, "solid"
        if o:
            return C_FANOUT, 2.8, "solid"
        if i:
            return C_FANIN, 2.8, "solid"
        return C_NORMAL, 1.1, "solid"

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    x0, x1 = min(xs) - HW, max(xs) + HW
    y0, y1 = min(ys) - HH, max(ys) + HH
    fig, ax = plt.subplots(figsize=((x1 - x0 + 3.0) * 0.66, (y1 - y0 + 4.0) * 0.66), dpi=140)
    ax.set_xlim(x0 - 1.4, x1 + 1.4)
    ax.set_ylim(y0 - 1.4, y1 + 5.0)
    ax.set_aspect("equal")
    ax.axis("off")

    for s, d, e in edges:
        (sx, sy), (dx, dy) = pos[s], pos[d]
        p0, p1 = (sx + HW, sy), (dx - HW, dy)
        implicit = _edge_implicit(e)
        span = layer[d] - layer[s]
        if span >= 2:
            rad = 0.18 if sy >= dy else -0.18
        else:
            rad = 0.0 if abs(dy - sy) < 1e-6 else (0.07 if dy > sy else -0.07)
        ax.add_patch(FancyArrowPatch(
            p0, p1, connectionstyle="arc3,rad=%s" % rad, arrowstyle="-|>",
            mutation_scale=13, lw=1.5, color=(C_IMPLICIT if implicit else C_EXPLICIT),
            linestyle=((0, (5, 3)) if implicit else "solid"),
            shrinkA=0, shrinkB=0, zorder=2, joinstyle="round", capstyle="round"))

    used_tactics = []
    titles = []
    for nid, (cx, cy) in pos.items():
        n = nodes[nid]
        tid = _tactic_id(n)
        if tid and tid not in used_tactics:
            used_tactics.append(tid)
        bcol, blw, bls = border(nid)
        ax.add_patch(FancyBboxPatch(
            (cx - HW, cy - HH), 2 * HW, 2 * HH,
            boxstyle="round,pad=0.015,rounding_size=0.16",
            facecolor=FILL.get(tid, "#EEEEEE"), edgecolor=bcol, lw=blw,
            linestyle=bls, zorder=4))
        label = n.get("mention") or nid
        lines = 1 if len(label) <= 13 else (2 if len(label) <= 28 else 3)
        wrapped = "\n".join(textwrap.wrap(label, width=max(9, -(-len(label) // lines)),
                                          break_long_words=False)) or label
        t = ax.text(cx, cy + 0.22, wrapped, ha="center", va="center", color=C_TITLE,
                    fontsize=11.5, fontweight="bold", linespacing=0.95, zorder=6)
        ax.text(cx, cy - 0.62, "[%s]" % (n.get("attack_id") or ""), ha="center", va="center",
                color=C_ID, fontsize=8.5, zorder=6)
        titles.append((t, cx, cy))

    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for t, cx, cy in titles:
        lo = ax.transData.transform((cx - HW * 0.90, cy - HH * 0.60))
        hi = ax.transData.transform((cx + HW * 0.90, cy + HH * 0.60))
        tgt_w, tgt_h = abs(hi[0] - lo[0]), abs(hi[1] - lo[1])
        ext = t.get_window_extent(r)
        if ext.width > 0 and ext.height > 0:
            fs = t.get_fontsize() * min(tgt_w / ext.width, tgt_h / ext.height)
            t.set_fontsize(max(6.5, min(13.5, fs)))

    nt = title or "%s — EDL-LLM attack-process DAG  (%d nodes / %d edges)" % (
        g.get("doc_id", ""), len(nodes), len(edges))
    ax.text((x0 + x1) / 2, y1 + 4.0, nt, ha="center", va="center",
            fontsize=18, fontweight="bold", color="#23262B")

    def hrow(items, ycen, base_fs):
        txts = [ax.text(0, ycen, it["label"], ha="left", va="center",
                        fontsize=base_fs + (1.0 if it["kind"] == "head" else 0.0),
                        fontweight=("bold" if it["kind"] == "head" else "normal"),
                        color=("#23262B" if it["kind"] == "head" else "#33373D"), zorder=9)
                for it in items]
        fig.canvas.draw()
        rr = fig.canvas.get_renderer()
        ppu = ax.transData.transform((1, ycen))[0] - ax.transData.transform((0, ycen))[0]
        sw = [0.0 if it["kind"] == "head" else (1.18 if it["kind"] in ("eline", "iline") else 0.80)
              for it in items]
        gap_st, item_gap = 0.26, 1.05
        tw = [t.get_window_extent(rr).width / ppu for t in txts]
        total = sum(sw[i] + (gap_st if sw[i] else 0.0) + tw[i] for i in range(len(items)))\
            + item_gap * (len(items) - 1)
        avail = x1 - x0
        if total > avail:
            f = avail / total * 0.98
            for t in txts:
                t.set_fontsize(t.get_fontsize() * f)
            fig.canvas.draw()
            tw = [t.get_window_extent(rr).width / ppu for t in txts]
            total = sum(sw[i] + (gap_st if sw[i] else 0.0) + tw[i] for i in range(len(items)))\
                + item_gap * (len(items) - 1)
        cur = (x0 + x1) / 2 - total / 2
        for i, it in enumerate(items):
            k, col = it["kind"], it.get("color")
            if k == "fill":
                ax.add_patch(Rectangle((cur, ycen - 0.21), 0.80, 0.42, facecolor=col,
                             edgecolor="#9AA0A8", lw=0.8, zorder=9))
            elif k == "border":
                ax.add_patch(FancyBboxPatch((cur, ycen - 0.22), 0.80, 0.44,
                             boxstyle="round,pad=0.01,rounding_size=0.10", facecolor="white",
                             edgecolor=col, lw=it.get("lw", 2.6),
                             linestyle=it.get("ls", "solid"), zorder=9))
            elif k in ("eline", "iline"):
                ax.add_line(Line2D([cur, cur + 1.18], [ycen, ycen], color=col, lw=2.2,
                            linestyle=((0, (5, 3)) if k == "iline" else "solid"), zorder=9))
            txts[i].set_x(cur + sw[i] + (gap_st if sw[i] else 0.0))
            cur += sw[i] + (gap_st if sw[i] else 0.0) + tw[i] + item_gap

    hrow([{"kind": "head", "label": "Tactic:"}]
         + [{"kind": "fill", "color": FILL[t], "label": "%s %s" % (t, TNAME[t])}
            for t in [t for t, _, _ in TACTIC if t in used_tactics]],
         y1 + 2.5, 10.5)
    hrow([{"kind": "head", "label": "Node / Edge:"},
          {"kind": "border", "color": C_FANOUT, "lw": 2.8, "label": "fan-out hub"},
          {"kind": "border", "color": C_FANIN, "lw": 2.8, "label": "fan-in hub"},
          {"kind": "border", "color": C_INF, "lw": 2.4, "ls": (0, (4, 2)), "label": "inferred node"},
          {"kind": "eline", "color": C_EXPLICIT, "label": "explicit edge"},
          {"kind": "iline", "color": C_IMPLICIT, "label": "implicit edge"}],
         y1 + 1.1, 10.5)

    fig.savefig(out_path, bbox_inches="tight", facecolor="white", pad_inches=0.25)
    plt.close(fig)
    print("wrote %s  (%d nodes / %d edges)" % (out_path, len(nodes), len(edges)))

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python tools/render_graph.py <graph.json> <out.png> [--title TITLE]")
        sys.exit(1)
    title = None
    if "--title" in sys.argv:
        title = sys.argv[sys.argv.index("--title") + 1]
    render(sys.argv[1], sys.argv[2], title)
