from __future__ import annotations
import argparse
from difflib import SequenceMatcher
from functools import lru_cache
import json
from pathlib import Path
import re

DEFAULT_MATCH_THRESHOLD = 0.6
MAX_EXACT_MATCHING_NODES = 24
MAX_TOPO_ORDERS = 20

def normalize_tech_id(tid: str) -> str:
    if not tid:
        return ""
    m = re.match(r'(T\d{4})', tid)
    return m.group(1) if m else tid

def load_graph(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def with_artifact_identity(
    metrics: dict,
    pred_path: str | Path,
    gold_path: str | Path,
    output_path: str | Path,
    command: str,
    doc_id: str | None = None,
) -> dict:
    result = dict(metrics)
    result["artifact_identity"] = {
        "doc_id": doc_id or Path(pred_path).stem.replace("_eval", ""),
        "prediction_artifact": str(pred_path),
        "gold_artifact": str(gold_path),
        "generated_output_path": str(output_path),
        "metric_protocol": metrics.get("metric_protocol"),
        "mutation_risk": "writes per-report evaluation artifact",
        "inspection_or_command": command,
    }
    return result

def f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def set_accuracy(tp: int, fp: int, fn: int) -> float:
    denom = tp + fp + fn
    if denom == 0:
        return 0.0
    return tp / denom

def node_id(node: dict, index: int) -> str:
    return str(node.get("node_id") or f"__node_{index}")

def node_ids(graph: dict) -> list:
    return [node_id(node, i) for i, node in enumerate(graph.get("nodes", []))]

def technique_id(node: dict) -> str:
    return str(node.get("attack_id") or node.get("technique_id") or "").strip()

def node_text(node: dict) -> str:
    metadata = node.get("metadata") or {}
    parts = [
        node.get("mention", ""),
        metadata.get("evidence_text", ""),
    ]
    procedure = metadata.get("procedure")
    if isinstance(procedure, dict):
        parts.append(" ".join(str(v) for v in procedure.values()))
    elif procedure:
        parts.append(str(procedure))
    return " ".join(str(part) for part in parts if part).lower()

def text_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()

def sentence_similarity(gold_node: dict, pred_node: dict) -> float:
    gold_sent = gold_node.get("sentence_id")
    pred_sent = pred_node.get("sentence_id")
    if not isinstance(gold_sent, int) or not isinstance(pred_sent, int):
        return 0.0
    return 1.0 / (1 + abs(gold_sent - pred_sent))

def node_match_score(gold_node: dict, pred_node: dict) -> float:
    gold_tid = technique_id(gold_node)
    pred_tid = technique_id(pred_node)
    gold_parent = normalize_tech_id(gold_tid)
    pred_parent = normalize_tech_id(pred_tid)
    semantic_score = text_similarity(node_text(gold_node), node_text(pred_node))

    if gold_tid and pred_tid:
        if gold_tid == pred_tid:
            base_score = 1.0
        elif gold_parent and pred_parent and gold_parent == pred_parent:
            base_score = 0.82
        else:
            base_score = 0.0
    else:
        base_score = semantic_score

    if base_score <= 0:
        return 0.0

    sent_sim = sentence_similarity(gold_node, pred_node)
    id_bonus = (
        0.08
        if gold_node.get("node_id")
        and gold_node.get("node_id") == pred_node.get("node_id")
        and sent_sim >= 0.5
        else 0.0
    )
    sent_bonus = 0.18 * sent_sim
    text_bonus = 0.08 * semantic_score
    return base_score + id_bonus + sent_bonus + text_bonus

def max_weight_node_matching(gold_nodes: list, pred_nodes: list, threshold: float) -> list:
    scores = [
        [node_match_score(gold_node, pred_node) for pred_node in pred_nodes]
        for gold_node in gold_nodes
    ]

    if len(pred_nodes) > MAX_EXACT_MATCHING_NODES:
        candidates = []
        for gi, row in enumerate(scores):
            for pi, score in enumerate(row):
                if score >= threshold:
                    candidates.append((score, gi, pi))
        candidates.sort(reverse=True)
        used_gold = set()
        used_pred = set()
        matches = []
        for score, gi, pi in candidates:
            if gi not in used_gold and pi not in used_pred:
                used_gold.add(gi)
                used_pred.add(pi)
                matches.append((gi, pi, score))
        return sorted(matches)

    @lru_cache(maxsize=None)
    def search(gold_index: int, used_pred_mask: int):
        if gold_index >= len(gold_nodes):
            return 0.0, 0, ()

        best_score, best_count, best_pairs = search(gold_index + 1, used_pred_mask)
        for pred_index, score in enumerate(scores[gold_index]):
            if score < threshold or used_pred_mask & (1 << pred_index):
                continue
            rest_score, rest_count, rest_pairs = search(
                gold_index + 1,
                used_pred_mask | (1 << pred_index),
            )
            candidate = (
                score + rest_score,
                1 + rest_count,
                ((gold_index, pred_index, score),) + rest_pairs,
            )
            if candidate[0] > best_score or (
                candidate[0] == best_score and candidate[1] > best_count
            ):
                best_score, best_count, best_pairs = candidate

        return best_score, best_count, best_pairs

    _, _, pairs = search(0, 0)
    return list(pairs)

def graph_edges(graph: dict) -> set:
    valid_nodes = set(node_ids(graph))
    edges = set()
    for edge in graph.get("edges", []):
        src = str(edge.get("src") or "")
        dst = str(edge.get("dst") or "")
        if src in valid_nodes and dst in valid_nodes:
            edges.add((src, dst))
    return edges

def is_optional_gold_node(node: dict) -> bool:
    metadata = node.get("metadata") or {}
    return str(metadata.get("evaluation_role") or "").lower() in {
        "optional_context",
        "optional_support",
    }

def graph_without_nodes(graph: dict, removed_node_ids: set[str]) -> dict:
    original_nodes = list(graph.get("nodes", []))
    original_ids = [node_id(node, index) for index, node in enumerate(original_nodes)]
    nodes = [
        node
        for index, node in enumerate(original_nodes)
        if node_id(node, index) not in removed_node_ids
    ]
    kept_ids = {
        node_id(node, index)
        for index, node in enumerate(original_nodes)
        if original_ids[index] not in removed_node_ids
    }
    contracted_edges = contract_removed_node_edges(graph, kept_ids, removed_node_ids)
    edges = [
        edge
        for edge in graph.get("edges", [])
        if str(edge.get("src") or "") in kept_ids and str(edge.get("dst") or "") in kept_ids
    ]
    existing_edges = {(str(edge.get("src") or ""), str(edge.get("dst") or "")) for edge in edges}
    for src, dst in sorted(contracted_edges - existing_edges):
        edges.append({
            "src": src,
            "dst": dst,
            "relation": "precedes",
            "metadata": {
                "evaluation_transform": "optional_context_contraction",
                "note": "Optional context node removed for required graph scoring; predecessor and successor were contracted.",
            },
        })
    paths = []
    for path in graph.get("paths", []):
        filtered_path = [
            str(nid)
            for nid in path.get("node_ids", [])
            if str(nid) in kept_ids
        ]
        if len(filtered_path) >= 2:
            updated_path = dict(path)
            updated_path["node_ids"] = filtered_path
            paths.append(updated_path)
    result = dict(graph)
    result["nodes"] = nodes
    result["edges"] = edges
    result["paths"] = paths
    return result

def contract_removed_node_edges(
    graph: dict,
    kept_ids: set[str],
    removed_node_ids: set[str],
) -> set[tuple[str, str]]:
    adjacency: dict[str, list[str]] = {}
    for edge in graph.get("edges", []):
        src = str(edge.get("src") or "")
        dst = str(edge.get("dst") or "")
        if src and dst:
            adjacency.setdefault(src, []).append(dst)

    contracted: set[tuple[str, str]] = set()
    for src in kept_ids:
        stack = list(adjacency.get(src, []))
        seen = set()
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            if current in kept_ids:
                if current != src:
                    contracted.add((src, current))
                continue
            if current not in removed_node_ids:
                continue
            stack.extend(adjacency.get(current, []))
    return contracted

def optional_gold_context(pred_graph: dict, gold_graph: dict, match_threshold: float) -> tuple[dict, dict, dict]:
    gold_nodes = gold_graph.get("nodes", [])
    optional_gold_indices = [i for i, node in enumerate(gold_nodes) if is_optional_gold_node(node)]
    if not optional_gold_indices:
        return pred_graph, gold_graph, {
            "optional_gold_nodes_count": 0,
            "optional_gold_edges_count": 0,
            "optional_gold_matched_count": 0,
            "optional_pred_nodes_ignored": 0,
        }

    gold_ids_all = node_ids(gold_graph)
    pred_ids_all = node_ids(pred_graph)
    optional_gold_nodes = [gold_nodes[i] for i in optional_gold_indices]
    optional_gold_ids = {gold_ids_all[i] for i in optional_gold_indices}

    required_gold = graph_without_nodes(gold_graph, optional_gold_ids)
    required_matches = max_weight_node_matching(required_gold.get("nodes", []), pred_graph.get("nodes", []), match_threshold)
    used_pred_indices = {pred_index for _, pred_index, _ in required_matches}
    remaining_pred_nodes = [
        node for index, node in enumerate(pred_graph.get("nodes", []))
        if index not in used_pred_indices
    ]
    remaining_pred_ids = [
        pred_ids_all[index]
        for index in range(len(pred_ids_all))
        if index not in used_pred_indices
    ]
    optional_matches = max_weight_node_matching(optional_gold_nodes, remaining_pred_nodes, match_threshold)
    optional_pred_ids = {
        remaining_pred_ids[pred_index]
        for _, pred_index, _ in optional_matches
    }

    pred_scoring = graph_without_nodes(pred_graph, optional_pred_ids)
    optional_gold_edges = [
        edge for edge in gold_graph.get("edges", [])
        if str(edge.get("src") or "") in optional_gold_ids or str(edge.get("dst") or "") in optional_gold_ids
    ]
    return pred_scoring, required_gold, {
        "optional_gold_nodes_count": len(optional_gold_nodes),
        "optional_gold_edges_count": len(optional_gold_edges),
        "optional_gold_matched_count": len(optional_matches),
        "optional_pred_nodes_ignored": len(optional_pred_ids),
    }

def component_prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "accuracy": round(set_accuracy(tp, fp, fn), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1(precision, recall), 4),
    }

def edge_component_counts(pred_graph: dict, gold_graph: dict, pred_to_gold: dict) -> tuple[int, int, int]:
    pred_edges = graph_edges(pred_graph)
    gold_edges = graph_edges(gold_graph)
    mapped_pred_edges = set()
    tp = 0

    for src, dst in pred_edges:
        mapped = None
        if src in pred_to_gold and dst in pred_to_gold:
            mapped = (pred_to_gold[src], pred_to_gold[dst])
            mapped_pred_edges.add(mapped)
        if mapped in gold_edges:
            tp += 1

    fp = len(pred_edges) - tp
    fn = len(gold_edges - mapped_pred_edges)
    return tp, fp, fn

def _node_parent_tech_map(graph: dict) -> dict[str, str]:
    ids = node_ids(graph)
    nodes = graph.get("nodes", [])
    return {
        ids[i]: normalize_tech_id(technique_id(node))
        for i, node in enumerate(nodes)
    }

def edge_component_counts_parent_normalized(
    pred_graph: dict,
    gold_graph: dict,
    pred_to_gold: dict,
) -> tuple[int, int, int]:
    pred_parent = _node_parent_tech_map(pred_graph)
    gold_parent = _node_parent_tech_map(gold_graph)

    def parent_pair_set(graph: dict, parent_map: dict[str, str]) -> set[tuple[str, str]]:
        valid_nodes = set(parent_map)
        pairs: set[tuple[str, str]] = set()
        for edge in graph.get("edges", []):
            src = str(edge.get("src") or "")
            dst = str(edge.get("dst") or "")
            if src in valid_nodes and dst in valid_nodes:
                src_p = parent_map[src]
                dst_p = parent_map[dst]
                if src_p and dst_p and src_p != dst_p:
                    pairs.add((src_p, dst_p))
        return pairs

    pred_parent_pairs = parent_pair_set(pred_graph, pred_parent)
    gold_parent_pairs = parent_pair_set(gold_graph, gold_parent)

    tp = len(pred_parent_pairs & gold_parent_pairs)
    fp = len(pred_parent_pairs - gold_parent_pairs)
    fn = len(gold_parent_pairs - pred_parent_pairs)
    return tp, fp, fn

def technique_set(graph: dict) -> set[str]:
    techniques = set()
    for node in graph.get("nodes", []):
        tid = normalize_tech_id(technique_id(node))
        if tid:
            techniques.add(tid)
    return techniques

def ttp_set_metrics(pred_graph: dict, gold_graph: dict) -> dict:
    pred_techniques = technique_set(pred_graph)
    gold_techniques = technique_set(gold_graph)
    tp = len(pred_techniques & gold_techniques)
    fp = len(pred_techniques - gold_techniques)
    fn = len(gold_techniques - pred_techniques)
    return component_prf(tp, fp, fn)

def transitive_closure(nodes: list[str], edges: set[tuple[str, str]]) -> set[tuple[str, str]]:
    children = {node: [] for node in nodes}
    for src, dst in edges:
        if src in children and dst in children:
            children[src].append(dst)

    closure = set()
    for start in nodes:
        seen = set()
        stack = list(children[start])
        while stack:
            current = stack.pop()
            if current in seen or current == start:
                continue
            seen.add(current)
            closure.add((start, current))
            stack.extend(children.get(current, []))
    return closure

def mapped_pred_edges(pred_graph: dict, pred_to_gold: dict) -> set[tuple[str, str]]:
    mapped = set()
    for src, dst in graph_edges(pred_graph):
        if src in pred_to_gold and dst in pred_to_gold:
            mapped.add((pred_to_gold[src], pred_to_gold[dst]))
    return mapped

def reachability_metrics(pred_graph: dict, gold_graph: dict, pred_to_gold: dict) -> dict:
    matched_gold_nodes = sorted(set(pred_to_gold.values()), key=lambda nid: node_ids(gold_graph).index(nid))
    gold_nodes_all = node_ids(gold_graph)
    pred_reachability = transitive_closure(matched_gold_nodes, mapped_pred_edges(pred_graph, pred_to_gold))
    gold_reachability = transitive_closure(gold_nodes_all, graph_edges(gold_graph))
    tp = len(pred_reachability & gold_reachability)
    fp = len(pred_reachability - gold_reachability)
    fn = len(gold_reachability - pred_reachability)
    return component_prf(tp, fp, fn)

def reachability_diagnostics(pred_graph: dict, gold_graph: dict, pred_to_gold: dict) -> dict:
    gold_ids = node_ids(gold_graph)
    matched_gold_nodes = sorted(
        set(pred_to_gold.values()),
        key=lambda nid: gold_ids.index(nid) if nid in gold_ids else 10**9,
    )
    pred_reachability = transitive_closure(
        matched_gold_nodes,
        mapped_pred_edges(pred_graph, pred_to_gold),
    )
    gold_reachability = transitive_closure(gold_ids, graph_edges(gold_graph))
    matched_set = set(matched_gold_nodes)
    matched_gold_reachability = {
        pair for pair in gold_reachability
        if pair[0] in matched_set and pair[1] in matched_set
    }

    matched_tp = pred_reachability & matched_gold_reachability
    matched_fp = pred_reachability - matched_gold_reachability
    matched_fn = matched_gold_reachability - pred_reachability
    matched_metrics = component_prf(len(matched_tp), len(matched_fp), len(matched_fn))

    def pairs(items: set[tuple[str, str]]) -> list[list[str]]:
        return [[src, dst] for src, dst in sorted(items)]

    return {
        "matched_gold_nodes": matched_gold_nodes,
        "true_positive_pairs": pairs(pred_reachability & gold_reachability),
        "false_positive_pairs": pairs(pred_reachability - gold_reachability),
        "false_negative_pairs": pairs(gold_reachability - pred_reachability),
        "matched_node_only": {
            "accuracy": matched_metrics["accuracy"],
            "precision": matched_metrics["precision"],
            "recall": matched_metrics["recall"],
            "f1": matched_metrics["f1"],
            "true_positive_pairs": pairs(matched_tp),
            "false_positive_pairs": pairs(matched_fp),
            "false_negative_pairs": pairs(matched_fn),
        },
    }

def graph_triples(graph: dict, id_map: dict[str, str] | None = None) -> set[tuple[str, str, str]]:
    id_map = id_map or {}
    triples = set()
    ids = node_ids(graph)
    for index, node in enumerate(graph.get("nodes", [])):
        raw_id = ids[index]
        mapped_id = id_map.get(raw_id, raw_id)
        triples.add((mapped_id, "type", str(node.get("node_type") or "action")))
        tid = normalize_tech_id(technique_id(node))
        if tid:
            triples.add((mapped_id, "ttp", tid))
    for src, dst in graph_edges(graph):
        triples.add((id_map.get(src, src), "rel", id_map.get(dst, dst)))
    return triples

def graph_triple_metrics(pred_graph: dict, gold_graph: dict, pred_to_gold: dict) -> dict:
    pred_triples = graph_triples(pred_graph, pred_to_gold)
    gold_triples = graph_triples(gold_graph)
    tp = len(pred_triples & gold_triples)
    fp = len(pred_triples - gold_triples)
    fn = len(gold_triples - pred_triples)
    return component_prf(tp, fp, fn)

def graph_edit_similarity(
    pred_graph: dict,
    gold_graph: dict,
    node_fp: int,
    node_fn: int,
    edge_fp: int,
    edge_fn: int,
) -> float:
    ged = float(node_fp + node_fn) + 0.5 * float(edge_fp + edge_fn)
    normalizer = (
        len(gold_graph.get("nodes", []))
        + len(pred_graph.get("nodes", []))
        + 0.5 * (len(gold_graph.get("edges", [])) + len(pred_graph.get("edges", [])))
    )
    if normalizer <= 0:
        return 0.0
    return round(max(0.0, min(1.0, 1.0 - ged / normalizer)), 4)

def edge_component_diagnostics(
    pred_graph: dict,
    gold_graph: dict,
    pred_to_gold: dict,
    matches: list[tuple[int, int, float]],
) -> dict:
    pred_ids = node_ids(pred_graph)
    gold_ids = node_ids(gold_graph)
    pred_nodes = pred_graph.get("nodes", [])
    gold_nodes = gold_graph.get("nodes", [])
    node_matches = []
    for gold_index, pred_index, score in matches:
        pred_node = pred_nodes[pred_index]
        gold_node = gold_nodes[gold_index]
        node_matches.append({
            "pred_node": pred_ids[pred_index],
            "gold_node": gold_ids[gold_index],
            "score": round(score, 4),
            "pred_technique": technique_id(pred_node),
            "gold_technique": technique_id(gold_node),
        })
    node_matches.sort(key=lambda item: (item["gold_node"], item["pred_node"]))

    pred_edges = graph_edges(pred_graph)
    gold_edges = graph_edges(gold_graph)
    mapped_pred_edges = set()
    true_positive_edges = []
    false_positive_edges = []

    for src, dst in sorted(pred_edges):
        mapped = None
        if src in pred_to_gold and dst in pred_to_gold:
            mapped = (pred_to_gold[src], pred_to_gold[dst])
            mapped_pred_edges.add(mapped)
        if mapped in gold_edges:
            true_positive_edges.append({
                "pred_edge": [src, dst],
                "mapped_gold_edge": [mapped[0], mapped[1]],
            })
        else:
            false_positive_edges.append({
                "pred_edge": [src, dst],
                "mapped_gold_edge": [mapped[0], mapped[1]] if mapped else None,
            })

    false_negative_edges = [
        [src, dst]
        for src, dst in sorted(gold_edges - mapped_pred_edges)
    ]

    return {
        "node_matches": node_matches,
        "true_positive_edges": true_positive_edges,
        "false_positive_edges": false_positive_edges,
        "false_negative_edges": false_negative_edges,
    }

def sample_topological_orders(graph: dict, limit: int = MAX_TOPO_ORDERS) -> list:
    nodes = node_ids(graph)
    index = {nid: i for i, nid in enumerate(nodes)}
    children = {nid: [] for nid in nodes}
    indegree = {nid: 0 for nid in nodes}

    for src, dst in graph_edges(graph):
        children[src].append(dst)
        indegree[dst] += 1
    for nid in nodes:
        children[nid].sort(key=index.get)

    orders = []

    def backtrack(order: list, indegrees: dict, available: list):
        if len(orders) >= limit:
            return
        if len(order) == len(nodes):
            orders.append(order[:])
            return
        if not available:
            return

        for current in sorted(available, key=index.get):
            next_available = [nid for nid in available if nid != current]
            next_indegrees = dict(indegrees)
            for child in children[current]:
                next_indegrees[child] -= 1
                if next_indegrees[child] == 0:
                    next_available.append(child)
            backtrack(order + [current], next_indegrees, next_available)

    initial = [nid for nid in nodes if indegree[nid] == 0]
    backtrack([], indegree, initial)
    return orders or [nodes]

def longest_increasing_subsequence_length(values: list) -> int:
    if not values:
        return 0

    dp = [1] * len(values)
    for i in range(len(values)):
        for j in range(i):
            if values[j] < values[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)

def chain_lis_length(pred_order: list, gold_orders: list, pred_to_gold: dict) -> int:
    mapped_gold_sequence = [pred_to_gold[nid] for nid in pred_order if nid in pred_to_gold]
    best = 0
    for gold_order in gold_orders:
        gold_rank = {nid: i for i, nid in enumerate(gold_order)}
        ranked_sequence = [
            gold_rank[nid]
            for nid in mapped_gold_sequence
            if nid in gold_rank
        ]
        best = max(best, longest_increasing_subsequence_length(ranked_sequence))
    return best

def fixed_mapping_mcis_size(pred_graph: dict, gold_graph: dict, pred_to_gold: dict) -> int:
    gold_order = {nid: i for i, nid in enumerate(node_ids(gold_graph))}
    matched_gold_nodes = sorted(set(pred_to_gold.values()), key=lambda nid: gold_order.get(nid, 10**9))
    if not matched_gold_nodes:
        return 0

    pred_edges_mapped = set()
    for src, dst in graph_edges(pred_graph):
        if src in pred_to_gold and dst in pred_to_gold:
            pred_edges_mapped.add((pred_to_gold[src], pred_to_gold[dst]))

    gold_edges = graph_edges(gold_graph)
    node_count = len(matched_gold_nodes)
    compatible = [0] * node_count
    for i, left in enumerate(matched_gold_nodes):
        compatible[i] |= 1 << i
        for j in range(i + 1, node_count):
            right = matched_gold_nodes[j]
            same_forward = ((left, right) in pred_edges_mapped) == ((left, right) in gold_edges)
            same_backward = ((right, left) in pred_edges_mapped) == ((right, left) in gold_edges)
            if same_forward and same_backward:
                compatible[i] |= 1 << j
                compatible[j] |= 1 << i

    best = 0

    def expand(candidates: int, size: int):
        nonlocal best
        if size + bin(candidates).count("1") <= best:
            return
        if candidates == 0:
            best = max(best, size)
            return

        remaining = candidates
        while remaining:
            if size + bin(remaining).count("1") <= best:
                return
            bit = remaining & -remaining
            vertex = bit.bit_length() - 1
            remaining &= ~bit
            expand(remaining & compatible[vertex], size + 1)

    expand((1 << node_count) - 1, 0)
    return best

# --- structural metrics (global perspective): make the failures that edge_parent_f1
#     cannot see — isolated nodes, lost/spurious start nodes, graph fragmentation —
#     measurable. All computed per-graph in parent-technique space, so they include
#     unmatched/isolated nodes that the node-matching metrics quietly drop.

def _degrees(graph: dict):
    ids = node_ids(graph)
    indeg = {nid: 0 for nid in ids}
    outdeg = {nid: 0 for nid in ids}
    for src, dst in graph_edges(graph):
        outdeg[src] = outdeg.get(src, 0) + 1
        indeg[dst] = indeg.get(dst, 0) + 1
    return ids, indeg, outdeg

def weak_component_count(graph: dict) -> int:
    ids = node_ids(graph)
    parent = {nid: nid for nid in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for src, dst in graph_edges(graph):
        if src in parent and dst in parent:
            parent[find(src)] = find(dst)
    return len({find(nid) for nid in ids})

def isolated_node_count(graph: dict) -> int:
    _, indeg, outdeg = _degrees(graph)
    return sum(1 for nid in indeg if indeg[nid] == 0 and outdeg[nid] == 0)

def _role_techs(graph: dict, role: str) -> set[str]:
    ids, indeg, outdeg = _degrees(graph)
    nodes = graph.get("nodes", [])
    out: set[str] = set()
    for i, nid in enumerate(ids):
        is_root = indeg[nid] == 0 and outdeg[nid] > 0
        is_sink = outdeg[nid] == 0 and indeg[nid] > 0
        if (role == "root" and is_root) or (role == "sink" and is_sink):
            t = normalize_tech_id(technique_id(nodes[i]))
            if t:
                out.add(t)
    return out

def _reachability_parent_pairs(graph: dict) -> set[tuple[str, str]]:
    ids = node_ids(graph)
    nodes = graph.get("nodes", [])
    par = {ids[i]: normalize_tech_id(technique_id(nodes[i])) for i in range(len(ids))}
    pairs: set[tuple[str, str]] = set()
    for src, dst in transitive_closure(ids, graph_edges(graph)):
        ps, pd = par.get(src), par.get(dst)
        if ps and pd and ps != pd:
            pairs.add((ps, pd))
    return pairs

def _subtechnique_fraction(graph: dict) -> float:
    techs = [technique_id(n) for n in graph.get("nodes", [])]
    techs = [t for t in techs if t]
    if not techs:
        return 0.0
    return round(sum(1 for t in techs if "." in t) / len(techs), 4)

def structural_metrics(pred_graph: dict, gold_graph: dict) -> dict:
    p_iso = isolated_node_count(pred_graph)
    g_iso = isolated_node_count(gold_graph)
    p_comp = weak_component_count(pred_graph)
    g_comp = weak_component_count(gold_graph)

    pr, gr = _role_techs(pred_graph, "root"), _role_techs(gold_graph, "root")
    root = component_prf(len(pr & gr), len(pr - gr), len(gr - pr))
    ps, gs = _role_techs(pred_graph, "sink"), _role_techs(gold_graph, "sink")
    sink = component_prf(len(ps & gs), len(ps - gs), len(gs - ps))
    prp, grp = _reachability_parent_pairs(pred_graph), _reachability_parent_pairs(gold_graph)
    reach = component_prf(len(prp & grp), len(prp - grp), len(grp - prp))

    return {
        "isolated_node_count": p_iso,
        "gold_isolated_node_count": g_iso,
        "connected_components": p_comp,
        "gold_connected_components": g_comp,
        "connected_components_match": p_comp == g_comp,
        # a structurally sane reconstruction has no isolated nodes and is not more
        # fragmented than the gold (an attack process is normally one component).
        "structure_valid": bool(p_iso == 0 and p_comp <= max(1, g_comp)),
        "root_set_precision": root["precision"],
        "root_set_recall": root["recall"],
        "root_set_f1": root["f1"],
        "sink_set_precision": sink["precision"],
        "sink_set_recall": sink["recall"],
        "sink_set_f1": sink["f1"],
        "reachability_parent_precision": reach["precision"],
        "reachability_parent_recall": reach["recall"],
        "reachability_parent_f1": reach["f1"],
        "subtechnique_fraction": _subtechnique_fraction(pred_graph),
        "gold_subtechnique_fraction": _subtechnique_fraction(gold_graph),
    }

def evaluate(
    pred_graph: dict,
    gold_graph: dict,
    match_threshold: float = DEFAULT_MATCH_THRESHOLD,
    topo_order_limit: int = MAX_TOPO_ORDERS,
) -> dict:
    pred_scoring_graph, gold_scoring_graph, optional_metrics = optional_gold_context(
        pred_graph,
        gold_graph,
        match_threshold,
    )
    results = evaluate_required(
        pred_scoring_graph,
        gold_scoring_graph,
        match_threshold,
        topo_order_limit,
    )
    results.update(optional_metrics)
    return results

def evaluate_required(
    pred_graph: dict,
    gold_graph: dict,
    match_threshold: float = DEFAULT_MATCH_THRESHOLD,
    topo_order_limit: int = MAX_TOPO_ORDERS,
) -> dict:
    pred_nodes = pred_graph.get("nodes", [])
    gold_nodes = gold_graph.get("nodes", [])
    pred_ids = node_ids(pred_graph)
    gold_ids = node_ids(gold_graph)

    matches = max_weight_node_matching(gold_nodes, pred_nodes, match_threshold)
    pred_to_gold = {
        pred_ids[pred_index]: gold_ids[gold_index]
        for gold_index, pred_index, _ in matches
    }

    node_tp = len(matches)
    node_fp = len(pred_nodes) - node_tp
    node_fn = len(gold_nodes) - node_tp
    node_metrics = component_prf(node_tp, node_fp, node_fn)
    ttp_metrics = ttp_set_metrics(pred_graph, gold_graph)

    edge_tp, edge_fp, edge_fn = edge_component_counts(pred_graph, gold_graph, pred_to_gold)
    edge_metrics = component_prf(edge_tp, edge_fp, edge_fn)
    edge_diagnostics = edge_component_diagnostics(pred_graph, gold_graph, pred_to_gold, matches)
    edge_parent_tp, edge_parent_fp, edge_parent_fn = edge_component_counts_parent_normalized(
        pred_graph, gold_graph, pred_to_gold,
    )
    edge_parent_metrics = component_prf(edge_parent_tp, edge_parent_fp, edge_parent_fn)
    reachability = reachability_metrics(pred_graph, gold_graph, pred_to_gold)
    reachability_details = reachability_diagnostics(pred_graph, gold_graph, pred_to_gold)
    triples = graph_triple_metrics(pred_graph, gold_graph, pred_to_gold)

    pred_order = sample_topological_orders(pred_graph, limit=1)[0]
    gold_orders = sample_topological_orders(gold_graph, limit=topo_order_limit)

    chain_len = chain_lis_length(pred_order, gold_orders, pred_to_gold)
    chain_precision = chain_len / len(pred_nodes) if pred_nodes else 0.0
    chain_recall = chain_len / len(gold_nodes) if gold_nodes else 0.0

    mcis_nodes = fixed_mapping_mcis_size(pred_graph, gold_graph, pred_to_gold)
    graph_precision = mcis_nodes / len(pred_nodes) if pred_nodes else 0.0
    graph_recall = mcis_nodes / len(gold_nodes) if gold_nodes else 0.0

    return {
        "metric_protocol": "WorFEval-adapted",
        "match_threshold": match_threshold,
        "ttp_accuracy": ttp_metrics["accuracy"],
        "ttp_precision": ttp_metrics["precision"],
        "ttp_recall": ttp_metrics["recall"],
        "ttp_f1": ttp_metrics["f1"],
        "node_tp": node_tp,
        "node_fp": node_fp,
        "node_fn": node_fn,
        "node_accuracy": node_metrics["accuracy"],
        "node_precision": node_metrics["precision"],
        "node_recall": node_metrics["recall"],
        "node_f1": node_metrics["f1"],
        "edge_tp": edge_tp,
        "edge_fp": edge_fp,
        "edge_fn": edge_fn,
        "edge_accuracy": edge_metrics["accuracy"],
        "edge_precision": edge_metrics["precision"],
        "edge_recall": edge_metrics["recall"],
        "edge_f1": edge_metrics["f1"],
        "edge_diagnostics": edge_diagnostics,
        "edge_parent_tp": edge_parent_tp,
        "edge_parent_fp": edge_parent_fp,
        "edge_parent_fn": edge_parent_fn,
        "edge_parent_accuracy": edge_parent_metrics["accuracy"],
        "edge_parent_precision": edge_parent_metrics["precision"],
        "edge_parent_recall": edge_parent_metrics["recall"],
        "edge_parent_f1": edge_parent_metrics["f1"],
        "reachability_accuracy": reachability["accuracy"],
        "reachability_precision": reachability["precision"],
        "reachability_recall": reachability["recall"],
        "reachability_f1": reachability["f1"],
        "reachability_diagnostics": reachability_details,
        "f1_chain": round(f1(chain_precision, chain_recall), 4),
        "chain_precision": round(chain_precision, 4),
        "chain_recall": round(chain_recall, 4),
        "chain_lis_length": chain_len,
        "f1_graph": round(f1(graph_precision, graph_recall), 4),
        "graph_precision": round(graph_precision, 4),
        "graph_recall": round(graph_recall, 4),
        "graph_mcis_nodes": mcis_nodes,
        "graph_triple_accuracy": triples["accuracy"],
        "graph_triple_precision": triples["precision"],
        "graph_triple_recall": triples["recall"],
        "graph_triple_f1": triples["f1"],
        "graph_edit_similarity": graph_edit_similarity(
            pred_graph,
            gold_graph,
            node_fp,
            node_fn,
            edge_fp,
            edge_fn,
        ),
        "attack_step_f1": node_metrics["f1"],
        "node_chain_score": round(f1(chain_precision, chain_recall), 4),
        "workflow_graph_score": round(f1(graph_precision, graph_recall), 4),
        "matched_node_count": len(matches),
        "gold_topological_orders_sampled": len(gold_orders),
        "pred_nodes_count": len(pred_nodes),
        "gold_nodes_count": len(gold_nodes),
        "pred_edges_count": len(pred_graph.get("edges", [])),
        "gold_edges_count": len(gold_graph.get("edges", [])),
        **structural_metrics(pred_graph, gold_graph),
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate EDL-LLM output against gold standard")
    parser.add_argument("--pred", required=True, help="Predicted graph JSON")
    parser.add_argument("--gold", required=True, help="Gold standard graph JSON")
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=DEFAULT_MATCH_THRESHOLD,
        help="Node matching threshold used before chain/graph scoring",
    )
    parser.add_argument(
        "--topo-order-limit",
        type=int,
        default=MAX_TOPO_ORDERS,
        help="Maximum number of gold topological orders sampled for f1_chain",
    )
    args = parser.parse_args()

    pred = load_graph(args.pred)
    gold = load_graph(args.gold)
    results = evaluate(pred, gold, args.match_threshold, args.topo_order_limit)
    out_path = args.pred.replace(".json", "_eval.json")
    command = (
        f"python scripts/evaluate.py --pred {args.pred} --gold {args.gold} "
        f"--match-threshold {args.match_threshold} --topo-order-limit {args.topo_order_limit}"
    )
    results = with_artifact_identity(
        results,
        pred_path=args.pred,
        gold_path=args.gold,
        output_path=out_path,
        command=command,
        doc_id=gold.get("doc_id") or pred.get("doc_id"),
    )

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    for k, v in results.items():
        print(f"  {k:30s}: {v}")
    print("=" * 50)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Saved to: {out_path}")

if __name__ == "__main__":
    main()
