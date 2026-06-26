from __future__ import annotations

from .metrics import (
    evaluate,
    evaluate_required,
    load_graph,
    with_artifact_identity,
    DEFAULT_MATCH_THRESHOLD,
    MAX_TOPO_ORDERS,
)

REPORTED_METRICS: dict[str, str] = {
    "TTP F1": "ttp_f1",
    "Attack Step F1": "attack_step_f1",
    "Node Chain": "node_chain_score",
    "Workflow Graph": "workflow_graph_score",
    "Graph Triple": "graph_triple_f1",
    "Edge F1 (strict)": "edge_f1",
    "Edge F1 (parent-normalized)": "edge_parent_f1",
    "Root-set F1": "root_set_f1",
    "Reachability (parent) F1": "reachability_parent_f1",
    "Isolated nodes": "isolated_node_count",
    "Connected components": "connected_components",
    "Structure valid": "structure_valid",
}

__all__ = [
    "evaluate",
    "evaluate_required",
    "load_graph",
    "with_artifact_identity",
    "REPORTED_METRICS",
    "DEFAULT_MATCH_THRESHOLD",
    "MAX_TOPO_ORDERS",
]
