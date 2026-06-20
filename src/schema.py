from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Any

@dataclass
class TtpCandidate:
    candidate_id: str
    technique_id: str
    mention: str
    tactic_id: str | None = None
    sentence_id: int | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(d: dict) -> "TtpCandidate":
        return TtpCandidate(
            candidate_id=d["candidate_id"],
            technique_id=d["technique_id"],
            mention=d.get("mention", d["technique_id"]),
            tactic_id=d.get("tactic_id") or d.get("metadata", {}).get("tactic_id"),
            sentence_id=d.get("sentence_id"),
            confidence=d.get("confidence"),
            metadata=d.get("metadata", {}),
        )

@dataclass
class StandardizedInputDocument:
    doc_id: str
    source_dataset: str
    source_path: str
    text: str | None
    sentences: list[str]
    ttp_candidates: list[TtpCandidate]
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(d: dict) -> "StandardizedInputDocument":
        return StandardizedInputDocument(
            doc_id=d["doc_id"],
            source_dataset=d["source_dataset"],
            source_path=d["source_path"],
            text=d.get("text"),
            sentences=d.get("sentences", []),
            ttp_candidates=[TtpCandidate.from_dict(c) for c in d.get("ttp_candidates", [])],
            metadata=d.get("metadata", {}),
        )

    @staticmethod
    def from_json(path: str) -> "StandardizedInputDocument":
        with open(path, "r", encoding="utf-8") as f:
            return StandardizedInputDocument.from_dict(json.load(f))

@dataclass
class GraphNode:
    node_id: str
    mention: str
    node_type: str
    attack_id: str | None = None
    sentence_id: int | None = None
    span: dict[str, int] | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "node_id": self.node_id,
            "mention": self.mention,
            "node_type": self.node_type,
        }
        if self.attack_id is not None:
            d["attack_id"] = self.attack_id
        if self.sentence_id is not None:
            d["sentence_id"] = self.sentence_id
        if self.span is not None:
            d["span"] = self.span
        if self.confidence is not None:
            d["confidence"] = self.confidence
        if self.metadata:
            d["metadata"] = self.metadata
        return d

@dataclass
class GraphEdge:
    src: str
    dst: str
    relation: str
    evidence_span: dict[str, int] | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    VALID_RELATIONS = {"precedes", "enables", "causes", "uses", "targets",
                       "phase_transition", "related"}

    def to_dict(self) -> dict:
        d = {"src": self.src, "dst": self.dst, "relation": self.relation}
        if self.evidence_span is not None:
            d["evidence_span"] = self.evidence_span
        if self.confidence is not None:
            d["confidence"] = self.confidence
        if self.metadata:
            d["metadata"] = self.metadata
        return d

@dataclass
class GraphPath:
    path_id: str
    node_ids: list[str]
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {"path_id": self.path_id, "node_ids": self.node_ids}
        if self.confidence is not None:
            d["confidence"] = self.confidence
        if self.metadata:
            d["metadata"] = self.metadata
        return d

@dataclass
class GraphDocument:
    doc_id: str
    source_dataset: str
    source_path: str
    text: str | None = None
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    paths: list[GraphPath] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "doc_id": self.doc_id,
            "source_dataset": self.source_dataset,
            "source_path": self.source_path,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "paths": [p.to_dict() for p in self.paths],
            "metadata": self.metadata,
        }
        if self.text is not None:
            d["text"] = self.text
        return d

    def to_json(self, path: str, indent: int = 2):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=indent, ensure_ascii=False)

    def validate(self) -> list[str]:
        errors = []
        node_ids = {n.node_id for n in self.nodes}
        if len(node_ids) != len(self.nodes):
            errors.append("Duplicate node_ids detected")
        for e in self.edges:
            if e.src not in node_ids:
                errors.append(f"Edge src '{e.src}' not in nodes")
            if e.dst not in node_ids:
                errors.append(f"Edge dst '{e.dst}' not in nodes")
            if e.relation not in GraphEdge.VALID_RELATIONS:
                errors.append(f"Invalid relation '{e.relation}'")
            if e.src == e.dst:
                errors.append(f"Self-loop: {e.src}")
        for p in self.paths:
            for nid in p.node_ids:
                if nid not in node_ids:
                    errors.append(f"Path '{p.path_id}' references unknown node '{nid}'")
            if len(p.node_ids) < 2:
                errors.append(f"Path '{p.path_id}' has fewer than 2 nodes")
        return errors

@dataclass
class ProcedureRecord:
    actor: str
    action: str
    obj: str
    purpose: str
    source_sentence_ids: list[int] = field(default_factory=list)
    source_text: str = ""

@dataclass
class ProcedureLedgerRecord:
    procedure_id: str
    actor: str = ""
    action: str = ""
    obj: str = ""
    purpose: str = ""
    evidence_sentence_ids: list[int] = field(default_factory=list)
    evidence_text: str = ""
    span: dict[str, int] | None = None
    binding_status: str = "unbound"
    bound_node_id: str = ""
    bound_technique_id: str = ""
    candidate_techniques: list[str] = field(default_factory=list)
    rejected_techniques: list[str] = field(default_factory=list)
    abstention_reason: str = ""
    binding_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "procedure_id": self.procedure_id,
            "actor": self.actor,
            "action": self.action,
            "object": self.obj,
            "purpose": self.purpose,
            "evidence_sentence_ids": self.evidence_sentence_ids,
            "binding_status": self.binding_status,
            "bound_node_id": self.bound_node_id,
            "bound_technique_id": self.bound_technique_id,
        }
        if self.evidence_text:
            d["evidence_text"] = self.evidence_text
        if self.span is not None:
            d["span"] = self.span
        if self.candidate_techniques:
            d["candidate_techniques"] = self.candidate_techniques
        if self.rejected_techniques:
            d["rejected_techniques"] = self.rejected_techniques
        if self.abstention_reason:
            d["abstention_reason"] = self.abstention_reason
        if self.binding_note:
            d["binding_note"] = self.binding_note
        return d

@dataclass
class EdgeEvidenceClaim:
    claim_id: str
    src: str
    dst: str
    src_procedure_id: str = ""
    dst_procedure_id: str = ""
    relation_hint: str = "precedes"
    decision_hint: str = "direct"
    dependency_role_hint: str = "temporal_progression"
    strict_eval: bool = True
    evidence_sentence_ids: list[int] = field(default_factory=list)
    evidence_reason: str = ""
    local_text: str = ""
    graph_edge_present: bool = False
    graph_relation: str = ""
    claim_status: str = "candidate"
    evidence_refs: list[str] = field(default_factory=list)
    necessity_test: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "src": self.src,
            "dst": self.dst,
            "src_node_id": self.src,
            "dst_node_id": self.dst,
            "src_procedure_id": self.src_procedure_id,
            "dst_procedure_id": self.dst_procedure_id,
            "relation_hint": self.relation_hint,
            "decision_hint": self.decision_hint,
            "dependency_role_hint": self.dependency_role_hint,
            "strict_eval": self.strict_eval,
            "evidence_sentence_ids": self.evidence_sentence_ids,
            "evidence_reason": self.evidence_reason,
            "local_text": self.local_text,
            "graph_edge_present": self.graph_edge_present,
            "graph_relation": self.graph_relation,
            "claim_status": self.claim_status,
            "evidence_refs": self.evidence_refs,
            "necessity_test": self.necessity_test,
            "decision_options": ["direct", "bridge", "support", "optional", "abstain"],
        }

@dataclass
class EdgeClaimAudit:
    claim_id: str
    verdict: str
    deviation_type: str = ""
    verdict_action: str = "accept"
    edge_action: str = "keep"
    edge_refs: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "verdict": self.verdict,
            "deviation_type": self.deviation_type,
            "verdict_action": self.verdict_action,
            "edge_action": self.edge_action,
            "edge_refs": self.edge_refs,
            "evidence_refs": self.evidence_refs,
            "reason": self.reason,
        }

@dataclass
class TypedPatch:
    patch_id: str
    finding_id: str = ""
    claim_id: str = ""
    patch_type: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    deviation_type: str = ""
    verdict_action: str = "revise"
    evidence_refs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "patch_id": self.patch_id,
            "finding_id": self.finding_id,
            "claim_id": self.claim_id,
            "patch_type": self.patch_type,
            "payload": self.payload,
            "deviation_type": self.deviation_type,
            "verdict_action": self.verdict_action,
            "evidence_refs": self.evidence_refs,
        }

@dataclass
class IdentifiedTtp:
    node_id: str
    technique_id: str
    technique_name: str
    tactic: str
    tactic_id: str
    node_type: str
    evidence_sentence_ids: list[int] = field(default_factory=list)
    evidence_text: str = ""
    procedure: ProcedureRecord | None = None
    confidence: float = 0.0
    candidates_considered: list[str] = field(default_factory=list)

@dataclass
class ProposedEdge:
    src: str
    dst: str
    relation: str
    evidence_type: str
    evidence_text: str = ""
    evidence_sentence_ids: list[int] = field(default_factory=list)
    kb_prior: dict | None = None
    confidence: float = 0.0

@dataclass
class CriticFinding:
    finding_id: str
    finding_type: str
    target: str
    target_type: str
    reason: str
    constraint_violated: str = ""
    route_to: str = ""
    severity: str = "medium"
    suggested_fix: str = ""
    deviation_type: str = ""
    verdict_action: str = "revise"
    repair_owner: str = ""
    repair_scope: list[str] = field(default_factory=list)
    frozen_scope: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    edge_action: str = ""
    edge_refs: list[str] = field(default_factory=list)

    IDENTIFIER_TYPES = {"misidentification", "unsupported_obs", "missing_node"}
    REASONER_TYPES = {"unsupported_inf", "wrong_edge", "missing_link", "redundant_node"}

    def __post_init__(self):
        if not self.deviation_type:
            self.deviation_type = self.constraint_violated
        if not self.constraint_violated:
            self.constraint_violated = self.deviation_type
        if not self.repair_owner:
            self.repair_owner = self.route_to
        if not self.route_to:
            self.route_to = self.repair_owner
