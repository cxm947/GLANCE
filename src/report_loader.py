from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

CANDIDATE_MODES = ("text-only", "auto", "gold")

def safe_doc_name(doc_id: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\s]+', "_", (doc_id or "").strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "report"

def split_sentences(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[。！？；;])\s*|(?<=[.!?])\s+", text)
    return [part.strip() for part in parts if part.strip()]

def read_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_case(name: str, base: str = "data/case_study") -> dict[str, Any]:
    root = Path(base)
    text = (root / "inputs" / f"{name}.txt").read_text(encoding="utf-8").strip()
    gold = read_json(root / "gold" / f"{name}.json")
    sentences = gold.get("sentences") or split_sentences(text)
    return {"doc_id": gold.get("doc_id") or name, "text": text, "sentences": sentences, "gold": gold}

def discover_gold_files(gold_dir: Path) -> list[Path]:
    gold_dir = Path(gold_dir)
    if not gold_dir.exists():
        raise FileNotFoundError(f"Report directory not found: {gold_dir}")
    return sorted(
        path for path in gold_dir.glob("*.json")
        if not path.name.endswith("_eval.json")
    )

AUTO_PATTERNS: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    ("T1566.001", "Spearphishing Attachment", "TA0001", ("email", "trojanized", "attachment", "document")),
    ("T1204.002", "User Execution: Malicious File", "TA0002", ("victim opens", "user opens", "opens the document", "opened the document")),
    ("T1105", "Ingress Tool Transfer", "TA0011", ("fetches", "fetched", "downloaded", "remote template", "actor-controlled website")),
    ("T1203", "Exploitation for Client Execution", "TA0002", ("cve-", "exploit", "exploited")),
    ("T1059.001", "Command and Scripting Interpreter: PowerShell", "TA0002", ("powershell",)),
    ("T1112", "Modify Registry", "TA0005", ("registry key", "currentversion\\run", "run registry")),
    ("T1547.001", "Registry Run Keys / Startup Folder", "TA0003", ("currentversion\\run", "run registry", "run key")),
    ("T1127.001", "Trusted Developer Utilities Proxy Execution: MSBuild", "TA0005", ("msbuild",)),
    ("T1140", "Deobfuscate/Decode Files or Information", "TA0005", ("deobfuscate", "deobfuscated", "decode", "decoded")),
    ("T1573.001", "Encrypted Channel: Symmetric Cryptography", "TA0011", ("encrypted rc4", "rc4", "encrypted", "byte stream")),
    ("T1583.001", "Acquire Infrastructure: Domains", "TA0042", ("c2 server was the domain", "c2 server", "domain")),
    ("T1082", "System Information Discovery", "TA0007", ("enumerate the host", "host information", "system information")),
    ("T1041", "Exfiltration Over C2 Channel", "TA0010", ("sent back to", "exfiltrat", "sent back to the threat actor")),
)

def build_auto_candidates(sentences: list[str]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for sentence_id, sentence in enumerate(sentences):
        lowered = sentence.lower()
        local_context = " ".join(
            sentences[index].lower()
            for index in range(max(0, sentence_id - 1), min(len(sentences), sentence_id + 2))
        )
        for technique_id, mention, tactic_id, needles in AUTO_PATTERNS:
            if not any(needle in lowered for needle in needles):
                continue
            key = (sentence_id, technique_id)
            if key in seen:
                continue
            seen.add(key)
            candidates.append({
                "candidate_id": f"auto_{sentence_id}_{technique_id.replace('.', '_')}",
                "technique_id": technique_id,
                "mention": mention,
                "tactic_id": tactic_id,
                "sentence_id": sentence_id,
                "confidence": 0.65,
                "metadata": {"source": "auto-text-pattern", "evidence_text": sentence},
            })
        if "stager" in lowered and "script" in lowered and "powershell" in local_context:
            key = (sentence_id, "T1059.001")
            if key not in seen:
                seen.add(key)
                candidates.append({
                    "candidate_id": f"auto_{sentence_id}_T1059_001",
                    "technique_id": "T1059.001",
                    "mention": "Command and Scripting Interpreter: PowerShell",
                    "tactic_id": "TA0002",
                    "sentence_id": sentence_id,
                    "confidence": 0.6,
                    "metadata": {"source": "auto-text-pattern", "evidence_text": sentence},
                })
    return candidates

def build_standardized_input(
    gold_graph: dict[str, Any],
    gold_path: Path,
    candidate_mode: str = "text-only",
) -> dict[str, Any]:
    if candidate_mode not in CANDIDATE_MODES:
        raise ValueError(f"candidate_mode must be one of {CANDIDATE_MODES}")

    text = gold_graph.get("text") or ""
    doc = {
        "doc_id": gold_graph.get("doc_id") or Path(gold_path).stem.replace("_DAG", ""),
        "source_dataset": gold_graph.get("source_dataset") or "gold",
        "source_path": str(gold_path),
        "text": text,
        "sentences": split_sentences(text),
        "ttp_candidates": [],
        "metadata": {
            "source_gold_artifact": str(gold_path),
            "candidate_mode": candidate_mode,
        },
    }

    if candidate_mode == "auto":
        doc["ttp_candidates"] = build_auto_candidates(doc["sentences"])
    elif candidate_mode == "gold":
        candidates = []
        for index, node in enumerate(gold_graph.get("nodes", [])):
            technique_id = str(node.get("attack_id") or node.get("technique_id") or "")
            if not technique_id:
                continue
            metadata = dict(node.get("metadata") or {})
            metadata["source"] = "gold-oracle-candidate"
            metadata["source_node_id"] = node.get("node_id", "")
            candidates.append({
                "candidate_id": node.get("node_id") or f"gold_cand_{index}",
                "technique_id": technique_id,
                "mention": node.get("mention") or technique_id,
                "tactic_id": metadata.get("tactic_id"),
                "sentence_id": node.get("sentence_id"),
                "confidence": node.get("confidence", 1.0),
                "metadata": metadata,
            })
        doc["ttp_candidates"] = candidates

    return doc

def write_standardized_input(input_doc: dict[str, Any], output_dir: Path) -> Path:
    input_path = Path(output_dir) / "inputs" / f"{safe_doc_name(input_doc['doc_id'])}.json"
    write_json(input_path, input_doc)
    return input_path
