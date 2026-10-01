"""
eval.py
-------
LUMIN evaluation pipeline: RAG conditions, KG traversal, zero-shot,
multi-model LLM routing, retrieval recall, and calibration curves.

Usage:
    python eval.py --condition rag_mapping_docs
    python eval.py --condition kg --calibration-curve
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import networkx as nx
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.traversal import EDGE_PRIORITY, EMBEDDING_MODEL, LUMINTraversal

load_dotenv()

# ── Constants ──────────────────────────────────────────────────────────────────

CONDITIONS = [
    "rag_concept_docs",
    "rag_mapping_docs",
    "rag_oracle",
    "kg",
    "zero_shot",
]

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TOP_K = 5
TIERS = ["T0", "T1", "T2", "T3"]

DEFAULT_OPEN_MODELS = [
    "meta-llama/Llama-3.2-3B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
]

RAG_SYSTEM_PROMPT = """You extract PDS4 query filters from natural language.
Given context documents and a user query, output ONLY valid JSON with this shape:
{"filters": {"field_name": {"min": <number>, "max": <number>, "value": <string>}}, "confidence": <number between 0 and 1>}
Include only fields clearly supported by the query and context.
Omit fields you are uncertain about. Use min/max for ranges and value for exact matches.
The "confidence" field is your own honest self-assessment of how likely it is that the
filters you produced are COMPLETELY correct — not just plausible. 1.0 means you are certain
every field and value is right. A low value (e.g. 0.2) means you are largely guessing, the
context didn't clearly support your answer, or you suspect you are missing required fields.
Use the full range — do not default to a fixed value like 0.5 or 1.0 out of habit."""

DEV_TEST_CASES = [
    {
        "query": "southern summer",
        "tier": "T0",
        "concept_label": "Martian Southern Summer",
        "ground_truth_filters": {"solar_longitude": {"min": 180, "max": 360}},
    },
    {
        "query": "HiRISE",
        "tier": "T1",
        "concept_label": "HiRISE High-Resolution Imaging",
        "ground_truth_filters": {"mission_name": {"value": "HiRISE"}},
    },
    {
        "query": "southern hemisphere warm season",
        "tier": "T2",
        "concept_label": "Martian Southern Summer",
        "ground_truth_filters": {"solar_longitude": {"min": 180, "max": 360}},
    },
    {
        "query": "NPLD stratigraphic profiles from orbit",
        "tier": "T3",
        "concept_label": "SHARAD Radargrams",
        "ground_truth_filters": {"mission_name": {"value": "SHARAD"}},
    },
]


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    query: str
    tier: str
    condition: str
    predicted_filters: dict
    ground_truth_filters: dict
    correct: bool
    confidence: float
    context_docs: list[str]
    retrieval_recall: Optional[bool] = None
    verbalized_confidence: Optional[float] = None


# ── Clients ────────────────────────────────────────────────────────────────────

def get_embedding_client() -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required for embeddings")
    return OpenAI(api_key=key)


def get_llm_client(model: str) -> OpenAI:
    if model.startswith("meta-llama"):
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is required for meta-llama models"
            )
        return OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1")
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required")
    return OpenAI(api_key=key)


# ── KG document builders ───────────────────────────────────────────────────────

def _parse_value_hint(raw: Any) -> Any:
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw


def format_value_hint(hint: Any) -> str:
    if hint is None:
        return "no specific value constraint"
    if not isinstance(hint, dict):
        return str(hint)

    unit = hint.get("unit", "")
    unit_str = f" {unit}" if unit else ""

    if "value" in hint:
        return f"value {hint['value']}"

    lo = hint.get("min")
    hi = hint.get("max")
    if lo is not None and hi is not None:
        return f"values between {lo} and {hi}{unit_str}"
    if lo is not None:
        return f"values at least {lo}{unit_str}"
    if hi is not None:
        return f"values at most {hi}{unit_str}"
    return "no specific value constraint"


def _concept_aliases(G: nx.DiGraph, concept_id: str) -> list[str]:
    aliases = []
    for pred in G.predecessors(concept_id):
        if G.nodes[pred].get("node_type") == "alias":
            edge = G.edges.get((pred, concept_id), {})
            if edge.get("edge_type") == "is_alias_of":
                aliases.append(G.nodes[pred].get("surface_form", pred))
    return aliases


def _concept_schema_leaves(G: nx.DiGraph, concept_id: str) -> list[tuple[str, str, Any]]:
    """
    Return (field_name, edge_type, value_hint) for each schema leaf this
    concept actually grounds to.

    v2 fix: concepts do NOT connect directly to schema_leaf nodes. The real
    path is concept --instance_of--> class --grounded_in--> schema_leaf,
    with the concept's instance_of edge carrying slot_values (JSON string)
    that says which of the class's slots are actually filled. Only filled
    slots are emitted — an unfilled slot has no value to ground, matching
    the traversal's own "unfilled slots are unreachable" rule.
    """
    leaves = []

    # Step 1: find the class via instance_of, and this concept's slot values
    class_id = None
    slot_values: dict = {}
    for succ in G.successors(concept_id):
        if G.nodes[succ].get("node_type") != "class":
            continue
        edge = G.edges.get((concept_id, succ), {})
        if edge.get("edge_type") != "instance_of":
            continue
        class_id = succ
        raw = edge.get("slot_values")
        if raw:
            slot_values = json.loads(raw) if isinstance(raw, str) else raw
        break
    if class_id is None:
        return leaves

    # Step 2: map this class's slots to schema fields via grounded_in
    slot_to_field = {}
    for succ in G.successors(class_id):
        if G.nodes[succ].get("node_type") != "schema_leaf":
            continue
        edge = G.edges.get((class_id, succ), {})
        if edge.get("edge_type") != "grounded_in":
            continue
        slot = edge.get("slot")
        slot_to_field[slot] = G.nodes[succ].get("name", succ)

    # Step 3: emit only the slots this concept actually fills
    for slot, value in slot_values.items():
        if slot == "variants" and isinstance(value, list):
            # multi-region concepts (e.g. Mars Polar Regions): each variant
            # is its own west/east/south/north box sharing the same slot map
            for variant in value:
                for sub_slot, sub_value in variant.items():
                    field_name = slot_to_field.get(sub_slot)
                    if field_name:
                        leaves.append((field_name, "grounded_in", _parse_value_hint(sub_value)))
        else:
            field_name = slot_to_field.get(slot)
            if field_name:
                leaves.append((field_name, "grounded_in", _parse_value_hint(value)))

    return leaves


def _pick_primary_leaf(
    leaves: list[tuple[str, str, Any]],
) -> tuple[str, str, Any] | None:
    if not leaves:
        return None
    return max(
        leaves,
        key=lambda item: EDGE_PRIORITY.get(item[1], 0.0),
    )


def build_concept_docs(G: nx.DiGraph) -> tuple[list[str], list[dict]]:
    docs: list[str] = []
    metadata: list[dict] = []

    for node_id, data in G.nodes(data=True):
        if data.get("node_type") != "concept":
            continue

        label = data.get("label", node_id)
        description = data.get("description", "")
        aliases = _concept_aliases(G, node_id)
        leaves = _concept_schema_leaves(G, node_id)
        field_names = [f for f, _, _ in leaves]

        alias_str = ", ".join(aliases) if aliases else "none"
        field_str = ", ".join(field_names) if field_names else "none"
        text = (
            f"{label}. {description} "
            f"Aliases: {alias_str}. Schema fields: {field_str}."
        )

        docs.append(text)
        metadata.append({
            "doc_type": "concept",
            "concept_id": node_id,
            "concept_label": label,
            "field_names": field_names,
        })

    print(f"  Built {len(docs)} concept docs")
    return docs, metadata


def build_mapping_docs(G: nx.DiGraph) -> tuple[list[str], list[dict]]:
    """
    One doc per alias, listing EVERY schema field the alias's concept
    grounds to — not just a single "primary" one.

    Fix: the original version picked one field via _pick_primary_leaf and
    built the doc around that alone. For any concept needing more than one
    field (HiRISE needs 3; a MissionPhase needs up to 4), that silently
    produced a document that structurally could not supply the rest —
    confirmed live: rag_mapping_docs and rag_oracle (which reuses this same
    doc set) both scored exactly 0% on every query requiring 2+ fields,
    with 95-100% on single-field queries. That's not a model weakness,
    it's this function withholding information the KG actually has.
    """
    docs: list[str] = []
    metadata: list[dict] = []

    for alias_id, data in G.nodes(data=True):
        if data.get("node_type") != "alias":
            continue

        alias = data.get("surface_form", alias_id)
        concept_id = None
        for succ in G.successors(alias_id):
            if G.edges.get((alias_id, succ), {}).get("edge_type") == "is_alias_of":
                concept_id = succ
                break
        if concept_id is None:
            continue

        concept_label = G.nodes[concept_id].get("label", concept_id)
        leaves = _concept_schema_leaves(G, concept_id)  # now returns ALL grounded leaves

        if not leaves:
            text = (
                f"{alias} refers to {concept_label}, "
                f"which has no mapped PDS field."
            )
            field_names: list[str] = []
        else:
            field_names = [f for f, _, _ in leaves]
            field_lines = [
                f"{field_name} ({format_value_hint(value_hint)})"
                for field_name, _, value_hint in leaves
            ]
            fields_str = "; ".join(field_lines)
            text = (
                f"{alias} refers to {concept_label}, "
                f"which maps to the following PDS field(s): {fields_str}."
            )

        docs.append(text)
        metadata.append({
            "doc_type": "mapping",
            "alias": alias,
            "concept_id": concept_id,
            "concept_label": concept_label,
            "field_names": field_names,
        })

    print(f"  Built {len(docs)} mapping docs")
    return docs, metadata


def find_oracle_doc(
    mapping_index: "FlatRAGIndex",
    ground_truth_filters: dict,
    concept_label: str | None = None,
) -> str:
    gt_fields = set(ground_truth_filters.keys())

    best = None
    best_score = -1
    for i, meta in enumerate(mapping_index.metadata):
        score = 0
        meta_fields = set(meta.get("field_names", []))
        if gt_fields & meta_fields:
            score += 2
        doc = mapping_index.docs[i]
        if any(f in doc for f in gt_fields):
            score += 1
        if concept_label:
            cl = concept_label.lower()
            if meta.get("concept_label", "").lower() == cl:
                score += 2
            if cl in doc.lower():
                score += 1
        if score > best_score:
            best_score = score
            best = mapping_index.docs[i]

    if best is not None:
        return best

    if mapping_index.docs:
        return mapping_index.docs[0]
    return ""


# ── Flat RAG index ───────────────────────────────────────────────────────────

class FlatRAGIndex:
    def __init__(
        self,
        docs: list[str],
        client: OpenAI,
        metadata: list[dict] | None = None,
    ):
        self.docs = docs
        self.metadata = metadata or [{} for _ in docs]
        self.client = client
        self.embeddings: np.ndarray | None = None
        self._build()

    def _build(self):
        if not self.docs:
            self.embeddings = np.zeros((0, 1), dtype=np.float32)
            return

        print(f"  Embedding {len(self.docs)} documents...")
        response = self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=self.docs,
        )
        vecs = [e.embedding for e in response.data]
        self.embeddings = np.array(vecs, dtype=np.float32)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / np.maximum(norms, 1e-9)

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        ground_truth_filters: dict | None = None,
        concept_label: str | None = None,
    ) -> tuple[list[str], bool | None, float]:
        if self.embeddings is None or len(self.docs) == 0:
            return [], None, 0.0

        response = self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[query],
        )
        q = np.array(response.data[0].embedding, dtype=np.float32)
        q = q / max(np.linalg.norm(q), 1e-9)

        sims = self.embeddings @ q
        k = min(top_k, len(self.docs))
        top_idx = np.argsort(sims)[::-1][:k]
        max_sim = float(sims[top_idx[0]]) if len(top_idx) else 0.0

        retrieved_docs = [self.docs[i] for i in top_idx]
        recall: bool | None = None
        if ground_truth_filters is not None:
            recall = any(
                _doc_is_correct(
                    self.docs[i],
                    self.metadata[i],
                    ground_truth_filters,
                    concept_label,
                )
                for i in top_idx
            )

        return retrieved_docs, recall, max_sim


def _doc_is_correct(
    doc: str,
    meta: dict,
    ground_truth_filters: dict,
    concept_label: str | None,
) -> bool:
    gt_fields = set(ground_truth_filters.keys())
    if gt_fields & set(meta.get("field_names", [])):
        return True
    if any(f in doc for f in gt_fields):
        return True
    if concept_label and concept_label.lower() in doc.lower():
        return True
    return False


# ── LLM pipeline ───────────────────────────────────────────────────────────────

def _join_docs(docs: list[str]) -> str:
    if not docs:
        return "(no context)"
    return "\n---\n".join(docs)


def parse_llm_json(text: str) -> dict:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return {"filters": {}}
        parsed = json.loads(match.group())

    if isinstance(parsed, dict) and "filters" in parsed:
        return parsed
    if isinstance(parsed, dict):
        return {"filters": parsed}
    return {"filters": {}}


def extract_verbalized_confidence(raw: dict) -> Optional[float]:
    """
    Pull the model's self-reported confidence out of a parsed LLM response,
    validating it rather than trusting it blindly — a model can return a
    string, an out-of-range number, or omit it entirely.
    Returns None (not 0.0) when genuinely absent, so a missing value is
    distinguishable from a model that confidently reported zero.
    """
    if not isinstance(raw, dict) or "confidence" not in raw:
        return None
    try:
        val = float(raw["confidence"])
    except (TypeError, ValueError):
        return None
    if val != val:  # NaN check without importing math
        return None
    return max(0.0, min(1.0, val))


def run_rag_query(
    query: str,
    context_docs: list[str],
    client: OpenAI,
    model: str,
) -> dict:
    user_msg = f"Context:\n{_join_docs(context_docs)}\n\nQuery: {query}"
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
    )
    raw = response.choices[0].message.content or ""
    try:
        return parse_llm_json(raw)
    except json.JSONDecodeError:
        print(f"  Warning: failed to parse LLM JSON for query: {query!r}")
        return {"filters": {}, "raw": raw}


# ── Filter scoring ─────────────────────────────────────────────────────────────

def normalize_filters(raw: Any) -> dict:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        if "filters" in raw and isinstance(raw["filters"], dict):
            return raw["filters"]
        if all(isinstance(v, dict) for v in raw.values()):
            return raw
        return raw

    if isinstance(raw, list):
        out: dict = {}
        for item in raw:
            if not isinstance(item, dict):
                continue
            field = item.get("field") or item.get("name")
            if not field:
                continue
            hint = item.get("value_hint")
            if hint is None and "value" in item:
                out[field] = {"value": item["value"]}
            elif isinstance(hint, dict):
                out[field] = dict(hint)
            elif hint is not None:
                out[field] = {"value": hint}
            else:
                out[field] = {}
        return out

    return {}


def _constraint_satisfied(predicted: dict, expected: dict) -> bool:
    if not predicted:
        return False

    if "value" in expected:
        exp = str(expected["value"]).lower()
        if "value" in predicted:
            return str(predicted["value"]).lower() == exp
        return False

    pmin = predicted.get("min")
    pmax = predicted.get("max")
    emin = expected.get("min")
    emax = expected.get("max")

    if emin is not None or emax is not None:
        if pmin is None and pmax is None:
            return False
        if emin is not None and pmax is not None and pmax < emin:
            return False
        if emax is not None and pmin is not None and pmin > emax:
            return False
        return True

    return bool(predicted)


def filter_match(predicted: dict, ground_truth: dict) -> bool:
    if not ground_truth:
        return not predicted
    for field, expected in ground_truth.items():
        if field not in predicted:
            return False
        if not _constraint_satisfied(predicted[field], expected):
            return False
    return True


# ── Calibration metrics ───────────────────────────────────────────────────────

def _auroc(pairs: list[tuple[float, bool]]) -> Optional[float]:
    """
    AUROC via the rank-sum formula — does confidence correctly rank correct
    answers above wrong ones? 0.5 = no better than random; 1.0 = perfect
    separation. Implemented from scratch (no scipy dependency) with average
    ranks for ties, since a constant confidence (e.g. rag_oracle's old
    hardcoded 1.0) produces nothing BUT ties and must resolve to exactly 0.5,
    not an error or an artificially high score.
    """
    scores = np.array([p[0] for p in pairs], dtype=np.float64)
    labels = np.array([bool(p[1]) for p in pairs], dtype=np.bool_)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return None  # undefined — every result was the same class

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-indexed average rank for the tied block
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1

    rank_sum_pos = ranks[labels].sum()
    auroc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auroc)


def _ece(pairs: list[tuple[float, bool]], n_bins: int = 10) -> float:
    """
    Expected Calibration Error: bins predictions by stated confidence, and
    measures how far each bin's actual accuracy is from its average stated
    confidence, weighted by bin size. 0 = perfectly calibrated.
    """
    scores = np.array([p[0] for p in pairs], dtype=np.float64)
    labels = np.array([bool(p[1]) for p in pairs], dtype=np.float64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(scores)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (scores >= lo) & (scores <= hi if hi == 1.0 else scores < hi)
        count = int(in_bin.sum())
        if count == 0:
            continue
        bin_acc = labels[in_bin].mean()
        bin_conf = scores[in_bin].mean()
        ece += (count / n) * abs(bin_acc - bin_conf)
    return float(ece)


# ── Evaluator ──────────────────────────────────────────────────────────────────

class Evaluator:
    def __init__(
        self,
        condition: str,
        kg_path: str | Path | None = None,
        model: str = DEFAULT_MODEL,
        top_k: int = DEFAULT_TOP_K,
    ):
        if condition not in CONDITIONS:
            raise ValueError(f"Unknown condition: {condition}")

        self.condition = condition
        self.model = model
        self.top_k = top_k

        if kg_path is None:
            kg_path = ROOT / "output" / "lumin_kg.json"
        kg_path = Path(kg_path)
        if not kg_path.exists():
            raise FileNotFoundError(
                f"KG not found at {kg_path}. Place lumin_kg.json in output/."
            )

        print(f"Loading KG from {kg_path}...")
        with open(kg_path, encoding="utf-8") as f:
            self.G = nx.node_link_graph(json.load(f))
        print(
            f"  {self.G.number_of_nodes()} nodes, "
            f"{self.G.number_of_edges()} edges"
        )

        self.embedding_client = get_embedding_client()
        self.llm_client = get_llm_client(model)

        self.concept_index: FlatRAGIndex | None = None
        self.mapping_index: FlatRAGIndex | None = None
        self.traversal: LUMINTraversal | None = None

        if condition == "rag_concept_docs":
            print("Building RAG indices...")
            docs, meta = build_concept_docs(self.G)
            self.concept_index = FlatRAGIndex(docs, self.embedding_client, meta)
            self._save_docs(docs, meta, kg_path.parent / "concept_docs.json")
        elif condition in ("rag_mapping_docs", "rag_oracle"):
            print("Building RAG indices...")
            docs, meta = build_mapping_docs(self.G)
            self.mapping_index = FlatRAGIndex(docs, self.embedding_client, meta)
            self._save_docs(docs, meta, kg_path.parent / "mapping_docs.json")
        elif condition == "kg":
            self.traversal = LUMINTraversal(str(kg_path))

    @staticmethod
    def _save_docs(docs: list[str], metadata: list[dict], path: Path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                [{"doc": d, "metadata": m} for d, m in zip(docs, metadata)],
                f,
                indent=2,
            )
        print(f"  Saved {len(docs)} docs → {path}")

    def run(self, test_cases: list[dict]) -> list[EvalResult]:
        results = []
        for i, case in enumerate(test_cases, 1):
            print(f"\n[{i}/{len(test_cases)}] {case['query']!r} ({case.get('tier', '?')})")
            result = self.run_one(case)
            tag = "OK" if result.correct else "MISS"
            recall_str = ""
            if result.retrieval_recall is not None:
                recall_str = f"  recall={'Y' if result.retrieval_recall else 'N'}"
            print(f"  {tag}  conf={result.confidence:.3f}{recall_str}")
            results.append(result)
        return results

    def run_one(self, case: dict) -> EvalResult:
        query = case["query"]
        tier = case.get("tier", "T0")
        gt = case.get("ground_truth_filters", {})
        concept_label = case.get("concept_label")

        context_docs: list[str] = []
        recall: bool | None = None
        confidence = 0.0
        predicted: dict = {}
        verbalized_confidence: Optional[float] = None

        if self.condition == "rag_concept_docs":
            assert self.concept_index is not None
            context_docs, recall, confidence = self.concept_index.retrieve(
                query, self.top_k, gt, concept_label
            )
            raw = run_rag_query(query, context_docs, self.llm_client, self.model)
            predicted = normalize_filters(raw)
            verbalized_confidence = extract_verbalized_confidence(raw)

        elif self.condition == "rag_mapping_docs":
            assert self.mapping_index is not None
            context_docs, recall, confidence = self.mapping_index.retrieve(
                query, self.top_k, gt, concept_label
            )
            raw = run_rag_query(query, context_docs, self.llm_client, self.model)
            predicted = normalize_filters(raw)
            verbalized_confidence = extract_verbalized_confidence(raw)

        elif self.condition == "rag_oracle":
            assert self.mapping_index is not None
            oracle_doc = find_oracle_doc(self.mapping_index, gt, concept_label)
            context_docs = [oracle_doc] if oracle_doc else []
            recall = None
            confidence = 1.0
            raw = run_rag_query(query, context_docs, self.llm_client, self.model)
            predicted = normalize_filters(raw)
            verbalized_confidence = extract_verbalized_confidence(raw)

        elif self.condition == "kg":
            assert self.traversal is not None
            traversal_result = self.traversal.query(query)
            predicted = normalize_filters(traversal_result.filters)
            confidence = traversal_result.confidence
            recall = None
            context_docs = []

        elif self.condition == "zero_shot":
            raw = run_rag_query(query, [], self.llm_client, self.model)
            predicted = normalize_filters(raw)
            confidence = 0.5
            recall = None
            context_docs = []
            verbalized_confidence = extract_verbalized_confidence(raw)

        correct = filter_match(predicted, gt)

        return EvalResult(
            query=query,
            tier=tier,
            condition=self.condition,
            predicted_filters=predicted,
            ground_truth_filters=gt,
            correct=correct,
            confidence=confidence,
            context_docs=context_docs,
            retrieval_recall=recall,
            verbalized_confidence=verbalized_confidence,
        )

    def aggregate(self, results: list[EvalResult]) -> dict:
        rag_with_recall = [r for r in results if r.retrieval_recall is not None]

        summary: dict[str, Any] = {
            "n": len(results),
            "accuracy": sum(r.correct for r in results) / max(1, len(results)),
            "retrieval_recall": (
                sum(r.retrieval_recall for r in rag_with_recall)
                / max(1, len(rag_with_recall))
            ),
            "by_tier": {},
        }

        for tier in TIERS:
            tier_results = [r for r in results if r.tier == tier]
            if not tier_results:
                continue
            tier_rag = [r for r in tier_results if r.retrieval_recall is not None]
            summary["by_tier"][tier] = {
                "n": len(tier_results),
                "accuracy": (
                    sum(r.correct for r in tier_results) / len(tier_results)
                ),
                "retrieval_recall": (
                    sum(r.retrieval_recall for r in tier_rag) / max(1, len(tier_rag))
                    if tier_rag
                    else None
                ),
            }

        return summary

    def calibration_curve(self, results: list[EvalResult], confidence_key: str = "confidence") -> list[dict]:
        confidences = [getattr(r, confidence_key) for r in results]
        if any(c is None for c in confidences):
            usable = [(r, c) for r, c in zip(results, confidences) if c is not None]
            skipped = len(results) - len(usable)
            if skipped:
                print(f"  [calibration] skipping {skipped}/{len(results)} results with no {confidence_key}")
            results = [r for r, _ in usable]

        curve = []
        for tau in np.arange(0.1, 1.0, 0.01):
            covered = [r for r in results if getattr(r, confidence_key) >= tau]
            n_covered = len(covered)
            n_total = len(results)
            n_correct = sum(r.correct for r in covered)
            n_wrong = n_covered - n_correct
            curve.append({
                "tau": round(float(tau), 2),
                "coverage": n_covered / max(1, n_total),
                "accuracy": n_correct / max(1, n_covered),
                "silent_failure_rate": n_wrong / max(1, n_covered),
            })
        return curve

    def calibration_metrics(self, results: list[EvalResult], confidence_key: str = "confidence") -> dict:
        """
        AUROC (does confidence rank correct above wrong?) and ECE (does a
        stated confidence of X correspond to being right X% of the time?).
        Separate from calibration_curve's threshold sweep — these summarize
        it into two numbers suitable for comparing conditions side by side.
        """
        pairs = [(getattr(r, confidence_key), r.correct) for r in results
                 if getattr(r, confidence_key) is not None]
        if not pairs:
            return {"auroc": None, "ece": None, "n": 0,
                    "note": f"no {confidence_key} values available"}

        return {
            "auroc": _auroc(pairs),
            "ece": _ece(pairs),
            "n": len(pairs),
        }


# ── Test data ──────────────────────────────────────────────────────────────────

def load_test_cases(path: str | Path | None) -> list[dict]:
    if path is None:
        return list(DEV_TEST_CASES)

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    cases = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt_raw = row.get("ground_truth_filters", "{}")
            if isinstance(gt_raw, str):
                gt = json.loads(gt_raw)
            else:
                gt = gt_raw
            cases.append({
                "query": row["query"],
                "tier": row.get("tier", "T0"),
                "ground_truth_filters": gt,
                "concept_label": row.get("concept_label") or None,
            })
    return cases


def _print_summary(summary: dict):
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Queries:          {summary['n']}")
    print(f"  Accuracy:         {summary['accuracy']:.3f}")
    if summary.get("retrieval_recall") is not None:
        print(f"  Retrieval recall: {summary['retrieval_recall']:.3f}")

    if summary.get("by_tier"):
        print("\n  By tier:")
        for tier, stats in summary["by_tier"].items():
            recall = stats.get("retrieval_recall")
            recall_s = f"{recall:.3f}" if recall is not None else "n/a"
            print(
                f"    {tier}: n={stats['n']}  "
                f"accuracy={stats['accuracy']:.3f}  "
                f"retrieval_recall={recall_s}"
            )


def _print_calibration_anchors(curve: list[dict]):
    anchors = [0.30, 0.55, 0.70, 0.90]
    print("\n  Calibration anchors:")
    print(f"  {'tau':>5}  {'coverage':>8}  {'accuracy':>8}  {'silent_fail':>11}")
    for anchor in anchors:
        point = min(curve, key=lambda p: abs(p["tau"] - anchor))
        print(
            f"  {point['tau']:5.2f}  "
            f"{point['coverage']:8.3f}  "
            f"{point['accuracy']:8.3f}  "
            f"{point['silent_failure_rate']:11.3f}"
        )


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LUMIN evaluation pipeline")
    parser.add_argument(
        "--condition",
        required=True,
        choices=CONDITIONS,
        help="Evaluation condition",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"LLM model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="CSV test dataset path (default: inline dev set)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"RAG retrieval depth (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--kg-path",
        default=None,
        help="Path to lumin_kg.json",
    )
    parser.add_argument(
        "--calibration-curve",
        action="store_true",
        help="Sweep confidence threshold and save calibration_curve.json",
    )
    parser.add_argument(
        "--confidence-key",
        default="confidence",
        choices=["confidence", "verbalized_confidence"],
        help=(
            "Which confidence signal to use for --calibration-curve and the "
            "calibration_metrics in the results file. 'confidence' is the "
            "existing signal (retrieval similarity for RAG, traversal "
            "rho for kg, or a hardcoded constant for rag_oracle/zero_shot). "
            "'verbalized_confidence' is the model's own self-reported "
            "confidence, elicited via the prompt — only present for the "
            "LLM-driven conditions (not kg), and the one to use for a fair "
            "calibration comparison against the KG's rho."
        ),
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "results"),
        help="Output directory for results JSON (default: <repo>/results)",
    )
    args = parser.parse_args()

    test_cases = load_test_cases(args.dataset)
    evaluator = Evaluator(
        condition=args.condition,
        kg_path=args.kg_path,
        model=args.model,
        top_k=args.top_k,
    )

    print(f"\nRunning {len(test_cases)} test cases "
          f"[condition={args.condition}, model={args.model}]")

    results = evaluator.run(test_cases)
    summary = evaluator.aggregate(results)
    _print_summary(summary)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.calibration_curve:
        curve = evaluator.calibration_curve(results, confidence_key=args.confidence_key)
        metrics = evaluator.calibration_metrics(results, confidence_key=args.confidence_key)
        cal_path = out_dir / "calibration_curve.json"
        payload = {
            "condition": args.condition,
            "model": args.model,
            "confidence_key": args.confidence_key,
            "n_queries": len(results),
            "auroc": metrics["auroc"],
            "ece": metrics["ece"],
            "curve": curve,
        }
        with open(cal_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved {cal_path}")
        print(f"  AUROC: {metrics['auroc']}")
        print(f"  ECE:   {metrics['ece']}")
        _print_calibration_anchors(curve)

        has_verbalized = any(r.verbalized_confidence is not None for r in results)
        if args.confidence_key == "confidence" and has_verbalized:
            print(
                "\n  Note: this condition also has verbalized_confidence available. "
                "Re-run with --confidence-key verbalized_confidence to see the "
                "model's own self-reported calibration instead."
            )
    else:
        # Always compute calibration_metrics on the default signal so every
        # results file carries a comparable AUROC/ECE, even without
        # --calibration-curve. Also compute on verbalized_confidence
        # whenever it's actually present, so RAG conditions get both.
        metrics_confidence = evaluator.calibration_metrics(results, confidence_key="confidence")
        metrics_verbalized = None
        if any(r.verbalized_confidence is not None for r in results):
            metrics_verbalized = evaluator.calibration_metrics(results, confidence_key="verbalized_confidence")

        results_path = out_dir / f"eval_results_{args.condition}.json"
        payload = {
            "condition": args.condition,
            "model": args.model,
            "summary": summary,
            "calibration_metrics": {
                "confidence": metrics_confidence,
                "verbalized_confidence": metrics_verbalized,
            },
            "results": [asdict(r) for r in results],
        }
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved {results_path}")
        print(f"  AUROC (confidence):            {metrics_confidence['auroc']}")
        print(f"  ECE   (confidence):            {metrics_confidence['ece']}")
        if metrics_verbalized:
            print(f"  AUROC (verbalized_confidence): {metrics_verbalized['auroc']}")
            print(f"  ECE   (verbalized_confidence): {metrics_verbalized['ece']}")


if __name__ == "__main__":
    main()