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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import networkx as nx
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

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
{"filters": {"field_name": {"min": <number>, "max": <number>, "value": <string>}}}
Include only fields clearly supported by the query and context.
Omit fields you are uncertain about. Use min/max for ranges and value for exact matches."""

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
    """Return (field_name, edge_type, value_hint) for each schema leaf."""
    leaves = []
    for succ in G.successors(concept_id):
        if G.nodes[succ].get("node_type") != "schema_leaf":
            continue
        edge = G.edges.get((concept_id, succ), {})
        edge_type = edge.get("edge_type", "")
        field_name = G.nodes[succ].get("name", succ)
        value_hint = _parse_value_hint(edge.get("value_hint"))
        leaves.append((field_name, edge_type, value_hint))
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
        leaves = _concept_schema_leaves(G, concept_id)
        primary = _pick_primary_leaf(leaves)
        if primary is None:
            text = (
                f"{alias} refers to {concept_label}, "
                f"which has no mapped PDS field."
            )
            field_names: list[str] = []
        else:
            field_name, edge_type, value_hint = primary
            field_names = [field_name]
            if edge_type == "instrument_of":
                instrument = "unknown"
                if isinstance(value_hint, dict):
                    instrument = value_hint.get("value", instrument)
                text = (
                    f"{alias} refers to {concept_label}, "
                    f"which is observed by the {instrument} instrument."
                )
            else:
                hint_str = format_value_hint(value_hint)
                text = (
                    f"{alias} refers to {concept_label}, "
                    f"which maps to the PDS field {field_name} with {hint_str}."
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
            kg_path = Path(__file__).parent / "output" / "lumin_kg.json"
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

        if self.condition == "rag_concept_docs":
            assert self.concept_index is not None
            context_docs, recall, confidence = self.concept_index.retrieve(
                query, self.top_k, gt, concept_label
            )
            raw = run_rag_query(query, context_docs, self.llm_client, self.model)
            predicted = normalize_filters(raw)

        elif self.condition == "rag_mapping_docs":
            assert self.mapping_index is not None
            context_docs, recall, confidence = self.mapping_index.retrieve(
                query, self.top_k, gt, concept_label
            )
            raw = run_rag_query(query, context_docs, self.llm_client, self.model)
            predicted = normalize_filters(raw)

        elif self.condition == "rag_oracle":
            assert self.mapping_index is not None
            oracle_doc = find_oracle_doc(self.mapping_index, gt, concept_label)
            context_docs = [oracle_doc] if oracle_doc else []
            recall = None
            confidence = 1.0
            raw = run_rag_query(query, context_docs, self.llm_client, self.model)
            predicted = normalize_filters(raw)

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

    def calibration_curve(self, results: list[EvalResult]) -> list[dict]:
        curve = []
        for tau in np.arange(0.1, 1.0, 0.01):
            covered = [r for r in results if r.confidence >= tau]
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
        "--output",
        default=".",
        help="Output directory for results JSON",
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
        curve = evaluator.calibration_curve(results)
        cal_path = out_dir / "calibration_curve.json"
        payload = {
            "condition": args.condition,
            "model": args.model,
            "n_queries": len(results),
            "curve": curve,
        }
        with open(cal_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved {cal_path}")
        _print_calibration_anchors(curve)
    else:
        results_path = out_dir / f"eval_results_{args.condition}.json"
        payload = {
            "condition": args.condition,
            "model": args.model,
            "summary": summary,
            "results": [asdict(r) for r in results],
        }
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved {results_path}")


if __name__ == "__main__":
    main()
