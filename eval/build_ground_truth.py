"""
Ground-truth generator — the missing bridge between the hand-written /
mined test query CSVs (id, tier, query_text, target_concept_or_alias,
notes) and what eval.py's load_test_cases() actually expects
(query, tier, ground_truth_filters [JSON], concept_label).

Ground truth is DERIVED FROM THE FROZEN GRAPH, never hand-typed — the
whole point of this script. A human only had to say which concept a query
targets; this script looks up what that concept actually grounds to.

Known, explicitly-handled limitation: eval.py's current filter_match() /
_constraint_satisfied() has no support for "OR over alternatives" — a
concept with multiple disjoint value sets (e.g. Mars Polar Regions: north
OR south, not both at once) can't be expressed as a single ground-truth
dict without the scorer either over- or under-crediting it. Rather than
silently pick one region and risk scoring queries wrong for the wrong
reason, those rows are routed to needs_review.csv instead of guessed.

Usage:
    python build_ground_truth.py data/lumin_test_queries.csv output/lumin_kg.json
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import networkx as nx


def load_graph(kg_path):
    with open(kg_path, encoding="utf-8") as f:
        return nx.node_link_graph(json.load(f))


def build_concept_lookup(G):
    """label (lowercased) -> concept_id, plus alias surface_form -> concept_id as fallback."""
    by_label = {}
    by_alias = {}
    for node_id, data in G.nodes(data=True):
        if data.get("node_type") == "concept":
            by_label[data["label"].lower()] = node_id
    for u, v, data in G.edges(data=True):
        if data.get("edge_type") == "is_alias_of":
            surface = G.nodes[u].get("surface_form", "")
            by_alias[surface.lower()] = v
    return by_label, by_alias


def ground_concept(G, concept_id):
    """
    Returns one of:
      ("ok", {field: value, ...})           — clean, single ground truth
      ("variants", [{field: value}, ...])   — multiple disjoint value sets;
                                               caller must NOT flatten these
      ("empty", {})                         — concept has no filled slots
    Mirrors the same traversal logic as the fixed eval.py
    _concept_schema_leaves, but kept standalone and more explicit about
    the variants case rather than silently expanding it.
    """
    class_id, slot_values = None, {}
    for succ in G.successors(concept_id):
        if G.nodes[succ].get("node_type") != "class":
            continue
        edge = G.edges.get((concept_id, succ), {})
        if edge.get("edge_type") != "instance_of":
            continue
        class_id = succ
        raw = edge.get("slot_values")
        slot_values = json.loads(raw) if isinstance(raw, str) else (raw or {})
        break
    if class_id is None:
        return "empty", {}

    slot_to_field = {}
    for succ in G.successors(class_id):
        if G.nodes[succ].get("node_type") != "schema_leaf":
            continue
        edge = G.edges.get((class_id, succ), {})
        if edge.get("edge_type") != "grounded_in":
            continue
        slot_to_field[edge.get("slot")] = G.nodes[succ].get("name", succ)

    if "variants" in slot_values and isinstance(slot_values["variants"], list):
        variant_dicts = []
        for variant in slot_values["variants"]:
            d = {}
            for sub_slot, sub_value in variant.items():
                field = slot_to_field.get(sub_slot)
                if field:
                    d[field] = sub_value
            variant_dicts.append(d)
        return "variants", variant_dicts

    result = {}
    for slot, value in slot_values.items():
        field = slot_to_field.get(slot)
        if field:
            result[field] = value
    if not result:
        return "empty", {}
    return "ok", result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_csv", help="hand-written or mined test query CSV")
    ap.add_argument("kg_path", nargs="?", default="output/lumin_kg.json")
    ap.add_argument("--out", default="eval_ready_dataset.csv")
    ap.add_argument("--review-out", default="needs_review.csv")
    args = ap.parse_args()

    G = load_graph(args.kg_path)
    by_label, by_alias = build_concept_lookup(G)

    rows = list(csv.DictReader(open(args.input_csv, encoding="utf-8")))
    ready, review = [], []

    for row in rows:
        target = row.get("target_concept_or_alias", "").strip()
        query = row.get("query_text", row.get("query", "")).strip()
        tier = row.get("tier", "T0")
        rid = row.get("id", "")

        # Compound-query rows (notes mention more than one concept, or the
        # target itself names more than one) — explicitly out of scope for
        # strict single-concept scoring, same as the original design's
        # "compound-vs-ambiguous is an analysis-section item" decision.
        if "+" in target or "compound" in row.get("notes", "").lower():
            review.append({**row, "reason": "compound query — references more than one concept, not scorable as a single ground truth"})
            continue

        concept_id = by_label.get(target.lower()) or by_alias.get(target.lower())
        if concept_id is None:
            review.append({**row, "reason": f"concept/alias '{target}' not found in graph"})
            continue

        status, grounding = ground_concept(G, concept_id)
        concept_label = G.nodes[concept_id]["label"]

        if status == "empty":
            review.append({**row, "reason": f"concept '{concept_label}' has no filled slots to ground to"})
        elif status == "variants":
            review.append({**row, "reason": f"concept '{concept_label}' has {len(grounding)} disjoint value sets (e.g. north/south) — current scorer has no OR support, needs manual ground truth or a scorer fix before this can be auto-generated"})
        else:  # ok
            ready.append({
                "query": query,
                "tier": tier,
                "ground_truth_filters": json.dumps(grounding),
                "concept_label": concept_label,
                "_source_id": rid,  # kept for traceability, eval.py ignores unknown columns
            })

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["query", "tier", "ground_truth_filters", "concept_label", "_source_id"])
        w.writeheader()
        w.writerows(ready)

    if review:
        review_fields = list(rows[0].keys()) + ["reason"]
        with open(args.review_out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=review_fields)
            w.writeheader()
            w.writerows(review)

    print(f"{len(ready)}/{len(rows)} rows converted -> {args.out} (eval.py-ready)")
    if review:
        print(f"{len(review)}/{len(rows)} rows need a human look -> {args.review_out}")
        reasons = {}
        for r in review:
            key = r["reason"].split(" — ")[0].split(" '")[0]
            reasons[key] = reasons.get(key, 0) + 1
        for reason, count in reasons.items():
            print(f"  {count}x: {reason}")


if __name__ == "__main__":
    main()
