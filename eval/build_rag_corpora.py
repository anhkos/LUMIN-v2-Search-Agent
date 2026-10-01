"""
Builds two RAG baseline corpora, mechanically, from the frozen graph export
(lumin_kg.json) — no hand-tuning, no test-query involvement.

  rag_concept_docs_corpus.jsonl  — realistic-status-quo baseline.
      One chunk per schema_leaf node: the raw PDS4 field definition
      (description, data type, min/max, units). Zero jargon-bridging
      information. This is what a scientist sees reading real PDS docs.
      1,580 chunks — also usable as the top tier of a distractor-scaling
      sweep (32 -> ~200 -> ~1,580), or grouped/subsampled for smaller tiers.

  rag_oracle_corpus.jsonl  — information-matched baseline.
      One chunk per concept: its description, every alias, its class, and
      the concrete field(s)+value(s) it grounds to — exactly the
      information LUMINTraversal itself uses to build a filter, nothing
      more. Deliberately does NOT include the schema_leaf's own formal PDS4
      description text — that's general archive documentation, not
      graph-specific bridging knowledge, and belongs in rag_concept_docs
      instead. Keeping that line clean is what makes "information-matched"
      actually mean something rather than quietly giving oracle extra help.
      ~19 chunks (one per concept).

Usage:
    python build_rag_corpora.py lumin_kg.json
"""

import json
import sys
from collections import defaultdict


def load_graph(path):
    data = json.load(open(path, encoding="utf-8"))
    nodes = {n["id"]: n for n in data["nodes"]}
    edges_by_type = defaultdict(list)
    for e in data["edges"]:
        edges_by_type[e["edge_type"]].append(e)
    return nodes, edges_by_type


def build_concept_docs_corpus(nodes):
    """One chunk per schema_leaf — raw PDS4 documentation, no bridging."""
    chunks = []
    for node_id, n in nodes.items():
        if n["node_type"] != "schema_leaf":
            continue
        parts = [
            f"Field: {n['class']}.{n['name']}",
            f"Description: {n['description']}",
            f"Data type: {n['data_type']}",
        ]
        if n.get("min_value") not in (None, ""):
            parts.append(f"Range: {n['min_value']} to {n['max_value']}")
        if n.get("unit"):
            parts.append(f"Unit type: {n['unit']}")
        if n.get("permissible_values"):
            values = [pv.get("value", str(pv)) if isinstance(pv, dict) else str(pv)
                      for pv in n["permissible_values"]]
            parts.append(f"Permissible values: {', '.join(values)}")

        chunks.append({
            "id": node_id,
            "source": "schema_leaf",
            "text": "\n".join(parts),
        })
    return chunks


def build_oracle_corpus(nodes, edges_by_type):
    """
    One chunk per concept — exactly what traversal uses: description,
    aliases, class, and the resolved field+value groundings. No more.
    """
    # alias -> concept (from is_alias_of edges)
    aliases_by_concept = defaultdict(list)
    for e in edges_by_type["is_alias_of"]:
        alias_node = nodes[e["source"]]
        aliases_by_concept[e["target"]].append(alias_node["surface_form"])

    # class -> {slot: schema_field} (from grounded_in edges)
    slot_to_field = defaultdict(dict)
    for e in edges_by_type["grounded_in"]:
        target = nodes[e["target"]]
        slot_to_field[e["source"]][e["slot"]] = f"{target['class']}.{target['name']}"

    # concept -> (class, slot_values) (from instance_of edges)
    concept_grounding = {}
    for e in edges_by_type["instance_of"]:
        concept_grounding[e["source"]] = {
            "class_id": e["target"],
            "slot_values": json.loads(e["slot_values"]) if e.get("slot_values") else {},
        }

    # concept -> [related concept via occurs_during]
    occurs_during = defaultdict(list)
    for e in edges_by_type["occurs_during"]:
        occurs_during[e["source"]].append(e["target"])

    chunks = []
    for node_id, n in nodes.items():
        if n["node_type"] != "concept":
            continue

        parts = [
            f"Concept: {n['label']}",
            f"Description: {n['description']}",
        ]

        aliases = aliases_by_concept.get(node_id, [])
        if aliases:
            parts.append(f"Also known as: {', '.join(aliases)}")

        grounding = concept_grounding.get(node_id)
        if grounding:
            class_id = grounding["class_id"]
            class_label = nodes[class_id]["label"]
            parts.append(f"Type: {class_label}")

            field_lines = []
            for slot, value in grounding["slot_values"].items():
                if slot == "variants" and isinstance(value, list):
                    for i, variant in enumerate(value, 1):
                        field_lines.append(f"  Region variant {i}:")
                        for sub_slot, sub_value in variant.items():
                            field = slot_to_field.get(class_id, {}).get(sub_slot, "(unmapped slot)")
                            field_lines.append(f"    {sub_slot} -> {field}: {sub_value}")
                else:
                    field = slot_to_field.get(class_id, {}).get(slot, "(unmapped slot)")
                    field_lines.append(f"  {slot} -> {field}: {value}")
            if field_lines:
                parts.append("Grounds to these fields:\n" + "\n".join(field_lines))

        related = occurs_during.get(node_id, [])
        if related:
            related_labels = [nodes[r]["label"] for r in related]
            parts.append(f"Occurs during: {', '.join(related_labels)}")

        chunks.append({
            "id": node_id,
            "source": "concept",
            "text": "\n".join(parts),
        })
    return chunks


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "lumin_kg.json"
    nodes, edges_by_type = load_graph(path)

    concept_docs = build_concept_docs_corpus(nodes)
    oracle = build_oracle_corpus(nodes, edges_by_type)

    with open("rag_concept_docs_corpus.jsonl", "w", encoding="utf-8") as f:
        for c in concept_docs:
            f.write(json.dumps(c) + "\n")

    with open("rag_oracle_corpus.jsonl", "w", encoding="utf-8") as f:
        for c in oracle:
            f.write(json.dumps(c) + "\n")

    print(f"rag_concept_docs_corpus.jsonl: {len(concept_docs)} chunks (schema fields, no bridging)")
    print(f"rag_oracle_corpus.jsonl:       {len(oracle)} chunks (graph's own bridging knowledge)")
    print()
    print("Sample oracle chunk:")
    print(oracle[0]["text"])


if __name__ == "__main__":
    main()
