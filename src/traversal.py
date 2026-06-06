"""
traversal.py
------------
LUMIN Knowledge Graph traversal module.

Three components:
    1. VectorIndex   — embeds alias/concept nodes, finds nearest entry point
                       for any query term (known or unknown jargon)
    2. Traverser     — BFS from entry point to schema leaf nodes,
                       following typed edges in priority order
    3. TraversalResult — structured output with filters + confidence signal

This is the self-RAG analog: instead of learned reflection tokens,
confidence is derived from graph topology (entry similarity, path length,
edge type quality). No supervision required.

Usage:
    python traversal.py

Requires:
    pip install networkx openai numpy scikit-learn
    export OPENAI_API_KEY=your_key
"""

import json
import os
import math
from collections import deque
from pathlib import Path
import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import Optional
from openai import OpenAI

# ── Configuration ──────────────────────────────────────────────────────────────

EMBEDDING_MODEL   = "text-embedding-3-small"   # cheap + fast; swap to large if needed
ENTRY_TOP_K       = 3       # number of candidate entry nodes to consider
BFS_MAX_DEPTH     = 5       # max hops before giving up
CONFIDENCE_THRESH = 0.55    # below this → flag as uncertain

# Edge type priority (higher = more confident traversal)
EDGE_PRIORITY = {
    "is_alias_of":    1.00,
    "corresponds_to": 1.00,
    "measured_by":    0.90,
    "instrument_of":  0.80,
    "related_to":     0.50,   # weak cross-concept edge
}

# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class TraversalResult:
    query_term:       str
    entry_node:       str                    # node id of entry point
    entry_node_type:  str                    # alias | concept
    entry_similarity: float                  # cosine sim to entry node
    path:             list[str]              # full node id path
    edge_types:       list[str]              # edge types along path
    schema_leaves:    list[dict]             # schema leaf nodes reached
    filters:          list[dict]             # extracted PDS filter dicts
    confidence:       float                  # 0–1 confidence signal
    uncertain:        bool                   # True if below threshold
    explanation:      str                    # human-readable reasoning

    def pretty(self):
        lines = [
            f"Query:       {self.query_term}",
            f"Entry node:  {self.entry_node}  ({self.entry_node_type})",
            f"Similarity:  {self.entry_similarity:.3f}",
            f"Path:        {' -> '.join(self.path)}",
            f"Edge types:  {' -> '.join(self.edge_types)}",
            f"Confidence:  {self.confidence:.3f}  {'[UNCERTAIN]' if self.uncertain else '[OK]'}",
            f"Filters:     {self.filters}",
            f"Explanation: {self.explanation}",
        ]
        return "\n".join(lines)


# ── Vector Index ───────────────────────────────────────────────────────────────

class VectorIndex:
    """
    Embeds all alias and concept nodes in the KG.
    At query time, finds the nearest node by cosine similarity.

    Search space is restricted to alias + concept nodes only —
    never schema leaves, to preserve the traversal path.
    """

    def __init__(self, G: nx.DiGraph, client: OpenAI):
        self.G = G
        self.client = client
        self.node_ids = []
        self.texts = []
        self.embeddings = None
        self._build()

    def _build(self):
        print("Building vector index over alias + concept nodes...")
        for node_id, data in self.G.nodes(data=True):
            node_type = data.get("node_type")
            if node_type == "alias":
                text = data.get("surface_form", node_id)
            elif node_type == "concept":
                # richer text for concept nodes: label + description
                text = f"{data.get('label', '')}. {data.get('description', '')}"
            else:
                continue   # skip schema leaves
            self.node_ids.append(node_id)
            self.texts.append(text)

        print(f"  Embedding {len(self.texts)} nodes...")
        response = self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=self.texts
        )
        vecs = [e.embedding for e in response.data]
        self.embeddings = np.array(vecs, dtype=np.float32)
        # L2-normalise for fast cosine via dot product
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / np.maximum(norms, 1e-9)
        print(f"  Done. Index size: {self.embeddings.shape}")

    def query(self, term: str, top_k: int = ENTRY_TOP_K) -> list[tuple[str, float]]:
        """
        Embed a query term and return top-k (node_id, similarity) pairs.
        """
        response = self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[term]
        )
        q = np.array(response.data[0].embedding, dtype=np.float32)
        q = q / max(np.linalg.norm(q), 1e-9)

        sims = self.embeddings @ q          # cosine similarity for all nodes
        top_k_idx = np.argsort(sims)[::-1][:top_k]
        return [(self.node_ids[i], float(sims[i])) for i in top_k_idx]


# ── BFS Traverser ──────────────────────────────────────────────────────────────

class Traverser:
    """
    BFS from an entry node toward schema leaf nodes.
    Follows edges in priority order, stops at first schema leaf reached.
    Returns the best path found across all BFS branches.
    """

    def __init__(self, G: nx.DiGraph):
        self.G = G

    def traverse(self, start_node: str) -> Optional[tuple[list[str], list[str]]]:
        """
        BFS from start_node to a schema leaf.
        Returns (path, edge_types) or None if no leaf reachable within depth.
        """
        # queue entries: (current_node, path_so_far, edge_types_so_far)
        queue = deque([(start_node, [start_node], [])])
        visited = {start_node}
        best_path = None
        best_score = -1.0

        while queue:
            current, path, edges = queue.popleft()

            if len(path) > BFS_MAX_DEPTH + 1:
                continue

            node_type = self.G.nodes[current].get("node_type")

            # termination: reached a schema leaf
            if node_type == "schema_leaf" and len(path) > 1:
                score = self._path_score(edges, path)
                if score > best_score:
                    best_score = score
                    best_path = (path, edges)
                continue   # don't expand further from leaf

            # expand neighbors, sorted by edge priority (high → low)
            neighbors = list(self.G.successors(current))
            neighbors.sort(
                key=lambda n: EDGE_PRIORITY.get(
                    self.G.edges[current, n].get("edge_type", ""), 0
                ),
                reverse=True
            )

            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    edge_type = self.G.edges[current, neighbor].get("edge_type", "unknown")
                    queue.append((neighbor, path + [neighbor], edges + [edge_type]))

        return best_path

    def _path_score(self, edge_types: list[str], path: list[str]) -> float:
        """Score a path by the product of its edge priorities, penalised by length."""
        if not edge_types:
            return 0.0
        priority_product = math.prod(
            EDGE_PRIORITY.get(et, 0.5) for et in edge_types
        )
        length_penalty = 1.0 / len(edge_types)
        terminal_edge = self.G.edges[path[-2], path[-1]]
        value_hint_bonus = 1.2 if terminal_edge.get("value_hint") else 1.0
        return priority_product * length_penalty * value_hint_bonus


# ── Filter Extractor ───────────────────────────────────────────────────────────

def extract_filters(G: nx.DiGraph, path: list[str], edge_types: list[str]) -> list[dict]:
    """
    Walk the path and collect value_hints from edges that have them.
    Returns a list of filter dicts, each with:
        field, class, value_hint, edge_type
    """
    filters = []
    for i, edge_type in enumerate(edge_types):
        u, v = path[i], path[i + 1]
        edge_data = G.edges[u, v]
        value_hint_raw = edge_data.get("value_hint")
        v_data = G.nodes[v]

        if v_data.get("node_type") == "schema_leaf":
            f = {
                "field":      v_data.get("name"),
                "class":      v_data.get("class"),
                "ldd":        v_data.get("ldd"),
                "edge_type":  edge_type,
                "data_type":  v_data.get("data_type"),
                "unit":       v_data.get("unit"),
            }
            if value_hint_raw:
                try:
                    f["value_hint"] = json.loads(value_hint_raw)
                except Exception:
                    f["value_hint"] = value_hint_raw
            else:
                f["value_hint"] = None

            # add permissible values if enumerated
            if v_data.get("enumerated") and v_data.get("permissible_values"):
                f["permissible_values"] = [
                    pv["value"] for pv in v_data["permissible_values"]
                ]
            filters.append(f)

    return filters


# ── Confidence Scorer ──────────────────────────────────────────────────────────

def score_confidence(
    entry_similarity: float,
    path: list[str],
    edge_types: list[str],
    G: nx.DiGraph
) -> tuple[float, str]:
    """
    Derives confidence from graph topology — no supervision needed.
    This is the self-RAG analog: instead of learned reflection tokens,
    we use structural signals.

    Factors:
        1. Entry similarity  — how close was the query to the entry node?
        2. Path length       — longer paths = more uncertain
        3. Edge type quality — weak edges (related_to) reduce confidence
        4. Exact alias match — did we land on a known alias vs concept node?

    Returns (confidence_score, explanation_string)
    """
    explanations = []

    # Factor 1: entry similarity (already 0-1, cosine)
    sim_score = entry_similarity
    if sim_score >= 0.90:
        explanations.append(f"strong entry match ({sim_score:.2f})")
    elif sim_score >= 0.70:
        explanations.append(f"moderate entry match ({sim_score:.2f})")
    else:
        explanations.append(f"weak entry match ({sim_score:.2f}) — possible wrong-node latch")

    # Factor 2: path length penalty (1 hop = 1.0, each extra hop * 0.85)
    n_hops = len(edge_types)
    length_score = 0.85 ** max(0, n_hops - 1)
    if n_hops <= 2:
        explanations.append(f"short path ({n_hops} hops)")
    elif n_hops <= 4:
        explanations.append(f"medium path ({n_hops} hops)")
    else:
        explanations.append(f"long path ({n_hops} hops) — higher uncertainty")

    # Factor 3: edge quality (penalise weak edges)
    edge_score = math.prod(
        EDGE_PRIORITY.get(et, 0.5) for et in edge_types
    ) if edge_types else 1.0
    if any(et == "related_to" for et in edge_types):
        explanations.append("traversed weak related_to edge — possible wrong concept")

    # Factor 4: entry node type (alias = more precise than concept)
    entry_type = G.nodes[path[0]].get("node_type") if path else "unknown"
    type_score = 1.0 if entry_type == "alias" else 0.85
    if entry_type == "alias":
        explanations.append("entered via known alias (precise)")
    else:
        explanations.append("entered via concept node (approximate — term may be unknown jargon)")

    # Combined score
    confidence = sim_score * length_score * edge_score * type_score
    confidence = min(1.0, max(0.0, confidence))

    explanation = "; ".join(explanations)
    return confidence, explanation


# ── Main Interface ─────────────────────────────────────────────────────────────

class LUMINTraversal:
    """
    Main interface. Given a query term, returns a TraversalResult with
    extracted PDS filters and a confidence signal.
    """

    def __init__(self, kg_path: str = None):
        if kg_path is None:
            kg_path = Path(__file__).parent.parent / "output" / "lumin_kg.json"
        print(f"Loading KG from {kg_path}...")
        with open(kg_path, encoding="utf-8") as f:
            kg_data = json.load(f)
        self.G = nx.node_link_graph(kg_data)
        print(f"  {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")

        self.client   = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.index    = VectorIndex(self.G, self.client)
        self.traverser = Traverser(self.G)

    def query(self, term: str) -> TraversalResult:
        """
        Given a natural language term (known or unknown jargon),
        return filters and confidence.
        """
        # Stage 1: exact alias lookup — skip embedding if we have a direct hit
        normalized = term.lower().replace(" ", "_")
        alias_id = f"alias:{normalized}"
        if alias_id in self.G:
            candidates = [(alias_id, 1.0)]
        else:
            # Stage 2: embedding similarity over alias + concept nodes
            candidates = self.index.query(term, top_k=ENTRY_TOP_K)

        best_result = None

        for entry_node, entry_sim in candidates:
            # Step 2: BFS from this entry node to a schema leaf
            traversal = self.traverser.traverse(entry_node)
            if traversal is None:
                continue

            path, edge_types = traversal

            # Step 3: extract filters from path
            filters = extract_filters(self.G, path, edge_types)

            # Step 4: score confidence
            confidence, explanation = score_confidence(
                entry_sim, path, edge_types, self.G
            )

            result = TraversalResult(
                query_term       = term,
                entry_node       = entry_node,
                entry_node_type  = self.G.nodes[entry_node].get("node_type", "unknown"),
                entry_similarity = entry_sim,
                path             = path,
                edge_types       = edge_types,
                schema_leaves    = [self.G.nodes[path[-1]]],
                filters          = filters,
                confidence       = confidence,
                uncertain        = confidence < CONFIDENCE_THRESH,
                explanation      = explanation,
            )

            # keep the highest-confidence result across candidates
            if best_result is None or result.confidence > best_result.confidence:
                best_result = result

        if best_result is None:
            # complete traversal failure — return uncertain empty result
            return TraversalResult(
                query_term       = term,
                entry_node       = candidates[0][0] if candidates else "none",
                entry_node_type  = "unknown",
                entry_similarity = candidates[0][1] if candidates else 0.0,
                path             = [],
                edge_types       = [],
                schema_leaves    = [],
                filters          = [],
                confidence       = 0.0,
                uncertain        = True,
                explanation      = "No schema leaf reachable within depth limit",
            )

        return best_result


# ── Demo ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    lumint = LUMINTraversal()

    test_queries = [
        # T0/T1 — should be high confidence
        "southern summer",
        "HiRISE",
        "north pole",

        # T2 — moderate confidence expected
        "southern hemisphere warm season",
        "subsurface ice layering radar",

        # T3 / unknown — lower confidence, may flag uncertain
        "aphelion season near perihelion transition",
        "NPLD stratigraphic profiles from orbit",
        "tau surge during global encirclement",

        # genuinely out-of-scope — should flag uncertain
        "coffee shop near JPL",
    ]

    print("\n" + "=" * 70)
    print("LUMIN KG Traversal -- Demo")
    print("=" * 70)

    for query in test_queries:
        print(f"\n{'-' * 70}")
        result = lumint.query(query)
        print(result.pretty())
