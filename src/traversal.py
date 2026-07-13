"""
traversal.py
------------
LUMIN Knowledge Graph traversal module.

Three components:
    1. VectorIndex    — embeds alias/concept nodes; finds nearest entry point
                        for any query term (known or unknown jargon) via
                        cosine similarity over topology + embedding features
    2. Traverser      — best-first search (Dijkstra) from entry point to
                        schema leaf, directly maximising the confidence score
    3. TraversalResult — structured output with filters + confidence signal

Confidence signal ρ = ρ_sim × ρ_path × ρ_type × ρ_margin × ρ_agreement
    ρ_sim       cosine similarity to entry node (embedding-based)
    ρ_path      Dijkstra-optimal path score: product(edge_priorities) × 0.85^(hops-1)
                — the traversal directly maximises this, so the reported ρ_path
                  is the best achievable from the entry node
    ρ_type      1.0 for alias entry (precise), 0.85 for concept entry (approximate)
    ρ_margin    entry ambiguity: gap between top-1 and top-2 embedding similarities
    ρ_agreement cross-candidate consistency: do the k entry candidates converge
                on the same terminal schema leaf?

Note on supervision: ρ requires no learned confidence model and no reflection-
token training data. A single scalar threshold τ is tuned on the validation
split; the full risk-coverage curve is reported so the choice of τ is
transparent. ρ_sim depends on the embedding model (not pure graph topology);
all other factors are derived from graph structure alone.

Usage:
    python traversal.py

Requires:
    pip install networkx openai numpy
    export OPENAI_API_KEY=your_key
"""

import json
import os
import math
import heapq
import numpy as np
import networkx as nx
from dataclasses import dataclass
from itertools import count
from typing import Optional
from openai import OpenAI

# ── Configuration ──────────────────────────────────────────────────────────────

EMBEDDING_MODEL   = "text-embedding-3-small"
ENTRY_TOP_K       = 3       # candidate entry nodes from embedding search
MAX_DEPTH         = 5       # max hops in Dijkstra before giving up
CONFIDENCE_THRESH = 0.55    # below this → flag uncertain (tuned on val split)

# Edge type priority — higher = lower Dijkstra cost = preferred traversal
# Choosing these is a design decision; ablation in the paper tests sensitivity
EDGE_PRIORITY = {
    "is_alias_of":    1.00,
    "corresponds_to": 1.00,
    "measured_by":    0.90,
    "instrument_of":  0.80,
    "related_to":     0.50,   # weak cross-concept edge; heavily penalised
}

# Length decay per hop (after the first) — tunable, ablated in paper
LENGTH_DECAY = 0.85

# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class TraversalResult:
    query_term:       str
    entry_node:       str
    entry_node_type:  str        # "alias" | "concept"
    entry_similarity: float      # cosine sim to entry node
    entry_margin:     float      # sim[0] - sim[1]; high = unambiguous entry
    path:             list       # full node-id path from entry to schema leaf
    edge_types:       list       # edge type labels along path
    path_rho:         float      # ρ_path = Dijkstra-optimal path score
    path_agreement:   float      # 1.0 if k candidates converge, 0.75 if not
    schema_leaves:    list       # schema leaf node data dicts
    filters:          list       # extracted PDS filter dicts
    confidence:       float      # combined ρ
    uncertain:        bool       # True if ρ < CONFIDENCE_THRESH
    explanation:      str

    def pretty(self):
        lines = [
            f"Query:        {self.query_term}",
            f"Entry node:   {self.entry_node}  ({self.entry_node_type})",
            f"Similarity:   {self.entry_similarity:.3f}  "
            f"(margin: {self.entry_margin:.3f})",
            f"Path:         {' → '.join(self.path)}",
            f"Edge types:   {' → '.join(self.edge_types)}",
            f"ρ_path:       {self.path_rho:.3f}  "
            f"agreement: {self.path_agreement:.2f}",
            f"Confidence:   {self.confidence:.3f}  "
            f"{'⚠ UNCERTAIN' if self.uncertain else '✓ OK'}",
            f"Filters:      {self.filters}",
            f"Explanation:  {self.explanation}",
        ]
        return "\n".join(lines)


# ── Vector Index ───────────────────────────────────────────────────────────────

class VectorIndex:
    """
    Embeds all alias and concept nodes. At query time, finds nearest nodes
    by cosine similarity. Search space restricted to alias + concept layers —
    never schema leaves — to preserve the traversal path.
    """

    def __init__(self, G: nx.DiGraph, client: OpenAI):
        self.G = G
        self.client = client
        self.node_ids: list[str] = []
        self.texts:    list[str] = []
        self.embeddings = None
        self._build()

    def _build(self):
        print("Building vector index over alias + concept nodes...")
        for node_id, data in self.G.nodes(data=True):
            nt = data.get("node_type")
            if nt == "alias":
                text = data.get("surface_form", node_id)
            elif nt == "concept":
                text = f"{data.get('label', '')}. {data.get('description', '')}"
            else:
                continue
            self.node_ids.append(node_id)
            self.texts.append(text)

        print(f"  Embedding {len(self.texts)} nodes...")
        resp = self.client.embeddings.create(
            model=EMBEDDING_MODEL, input=self.texts)
        vecs = np.array([e.embedding for e in resp.data], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        self.embeddings = vecs / np.maximum(norms, 1e-9)
        print(f"  Index ready: {self.embeddings.shape}")

    def query(self, term: str, top_k: int = ENTRY_TOP_K
              ) -> list[tuple[str, float]]:
        """Return top-k (node_id, cosine_similarity) pairs."""
        resp = self.client.embeddings.create(
            model=EMBEDDING_MODEL, input=[term])
        q = np.array(resp.data[0].embedding, dtype=np.float32)
        q = q / max(np.linalg.norm(q), 1e-9)
        sims = self.embeddings @ q
        idx  = np.argsort(sims)[::-1][:top_k]
        return [(self.node_ids[i], float(sims[i])) for i in idx]


# ── Dijkstra Traverser ─────────────────────────────────────────────────────────

class Traverser:
    """
    Best-first search from an entry node to a schema leaf, directly
    maximising the path confidence score ρ_path.

    ρ_path = product(edge_priorities) × LENGTH_DECAY^(max(0, hops-1))

    Taking -log converts this to a sum of non-negative edge costs, which
    Dijkstra minimises. The first schema leaf popped from the priority queue
    is the ρ_path-optimal one — search and calibration use the same objective,
    so the reported ρ_path is the maximum achievable from the entry node.

    This resolves the inconsistency in BFS, where the shortest-hop path may
    have lower ρ_path than a longer path through higher-priority edges.
    """

    # Pre-compute per-edge costs (-log of priority)
    LOG_DECAY = -math.log(LENGTH_DECAY)  # -log(0.85) ≈ 0.163 per extra hop
    EDGE_COST = {et: -math.log(max(p, 1e-9))
                 for et, p in EDGE_PRIORITY.items()}

    def __init__(self, G: nx.DiGraph):
        self.G = G

    def traverse(self, start_node: str
                 ) -> Optional[tuple[list[str], list[str], float]]:
        """
        Dijkstra from start_node to the ρ_path-optimal schema leaf.
        Returns (path, edge_types, rho_path) or None if unreachable.
        """
        _cnt = count()   # tie-breaker — avoids comparing list elements

        # (cost, counter, node_id, path, edge_types)
        pq = [(0.0, next(_cnt), start_node, [start_node], [])]
        best_cost: dict[str, float] = {start_node: 0.0}

        while pq:
            cost, _, current, path, edges = heapq.heappop(pq)

            # Stale entry
            if cost > best_cost.get(current, math.inf) + 1e-9:
                continue

            nt = self.G.nodes[current].get("node_type")

            # Reached a schema leaf — this is the ρ_path-optimal path
            if nt == "schema_leaf" and len(path) > 1:
                rho_path = math.exp(-cost)
                return path, edges, rho_path

            if len(path) > MAX_DEPTH + 1:
                continue

            for neighbor in self.G.successors(current):
                et          = self.G.edges[current, neighbor].get(
                                "edge_type", "unknown")
                edge_cost   = self.EDGE_COST.get(et, -math.log(0.5))
                # Length decay applies from the second hop onward
                length_cost = self.LOG_DECAY if len(edges) >= 1 else 0.0
                new_cost    = cost + edge_cost + length_cost

                if new_cost < best_cost.get(neighbor, math.inf):
                    best_cost[neighbor] = new_cost
                    heapq.heappush(
                        pq,
                        (new_cost, next(_cnt), neighbor,
                         path + [neighbor], edges + [et])
                    )

        return None


# ── Filter Extractor ───────────────────────────────────────────────────────────

def extract_filters(G: nx.DiGraph,
                    path: list[str],
                    edge_types: list[str]) -> list[dict]:
    filters = []
    for i, et in enumerate(edge_types):
        u, v     = path[i], path[i + 1]
        vd       = G.nodes[v]
        edge_d   = G.edges[u, v]
        if vd.get("node_type") != "schema_leaf":
            continue
        hint_raw = edge_d.get("value_hint")
        f = {
            "field":     vd.get("name"),
            "class":     vd.get("class"),
            "ldd":       vd.get("ldd"),
            "edge_type": et,
            "data_type": vd.get("data_type"),
            "unit":      vd.get("unit"),
            "value_hint": None,
        }
        if hint_raw:
            try:
                f["value_hint"] = json.loads(hint_raw)
            except Exception:
                f["value_hint"] = hint_raw
        if vd.get("enumerated") and vd.get("permissible_values"):
            f["permissible_values"] = [
                pv["value"] for pv in vd["permissible_values"]]
        filters.append(f)
    return filters


# ── Confidence Scorer ──────────────────────────────────────────────────────────

def score_confidence(
    entry_similarity: float,
    path:             list[str],
    edge_types:       list[str],
    path_rho:         float,      # Dijkstra-optimal ρ_path (ρ_edge × ρ_len)
    entry_margin:     float,      # sim[0] - sim[1]; entry ambiguity signal
    path_agreement:   float,      # 1.0 if k candidates converge, else 0.75
    G:                nx.DiGraph,
) -> tuple[float, str]:
    """
    Combines six interpretable factors into a scalar confidence score ρ.

    ρ = ρ_sim × ρ_path × ρ_type × ρ_margin × ρ_agreement

    Each factor is independently inspectable — unlike a learned scalar,
    a low ρ can be traced to its source:
        low ρ_sim     → query far from any ontology entry point; add aliases
        low ρ_path    → traversal relied on weak/long edges; review graph
        low ρ_type    → unknown jargon entry (expected for T2/T3)
        low ρ_margin  → ambiguous entry; two concepts equally close
        low ρ_agreement → k candidates disagree on terminal leaf

    Threshold τ = CONFIDENCE_THRESH is tuned on the validation split.
    The full risk-coverage curve is reported so the choice is transparent.
    """
    expl = []

    # ── ρ_sim: embedding similarity to entry node ──────────────────────────
    rho_sim = entry_similarity
    if rho_sim >= 0.90:
        expl.append(f"strong entry match ({rho_sim:.2f})")
    elif rho_sim >= 0.70:
        expl.append(f"moderate entry match ({rho_sim:.2f})")
    else:
        expl.append(f"weak entry match ({rho_sim:.2f}) — possible wrong-node latch")

    # ── ρ_path: Dijkstra-optimal path score (ρ_edge × ρ_len) ──────────────
    # Already computed by Traverser — directly reflects best achievable path
    n_hops = len(edge_types)
    if n_hops <= 2:
        expl.append(f"short path ({n_hops} hops)")
    elif n_hops <= 4:
        expl.append(f"medium path ({n_hops} hops)")
    else:
        expl.append(f"long path ({n_hops} hops)")
    if any(et == "related_to" for et in edge_types):
        expl.append("traversed weak related_to edge")

    # ── ρ_type: alias entry vs concept entry ──────────────────────────────
    entry_type = G.nodes[path[0]].get("node_type") if path else "unknown"
    rho_type = 1.0 if entry_type == "alias" else 0.85
    if entry_type == "alias":
        expl.append("entered via known alias (precise)")
    else:
        expl.append("entered via concept node (approximate — unknown jargon)")

    # ── ρ_margin: entry ambiguity (gap between top-1 and top-2 similarity) ─
    # High margin = one clear winner; low margin = two concepts equally close
    # ρ_margin ∈ [0.5, 1.0]; saturates at margin ≥ 0.2
    rho_margin = 0.5 + min(0.5, entry_margin * 2.5)
    if entry_margin >= 0.20:
        expl.append(f"clear entry margin ({entry_margin:.2f})")
    elif entry_margin < 0.08:
        expl.append(f"ambiguous entry (margin {entry_margin:.2f}) — two concepts equally close")
    else:
        expl.append(f"moderate entry margin ({entry_margin:.2f})")

    # ── ρ_agreement: path convergence across k entry candidates ───────────
    rho_agreement = path_agreement
    if path_agreement < 1.0:
        expl.append("candidates disagree on terminal leaf — reduced confidence")
    else:
        expl.append("all candidates converge on same leaf")

    # ── Combined ρ ─────────────────────────────────────────────────────────
    rho = rho_sim * path_rho * rho_type * rho_margin * rho_agreement
    rho = min(1.0, max(0.0, rho))

    return rho, "; ".join(expl)


# ── Main Interface ─────────────────────────────────────────────────────────────

class LUMINTraversal:
    """
    Given a query term, returns a TraversalResult with PDS filters
    and a six-factor confidence signal.
    """

    def __init__(self, kg_path: str = "lumin_kg.json"):
        print(f"Loading KG from {kg_path}...")
        with open(kg_path) as f:
            kg_data = json.load(f)
        self.G         = nx.node_link_graph(kg_data)
        self.client    = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.index     = VectorIndex(self.G, self.client)
        self.traverser = Traverser(self.G)
        print(f"  {self.G.number_of_nodes()} nodes, "
              f"{self.G.number_of_edges()} edges")

    def query(self, term: str) -> TraversalResult:
        # ── Stage 1: exact alias lookup ────────────────────────────────────
        normalized = term.lower().replace(" ", "_")
        alias_id   = f"alias:{normalized}"
        if alias_id in self.G:
            candidates   = [(alias_id, 1.0)]
            entry_margin = 1.0   # exact match — no ambiguity
        else:
            # ── Stage 2: embedding-based fallback ─────────────────────────
            candidates = self.index.query(term, top_k=ENTRY_TOP_K)
            if len(candidates) >= 2:
                entry_margin = candidates[0][1] - candidates[1][1]
            else:
                entry_margin = 1.0

        # ── Dijkstra from each candidate ───────────────────────────────────
        traversal_results = []
        for entry_node, entry_sim in candidates:
            result = self.traverser.traverse(entry_node)
            if result is not None:
                path, edge_types, path_rho = result
                traversal_results.append(
                    (entry_node, entry_sim, path, edge_types, path_rho))

        if not traversal_results:
            entry_node, entry_sim = (candidates[0] if candidates
                                     else ("none", 0.0))
            return TraversalResult(
                query_term=term, entry_node=entry_node,
                entry_node_type="unknown", entry_similarity=entry_sim,
                entry_margin=entry_margin, path=[], edge_types=[],
                path_rho=0.0, path_agreement=1.0, schema_leaves=[],
                filters=[], confidence=0.0, uncertain=True,
                explanation="No schema leaf reachable within depth limit",
            )

        # ── Path agreement across candidates ───────────────────────────────
        terminal_leaves  = [r[2][-1] for r in traversal_results]
        unique_terminals = set(terminal_leaves)
        path_agreement   = 1.0 if len(unique_terminals) == 1 else 0.75

        # ── Score each candidate and pick the best ─────────────────────────
        best_result = None
        for entry_node, entry_sim, path, edge_types, path_rho in traversal_results:
            filters = extract_filters(self.G, path, edge_types)
            confidence, explanation = score_confidence(
                entry_similarity=entry_sim,
                path=path,
                edge_types=edge_types,
                path_rho=path_rho,
                entry_margin=entry_margin,
                path_agreement=path_agreement,
                G=self.G,
            )
            result = TraversalResult(
                query_term       = term,
                entry_node       = entry_node,
                entry_node_type  = self.G.nodes[entry_node].get(
                                       "node_type", "unknown"),
                entry_similarity = entry_sim,
                entry_margin     = entry_margin,
                path             = path,
                edge_types       = edge_types,
                path_rho         = path_rho,
                path_agreement   = path_agreement,
                schema_leaves    = [self.G.nodes[path[-1]]],
                filters          = filters,
                confidence       = confidence,
                uncertain        = confidence < CONFIDENCE_THRESH,
                explanation      = explanation,
            )
            if best_result is None or confidence > best_result.confidence:
                best_result = result

        return best_result


# ── Demo ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    lumint = LUMINTraversal("lumin_kg.json")

    test_queries = [
        "southern summer",                            # T1 exact alias
        "HiRISE",                                     # T1 exact alias
        "north pole",                                 # T1 exact alias
        "southern hemisphere warm season",            # T2
        "subsurface ice layering radar",              # T2
        "aphelion season near perihelion transition", # T3-in
        "NPLD stratigraphic profiles from orbit",     # T3-in
        "tau surge during global encirclement",       # T3-in
        "cryosphere thickness estimates",             # T3-out (not in alias list)
        "coffee shop near JPL",                       # OOD
    ]

    print("\n" + "=" * 72)
    print("LUMIN KG Traversal — Demo (Dijkstra + 6-factor confidence)")
    print("=" * 72)

    for q in test_queries:
        print(f"\n{'─' * 72}")
        print(lumint.query(q).pretty())