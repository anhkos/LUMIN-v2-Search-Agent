"""
traversal.py  (v2.1 — slot-aware termination + calibration fixes)
-----------------------------------------------------------------
LUMIN Knowledge Graph traversal module.

Three components:
    1. VectorIndex    — embeds alias/concept nodes; finds nearest entry point
                        for any query term (known or unknown jargon)
    2. Traverser      — best-first search (Dijkstra) from entry point to a
                        *complete grounding*: a class node whose filled slots
                        fan out to schema leaves
    3. TraversalResult — structured output with composed filters + confidence

Changes in v2.1 (fixes issues surfaced by the first v2 demo run):

  [T1] TERMINAL CONDITION: the goal state is now the CLASS node, not the
       first schema leaf popped. On reaching a class, the traverser computes
       the slots the concept (or alias override) actually fills, and fans out
       across ALL matching grounded_in edges in parallel. Consequences:
         * multi-slot concepts emit complete filter sets (a bounding box is
           four sides, not whichever side Dijkstra popped first)
         * slots the concept never filled are unreachable by construction —
           a filter with a field but no value can no longer be emitted
       If a class has no filled slots for this concept, it is NOT a terminal;
       search continues (e.g. through occurs_during to another concept).

  [T2] ENTRY MARGIN over distinct CONCEPTS: top-k raw embedding hits are
       deduplicated by parent concept before computing the top-1/top-2 gap.
       Two aliases of the same concept are redundancy, not ambiguity.

  [T3] ρ_type keys on the entry MECHANISM (exact lookup vs embedding
       fallback), not the node type the embedding happened to land on.

  [T4] ρ_path is NORMALIZED by the best achievable score for the entry
       mechanism, so a minimal clean path scores 1.0 rather than a constant
       0.85^(hops-1) ceiling that silently compresses every ρ below τ.

  [T5] ρ_agreement is class-aware: candidates grounding in DIFFERENT classes
       (e.g. a region box + an instrument name) are flagged as a possible
       compound/conjunctive query (mild 0.9), distinct from candidates
       disagreeing within the same class (genuine ambiguity, 0.75).

Confidence signal ρ = ρ_sim × ρ_path × ρ_type × ρ_margin × ρ_agreement
    ρ_sim       cosine similarity to entry node (embedding-based)
    ρ_path      Dijkstra-optimal path score, normalized per [T4]
    ρ_type      1.0 exact alias lookup; 0.85 embedding fallback  [T3]
    ρ_margin    entry ambiguity across distinct concepts         [T2]
    ρ_agreement cross-candidate consistency, class-aware         [T5]

Note on supervision: ρ requires no learned confidence model and no reflection-
token training data. A single scalar threshold τ is tuned on the validation
split; the full risk-coverage curve is reported so the choice of τ is
transparent. ρ_sim depends on the embedding model (not pure graph topology);
all other factors are derived from graph structure alone.

!! τ (CONFIDENCE_THRESH) was tuned against the pre-normalization ρ scale.
   After [T4] the scale shifts upward; re-run the risk–coverage sweep on the
   validation split before quoting abstention numbers.

Usage:
    python traversal.py

Requires:
    pip install networkx numpy openai
    export OPENAI_API_KEY=your_key
    (without a key, a deterministic hash-based embedder is used — TESTING
     ONLY: exact-alias behavior is identical, fuzzy-match similarities are
     NOT comparable to the real embedding model)
"""

import json
import os
import math
import heapq
import hashlib
import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from itertools import count
from typing import Optional

# ── Configuration ──────────────────────────────────────────────────────────────

EMBEDDING_MODEL   = "text-embedding-3-small"
ENTRY_TOP_K       = 3       # distinct-concept entry candidates   [T2]
RAW_TOP_K         = 10      # raw embedding hits before concept dedup
MAX_DEPTH         = 5       # max hops in Dijkstra before giving up
CONFIDENCE_THRESH = 0.55    # below this → flag uncertain
                            # !! re-tune on val split after [T4] rescaling

# Fallback edge priorities — the v2 KG exports its own in
# G.graph["edge_priorities"]; these are used only for graphs that don't.
EDGE_PRIORITY = {
    "is_alias_of":    1.00,
    "instance_of":    1.00,
    "grounded_in":    1.00,
    "occurs_during":  0.85,
    "located_in":     0.85,
    "observed_by":    0.80,
    "related_to":     0.50,
}

LENGTH_DECAY = 0.85   # per hop after the first — tunable, ablated in paper

# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class TraversalResult:
    query_term:       str
    entry_node:       str
    entry_node_type:  str        # "alias" | "concept"
    entry_mode:       str        # "exact" | "embedding"           [T3]
    entry_similarity: float
    entry_margin:     float      # top-1/top-2 gap over distinct concepts [T2]
    path:             list       # node-id path from entry to CLASS node
    edge_types:       list       # edge type labels along that path
    path_rho:         float      # normalized ρ_path               [T4]
    rho_ambiguity:    float      # [T6] consequence-aware ambiguity factor
    schema_leaves:    list       # ALL leaf node dicts reached via filled slots
    filters:          list       # composed PDS filter dicts (one per leaf)
    confidence:       float
    uncertain:        bool
    explanation:      str

    def pretty(self):
        leaf_names = ", ".join(
            f"{d.get('class')}.{d.get('name')}" for d in self.schema_leaves)
        lines = [
            f"Query:        {self.query_term}",
            f"Entry node:   {self.entry_node}  "
            f"({self.entry_node_type}, {self.entry_mode})",
            f"Similarity:   {self.entry_similarity:.3f}  "
            f"(margin: {self.entry_margin:.3f})",
            f"Path:         {' → '.join(self.path)}",
            f"Edge types:   {' → '.join(self.edge_types)}",
            f"Leaves:       {leaf_names or '(none)'}",
            f"ρ_path:       {self.path_rho:.3f}  "
            f"ambiguity: {self.rho_ambiguity:.2f}",
            f"Confidence:   {self.confidence:.3f}  "
            f"{'⚠ UNCERTAIN' if self.uncertain else '✓ OK'}",
            f"Filters:      {self.filters}",
            f"Explanation:  {self.explanation}",
        ]
        return "\n".join(lines)


# ── Embedding backends ─────────────────────────────────────────────────────────

class OpenAIEmbedder:
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def embed(self, texts: list[str]) -> np.ndarray:
        resp = self.client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        return np.array([e.embedding for e in resp.data], dtype=np.float32)


class HashEmbedder:
    """
    Deterministic character-trigram hashing embedder. TESTING ONLY — lets the
    pipeline run without network access. Exact-alias lookups (Stage 1) never
    touch embeddings, so those paths are faithful; fuzzy-match similarities
    are NOT comparable to the real embedding model.
    """
    DIM = 512

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            t = f"  {text.lower()}  "
            for i in range(len(t) - 2):
                h = int(hashlib.md5(t[i:i + 3].encode()).hexdigest()[:8], 16)
                out[row, h % self.DIM] += 1.0
        return out


def make_embedder():
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAIEmbedder()
    print("⚠  OPENAI_API_KEY not set — using hash embedder (TESTING ONLY; "
          "fuzzy similarities not comparable to real embeddings)")
    return HashEmbedder()


# ── Vector Index ───────────────────────────────────────────────────────────────

class VectorIndex:
    """
    Embeds all alias and concept nodes. Search space is restricted to the
    alias + concept layers — never schema leaves — to preserve the traversal
    path. Returns RAW hits; concept-level dedup happens in LUMINTraversal [T2].
    """

    def __init__(self, G: nx.DiGraph, embedder):
        self.G = G
        self.embedder = embedder
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
        vecs  = self.embedder.embed(self.texts)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        self.embeddings = vecs / np.maximum(norms, 1e-9)
        print(f"  Index ready: {self.embeddings.shape}")

    def query(self, term: str, top_k: int = RAW_TOP_K
              ) -> list[tuple[str, float]]:
        """Return top-k raw (node_id, cosine_similarity) pairs."""
        q = self.embedder.embed([term])[0]
        q = q / max(np.linalg.norm(q), 1e-9)
        sims = self.embeddings @ q
        idx  = np.argsort(sims)[::-1][:top_k]
        return [(self.node_ids[i], float(sims[i])) for i in idx]


# ── Slot-value resolution (shared by Traverser + extractor) ────────────────────

def _iter_slot_dicts(slot_values: dict):
    """Yield flat {slot: value} dicts, expanding a variants wrapper."""
    if "variants" in slot_values:
        for v in slot_values["variants"]:
            yield v
    else:
        yield slot_values


def effective_slot_values(G: nx.DiGraph, path: list[str],
                          edge_types: list[str]) -> dict:
    """
    The slot values in effect along a path: the concept's instance_of values,
    overridden wholesale by an alias-level value_hint if the entry alias
    carried one. Alias override takes precedence over (often more generic)
    concept-level values — never the reverse.
    """
    concept_sv: dict = {}
    alias_sv: Optional[dict] = None
    for i, et in enumerate(edge_types):
        edge_d = G.edges[path[i], path[i + 1]]
        if et == "instance_of":
            try:
                concept_sv = json.loads(edge_d.get("slot_values") or "{}")
            except (json.JSONDecodeError, TypeError):
                concept_sv = {}
        elif et == "is_alias_of" and edge_d.get("value_hint"):
            try:
                alias_sv = json.loads(edge_d["value_hint"])
            except (json.JSONDecodeError, TypeError):
                pass
    return alias_sv if alias_sv is not None else concept_sv


def filled_slots(slot_values: dict) -> set:
    """Slot names carrying a value in any variant."""
    slots = set()
    for sv in _iter_slot_dicts(slot_values):
        slots.update(sv.keys())
    return slots


# ── Dijkstra Traverser ─────────────────────────────────────────────────────────

class Traverser:
    """
    Best-first search maximising ρ_path, with a slot-aware goal state [T1]:

    A CLASS node is terminal iff the slot values in effect along the path fill
    at least one of the class's grounded slots. On termination the traverser
    fans out across ALL grounded_in edges whose slot is filled — those leaves
    are reached in parallel and share one grounded_in hop in the path score.
    A class with no filled slots is a through-node, not a terminal (search may
    continue via occurs_during etc. to another concept's grounding).

    Legacy v1 graphs (concept -> leaf edges, no class layer) still terminate
    on the first schema leaf popped.

    ρ_path = product(edge_priorities) × LENGTH_DECAY^(max(0, hops-1)),
    computed via -log costs so Dijkstra directly maximises it. The first
    terminal popped is ρ_path-optimal: search and calibration share one
    objective, so the reported ρ_path is the best achievable from the entry.
    """

    LOG_DECAY = -math.log(LENGTH_DECAY)

    def __init__(self, G: nx.DiGraph):
        self.G = G
        edge_priority = G.graph.get("edge_priorities", EDGE_PRIORITY)
        self.EDGE_COST = {et: -math.log(max(p, 1e-9))
                          for et, p in edge_priority.items()}

    def _grounded_edges(self, class_node: str) -> dict:
        """{slot: leaf_node_id} for a class node."""
        out = {}
        for succ in self.G.successors(class_node):
            ed = self.G.edges[class_node, succ]
            if ed.get("edge_type") == "grounded_in":
                out[ed.get("slot")] = succ
        return out

    def traverse(self, start_node: str) -> Optional[dict]:
        """
        Dijkstra from start_node to the ρ_path-optimal complete grounding.
        Returns dict(path, edge_types, rho_path_raw, leaves={slot: leaf_id},
        slot_values) or None if no grounding is reachable.
        """
        _cnt = count()
        pq = [(0.0, next(_cnt), start_node, [start_node], [])]
        best_cost: dict[str, float] = {start_node: 0.0}

        while pq:
            cost, _, current, path, edges = heapq.heappop(pq)
            if cost > best_cost.get(current, math.inf) + 1e-9:
                continue

            nt = self.G.nodes[current].get("node_type")

            # [T1] Class node: terminal iff the concept fills >=1 grounded slot
            if nt == "class":
                sv        = effective_slot_values(self.G, path, edges)
                grounded  = self._grounded_edges(current)
                usable    = filled_slots(sv) & set(grounded)
                if usable:
                    hop_cost = (self.EDGE_COST.get("grounded_in", 0.0)
                                + (self.LOG_DECAY if len(edges) >= 1 else 0.0))
                    return {
                        "path":         path,
                        "edge_types":   edges + ["grounded_in"],
                        "rho_path_raw": math.exp(-(cost + hop_cost)),
                        "leaves":       {s: grounded[s] for s in sorted(usable)},
                        "slot_values":  sv,
                    }
                # else: fall through — expand successors (not grounded_in,
                # which only lead to leaves we have no values for)

            # Legacy v1 graphs: leaf terminal
            if nt == "schema_leaf" and len(path) > 1:
                return {
                    "path":         path,
                    "edge_types":   edges,
                    "rho_path_raw": math.exp(-cost),
                    "leaves":       {edges[-1]: current} if edges else {},
                    "slot_values":  effective_slot_values(self.G, path, edges),
                }

            if len(path) > MAX_DEPTH + 1:
                continue

            for neighbor in self.G.successors(current):
                et = self.G.edges[current, neighbor].get("edge_type", "unknown")
                if nt == "class" and et == "grounded_in":
                    continue   # [T1] unfilled slots are unreachable by construction
                edge_cost   = self.EDGE_COST.get(et, -math.log(0.5))
                length_cost = self.LOG_DECAY if len(edges) >= 1 else 0.0
                new_cost    = cost + edge_cost + length_cost

                if new_cost < best_cost.get(neighbor, math.inf):
                    best_cost[neighbor] = new_cost
                    heapq.heappush(
                        pq, (new_cost, next(_cnt), neighbor,
                             path + [neighbor], edges + [et]))
        return None

    def max_achievable_rho(self, entry_node: str) -> float:
        """
        [T4] Best-case ρ_path for this entry's minimal complete grounding:
        alias entry = 3 hops (alias→concept→class→leaves), concept entry = 2.
        All-priority-1.0 edges assumed. Used to normalize ρ_path so a minimal
        clean path scores 1.0 instead of a constant sub-τ ceiling.
        """
        nt = self.G.nodes[entry_node].get("node_type")
        min_hops = 3 if nt == "alias" else 2
        return math.exp(-(min_hops - 1) * self.LOG_DECAY)


# ── Filter Extractor ───────────────────────────────────────────────────────────

def extract_filters(G: nx.DiGraph, leaves: dict, slot_values: dict
                    ) -> list[dict]:
    """
    Compose one filter per grounded leaf: field metadata from the leaf node,
    value from the slot values in effect. Disjoint variants that disagree on
    a slot (e.g. north/south polar caps) surface as a list of alternatives —
    downstream query construction must OR them.
    """
    filters = []
    for slot, leaf_id in leaves.items():
        vd = G.nodes[leaf_id]

        value_hint = None
        for sv in _iter_slot_dicts(slot_values):
            if slot not in sv:
                continue
            if value_hint is None:
                value_hint = sv[slot]
            elif value_hint != sv[slot]:
                if not isinstance(value_hint, list):
                    value_hint = [value_hint]
                if sv[slot] not in value_hint:
                    value_hint.append(sv[slot])

        f = {
            "field":      vd.get("name"),
            "class":      vd.get("class"),
            "ldd":        vd.get("ldd"),
            "slot":       slot,
            "data_type":  vd.get("data_type"),
            "unit":       vd.get("unit"),
            "value_hint": value_hint,
        }
        if vd.get("enumerated") and vd.get("permissible_values"):
            f["permissible_values"] = [
                pv["value"] for pv in vd["permissible_values"]]
        filters.append(f)
    return filters


# ── Confidence Scorer ──────────────────────────────────────────────────────────

def score_confidence(
    entry_similarity: float,
    entry_mode:       str,        # "exact" | "embedding"          [T3]
    edge_types:       list[str],
    path_rho_norm:    float,      # normalized ρ_path              [T4]
    ambiguity:        tuple,      # (rho_amb, note) from _ambiguity [T6]
) -> tuple[float, str]:
    """
    ρ = ρ_sim × ρ_path × ρ_type × ρ_ambiguity

    Each factor is independently inspectable — a low ρ traces to its source:
        low ρ_sim       → query far from any entry point; add aliases
        low ρ_path      → traversal relied on weak/long edges; review graph
        low ρ_type      → embedding-fallback entry (expected for T2/T3)
        low ρ_ambiguity → a close competing concept grounds DIFFERENTLY [T6]

    [T6] ρ_ambiguity merges the former ρ_margin and ρ_agreement, which after
    concept-dedup measured overlapping quantities and double-penalized:
    closeness in embedding space only matters when the close candidates lead
    to different filters. Computed in LUMINTraversal._ambiguity by comparing
    filter SIGNATURES (field + value), not leaf ids — two instruments filling
    identical slots with different values are genuine ambiguity even though
    their leaf sets are equal.

    Threshold τ = CONFIDENCE_THRESH; report the full risk–coverage curve.
    """
    expl = []

    rho_sim = entry_similarity
    if rho_sim >= 0.90:
        expl.append(f"strong entry match ({rho_sim:.2f})")
    elif rho_sim >= 0.70:
        expl.append(f"moderate entry match ({rho_sim:.2f})")
    else:
        expl.append(f"weak entry match ({rho_sim:.2f}) — possible wrong-node latch")

    n_hops = len(edge_types)
    if path_rho_norm >= 0.999:
        expl.append(f"minimal clean path ({n_hops} hops)")
    else:
        expl.append(f"non-minimal path ({n_hops} hops, ρ_path {path_rho_norm:.2f})")
    if any(et == "related_to" for et in edge_types):
        expl.append("traversed weak related_to edge")

    # [T3] mechanism, not landed-node type
    rho_type = 1.0 if entry_mode == "exact" else 0.85
    if entry_mode == "exact":
        expl.append("exact alias lookup (precise)")
    else:
        expl.append("embedding-fallback entry (unknown surface form)")

    # [T6] consequence-aware ambiguity: closeness only costs confidence when
    # the close candidates lead to DIFFERENT groundings (see _ambiguity).
    rho_amb, amb_note = ambiguity
    expl.append(amb_note)

    rho = rho_sim * path_rho_norm * rho_type * rho_amb
    return min(1.0, max(0.0, rho)), "; ".join(expl)


# ── Main Interface ─────────────────────────────────────────────────────────────

class LUMINTraversal:
    """
    Given a query term, returns a TraversalResult with composed PDS filters
    and a decomposable confidence signal.
    """

    def __init__(self, kg_path: str = "output/lumin_kg.json"):
        print(f"Loading KG from {kg_path}...")
        with open(kg_path) as f:
            kg_data = json.load(f)
        edges_kw  = "edges" if "edges" in kg_data else "links"
        self.G    = nx.node_link_graph(kg_data, edges=edges_kw)
        self.index     = VectorIndex(self.G, make_embedder())
        self.traverser = Traverser(self.G)
        print(f"  {self.G.number_of_nodes()} nodes, "
              f"{self.G.number_of_edges()} edges")

    def _parent_concept(self, node_id: str) -> str:
        d = self.G.nodes[node_id]
        if d.get("node_type") == "alias":
            return d.get("concept", node_id)
        return node_id

    def _entry_candidates(self, term: str
                          ) -> tuple[list[tuple[str, float]], float, str]:
        """
        Stage 1: exact alias lookup. Stage 2: embedding fallback, deduped to
        the best-scoring node per DISTINCT concept [T2]. Margin is the
        top-1/top-2 similarity gap across distinct concepts.
        """
        normalized = term.lower().replace(" ", "_")
        alias_id   = f"alias:{normalized}"
        if alias_id in self.G:
            return [(alias_id, 1.0)], 1.0, "exact"

        raw = self.index.query(term, top_k=RAW_TOP_K)
        seen, deduped = set(), []
        for node_id, sim in raw:                     # raw is sim-descending
            concept = self._parent_concept(node_id)
            if concept in seen:
                continue
            seen.add(concept)
            deduped.append((node_id, sim))
            if len(deduped) == ENTRY_TOP_K:
                break
        margin = (deduped[0][1] - deduped[1][1]) if len(deduped) >= 2 else 1.0
        return deduped, margin, "embedding"

    LIVE_BAND    = 0.15   # runner-up counts as "live" if within this sim gap
    COMPOUND_MIN = 0.60   # min top similarity to label disjoint hits compound

    def _ambiguity(self, candidates, filters_per_candidate, entry_mode
                   ) -> tuple[float, str]:
        """
        [T6] ρ_ambiguity: penalize entry closeness only when close candidates
        lead to DIFFERENT groundings. Candidates are compared by filter
        SIGNATURE — frozenset of (field, value) — not leaf ids, so two
        instruments filling identical slots with different values register
        as conflicting (leaf sets alone cannot see this).

        Cases (over candidates within LIVE_BAND of the top similarity):
          no live runner-up            → 1.0   closeness is moot
          identical signatures         → 1.0   near-neighbors agree
          disjoint fields              → 0.95  possible compound query
          shared field, differing value → 0.5 + 0.5·min(1, margin/0.2)
                                          (margin-scaled: the closer the
                                           conflicting alternative, the
                                           larger the penalty)
        """
        if entry_mode == "exact" or len(candidates) < 2:
            return 1.0, "no competing entry (exact or single candidate)"

        top_sim = candidates[0][1]
        margin  = top_sim - candidates[1][1]

        def signature(filters):
            return frozenset(
                (f["field"], json.dumps(f["value_hint"], sort_keys=True))
                for f in filters)

        def fields(filters):
            return {f["field"] for f in filters}

        top_sig, top_fields = (signature(filters_per_candidate[0]),
                               fields(filters_per_candidate[0]))
        live = [(sim, signature(fl), fields(fl))
                for (_, sim), fl in zip(candidates[1:],
                                        filters_per_candidate[1:])
                if top_sim - sim <= self.LIVE_BAND]

        if not live:
            return 1.0, (f"no competing concept within similarity band "
                         f"(margin {margin:.2f})")
        if all(sig == top_sig for _, sig, _ in live):
            return 1.0, "near candidates converge on identical grounding"
        if any(fld & top_fields for _, sig, fld in live if sig != top_sig):
            rho = 0.5 + 0.5 * min(1.0, margin / 0.2)
            return rho, (f"close competing concept grounds the same field(s) "
                         f"differently (margin {margin:.2f}) — ambiguity "
                         f"penalty scaled by margin")
        if top_sim >= self.COMPOUND_MIN:
            return 0.95, ("close candidate grounds disjoint fields — "
                          "possible compound query (composable filters)")
        return 0.95, "weak scattered candidates ground disjoint fields"

    def query(self, term: str) -> TraversalResult:
        candidates, entry_margin, entry_mode = self._entry_candidates(term)

        traversal_results = []
        for entry_node, entry_sim in candidates:
            t = self.traverser.traverse(entry_node)
            if t is not None:
                traversal_results.append((entry_node, entry_sim, t))

        if not traversal_results:
            entry_node, entry_sim = (candidates[0] if candidates
                                     else ("none", 0.0))
            return TraversalResult(
                query_term=term, entry_node=entry_node,
                entry_node_type="unknown", entry_mode=entry_mode,
                entry_similarity=entry_sim, entry_margin=entry_margin,
                path=[], edge_types=[], path_rho=0.0, rho_ambiguity=1.0,
                schema_leaves=[], filters=[], confidence=0.0, uncertain=True,
                explanation="No complete grounding reachable within depth limit",
            )

        # ── [T6] Consequence-aware ambiguity across candidates ─────────────
        per_cand_filters = [
            extract_filters(self.G, t["leaves"], t["slot_values"])
            for _, _, t in traversal_results]
        ambiguity = self._ambiguity(
            candidates=[(n, s) for n, s, _ in traversal_results],
            filters_per_candidate=per_cand_filters,
            entry_mode=entry_mode)

        # ── Score each candidate; keep the best ────────────────────────────
        best_result = None
        for (entry_node, entry_sim, t), filters in zip(
                traversal_results, per_cand_filters):
            rho_norm = min(1.0, t["rho_path_raw"]
                           / self.traverser.max_achievable_rho(entry_node))
            confidence, explanation = score_confidence(
                entry_similarity=entry_sim,
                entry_mode=entry_mode,
                edge_types=t["edge_types"],
                path_rho_norm=rho_norm,
                ambiguity=ambiguity,
            )
            result = TraversalResult(
                query_term       = term,
                entry_node       = entry_node,
                entry_node_type  = self.G.nodes[entry_node].get(
                                       "node_type", "unknown"),
                entry_mode       = entry_mode,
                entry_similarity = entry_sim,
                entry_margin     = entry_margin,
                path             = t["path"],
                edge_types       = t["edge_types"],
                path_rho         = rho_norm,
                rho_ambiguity    = ambiguity[0],
                schema_leaves    = [self.G.nodes[l]
                                    for l in t["leaves"].values()],
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
    lumint = LUMINTraversal("output/lumin_kg.json")

    test_queries = [
        "southern summer",                            # T1 exact alias
        "HiRISE",                                     # T1 exact alias
        "north pole",                                 # T1 exact alias
        "MY34",                                       # T1 exact, alias override
        "southern hemisphere warm season",            # T2
        "subsurface ice layering radar",              # T2
        "aphelion season near perihelion transition", # T3-in
        "NPLD stratigraphic profiles from orbit",     # T3-in (compound-ish)
        "tau surge during global encirclement",       # T3-in
        "cryosphere thickness estimates",             # T3-out (not in alias list)
        "coffee shop near JPL",                       # OOD
    ]

    print("\n" + "=" * 72)
    print("LUMIN KG Traversal — Demo (slot-aware Dijkstra + 5-factor ρ)")
    print("=" * 72)

    for q in test_queries:
        print(f"\n{'─' * 72}")
        print(lumint.query(q).pretty())