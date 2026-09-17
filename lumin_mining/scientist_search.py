"""
Consumer 1 — Paper search for interested scientists.

Given a term a scientist actually types (which might be an exact alias, a
rough paraphrase, or something not in the graph at all), find the graph
concept it most likely refers to, then surface real papers about it —
ranked, with links, not just a raw ADS dump.

This deliberately reuses the SAME anchors + harvest modules as the dataset
miner (consumer 2). They are two views on one pipeline, not two pipelines.

Usage:
    python scientist_search.py "aphelion cloud belt" lumin_kg_summary.txt
    python scientist_search.py "MY34" lumin_kg_summary.txt --papers 10
"""

import argparse
import difflib

from anchors import parse_summary, DEFAULT_SUMMARY_PATH
from ads_harvest import harvest_anchor, search


def find_concept(query: str, anchors, cutoff: float = 0.6):
    """
    Loose match against every known alias. Returns the best-matching Anchor,
    or None if nothing clears the similarity cutoff (caller should fall back
    to a raw, ungrounded ADS search in that case).
    """
    query_lower = query.lower().strip()
    best_anchor, best_score = None, 0.0
    for a in anchors:
        for alias in a.aliases:
            score = difflib.SequenceMatcher(None, query_lower, alias).ratio()
            if score > best_score:
                best_anchor, best_score = a, score
    return (best_anchor, best_score) if best_score >= cutoff else (None, best_score)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", help="what the scientist typed")
    ap.add_argument("summary_path", nargs="?", default=DEFAULT_SUMMARY_PATH)
    ap.add_argument("--papers", type=int, default=8)
    args = ap.parse_args()

    anchors = parse_summary(args.summary_path)
    anchor, score = find_concept(args.query, anchors)

    if anchor:
        print(f"Matched to graph concept: {anchor.concept}  (similarity {score:.2f})")
        print(f"  {anchor.description}\n")
        hits = harvest_anchor(anchor, rows_per_query=args.papers)
    else:
        print(f"No confident graph match (best similarity {score:.2f}) — "
              f"searching literature directly on your query text instead.\n")
        hits = [{**doc, "matched_query": args.query}
                for doc in search(f'"{args.query}"', rows=args.papers)]

    if not hits:
        print("No papers found.")
        return

    hits.sort(key=lambda h: h.get("citation_count") or 0, reverse=True)
    print(f"Found {len(hits)} papers:\n")
    for h in hits[:args.papers]:
        title = h.get("title", ["(no title)"])[0]
        print(f"  {title}")
        print(f"    {h.get('year', '?')}  ·  cited {h.get('citation_count', 0)}  ·  "
              f"https://ui.adsabs.harvard.edu/abs/{h.get('bibcode')}/abstract")
        if h.get("abstract"):
            snippet = h["abstract"][:180].rsplit(" ", 1)[0] + "…"
            print(f"    {snippet}")
        print()


if __name__ == "__main__":
    main()
