"""
Consumer 2 — Dataset mining for testing the KG.

Runs the full anchor -> harvest -> extract pipeline across (some or all)
concepts, and writes two files:

  1. candidate_aliases.csv — new jargon found in real literature, for a human
     to triage before it's ever added to the graph as a real alias. Columns:
     concept, phrase, explicit, source_title, source_url, justification

  2. mined_dev_rows.csv — T3-tier query rows in the SAME schema as the
     hand-written tutorial dataset (id, tier, query_text, target_concept_or_alias,
     notes), so this batch merges directly into the same eval dev set A and B
     are hand-writing. query_text here is the actual sentence the phrase was
     found in (lightly cleaned) — a real, naturally-occurring test case,
     not an invented one.

Usage:
    python mine_dataset.py lumin_kg_summary.txt
    python mine_dataset.py lumin_kg_summary.txt --concept "Global Dust Storm"
    python mine_dataset.py lumin_kg_summary.txt --papers-per-query 5
"""

import argparse
import csv
import re

from anchors import parse_summary, DEFAULT_SUMMARY_PATH
from ads_harvest import harvest_anchor
from llm_extract import extract_candidates


def clean_sentence(excerpt: str, phrase: str, max_len: int = 240) -> str:
    """Pull just the sentence containing `phrase` out of a longer excerpt."""
    sentences = re.split(r"(?<=[.!?])\s+", excerpt)
    for s in sentences:
        if phrase.lower() in s.lower():
            return s.strip()[:max_len]
    return excerpt.strip()[:max_len]  # fallback: truncated excerpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summary_path", nargs="?", default=DEFAULT_SUMMARY_PATH)
    ap.add_argument("--concept", help="only mine this concept (substring match)")
    ap.add_argument("--papers-per-query", type=int, default=5)
    args = ap.parse_args()

    anchors = parse_summary(args.summary_path)
    if args.concept:
        anchors = [a for a in anchors if args.concept.lower() in a.concept.lower()]
    if not anchors:
        print("No matching concepts.")
        return

    alias_rows, dev_rows = [], []
    next_id = 1

    for anchor in anchors:
        print(f"── {anchor.concept} ──")
        try:
            hits = harvest_anchor(anchor, rows_per_query=args.papers_per_query)
        except RuntimeError as e:
            print(f"   skipped: {e}")
            continue

        for hit in hits:
            excerpt = hit.get("abstract") or ""
            if not excerpt:
                continue
            candidates = extract_candidates(anchor, excerpt)
            title = hit.get("title", ["?"])[0]
            url = f"https://ui.adsabs.harvard.edu/abs/{hit.get('bibcode')}/abstract"

            for c in candidates:
                print(f"   found: '{c['phrase']}'  (explicit={c['explicit']})")
                alias_rows.append({
                    "concept": anchor.concept,
                    "phrase": c["phrase"],
                    "explicit": c["explicit"],
                    "source_title": title,
                    "source_url": url,
                    "justification": c["justification"],
                })
                sentence = clean_sentence(excerpt, c["phrase"])
                dev_rows.append({
                    "id": f"MINED-{next_id:03d}",
                    "tier": "T3",
                    "query_text": sentence,
                    "target_concept_or_alias": anchor.concept,
                    "notes": f"mined from {hit.get('bibcode')}; candidate phrase: '{c['phrase']}'"
                             + ("" if c["explicit"] else " (implicit use — verify before trusting)"),
                })
                next_id += 1
        print()

    with open("candidate_aliases.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["concept", "phrase", "explicit", "source_title", "source_url", "justification"])
        w.writeheader()
        w.writerows(alias_rows)

    with open("mined_dev_rows.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "tier", "query_text", "target_concept_or_alias", "notes"])
        w.writeheader()
        w.writerows(dev_rows)

    print(f"Wrote {len(alias_rows)} candidate aliases -> candidate_aliases.csv")
    print(f"Wrote {len(dev_rows)} dev-set rows -> mined_dev_rows.csv")
    print("\nBoth need a human pass before trusting them:")
    print("  - candidate_aliases.csv: review before adding anything to the graph")
    print("  - mined_dev_rows.csv: spot-check a sample before merging into the real eval set")


if __name__ == "__main__":
    main()
