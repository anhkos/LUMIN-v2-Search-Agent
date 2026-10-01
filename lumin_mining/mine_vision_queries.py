"""
Consumer 3 — Vision query mining for mars-clip-bench.

Same anchor -> harvest -> extract orchestration as mine_dataset.py, pointed at MSL/HiRISE
visual classes (vision_anchors.py) instead of PDS4 schema concepts, and writing output in the
vision project's own schema rather than the KG project's dev-set schema.

Writes two files:

  1. vision_candidate_terms.csv — every candidate phrase the extractor found, explicit and
     implicit alike, with its source paper — a human-reviewable trail of what was found and
     rejected, not just a black box that spat out a CSV. Columns: concept, mission, class_label,
     phrase, explicit, source_title, source_url, justification.

  2. vision_mined_queries.csv — only the explicit candidates, in queries.csv's own schema
     (query_text, class_label, mission, query_type), query_type always "authentic". Implicit
     candidates are NOT included here — see llm_extract_vision.py's docstring on why an implicit
     mention would be a circular, meaningless "visual description." Spot-check this file before
     merging any of it into eval/queries/msl_queries.csv or hirise_queries.csv.

Usage:
    python mine_vision_queries.py --mission HiRISE
    python mine_vision_queries.py --mission MSL
    python mine_vision_queries.py --mission all --papers-per-query 5
"""

import argparse
import csv
import re
import sys

from vision_anchors import ALL_ANCHORS, HIRISE_ANCHORS, MSL_ANCHORS, class_label_for
from ads_harvest import harvest_anchor
from llm_extract_vision import extract_candidates

# LLM-extracted text can contain arbitrary Unicode (≈, em dashes, degree signs, ...) that
# Windows' default console codepage (cp1252) can't encode — without this, a single such
# character crashes the run mid-way through, after real (paid) API calls have already been
# spent on the anchors processed so far. Force UTF-8 stdout so that can't happen.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CANDIDATE_TERMS_PATH = "vision_candidate_terms.csv"
MINED_QUERIES_PATH = "vision_mined_queries.csv"
CANDIDATE_FIELDS = ["concept", "mission", "class_label", "phrase", "explicit",
                     "source_title", "source_url", "justification"]
QUERY_FIELDS = ["query_text", "class_label", "mission", "query_type"]


def write_outputs(candidate_rows, query_rows):
    """Rewritten after every anchor so a mid-run crash only loses the in-flight anchor's
    work, not the whole run's worth of already-spent API calls."""
    with open(CANDIDATE_TERMS_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CANDIDATE_FIELDS)
        w.writeheader()
        w.writerows(candidate_rows)

    with open(MINED_QUERIES_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=QUERY_FIELDS)
        w.writeheader()
        w.writerows(query_rows)


def clean_sentence(excerpt: str, phrase: str, max_len: int = 240) -> str:
    """Pull just the sentence containing `phrase` out of a longer excerpt."""
    sentences = re.split(r"(?<=[.!?])\s+", excerpt)
    for s in sentences:
        if phrase.lower() in s.lower():
            return s.strip()[:max_len]
    return excerpt.strip()[:max_len]  # fallback: truncated excerpt


def anchors_for_mission(mission: str) -> list:
    if mission == "HiRISE":
        return HIRISE_ANCHORS
    if mission == "MSL":
        return MSL_ANCHORS
    return ALL_ANCHORS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", choices=["HiRISE", "MSL", "all"], default="all")
    ap.add_argument("--papers-per-query", type=int, default=5)
    args = ap.parse_args()

    anchors = anchors_for_mission(args.mission)

    candidate_rows, query_rows = [], []
    empty_classes = []

    for anchor in anchors:
        print(f"-- {anchor.concept} [{anchor.mission}] --")
        class_label = class_label_for(anchor)

        try:
            hits = harvest_anchor(anchor, rows_per_query=args.papers_per_query)
        except RuntimeError as e:
            print(f"   skipped: {e}")
            continue

        found_any = False
        for hit in hits:
            excerpt = hit.get("abstract") or ""
            if not excerpt:
                continue
            candidates = extract_candidates(anchor, excerpt)
            title = hit.get("title", ["?"])[0]
            url = f"https://ui.adsabs.harvard.edu/abs/{hit.get('bibcode')}/abstract"

            for c in candidates:
                found_any = True
                print(f"   found: '{c['phrase']}'  (explicit={c['explicit']})")
                candidate_rows.append({
                    "concept": anchor.concept,
                    "mission": anchor.mission,
                    "class_label": class_label,
                    "phrase": c["phrase"],
                    "explicit": c["explicit"],
                    "source_title": title,
                    "source_url": url,
                    "justification": c["justification"],
                })

                if not c["explicit"]:
                    continue  # implicit mentions never go into the query set

                sentence = clean_sentence(excerpt, c["phrase"])
                query_rows.append({
                    "query_text": sentence,
                    "class_label": class_label,
                    "mission": anchor.mission,
                    "query_type": "authentic",
                })

        if not found_any:
            empty_classes.append(f"{anchor.concept} [{anchor.mission}]")
        print()

        write_outputs(candidate_rows, query_rows)  # checkpoint after every anchor

    n_explicit = len(query_rows)
    n_implicit = len(candidate_rows) - n_explicit
    print(f"Wrote {len(candidate_rows)} candidate terms ({n_explicit} explicit, {n_implicit} implicit) "
          f"-> vision_candidate_terms.csv")
    print(f"Wrote {n_explicit} authentic query rows -> vision_mined_queries.csv")
    if empty_classes:
        print(f"\n{len(empty_classes)} classes came back with nothing found:")
        for c in empty_classes:
            print(f"  - {c}")
    print("\nvision_mined_queries.csv still needs a human spot-check before merging into "
          "eval/queries/msl_queries.csv or hirise_queries.csv.")


if __name__ == "__main__":
    main()
