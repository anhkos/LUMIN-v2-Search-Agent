"""
T3-out mining prototype — "feel it out" version.

Core idea: search literature BY VALUE (e.g. "MY34", "Ls 270") instead of by
keyword, then harvest the surrounding phrases. Since the value is unambiguous,
whatever jargon shows up near it is real, naturally-occurring vocabulary we
didn't anticipate — the actual test cases for schema-opacity.

This script is stage 1 (anchors) + stage 3 (harvest) + stage 4 (prefilter),
compressed into one file for quick tinkering. It runs in two modes:

  - LIVE mode: set ADS_API_KEY and it hits the real NASA ADS API.
  - MOCK mode (default here): no key needed, uses a couple of hand-written
    sample excerpts so you can see the full shape of the pipeline right now.

Get a free ADS key at https://ui.adsabs.harvard.edu/user/settings/token
"""

import os
import re
import json
import textwrap

from dotenv import load_dotenv

load_dotenv()

ADS_API_KEY = os.environ.get("ADS_API_KEY")
ADS_SEARCH_URL = "https://api.adsabs.harvard.edu/v1/search/query"

# ---------------------------------------------------------------------------
# Stage 1 — Anchors: concrete values pulled from the frozen graph.
# In the real pipeline these come from lumin_kg_summary.txt automatically;
# hand-picked here from the actual graph content for a quick test.
# ---------------------------------------------------------------------------

ANCHORS = [
    {
        "concept": "MartianDustStormSeason",
        "value": "MY34",
        "query": '"MY34"',
        "known_aliases": {"my34", "mars year 34", "2018 dust storm year", "2018 global storm"},
    },
    {
        "concept": "MartianSouthernSummer",
        "value": "Ls 270-360",
        "query": '"Ls 270" OR "solar longitude 270"',
        "known_aliases": {"southern summer", "southern hemisphere summer", "martian summer", "southern warm season"},
    },
    {
        "concept": "WaterIceClouds",
        "value": "aphelion cloud belt",
        "query": '"aphelion cloud belt"',
        "known_aliases": {"water ice clouds", "ice clouds", "aphelion cloud belt", "acb", "martian clouds",
                           "equatorial clouds", "cloud belt", "cloud opacity"},
    },
]

# ---------------------------------------------------------------------------
# Mock corpus — stand-ins for real ADS full-text hits, so the pipeline has
# something to chew on without network access. Swap for real API calls below.
# ---------------------------------------------------------------------------

MOCK_HITS = {
    "MY34": [
        "During the MY34 planet-encircling dust event, opacity at Gale crater "
        "exceeded tau=8, forcing Curiosity into a low-power survival mode for "
        "several weeks before nominal operations resumed.",
    ],
    "Ls 270-360": [
        "Observations were restricted to the perihelic warm season (Ls ~270-300), "
        "a period colloquially referred to by the team as 'dust season onset', "
        "coinciding with elevated regional storm frequency.",
    ],
    "aphelion cloud belt": [
        "The equatorial cloud band, sometimes called the 'tropical cloud belt' in "
        "earlier literature, forms reliably during the aphelion cloud belt season "
        "and has been linked to reduced daytime surface temperatures.",
    ],
}

# ---------------------------------------------------------------------------
# Stage 3 — Harvest
# ---------------------------------------------------------------------------

def harvest_live(query, rows=5):
    """Real ADS call. Requires ADS_API_KEY. Uncomment `requests` usage to run."""
    import requests
    headers = {"Authorization": f"Bearer {ADS_API_KEY}"}
    params = {"q": f'full:({query})', "fl": "title,abstract,bibcode", "rows": rows}
    resp = requests.get(ADS_SEARCH_URL, headers=headers, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()["response"]["docs"]


def harvest_mock(anchor):
    """Offline stand-in: returns the hand-written excerpts for this anchor."""
    return MOCK_HITS.get(anchor["value"], [])


# ---------------------------------------------------------------------------
# Stage 4 — Prefilter: regex window around the anchor value (±2 sentences,
# simplified here to whole-excerpt since our mock excerpts are short)
# ---------------------------------------------------------------------------

def prefilter(excerpt, anchor):
    sentences = re.split(r"(?<=[.!?])\s+", excerpt)
    hits = [s for s in sentences if anchor["value"].split()[0].lower() in s.lower()
            or any(w.lower() in s.lower() for w in anchor["value"].split())]
    return hits or sentences  # fall back to whole excerpt if no direct hit


# ---------------------------------------------------------------------------
# Stage 5 (stub) — naive candidate-phrase flagging.
# Real pipeline uses an LLM with an explicit-flag prompt; this is a cheap
# heuristic just to see the shape: flag quoted or scare-quoted terms, and
# terms not already in known_aliases.
# ---------------------------------------------------------------------------

def flag_candidates(sentences, anchor):
    candidates = []
    for s in sentences:
        quoted = re.findall(r"'([^']+)'|\"([^\"]+)\"", s)
        for pair in quoted:
            phrase = (pair[0] or pair[1]).strip()
            if phrase.lower() not in anchor["known_aliases"]:
                candidates.append(phrase)
    return candidates


# ---------------------------------------------------------------------------
# Run it
# ---------------------------------------------------------------------------

def main():
    live = bool(ADS_API_KEY)
    print(f"Mode: {'LIVE (real ADS API)' if live else 'MOCK (offline sample data)'}\n")

    results = []
    for anchor in ANCHORS:
        print(f"── {anchor['concept']}  (anchor: {anchor['value']}) " + "─" * 20)
        print(f"   ADS query would be: full:({anchor['query']})")

        if live:
            docs = harvest_live(anchor["query"])
            excerpts = [d.get("abstract", "") for d in docs if d.get("abstract")]
        else:
            excerpts = harvest_mock(anchor)

        for excerpt in excerpts:
            sentences = prefilter(excerpt, anchor)
            candidates = flag_candidates(sentences, anchor)

            print("   Harvested excerpt:")
            print(textwrap.indent(textwrap.fill(excerpt, 70), "     "))
            if candidates:
                print(f"   -> Candidate NEW jargon (not in known aliases): {candidates}")
            else:
                print("   -> No new candidate phrases flagged (heuristic only — real")
                print("      pipeline uses an LLM extraction pass here, not regex)")
            print()

            results.append({
                "concept": anchor["concept"],
                "anchor_value": anchor["value"],
                "excerpt": excerpt,
                "candidates": candidates,
            })

    with open("mining_demo_output.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved: mining_demo_output.json")


if __name__ == "__main__":
    main()