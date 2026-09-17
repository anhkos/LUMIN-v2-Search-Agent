"""
Stage 2 — Harvest.

Thin wrapper around the NASA ADS search API, with disk caching so repeated
runs (or both consumers hitting the same anchor) don't re-spend API quota.
Requires ADS_API_KEY in the environment — free key at
https://ui.adsabs.harvard.edu/user/settings/token

This can't be exercised from Claude's sandbox (adsabs.harvard.edu isn't on
its allowed network list) — run and test this module locally.
"""

import hashlib
import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

ADS_API_KEY = os.environ.get("ADS_API_KEY")
ADS_SEARCH_URL = "https://api.adsabs.harvard.edu/v1/search/query"
CACHE_DIR = Path(os.environ.get("LUMIN_MINING_CACHE", ".ads_cache"))
CACHE_TTL_DAYS = 30


def _cache_path(query: str, rows: int) -> Path:
    key = hashlib.sha256(f"{query}|{rows}".encode()).hexdigest()[:16]
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / f"{key}.json"


def search(query: str, rows: int = 10, fields: str = "title,abstract,bibcode,year,citation_count") -> list[dict]:
    """
    Full-text search against ADS. Returns a list of doc dicts. Cached to disk
    for CACHE_TTL_DAYS — delete the cache file (or the whole cache dir) to
    force a fresh fetch.
    """
    cache_file = _cache_path(query, rows)
    if cache_file.exists():
        age_days = (time.time() - cache_file.stat().st_mtime) / 86400
        if age_days < CACHE_TTL_DAYS:
            return json.loads(cache_file.read_text())

    if not ADS_API_KEY:
        raise RuntimeError(
            "ADS_API_KEY not set. Get a free key at "
            "https://ui.adsabs.harvard.edu/user/settings/token and "
            "`export ADS_API_KEY=...` before running."
        )

    headers = {"Authorization": f"Bearer {ADS_API_KEY}"}
    params = {"q": f"full:({query})", "fl": fields, "rows": rows,
              "sort": "citation_count desc"}
    resp = requests.get(ADS_SEARCH_URL, headers=headers, params=params, timeout=20)
    resp.raise_for_status()
    docs = resp.json()["response"]["docs"]

    cache_file.write_text(json.dumps(docs, indent=2))
    return docs


def harvest_anchor(anchor, rows_per_query: int = 8) -> list[dict]:
    """
    Run every query for one Anchor (see anchors.py), tag each returned doc
    with which concept/query it came from, and dedupe by bibcode across the
    anchor's own queries (an MY34 paper can legitimately match two of its
    own query variants — no need to process it twice).
    """
    seen_bibcodes = set()
    hits = []
    for query in anchor.queries:
        query = query.split("  #")[0].strip()  # strip the low-precision comment tag
        for doc in search(query, rows=rows_per_query):
            bibcode = doc.get("bibcode")
            if bibcode in seen_bibcodes:
                continue
            seen_bibcodes.add(bibcode)
            hits.append({**doc, "matched_query": query, "concept": anchor.concept})
    return hits


if __name__ == "__main__":
    import sys
    from anchors import parse_summary, DEFAULT_SUMMARY_PATH

    summary_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SUMMARY_PATH
    concept_filter = sys.argv[2] if len(sys.argv) > 2 else None

    anchors = parse_summary(summary_path)
    if concept_filter:
        anchors = [a for a in anchors if concept_filter.lower() in a.concept.lower()]

    for a in anchors:
        print(f"── {a.concept} ──")
        try:
            hits = harvest_anchor(a, rows_per_query=5)
        except RuntimeError as e:
            print(f"   {e}")
            continue
        for h in hits:
            print(f"   [{h['matched_query']}] {h.get('title', ['?'])[0]} ({h.get('year')}, cited {h.get('citation_count')})")
        print()
