"""
Stage 1 — Anchors.

Parses lumin_kg_summary.txt (or eventually the graph JSON directly) into one
record per concept: its description, its known aliases, and a couple of
candidate ADS full-text queries built from its most distinctive values.

This replaces the earlier hand-picked ANCHORS list in the prototype with
something that walks the *whole* graph, so both downstream consumers (paper
search and dataset mining) can operate on every concept, not just three.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_SUMMARY_PATH = str(Path(__file__).parent / "lumin_kg_summary.txt")


@dataclass
class Anchor:
    concept: str
    category: str            # e.g. "seasonal_temporal", "instrument_mode"
    description: str
    aliases: list = field(default_factory=list)
    slot_values: str = ""    # raw slot-value string, kept for reference
    queries: list = field(default_factory=list)  # candidate ADS full-text queries


# One block looks like:
#
# ── Global Dust Storm [atmospheric]  instance_of AtmosphericEvent
#    A planet-encircling dust event on Mars where dust optical depth ...
#    Slot values: {"opacity": {"min": 3.0, "unit": "tau"}}
#    occurs_during -> MartianDustStormSeason
#    Aliases (10):
#      global dust storm, planet-encircling dust event, ...
#
BLOCK_RE = re.compile(
    r"^── (?P<name>.+?) \[(?P<category>[^\]]+)\]\s+instance_of\s+\S+\n"
    r"(?P<body>(?:^(?!──).*\n?)*)",
    re.MULTILINE,
)
ALIASES_RE = re.compile(r"Aliases \(\d+\):\s*\n\s*(.+)", re.MULTILINE)
SLOTVALUES_RE = re.compile(r"Slot values:\s*(\{.*\})")


def parse_summary(path: str) -> list[Anchor]:
    text = open(path, encoding="utf-8").read()
    # skip the "Class layer" section entirely — we only want concept blocks
    if "Aliases" not in text:
        raise ValueError("No concept blocks with aliases found — wrong file?")

    anchors = []
    for m in BLOCK_RE.finditer(text):
        name, category, body = m["name"], m["category"], m["body"]
        if "Aliases" not in body:
            continue  # e.g. malformed / class-layer leakage

        desc_line = body.strip().split("\n")[0].strip()
        alias_match = ALIASES_RE.search(body)
        aliases = [a.strip().lower() for a in alias_match.group(1).split(",")] if alias_match else []
        slotval_match = SLOTVALUES_RE.search(body)
        slot_values = slotval_match.group(1) if slotval_match else ""

        anchors.append(Anchor(
            concept=name.strip(),
            category=category.strip(),
            description=desc_line,
            aliases=aliases,
            slot_values=slot_values,
        ))

    for a in anchors:
        a.queries = _build_queries(a)
    return anchors


# Words too common in planetary-science literature to work as a standalone
# anchor — they need a value or a second word attached, or they'll flood
# every harvest with irrelevant hits.
GENERIC_WORDS = {"aphelion", "perihelion", "dust event", "storm season",
                  "cloud belt", "polar region", "dust storm season"}


def _value_queries(anchor: Anchor) -> list[str]:
    """
    Prefer anchoring by VALUE (Ls degrees) over anchoring by word — this is
    the actual design principle: an alias is opinion, a value is fact. Only
    fires for concepts with a numeric ls_range / seasonal_window slot.
    """
    try:
        slots = json.loads(anchor.slot_values) if anchor.slot_values else {}
    except json.JSONDecodeError:
        return []

    for key in ("ls_range", "seasonal_window"):
        rng = slots.get(key)
        if isinstance(rng, dict) and "min" in rng and "max" in rng:
            lo, hi = rng["min"], rng["max"]
            return [f'"Ls {lo}" OR "solar longitude {lo}"',
                    f'"Ls {hi}" OR "solar longitude {hi}"']
    return []


def _build_queries(anchor: Anchor) -> list[str]:
    """
    Candidate ADS full-text queries for this concept, value-anchored queries
    first (more precise), then distinctive alias/acronym queries as backup.
    Generic single words are skipped entirely rather than shipped as a
    misleadingly "specific-looking" query that will just flood with noise.
    """
    aliases = list(dict.fromkeys(anchor.aliases))  # dedupe, preserve order
    queries = _value_queries(anchor)

    for alias in aliases:
        if alias in GENERIC_WORDS:
            continue
        # short, alphanumeric, likely a code/acronym (MY34, NPLD, SHARAD...)
        if re.fullmatch(r"[a-z0-9]{2,8}", alias.replace(" ", "")) and len(alias) <= 8:
            q = f'"{alias.upper()}"'
            if q not in queries:
                queries.append(q)

    if not queries:
        # last resort: shortest non-generic alias, flagged as low-precision
        candidates = [a for a in aliases if a not in GENERIC_WORDS]
        if candidates:
            queries.append(f'"{min(candidates, key=len)}"  # low-precision fallback')

    return queries[:4]


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SUMMARY_PATH
    anchors = parse_summary(path)
    print(f"Parsed {len(anchors)} concept anchors:\n")
    for a in anchors:
        print(f"── {a.concept} [{a.category}]")
        print(f"   aliases: {a.aliases}")
        print(f"   queries: {a.queries}")
        print()
