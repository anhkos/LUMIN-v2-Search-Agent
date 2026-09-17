"""
Stage 4 — Extract (the piece the live test proved regex can't do).

Given a harvested excerpt and the concept it was found under, ask an LLM to
find candidate jargon phrases that plausibly refer to the concept but aren't
already in its known alias list — the "GDS" / "WICs" kind of finding.

Explicit-flag design (anti-circularity, from the original pipeline spec):
we ask the model to distinguish "this excerpt explicitly defines/introduces
the term" from "this excerpt just uses a term we're guessing might apply" —
only explicit=true extractions should ever be promoted to a real alias
without a human looking at it first. Never launder the extractor's own
inference into a label.

Requires ANTHROPIC_API_KEY. Untested in Claude's sandbox — no outbound key
available there; run and sanity-check this locally.
"""

import json
import os

import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
MODEL = "claude-sonnet-4-6"

EXTRACTION_PROMPT = """You are helping build a knowledge graph that maps planetary-science jargon to formal PDS4 archive fields.

Concept: {concept}
Description: {description}
Already-known aliases for this concept: {known_aliases}

Below is an excerpt from a real scientific paper that was found by searching for a known value or alias associated with this concept.

Excerpt:
\"\"\"
{excerpt}
\"\"\"

Find any words or short phrases in this excerpt that:
- plausibly refer to the SAME concept described above
- are NOT already in the known-aliases list (including obvious case/pluralization variants)
- are used as a real term in the text, not a one-off description you're inferring

For each one, report:
- "phrase": the exact phrase as it appears in the text
- "explicit": true if the paper explicitly introduces/defines the term (e.g. "also called X", "known as X (Y)"), false if it just uses the term in passing without defining it
- "justification": one short sentence on why this refers to the concept

Respond with ONLY a JSON array (no markdown, no preamble). If nothing qualifies, respond with an empty array: []
"""


def extract_candidates(anchor, excerpt: str) -> list[dict]:
    prompt = EXTRACTION_PROMPT.format(
        concept=anchor.concept,
        description=anchor.description,
        known_aliases=", ".join(anchor.aliases),
        excerpt=excerpt,
    )
    resp = client.messages.create(
        model=MODEL,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text.strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        candidates = json.loads(text)
    except json.JSONDecodeError:
        print(f"  [warn] could not parse LLM output as JSON, skipping: {text[:120]}")
        return []

    # Guard against the model echoing something already in known_aliases
    # despite instructions — cheap, deterministic, worth keeping regardless.
    known_lower = {a.lower() for a in anchor.aliases}
    return [c for c in candidates if c.get("phrase", "").lower() not in known_lower]
