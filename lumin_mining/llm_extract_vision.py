"""
Stage 4 (vision variant) — Extract.

Adapted from llm_extract.py for the retrieval-benchmark side of the project. Same shape
(excerpt in, candidate phrases out, explicit/implicit flag) but a different question: the KG
version asks "does this map to a PDS4 field," this version asks "does this excerpt actually
describe what the visual class looks like."

Explicit-flag design (anti-circularity, unchanged from the original pipeline spec): explicit
means the excerpt clearly describes the class's visual appearance (shape, texture, color, size,
visual context); implicit means the term is just used in passing without real descriptive
content. Only explicit=true extractions should ever be promoted into the query set without a
human double-check — an implicit mention (e.g. "the drill was deployed on sol 400") tells you
nothing about what a drill LOOKS like, and shipping it as a "visual description" would be
circular: the model would just be quizzed on jargon co-occurrence, not appearance.

Requires ANTHROPIC_API_KEY. Run and sanity-check this locally before trusting its output.
"""

import json
import os

import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
MODEL = "claude-sonnet-4-6"

EXTRACTION_PROMPT = """You are helping build an authentic (non-template) natural-language retrieval query set for a Mars image classification/retrieval benchmark.

Visual class: {concept}
Mission: {mission}
Description: {description}

Below is an excerpt from a real scientific paper that was found by searching for a term associated with this visual class.

Excerpt:
\"\"\"
{excerpt}
\"\"\"

Given this excerpt and the visual class it was found under, does the excerpt describe what {concept} actually looks like — its shape, texture, color, size, or visual context? Extract the descriptive phrase or sentence if so.

Find any phrase or sentence in this excerpt that:
- actually describes the visual appearance of {concept} (what a photo of it would show), not just
  that it exists, was used, or was studied
- is a real, naturally-occurring sentence from the text, not something you are inferring or
  paraphrasing from scattered context

For each one, report:
- "phrase": the exact descriptive phrase or sentence as it appears in the text
- "explicit": true if the excerpt clearly describes the class's visual appearance (shape,
  texture, color, size, visual context), false if the term is just used in passing without real
  descriptive content (e.g. mentioning it was deployed, sampled, or studied on a given sol,
  without saying what it looks like)
- "justification": one short sentence on why this phrase describes the class's appearance

Respond with ONLY a JSON array (no markdown, no preamble). If nothing qualifies, respond with an empty array: []
"""


def extract_candidates(anchor, excerpt: str) -> list[dict]:
    prompt = EXTRACTION_PROMPT.format(
        concept=anchor.concept,
        mission=anchor.mission,
        description=anchor.description,
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

    return [c for c in candidates if c.get("phrase")]
