"""
Vision-class anchors — adapted from lumin_mining/anchors.py for the retrieval-benchmark side
of the project.

The KG tool's anchors are PDS4 schema concepts ("southern summer" -> a Ls-degree range). These
anchors are visual/geological classes from the actual MSL and HiRISE label sets, and the goal
isn't a schema filter — it's a real sentence from a real paper that describes what the class
looks like, to use as an authentic (non-template) retrieval query.

Same underlying principle as the original anchors.py: anchor searches on something distinctive
that a paper would actually contain, not a bare generic word that floods ADS with noise. Most
of the MSL class names ARE bare generic words (Ground, Wheel, Horizon, Scoop...), so this module
leans harder on the "generic word needs a qualifier" guard than the KG version did.

ID MAPPING — verified 2026-09-19 against msl_synset_words-indexed.txt (Zenodo 1049137, inside
msl-images.zip) and cross-checked against real per-image counts in the train/val/test label
files. This confirmed the MSL_ANCHORS concepts below are already the correct 24-class v1
taxonomy (not the newer 19-class Lu & Wagstaff 2020 set) — the concept list itself didn't need
fixing. What WAS wrong was mars-clip-bench's own `finetune/finetune_clip.py` CLASS_NAMES dict,
which had drifted onto a different, partly-fabricated set of names (now fixed there too). See
MSL_CONCEPT_TO_ID / HIRISE_CONCEPT_TO_ID below for the concept -> class_label mapping used when
writing mined rows.
"""

from dataclasses import dataclass, field


@dataclass
class VisionAnchor:
    concept: str            # class name, matched against your manifest after the ID check above
    mission: str             # "HiRISE" or "MSL"
    description: str
    seed_aliases: list = field(default_factory=list)   # phrasings already known to describe it
    queries: list = field(default_factory=list)         # candidate ADS full-text queries

    def __post_init__(self):
        if not self.queries:
            self.queries = _build_queries(self)


# Bare words that will flood ADS with irrelevant hits unless paired with a qualifier — mirrors
# GENERIC_WORDS in the original anchors.py, but bigger, since almost every MSL class name is a
# plain English word rather than jargon.
GENERIC_WORDS = {
    "ground", "horizon", "wheel", "scoop", "inlet", "turret", "drill", "other",
}

# Every MSL query gets qualified with the rover/mission context, since "wheel" or "drill" alone
# means nothing to a search engine. Rotate through these rather than repeating one phrase, so the
# 4 queries per class aren't near-duplicates of each other.
MSL_QUALIFIERS = ['"Curiosity rover"', '"Mars Science Laboratory"', '"MSL rover"']


def _build_queries(anchor: "VisionAnchor") -> list:
    if anchor.mission == "HiRISE":
        # HiRISE classes are real, specific geomorphology terms — they can mostly stand on their
        # own as ADS queries, the same way the KG tool anchors on a distinctive alias.
        base = [f'"{a}"' for a in anchor.seed_aliases[:3]]
        return (base + [f'"{anchor.concept.replace("_", " ")}" AND "HiRISE"'])[:4]

    # MSL: pair every seed alias with a rotating mission qualifier, and skip anything that would
    # otherwise be shipped as a bare generic word.
    queries = []
    for i, alias in enumerate(anchor.seed_aliases[:4]):
        if alias.lower() in GENERIC_WORDS and "curiosity" not in alias.lower() and "msl" not in alias.lower():
            qualifier = MSL_QUALIFIERS[i % len(MSL_QUALIFIERS)]
            queries.append(f'"{alias}" AND {qualifier}')
        else:
            queries.append(f'"{alias}"')
    return queries[:4]


# ---------------------------------------------------------------------------------------------
# HiRISE — 8 classes, from the original paper's confusion matrix (Fig. 2) / Table II.
# These are established Mars geomorphology terms with a real descriptive literature, so I'd
# expect ADS mining to work well here.
# ---------------------------------------------------------------------------------------------

HIRISE_ANCHORS = [
    VisionAnchor(
        concept="other", mission="HiRISE",
        description="Catch-all background terrain not matching any of the other 7 classes "
                     "(≈84% of the corpus). Not a coherent visual concept, so mining a "
                     "meaningful natural-language description for it may not be possible — "
                     "worth trying, but don't be surprised if this one comes back empty.",
        seed_aliases=["martian surface terrain"],
    ),
    VisionAnchor(
        concept="crater", mission="HiRISE",
        description="Impact crater — a circular depression, often with a raised rim, formed by "
                     "meteorite impact.",
        seed_aliases=["impact crater", "crater rim", "impact structure"],
    ),
    VisionAnchor(
        concept="dark_dune", mission="HiRISE",
        description="A dune or dune field with low albedo (dark-toned), typically basaltic sand.",
        seed_aliases=["dark dune", "dark sand dune", "basaltic dune"],
    ),
    VisionAnchor(
        concept="slope_streak", mission="HiRISE",
        description="A dark or bright linear marking on a slope, generally attributed to dry "
                     "granular flow. NOTE: this is distinct from 'recurring slope lineae (RSL)', "
                     "a separate, later-discovered seasonal phenomenon — don't conflate the two "
                     "without checking which one your class actually depicts; worth a quick "
                     "confirmation with Steven Lu or the original label definitions.",
        seed_aliases=["slope streak", "slope streaks"],
    ),
    VisionAnchor(
        concept="bright_dune", mission="HiRISE",
        description="A dune or dune field with high albedo (bright-toned), often associated with "
                     "different composition or frost cover relative to dark dunes.",
        seed_aliases=["bright dune", "bright sand dune", "light-toned dune"],
    ),
    VisionAnchor(
        concept="impact_ejecta", mission="HiRISE",
        description="Material thrown out and deposited around an impact crater, often visible "
                     "as a blanket or rays extending from the rim.",
        seed_aliases=["impact ejecta", "ejecta blanket", "ejecta rays"],
    ),
    VisionAnchor(
        concept="swiss_cheese", mission="HiRISE",
        description="Pitted terrain in the south polar CO2 ice cap, formed by seasonal "
                     "sublimation, resembling holes in Swiss cheese.",
        seed_aliases=["swiss cheese terrain", "south polar pits", "CO2 ice sublimation pits"],
    ),
    VisionAnchor(
        concept="spider", mission="HiRISE",
        description="Araneiform (spider-shaped) channel networks in the south polar region, "
                     "thought to form from CO2 gas jet erosion beneath seasonal ice.",
        seed_aliases=["araneiform terrain", "spiders", "araneiform channels"],
    ),
]

# ---------------------------------------------------------------------------------------------
# MSL — 24 classes, confirmed against Wagstaff et al. 2018 ("Deep Mars: CNN Classification of
# Mars Imagery for the PDS Imaging Atlas"), Figure 1 — the same v1/6,691-image/24-class dataset
# the original LUMIN paper cites (Zenodo 1049137). This is the rover-hardware-focused set, NOT
# the newer 19-class science-target set (Lu & Wagstaff 2020) — don't mix the two up if you see
# other MSL class lists online, several exist.
#
# These are almost all rover hardware/instrument names, which is exactly the coverage gap
# hypothesis from earlier: there may not be much natural "here's what this looks like" language
# about a drill or a wheel in the literature the way there is for geological features. If this
# comes back much thinner than HiRISE's, that's a finding, not a bug.
# ---------------------------------------------------------------------------------------------

MSL_ANCHORS = [
    VisionAnchor("APXS", "MSL", "Alpha Particle X-Ray Spectrometer — an element-analysis "
                 "instrument mounted on the rover's robotic arm.",
                 ["APXS", "Alpha Particle X-Ray Spectrometer"]),
    VisionAnchor("APXS CT", "MSL", "Calibration target used to check the APXS instrument.",
                 ["APXS calibration target", "APXS CT"]),
    VisionAnchor("ChemCam CT", "MSL", "Calibration target used by the ChemCam laser instrument.",
                 ["ChemCam calibration target", "ChemCam CT"]),
    VisionAnchor("Chemin inlet", "MSL", "The sample inlet of the CheMin mineralogy instrument, "
                 "shown in its open position.",
                 ["CheMin inlet", "CheMin sample inlet"]),
    VisionAnchor("Drill", "MSL", "The rover's rotary-percussive drill, used to bore into rock "
                 "and collect powder samples.",
                 ["drill", "rock drill", "sample drill"]),
    VisionAnchor("Drill hole", "MSL", "The hole left in a rock after drilling.",
                 ["drill hole", "drilled hole", "borehole"]),
    VisionAnchor("DRT front", "MSL", "The Dust Removal Tool (a motorized brush) viewed from the "
                 "front.",
                 ["Dust Removal Tool", "DRT"]),
    VisionAnchor("DRT side", "MSL", "The Dust Removal Tool viewed from the side.",
                 ["Dust Removal Tool", "DRT"]),
    VisionAnchor("Ground", "MSL", "Flat, close-range terrain directly beneath or near the rover.",
                 ["ground", "surface terrain", "regolith surface"]),
    VisionAnchor("Horizon", "MSL", "The distant skyline/horizon as seen from the rover.",
                 ["horizon", "distant horizon", "martian skyline"]),
    VisionAnchor("Inlet", "MSL", "A sample-intake port on the rover (distinct from the CheMin "
                 "inlet specifically).",
                 ["inlet", "sample inlet"]),
    VisionAnchor("MAHLI", "MSL", "The Mars Hand Lens Imager — a close-up camera on the robotic "
                 "arm's turret.",
                 ["MAHLI", "Mars Hand Lens Imager"]),
    VisionAnchor("MAHLI CT", "MSL", "Calibration target used by the MAHLI camera.",
                 ["MAHLI calibration target", "MAHLI CT"]),
    VisionAnchor("Mastcam", "MSL", "The rover's mast-mounted stereo camera system.",
                 ["Mastcam", "Mast Camera"]),
    VisionAnchor("Mastcam CT", "MSL", "Calibration target used by Mastcam.",
                 ["Mastcam calibration target", "Mastcam CT"]),
    VisionAnchor("Observ. tray", "MSL", "The observation tray used to hold sample material for "
                 "imaging.",
                 ["observation tray"]),
    VisionAnchor("Portion box", "MSL", "A container used to portion out collected sample "
                 "material.",
                 ["portion box"]),
    VisionAnchor("Portion tube", "MSL", "A tube used in the sample portioning process.",
                 ["portion tube"]),
    VisionAnchor("P. tube op.", "MSL", "The opening of the portion tube.",
                 ["portion tube opening"]),
    VisionAnchor("REMS-UV", "MSL", "The ultraviolet sensor of the Rover Environmental "
                 "Monitoring Station.",
                 ["REMS-UV", "Rover Environmental Monitoring Station"]),
    VisionAnchor("Rear deck", "MSL", "The rear deck of the rover body.",
                 ["rover rear deck", "rear deck"]),
    VisionAnchor("Scoop", "MSL", "A scoop tool used to collect loose surface material.",
                 ["scoop", "sample scoop"]),
    VisionAnchor("Turret", "MSL", "The instrument turret at the end of the robotic arm, holding "
                 "MAHLI, APXS, drill, and DRT.",
                 ["turret", "arm turret", "instrument turret"]),
    VisionAnchor("Wheel", "MSL", "One of the rover's six wheels — a subject of particular "
                 "interest due to unexpected wear.",
                 ["wheel", "rover wheel", "wheel damage"]),
]

ALL_ANCHORS = HIRISE_ANCHORS + MSL_ANCHORS

# ---------------------------------------------------------------------------------------------
# concept -> class_label (numeric ID), verified against msl_synset_words-indexed.txt / the
# dataset's own labels-map-proj-v3.txt. class 22 ("sun") is skipped — 0 images use it, and it
# has no MSL_ANCHORS entry to map. Explicit dict rather than positional zip against MSL_ANCHORS,
# so a future edit to the anchor list order can't silently mis-map an ID.
# ---------------------------------------------------------------------------------------------
MSL_CONCEPT_TO_ID = {
    "APXS": 0,
    "APXS CT": 1,
    "ChemCam CT": 2,
    "Chemin inlet": 3,
    "Drill": 4,
    "Drill hole": 5,
    "DRT front": 6,
    "DRT side": 7,
    "Ground": 8,
    "Horizon": 9,
    "Inlet": 10,
    "MAHLI": 11,
    "MAHLI CT": 12,
    "Mastcam": 13,
    "Mastcam CT": 14,
    "Observ. tray": 15,
    "Portion box": 16,
    "Portion tube": 17,
    "P. tube op.": 18,
    "REMS-UV": 19,
    "Rear deck": 20,
    "Scoop": 21,
    "Turret": 23,
    "Wheel": 24,
}

HIRISE_CONCEPT_TO_ID = {
    "other": 0,
    "crater": 1,
    "dark_dune": 2,
    "slope_streak": 3,
    "bright_dune": 4,
    "impact_ejecta": 5,
    "swiss_cheese": 6,
    "spider": 7,
}


def class_label_for(anchor: "VisionAnchor") -> int:
    table = MSL_CONCEPT_TO_ID if anchor.mission == "MSL" else HIRISE_CONCEPT_TO_ID
    return table[anchor.concept]


if __name__ == "__main__":
    # Sanity-check both tables cover every anchor before anything downstream relies on them.
    for anchor in ALL_ANCHORS:
        class_label_for(anchor)
    print(f"ID mapping OK for all {len(ALL_ANCHORS)} anchors.\n")

    for a in ALL_ANCHORS:
        print(f"-- {a.concept} [{a.mission}]")
        print(f"   {a.description}")
        print(f"   queries: {a.queries}")
        print()
