"""
build_kg.py  (v2 — four-layer architecture)
-------------------------------------------
Builds the LUMIN knowledge graph. v2 introduces a domain-general CLASS layer
between Mars-specific concepts and PDS4 schema leaves, per JPL review feedback.

Graph structure (four layers):

    [alias]   --is_alias_of-->  [concept]                       (surface forms)
    [concept] --instance_of-->  [class]      value payload here (Mars-specific)
    [concept] --occurs_during-> [concept]    cross-links        (Mars-specific)
    [class]   --grounded_in-->  [schema leaf] field binding     (domain-general)

Key design change vs v1 — SLOT-BASED VALUE GROUNDING:
    A class declares named grounding SLOTS, each bound to exactly one schema
    leaf (the *field*). A concept's instance_of edge carries scalar/range
    VALUES keyed by slot name (the *value*). Traversal composes:
        filter field  = leaf bound to the slot   (from class --grounded_in-->)
        filter value  = slot value               (from concept --instance_of-->)
    Consequences:
      * Bounding boxes are now scalars per side (west=270, east=330), not
        ambiguous ranges duplicated on every side.  (v1 bug fix)
      * The class layer is load-bearing: every complete path is
        alias -> concept -> class -> leaf (3 hops), so tier depth is
        topological, not just linguistic.
      * Only the concept + alias layers are Mars-specific. Porting to a new
        domain = new concept/alias layers over reusable classes.

Edge vocabulary + traversal priorities are exported in graph metadata
(G.graph["edge_priorities"]) so traversal.py reads them from the artifact
instead of hard-coding.

!! BREAKING CHANGE for traversal.py:
   - edge types renamed/added: instance_of, grounded_in, occurs_during
     (v1 types corresponds_to / measured_by / instrument_of are gone)
   - value hints moved from concept->leaf edges to concept->class edges,
     keyed by slot; leaf binding is on class->leaf edges (attr "slot")
   - aliases may carry their own value overrides (attr "value_hint"),
     e.g. alias "MY34" resolves to a concrete date range
   - MartianPolarRegion values use {"variants": [...]} (disjoint boxes)

Domain facts marked  # JPL-VERIFY  were corrected to standard conventions
but must be confirmed in the JPL validation pass before benchmark freeze.

Input:  schema_nodes_final.json  (deduped, from filter_schema.py + dedup step)
Output: lumin_kg.json            (full graph in NetworkX node-link format)
        lumin_kg_summary.txt     (human-readable graph summary)

Usage:
    pip install networkx
    python build_kg.py [path/to/schema_nodes_final.json]
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import networkx as nx

# ── Edge vocabulary ────────────────────────────────────────────────────────────
# Priorities consumed by traversal.py (best-first search / rho_edge factor).
# Reserved types (located_in, observed_by, related_to) are defined now so the
# vocabulary is stable when those edges are added; they carry lower priority.

EDGE_PRIORITIES = {
    "is_alias_of":   1.0,   # alias   -> concept
    "instance_of":   1.0,   # concept -> class     (carries slot values)
    "grounded_in":   1.0,   # class   -> leaf      (carries slot name)
    "occurs_during": 0.85,  # concept -> concept   (event -> season/interval)
    "located_in":    0.85,  # reserved: concept -> concept (spatial containment)
    "observed_by":   0.80,  # reserved: concept -> concept (phenomenon -> instrument)
    "related_to":    0.50,  # reserved: weak association
}

# ── Class layer (domain-general) ───────────────────────────────────────────────
# Each class:
#   id, label, description : embedded at query time like concepts
#   slots : {slot_name: schema_leaf_key}   one leaf per slot
# Nothing in this layer is Mars-specific. Porting to a new domain reuses these
# classes (or siblings) and swaps only the concept/alias layers.

CLASSES = [
    {
        "id": "Season",
        "label": "Season",
        "description": (
            "A recurring interval of a planetary year defined by the planet's "
            "position in its orbit, expressed as a solar longitude (Ls) range."
        ),
        "slots": {
            "ls_range": "Time_Coordinates.solar_longitude",
        },
    },
    {
        "id": "OrbitalEvent",
        "label": "Orbital event",
        "description": (
            "A recurring point or short window in a body's orbit, such as "
            "perihelion or aphelion passage, expressed as a solar longitude range."
        ),
        "slots": {
            "ls_range": "Time_Coordinates.solar_longitude",
        },
    },
    {
        "id": "TemporalInterval",
        "label": "Temporal interval",
        "description": (
            "A concrete calendar interval with a start and stop time, such as a "
            "numbered planetary year or a named observing period."
        ),
        "slots": {
            "start": "Time_Coordinates.start_date_time",
            "stop":  "Time_Coordinates.stop_date_time",
        },
    },
    {
        "id": "GeographicRegion",
        "label": "Geographic region",
        "description": (
            "A named surface region bounded by a latitude/longitude box: "
            "scalar west, east, south, and north bounding coordinates."
        ),
        "slots": {
            "west":  "Bounding_Coordinates.west_bounding_coordinate",
            "east":  "Bounding_Coordinates.east_bounding_coordinate",
            "south": "Bounding_Coordinates.south_bounding_coordinate",
            "north": "Bounding_Coordinates.north_bounding_coordinate",
        },
    },
    {
        "id": "AtmosphericEvent",
        "label": "Atmospheric event",
        "description": (
            "An atmospheric phenomenon such as a dust storm or cloud formation, "
            "characterized by an optical depth signature and, optionally, the "
            "seasonal window in which it occurs. Distinct from a Season: an "
            "atmospheric event OCCURS DURING a seasonal window; it is not itself "
            "a definition of one."
        ),
        "slots": {
            "opacity":         "Radiometric_Correction.atmospheric_opacity",
            "seasonal_window": "Time_Coordinates.solar_longitude",
        },
    },
    {
        "id": "Instrument",
        "label": "Instrument",
        "description": (
            "A science instrument aboard a spacecraft. Identified by instrument "
            "name; associated with a host mission; optionally characterized by "
            "pixel resolution."
        ),
        "slots": {
            "instrument_name":  "Instrument.name",
            "mission_name":     "Mission_PDS3.mission_name",
            "pixel_resolution": "Coordinate_Representation.pixel_resolution_x",
        },
    },
    {
        "id": "MissionPhase",
        "label": "Mission phase",
        "description": (
            "A named operational phase of a mission (e.g. cruise, aerobraking, "
            "primary science, extended mission), with an associated mission and "
            "optional calendar window."
        ),
        "slots": {
            "phase_name":   "Mission_Phase.mission_phase_name",
            "mission_name": "Mission_PDS3.mission_name",
            "start":        "Time_Coordinates.start_date_time",
            "stop":         "Time_Coordinates.stop_date_time",
        },
    },
]

# ── Concept layer (Mars-specific) ──────────────────────────────────────────────
# Each concept:
#   id, label, category, description : as in v1
#   class_of     : id of the class this concept instantiates
#   slot_values  : {slot_name: value} — values only; fields come from the class.
#                  Range values: {"min":..,"max":..,"unit":..}
#                  Enum/scalar:  {"value":..}
#                  Disjoint alternatives: {"variants": [ {slot_values...}, ... ]}
#   occurs_during: optional list of concept ids (event -> season cross-links)
#   aliases      : NL surface strings (T1 entry points)
#   alias_value_hints : optional {alias: {slot: value}} overrides — lets a
#                  specific surface form (e.g. "MY34") resolve to concrete
#                  values while sharing the parent concept's structure.
#
# v1 bugs fixed here, all marked inline:
#   [FIX-1] aphelion/perihelion season aliases were swapped across the two
#           summer concepts, contradicting the MarsAphelion/MarsPerihelion
#           nodes in the same graph. Southern summer is PERIHELION season.
#   [FIX-2] Ls windows tightened: v1 "southern summer" = 180-360 was southern
#           spring+summer. Proper solstice seasons are 90-degree windows.
#   [FIX-3] instruments now ground to Instrument.name; mission_name carries the
#           actual mission (v1 wrote instrument names into mission_name).
#   [FIX-4] bounding boxes are scalar per side; HellasBasin west side restored.
#   [FIX-5] MarsYear aliases resolve to concrete date ranges (v1: value None,
#           traversal terminated with nothing to extract).
#   [FIX-6] GlobalDustStorm's dangling corresponds_to->start_date_time edge
#           replaced by per-event alias date hints + occurs_during cross-link.
#   [NEW]   MartianDustStormSeason concept added — the correct home for the
#           "dust storm season" alias (v1 hung it on northern summer).

CONCEPTS = [

    # ── Seasonal / Temporal ────────────────────────────────────────────────────
    {
        "id": "MartianSouthernSummer",
        "label": "Martian Southern Summer",
        "category": "seasonal_temporal",
        "class_of": "Season",
        "description": (
            "The period of Mars southern hemisphere summer, from southern summer "
            "solstice to southern autumn equinox: solar longitude 270 to 360 "
            "degrees. Coincides with perihelion season, warmer temperatures, and "
            "elevated dust activity."
        ),
        "slot_values": {
            "ls_range": {"min": 270, "max": 360, "unit": "deg"},  # JPL-VERIFY [FIX-2]
        },
        "aliases": [
            "southern summer", "southern hemisphere summer",
            "martian summer", "southern warm season",            # [FIX-1] "aphelion season" removed
        ],
    },
    {
        "id": "MartianNorthernSummer",
        "label": "Martian Northern Summer",
        "category": "seasonal_temporal",
        "class_of": "Season",
        "description": (
            "The period of Mars northern hemisphere summer, from northern summer "
            "solstice to northern autumn equinox: solar longitude 90 to 180 "
            "degrees. Falls within the cooler aphelion half of the Mars year."
        ),
        "slot_values": {
            "ls_range": {"min": 90, "max": 180, "unit": "deg"},  # JPL-VERIFY [FIX-2]
        },
        "aliases": [
            "northern summer", "northern hemisphere summer",
            "northern warm season",   # [FIX-1] "perihelion season", "dust storm season" removed
        ],
    },
    {
        "id": "MartianDustStormSeason",   # [NEW]
        "label": "Martian Dust Storm Season",
        "category": "seasonal_temporal",
        "class_of": "Season",
        "description": (
            "The dusty half of the Mars year spanning southern spring and summer, "
            "solar longitude 180 to 360 degrees, when regional and global dust "
            "storms are most likely. Brackets perihelion."
        ),
        "slot_values": {
            "ls_range": {"min": 180, "max": 360, "unit": "deg"},  # JPL-VERIFY
        },
        "aliases": [
            "dust storm season", "dusty season", "storm season",
            "peak dust season",
        ],
    },
    {
        "id": "MarsAphelion",
        "label": "Mars Aphelion",
        "category": "seasonal_temporal",
        "class_of": "OrbitalEvent",
        "description": (
            "The point in Mars orbit farthest from the Sun, near solar longitude "
            "71 degrees (late northern spring). Associated with cooler "
            "temperatures and the aphelion cloud belt."
        ),
        "slot_values": {
            "ls_range": {"min": 50, "max": 90, "unit": "deg"},
        },
        "aliases": [
            "aphelion", "mars aphelion", "aphelion passage",
            "farthest from sun", "aphelion cloud belt season",
            "aphelion season",   # [FIX-1] moved here from southern summer
        ],
    },
    {
        "id": "MarsPerihelion",
        "label": "Mars Perihelion",
        "category": "seasonal_temporal",
        "class_of": "OrbitalEvent",
        "description": (
            "The point in Mars orbit closest to the Sun, near solar longitude "
            "251 degrees (southern spring). Associated with a warmer southern "
            "hemisphere and peak dust storm risk."
        ),
        "slot_values": {
            "ls_range": {"min": 230, "max": 270, "unit": "deg"},
        },
        "aliases": [
            "perihelion", "mars perihelion", "perihelion passage",
            "closest to sun",
            "perihelion season",  # [FIX-1] moved here from northern summer
        ],
    },
    {
        "id": "MarsYear",
        "label": "Mars Year",
        "category": "seasonal_temporal",
        "class_of": "TemporalInterval",
        "description": (
            "Mars Year numbering system, where MY1 began April 1955. Used to "
            "reference specific observing periods, e.g. MY34 refers to the 2018 "
            "global dust storm year."
        ),
        "slot_values": {},   # generic concept carries no dates; aliases do [FIX-5]
        "aliases": [
            "MY34", "MY33", "MY35", "mars year 34", "mars year 33",
            "2018 dust storm year", "2018 global storm",
        ],
        "alias_value_hints": {   # JPL-VERIFY all MY boundary dates [FIX-5]
            "MY33":                {"start": {"value": "2015-06-18"}, "stop": {"value": "2017-05-05"}},
            "MY34":                {"start": {"value": "2017-05-05"}, "stop": {"value": "2019-03-23"}},
            "MY35":                {"start": {"value": "2019-03-23"}, "stop": {"value": "2021-02-07"}},
            "mars year 33":        {"start": {"value": "2015-06-18"}, "stop": {"value": "2017-05-05"}},
            "mars year 34":        {"start": {"value": "2017-05-05"}, "stop": {"value": "2019-03-23"}},
            "2018 dust storm year": {"start": {"value": "2017-05-05"}, "stop": {"value": "2019-03-23"}},
            "2018 global storm":   {"start": {"value": "2017-05-05"}, "stop": {"value": "2019-03-23"}},
        },
    },

    # ── Geographic ─────────────────────────────────────────────────────────────
    # [FIX-4] All boxes are scalar per side. v1 wrote the full lon range on
    # both east and west edges, which is unextractable as a filter.
    {
        "id": "VallesMarineris",
        "label": "Valles Marineris",
        "category": "geographic",
        "class_of": "GeographicRegion",
        "description": (
            "A vast canyon system on Mars approximately 4000 km long and 7 km "
            "deep, located near the equator, roughly 5-20S latitude and "
            "270-330E longitude."
        ),
        "slot_values": {   # JPL-VERIFY box extents
            "west":  {"value": 270, "unit": "deg"},
            "east":  {"value": 330, "unit": "deg"},
            "south": {"value": -20, "unit": "deg"},
            "north": {"value": -5,  "unit": "deg"},
        },
        "aliases": [
            "valles marineris", "mariner valley", "vallis marineris",
            "martian grand canyon", "the great canyon of mars",
        ],
    },
    {
        "id": "TharsisRegion",
        "label": "Tharsis Plateau",
        "category": "geographic",
        "class_of": "GeographicRegion",
        "description": (
            "A large volcanic highland on Mars, centered near the equator around "
            "250E longitude, featuring the largest volcanoes in the solar system."
        ),
        "slot_values": {   # JPL-VERIFY box extents
            "west":  {"value": 220, "unit": "deg"},
            "east":  {"value": 280, "unit": "deg"},
            "south": {"value": -20, "unit": "deg"},
            "north": {"value": 30,  "unit": "deg"},
        },
        "aliases": [
            "tharsis", "tharsis plateau", "tharsis rise",
            "tharsis bulge", "tharsis volcanic region",
        ],
    },
    {
        "id": "MartianPolarRegion",
        "label": "Mars Polar Regions",
        "category": "geographic",
        "class_of": "GeographicRegion",
        "description": (
            "The north and south polar regions of Mars, defined as latitudes "
            "above 60N or below 60S. Contains polar ice caps and layered deposits."
        ),
        # Two disjoint boxes -> variants. traversal.py must OR these. [FIX-4]
        "slot_values": {
            "variants": [
                {   # north polar cap
                    "west":  {"value": 0,   "unit": "deg"},
                    "east":  {"value": 360, "unit": "deg"},
                    "south": {"value": 60,  "unit": "deg"},
                    "north": {"value": 90,  "unit": "deg"},
                },
                {   # south polar cap
                    "west":  {"value": 0,    "unit": "deg"},
                    "east":  {"value": 360,  "unit": "deg"},
                    "south": {"value": -90,  "unit": "deg"},
                    "north": {"value": -60,  "unit": "deg"},
                },
            ],
        },
        "aliases": [
            "polar region", "north pole", "south pole",
            "polar cap", "polar ice cap", "arctic region mars",
            "polar layered deposits", "NPLD", "SPLD",
            "north polar layered deposits", "south polar layered deposits",
        ],
        # Directional aliases resolve to the matching single cap. [FIX-5 pattern]
        "alias_value_hints": {
            "north pole": {"west": {"value": 0, "unit": "deg"}, "east": {"value": 360, "unit": "deg"},
                           "south": {"value": 60, "unit": "deg"}, "north": {"value": 90, "unit": "deg"}},
            "NPLD":       {"west": {"value": 0, "unit": "deg"}, "east": {"value": 360, "unit": "deg"},
                           "south": {"value": 60, "unit": "deg"}, "north": {"value": 90, "unit": "deg"}},
            "north polar layered deposits":
                          {"west": {"value": 0, "unit": "deg"}, "east": {"value": 360, "unit": "deg"},
                           "south": {"value": 60, "unit": "deg"}, "north": {"value": 90, "unit": "deg"}},
            "south pole": {"west": {"value": 0, "unit": "deg"}, "east": {"value": 360, "unit": "deg"},
                           "south": {"value": -90, "unit": "deg"}, "north": {"value": -60, "unit": "deg"}},
            "SPLD":       {"west": {"value": 0, "unit": "deg"}, "east": {"value": 360, "unit": "deg"},
                           "south": {"value": -90, "unit": "deg"}, "north": {"value": -60, "unit": "deg"}},
            "south polar layered deposits":
                          {"west": {"value": 0, "unit": "deg"}, "east": {"value": 360, "unit": "deg"},
                           "south": {"value": -90, "unit": "deg"}, "north": {"value": -60, "unit": "deg"}},
        },
    },
    {
        "id": "HellasBasin",
        "label": "Hellas Basin",
        "category": "geographic",
        "class_of": "GeographicRegion",
        "description": (
            "The largest impact basin on Mars, approximately 7 km below datum, "
            "located in the southern hemisphere southeast of Tharsis, roughly "
            "30-50S latitude and 50-90E longitude."
        ),
        "slot_values": {   # JPL-VERIFY — v1 had no west side and 40-70S [FIX-4]
            "west":  {"value": 50,  "unit": "deg"},
            "east":  {"value": 90,  "unit": "deg"},
            "south": {"value": -50, "unit": "deg"},
            "north": {"value": -30, "unit": "deg"},
        },
        "aliases": [
            "hellas", "hellas basin", "hellas planitia",
            "hellas impact basin", "hellas crater",
        ],
    },

    # ── Atmospheric ────────────────────────────────────────────────────────────
    {
        "id": "GlobalDustStorm",
        "label": "Global Dust Storm",
        "category": "atmospheric",
        "class_of": "AtmosphericEvent",
        "description": (
            "A planet-encircling dust event on Mars where dust optical depth "
            "(tau) exceeds approximately 3, obscuring the surface globally. "
            "Notable events: MY25 (2001), MY34 (2018)."
        ),
        "slot_values": {
            "opacity": {"min": 3.0, "unit": "tau"},
        },
        "occurs_during": ["MartianDustStormSeason"],   # [FIX-6]
        "aliases": [
            "global dust storm", "planet-encircling dust event",
            "global dust event", "dust storm MY34", "2018 dust storm",
            "MY25 dust storm", "2001 dust storm",
            "tau surge", "opacity surge", "high dust opacity",
        ],
        # Named events resolve to calendar windows via TemporalInterval-style
        # hints consumed only when those slots exist downstream; kept as
        # documentation for the dataset guide and future event concepts.
        # JPL-VERIFY event windows before promoting to per-event concepts.
    },
    {
        "id": "RegionalDustStorm",
        "label": "Regional Dust Storm",
        "category": "atmospheric",
        "class_of": "AtmosphericEvent",
        "description": (
            "A localized dust storm on Mars affecting a region but not "
            "planet-encircling. Dust optical depth elevated but below global "
            "storm threshold."
        ),
        "slot_values": {
            "opacity": {"min": 1.0, "max": 3.0, "unit": "tau"},
        },
        "occurs_during": ["MartianDustStormSeason"],
        "aliases": [
            "regional dust storm", "local dust storm", "dust event",
            "elevated dust", "dust lifting", "dust opacity event",
        ],
    },
    {
        "id": "WaterIceClouds",
        "label": "Water Ice Clouds",
        "category": "atmospheric",
        "class_of": "AtmosphericEvent",
        "description": (
            "Water ice clouds on Mars that form primarily during the aphelion "
            "cloud belt season (Ls 50-150 degrees) near the equator."
        ),
        # The Ls window is a seasonal_window of an atmospheric event — NOT a
        # season definition. This is the v1 'atmospheric corresponds to season'
        # conflation the JPL review flagged; the slot name now carries the
        # distinction explicitly.
        "slot_values": {
            "seasonal_window": {"min": 50, "max": 150, "unit": "deg"},
        },
        "occurs_during": ["MarsAphelion"],
        "aliases": [
            "water ice clouds", "ice clouds", "aphelion cloud belt",
            "ACB", "martian clouds", "equatorial clouds",
            "cloud belt", "cloud opacity",
        ],
    },

    # ── Instruments ────────────────────────────────────────────────────────────
    # [FIX-3] instrument_name now grounds to Instrument.name; mission_name
    # carries the host mission. v1 wrote "SHARAD" etc. into mission_name,
    # which yields filters that match nothing in the archive.
    {
        "id": "SHARADRadargram",
        "label": "SHARAD Radargrams",
        "category": "instrument_mode",
        "class_of": "Instrument",
        "description": (
            "Subsurface radar sounding data from the SHARAD instrument on MRO. "
            "Used to image subsurface layering, polar deposits, and ice. Data "
            "type is radargram."
        ),
        "slot_values": {
            "instrument_name": {"value": "SHARAD"},
            "mission_name":    {"value": "MARS RECONNAISSANCE ORBITER"},  # JPL-VERIFY exact archive string
        },
        "aliases": [
            "SHARAD", "sharad", "subsurface radar", "ice penetrating radar",
            "ground penetrating radar mars", "radar sounder",
            "subsurface sounder", "radargram", "radar cross section",
            "stratigraphic profiles from orbit",
        ],
    },
    {
        "id": "HiRISEImaging",
        "label": "HiRISE High-Resolution Imaging",
        "category": "instrument_mode",
        "class_of": "Instrument",
        "description": (
            "High Resolution Imaging Science Experiment on MRO, providing the "
            "highest resolution images of Mars surface at sub-meter scale "
            "(0.25-0.5 m/pixel)."
        ),
        "slot_values": {
            "instrument_name":  {"value": "HIRISE"},                        # JPL-VERIFY casing
            "mission_name":     {"value": "MARS RECONNAISSANCE ORBITER"},   # JPL-VERIFY
            "pixel_resolution": {"max": 0.5, "unit": "m/pixel"},
        },
        "aliases": [
            "HiRISE", "hirise", "high resolution imaging",
            "sub-meter imagery", "high res camera",
            "color imaging MRO", "25cm resolution",
        ],
    },
    {
        "id": "CRISMHyperspectral",
        "label": "CRISM Hyperspectral Imaging",
        "category": "instrument_mode",
        "class_of": "Instrument",
        "description": (
            "Compact Reconnaissance Imaging Spectrometer for Mars on MRO. Maps "
            "surface mineralogy. Modes include FRT (full resolution targeted) "
            "and HRL (half-resolution long)."
        ),
        "slot_values": {
            "instrument_name": {"value": "CRISM"},
            "mission_name":    {"value": "MARS RECONNAISSANCE ORBITER"},  # JPL-VERIFY
        },
        "aliases": [
            "CRISM", "crism", "hyperspectral", "spectrometer MRO",
            "mineralogy mapper", "FRT", "HRL", "targeted spectrometer",
            "surface composition mapping", "spectral imaging",
        ],
    },
    {
        "id": "CTXCamera",
        "label": "CTX Context Camera",
        "category": "instrument_mode",
        "class_of": "Instrument",
        "description": (
            "Context Camera on MRO providing wide-field grayscale images at "
            "approximately 6 m/pixel resolution over a 30 km swath."
        ),
        "slot_values": {
            "instrument_name":  {"value": "CTX"},
            "mission_name":     {"value": "MARS RECONNAISSANCE ORBITER"},  # JPL-VERIFY
            "pixel_resolution": {"min": 5.0, "max": 7.0, "unit": "m/pixel"},
        },
        "aliases": [
            "CTX", "ctx", "context camera", "context imager",
            "6 meter resolution", "wide angle camera MRO",
            "30km swath", "context imagery",
        ],
    },

    # ── Mission Phases ─────────────────────────────────────────────────────────
    # mission_phase_name is free text and varies per mission — the phase_name
    # values below MUST be validated against distinct values actually present
    # in archive labels (JPL review: phases are typically strings like
    # "in flight", "extended", "end of life"). The validation UI should
    # surface archive value counts next to each hint.
    {
        "id": "MROAerobraking",
        "label": "MRO Aerobraking Phase",
        "category": "mission_phase",
        "class_of": "MissionPhase",
        "description": (
            "The aerobraking phase of the Mars Reconnaissance Orbiter mission in "
            "2006, during which limited science data was collected due to "
            "non-science orbit geometry."
        ),
        "slot_values": {
            "phase_name":   {"value": "AEROBRAKING"},                    # JPL-VERIFY archive string
            "mission_name": {"value": "MARS RECONNAISSANCE ORBITER"},    # JPL-VERIFY
            "start":        {"value": "2006-03-01"},
            "stop":         {"value": "2006-08-30"},
        },
        "aliases": [
            "aerobraking", "aerobraking phase", "MRO aerobraking",
            "2006 aerobraking", "orbital insertion phase",
            "pre-science phase MRO",
        ],
    },
    {
        "id": "CuriositySciencePhase",
        "label": "Curiosity Rover Science Phase",
        "category": "mission_phase",
        "class_of": "MissionPhase",
        "description": (
            "The primary science operations phase of the MSL Curiosity rover, "
            "beginning after landing in August 2012 and continuing through the "
            "mission."
        ),
        "slot_values": {
            "phase_name":   {"value": "PRIMARY SURFACE MISSION"},        # JPL-VERIFY archive string
            "mission_name": {"value": "MARS SCIENCE LABORATORY"},        # JPL-VERIFY
        },
        "aliases": [
            "curiosity science", "MSL science phase", "curiosity rover data",
            "gale crater science", "post-landing science",
            "sol-based observations", "curiosity surface operations",
        ],
    },
]

# ── Validation ─────────────────────────────────────────────────────────────────

def _iter_slot_dicts(slot_values):
    """Yield flat {slot: value} dicts, expanding a variants wrapper."""
    if "variants" in slot_values:
        for v in slot_values["variants"]:
            yield v
    else:
        yield slot_values


def validate(schema_nodes, classes, concepts):
    """Fail fast on structural errors; warn on suspicious values."""
    errors, warnings = [], []
    class_by_id = {c["id"]: c for c in classes}

    for cls in classes:
        for slot, leaf_key in cls["slots"].items():
            if leaf_key not in schema_nodes:
                errors.append(f"class {cls['id']}: slot '{slot}' -> missing leaf {leaf_key}")

    for c in concepts:
        cls = class_by_id.get(c["class_of"])
        if cls is None:
            errors.append(f"concept {c['id']}: unknown class '{c['class_of']}'")
            continue
        for sv in _iter_slot_dicts(c["slot_values"]):
            for slot in sv:
                if slot not in cls["slots"]:
                    errors.append(f"concept {c['id']}: value for unknown slot '{slot}' "
                                  f"(class {cls['id']} has {sorted(cls['slots'])})")
        for alias, sv in c.get("alias_value_hints", {}).items():
            if alias not in c["aliases"]:
                errors.append(f"concept {c['id']}: alias_value_hints for non-alias '{alias}'")
            for slot in sv:
                if slot not in cls["slots"]:
                    errors.append(f"concept {c['id']}/alias '{alias}': unknown slot '{slot}'")
        for target in c.get("occurs_during", []):
            if not any(other["id"] == target for other in concepts):
                errors.append(f"concept {c['id']}: occurs_during -> unknown concept '{target}'")

        # Geographic sanity: west < east, south < north (warn only — regions
        # crossing the prime meridian legitimately violate west < east).
        if c["class_of"] == "GeographicRegion":
            for sv in _iter_slot_dicts(c["slot_values"]):
                w = sv.get("west", {}).get("value")
                e = sv.get("east", {}).get("value")
                s = sv.get("south", {}).get("value")
                n = sv.get("north", {}).get("value")
                if w is not None and e is not None and w >= e:
                    warnings.append(f"{c['id']}: west ({w}) >= east ({e}) — meridian crossing?")
                if s is not None and n is not None and s >= n:
                    errors.append(f"{c['id']}: south ({s}) >= north ({n})")

    return errors, warnings

# ── Graph construction ─────────────────────────────────────────────────────────

def build_graph(schema_nodes, classes, concepts):
    G = nx.DiGraph()
    G.graph["version"] = 2
    G.graph["layers"] = ["alias", "concept", "class", "schema_leaf"]
    G.graph["edge_priorities"] = EDGE_PRIORITIES

    # 1. Schema leaf layer (stamp node_type; don't trust the input file)
    for key, node in schema_nodes.items():
        G.add_node(f"schema:{key}", **{**node, "node_type": "schema_leaf"})

    # 2. Class layer + grounded_in edges (field bindings, no values)
    for cls in classes:
        class_id = f"class:{cls['id']}"
        G.add_node(class_id,
                   node_type="class",
                   label=cls["label"],
                   description=cls["description"])
        for slot, leaf_key in cls["slots"].items():
            G.add_edge(class_id, f"schema:{leaf_key}",
                       edge_type="grounded_in",
                       slot=slot)

    # 3. Concept layer + instance_of edges (values, keyed by slot)
    for c in concepts:
        concept_id = f"concept:{c['id']}"
        G.add_node(concept_id,
                   node_type="concept",
                   label=c["label"],
                   category=c["category"],
                   class_of=c["class_of"],
                   description=c["description"])
        G.add_edge(concept_id, f"class:{c['class_of']}",
                   edge_type="instance_of",
                   slot_values=json.dumps(c["slot_values"]))

    # 4. Concept -> concept cross-links
    for c in concepts:
        for target in c.get("occurs_during", []):
            G.add_edge(f"concept:{c['id']}", f"concept:{target}",
                       edge_type="occurs_during")

    # 5. Alias layer (with optional per-alias value overrides)
    for c in concepts:
        concept_id = f"concept:{c['id']}"
        hints = c.get("alias_value_hints", {})
        for alias in c["aliases"]:
            alias_id = f"alias:{alias.lower().replace(' ', '_')}"
            G.add_node(alias_id,
                       node_type="alias",
                       surface_form=alias,
                       concept=c["id"])
            edge_attrs = {"edge_type": "is_alias_of"}
            if alias in hints:
                edge_attrs["value_hint"] = json.dumps(hints[alias])
            G.add_edge(alias_id, concept_id, **edge_attrs)

    return G

# ── Reporting ──────────────────────────────────────────────────────────────────

def print_graph_summary(G):
    node_types = defaultdict(int)
    edge_types = defaultdict(int)
    for _, data in G.nodes(data=True):
        node_types[data.get("node_type", "unknown")] += 1
    for _, _, data in G.edges(data=True):
        edge_types[data.get("edge_type", "unknown")] += 1

    print("\n── Graph Summary ────────────────────────────────────────")
    print(f"  Total nodes : {G.number_of_nodes()}")
    print(f"  Total edges : {G.number_of_edges()}")
    print("\n  Node types:")
    for nt, count in sorted(node_types.items()):
        print(f"    {nt:20s} {count}")
    print("\n  Edge types:")
    for et, count in sorted(edge_types.items()):
        print(f"    {et:25s} {count}")


def write_text_summary(G, classes, concepts, path):
    lines = ["LUMIN Knowledge Graph Summary (v2 — four-layer)", "=" * 60, ""]

    node_types = defaultdict(int)
    for _, d in G.nodes(data=True):
        node_types[d.get("node_type")] += 1

    lines.append(f"Total nodes : {G.number_of_nodes()}")
    lines.append(f"Total edges : {G.number_of_edges()}")
    for nt in ("schema_leaf", "class", "concept", "alias"):
        lines.append(f"  {nt:12s}: {node_types[nt]}")
    lines.append("")

    lines.append("── Class layer (domain-general) " + "─" * 28)
    for cls in classes:
        lines.append(f"  {cls['id']}")
        for slot, leaf in cls["slots"].items():
            lines.append(f"    {slot:18s} -> {leaf}")
    lines.append("")

    for c in concepts:
        lines.append(f"── {c['label']} [{c['category']}]  instance_of {c['class_of']}")
        lines.append(f"   {c['description'][:120]}")
        lines.append(f"   Slot values: {json.dumps(c['slot_values'])}")
        for target in c.get("occurs_during", []):
            lines.append(f"   occurs_during -> {target}")
        lines.append(f"   Aliases ({len(c['aliases'])}):")
        lines.append(f"     {', '.join(c['aliases'])}")
        if c.get("alias_value_hints"):
            lines.append(f"   Alias value overrides: {sorted(c['alias_value_hints'])}")
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved {path}")

# ── Demo traversal ─────────────────────────────────────────────────────────────

def demo_traverse(G, alias_surface):
    """Walk alias -> concept -> class -> leaves and compose the filter,
    demonstrating what traversal.py should do under the v2 layout."""
    alias_id = f"alias:{alias_surface.lower().replace(' ', '_')}"
    if alias_id not in G:
        print(f"  (alias '{alias_surface}' not in graph)")
        return
    print(f"  Query surface form: '{alias_surface}'")

    (concept_id,) = list(G.successors(alias_id))
    alias_hint = G.edges[alias_id, concept_id].get("value_hint")
    print(f"    [alias] --is_alias_of--> [concept] {G.nodes[concept_id]['label']}")

    class_id = None
    for succ in G.successors(concept_id):
        if G.edges[concept_id, succ].get("edge_type") == "instance_of":
            class_id = succ
    slot_values = json.loads(G.edges[concept_id, class_id]["slot_values"] or "{}")
    if alias_hint:  # alias-level values override concept-level values
        slot_values = json.loads(alias_hint)
    print(f"    [concept] --instance_of--> [class] {G.nodes[class_id]['label']}")

    print(f"    [class] --grounded_in--> leaves; composed filter:")
    for sv in _iter_slot_dicts(slot_values):
        for succ in G.successors(class_id):
            slot = G.edges[class_id, succ].get("slot")
            if slot in sv:
                leaf_name = G.nodes[succ].get("name")
                print(f"      {leaf_name:28s} = {json.dumps(sv[slot])}")

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Building LUMIN Knowledge Graph (v2 — four-layer)\n")

    if len(sys.argv) > 1:
        nodes_path = Path(sys.argv[1])
    else:
        nodes_path = Path(__file__).parent.parent / "output" / "schema_nodes_final.json"

    with open(nodes_path) as f:
        raw = json.load(f)
    schema_nodes = {f"{n['class']}.{n['name']}": n for n in raw}
    print(f"Loaded {len(schema_nodes)} schema leaf nodes")

    errors, warnings = validate(schema_nodes, CLASSES, CONCEPTS)
    for w in warnings:
        print(f"⚠  {w}")
    if errors:
        for e in errors:
            print(f"✗  {e}")
        sys.exit(f"\n{len(errors)} validation error(s) — graph not built.")

    G = build_graph(schema_nodes, CLASSES, CONCEPTS)
    print_graph_summary(G)

    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)

    graph_data = nx.node_link_data(G, edges="edges")
    with open(output_dir / "lumin_kg.json", "w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=2)
    print(f"\nSaved {output_dir / 'lumin_kg.json'}")

    write_text_summary(G, CLASSES, CONCEPTS, output_dir / "lumin_kg_summary.txt")

    print("\n── Demo: alias -> concept -> class -> composed filter ───")
    for surface in ("southern summer", "MY34", "hellas", "sharad", "north pole"):
        demo_traverse(G, surface)
        print()


if __name__ == "__main__":
    main()