"""
build_kg.py
-----------
Builds the LUMIN knowledge graph by layering concept nodes and alias nodes
on top of the parsed PDS4 schema leaf nodes.

Graph structure:
    [alias nodes]   → is_alias_of   → [concept nodes]
    [concept nodes] → measured_by   → [schema leaves]
    [concept nodes] → corresponds_to→ [schema leaves]  (direct value mapping)
    [concept nodes] → instrument_of → [schema leaves]  (instrument queries)

Input:  schema_nodes_final.json  (deduped, from filter_schema.py + dedup step)
Output: lumin_kg.json            (full graph in NetworkX node-link format)
        lumin_kg_summary.txt     (human-readable graph summary)

Usage:
    pip install networkx
    python build_kg.py
"""

import json
from pathlib import Path
import networkx as nx
from collections import defaultdict

# ── Load schema leaf nodes ─────────────────────────────────────────────────────

def load_schema_nodes(path="schema_nodes_final.json"):
    with open(path) as f:
        nodes = json.load(f)
    print(f"Loaded {len(nodes)} schema leaf nodes")
    return {f"{n['class']}.{n['name']}": n for n in nodes}

# ── Concept definitions ────────────────────────────────────────────────────────
# Each concept has:
#   id          : unique node identifier
#   label       : human-readable name
#   category    : concept category (matches dataset guide)
#   description : brief definition (used for embedding at query time)
#   maps_to     : list of (schema_leaf_key, edge_type, value_hint) tuples
#                 schema_leaf_key = "ClassName.attr_name" from schema nodes
#                 edge_type = measured_by | corresponds_to | instrument_of
#                 value_hint = the actual filter value/range (None if open)
#   aliases     : list of NL surface strings (T1 entry points)
#                 Add T2/T3 aliases here as dataset generation progresses

CONCEPTS = [

    # ── Seasonal / Temporal ────────────────────────────────────────────────────
    {
        "id": "MartianSouthernSummer",
        "label": "Martian Southern Summer",
        "category": "seasonal_temporal",
        "description": "The period of Mars southern hemisphere summer, defined by solar longitude between 180 and 360 degrees. Also called aphelion season.",
        "maps_to": [
            ("Time_Coordinates.solar_longitude", "corresponds_to", {"min": 180, "max": 360, "unit": "deg"}),
        ],
        "aliases": [
            "southern summer", "southern hemisphere summer", "aphelion season",
            "martian summer", "southern warm season",
        ],
    },
    {
        "id": "MartianNorthernSummer",
        "label": "Martian Northern Summer",
        "category": "seasonal_temporal",
        "description": "The period of Mars northern hemisphere summer, defined by solar longitude between 0 and 180 degrees. Associated with perihelion season and elevated dust storm risk.",
        "maps_to": [
            ("Time_Coordinates.solar_longitude", "corresponds_to", {"min": 0, "max": 180, "unit": "deg"}),
        ],
        "aliases": [
            "northern summer", "northern hemisphere summer", "perihelion season",
            "dust storm season", "northern warm season",
        ],
    },
    {
        "id": "MarsAphelion",
        "label": "Mars Aphelion",
        "category": "seasonal_temporal",
        "description": "The point in Mars orbit farthest from the Sun, occurring around solar longitude 70 degrees. Associated with cooler temperatures and the aphelion cloud belt.",
        "maps_to": [
            ("Time_Coordinates.solar_longitude", "corresponds_to", {"min": 50, "max": 90, "unit": "deg"}),
        ],
        "aliases": [
            "aphelion", "mars aphelion", "aphelion passage",
            "farthest from sun", "aphelion cloud belt season",
        ],
    },
    {
        "id": "MarsPerihelion",
        "label": "Mars Perihelion",
        "category": "seasonal_temporal",
        "description": "The point in Mars orbit closest to the Sun, occurring around solar longitude 250 degrees. Associated with warmer southern hemisphere and peak dust storm risk.",
        "maps_to": [
            ("Time_Coordinates.solar_longitude", "corresponds_to", {"min": 230, "max": 270, "unit": "deg"}),
        ],
        "aliases": [
            "perihelion", "mars perihelion", "perihelion passage",
            "closest to sun", "peak dust season",
        ],
    },
    {
        "id": "MarsYear",
        "label": "Mars Year",
        "category": "seasonal_temporal",
        "description": "Mars Year numbering system, where MY1 began April 1955. Used to reference specific observing periods, e.g. MY34 refers to the 2018 global dust storm year.",
        "maps_to": [
            ("Time_Coordinates.start_date_time", "corresponds_to", None),
            ("Time_Coordinates.stop_date_time", "corresponds_to", None),
        ],
        "aliases": [
            "MY34", "MY33", "MY35", "mars year 34", "mars year 33",
            "2018 dust storm year", "2018 global storm",
        ],
    },

    # ── Geographic ─────────────────────────────────────────────────────────────
    {
        "id": "VallesMarineris",
        "label": "Valles Marineris",
        "category": "geographic",
        "description": "A vast canyon system on Mars approximately 4000 km long and 7 km deep, located near the equator between 10S-20S latitude and 270-330E longitude.",
        "maps_to": [
            ("Bounding_Coordinates.south_bounding_coordinate", "corresponds_to", {"min": -20, "max": -5, "unit": "deg"}),
            ("Bounding_Coordinates.north_bounding_coordinate", "corresponds_to", {"min": -5, "max": 5, "unit": "deg"}),
            ("Bounding_Coordinates.east_bounding_coordinate", "corresponds_to", {"min": 270, "max": 330, "unit": "deg"}),
            ("Bounding_Coordinates.west_bounding_coordinate", "corresponds_to", {"min": 270, "max": 330, "unit": "deg"}),
        ],
        "aliases": [
            "valles marineris", "mariner valley", "vallis marineris",
            "martian grand canyon", "the great canyon of mars",
        ],
    },
    {
        "id": "TharsisRegion",
        "label": "Tharsis Plateau",
        "category": "geographic",
        "description": "A large volcanic highland on Mars, centered near the equator around 250E longitude, featuring the largest volcanoes in the solar system.",
        "maps_to": [
            ("Bounding_Coordinates.south_bounding_coordinate", "corresponds_to", {"min": -20, "max": 0, "unit": "deg"}),
            ("Bounding_Coordinates.north_bounding_coordinate", "corresponds_to", {"min": 0, "max": 30, "unit": "deg"}),
            ("Bounding_Coordinates.east_bounding_coordinate", "corresponds_to", {"min": 220, "max": 280, "unit": "deg"}),
            ("Bounding_Coordinates.west_bounding_coordinate", "corresponds_to", {"min": 220, "max": 280, "unit": "deg"}),
        ],
        "aliases": [
            "tharsis", "tharsis plateau", "tharsis rise",
            "tharsis bulge", "tharsis volcanic region",
        ],
    },
    {
        "id": "MartianPolarRegion",
        "label": "Mars Polar Regions",
        "category": "geographic",
        "description": "The north and south polar regions of Mars, defined as latitudes above 60N or below 60S. Contains polar ice caps and layered deposits.",
        "maps_to": [
            ("Bounding_Coordinates.north_bounding_coordinate", "corresponds_to", {"min": 60, "max": 90, "unit": "deg"}),
            ("Bounding_Coordinates.south_bounding_coordinate", "corresponds_to", {"min": -90, "max": -60, "unit": "deg"}),
        ],
        "aliases": [
            "polar region", "north pole", "south pole",
            "polar cap", "polar ice cap", "arctic region mars",
            "polar layered deposits", "NPLD", "SPLD",
            "north polar layered deposits", "south polar layered deposits",
        ],
    },
    {
        "id": "HellasBasin",
        "label": "Hellas Basin",
        "category": "geographic",
        "description": "The largest impact crater on Mars, approximately 7 km below datum, located in the southern hemisphere around 40-70S latitude.",
        "maps_to": [
            ("Bounding_Coordinates.south_bounding_coordinate", "corresponds_to", {"min": -70, "max": -40, "unit": "deg"}),
            ("Bounding_Coordinates.north_bounding_coordinate", "corresponds_to", {"min": -40, "max": -30, "unit": "deg"}),
            ("Bounding_Coordinates.east_bounding_coordinate", "corresponds_to", {"min": 60, "max": 110, "unit": "deg"}),
        ],
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
        "description": "A planet-encircling dust event on Mars where dust optical depth (tau) exceeds approximately 3, obscuring the surface globally. Notable events: MY25 (2001), MY34 (2018).",
        "maps_to": [
            ("Radiometric_Correction.atmospheric_opacity", "measured_by", {"min": 3.0, "unit": "tau"}),
            ("Time_Coordinates.start_date_time", "corresponds_to", None),
        ],
        "aliases": [
            "global dust storm", "planet-encircling dust event",
            "global dust event", "dust storm MY34", "2018 dust storm",
            "MY25 dust storm", "2001 dust storm",
            "tau surge", "opacity surge", "high dust opacity",
        ],
    },
    {
        "id": "RegionalDustStorm",
        "label": "Regional Dust Storm",
        "category": "atmospheric",
        "description": "A localized dust storm on Mars affecting a region but not planet-encircling. Dust optical depth elevated but below global storm threshold.",
        "maps_to": [
            ("Radiometric_Correction.atmospheric_opacity", "measured_by", {"min": 1.0, "max": 3.0, "unit": "tau"}),
        ],
        "aliases": [
            "regional dust storm", "local dust storm", "dust event",
            "elevated dust", "dust lifting", "dust opacity event",
        ],
    },
    {
        "id": "WaterIceClouds",
        "label": "Water Ice Clouds",
        "category": "atmospheric",
        "description": "Water ice clouds on Mars that form primarily during the aphelion cloud belt season (Ls 50-150 degrees) near the equator.",
        "maps_to": [
            ("Time_Coordinates.solar_longitude", "corresponds_to", {"min": 50, "max": 150, "unit": "deg"}),
        ],
        "aliases": [
            "water ice clouds", "ice clouds", "aphelion cloud belt",
            "ACB", "martian clouds", "equatorial clouds",
            "cloud belt", "cloud opacity",
        ],
    },

    # ── Instrument Modes ───────────────────────────────────────────────────────
    {
        "id": "SHARADRadargram",
        "label": "SHARAD Radargrams",
        "category": "instrument_mode",
        "description": "Subsurface radar sounding data from the SHARAD instrument on MRO. Used to image subsurface layering, polar deposits, and ice. Data type is radargram.",
        "maps_to": [
            ("Mission_PDS3.mission_name", "instrument_of", {"value": "SHARAD"}),
        ],
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
        "description": "High Resolution Imaging Science Experiment on MRO, providing the highest resolution images of Mars surface at sub-meter scale (0.25-0.5 m/pixel).",
        "maps_to": [
            ("Mission_PDS3.mission_name", "instrument_of", {"value": "HiRISE"}),
            ("Coordinate_Representation.pixel_resolution_x", "corresponds_to", {"max": 0.5, "unit": "m/pixel"}),
        ],
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
        "description": "Compact Reconnaissance Imaging Spectrometer for Mars on MRO. Maps surface mineralogy. Modes include FRT (full resolution targeted) and HRL (half-resolution long).",
        "maps_to": [
            ("Mission_PDS3.mission_name", "instrument_of", {"value": "CRISM"}),
        ],
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
        "description": "Context Camera on MRO providing wide-field grayscale images at approximately 6 m/pixel resolution over a 30 km swath.",
        "maps_to": [
            ("Mission_PDS3.mission_name", "instrument_of", {"value": "CTX"}),
            ("Coordinate_Representation.pixel_resolution_x", "corresponds_to", {"min": 5.0, "max": 7.0, "unit": "m/pixel"}),
        ],
        "aliases": [
            "CTX", "ctx", "context camera", "context imager",
            "6 meter resolution", "wide angle camera MRO",
            "30km swath", "context imagery",
        ],
    },

    # ── Mission Phases ─────────────────────────────────────────────────────────
    {
        "id": "MROAerobraking",
        "label": "MRO Aerobraking Phase",
        "category": "mission_phase",
        "description": "The aerobraking phase of the Mars Reconnaissance Orbiter mission in 2006, during which limited science data was collected due to non-science orbit geometry.",
        "maps_to": [
            ("Mission_Phase.mission_phase_name", "corresponds_to", {"value": "Aerobraking"}),
            ("Time_Coordinates.start_date_time", "corresponds_to", {"value": "2006-03-01"}),
            ("Time_Coordinates.stop_date_time", "corresponds_to", {"value": "2006-08-30"}),
        ],
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
        "description": "The primary science operations phase of the MSL Curiosity rover, beginning after landing in August 2012 and continuing through the mission.",
        "maps_to": [
            ("Mission_Phase.mission_phase_name", "corresponds_to", {"value": "Science Phase"}),
            ("Mission_PDS3.mission_name", "corresponds_to", {"value": "MSL"}),
        ],
        "aliases": [
            "curiosity science", "MSL science phase", "curiosity rover data",
            "gale crater science", "post-landing science",
            "sol-based observations", "curiosity surface operations",
        ],
    },
]

# ── Graph construction ─────────────────────────────────────────────────────────

def build_graph(schema_nodes, concepts):
    G = nx.DiGraph()

    # 1. Add schema leaf nodes
    for key, node in schema_nodes.items():
        node_id = f"schema:{key}"
        G.add_node(node_id, **node)

    # 2. Add concept nodes + edges to schema leaves
    missing_leaves = []
    for c in concepts:
        concept_id = f"concept:{c['id']}"
        G.add_node(concept_id,
                   node_type="concept",
                   label=c["label"],
                   category=c["category"],
                   description=c["description"])

        for leaf_key, edge_type, value_hint in c["maps_to"]:
            schema_id = f"schema:{leaf_key}"
            if schema_id in G:
                G.add_edge(concept_id, schema_id,
                           edge_type=edge_type,
                           value_hint=json.dumps(value_hint) if value_hint else None)
            else:
                missing_leaves.append((c["id"], leaf_key))

    # 3. Add alias nodes + edges to concepts
    for c in concepts:
        concept_id = f"concept:{c['id']}"
        for alias in c["aliases"]:
            alias_id = f"alias:{alias.lower().replace(' ','_')}"
            G.add_node(alias_id,
                       node_type="alias",
                       surface_form=alias,
                       concept=c["id"])
            G.add_edge(alias_id, concept_id, edge_type="is_alias_of")

    return G, missing_leaves

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

def write_text_summary(G, concepts, path):
    lines = ["LUMIN Knowledge Graph Summary", "=" * 60, ""]

    node_types = defaultdict(int)
    for _, d in G.nodes(data=True): node_types[d.get("node_type")] += 1

    lines.append(f"Total nodes : {G.number_of_nodes()}")
    lines.append(f"Total edges : {G.number_of_edges()}")
    lines.append(f"  schema_leaf : {node_types['schema_leaf']}")
    lines.append(f"  concept     : {node_types['concept']}")
    lines.append(f"  alias       : {node_types['alias']}")
    lines.append("")

    for c in concepts:
        lines.append(f"── {c['label']} [{c['category']}]")
        lines.append(f"   {c['description'][:120]}")
        lines.append(f"   Schema leaves:")
        for leaf_key, edge_type, value_hint in c["maps_to"]:
            hint_str = f"  → {value_hint}" if value_hint else ""
            lines.append(f"     {edge_type:20s} → {leaf_key}{hint_str}")
        lines.append(f"   Aliases ({len(c['aliases'])}):")
        lines.append(f"     {', '.join(c['aliases'])}")
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved {path}")

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Building LUMIN Knowledge Graph\n")

    nodes_path = Path(__file__).parent.parent / "output" / "schema_nodes_final.json"
    schema_nodes = load_schema_nodes(nodes_path)

    G, missing = build_graph(schema_nodes, CONCEPTS)

    print_graph_summary(G)

    if missing:
        print(f"\n⚠  Missing schema leaves (add to schema_nodes_final.json or fix key):")
        for concept_id, leaf_key in missing:
            print(f"   [{concept_id}] → {leaf_key}")

    # Save graph
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)

    graph_data = nx.node_link_data(G)
    with open(output_dir / "lumin_kg.json", "w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=2)
    print(f"\nSaved output/lumin_kg.json")

    write_text_summary(G, CONCEPTS, output_dir / "lumin_kg_summary.txt")

    # Quick traversal demo
    print("\n── Demo: traverse from alias → schema leaf ──────────────")
    demo_alias = "alias:southern_summer"
    if demo_alias in G:
        path = list(nx.dfs_edges(G, source=demo_alias))
        print(f"  Starting from: 'southern summer'")
        for u, v in path:
            u_type = G.nodes[u].get("node_type")
            v_type = G.nodes[v].get("node_type")
            edge_type = G.edges[u, v].get("edge_type")
            v_label = G.nodes[v].get("label") or G.nodes[v].get("name") or v
            print(f"    [{u_type}] → {edge_type} → [{v_type}] {v_label}")
            hint = G.edges[u, v].get("value_hint")
            if hint:
                print(f"      value hint: {hint}")

if __name__ == "__main__":
    main()
