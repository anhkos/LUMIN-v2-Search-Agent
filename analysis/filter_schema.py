"""
filter_schema.py
----------------
Filters schema_nodes.json down to query-relevant nodes only —
strips out internal data dictionary machinery and keeps science/query fields.

Input:  schema_nodes.json  (from parse_ldd.py)
Output: schema_nodes_clean.json   — filtered flat list
        schema_nodes_by_concept.json — grouped by concept category
        schema_nodes_summary.txt  — human-readable overview for the team

Usage:
    python filter_schema.py
"""

import json
from collections import defaultdict
from pathlib import Path

# ── Classes to exclude (internal DD machinery, not query fields) ───────────────
EXCLUDE_CLASSES = {
    "DD_Attribute_Full", "DD_Class_Full", "DD_Value_Domain_Full",
    "DD_Association", "DD_Association_External", "DD_Associate_External_Class",
    "DD_Value_Domain", "Schematron_Rule", "Schematron_Assert",
    "Volume_PDS3", "Data_Set_PDS3", "PDS_Affiliate", "PDS3_Catalog",
    "Software_Source", "Terminological_Entry_SKOS",
    "Resource", "Service", "Tracking_Detail",
    "DD_Permissible_Value_Full",
}

# ── Concept category → keywords to match in class name or attr name ────────────
CONCEPT_CATEGORIES = {
    "seasonal_temporal": [
        "solar_longitude", "solar_lon", "start_date", "stop_date",
        "start_time", "stop_time", "time_coordinate", "mission_phase",
        "season", "sampling_parameter"
    ],
    "geographic": [
        "bounding_coordinate", "latitude", "longitude", "north_bound",
        "south_bound", "east_bound", "west_bound", "spatial_domain",
        "center_latitude", "center_longitude", "geographic"
    ],
    "atmospheric": [
        "atmospheric_opacity", "opacity", "tau", "atmosphere",
        "dust", "inband_fsun", "radiometric", "zenith_scaling"
    ],
    "instrument_mode": [
        "filter", "optical_filter", "detector", "exposure", "gain",
        "bandwidth", "instrument", "companding", "shutter",
        "pixel_resolution", "pixel_scale", "focus", "zoom",
        "band_bin", "compression"
    ],
    "mission_phase": [
        "mission_phase", "mission_name", "mission_start",
        "mission_stop", "investigation", "product_type"
    ],
    "image_quality": [
        "data_quality", "saturated", "missing_pixel", "corrupted",
        "bad_pixel", "hot_pixel", "nonlinear", "dark_current"
    ],
    "geometry_projection": [
        "map_projection", "coordinate_system", "geodetic_model",
        "reference_meridian", "standard_parallel", "scale_factor",
        "latitude_type", "longitude_direction"
    ],
}

def matches_category(node, keywords):
    name = node["name"].lower()
    cls  = node["class"].lower()
    desc = node.get("description", "").lower()
    return any(kw in name or kw in cls or kw in desc for kw in keywords)

def load_and_filter(path):
    with open(path) as f:
        nodes = json.load(f)

    print(f"Loaded {len(nodes)} nodes")

    # Step 1: remove excluded classes
    filtered = [n for n in nodes if n["class"] not in EXCLUDE_CLASSES]
    print(f"After removing internal classes: {len(filtered)}")

    # Step 2: remove obvious schema-machinery by name patterns
    def is_machinery(n):
        name = n["name"].lower()
        machinery_terms = [
            "field_format", "field_number", "group_number", "group_length",
            "field_length", "field_delimiter", "record_length", "record_delimiter",
            "maximum_occurrences", "minimum_occurrences",
            "given_name", "family_name", "team_name", "affiliation",
            "citation_text", "reference_text", "archive_status",
            "registration_authority", "stewardship"
        ]
        return any(t in name for t in machinery_terms)

    filtered = [n for n in filtered if not is_machinery(n)]
    print(f"After removing schema machinery attrs: {len(filtered)}")

    return filtered

def categorize(nodes):
    """Assign each node to one or more concept categories."""
    categorized = defaultdict(list)
    uncategorized = []

    for node in nodes:
        found = False
        for category, keywords in CONCEPT_CATEGORIES.items():
            if matches_category(node, keywords):
                categorized[category].append(node)
                found = True
        if not found:
            uncategorized.append(node)

    return dict(categorized), uncategorized

def write_summary(categorized, uncategorized, path):
    lines = []
    lines.append("PDS4 Schema Nodes — Clean Summary for LUMIN KG")
    lines.append("=" * 60)
    lines.append("")

    total = sum(len(v) for v in categorized.values()) + len(uncategorized)
    lines.append(f"Total query-relevant nodes: {total}")
    lines.append("")

    for cat, nodes in sorted(categorized.items()):
        lines.append(f"── {cat.upper().replace('_',' ')} ({len(nodes)} nodes)")
        for n in nodes:
            pv_str = ""
            if n.get("permissible_values"):
                vals = [v["value"] for v in n["permissible_values"][:4]]
                pv_str = f"  → values: {vals}"
            range_str = ""
            if n.get("min_value") and n.get("unit"):
                range_str = f"  → range: [{n['min_value']}, {n['max_value']}] {n['unit']}"
            lines.append(f"  [{n['ldd']}] {n['class']}.{n['name']}")
            lines.append(f"    {n['description'][:100]}")
            if pv_str:  lines.append(f"   {pv_str}")
            if range_str: lines.append(f"   {range_str}")
        lines.append("")

    if uncategorized:
        lines.append(f"── UNCATEGORIZED ({len(uncategorized)} nodes) — review manually")
        for n in uncategorized[:20]:
            lines.append(f"  [{n['ldd']}] {n['class']}.{n['name']}")
        if len(uncategorized) > 20:
            lines.append(f"  ... and {len(uncategorized)-20} more")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"✓ Saved {path}")

def main():
    print("Filtering schema nodes...\n")


    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)

    schema_path = output_dir / "schema_nodes.json"
    nodes = load_and_filter(schema_path)

    categorized, uncategorized = categorize(nodes)

    print(f"\nCategorized nodes by concept area:")
    for cat, ns in sorted(categorized.items()):
        print(f"  {cat:30s} {len(ns):4d} nodes")
    print(f"  {'uncategorized':30s} {len(uncategorized):4d} nodes")

    # save clean flat list (all filtered nodes)
    all_clean = []
    seen_ids = set()
    for ns in categorized.values():
        for n in ns:
            if n["identifier"] not in seen_ids:
                all_clean.append(n)
                seen_ids.add(n["identifier"])
    for n in uncategorized:
        if n["identifier"] not in seen_ids:
            all_clean.append(n)
            seen_ids.add(n["identifier"])

    with open(output_dir / "schema_nodes_clean.json", "w", encoding="utf-8") as f:
        json.dump(all_clean, f, indent=2)
    print(f"\n✓ Saved output/schema_nodes_clean.json ({len(all_clean)} nodes)")

    # save by-concept grouping
    by_concept_serializable = {
        cat: nodes for cat, nodes in categorized.items()
    }
    by_concept_serializable["uncategorized"] = uncategorized
    with open(output_dir / "schema_nodes_by_concept.json", "w", encoding="utf-8") as f:
        json.dump(by_concept_serializable, f, indent=2)
    print(f"✓ Saved output/schema_nodes_by_concept.json")

    # save human-readable summary
    write_summary(categorized, uncategorized, output_dir / "schema_nodes_summary.txt")

    # print the most important nodes for the paper's core concepts
    print("\n── Key nodes for your core concepts ─────────────────")
    key_names = [
        "solar_longitude", "atmospheric_opacity", "mission_phase_name",
        "north_bounding_coordinate", "south_bounding_coordinate",
        "east_bounding_coordinate", "west_bounding_coordinate",
        "exposure_duration", "bandwidth", "filter_name",
        "gain_number", "detector_id", "product_type_name"
    ]
    for n in all_clean:
        if n["name"] in key_names:
            print(f"  [{n['ldd']}] {n['class']}.{n['name']}")
            print(f"    {n['description'][:120]}")
            if n.get("permissible_values"):
                vals = [v["value"] for v in n["permissible_values"][:5]]
                print(f"    values: {vals}")

if __name__ == "__main__":
    main()
