"""
parse_ldd.py
------------
Fetches PDS4 Local Data Dictionary (LDD) JSON files and extracts
schema nodes for the LUMIN knowledge graph.

Output: schema_nodes.json — one entry per attribute, structured for
        direct use as schema leaf nodes in NetworkX KG construction.

Usage:
    pip install requests networkx
    python parse_ldd.py
    python parse_ldd.py --local cart=data/PDS4_CART_1P00_1970.json --local img=data/PDS4_IMG_1P00_1930.json

Outputs:
    schema_nodes.json   — flat list of all parsed schema nodes
    schema_graph.json   — NetworkX node-link format (load with nx.node_link_graph)
"""

import json
import argparse
from pathlib import Path
import requests
import networkx as nx
from collections import defaultdict

# ── LDDs to fetch ─────────────────────────────────────────────────────────────
LDDS = {
    "cart": "https://pds.nasa.gov/pds4/cart/v1/PDS4_CART_1P00_1970.JSON",
    "img":  "https://pds.nasa.gov/pds4/img/v1/PDS4_IMG_1P00_1930.JSON",
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def fetch_ldd(name, url):
    print(f"  Fetching {name}...", end=" ")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    print("OK")
    return data

def load_ldd_file(name, path):
    print(f"  Loading {name} from {path}...", end=" ")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print("OK")
    return data

def parse_local_source(value):
    """Parse --local input in form NAME=PATH."""
    if "=" not in value:
        raise argparse.ArgumentTypeError("--local must be in form name=path")
    name, path = value.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise argparse.ArgumentTypeError("--local must be in form name=path")
    return name, path

def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse PDS4 LDD JSON files into schema nodes and a schema graph."
    )
    parser.add_argument(
        "--local",
        action="append",
        default=[],
        type=parse_local_source,
        metavar="NAME=PATH",
        help="Load local LDD JSON file instead of fetching defaults. Can be repeated.",
    )
    return parser.parse_args()

def extract_class_name(identifier):
    """
    Identifiers look like:
      '0001_NASA_PDS_1.cart.Bounding_Coordinates.cart.east_bounding_coordinate'
    We want the class name: 'Bounding_Coordinates'
    Strategy: find the first segment that starts with an uppercase letter.
    """
    parts = identifier.split(".")
    for part in parts:
        if part and part[0].isupper():
            return part
    return "Unknown"

def parse_permissible_values(attr):
    """Extract controlled vocabulary from PermissibleValueList if present."""
    pv_list = attr.get("PermissibleValueList", [])
    if not pv_list:
        return []
    values = []
    for pv in pv_list:
        entry = pv.get("PermissibleValue", {})
        values.append({
            "value": entry.get("value", ""),
            "meaning": entry.get("valueMeaning", ""),
            "deprecated": entry.get("isDeprecated", "false") == "true"
        })
    return values

def parse_attribute(attr_wrapper, ldd_name):
    """Parse a single attributeDictionary entry into a clean schema node dict."""
    attr = attr_wrapper.get("attribute", {})
    if not attr:
        return None

    identifier  = attr.get("identifier", "")
    title       = attr.get("title", "")
    description = attr.get("description", "").strip()
    data_type   = attr.get("dataType", "")
    unit        = attr.get("unitOfMeasure", "")
    unit_ids    = attr.get("unitId", "")          # e.g. "deg, rad, arcmin"
    min_val     = attr.get("minimumValue", None)
    max_val     = attr.get("maximumValue", None)
    enumerated  = attr.get("isEnumerated", "false") == "true"
    namespace   = attr.get("nameSpaceId", ldd_name)
    deprecated  = attr.get("isDeprecated", "false") == "true"

    if deprecated:
        return None  # skip deprecated attributes

    class_name = extract_class_name(identifier)

    node = {
        # identity
        "node_type":   "schema_leaf",
        "ldd":         ldd_name,
        "namespace":   namespace,
        "class":       class_name,
        "name":        title,
        "identifier":  identifier,

        # semantics — used for embedding + description
        "description": description,

        # typing
        "data_type":   data_type,
        "enumerated":  enumerated,

        # constraints (useful for filter generation)
        "min_value":   min_val if min_val not in ("Unbounded", None) else None,
        "max_value":   max_val if max_val not in ("Unbounded", None) else None,
        "unit":        unit if unit != "null" else None,
        "unit_ids":    [u.strip() for u in unit_ids.split(",")] if unit_ids and unit_ids != "null" else [],

        # controlled vocabulary (for enumerated attributes)
        "permissible_values": parse_permissible_values(attr) if enumerated else [],
    }
    return node

def parse_ldd_json(data, ldd_name):
    """Parse a full LDD JSON response into a list of schema nodes."""
    dd = data[0].get("dataDictionary", {})
    attr_dict = dd.get("attributeDictionary", [])

    nodes = []
    skipped = 0
    for entry in attr_dict:
        node = parse_attribute(entry, ldd_name)
        if node:
            nodes.append(node)
        else:
            skipped += 1

    print(f"    → {len(nodes)} attributes extracted, {skipped} skipped (deprecated)")
    return nodes

# ── Graph construction ─────────────────────────────────────────────────────────

def build_graph(all_nodes):
    """
    Build a NetworkX DiGraph with:
      - LDD nodes (top level)
      - Class nodes (mid level)
      - Schema leaf nodes (attribute level)
    
    Edges:
      ldd → class:   'contains_class'
      class → attr:  'has_attribute'
    
    NL alias and concept nodes will be added on top of this
    by Shaurya in the next phase.
    """
    G = nx.DiGraph()

    # group by ldd → class
    by_ldd_class = defaultdict(lambda: defaultdict(list))
    for node in all_nodes:
        by_ldd_class[node["ldd"]][node["class"]].append(node)

    for ldd_name, classes in by_ldd_class.items():
        ldd_node_id = f"ldd:{ldd_name}"
        G.add_node(ldd_node_id, node_type="ldd", name=ldd_name)

        for class_name, attrs in classes.items():
            class_node_id = f"class:{ldd_name}.{class_name}"
            G.add_node(class_node_id,
                       node_type="schema_class",
                       name=class_name,
                       ldd=ldd_name)
            G.add_edge(ldd_node_id, class_node_id, edge_type="contains_class")

            for attr in attrs:
                attr_node_id = f"attr:{attr['identifier']}"
                G.add_node(attr_node_id, **attr)
                G.add_edge(class_node_id, attr_node_id,
                           edge_type="has_attribute")

    return G

# ── Summary ────────────────────────────────────────────────────────────────────

def print_summary(all_nodes, G):
    print("\n-- Summary ---------------------------------------------------")
    by_ldd = defaultdict(int)
    enumerated_count = 0
    with_units = 0
    with_constraints = 0

    for n in all_nodes:
        by_ldd[n["ldd"]] += 1
        if n["enumerated"]:
            enumerated_count += 1
        if n["unit"]:
            with_units += 1
        if n["min_value"] or n["max_value"]:
            with_constraints += 1

    for ldd, count in sorted(by_ldd.items()):
        print(f"  {ldd:10s}  {count:4d} attributes")

    print(f"\n  Total attributes : {len(all_nodes)}")
    print(f"  Enumerated       : {enumerated_count}  (controlled vocab — good for T0/T1 exact match)")
    print(f"  With units       : {with_units}  (numeric — good for range filters)")
    print(f"  With constraints : {with_constraints}  (min/max bounds available)")
    print(f"\n  Graph nodes      : {G.number_of_nodes()}")
    print(f"  Graph edges      : {G.number_of_edges()}")
    print("------------------------------------------------------")

    # show a few interesting enumerated ones
    print("\n-- Sample enumerated attributes (controlled vocab) ----")
    shown = 0
    for n in all_nodes:
        if n["enumerated"] and n["permissible_values"] and shown < 5:
            vals = [v["value"] for v in n["permissible_values"]]
            print(f"  [{n['ldd']}] {n['class']}.{n['name']}")
            print(f"    values: {vals}")
            shown += 1

    # show a few with units/ranges
    print("\n-- Sample numeric attributes (range filters) ----------")
    shown = 0
    for n in all_nodes:
        if n["unit"] and n["min_value"] and shown < 5:
            print(f"  [{n['ldd']}] {n['class']}.{n['name']}")
            print(f"    range: [{n['min_value']}, {n['max_value']}]  unit: {n['unit']}")
            shown += 1

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("PDS4 LDD Parser — LUMIN KG Schema Layer\n")

    all_nodes = []

    if args.local:
        print("Loading local LDD files:")
        for ldd_name, path in args.local:
            try:
                data = load_ldd_file(ldd_name, path)
                nodes = parse_ldd_json(data, ldd_name)
                all_nodes.extend(nodes)
            except Exception as e:
                print(f"  WARNING: Could not load {ldd_name} from {path}: {e}")
    else:
        print("Fetching LDDs:")
        for ldd_name, url in LDDS.items():
            try:
                data = fetch_ldd(ldd_name, url)
                nodes = parse_ldd_json(data, ldd_name)
                all_nodes.extend(nodes)
            except Exception as e:
                print(f"  WARNING: Could not fetch {ldd_name}: {e}")

    print(f"\nBuilding graph...")
    G = build_graph(all_nodes)

    print_summary(all_nodes, G)

    # ── Save outputs ────────────────────────────────────────────────────────
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "schema_nodes.json", "w", encoding="utf-8") as f:
        json.dump(all_nodes, f, indent=2)
    print("\nSaved output/schema_nodes.json")

    graph_data = nx.node_link_data(G)
    with open(output_dir / "schema_graph.json", "w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=2)
    print("Saved output/schema_graph.json")

    print("\nNext step for Shaurya:")
    print("  Load schema_graph.json with nx.node_link_graph()")
    print("  Add concept nodes and alias nodes on top.")
    print("  Edge types to add: is_alias_of, measured_by, corresponds_to")

if __name__ == "__main__":
    main()
