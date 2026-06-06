import json
from pathlib import Path

output_dir = Path(__file__).parent.parent / "output"
output_dir.mkdir(exist_ok=True)

with open(output_dir / "schema_nodes.json", "r", encoding="utf-8") as f:
    nodes = json.load(f)

# deduplicate by class + name (keep first occurrence)
seen = set()
deduped = []
for n in nodes:
    key = f"{n['class']}.{n['name']}"
    if key not in seen:
        seen.add(key)
        deduped.append(n)

print(f"Before: {len(nodes)}  After: {len(deduped)}")
with open(output_dir / "schema_nodes_final.json", "w", encoding="utf-8") as f:
    json.dump(deduped, f, indent=2)
print(f"Saved output/schema_nodes_final.json")