# LUMIN v2: Neurosymbolic Query Engine for NASA PDS

LUMIN converts natural language queries into structured PDS4 schema field filters using a hand-curated knowledge graph and embedding-based traversal.

A query like *"southern summer"* resolves to `solar_longitude ∈ [180, 360] deg`. A query like *"HiRISE"* resolves to `instrument_id = HiRISE`. Unknown jargon falls through to cosine-similarity search over concept embeddings and returns a confidence score so downstream systems know when to ask for clarification.

---

## How it works

```
NL query
   │
   ├─ Stage 1: exact alias lookup  (e.g. "southern summer" → alias:southern_summer)
   │
   └─ Stage 2: embedding similarity over concept + alias nodes
                       │
                       ▼
              BFS traversal through KG
              alias → concept → schema leaf
                       │
                       ▼
              PDS4 filter dict  +  confidence score
```

The knowledge graph has three node layers:

| Layer | Example | Role |
|---|---|---|
| Alias | `alias:southern_summer` | Surface string entry points (T0/T1) |
| Concept | `concept:MartianSouthernSummer` | Semantic grouping with description |
| Schema leaf | `schema:Time_Coordinates.solar_longitude` | Actual PDS4 field + constraints |

---

## Setup

```bash
git clone <repo>
cd lumin-v2-core
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

Copy `example.env` to `.env` and fill in your keys:

```
OPENAI_API_KEY=...          # used for embeddings in traversal.py
OPENROUTER_API_KEY=...      # used for the LLM (OLMo / eval pipeline)
```

---

## Pipeline

Run these in order from the repo root. All outputs go to `output/`.

### 1. Parse PDS4 LDD files → schema nodes

```bash
python analysis/parse_ldd.py --local cart=data/PDS4_CART_1P00_1970.json --local img=data/PDS4_IMG_1P00_1930.json
```

Outputs: `output/schema_nodes.json`, `output/schema_graph.json`

### 2. Filter to query-relevant nodes

```bash
python analysis/filter_schema.py
```

Removes internal DD machinery, groups nodes by concept category (geographic, instrument\_mode, seasonal\_temporal, etc.).

Outputs: `output/schema_nodes_clean.json`, `output/schema_nodes_by_concept.json`, `output/schema_nodes_summary.txt`

### 3. Deduplicate

```bash
python analysis/dedup.py
```

Removes duplicate `class.name` entries, keeping first occurrence.

Output: `output/schema_nodes_final.json`

### 4. Build the knowledge graph

```bash
python build/build_kg.py
```

Layers concept nodes and alias nodes on top of schema leaves. Edges:
- `alias → concept` via `is_alias_of`
- `concept → schema_leaf` via `measured_by`, `corresponds_to`, or `instrument_of`

Outputs: `output/lumin_kg.json`, `output/lumin_kg_summary.txt`

### 5. Run the traversal engine

```bash
python src/traversal.py
```

Runs the demo query set and prints filters + confidence scores for each.

### Visualize the knowledge graph

After `output/lumin_kg.json` exists, start the local static server:

```bash
node serve.js
```

Open `http://localhost:3000/` to explore aliases, concepts, schema leaves, and typed edges.

---

## Project structure

```
lumin-v2-core/
├── analysis/
│   ├── parse_ldd.py          # Parse PDS4 LDD JSON → schema nodes + graph
│   ├── filter_schema.py      # Filter to query-relevant nodes by concept
│   └── dedup.py              # Deduplicate by class.name
├── build/
│   └── build_kg.py           # Build full KG (aliases + concepts + schema leaves)
├── src/
│   └── traversal.py          # Query engine: VectorIndex + BFS traverser
├── data/
│   ├── PDS4_CART_1P00_1970.json   # Cartography LDD
│   ├── PDS4_IMG_1P00_1930.json    # Imaging LDD
│   └── normalized_pds_test_dataset.csv
├── output/                   # All generated artifacts (gitignored)

```
