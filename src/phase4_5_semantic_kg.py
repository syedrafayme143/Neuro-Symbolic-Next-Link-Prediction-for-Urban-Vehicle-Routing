"""
phase4_5_semantic_kg.py
========================
Phase 4.5 of the Semantic Road Graph-Based Next-Link Prediction project.

Converts the flat tabular road network (nodes + edges CSVs from Phase 1)
into a W3C-standard RDF Knowledge Graph. This is the symbolic half of the
project's "neuro-symbolic" architecture — later phases combine these
explicit logical constraints (legal topology, speed limits, one-way
restrictions) with the learned Node2Vec embeddings.

Ontology design
----------------
Namespace: EX = http://smartmobility.bmw.org/ontology#
(NOTE: swap this for your own placeholder domain, e.g.
 http://smartmobility.org/ontology#, before publishing publicly —
 this is a personal portfolio project, not BMW-affiliated work.)

Classes:
  EX.Intersection   — a road network node (OSM intersection)
  EX.RoadSegment    — a directed road edge between two intersections

Individuals:
  EX.Intersection_<osmid>          — one per node
  EX.Road_<u>_<v>_<key>            — one per edge (u, v, key uniquely
                                       identifies parallel edges between
                                       the same node pair, e.g. dual
                                       carriageways)

Object properties (relations between individuals):
  EX.startsFrom   RoadSegment -> Intersection   (the edge's origin node u)
  EX.connectsTo   RoadSegment -> Intersection   (the edge's destination node v)

Datatype properties (literals):
  EX.hasSpeedLimit  RoadSegment -> xsd:integer   (km/h)
  EX.isOneWay       RoadSegment -> xsd:boolean
  EX.roadType       RoadSegment -> xsd:string    (OSM highway classification)
  EX.hasLength      RoadSegment -> xsd:float     (metres)
  EX.hasLatitude    Intersection -> xsd:float
  EX.hasLongitude   Intersection -> xsd:float

Inputs:
  - data/raw/ingolstadt_nodes.csv
  - data/raw/ingolstadt_edges.csv

Output:
  - data/processed/ingolstadt_semantic_map.ttl   (Turtle-serialized RDF graph)
"""

import os

import pandas as pd
from rdflib import Graph, Namespace, RDF, Literal, URIRef
from rdflib.namespace import RDFS, XSD

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
NODES_CSV     = os.path.join("data", "raw",       "ingolstadt_nodes.csv")
EDGES_CSV     = os.path.join("data", "raw",       "ingolstadt_edges.csv")
PROCESSED_DIR = os.path.join("data", "processed")
OUTPUT_TTL    = os.path.join(PROCESSED_DIR, "ingolstadt_semantic_map.ttl")

ONTOLOGY_URI = "http://smartmobility.bmw.org/ontology#"
EX = Namespace(ONTOLOGY_URI)


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def intersection_uri(osmid) -> URIRef:
    """Build the URI for a node/intersection individual."""
    return EX[f"Intersection_{int(osmid)}"]


def road_uri(u, v, key) -> URIRef:
    """Build the URI for an edge/road-segment individual using the
    (u, v, key) tuple — `key` disambiguates parallel edges (e.g. dual
    carriageways) between the same pair of nodes."""
    return EX[f"Road_{int(u)}_{int(v)}_{int(key)}"]


def to_bool(value) -> bool:
    """Robustly coerce a CSV cell (which may already be bool, or the
    strings 'True'/'False'/'1'/'0') into a Python bool."""
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in ("true", "1", "yes")


# ---------------------------------------------------------------------------
# ONTOLOGY SCHEMA (lightweight — classes & property declarations)
# ---------------------------------------------------------------------------

def declare_schema(g: Graph):
    """
    Add minimal RDFS schema triples so the graph is self-describing:
    declares the two classes and gives human-readable labels to the
    key properties. Optional for SPARQL querying, but makes the .ttl
    file far more inspectable/maintainable for a portfolio reviewer.
    """
    print("[Phase 4.5] Declaring ontology schema (classes & property labels) …")

    g.add((EX.Intersection, RDF.type, RDFS.Class))
    g.add((EX.Intersection, RDFS.label, Literal("Road network intersection (OSM node)")))

    g.add((EX.RoadSegment, RDF.type, RDFS.Class))
    g.add((EX.RoadSegment, RDFS.label, Literal("Directed road segment (OSM edge)")))

    property_labels = {
        EX.startsFrom:    "Road segment originates at this intersection",
        EX.connectsTo:    "Road segment terminates at this intersection",
        EX.hasSpeedLimit: "Legal speed limit in km/h",
        EX.isOneWay:      "Whether the road segment is one-way only",
        EX.roadType:      "OSM highway classification (e.g. residential, primary)",
        EX.hasLength:     "Physical length of the segment in metres",
        EX.hasLatitude:   "WGS84 latitude of the intersection",
        EX.hasLongitude:  "WGS84 longitude of the intersection",
    }
    for prop, label in property_labels.items():
        g.add((prop, RDFS.label, Literal(label)))

    print(f"[Phase 4.5] Schema declared: 2 classes, {len(property_labels)} properties")


# ---------------------------------------------------------------------------
# TRIPLE GENERATION
# ---------------------------------------------------------------------------

def add_intersections(g: Graph, nodes_df: pd.DataFrame):
    """Serialize every node as an EX.Intersection individual with lat/lon."""
    print(f"[Phase 4.5] Serializing {len(nodes_df):,} intersections …")

    for i, row in enumerate(nodes_df.itertuples(index=False)):
        node_uri = intersection_uri(row.osmid)

        g.add((node_uri, RDF.type, EX.Intersection))
        g.add((node_uri, EX.hasLatitude, Literal(float(row.y), datatype=XSD.float)))
        g.add((node_uri, EX.hasLongitude, Literal(float(row.x), datatype=XSD.float)))

        if (i + 1) % 5000 == 0:
            print(f"[Phase 4.5]   …{i + 1:,}/{len(nodes_df):,} intersections serialized")

    print(f"[Phase 4.5] ✓ All {len(nodes_df):,} intersections serialized")


def add_road_segments(g: Graph, edges_df: pd.DataFrame):
    """
    Serialize every edge as an EX.RoadSegment individual, linking it to
    its start/end intersections and attaching speed limit, one-way flag,
    road type, and length as literal datatype properties.
    """
    print(f"[Phase 4.5] Serializing {len(edges_df):,} road segments …")

    for i, row in enumerate(edges_df.itertuples(index=False)):
        # `key` disambiguates parallel edges between the same (u, v) pair.
        # Guard against a missing column (default to 0) for robustness.
        key = getattr(row, "key", 0)
        if pd.isna(key):
            key = 0

        seg_uri = road_uri(row.u, row.v, key)
        u_uri = intersection_uri(row.u)
        v_uri = intersection_uri(row.v)

        # --- Type assertion ---
        g.add((seg_uri, RDF.type, EX.RoadSegment))

        # --- Topological constraints ---
        g.add((seg_uri, EX.startsFrom, u_uri))
        g.add((seg_uri, EX.connectsTo, v_uri))

        # --- Physical / legal properties ---
        speed = row.maxspeed if not pd.isna(row.maxspeed) else 50
        g.add((seg_uri, EX.hasSpeedLimit, Literal(int(speed), datatype=XSD.integer)))

        g.add((seg_uri, EX.isOneWay, Literal(to_bool(row.oneway), datatype=XSD.boolean)))

        road_type = str(row.highway) if not pd.isna(row.highway) else "unclassified"
        g.add((seg_uri, EX.roadType, Literal(road_type, datatype=XSD.string)))

        length = row.length if not pd.isna(row.length) else 0.0
        g.add((seg_uri, EX.hasLength, Literal(float(length), datatype=XSD.float)))

        if (i + 1) % 5000 == 0:
            print(f"[Phase 4.5]   …{i + 1:,}/{len(edges_df):,} road segments serialized")

    print(f"[Phase 4.5] ✓ All {len(edges_df):,} road segments serialized")


# ---------------------------------------------------------------------------
# VERIFICATION ENGINE — native SPARQL query
# ---------------------------------------------------------------------------

def query_outgoing_roads(g: Graph, intersection_osmid: int):
    """
    Run a native SPARQL query against the in-memory RDF graph:
    given a mock intersection ID, return every legally accessible
    outgoing road segment from that intersection, along with its
    speed limit, road type, and one-way flag.

    This demonstrates the "symbolic reasoning" half of the
    neuro-symbolic architecture: rather than relying on a learned
    model, we can deterministically and verifiably query the legal
    road topology.
    """
    node_uri = intersection_uri(intersection_osmid)

    query = f"""
    PREFIX ex: <{ONTOLOGY_URI}>
    SELECT ?road ?destination ?speedLimit ?roadType ?oneWay
    WHERE {{
        ?road ex:startsFrom ?origin .
        ?road ex:connectsTo ?destination .
        ?road ex:hasSpeedLimit ?speedLimit .
        ?road ex:roadType ?roadType .
        ?road ex:isOneWay ?oneWay .
        FILTER (?origin = <{node_uri}>)
    }}
    ORDER BY ?road
    """

    print(f"\n[Phase 4.5] ── SPARQL Verification Query ──────────────────────")
    print(f"[Phase 4.5] Querying all legally accessible outgoing roads from:")
    print(f"[Phase 4.5]   {node_uri}")

    results = list(g.query(query))

    if not results:
        print(f"[Phase 4.5] ⚠  No outgoing roads found for intersection {intersection_osmid}. "
              f"(Check that this OSM node ID exists in the loaded graph.)")
        return results

    print(f"[Phase 4.5] Found {len(results)} outgoing road segment(s):\n")
    print(f"[Phase 4.5] {'Road URI':<45} {'→ Destination':<28} {'Speed':>6}  {'Type':<14} {'OneWay':>7}")
    print(f"[Phase 4.5] {'-'*45} {'-'*28} {'-'*6}  {'-'*14} {'-'*7}")

    for row in results:
        road_short = str(row.road).replace(ONTOLOGY_URI, "ex:")
        dest_short = str(row.destination).replace(ONTOLOGY_URI, "ex:")
        print(f"[Phase 4.5] {road_short:<45} {dest_short:<28} "
              f"{int(row.speedLimit):>4} km/h  {str(row.roadType):<14} {str(row.oneWay):>7}")

    print(f"[Phase 4.5] ── End of SPARQL verification ─────────────────────\n")
    return results


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load cleaned tabular data from Phase 1
    # ------------------------------------------------------------------
    print(f"[Phase 4.5] Loading nodes from '{NODES_CSV}' …")
    nodes_df = pd.read_csv(NODES_CSV)
    print(f"[Phase 4.5] Loaded {len(nodes_df):,} nodes")

    print(f"[Phase 4.5] Loading edges from '{EDGES_CSV}' …")
    edges_df = pd.read_csv(EDGES_CSV)
    print(f"[Phase 4.5] Loaded {len(edges_df):,} edges")

    # ------------------------------------------------------------------
    # 2. Initialize RDF graph & bind namespace
    # ------------------------------------------------------------------
    print(f"\n[Phase 4.5] Initializing rdflib.Graph() with ontology namespace:")
    print(f"[Phase 4.5]   EX = {ONTOLOGY_URI}")
    g = Graph()
    g.bind("ex", EX)

    # ------------------------------------------------------------------
    # 3. Declare lightweight schema
    # ------------------------------------------------------------------
    declare_schema(g)

    # ------------------------------------------------------------------
    # 4. Serialize nodes & edges as triples
    # ------------------------------------------------------------------
    add_intersections(g, nodes_df)
    add_road_segments(g, edges_df)

    print(f"\n[Phase 4.5] Knowledge graph constructed  →  {len(g):,} total triples")

    # ------------------------------------------------------------------
    # 5. Verification engine — run a SPARQL query on a sample intersection
    # ------------------------------------------------------------------
    # Use the first node in the dataset as a mock query target so this
    # script is fully self-contained and runnable without manual input.
    mock_intersection_id = int(nodes_df.iloc[0]["osmid"])
    query_outgoing_roads(g, mock_intersection_id)

    # ------------------------------------------------------------------
    # 6. Serialize to Turtle (.ttl)
    # ------------------------------------------------------------------
    print(f"[Phase 4.5] Serializing knowledge graph to Turtle format …")
    g.serialize(destination=OUTPUT_TTL, format="turtle")

    file_size_mb = os.path.getsize(OUTPUT_TTL) / (1024 ** 2)
    print(f"[Phase 4.5] Saved  →  {OUTPUT_TTL}  ({file_size_mb:.2f} MB, {len(g):,} triples)")
    print("[Phase 4.5] ✓  Phase 4.5 complete.\n")


if __name__ == "__main__":
    main()
