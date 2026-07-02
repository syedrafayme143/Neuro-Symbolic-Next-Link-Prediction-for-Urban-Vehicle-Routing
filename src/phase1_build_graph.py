"""
phase1_build_graph.py
=====================
Phase 1 of the Semantic Road Graph-Based Next-Link Prediction project.

Downloads the drivable road network for Ingolstadt, Germany using OSMnx,
cleans edge attributes (length, oneway, highway, maxspeed), imputes missing
speed limits based on road type, and saves:
  - data/raw/ingolstadt_nodes.csv
  - data/raw/ingolstadt_edges.csv
  - data/raw/ingolstadt_graph.pkl
"""

import os
import pickle
import re

import numpy as np
import osmnx as ox
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
LOCATION = "Ingolstadt, Germany"
RAW_DIR = os.path.join("data", "raw")
NODES_CSV = os.path.join(RAW_DIR, "ingolstadt_nodes.csv")
EDGES_CSV = os.path.join(RAW_DIR, "ingolstadt_edges.csv")
GRAPH_PKL = os.path.join(RAW_DIR, "ingolstadt_graph.pkl")

# Default speed limits (km/h) by OSM highway type, used when maxspeed is missing.
HIGHWAY_SPEED_DEFAULTS = {
    "motorway": 130,
    "motorway_link": 80,
    "trunk": 100,
    "trunk_link": 60,
    "primary": 70,
    "primary_link": 50,
    "secondary": 60,
    "secondary_link": 50,
    "tertiary": 50,
    "tertiary_link": 50,
    "unclassified": 50,
    "residential": 30,
    "living_street": 10,
    "service": 20,
    "road": 50,           
}
GLOBAL_DEFAULT_SPEED = 50  


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _extract_first(value):
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _parse_maxspeed(raw_value):
    if raw_value is None or (isinstance(raw_value, float) and np.isnan(raw_value)):
        return None

    raw_value = _extract_first(raw_value)
    if raw_value is None:
        return None

    s = str(raw_value).strip().lower()

    if s.isdigit():
        return int(s)

    m = re.match(r"^(\d+)\s*(km/?h|mph)?", s)
    if m:
        speed = int(m.group(1))
        if "mph" in s:
            speed = round(speed * 1.60934)  
        return speed

    conditional_map = {
        "de:urban": 50, "de:rural": 100, "de:motorway": 130,
        "de:living_street": 10, "de:pedestrian": 10, "urban": 50,
        "rural": 100, "motorway": 130, "walk": 10, "none": 130,
    }
    for key, val in conditional_map.items():
        if key in s:
            return val

    return None  


def clean_edges(edges_gdf: pd.DataFrame) -> pd.DataFrame:
    print("[Phase 1] Cleaning edge attributes ...")
    df = edges_gdf.reset_index().copy()

    # Retained 'key' to uniquely identify parallel lines in MultiDiGraphs
    keep = ["u", "v", "key", "length", "oneway", "highway", "maxspeed"]
    for col in keep:
        if col not in df.columns:
            df[col] = None  

    df = df[keep].copy()
    df["highway"] = df["highway"].apply(_extract_first).fillna("road")
    df["oneway"] = df["oneway"].apply(_extract_first).fillna(False).astype(bool)
    df["length"] = pd.to_numeric(df["length"], errors="coerce")
    df["maxspeed"] = df["maxspeed"].apply(_parse_maxspeed)

    missing_mask = df["maxspeed"].isna()
    print(f"[Phase 1]   maxspeed missing before imputation : {missing_mask.sum():,} / {len(df):,} edges")

    df.loc[missing_mask, "maxspeed"] = df.loc[missing_mask, "highway"].map(HIGHWAY_SPEED_DEFAULTS)

    still_missing = df["maxspeed"].isna()
    df.loc[still_missing, "maxspeed"] = GLOBAL_DEFAULT_SPEED
    df["maxspeed"] = df["maxspeed"].astype(int)

    print(f"[Phase 1] Edge cleaning complete. Shape: {df.shape}")
    return df


def clean_nodes(nodes_gdf: pd.DataFrame) -> pd.DataFrame:
    df = nodes_gdf.reset_index().copy()
    keep = [c for c in ["osmid", "x", "y", "street_count"] if c in df.columns]
    return df[keep].copy()


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    print(f"[Phase 1] Downloading drivable road network for '{LOCATION}' ...")
    
    G = ox.graph_from_place(LOCATION, network_type="drive", simplify=True)
    print(f"[Phase 1] Graph downloaded  →  nodes: {G.number_of_nodes():,}  |  edges: {G.number_of_edges():,}")

    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G, nodes=True, edges=True)

    nodes_df = clean_nodes(nodes_gdf)
    edges_df = clean_edges(edges_gdf)

    nodes_df.to_csv(NODES_CSV, index=False)
    edges_df.to_csv(EDGES_CSV, index=False)

    with open(GRAPH_PKL, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[Phase 1] Graph pickle saved  →  {GRAPH_PKL} \n✓ Phase 1 complete.\n")


if __name__ == "__main__":
    main()