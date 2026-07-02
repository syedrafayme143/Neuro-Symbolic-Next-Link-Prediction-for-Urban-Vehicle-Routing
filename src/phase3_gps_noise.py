"""
phase3_gps_noise.py
===================
Phase 3 of the Semantic Road Graph-Based Next-Link Prediction project.
Generates noisy GPS tracks interpolated continuously along physical road geometries,
explicitly tracking the true edge [u, v, key] for EVERY individual point.
"""

import json
import os
import pickle
import numpy as np
import networkx as nx
from shapely.geometry import LineString

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
GRAPH_PKL        = os.path.join("data", "raw",       "ingolstadt_graph.pkl")
GT_JSON          = os.path.join("data", "synthetic", "ground_truth_paths.json")
OUTPUT_JSON      = os.path.join("data", "synthetic", "noisy_gps_data.json")

NOISE_LEVELS = {"low_5m": 5, "medium_10m": 10, "high_20m": 20}
RANDOM_SEED = 42

METRES_PER_DEGREE_LAT = 111_320.0
INGOLSTADT_LAT_RAD = np.radians(48.77)
METRES_PER_DEGREE_LON = METRES_PER_DEGREE_LAT * np.cos(INGOLSTADT_LAT_RAD)

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def add_noise(lat: float, lon: float, sigma_m: float, rng: np.random.Generator):
    delta_north_m = rng.normal(0.0, sigma_m)
    delta_east_m  = rng.normal(0.0, sigma_m)
    noisy_lat = lat + (delta_north_m / METRES_PER_DEGREE_LAT)
    noisy_lon = lon + (delta_east_m / METRES_PER_DEGREE_LON)
    return (round(noisy_lat, 7), round(noisy_lon, 7))

def extract_trajectory_points_and_labels(G: nx.MultiDiGraph, edge_sequence: list) -> tuple:
    """
    Extracts continuous true coordinates along edge geometries, and pairs 
    EVERY individual coordinate point with its exact generating true edge [u, v, key].
    """
    true_coords = []
    point_edge_labels = [] # Tracks the true edge matching each point index
    
    for idx, (u, v, key) in enumerate(edge_sequence):
        edge_data = G[u][v][key]
        
        if "geometry" in edge_data and isinstance(edge_data["geometry"], LineString):
            coords = list(edge_data["geometry"].coords)
            formatted_coords = [[lat, lon] for lon, lat in coords]
        else:
            start_node = G.nodes[u]
            end_node = G.nodes[v]
            formatted_coords = [[start_node['y'], start_node['x']], [end_node['y'], end_node['x']]]
        
        # Deduplicate intersection connection points while keeping correct labels
        if idx > 0:
            formatted_coords = formatted_coords[1:]
            
        for coord in formatted_coords:
            true_coords.append(coord)
            point_edge_labels.append([u, v, key]) # Point-level ground truth label
            
    return true_coords, point_edge_labels

def main():
    rng = np.random.default_rng(RANDOM_SEED)
    
    print(f"[Phase 3] Loading graph structure...")
    with open(GRAPH_PKL, "rb") as f:
        G = pickle.load(f)

    with open(GT_JSON, "r", encoding="utf-8") as f:
        trajectories = json.load(f)

    print(f"[Phase 3] Generating geometric noise traces with point-level tracking labels...")
    enriched = []

    for idx, traj in enumerate(trajectories):
        edge_seq = traj["edge_sequence"]
        
        # FIXED: Returns paired spatial points and labels
        true_coords, point_labels = extract_trajectory_points_and_labels(G, edge_seq)

        noisy_gps = {}
        for level_name, sigma_m in NOISE_LEVELS.items():
            noisy_gps[level_name] = [
                list(add_noise(lat, lon, sigma_m, rng)) for (lat, lon) in true_coords
            ]

        record = {
            "traj_id"       : traj["traj_id"],
            "source_node"   : traj["source_node"],
            "target_node"   : traj["target_node"],
            "node_sequence" : traj["node_sequence"],
            "edge_sequence" : traj["edge_sequence"],
            "total_length_m": traj["total_length_m"],
            "num_edges"     : traj["num_edges"],
            "true_coords"   : true_coords,
            "point_labels"  : point_labels, # Point-by-point accurate alignment target
            "noisy_gps"     : noisy_gps,
        }
        enriched.append(record)

    output = {
        "metadata": {
            "num_trajectories" : len(enriched),
            "noise_levels"     : {k: f"±{v}m Gaussian" for k, v in NOISE_LEVELS.items()},
            "alignment_mode"   : "Point-to-Edge aligned truth map data",
        },
        "trajectories": enriched,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
        
    print(f"[Phase 3] Saved updated labels to '{OUTPUT_JSON}'\n✓ Phase 3 complete.")

if __name__ == "__main__":
    main()