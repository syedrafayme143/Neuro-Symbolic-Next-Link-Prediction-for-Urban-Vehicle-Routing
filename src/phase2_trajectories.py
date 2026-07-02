"""
phase2_trajectories.py
======================
Phase 2 of the Semantic Road Graph-Based Next-Link Prediction project.
Loads the NetworkX graph and generates 5,000 synthetic driving trajectories.
"""

import json
import os
import pickle
import random
import time
import networkx as nx
import numpy as np

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
GRAPH_PKL = os.path.join("data", "raw", "ingolstadt_graph.pkl")
SYNTHETIC_DIR = os.path.join("data", "synthetic")
OUTPUT_JSON = os.path.join(SYNTHETIC_DIR, "ground_truth_paths.json")

NUM_TRAJECTORIES = 5_000
MIN_PATH_LENGTH_NODES = 3    
MAX_SAMPLE_ATTEMPTS = 20     
RANDOM_SEED = 42

def load_graph(path: str) -> nx.MultiDiGraph:
    with open(path, "rb") as f:
        G = pickle.load(f)
    return G

def get_unique_edge_sequence(G: nx.MultiDiGraph, node_sequence: list) -> list:
    """
    Finds the exact (u, v, key) edge sequence taken. If parallel choices exist,
    selects the edge with the shortest physical length attribute.
    """
    edge_sequence = []
    for i in range(len(node_sequence) - 1):
        u, v = node_sequence[i], node_sequence[i+1]
        edge_choices = G[u][v]
        # Find key with minimum length
        best_key = min(edge_choices.keys(), key=lambda k: edge_choices[k].get("length", 1.0))
        edge_sequence.append([u, v, best_key]) # Stored as list for valid JSON compliance
    return edge_sequence

def compute_path_length(G: nx.MultiDiGraph, edge_sequence: list) -> float:
    return sum(G[e[0]][e[1]][e[2]].get("length", 1.0) for e in edge_sequence)

def generate_trajectories(G: nx.MultiDiGraph, n: int, seed: int) -> list:
    random.seed(seed)
    np.random.seed(seed)

    print("[Phase 2] Extracting largest strongly connected component (SCC) ...")
    largest_scc = max(nx.strongly_connected_components(G), key=len)
    G_scc = G.subgraph(largest_scc).copy()
    node_list = list(G_scc.nodes())

    trajectories = []
    traj_id = 0
    
    print(f"[Phase 2] Generating {n:,} trajectories ...")
    while traj_id < n:
        source, target = random.sample(node_list, 2)
        success = False
        
        for _ in range(MAX_SAMPLE_ATTEMPTS):
            try:
                node_seq = nx.shortest_path(G_scc, source, target, weight="length")
                if len(node_seq) >= MIN_PATH_LENGTH_NODES:
                    success = True
                    break
                target = random.choice(node_list)
            except nx.NetworkXNoPath:
                source, target = random.sample(node_list, 2)

        if not success:
            continue

        edge_seq = get_unique_edge_sequence(G_scc, node_seq)
        total_len = compute_path_length(G_scc, edge_seq)

        trajectories.append({
            "traj_id": traj_id,
            "source_node": source,
            "target_node": target,
            "node_sequence": node_seq,
            "edge_sequence": edge_seq, # Elements are now explicit [u, v, key]
            "total_length_m": round(total_len, 2),
            "num_edges": len(edge_seq),
        })
        traj_id += 1

    return trajectories

def main():
    os.makedirs(SYNTHETIC_DIR, exist_ok=True)
    G = load_graph(GRAPH_PKL)
    trajectories = generate_trajectories(G, NUM_TRAJECTORIES, RANDOM_SEED)

    print(f"[Phase 2] Saving trajectories to '{OUTPUT_JSON}' ...")
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(trajectories, f, indent=2)
    print("[Phase 2] ✓ Phase 2 complete.\n")

if __name__ == "__main__":
    main()
 