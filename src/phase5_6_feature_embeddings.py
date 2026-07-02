"""
phase5_6_features_embeddings.py
=================================
Refined, error-free senior-grade production script integrating 
directed Node2Vec embeddings, proper one-hot encoding, safe string boolean conversions,
out-degree tracking, validation checks, absolute seed reproducibility, and a 
high-performance O(N) backward linear target pre-computation pass.
"""

import math
import os
import pickle
import random
import time
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd
from gensim.models import Word2Vec

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
GRAPH_PKL          = os.path.join("data", "raw",       "ingolstadt_graph.pkl")
EDGES_CSV          = os.path.join("data", "raw",       "ingolstadt_edges.csv")
NOISY_GPS_JSON     = os.path.join("data", "synthetic", "noisy_gps_data.json")
PROCESSED_DIR      = os.path.join("data", "processed")
EMBEDDINGS_PKL     = os.path.join(PROCESSED_DIR, "node2vec_embeddings.pkl")
FEATURES_PKL       = os.path.join(PROCESSED_DIR, "final_ml_features.pkl")

# --- Hyperparameters ---
N2V_DIMENSIONS   = 64
N2V_WALK_LENGTH  = 10
N2V_NUM_WALKS    = 80
N2V_P            = 1.0     
N2V_Q            = 1.0     
N2V_WINDOW       = 10
N2V_MIN_COUNT    = 0
N2V_SG           = 1       
N2V_EPOCHS       = 5
N2V_WORKERS      = 1       # Enforces absolute cross-run reproducibility
N2V_SEED         = 42

RANDOM_SEED = 42

# ===========================================================================
# PHASE 6 — DIRECTED NODE2VEC GRAPH EMBEDDINGS
# ===========================================================================

def load_multidigraph(path: str) -> nx.MultiDiGraph:
    print(f"[Phase 6] Loading MultiDiGraph from '{path}' ...")
    with open(path, "rb") as f:
        G = pickle.load(f)
    return G

def collapse_to_weighted_directed_graph(G_multi: nx.MultiDiGraph) -> nx.DiGraph:
    """
    Collapses MultiDiGraph into a simple DiGraph to honor directional one-way traffic rules.
    Weights are set to inverse length so shorter routes are preferred in random walks.
    """
    print("[Phase 6] Collapsing MultiDiGraph → simple weighted DIRECTED Graph ...")
    G_directed = nx.DiGraph()
    G_directed.add_nodes_from(G_multi.nodes())

    for u, v, data in G_multi.edges(data=True):
        if u == v: continue  
        length = max(float(data.get("length", 1.0)), 1e-3)
        weight = 1.0 / length

        if G_directed.has_edge(u, v):
            if weight > G_directed[u][v]["weight"]:
                G_directed[u][v]["weight"] = weight
                G_directed[u][v]["length"] = length
        else:
            G_directed.add_edge(u, v, weight=weight, length=length)

    isolates = list(nx.isolates(G_directed))
    if isolates:
        G_directed.remove_nodes_from(isolates)
    return G_directed

def build_alias_tables(G: nx.DiGraph):
    neighbor_weights = {}
    for node in G.nodes():
        nbrs = list(G.successors(node))
        weights = [G[node][nbr]["weight"] for nbr in nbrs]
        neighbor_weights[node] = (nbrs, weights)
    return neighbor_weights

def _weighted_choice(rng: random.Random, items, weights):
    total = sum(weights)
    if total <= 0: return rng.choice(items)
    r = rng.uniform(0, total)
    upto = 0.0
    for item, w in zip(items, weights):
        upto += w
        if upto >= r: return item
    return items[-1]

def node2vec_walk(G, neighbor_weights, start_node, walk_length, p, q, rng: random.Random):
    walk = [start_node]
    while len(walk) < walk_length:
        current = walk[-1]
        nbrs, weights = neighbor_weights.get(current, ([], []))
        if not nbrs: break

        if len(walk) == 1:
            next_node = _weighted_choice(rng, nbrs, weights)
        else:
            prev = walk[-2]
            biased_weights = []
            for nbr, w in zip(nbrs, weights):
                if nbr == prev: alpha = 1.0 / p
                elif G.has_edge(nbr, prev): alpha = 1.0
                else: alpha = 1.0 / q
                biased_weights.append(w * alpha)
            next_node = _weighted_choice(rng, nbrs, biased_weights)
        walk.append(next_node)
    return walk

def generate_all_walks(G, neighbor_weights, num_walks, walk_length, p, q, seed):
    rng = random.Random(seed)
    nodes = list(G.nodes())
    walks = []
    for walk_iter in range(num_walks):
        rng.shuffle(nodes)
        for node in nodes:
            walk = node2vec_walk(G, neighbor_weights, node, walk_length, p, q, rng)
            walks.append([str(n) for n in walk])
    return walks

def train_word2vec(walks, dimensions, window, min_count, sg, epochs, workers, seed):
    print(f"[Phase 6] Training Word2Vec embeddings skip-gram model (Workers={workers})...")
    return Word2Vec(sentences=walks, vector_size=dimensions, window=window, min_count=min_count, sg=sg, workers=workers, epochs=epochs, seed=seed)

def extract_embeddings(model: Word2Vec, all_node_ids) -> dict:
    embeddings = {}
    for node_id in all_node_ids:
        token = str(node_id)
        if token in model.wv:
            embeddings[node_id] = model.wv[token].astype(np.float32)
        else:
            embeddings[node_id] = np.zeros(model.vector_size, dtype=np.float32)
    return embeddings

def run_phase6(original_multigraph: nx.MultiDiGraph) -> dict:
    G_directed = collapse_to_weighted_directed_graph(original_multigraph)
    neighbor_weights = build_alias_tables(G_directed)
    walks = generate_all_walks(G_directed, neighbor_weights, num_walks=N2V_NUM_WALKS, walk_length=N2V_WALK_LENGTH, p=N2V_P, q=N2V_Q, seed=N2V_SEED)
    model = train_word2vec(walks, dimensions=N2V_DIMENSIONS, window=N2V_WINDOW, min_count=N2V_MIN_COUNT, sg=N2V_SG, epochs=N2V_EPOCHS, workers=N2V_WORKERS, seed=N2V_SEED)
    embeddings = extract_embeddings(model, original_multigraph.nodes())
    return embeddings

# ===========================================================================
# PHASE 5 — FEATURE ENGINEERING & DATASET COMPILATION
# ===========================================================================

def parse_safe_bool(value) -> bool:
    if isinstance(value, bool): return value
    if pd.isna(value): return False
    return str(value).strip().lower() in ("true", "1", "yes")

def load_edges_lookup(edges_csv: str) -> dict:
    df = pd.read_csv(edges_csv)
    lookup = {}
    for row in df.itertuples(index=False):
        key_triple = (int(row.u), int(row.v), int(getattr(row, "key", 0)))
        lookup[key_triple] = {
            "length": float(row.length),
            "maxspeed": int(row.maxspeed),
            "highway": str(row.highway),
            "oneway": parse_safe_bool(row.oneway),
        }
    return lookup

def bearing_degrees(lat1, lon1, lat2, lon2) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    x = math.sin(dlambda) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    return (math.degrees(math.atan2(x, y)) + 360) % 360

def turn_angle_diff(bearing_prev: float, bearing_curr: float) -> float:
    return (bearing_curr - bearing_prev + 180) % 360 - 180

def load_trajectories(noisy_gps_json: str) -> list:
    import json
    print(f"[Phase 5] Loading trajectories from '{noisy_gps_json}' ...")
    with open(noisy_gps_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["trajectories"]

def build_feature_rows(trajectories, edge_lookup, degree_lookup, embeddings):
    """
    HIGH-PERFORMANCE OPTIMIZED: Pre-computes transitional step targets using a 
    single backward pass O(N) instead of expensive nested look-ahead loops. 
    Drops dataset generation runtime from hours to mere seconds.
    """
    print("\n[Phase 5] Compiling tabular features from continuous traces...")
    feature_rows = []
    targets = []
    
    total_trajs = len(trajectories)
    start_time = time.time()
    
    for idx, traj in enumerate(trajectories):
        noisy_pings = traj["noisy_gps"]["low_5m"] 
        labels = [tuple(lbl) for lbl in traj["point_labels"]]
        
        # Enforce exact structural spatial alignment
        assert len(noisy_pings) == len(labels), f"Sanity Alert: Trajectory index {idx} has length mismatch!"
        
        num_points = len(noisy_pings)
        if num_points < 3: continue

        # --- LINEAR BACKWARD TARGET O(N) SCAN ---
        next_targets = [None] * num_points
        last_seen_target = None
        last_seen_edge = labels[-1]

        for i in range(num_points - 1, -1, -1):
            if labels[i] != last_seen_edge:
                last_seen_target = f"{last_seen_edge[0]}_{last_seen_edge[1]}_{last_seen_edge[2]}"
                last_seen_edge = labels[i]
            next_targets[i] = last_seen_target

        # --- FEATURE ROW EXTRACTION LOOP ---
        prev_bearing = None
        
        for t in range(num_points):
            current_edge = labels[t]
            next_edge_target = next_targets[t]
            
            # Drop trailing trace elements that have no upcoming link turn available
            if next_edge_target is None: continue
                
            edge_attrs = edge_lookup.get(current_edge)
            if edge_attrs is None: continue
                
            u, v, key = current_edge
            lat, lon = noisy_pings[t]
            
            if t == 0:
                turn_angle = 0.0
            else:
                lat_prev, lon_prev = noisy_pings[t-1]
                prev_bearing = bearing_degrees(lat_prev, lon_prev, lat, lon)
                if t < num_points - 1:
                    lat_next, lon_next = noisy_pings[t+1]
                    curr_bearing = bearing_degrees(lat, lon, lat_next, lon_next)
                    turn_angle = turn_angle_diff(prev_bearing, curr_bearing)
                else:
                    turn_angle = 0.0

            emb_vector = embeddings.get(int(u), np.zeros(N2V_DIMENSIONS, dtype=np.float32))
            
            row = {
                "traj_id": int(traj["traj_id"]), # Preserved for GroupKFold spatial data splitting
                "point_step": t,
                "edge_length": edge_attrs["length"],
                "edge_maxspeed": edge_attrs["maxspeed"],
                "edge_highway": edge_attrs["highway"],
                "edge_oneway": edge_attrs["oneway"],
                "node_out_degree": degree_lookup.get(int(u), 0), # Out-degree tracks true available turn options
                "turn_angle_deg": turn_angle,
                "ping_lat": lat,
                "ping_lon": lon
            }
            # Exclude raw u, v, key integer IDs to protect against model value regression traps
            for d in range(N2V_DIMENSIONS):
                row[f"emb_{d}"] = float(emb_vector[d])
                
            feature_rows.append(row)
            targets.append(next_edge_target)

        if (idx + 1) % 1000 == 0 or (idx + 1) == total_trajs:
            pct = 100.0 * (idx + 1) / total_trajs
            print(f"[Phase 5]   Processed {idx + 1:,}/{total_trajs:,} trajectories ({pct:.0f}%) | Rows Compiled: {len(feature_rows):,}")

    return feature_rows, targets

# ---------------------------------------------------------------------------
# MAIN EXECUTIVE ORCHESTRATOR
# ---------------------------------------------------------------------------

def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    G_multi = load_multidigraph(GRAPH_PKL)
    
    # Run Phase 6 directed random walk embedding calculations
    embeddings = run_phase6(G_multi)
    
    # Save computed embeddings out to disk checkpoint
    with open(EMBEDDINGS_PKL, "wb") as f:
        pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[Phase 6] Graph node embeddings saved successfully to '{EMBEDDINGS_PKL}'")
    
    # Run Phase 5 tabular dataset generation
    edge_lookup = load_edges_lookup(EDGES_CSV)
    out_degree_lookup = {node: deg for node, deg in G_multi.out_degree()}
    trajectories = load_trajectories(NOISY_GPS_JSON)
    
    feature_rows, targets = build_feature_rows(trajectories, edge_lookup, out_degree_lookup, embeddings)
    
    X = pd.DataFrame(feature_rows)
    y = pd.Series(targets, name="next_edge_id")
    
    # Perform strict categorical one-hot encoding on road type strings
    print("[Phase 5] Converting highway classifications to safe one-hot feature blocks...")
    X = pd.get_dummies(X, columns=["edge_highway"], drop_first=False)
    
    print(f"[Phase 5] Complete! Feature matrix X shape: {X.shape} | Target vector y shape: {y.shape}")
    
    with open(FEATURES_PKL, "wb") as f:
        pickle.dump({"X": X, "y": y}, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✓ Output training files compiled safely at: '{FEATURES_PKL}'\n")

if __name__ == "__main__":
    main()