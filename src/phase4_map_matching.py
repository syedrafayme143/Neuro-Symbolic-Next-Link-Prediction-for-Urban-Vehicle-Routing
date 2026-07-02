"""
phase4_map_matching.py
=======================
Phase 4 of the Semantic Road Graph-Based Next-Link Prediction project.

Implements a refined, high-performance cross-track geometric map-matching baseline 
evaluating with exact full-key matching and saving results to disk. Optimized to 
prevent CPU hanging.
"""

import json
import os
import pickle
import time
import networkx as nx
import numpy as np
import osmnx as ox

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
GRAPH_PKL        = os.path.join("data", "raw",       "ingolstadt_graph.pkl")
NOISY_GPS_JSON  = os.path.join("data", "synthetic", "noisy_gps_data.json")
PROCESSED_DIR   = os.path.join("data", "processed")
OUTPUT_METRICS  = os.path.join(PROCESSED_DIR, "map_matching_metrics.json")

NOISE_LEVELS = ["low_5m", "medium_10m", "high_20m"]
FALLBACK_MAX_DISTANCE_M = 35.0
MAX_TRAJECTORIES = None   

# ---------------------------------------------------------------------------
# FAST MATH HELPERS (Vectorized & Highly Performant)
# ---------------------------------------------------------------------------

def load_graph(path: str) -> nx.MultiDiGraph:
    with open(path, "rb") as f:
        G = pickle.load(f)
    return G

def load_noisy_data(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_edge_index(G: nx.MultiDiGraph):
    """
    Pre-computes adjacency along with midpoints of roads for lightning-fast 
    spatial calculations without constructing heavy Shapely geometry objects in loops.
    """
    out_edges = {}
    for u, v, k, data in G.edges(keys=True, data=True):
        # Calculate a fast geometric midpoint proxy for the road segment
        if "geometry" in data:
            coords = list(data["geometry"].coords)
            mid_idx = len(coords) // 2
            mid_lon, mid_lat = coords[mid_idx]
        else:
            mid_lat = (G.nodes[u]['y'] + G.nodes[v]['y']) / 2.0
            mid_lon = (G.nodes[u]['x'] + G.nodes[v]['x']) / 2.0
            
        out_edges.setdefault(u, []).append((v, k, mid_lat, mid_lon))
    return out_edges

def is_connected(prev_edge, cand_edge):
    if prev_edge is None:
        return True
    _, prev_v, _ = prev_edge
    cand_u, _, _ = cand_edge
    return prev_v == cand_u or prev_edge[:2] == cand_edge[:2]

def haversine_m(lat1, lon1, lat2, lon2):
    """Fast Haversine formula for distance in meters."""
    R = 6371000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

def topological_fallback(G, out_edges, prev_edge, cand_edge, ping_lat, ping_lon):
    """
    Evaluates fallback candidate roads using pre-computed segment midpoints 
    to guarantee real-time execution speeds.
    """
    if is_connected(prev_edge, cand_edge):
        return cand_edge, False  

    if prev_edge is None:
        return cand_edge, False

    _, prev_v, _ = prev_edge
    neighbours = out_edges.get(prev_v, [])

    if not neighbours:
        return cand_edge, False  

    best_alt = None
    best_dist = float("inf")

    # Fast loop over pre-indexed primitive floats (no object overhead)
    for (nv, nk, mid_lat, mid_lon) in neighbours:
        dist = haversine_m(ping_lat, ping_lon, mid_lat, mid_lon)
        if dist < best_dist:
            best_dist = dist
            best_alt = (prev_v, nv, nk)

    if best_alt is not None and best_dist <= FALLBACK_MAX_DISTANCE_M:
        return best_alt, True  

    return cand_edge, False

def match_trajectory(G, out_edges, lats, lons):
    """Vectorized bulk spatial query combined with fast fallback routing."""
    X = np.asarray(lons)
    Y = np.asarray(lats)
    
    # OSMnx native nearest_edges uses a highly optimized internal spatial R-Tree index
    nearest = ox.distance.nearest_edges(G, X=X, Y=Y)  

    matched_edges = []
    num_fallbacks = 0
    prev_edge = None

    for i, cand in enumerate(nearest):
        cand_edge = tuple(cand)  
        final_edge, fell_back = topological_fallback(
            G, out_edges, prev_edge, cand_edge, lats[i], lons[i]
        )
        if fell_back:
            num_fallbacks += 1

        matched_edges.append(final_edge)
        prev_edge = final_edge

    return matched_edges, num_fallbacks

# ---------------------------------------------------------------------------
# MAIN EVALUATION PIPELINE
# ---------------------------------------------------------------------------

def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    G = load_graph(GRAPH_PKL)
    out_edges = build_edge_index(G)
    data = load_noisy_data(NOISY_GPS_JSON)

    trajectories = data["trajectories"]
    if MAX_TRAJECTORIES is not None:
        trajectories = trajectories[:MAX_TRAJECTORIES]

    results = {level: [] for level in NOISE_LEVELS}
    fallback_counts = {level: 0 for level in NOISE_LEVELS}
    
    print(f"\n[Phase 4] Running optimized point-aligned Map-Matching sandbox...")
    start_time = time.time()

    for level in NOISE_LEVELS:
        print(f"[Phase 4] ── Processing: {level} ──────────────────")
        level_start = time.time()
        
        for idx, traj in enumerate(trajectories):
            true_point_labels = [tuple(edge) for edge in traj["point_labels"]]

            pings = traj["noisy_gps"][level]
            lats = [p[0] for p in pings]
            lons = [p[1] for p in pings]

            # Match point sequences
            matched_edges, n_fb = match_trajectory(G, out_edges, lats, lons)
            fallback_counts[level] += n_fb

            # Metric evaluation comparing full keys (u, v, key)
            correct_points = sum(1 for i in range(len(true_point_labels)) if true_point_labels[i] == matched_edges[i])
            acc = correct_points / len(true_point_labels)
            results[level].append(acc)

            # Progress log every 1000 paths
            if (idx + 1) % 1000 == 0:
                print(f"[Phase 4]   Computed {idx + 1:,} / {len(trajectories):,} trajectories...")

        print(f"[Phase 4] Finished {level} in {time.time() - level_start:.1f} seconds.")

    # ------------------------------------------------------------------
    # SAVE METRICS TO DISK & SHOW TABLE
    # ------------------------------------------------------------------
    metrics_summary = {}
    
    print("\n[Phase 4] ══════════════════════════════════════════════════════")
    print("[Phase 4]            MAP-MATCHING ACCURACY SUMMARY MATRIX")
    print("[Phase 4] ══════════════════════════════════════════════════════")
    print(f"[Phase 4] {'Noise Level':<14} | {'Mean Top-1 Acc':>15} | {'Std Dev':>8} | {'Fallbacks':>10}")
    print(f"[Phase 4] {'-'*14}-+-{'-'*15}-+-{'-'*8}-+-{'-'*10}")
    
    for level in NOISE_LEVELS:
        accs = np.array(results[level], dtype=float)
        mean_acc = float(np.nanmean(accs))
        std_acc = float(np.nanstd(accs))
        
        print(f"[Phase 4] {level:<14} | {mean_acc:>14.3f}  | {std_acc:>7.3f}  | {fallback_counts[level]:>10,}")
        
        metrics_summary[level] = {
            "mean_accuracy": round(mean_acc, 4),
            "std_deviation": round(std_acc, 4),
            "total_fallbacks_triggered": fallback_counts[level]
        }
    print("[Phase 4] ══════════════════════════════════════════════════════")

    with open(OUTPUT_METRICS, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"[Phase 4] ✓ Performance evaluation metrics saved out to '{OUTPUT_METRICS}'")
    print(f"[Phase 4] Total execution time: {time.time() - start_time:.1f} seconds.\n")

if __name__ == "__main__":
    main()