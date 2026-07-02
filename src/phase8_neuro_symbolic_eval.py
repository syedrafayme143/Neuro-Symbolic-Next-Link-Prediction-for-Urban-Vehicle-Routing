"""
phase8_neuro_symbolic_eval.py
===============================
Phase 8 of the Semantic Road Graph-Based Next-Link Prediction project.

Implements a Neuro-Symbolic Veto Filter: a post-hoc correction layer that
checks every "neural" model's Top-1 prediction against the legal road
topology encoded in the RDF Knowledge Graph (Phase 4.5), and substitutes
the highest-ranked LEGAL choice from that model's Top-3 list.

Refined with path auto-discovery and exception safety hooks for RDF loading.
"""

import json
import os
import pickle
import random
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from rdflib import Graph as RDFGraph, Namespace

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
GRAPH_PKL        = os.path.join("data", "raw",       "ingolstadt_graph.pkl")
FEATURES_PKL     = os.path.join("data", "processed", "final_ml_features.pkl")
TTL_NAME         = "ingolstadt_semantic_map.ttl"
PHASE7_JSON      = os.path.join("data", "processed", "phase7_model_benchmark.json")
NOISY_GPS_JSON   = os.path.join("data", "synthetic", "noisy_gps_data.json")
PROCESSED_DIR    = os.path.join("data", "processed")
OUTPUT_JSON      = os.path.join(PROCESSED_DIR, "phase8_neuro_symbolic_metrics.json")

RANDOM_SEED          = 42
TRAIN_GROUP_FRACTION = 0.8

NON_PREDICTIVE_COLS = ["traj_id"]

ONTOLOGY_URI = "http://smartmobility.bmw.org/ontology#"
EX = Namespace(ONTOLOGY_URI)

# Must mirror Phase 7 exactly for valid delta reporting
RF_PARAMS = dict(
    n_estimators=20,
    max_depth=10,
    min_samples_split=10,
    max_samples=0.1,
    n_jobs=2,
    random_state=RANDOM_SEED,
)

LSTM_HIDDEN_SIZE  = 128
LSTM_NUM_LAYERS   = 2
LSTM_DROPOUT      = 0.2
LSTM_BATCH_SIZE   = 64
LSTM_EPOCHS       = 5
LSTM_LR           = 1e-3
LSTM_MAX_SEQ_LEN  = 30


# ===========================================================================
# SHARED UTILITIES 
# ===========================================================================

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_features(path: str):
    print(f"[Phase 8] Loading engineered feature dataset from '{path}' …")
    with open(path, "rb") as f:
        data = pickle.load(f)
    X, y = data["X"], data["y"]
    print(f"[Phase 8] Loaded  →  X: {X.shape}  |  y: {y.shape}  |  unique classes: {y.nunique():,}")
    return X, y


def load_nx_graph(path: str) -> nx.MultiDiGraph:
    print(f"[Phase 8] Loading road graph from '{path}' (used for veto fallback) …")
    with open(path, "rb") as f:
        G = pickle.load(f)
    print(f"[Phase 8] Graph loaded  →  nodes: {G.number_of_nodes():,}  |  edges: {G.number_of_edges():,}")
    return G


def group_train_test_split(X: pd.DataFrame, y: pd.Series, group_col: str,
                            train_fraction: float, seed: int):
    print(f"\n[Phase 8] Reproducing Phase 7's GROUP-AWARE train/test split on '{group_col}' …")

    unique_groups = X[group_col].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)

    n_train_groups = int(len(unique_groups) * train_fraction)
    train_groups = set(unique_groups[:n_train_groups])
    test_groups = set(unique_groups[n_train_groups:])

    train_mask = X[group_col].isin(train_groups)
    test_mask = X[group_col].isin(test_groups)

    print(f"[Phase 8]   Total trajectories      : {len(unique_groups):,}")
    print(f"[Phase 8]   Train trajectories ({train_fraction:.0%})  : {len(train_groups):,}  → {train_mask.sum():,} rows")
    print(f"[Phase 8]   Test trajectories ({1-train_fraction:.0%})   : {len(test_groups):,}  → {test_mask.sum():,} rows")

    overlap = train_groups & test_groups
    assert len(overlap) == 0, "CRITICAL: trajectory leakage detected between train/test groups!"
    return train_mask.values, test_mask.values, train_groups, test_groups


def drop_non_predictive(X: pd.DataFrame) -> pd.DataFrame:
    identity_cols = ["u", "v", "step", "point_step"]
    cols_to_drop = [c for c in (NON_PREDICTIVE_COLS + identity_cols) if c in X.columns]
    if cols_to_drop:
        print(f"[Phase 8] Dropping non-predictive/identifier columns: {cols_to_drop}")
    return X.drop(columns=cols_to_drop, errors="ignore")


def top_k_idx_matrix(proba: np.ndarray, k: int) -> np.ndarray:
    k = min(k, proba.shape[1])
    unordered = np.argpartition(proba, -k, axis=1)[:, -k:]
    row_idx = np.arange(proba.shape[0])[:, None]
    order = np.argsort(-proba[row_idx, unordered], axis=1)
    return unordered[row_idx, order]


# ===========================================================================
# RE-TRAINING METHODS
# ===========================================================================

def train_random_forest(X_train, y_train_enc):
    print("\n[Phase 8] [Random Forest] Re-training with Phase 7's exact hyperparameters …")
    X_train_f = X_train.astype({c: "float32" for c in X_train.select_dtypes(include="bool").columns})

    model = RandomForestClassifier(**RF_PARAMS)
    start = time.time()
    model.fit(X_train_f, y_train_enc)
    print(f"[Phase 8] [Random Forest] ✓ Re-trained in {time.time() - start:.1f}s")
    return model


class TrajectorySequenceDataset(Dataset):
    def __init__(self, X: pd.DataFrame, y_enc: np.ndarray, traj_ids: pd.Series, max_seq_len: int):
        self.max_seq_len = max_seq_len
        X_clean = X.astype({c: "float32" for c in X.select_dtypes(include="bool").columns})
        self.n_features = X_clean.shape[1]

        step_col = "point_step" if "point_step" in X_clean.columns else "step"
        step_values = X_clean[step_col].values if step_col in X_clean.columns else np.arange(len(X_clean))

        sort_df = pd.DataFrame({"traj_id": traj_ids.values, "point_step": step_values})
        sorted_indices = sort_df.sort_values(by=["traj_id", "point_step"]).index.to_numpy()

        feature_values = X_clean.values.astype(np.float32)[sorted_indices]
        y_enc_sorted = y_enc[sorted_indices]
        traj_ids_sorted = traj_ids.values[sorted_indices]

        self.original_row_index = np.asarray(sorted_indices)

        groups = defaultdict(list)
        for pos, tid in enumerate(traj_ids_sorted):
            groups[tid].append(pos)

        self.sequences, self.labels, self.lengths, self.orig_idx_per_seq = [], [], [], []
        for tid, positions in groups.items():
            seq_len = min(len(positions), max_seq_len)
            positions = positions[:seq_len]
            self.sequences.append(feature_values[positions])
            self.labels.append(y_enc_sorted[positions])
            self.lengths.append(seq_len)
            self.orig_idx_per_seq.append(self.original_row_index[positions])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq, labels, length = self.sequences[idx], self.labels[idx], self.lengths[idx]
        padded_seq = np.zeros((self.max_seq_len, self.n_features), dtype=np.float32)
        padded_labels = np.full((self.max_seq_len,), -100, dtype=np.int64)
        padded_seq[:length] = seq
        padded_labels[:length] = labels
        return torch.from_numpy(padded_seq), torch.from_numpy(padded_labels), length, idx


class LSTMNextEdgeClassifier(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, num_classes, dropout):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_head = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = torch.relu(self.input_proj(x))
        lstm_out, _ = self.lstm(x)
        return self.output_head(lstm_out)


def train_lstm(X_train, y_train_enc, train_traj_ids, num_classes, device):
    print("\n[Phase 8] [LSTM] Re-training sequential neural networks …")

    train_dataset = TrajectorySequenceDataset(X_train, y_train_enc, train_traj_ids, LSTM_MAX_SEQ_LEN)
    train_loader = DataLoader(train_dataset, batch_size=LSTM_BATCH_SIZE, shuffle=True)

    model = LSTMNextEdgeClassifier(
        input_dim=X_train.shape[1], hidden_size=LSTM_HIDDEN_SIZE, num_layers=LSTM_NUM_LAYERS,
        num_classes=num_classes, dropout=LSTM_DROPOUT,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LSTM_LR)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    model.train()
    for epoch in range(LSTM_EPOCHS):
        epoch_start = time.time()
        total_loss, n_batches = 0.0, 0
        for batch_seq, batch_labels, _lengths, _idx in train_loader:
            batch_seq, batch_labels = batch_seq.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            logits = model(batch_seq)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), batch_labels.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        print(f"[Phase 8] [LSTM]   Epoch {epoch + 1}/{LSTM_EPOCHS}  |  avg loss: {total_loss / max(n_batches,1):.4f}  |  {time.time() - epoch_start:.1f}s")

    return model


# ===========================================================================
# NEURO-SYMBOLIC VETO ENGINE
# ===========================================================================

class NeuroSymbolicVetoEngine:
    def __init__(self, rdf_graph: RDFGraph, nx_graph: nx.MultiDiGraph):
        self.rdf_graph = rdf_graph
        self.nx_graph = nx_graph
        self._legal_cache = {}      
        self._fallback_cache = {}   
        self.cache_hits = 0
        self.cache_misses = 0

    def _query_legal_next_edges(self, v: int) -> set:
        node_uri = f"{ONTOLOGY_URI}Intersection_{v}"
        query = f"""
        PREFIX ex: <{ONTOLOGY_URI}>
        SELECT ?road
        WHERE {{
            ?road ex:startsFrom <{node_uri}> .
        }}
        """
        results = self.rdf_graph.query(query)
        legal_labels = set()
        road_prefix = f"{ONTOLOGY_URI}Road_"
        for row in results:
            road_uri = str(row.road)
            if road_uri.startswith(road_prefix):
                legal_labels.add(road_uri[len(road_prefix):])
        return legal_labels

    def get_legal_next_edges(self, v: int) -> set:
        if v in self._legal_cache:
            self.cache_hits += 1
            return self._legal_cache[v]

        self.cache_misses += 1
        legal_set = self._query_legal_next_edges(v)
        self._legal_cache[v] = legal_set
        return legal_set

    def get_shortest_hop_fallback(self, v: int):
        if v in self._fallback_cache:
            return self._fallback_cache[v]

        best_label, best_length = None, float("inf")
        if v in self.nx_graph:
            for _, nv, nk, data in self.nx_graph.edges(v, keys=True, data=True):
                length = float(data.get("length", 1.0))
                if length < best_length:
                    best_length = length
                    best_label = f"{v}_{nv}_{nk}"

        self._fallback_cache[v] = best_label
        return best_label

    def veto_correct(self, current_v: int, top3_predictions: list):
        legal_set = self.get_legal_next_edges(current_v)
        raw_top1 = top3_predictions[0] if top3_predictions else None

        if raw_top1 is not None and raw_top1 in legal_set:
            return raw_top1, top3_predictions, False, False

        for candidate in top3_predictions:
            if candidate is not None and candidate in legal_set:
                corrected_top3 = [candidate] + [p for p in top3_predictions if p != candidate]
                return candidate, corrected_top3, True, False

        fallback = self.get_shortest_hop_fallback(current_v)
        corrected_top3 = [fallback] + [p for p in top3_predictions if p != fallback]
        return fallback, corrected_top3, True, True


def load_all_v_nodes_from_json(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    v_list = []
    for traj in data["trajectories"]:
        noisy_pings = traj["noisy_gps"]["low_5m"]
        labels = [tuple(lbl) for lbl in traj["point_labels"]]
        num_points = len(noisy_pings)
        if num_points < 3: continue
            
        for t in range(num_points):
            next_edge_target = None
            for look_ahead in range(t + 1, num_points):
                if labels[look_ahead] != labels[t]:
                    next_edge_target = f"{labels[look_ahead][0]}_{labels[look_ahead][1]}_{labels[look_ahead][2]}"
                    break
            if next_edge_target is None: continue
            
            _, v, _ = labels[t]
            v_list.append(v)
    return v_list


def load_rdf_graph(filename: str) -> RDFGraph:
    """
    FIXED: Implements path auto-discovery to prevent FileNotFoundError and 
    syntax handling safety wrappers when rdflib boots up.
    """
    # Define possible candidate locations where the .ttl file could be residing
    candidate_paths = [
        os.path.join("data", "processed", filename),
        os.path.join("data", "raw", filename),
        os.path.join("data", filename)
    ]
    
    actual_path = None
    for path in candidate_paths:
        if os.path.exists(path):
            actual_path = path
            break
            
    if actual_path is None:
        raise FileNotFoundError(
            f"\n[Phase 8] CRITICAL ERROR: Could not locate '{filename}' anywhere!\n"
            f"  Searched locations:\n"
            f"    1. {candidate_paths[0]}\n"
            f"    2. {candidate_paths[1]}\n"
            f"    3. {candidate_paths[2]}\n"
            f"  FIX: Please verify that you generated the Knowledge Graph in Phase 4.5.\n"
            f"  If it exists under a different name, rename it to '{filename}' or move it into 'data/processed/'."
        )

    print(f"\n[Phase 8] Loading RDF Knowledge Graph from '{actual_path}' …")
    g = RDFGraph()
    start = time.time()
    try:
        g.parse(actual_path, format="turtle")
    except Exception as e:
        raise RuntimeError(f"[Phase 8] Graph Parsing Failed! Check for bad syntax inside '{actual_path}'. Error details: {e}")
        
    print(f"[Phase 8] ✓ RDF graph loaded  →  {len(g):,} triples in {time.time() - start:.1f}s")
    return g


def evaluate_model_with_veto(model_name: str, top3_label_lists: list, y_test_raw: list,
                             current_v_array: np.ndarray, veto_engine: NeuroSymbolicVetoEngine):
    print(f"\n[Phase 8] Passing validations through Knowledge Graph layer: {model_name} …")

    n = len(top3_label_lists)
    pre_top1_correct = pre_top3_correct = post_top1_correct = post_top3_correct = 0
    n_vetoed = n_fallback_used = 0

    for i in range(n):
        top3 = top3_label_lists[i]
        true_label = y_test_raw[i]
        v = int(current_v_array[i])

        if top3 and top3[0] == true_label: pre_top1_correct += 1
        if true_label in top3: pre_top3_correct += 1

        corrected_top1, corrected_top3, was_vetoed, used_fallback = veto_engine.veto_correct(v, top3)

        if was_vetoed: n_vetoed += 1
        if used_fallback: n_fallback_used += 1

        if corrected_top1 == true_label: post_top1_correct += 1
        if true_label in corrected_top3: post_top3_correct += 1

        if (i + 1) % 100_000 == 0:
            print(f"[Phase 8]   …{i + 1:,}/{n:,} rows evaluated (Cache Hits: {veto_engine.cache_hits:,})")

    pre_top1, pre_top3 = pre_top1_correct / n, pre_top3_correct / n
    post_top1, post_top3 = post_top1_correct / n, post_top3_correct / n
    veto_rate = n_vetoed / n

    print(f"[Phase 8] [{model_name}] Pre-veto  Top-1 Acc: {pre_top1:.4f} | Top-3 Acc: {pre_top3:.4f}")
    print(f"[Phase 8] [{model_name}] Post-veto Top-1 Acc: {post_top1:.4f} | Top-3 Acc: {post_top3:.4f}")
    print(f"[Phase 8] [{model_name}] Veto rate: {veto_rate:.4%} ({n_vetoed:,} anomalies corrected)")

    return {
        "pre_kg_top1_accuracy":  round(pre_top1, 4),
        "pre_kg_top3_accuracy":  round(pre_top3, 4),
        "post_kg_top1_accuracy": round(post_top1, 4),
        "post_kg_top3_accuracy": round(post_top3, 4),
        "veto_rate":              round(veto_rate, 4),
    }


def get_top3_labels_rf(model, X_test, label_encoder, batch_size=20000) -> list:
    print("[Phase 8] [Random Forest] Generating tabular matrix predictions…")
    X_test_f = X_test.astype({c: "float32" for c in X_test.select_dtypes(include="bool").columns})
    all_top3 = []
    total = X_test_f.shape[0]
    for i in range(0, total, batch_size):
        batch = X_test_f.iloc[i:i + batch_size]
        proba = model.predict_proba(batch)
        top3_idx = top_k_idx_matrix(proba, k=3)
        for row in top3_idx:
            all_top3.append(list(label_encoder.inverse_transform(row)))
    return all_top3


def get_top3_labels_lstm(model, X_test_with_v, y_test_enc, test_traj_ids, label_encoder, device) -> tuple:
    print("[Phase 8] [LSTM] Generating trajectory batch matrix predictions…")
    feature_cols = [c for c in X_test_with_v.columns if c != "v"]
    X_model_input = X_test_with_v[feature_cols]

    dataset = TrajectorySequenceDataset(X_model_input, y_test_enc, test_traj_ids, LSTM_MAX_SEQ_LEN)
    loader = DataLoader(dataset, batch_size=LSTM_BATCH_SIZE, shuffle=False)

    edge_v_values = X_test_with_v["v"].astype(int).to_numpy()
    y_test_raw_values = label_encoder.inverse_transform(y_test_enc)

    model.eval()
    top3_by_orig_idx = {}

    with torch.no_grad():
        for batch_seq, batch_labels, batch_lengths, batch_seq_idx in loader:
            batch_seq = batch_seq.to(device)
            logits = model(batch_seq)

            for b in range(batch_seq.shape[0]):
                seq_idx = batch_seq_idx[b].item()
                length = batch_lengths[b].item()
                orig_indices = dataset.orig_idx_per_seq[seq_idx]

                seq_logits = logits[b, :length]
                proba = torch.softmax(seq_logits, dim=-1).cpu().numpy()
                top3_idx = top_k_idx_matrix(proba, k=3)

                for pos in range(length):
                    orig_row = orig_indices[pos]
                    labels_for_row = list(label_encoder.inverse_transform(top3_idx[pos]))
                    top3_by_orig_idx[orig_row] = labels_for_row

    n_rows = len(X_test_with_v)
    top3_label_lists = [top3_by_orig_idx.get(i, [None, None, None]) for i in range(n_rows)]
    return top3_label_lists, edge_v_values, list(y_test_raw_values)


def print_delta_table(phase7_metrics: dict, phase8_results: dict):
    print("\n[Phase 8] ══════════════════════════════════════════════════════════════════")
    print("[Phase 8]            NEURO-SYMBOLIC VETO — PERFORMANCE DELTA REPORT")
    print("[Phase 8] ══════════════════════════════════════════════════════════════════")
    print(f"[Phase 8] {'Model':<24} | {'Raw Top-1':>10} | {'+KG Top-1':>10} | "
          f"{'Raw Top-3':>10} | {'+KG Top-3':>10} | {'Veto Rate':>10}")
    print(f"[Phase 8] {'-'*24}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for model_name, metrics in phase8_results.items():
        print(f"[Phase 8] {model_name:<24} | {metrics['pre_kg_top1_accuracy']*100:>9.2f}% | {metrics['post_kg_top1_accuracy']*100:>9.2f}% | {metrics['pre_kg_top3_accuracy']*100:>9.2f}% | {metrics['post_kg_top3_accuracy']*100:>9.2f}% | {metrics['veto_rate']*100:>9.2f}%")
    print("[Phase 8] ══════════════════════════════════════════════════════════════════\n")


# ===========================================================================
# MAIN EXECUTIVE ORCHESTRATOR
# ===========================================================================

def main():
    set_all_seeds(RANDOM_SEED)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("═" * 70)
    print("  PHASE 8 — NEURO-SYMBOLIC VETO EVALUATION")
    print("═" * 70)

    X_full, y_full_raw = load_features(FEATURES_PKL)
    nx_graph = load_nx_graph(GRAPH_PKL)
    
    # FIXED: Runs through discovery pathways to load the RDF file safely
    rdf_graph = load_rdf_graph(TTL_NAME)

    if os.path.exists(PHASE7_JSON):
        with open(PHASE7_JSON, "r", encoding="utf-8") as f:
            phase7_metrics = json.load(f)
    else:
        phase7_metrics = {}

    print("[Phase 8] Dynamically reconstructing current intersection 'v' nodes from logs...")
    X_full["v"] = load_all_v_nodes_from_json(NOISY_GPS_JSON)
    X_full["step"] = X_full["point_step"] if "point_step" in X_full.columns else X_full["step"]

    train_mask, test_mask, train_groups, test_groups = group_train_test_split(
        X_full, y_full_raw, group_col="traj_id", train_fraction=TRAIN_GROUP_FRACTION, seed=RANDOM_SEED,
    )

    train_traj_ids = X_full.loc[train_mask, "traj_id"].reset_index(drop=True)
    test_traj_ids  = X_full.loc[test_mask, "traj_id"].reset_index(drop=True)

    X_train_raw = X_full.loc[train_mask].reset_index(drop=True)
    X_test_raw  = X_full.loc[test_mask].reset_index(drop=True)
    y_train_raw = y_full_raw.loc[train_mask].reset_index(drop=True)
    y_test_raw  = y_full_raw.loc[test_mask].reset_index(drop=True)

    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train_raw)
    num_classes = len(label_encoder.classes_)

    seen_classes = set(label_encoder.classes_)
    test_keep_mask = y_test_raw.isin(seen_classes)

    X_test_filtered = X_test_raw[test_keep_mask].reset_index(drop=True)
    y_test_raw_filtered = y_test_raw[test_keep_mask].reset_index(drop=True)
    test_traj_ids_filtered = test_traj_ids[test_keep_mask].reset_index(drop=True)
    y_test_enc = label_encoder.transform(y_test_raw_filtered)

    current_v_array = X_test_filtered["v"].astype(int).to_numpy()

    X_train_clean = drop_non_predictive(X_train_raw)
    X_test_clean  = drop_non_predictive(X_test_filtered)

    rf_model = train_random_forest(X_train_clean, y_train_enc)
    lstm_model = train_lstm(X_train_clean, y_train_enc, train_traj_ids, num_classes, device)

    veto_engine = NeuroSymbolicVetoEngine(rdf_graph, nx_graph)

    rf_top3 = get_top3_labels_rf(rf_model, X_test_clean, label_encoder)
    rf_results = evaluate_model_with_veto("Random Forest", rf_top3, list(y_test_raw_filtered), current_v_array, veto_engine)

    X_test_for_lstm = X_test_clean.copy()
    X_test_for_lstm["v"] = X_test_filtered["v"].values

    lstm_top3, lstm_current_v, lstm_y_true = get_top3_labels_lstm(
        lstm_model, X_test_for_lstm, y_test_enc, test_traj_ids_filtered, label_encoder, device,
    )

    valid_idx = [i for i, p in enumerate(lstm_top3) if p[0] is not None]
    lstm_top3_valid = [lstm_top3[i] for i in valid_idx]
    lstm_v_valid = np.array([lstm_current_v[i] for i in valid_idx])
    lstm_y_valid = [lstm_y_true[i] for i in valid_idx]

    lstm_results = evaluate_model_with_veto("LSTM", lstm_top3_valid, lstm_y_valid, lstm_v_valid, veto_engine)

    phase8_results = {
        "Random Forest": rf_results,
        "LSTM": lstm_results,
    }

    print_delta_table(phase7_metrics, phase8_results)

    output = {
        "phase7_baseline_reference": phase7_metrics,
        "phase8_neuro_symbolic_results": phase8_results,
        "veto_engine_cache_stats": {
            "unique_intersections_queried": veto_engine.cache_misses,
            "cache_hits": veto_engine.cache_hits,
        }
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"[Phase 8] Saved neuro-symbolic metrics  →  {OUTPUT_JSON}\n✓ Phase 8 complete.")


if __name__ == "__main__":
    main()