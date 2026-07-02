"""
phase7_train_models.py
========================
Phase 7 of the Semantic Road Graph-Based Next-Link Prediction project.

Trains and benchmarks three separate next-link prediction baselines:
  1. Rule-Based Transition Lookup (Extracted directly from JSON logs)
  2. Random Forest Tabular Classifier (Resource-optimized for consumer laptops)
  3. LSTM Recurrent Sequence Classifier (PyTorch sequence-to-sequence)

Optimized with row bootstrapping, capped CPU threads, and structural constraints 
to guarantee execution safety on standard laptops without thermal shutdown.
"""

import json
import os
import pickle
import random
import time
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import networkx as nx

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
GRAPH_PKL      = os.path.join("data", "raw",       "ingolstadt_graph.pkl")
FEATURES_PKL   = os.path.join("data", "processed", "final_ml_features.pkl")
NOISY_GPS_JSON = os.path.join("data", "synthetic", "noisy_gps_data.json")
PROCESSED_DIR  = os.path.join("data", "processed")
BENCHMARK_JSON = os.path.join(PROCESSED_DIR, "phase7_model_benchmark.json")

RANDOM_SEED   = 42
TRAIN_GROUP_FRACTION = 0.8     

NON_PREDICTIVE_COLS = ["traj_id", "point_step"]

# LAPTOP-SAFE OPTIMIZED PARAMETERS (Prevents OOM Crashes and System Meltdowns)
RF_PARAMS = dict(
    n_estimators=20,          # Reduced from 50 to shorten sustained heat generation windows
    max_depth=10,             # Capped depth to restrict heavy tree array allocations
    min_samples_split=10,
    max_samples=0.1,          # LAPTOP LIFE-SAVER: Uses only 10% of rows per tree (cuts RAM/CPU load by 90%)
    n_jobs=2,                 # FIXED: Never use -1 on standard laptops. Restrict to 2 cores for cooling safety.
    random_state=RANDOM_SEED,
)

# LSTM hyperparameters
LSTM_HIDDEN_SIZE   = 128
LSTM_NUM_LAYERS    = 2
LSTM_DROPOUT       = 0.2
LSTM_BATCH_SIZE    = 64
LSTM_EPOCHS        = 5        
LSTM_LR            = 1e-3
LSTM_MAX_SEQ_LEN   = 30      


# ===========================================================================
# SHARED UTILITIES
# ===========================================================================

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_features(path: str):
    print(f"[Phase 7] Loading engineered feature dataset from '{path}' …")
    with open(path, "rb") as f:
        data = pickle.load(f)
    X, y = data["X"], data["y"]
    print(f"[Phase 7] Loaded  →  X: {X.shape}  |  y: {y.shape}  |  unique classes: {y.nunique():,}")
    return X, y


def load_graph(path: str) -> nx.MultiDiGraph:
    print(f"[Phase 7] Loading road graph from '{path}' for rule-based fallback …")
    with open(path, "rb") as f:
        G = pickle.load(f)
    return G


def group_train_test_split(X: pd.DataFrame, y: pd.Series, group_col: str,
                            train_fraction: float, seed: int):
    print(f"\n[Phase 7] Performing GROUP-AWARE train/test split on '{group_col}' …")

    unique_groups = X[group_col].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)

    n_train_groups = int(len(unique_groups) * train_fraction)
    train_groups = set(unique_groups[:n_train_groups])
    test_groups = set(unique_groups[n_train_groups:])

    train_mask = X[group_col].isin(train_groups)
    test_mask = X[group_col].isin(test_groups)

    print(f"[Phase 7]   Total trajectories      : {len(unique_groups):,}")
    print(f"[Phase 7]   Train trajectories ({train_fraction:.0%})  : {len(train_groups):,}  → {train_mask.sum():,} rows")
    print(f"[Phase 7]   Test trajectories ({1-train_fraction:.0%})   : {len(test_groups):,}  → {test_mask.sum():,} rows")

    overlap = train_groups & test_groups
    assert len(overlap) == 0, "CRITICAL: trajectory leakage detected between train/test groups!"
    return train_mask.values, test_mask.values, train_groups, test_groups


def drop_non_predictive(X: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [c for c in NON_PREDICTIVE_COLS if c in X.columns]
    if cols_to_drop:
        print(f"[Phase 7] Dropping non-predictive/identifier columns: {cols_to_drop}")
    return X.drop(columns=cols_to_drop, errors="ignore")


def top_k_correct_count(y_true_idx: np.ndarray, proba: np.ndarray, k: int) -> int:
    if k >= proba.shape[1]:
        return len(y_true_idx)

    top_k_idx = np.argpartition(proba, -k, axis=1)[:, -k:]
    correct = (top_k_idx == y_true_idx[:, None]).any(axis=1)
    return int(correct.sum())


def print_benchmark_table(results: dict):
    print("\n[Phase 7] ══════════════════════════════════════════════════════")
    print("[Phase 7]             FINAL MODEL BENCHMARK SUMMARY")
    print("[Phase 7] ══════════════════════════════════════════════════════")
    print(f"[Phase 7] {'Model':<28} | {'Top-1 Acc':>10} | {'Top-3 Acc':>10} | {'Train Time (s)':>15}")
    print(f"[Phase 7] {'-'*28}-+-{'-'*10}-+-{'-'*10}-+-{'-'*15}")
    for name, metrics in results.items():
        print(f"[Phase 7] {name:<28} | {metrics['top1_accuracy']:>10.4f} | "
              f"{metrics['top3_accuracy']:>10.4f} | {metrics['train_time_s']:>15.1f}")
    print("[Phase 7] ══════════════════════════════════════════════════════\n")


# ===========================================================================
# MODEL 1 — RULE-BASED TRANSITION BASELINE
# ===========================================================================

class RuleBasedTransitionModel:
    def __init__(self, graph: nx.MultiDiGraph, seed: int = RANDOM_SEED):
        self.graph = graph
        self.rng = random.Random(seed)
        self.transition_counts = defaultdict(Counter)   
        self.global_label_pool = []                     
        self._fitted = False

    def fit(self, edge_keys: list, y_train: list):
        start = time.time()
        for current_edge, next_label in zip(edge_keys, y_train):
            self.transition_counts[current_edge][next_label] += 1
        self.global_label_pool = list(y_train)
        self._fitted = True
        return time.time() - start

    def _graph_fallback_neighbours(self, u, v):
        if v not in self.graph:
            return []
        candidates = []
        for _, nv, nk, data in self.graph.edges(v, keys=True, data=True):
            length = float(data.get("length", 1.0))
            label = f"{v}_{nv}_{nk}"
            candidates.append((label, length))
        candidates.sort(key=lambda x: x[1])
        return candidates

    def predict_topk(self, edge_keys: list, k: int = 3) -> list:
        assert self._fitted, "Must call .fit() before predicting"
        predictions = []

        for (u, v, key) in edge_keys:
            counter = self.transition_counts.get((u, v, key))
            if counter:
                ranked = [label for label, _ in counter.most_common(k)]
            else:
                ranked = []

            if len(ranked) < k:
                fallback_candidates = self._graph_fallback_neighbours(u, v)
                for label, _ in fallback_candidates:
                    if label not in ranked:
                        ranked.append(label)
                    if len(ranked) >= k:
                        break

            if len(ranked) < k and self.global_label_pool:
                while len(ranked) < k:
                    pick = self.rng.choice(self.global_label_pool)
                    if pick not in ranked:
                        ranked.append(pick)

            predictions.append(ranked[:k] if ranked else [None] * k)
        return predictions


def load_alignment_keys_from_json(path: str, subset_traj_ids: set):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    keys_list = []
    for traj in data["trajectories"]:
        if int(traj["traj_id"]) in subset_traj_ids:
            pings_count = len(traj["noisy_gps"]["low_5m"])
            labels = [tuple(lbl) for lbl in traj["point_labels"]]
            
            for t in range(pings_count):
                current_edge = labels[t]
                has_next = False
                for look_ahead in range(t + 1, pings_count):
                    if labels[look_ahead] != current_edge:
                        has_next = True
                        break
                if has_next:
                    keys_list.append(current_edge)
    return keys_list


def evaluate_rule_based_model(y_train_raw, y_test_raw, train_groups, test_groups, graph):
    print("\n" + "─" * 70)
    print("  MODEL 1 — RULE-BASED TRANSITION BASELINE")
    print("─" * 70)

    print("[Phase 7] Extracting spatial sequence lookup tuples from JSON...")
    train_keys = load_alignment_keys_from_json(NOISY_GPS_JSON, train_groups)
    test_keys = load_alignment_keys_from_json(NOISY_GPS_JSON, test_groups)

    assert len(train_keys) == len(y_train_raw), f"Train alignment mismatch: {len(train_keys)} keys vs {len(y_train_raw)} labels"
    assert len(test_keys) == len(y_test_raw), f"Test alignment mismatch: {len(test_keys)} keys vs {len(y_test_raw)} labels"

    model = RuleBasedTransitionModel(graph, seed=RANDOM_SEED)
    train_time = model.fit(train_keys, list(y_train_raw))

    print(f"[Phase 7] Predicting on {len(test_keys):,} test items...")
    start = time.time()
    topk_preds = model.predict_topk(test_keys, k=3)

    y_test_list = list(y_test_raw)
    top1_correct = sum(1 for true, preds in zip(y_test_list, topk_preds) if preds and preds[0] == true)
    top3_correct = sum(1 for true, preds in zip(y_test_list, topk_preds) if true in preds)

    top1_acc = top1_correct / len(y_test_list)
    top3_acc = top3_correct / len(y_test_list)

    print(f"[Phase 7] [Rule-Based] Top-1 Accuracy: {top1_acc:.4f}  |  Top-3 Accuracy: {top3_acc:.4f}")
    return {"top1_accuracy": top1_acc, "top3_accuracy": top3_acc, "train_time_s": train_time}


# ===========================================================================
# MODEL 2 — SAFE LAPTOP RANDOM FOREST TABULAR CLASSIFIER
# ===========================================================================

def evaluate_random_forest_model(X_train, y_train_enc, X_test, y_test_enc, batch_size=20000):
    print("\n" + "─" * 70)
    print("  MODEL 2 — RANDOM FOREST TABULAR CLASSIFIER")
    print("─" * 70)

    X_train = X_train.astype({col: 'float32' for col in X_train.select_dtypes(include='bool').columns})
    X_test = X_test.astype({col: 'float32' for col in X_test.select_dtypes(include='bool').columns})

    model = RandomForestClassifier(**RF_PARAMS)

    print("[Phase 7] [Random Forest] Training multi-class ensemble trees safely...")
    start = time.time()
    model.fit(X_train, y_train_enc)
    train_time = time.time() - start
    print(f"[Phase 7] [Random Forest] ✓ Training complete in {train_time:.1f}s")

    print("[Phase 7] [Random Forest] Extracting prediction probabilities batch-by-batch...")
    total_rows = X_test.shape[0]
    top1_correct = 0
    top3_correct = 0

    for i in range(0, total_rows, batch_size):
        batch_end = min(i + batch_size, total_rows)
        X_batch = X_test.iloc[i:batch_end]
        y_batch = y_test_enc[i:batch_end]

        proba_batch = model.predict_proba(X_batch)
        
        top1_correct += top_k_correct_count(y_batch, proba_batch, k=1)
        top3_correct += top_k_correct_count(y_batch, proba_batch, k=3)

    top1_acc = top1_correct / total_rows
    top3_acc = top3_correct / total_rows

    print(f"[Phase 7] [Random Forest] Top-1 Accuracy: {top1_acc:.4f}  |  Top-3 Accuracy: {top3_acc:.4f}")
    return {"top1_accuracy": top1_acc, "top3_accuracy": top3_acc, "train_time_s": train_time}


# ===========================================================================
# MODEL 3 — LSTM SEQUENCE CLASSIFIER (PyTorch)
# ===========================================================================

class TrajectorySequenceDataset(Dataset):
    def __init__(self, X: pd.DataFrame, y_enc: np.ndarray, traj_ids: pd.Series, max_seq_len: int):
        self.max_seq_len = max_seq_len
        X_clean = X.astype({col: 'float32' for col in X.select_dtypes(include='bool').columns})
        self.n_features = X_clean.shape[1]
        
        # Enforce structural chronological ordering to ensure data sequence integrity
        sort_df = pd.DataFrame({'traj_id': traj_ids.values, 'point_step': X_clean['point_step'].values if 'point_step' in X_clean.columns else range(len(X_clean))})
        sorted_indices = sort_df.sort_values(by=["traj_id", "point_step"]).index.tolist()
        
        feature_values = X_clean.values.astype(np.float32)[sorted_indices]
        y_enc_sorted = y_enc[sorted_indices]
        traj_ids_sorted = traj_ids.values[sorted_indices]

        groups = defaultdict(list)
        for row_idx, tid in enumerate(traj_ids_sorted):
            groups[tid].append(row_idx)

        self.sequences = []
        self.labels = []
        self.lengths = []

        for tid, row_indices in groups.items():
            seq_feats = feature_values[row_indices]
            seq_labels = y_enc_sorted[row_indices]

            # Linear window constraint processing
            seq_len = min(len(row_indices), max_seq_len)
            self.sequences.append(seq_feats[:seq_len])
            self.labels.append(seq_labels[:seq_len])
            self.lengths.append(seq_len)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        labels = self.labels[idx]
        length = self.lengths[idx]

        padded_seq = np.zeros((self.max_seq_len, self.n_features), dtype=np.float32)
        padded_labels = np.full((self.max_seq_len,), -100, dtype=np.int64)  

        padded_seq[:length] = seq
        padded_labels[:length] = labels

        return torch.from_numpy(padded_seq), torch.from_numpy(padded_labels), length


class LSTMNextEdgeClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_head = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = torch.relu(self.input_proj(x))           
        lstm_out, _ = self.lstm(x)                    
        return self.output_head(lstm_out)


def train_lstm_model(model, train_loader, device, epochs, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    model.train()
    for epoch in range(epochs):
        epoch_start = time.time()
        total_loss = 0.0
        n_batches = 0

        for batch_seq, batch_labels, _ in train_loader:
            batch_seq, batch_labels = batch_seq.to(device), batch_labels.to(device)

            optimizer.zero_grad()
            logits = model(batch_seq)                          
            loss = criterion(logits.reshape(-1, logits.shape[-1]), batch_labels.reshape(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        print(f"[Phase 7] [LSTM]    Epoch {epoch + 1}/{epochs}  |  avg loss: {total_loss / n_batches:.4f}  |  {time.time() - epoch_start:.1f}s")


def evaluate_lstm_topk(model, test_loader, device, k_values=(1, 3)):
    model.eval()
    correct_at_k = {k: 0 for k in k_values}
    total_valid = 0

    with torch.no_grad():
        for batch_seq, batch_labels, _ in test_loader:
            batch_seq, batch_labels = batch_seq.to(device), batch_labels.to(device)
            logits = model(batch_seq)                          
            mask = batch_labels != -100                        

            valid_logits = logits[mask]                        
            valid_labels = batch_labels[mask]                   

            if valid_logits.shape[0] == 0: continue

            max_k = max(k_values)
            topk = torch.topk(valid_logits, k=min(max_k, valid_logits.shape[1]), dim=1).indices

            for k in k_values:
                k_eff = min(k, topk.shape[1])
                hit = (topk[:, :k_eff] == valid_labels.unsqueeze(1)).any(dim=1)
                correct_at_k[k] += hit.sum().item()

            total_valid += valid_labels.shape[0]

    return {k: (correct_at_k[k] / total_valid if total_valid > 0 else 0.0) for k in k_values}, total_valid


def evaluate_lstm_model(X_train, y_train_enc, train_traj_ids, X_test, y_test_enc, test_traj_ids, num_classes):
    print("\n" + "─" * 70)
    print("  MODEL 3 — LSTM SEQUENCE CLASSIFIER (PyTorch)")
    print("─" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = TrajectorySequenceDataset(X_train, y_train_enc, train_traj_ids, LSTM_MAX_SEQ_LEN)
    test_dataset  = TrajectorySequenceDataset(X_test, y_test_enc, test_traj_ids, LSTM_MAX_SEQ_LEN)

    train_loader = DataLoader(train_dataset, batch_size=LSTM_BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=LSTM_BATCH_SIZE, shuffle=False)

    input_dim = X_train.shape[1]
    model = LSTMNextEdgeClassifier(
        input_dim=input_dim, hidden_size=LSTM_HIDDEN_SIZE, num_layers=LSTM_NUM_LAYERS, num_classes=num_classes, dropout=LSTM_DROPOUT
    ).to(device)

    start = time.time()
    train_lstm_model(model, train_loader, device, epochs=LSTM_EPOCHS, lr=LSTM_LR)
    train_time = time.time() - start

    print("[Phase 7] [LSTM] Evaluating on held-out test data …")
    accuracies, n_valid_points = evaluate_lstm_topk(model, test_loader, device, k_values=(1, 3))
    print(f"[Phase 7] [LSTM] Top-1 Accuracy: {accuracies[1]:.4f}  |  Top-3 Accuracy: {accuracies[3]:.4f}")

    return {"top1_accuracy": accuracies[1], "top3_accuracy": accuracies[3], "train_time_s": train_time}


# ===========================================================================
# MAIN EXECUTIVE ORCHESTRATOR
# ===========================================================================

def main():
    set_all_seeds(RANDOM_SEED)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("═" * 70)
    print("  PHASE 7 — MODEL TRAINING & BENCHMARKING")
    print("═" * 70)

    X_full, y_full_raw = load_features(FEATURES_PKL)
    graph = load_graph(GRAPH_PKL)

    train_mask, test_mask, train_groups, test_groups = group_train_test_split(
        X_full, y_full_raw, group_col="traj_id", train_fraction=TRAIN_GROUP_FRACTION, seed=RANDOM_SEED,
    )

    train_traj_ids = X_full.loc[train_mask, "traj_id"].reset_index(drop=True)
    test_traj_ids  = X_full.loc[test_mask, "traj_id"].reset_index(drop=True)

    X_train_raw = X_full.loc[train_mask].reset_index(drop=True)
    X_test_raw  = X_full.loc[test_mask].reset_index(drop=True)
    y_train_raw = y_full_raw.loc[train_mask].reset_index(drop=True)
    y_test_raw  = y_full_raw.loc[test_mask].reset_index(drop=True)

    print("\n[Phase 7] Fitting target LabelEncoder variables on training set...")
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train_raw)
    num_classes = len(label_encoder.classes_)
    
    seen_classes = set(label_encoder.classes_)
    test_keep_mask = y_test_raw.isin(seen_classes)
    
    print(f"[Phase 7] Filtering test set rows to seen target classes...")
    print(f"[Phase 7]   Unseen target classes dropped from evaluation: {y_test_raw.nunique() - y_test_raw[test_keep_mask].nunique():,}")
    
    X_test_raw_filtered = X_test_raw[test_keep_mask].reset_index(drop=True)
    y_test_raw_filtered = y_test_raw[test_keep_mask].reset_index(drop=True)
    test_traj_ids_filtered = test_traj_ids[test_keep_mask].reset_index(drop=True)
    
    y_test_enc = label_encoder.transform(y_test_raw_filtered)

    results = {}
    
    results["Rule-Based Lookup Baseline"] = evaluate_rule_based_model(
        y_train_raw, y_test_raw, train_groups, test_groups, graph
    )

    X_train_clean = drop_non_predictive(X_train_raw)
    X_test_clean  = drop_non_predictive(X_test_raw_filtered)

    results["Random Forest Tabular Baseline"] = evaluate_random_forest_model(
        X_train_clean, y_train_enc, X_test_clean, y_test_enc
    )

    results["LSTM Recurrent Classifier"] = evaluate_lstm_model(
        X_train_clean, y_train_enc, train_traj_ids, X_test_clean, y_test_enc, test_traj_ids_filtered, num_classes
    )

    print_benchmark_table(results)
    with open(BENCHMARK_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Performance benchmarks successfully persisted at: '{BENCHMARK_JSON}'\n")

if __name__ == "__main__":
    main()