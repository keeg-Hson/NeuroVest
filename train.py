#!/usr/bin/env python3
# train.py
"""
NeuroVest — model training

Purpose
-------
Trains the main XGBoost-based classifier for event prediction on SPY.

Two primary labeling branches are supported:

1) Forward-returns branch (default, controlled by TRAIN_CFG["use_forward_returns"]):
   - Uses add_forward_returns_and_labels to derive a binary target y ∈ {0,1}
     based on forward net returns over TRAIN_CFG["horizon"] days and
     TRAIN_CFG["pos_threshold"].
   - Builds features via utils.add_features / utils.finalize_features.
   - Uses logs/top_signals.txt (if present) to gate the feature set based on
     prior correlation diagnostics.
   - Applies time-series cross-validation, optional SMOTE, KBest feature selection,
     sample-weighting, and probability calibration.
   - Saves:
       models/market_crash_model_fwd.pkl       (model + feature schema)
       models/input_features_fwd.txt           (input feature column names)
       models/label_map_fwd.json               (identity map for y ∈ {0,1})
       models/thresholds_fwd.json              (OOF-tuned probability threshold)

2) Triple-barrier branch (fallback):
   - Uses label_events_triple_barrier to derive multi-class Event labels.
   - Trains an XGBoost classifier with similar pipeline components.
   - Saves:
       models/market_crash_model.pkl
       models/input_features.txt
       models/label_map.json
       models/thresholds.json

This module is intended to be run via `python train.py` or imported and
called through train_model / train_best_xgboost_model.
"""

from dotenv import load_dotenv

load_dotenv(".env", override=True)

import json
import os
import socket
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import (
    SelectFromModel,
    SelectKBest,
    VarianceThreshold,
    f_classif,
    mutual_info_classif,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import BaseCrossValidator, GridSearchCV
from xgboost import XGBClassifier

socket.setdefaulttimeout(float(os.getenv("NET_TIMEOUT", 3)))

<<<<<<< HEAD
from config import MODELS_DIR, LOGS_DIR, TRAIN_CFG
=======
from config import MODELS_DIR, TRAIN_CFG
>>>>>>> f1644007
from utils import (
    add_features,
    add_forward_returns_and_labels,
    compute_sample_weights,
    ensure_no_future_leakage,
    finalize_features,
    load_SPY_data,
)

# === GLOBAL: forward-looking feature blacklist (never in model inputs) =========
FWD_BLACKLIST = {"y", "fwd_price", "fwd_ret_raw", "fwd_ret_net", "horizon_forward"}

# Allow env var to override the flag for quick experiments
if os.getenv("TRAIN_USE_FORWARD_RETURNS", "").strip() in {"1", "true", "True"}:
    TRAIN_CFG["use_forward_returns"] = True

# --- tolerate unknown CLI args when invoked like: python -m train --models xgb
import sys

if len(sys.argv) > 1:
    sys.argv = sys.argv[:1]


# ====================== Time-series CV helpers ======================
class PurgedWalkForwardSplit(BaseCrossValidator):
    def __init__(self, n_splits=5, min_train_size=250, test_size=None, embargo=3):
        self.n_splits = int(n_splits)
        self.min_train_size = int(min_train_size)
        self.test_size = None if test_size is None else int(test_size)
        self.embargo = int(embargo)

    def _resolved_test_size(self, n):
        if self.test_size is not None:
            return max(1, self.test_size)
        remainder = max(1, n - self.min_train_size)
        return max(1, remainder // max(1, self.n_splits))

    def _count_possible_splits(self, n):
        if n <= self.min_train_size + 1:
            return 0
        test_size = self._resolved_test_size(n)
<<<<<<< HEAD
        # FIX: Match the fix in split() method
        start_test = self.min_train_size + self.embargo
=======
        start_test = self.min_train_size
>>>>>>> f1644007
        made = 0
        for _ in range(self.n_splits):
            test_start = start_test
            test_end = min(n, test_start + test_size)
            if test_end - test_start < 1:
                break
            train_end = max(0, test_start - self.embargo)
            if train_end >= self.min_train_size:
                made += 1
            start_test = test_end
        return made

    def split(self, X, y=None, groups=None):
        n = len(X)
        test_size = self._resolved_test_size(n)
<<<<<<< HEAD
        # FIX: Start test after min_train_size + embargo to ensure first split is valid
        # Previous: start_test = self.min_train_size caused first split to fail
        # Now: start_test = self.min_train_size + self.embargo ensures train_end >= min_train_size
        start_test = self.min_train_size + self.embargo
=======
        start_test = self.min_train_size
>>>>>>> f1644007
        made = 0
        for _ in range(self.n_splits):
            test_start = start_test
            test_end = min(n, test_start + test_size)
            if test_end - test_start < 1:
                break
            train_end = max(0, test_start - self.embargo)
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)
            if len(train_idx) >= self.min_train_size:
                yield (train_idx, test_idx)
                made += 1
            start_test = test_end
            if made >= self.n_splits:
                break

    def get_n_splits(self, X=None, y=None, groups=None):
        if X is not None:
            return self._count_possible_splits(len(X))
        return self.n_splits


warnings.filterwarnings(
    "ignore", message=r"\[.*\] WARNING: .*Parameters: { \"use_label_encoder\" } are not used\."
)
xgb.set_config(verbosity=0)

os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)


class SingleFoldTimeSplit(BaseCrossValidator):
    def __init__(self, min_train_size=50, test_size=10, embargo=0):
        self.min_train_size = int(min_train_size)
        self.test_size = int(test_size)
        self.embargo = int(embargo)

    def split(self, X, y=None, groups=None):
        n = len(X)
        if n < self.min_train_size + self.test_size:
            return
        train_end = max(self.min_train_size, n - self.test_size - self.embargo)
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(train_end + self.embargo, n)
        if len(test_idx) > 0 and len(train_idx) >= self.min_train_size:
            yield (train_idx, test_idx)

    def get_n_splits(self, X=None, y=None, groups=None):
        if X is None:
            return 1
        n = len(X)
        return 1 if n >= self.min_train_size + self.test_size else 0


def _iter_splits(cv, n_rows):
    if hasattr(cv, "split"):
        yield from cv.split(np.arange(n_rows))
    else:
        yield from cv


def _n_splits(cv, n_rows):
    if hasattr(cv, "get_n_splits"):
        try:
            return int(cv.get_n_splits(np.arange(n_rows)))
        except Exception:
            return 0
    try:
        return sum(1 for _ in _iter_splits(cv, n_rows))
    except Exception:
        return 0


def _build_adaptive_cv(n_rows: int) -> BaseCrossValidator:
    min_train = max(200, int(0.2 * n_rows))
    n_splits = min(5, max(3, (n_rows - min_train) // 100))
    embargo = min(10, max(2, n_rows // 150))
    remainder = max(1, n_rows - min_train)
    test_size = max(25, remainder // max(1, n_splits))
    return PurgedWalkForwardSplit(
        n_splits=n_splits, min_train_size=min_train, test_size=test_size, embargo=embargo
    )


def _cv_or_holdout(n_rows, embargo=2, min_train_floor=30):
    cv = _build_adaptive_cv(n_rows)
    if _n_splits(cv, n_rows) > 0:
        return cv
    min_train = max(min_train_floor, int(0.5 * n_rows))
    test_size = max(1, n_rows - min_train - embargo)
    if min_train >= 1 and test_size >= 1 and (min_train + embargo + test_size) <= n_rows:
        return SingleFoldTimeSplit(min_train_size=min_train, test_size=test_size, embargo=embargo)
    if n_rows >= 3:
        tr = np.arange(0, n_rows - 1)
        te = np.array([n_rows - 1])
        return [(tr, te)]
    return []


def _min_minority_per_fold(y: pd.Series, cv) -> int:
    min_count = np.inf
    n = len(y)
    for tr, _ in _iter_splits(cv, n):
        vc = y.iloc[tr].value_counts()
        if len(vc) < 2:
            return 0
        min_count = min(min_count, int(vc.min()))
    return int(min_count if min_count != np.inf else 0)


def _safe_smote_from_fold(y: pd.Series, cv: BaseCrossValidator):
    m = _min_minority_per_fold(y, cv)
    if m < 2:
        print(f"ℹ️ SMOTE disabled (min minority per fold = {m}).")
        return False, "passthrough"
    k = max(1, min(5, m - 1))
    print(f"ℹ️ SMOTE enabled with k_neighbors={k} (min minority per fold={m}).")
    return True, SMOTE(random_state=42, k_neighbors=k)


def _write_split_meta(dates, cv, out_path=None):
    """
    Persist a simple split_meta.json so that downstream evaluation/backtests
    can optionally restrict to out-of-sample history.
    """
    try:
        dates = pd.to_datetime(dates)
        n = len(dates)
        splits = list(_iter_splits(cv, n))
        if not splits:
            return
        _, te = splits[-1]
        if len(te) == 0:
            return
        split_idx = int(np.min(te))
        split_dt = dates[split_idx]
        split_date_str = split_dt.date().isoformat() if hasattr(split_dt, "date") else str(split_dt)

        payload = {
            "split_index": int(split_idx),
            "split_date": split_date_str,
            "n_rows": int(n),
            "cv_folds": int(len(splits)),
        }
        target = out_path or (MODELS_DIR / "split_meta.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"💾 Saved split metadata → {target}")
    except Exception as e:
        print(f"⚠️ Could not write split_meta.json: {e}")


def pick_threshold_from_oof(pipe, X, y, cv, pos_label=1):
    n = len(X)
    proba_oof = np.full(n, np.nan, dtype=float)
    seen = np.zeros(n, dtype=bool)
    classes_seen, col_idx = None, None

    for tr, te in _iter_splits(cv, n):
        est = clone(pipe)
        est.fit(X.iloc[tr], y.iloc[tr])
        probs = est.predict_proba(X.iloc[te])
        if classes_seen is None:
            classes_seen = list(getattr(est, "classes_", [0, 1]))
            try:
                col_idx = classes_seen.index(pos_label)
            except ValueError:
                col_idx = 1 if probs.shape[1] > 1 else 0

        write_mask = ~seen[te]
        idxs = np.asarray(te)[write_mask]
        if idxs.size:
            proba_oof[idxs] = probs[write_mask, col_idx]
            seen[idxs] = True

    mask = ~np.isnan(proba_oof)
    if not mask.any():
        raise RuntimeError("OOF builder produced no predictions. Check CV splits.")

    y_pos = (np.asarray(y)[mask] == pos_label).astype(int)
    p = proba_oof[mask]

    # Method 1: Precision-Recall curve F1 optimization (more accurate)
    prec_curve, rec_curve, thresholds = precision_recall_curve(y_pos, p)
    # Calculate F1 scores for each threshold
    f1_scores = 2 * (prec_curve * rec_curve) / (prec_curve + rec_curve + 1e-9)

    # Find threshold with best F1 score (with minimum precision constraint)
    min_precision = 0.40
    valid_idx = np.where(prec_curve >= min_precision)[0]

    if len(valid_idx) > 0:
        # Among valid thresholds, pick the one with best F1
        best_idx = valid_idx[np.argmax(f1_scores[valid_idx])]
        best_t_pr = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        best_f1_pr = f1_scores[best_idx]
        best_prec_pr = prec_curve[best_idx]
        best_rec_pr = rec_curve[best_idx]
    else:
        # Fallback to 0.5 if no valid threshold found
        best_t_pr, best_f1_pr, best_prec_pr, best_rec_pr = 0.5, 0.0, 0.0, 0.0

    # Method 2: Linear search (original method, kept for comparison)
    ts = np.linspace(0.30, 0.70, 41)
    best_t_linear, best_f1_linear, best_prec_linear, best_rec_linear = 0.50, -1.0, 0.0, 0.0

    for t_ in ts:
        y_hat = (p >= t_).astype(int)
        prec = precision_score(y_pos, y_hat, zero_division=0)
        rec = recall_score(y_pos, y_hat, zero_division=0)

        # Only consider thresholds with acceptable precision
        if prec < min_precision:
            continue

        f1 = f1_score(y_pos, y_hat, zero_division=0)

        # Prefer balanced precision/recall by penalizing extreme imbalances
        balance_penalty = abs(prec - rec) / (prec + rec + 1e-9)
        adjusted_f1 = f1 * (1.0 - 0.2 * balance_penalty)

        if adjusted_f1 > best_f1_linear:
            best_f1_linear = f1  # Store original F1, not adjusted
            best_t_linear = t_
            best_prec_linear = prec
            best_rec_linear = rec

    # Use PR curve method if it found a better F1 score, otherwise use linear search
    if best_f1_pr > best_f1_linear:
        best_t, best_f1, best_prec, best_rec = best_t_pr, best_f1_pr, best_prec_pr, best_rec_pr
    else:
        best_t, best_f1, best_prec, best_rec = (
            best_t_linear,
            best_f1_linear,
            best_prec_linear,
            best_rec_linear,
        )

    return best_t, {
        "precision": float(best_prec),
        "recall": float(best_rec),
        "f1": float(best_f1),
        "proba_col_index": int(col_idx if col_idx is not None else 1),
        "pos_enc": int(pos_label),
    }


# ====================== Top-signal utilities ======================
<<<<<<< HEAD
def _load_top_signals(path = None) -> list[str]:
=======
def _load_top_signals(path: str = "logs/top_signals.txt") -> list[str]:
>>>>>>> f1644007
    """
    Loads top feature names from analyze_signals/select_top_signals output.

    Expected format (per line):
        feature_name,correlation,...

    Lines starting with "Top" or empty lines are ignored.
    """
<<<<<<< HEAD
    if path is None:
        path = LOGS_DIR / "top_signals.txt"
    p = Path(path)
=======
    p = os.path.join(path)
>>>>>>> f1644007
    try:
        with open(p) as f:
            lines = f.readlines()
    except FileNotFoundError:
        print("⚠️ logs/top_signals.txt not found. Proceeding without correlation-based gating.")
        return []

    top = [
        ln.strip().split(",")[0] for ln in lines if ln.strip() and not ln.lstrip().startswith("Top")
    ]
    top = [c for c in dict.fromkeys(top) if c]
    print(f"✅ Loaded {len(top)} top signals for potential feature gating.")
    return top


# ====================== Public entry points ======================
def train_model(models=None, fast=False):
    df = load_SPY_data()
    return train_best_xgboost_model(df)


def run(models=None, fast=False):
    return train_model(models=models, fast=fast)


def main(models=None, fast=False):
    ok = train_model(models=models, fast=fast)
    return 0 if ok else 1


# ====================== Trainer ======================
def train_best_xgboost_model(df: pd.DataFrame) -> bool:
    print("\n📊 Generating features...")
    df, all_feature_cols = add_features(df)

    # Optional: top-signal gating from prior correlation diagnostics
    top_signals = _load_top_signals()

    N = len(df)
    MIN_VALID_ROWS = max(5, min(60, int(N * 0.40)))

    # Start from the rich feature list returned by add_features,
    # filtered only for minimum data availability.
    feature_cols = [
        c for c in all_feature_cols if c in df.columns and df[c].notna().sum() >= MIN_VALID_ROWS
    ]

    if not feature_cols:
        minimal_base = [
            "Daily_Return",
            "Return_Lag1",
            "Return_Lag3",
            "Return_Lag5",
            "ZMomentum",
            "RSI",
            "MACD",
            "MACD_Signal",
            "Stoch_K",
            "Stoch_D",
            "Gap_Pct",
            "Acceleration",
        ]
        feature_cols = [c for c in minimal_base if c in df.columns and df[c].notna().sum() >= 5]

    if not feature_cols:
        raise RuntimeError("No features available after dynamic fallback.")

    feature_cols = list(dict.fromkeys(feature_cols))

    # Apply intersection with top_signals when the overlap is reasonably sized.
    if top_signals:
        allowed = set(top_signals)
        intersect = [c for c in feature_cols if c in allowed]
        min_overlap = max(10, len(feature_cols) // 3)
        if len(intersect) >= min_overlap:
            print(
                f"🔎 Using intersection with top_signals for training features "
                f"({len(intersect)} of {len(feature_cols)})."
            )
            feature_cols = intersect
        else:
            print(
                f"ℹ️ top_signals intersection too small ({len(intersect)}); "
                f"retaining broader feature set ({len(feature_cols)})."
            )

    print(
        f"🧪 Final feature set ({len(feature_cols)}): "
        f"{feature_cols[:20]}{'...' if len(feature_cols) > 20 else ''}"
    )

    # CLEAN features BEFORE labeling/splitting
    df = finalize_features(df, feature_cols)

    # Ensure Close exists for labeling
    import pandas as _pd

    try:
        _raw = load_SPY_data()
        _raw_idxed = _raw["Close"].astype(float)
        df.index = _pd.to_datetime(df.index, errors="coerce")
        _raw_idxed.index = _pd.to_datetime(_raw_idxed.index, errors="coerce")
        df["Close"] = _raw_idxed.reindex(df.index)
    except Exception as _e:
        if "Close" not in df.columns:
            raise RuntimeError(f"Could not attach Close for labeling: {_e}") from _e
    df = df.dropna(subset=["Close"])

    feature_cols = [c for c in feature_cols if c in df.columns and df[c].notna().any()]

    # ====== Forward-returns branch ======
    if TRAIN_CFG.get("use_forward_returns", False):
        df = df.replace([np.inf, -np.inf], np.nan)

        print(
            f"[FWD] Labeling with horizon={TRAIN_CFG['horizon']}d, "
            f"pos_threshold={TRAIN_CFG['pos_threshold']:.4f}, "
            f"price_col='{TRAIN_CFG['price_col']}'."
        )

        df = add_forward_returns_and_labels(
            df,
            price_col=TRAIN_CFG["price_col"],
            horizon=TRAIN_CFG["horizon"],
            fee_bps=TRAIN_CFG["fee_bps"],
            slippage_bps=TRAIN_CFG["slippage_bps"],
            long_only=TRAIN_CFG["long_only"],
            pos_threshold=TRAIN_CFG["pos_threshold"],
        )

<<<<<<< HEAD
        INPUT_SCHEMA_FPATH = MODELS_DIR / "input_features_fwd.txt"
=======
        INPUT_SCHEMA_FPATH = "models/input_features_fwd.txt"
>>>>>>> f1644007

        def _clean_names(names):
            return [c for c in names if c not in FWD_BLACKLIST]

        try:
            with open(INPUT_SCHEMA_FPATH) as f:
                input_cols = _clean_names([line.strip() for line in f if line.strip()])
                print(f"📄 Loaded prior input schema (cleaned) with {len(input_cols)} cols.")
        except Exception:
            base = list(dict.fromkeys(feature_cols + [c for c in ["Close"] if c in df.columns]))
            input_cols = _clean_names([c for c in base if c in df.columns])

        for c in input_cols:
            if c not in df.columns:
                df[c] = np.nan

        X = df[input_cols].astype(float).replace([np.inf, -np.inf], np.nan)
        y = df["y"].astype(int)
        mask_ok = pd.Series(y).notna()
        X, y = X.loc[mask_ok], y.loc[mask_ok]

        if any(c in FWD_BLACKLIST for c in X.columns):
            raise RuntimeError(f"Leaky features detected in X: {set(X.columns) & FWD_BLACKLIST}")

        ensure_no_future_leakage(df, list(X.columns), ["y"], horizon_col="horizon_forward")

        os.makedirs("models", exist_ok=True)
        pd.Series(list(X.columns), dtype=str).to_csv(INPUT_SCHEMA_FPATH, index=False, header=False)
        print(f"💾 Saved CLEAN input schema → {INPUT_SCHEMA_FPATH} ({len(X.columns)} cols)")

        tscv_local = _cv_or_holdout(len(X), embargo=2, min_train_floor=30)
        n_folds = _n_splits(tscv_local, len(X))
        if n_folds == 0:
            if len(X) >= 2:
                tr = np.arange(0, len(X) - 1)
                te = np.array([len(X) - 1])
                tscv_local = [(tr, te)]
                n_folds = 1
                print("ℹ️ [FWD] Using last-row holdout (1 split).")
            else:
                raise RuntimeError("Not enough rows to train. Need at least 2.")

        _write_split_meta(X.index, tscv_local)

        # SMOTE optionally (guarded)
        try:
            use_smote, smote_step = _safe_smote_from_fold(y, tscv_local)
        except Exception:
            use_smote, smote_step = (False, "passthrough")

        xgb_common = dict(
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            tree_method="hist",
            use_label_encoder=False,
            early_stopping_rounds=75,  # Stop if no improvement for 75 rounds (allow more exploration)
        )
        xgb_obj = dict(objective="binary:logistic", eval_metric="logloss")

        use_kbest = X.shape[1] >= 2
        if use_kbest:
            max_k = X.shape[1]
<<<<<<< HEAD
            # Optimized for ~114 features (Nov 2025): Focus on 30-60 range
            # Analysis showed 20-30 features was too aggressive for expanded feature set
            k_choices = sorted(set([30, 40, 50, 60, max(30, max_k // 2), max_k]))
            k_choices = [k for k in k_choices if 5 <= k <= max_k]
            # Ensure minimum choices for smaller feature sets
=======
            # Expanded feature set: allow more features to improve model capacity
            k_choices = sorted(set([15, 20, 25, 30, max(15, max_k // 2), max_k]))
            k_choices = [k for k in k_choices if 5 <= k <= max_k]
            # Ensure we have at least some choices even with smaller feature sets
>>>>>>> f1644007
            if not k_choices:
                k_choices = sorted(set([min(5, max_k), min(10, max_k), max_k]))
            print(f"🔧 Feature selection k_choices: {k_choices}")

<<<<<<< HEAD
            # Optional: Pre-filter with tree-based importance for large feature sets
=======
            # Optional: Pre-filter with tree-based importance if we have many features
>>>>>>> f1644007
            use_tree_prefilter = X.shape[1] > 40
            if use_tree_prefilter:
                print("🌲 Using tree-based feature pre-filtering (ExtraTrees importance)")
                tree_selector = SelectFromModel(
                    ExtraTreesClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1),
                    threshold="median",  # Keep top 50% of features by importance
                )
                steps = [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("varth", VarianceThreshold(threshold=0.0)),
                    ("tree_selector", tree_selector),
                    ("smote", smote_step),
                    ("kbest", SelectKBest(score_func=mutual_info_classif)),
                    ("clf", XGBClassifier(**xgb_common, **xgb_obj)),
                ]
            else:
                steps = [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("varth", VarianceThreshold(threshold=0.0)),
                    ("smote", smote_step),
                    ("kbest", SelectKBest(score_func=mutual_info_classif)),
                    ("clf", XGBClassifier(**xgb_common, **xgb_obj)),
                ]
            pipe = Pipeline(steps=steps)
<<<<<<< HEAD
            # FIX: Reduced hyperparameter grid to prevent overfitting
            # Previous: 118,098 combinations (massive overfitting!)
            # Current: ~48 combinations (focus on key parameters)
            param_grid = {
                "kbest__k": k_choices,  # Keep all k choices for feature selection
                "clf__n_estimators": [500],  # Fix to middle value
                "clf__max_depth": [4, 6],  # MOST IMPORTANT - keep 2 values
                "clf__learning_rate": [0.02, 0.03],  # MOST IMPORTANT - keep 2 values
                "clf__subsample": [0.8],  # Fix to best value
                "clf__colsample_bytree": [0.8],  # Fix to best value
                "clf__min_child_weight": [10],  # Fix to best for class imbalance
                "clf__gamma": [0],  # Fix to 0 (usually best)
                "clf__reg_alpha": [0, 0.05],  # L1 regularization - keep 2 values
                "clf__reg_lambda": [1.5],  # L2 regularization - fix to middle
=======
            # Enhanced hyperparameter grid with regularization and better exploration
            param_grid = {
                "kbest__k": k_choices,
                "clf__n_estimators": [300, 500, 700],  # More trees for better learning
                "clf__max_depth": [4, 6, 8],  # Slightly deeper trees
                "clf__learning_rate": [0.01, 0.02, 0.03],  # Focus on proven optimal range
                "clf__subsample": [0.7, 0.8, 0.9],  # Finer gradations
                "clf__colsample_bytree": [0.7, 0.8, 0.9],  # Finer gradations
                "clf__min_child_weight": [5, 10, 15],  # Prevent overfitting on class imbalance
                "clf__gamma": [0, 0.5, 1.0],  # More exploration
                "clf__reg_alpha": [
                    0,
                    0.05,
                    0.2,
                ],  # L1 regularization - better granularity in critical range
                "clf__reg_lambda": [
                    0.5,
                    1.5,
                    3.0,
                ],  # L2 regularization - lower starting point for 80+ features
>>>>>>> f1644007
            }
            print(
                f"🔧 Hyperparameter grid size: {np.prod([len(v) for v in param_grid.values()])} combinations"
            )
        else:
            steps = [
                ("imputer", SimpleImputer(strategy="median")),
                ("varth", VarianceThreshold(threshold=0.0)),
                ("smote", smote_step),
                ("clf", XGBClassifier(**xgb_common, **xgb_obj)),
            ]
            pipe = Pipeline(steps=steps)
<<<<<<< HEAD
            # FIX: Reduced hyperparameter grid (no KBest version)
            # Previous: 3^9 = 19,683 combinations
            # Current: ~8 combinations
            param_grid = {
                "clf__n_estimators": [500],  # Fix to middle value
                "clf__max_depth": [4, 6],  # MOST IMPORTANT - keep 2 values
                "clf__learning_rate": [0.02, 0.03],  # MOST IMPORTANT - keep 2 values
                "clf__subsample": [0.8],  # Fix to best value
                "clf__colsample_bytree": [0.8],  # Fix to best value
                "clf__min_child_weight": [10],  # Fix to best value
                "clf__gamma": [0],  # Fix to 0
                "clf__reg_alpha": [0, 0.05],  # L1 regularization - keep 2 values
                "clf__reg_lambda": [1.5],  # L2 regularization - fix to middle
=======
            # Enhanced hyperparameter grid (no KBest) with regularization
            param_grid = {
                "clf__n_estimators": [300, 500, 700],
                "clf__max_depth": [4, 6, 8],
                "clf__learning_rate": [0.01, 0.03, 0.05],
                "clf__subsample": [0.7, 0.8, 0.9],
                "clf__colsample_bytree": [0.7, 0.8, 0.9],
                "clf__min_child_weight": [3, 5, 10],
                "clf__gamma": [0, 0.5, 1.0],
                "clf__reg_alpha": [0, 0.1, 0.5],
                "clf__reg_lambda": [1.0, 2.0, 5.0],
>>>>>>> f1644007
            }
            print(
                f"🔧 Hyperparameter grid size: {np.prod([len(v) for v in param_grid.values()])} combinations"
            )

        # IMPORTANT FIX: align profit-based sample weights to X.index
        sample_weight_profit = compute_sample_weights(
            df.loc[X.index],
            min_weight=TRAIN_CFG["min_weight"],
            max_weight=TRAIN_CFG["max_weight"],
            power=TRAIN_CFG["weight_power"],
            long_only=TRAIN_CFG["long_only"],
        )

        print(
            "🧮 Sample weight stats → "
            f"min={float(sample_weight_profit.min()):.3f}, "
            f"max={float(sample_weight_profit.max()):.3f}, "
            f"mean={float(sample_weight_profit.mean()):.3f}"
        )

        grid_search = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring="f1_macro",
            cv=tscv_local,
            n_jobs=-1,
            verbose=2,
            error_score=0,
        )
        print(f"[{datetime.now():%H:%M:%S}] starting GridSearchCV...")
        grid_search.fit(X, y)
        print(f"[{datetime.now():%H:%M:%S}] gridsearch done.")

        print(f"\n✅ [FWD] Best Params: {grid_search.best_params_}")
        print(f"🎯 [FWD] Best Score (F1 Macro): {grid_search.best_score_:.4f}")

        best_model = grid_search.best_estimator_

        from sklearn.utils.class_weight import compute_sample_weight as _csw

        y_pred0 = best_model.predict(X)
        w_miss = 1.0 + 2.0 * (y_pred0 != y).astype(float)
        w_bal = _csw(class_weight="balanced", y=y)
        w_final = w_miss * w_bal * sample_weight_profit

        from copy import deepcopy

        best_model_wn = deepcopy(best_model)
        if hasattr(best_model_wn, "steps"):
            steps_map = dict(best_model_wn.steps)
            if "smote" in steps_map:
                best_model_wn.set_params(smote="passthrough")

        best_model_wn.fit(X, y, **{"clf__sample_weight": w_final})

        # Check if calibration will actually help (quality gate)
        from sklearn.calibration import CalibratedClassifierCV, calibration_curve

        p_base = best_model_wn.predict_proba(X)[:, 1]
        prob_true, prob_pred = calibration_curve(y, p_base, n_bins=10, strategy="uniform")
        calibration_error = np.mean(np.abs(prob_true - prob_pred))

        print(f"📊 Calibration error: {calibration_error:.4f}")

        if calibration_error < 0.03:
            # Model is already well-calibrated, skip calibration to avoid degradation
            print("✅ Model already well-calibrated (error < 0.03), skipping calibration")
            best_model = best_model_wn
        else:
            print(f"📈 Applying calibration (error = {calibration_error:.4f} >= 0.03)")
            try:
                cal = CalibratedClassifierCV(best_model_wn, cv=3, method="isotonic")
                cal.fit(X, y, sample_weight=w_final)
            except Exception as e:
                print(f"⚠️ [FWD] Isotonic calibration failed ({e}) — falling back to sigmoid.")
                cal = CalibratedClassifierCV(best_model_wn, cv=3, method="sigmoid")
                cal.fit(X, y, sample_weight=w_final)

            best_model = cal

        MODEL_DIR = "models"
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path_fwd = os.getenv(
            "MODEL_PATH_FWD", os.path.join(MODEL_DIR, "market_crash_model_fwd.pkl")
        )

        payload = {"model": best_model, "features": list(X.columns)}
        joblib.dump(payload, model_path_fwd)
        print(f"💾 [FWD] Model saved to {model_path_fwd} (with feature schema)")

        label_values = sorted(pd.Series(y).unique().tolist())
        label_map = {int(v): int(v) for v in label_values}
        inv_label_map = {int(v): int(v) for v in label_values}
<<<<<<< HEAD
        with open(MODELS_DIR / "label_map_fwd.json", "w") as f:
=======
        with open("models/label_map_fwd.json", "w") as f:
>>>>>>> f1644007
            json.dump(
                {
                    "label_map": {str(k): int(v) for k, v in label_map.items()},
                    "inv_label_map": {str(k): int(v) for k, v in inv_label_map.items()},
                },
                f,
                indent=2,
            )
<<<<<<< HEAD
        print(f"💾 [FWD] Label maps → {MODELS_DIR}/label_map_fwd.json")
=======
        print("💾 [FWD] Label maps → models/label_map_fwd.json")
>>>>>>> f1644007

        try:
            best_pipe_for_oof = grid_search.best_estimator_
            t_star, metr = pick_threshold_from_oof(best_pipe_for_oof, X, y, tscv_local, pos_label=1)
        except Exception as e:
            print(f"⚠️ [FWD] OOF threshold selection failed ({e}) — falling back to 0.50.")
            t_star, metr = (
                0.50,
                {"precision": 0.0, "recall": 0.0, "f1": 0.0, "proba_col_index": 1, "pos_enc": 1},
            )

        thr_payload = {
            "pos_orig": 1,
            "pos_enc": metr.get("pos_enc", 1),
            "proba_col_index": metr.get("proba_col_index", 1),
            "threshold": float(t_star),
            "metric": "f1_positive_only_oof",
            "precision_oof": float(metr.get("precision", 0.0)),
            "recall_oof": float(metr.get("recall", 0.0)),
            "f1_oof": float(metr.get("f1", 0.0)),
        }
<<<<<<< HEAD
        with open(MODELS_DIR / "thresholds_fwd.json", "w") as f:
            json.dump(thr_payload, f, indent=2)
        print(
            f"💾 [FWD] Thresholds → {MODELS_DIR}/thresholds_fwd.json: "
=======
        with open("models/thresholds_fwd.json", "w") as f:
            json.dump(thr_payload, f, indent=2)
        print(
            f"💾 [FWD] Thresholds → models/thresholds_fwd.json: "
>>>>>>> f1644007
            f"t={t_star:.3f} (P_oof={thr_payload['precision_oof']:.3f}, "
            f"R_oof={thr_payload['recall_oof']:.3f}, F1_oof={thr_payload['f1_oof']:.3f})"
        )

        print("✅ [FWD] Forward-returns training completed.")
        print(
            "ℹ️ Predictor will automatically load this variant when PREDICT_VARIANT=forward_returns."
        )
        return True

    # ====== Triple-barrier branch (unchanged logic, but save model WITH schema) ======
    from utils import label_events_triple_barrier

    df["Volatility"] = df["Close"].rolling(window=20).std()
    print("\n🧪 Sample volatility (tail):")
    print(df["Volatility"].dropna().tail(10))

    df = label_events_triple_barrier(df, vol_col="ATR_14", pt_mult=1.0, sl_mult=1.0, t_max=10)

    view = (df if "Date" in df.columns else df.reset_index().rename(columns={"index": "Date"}))[
        ["Date", "Close", "Event"]
    ].tail(15)
    print(view)
    print("\n📊 Distribution of Event labels (incl. NaNs):")
    print(df["Event"].value_counts(dropna=False))
    print("\n📊 Number of unique Events:")
    print(df["Event"].nunique(dropna=False))

    if df["Event"].nunique() <= 1:
        print("❌ Not enough class diversity in Event labels — training aborted.")
        return False

    print("\n🧪 Missing values per feature column (post-clean):")
    print(df[feature_cols].isna().sum())
    print(f"\n🧪 Total rows before dropna: {len(df)}")

    valid_feature_cols = [c for c in feature_cols if df[c].notna().sum() > 0]
    required_cols = ["Event"] + valid_feature_cols
    df = df.dropna(subset=required_cols)
    if len(df) == 0:
        print("⚠️ Triple-barrier produced 0 usable rows — falling back to forward-returns.")
        os.environ["TRAIN_USE_FORWARD_RETURNS"] = "1"
        from config import TRAIN_CFG as _TC

        _TC["use_forward_returns"] = True
        from train import train_best_xgboost_model as _tbxm

        return _tbxm(load_SPY_data())

    print(f"\n🧹 Rows remaining after dropna: {len(df)}")
    if len(df) == 0:
        print("❌ No data left after dropping NaNs.")
        return False

    df = df[df["Event"] != 0].copy()

    try:
<<<<<<< HEAD
        with open(MODELS_DIR / "input_features.txt") as f:
            input_cols = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(
            f"⚠️ {MODELS_DIR}/input_features.txt missing or unreadable ({e}); using available columns as-is."
=======
        with open("models/input_features.txt") as f:
            input_cols = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(
            f"⚠️ models/input_features.txt missing or unreadable ({e}); using available columns as-is."
>>>>>>> f1644007
        )
        input_cols = [c for c in df.columns if c not in ("Event",)]

    for c in input_cols:
        if c not in df.columns:
            df[c] = 0.0

    X = df[input_cols].astype(float).replace([np.inf, -np.inf], np.nan)
    y_orig = df["Event"]

<<<<<<< HEAD
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    pd.Series(list(X.columns), dtype=str).to_csv(
        MODELS_DIR / "input_features.txt", index=False, header=False
    )
    print(f"💾 Input feature columns saved to {MODELS_DIR}/input_features.txt")
=======
    os.makedirs("models", exist_ok=True)
    pd.Series(list(X.columns), dtype=str).to_csv(
        "models/input_features.txt", index=False, header=False
    )
    print("💾 Input feature columns saved to models/input_features.txt")
>>>>>>> f1644007

    label_values = sorted(y_orig.unique())
    label_map = {lab: i for i, lab in enumerate(label_values)}
    inv_label_map = {v: k for k, v in label_map.items()}
    y = y_orig.map(label_map).astype(int)

    pos = int((y == 1).sum()) if 1 in y.unique() else 1
    neg = int((y == 0).sum()) if 0 in y.unique() else 1
    spw = max(1.0, neg / max(1, pos))
    print(f"🔧 scale_pos_weight for XGB: {spw:.2f}")

    print(f"🔤 Label encoding map: {label_map} (train on 0..{y.nunique() - 1})")
    print("\n📊 Encoded class distribution (0..K-1):")
    print(y.value_counts())

<<<<<<< HEAD
    with open(MODELS_DIR / "label_map.json", "w") as f:
=======
    with open("models/label_map.json", "w") as f:
>>>>>>> f1644007
        json.dump(
            {
                "label_map": {str(k): int(v) for k, v in label_map.items()},
                "inv_label_map": {str(k): int(v) for k, v in inv_label_map.items()},
            },
            f,
            indent=2,
        )
<<<<<<< HEAD
    print(f"💾 Saved label maps to {MODELS_DIR}/label_map.json")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "market_crash_model.pkl"
=======
    print("💾 Saved label maps to models/label_map.json")

    MODEL_DIR = "models"
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "market_crash_model.pkl")
>>>>>>> f1644007

    if X.shape[0] == 0:
        raise RuntimeError("Empty training matrix after cleaning.")
    if X.shape[1] == 0:
        raise RuntimeError("No features selected after filtering.")

    print("\n📊 Original label distribution (1/2):")
    print(y_orig.value_counts())
    print("\n📊 Encoded label distribution (0/1):")
    print(y.value_counts())

    tscv_local = _cv_or_holdout(len(X), embargo=2, min_train_floor=30)
    n_folds = _n_splits(tscv_local, len(X))
    if n_folds == 0:
        if len(X) >= 3:
            tr = np.arange(0, len(X) - 1)
            te = np.array([len(X) - 1])
            tscv_local = [(tr, te)]
            n_folds = 1
            print("ℹ️ Using last-row holdout (1 split).")
        else:
            raise RuntimeError("Not enough rows to train. Need at least 3.")
    print(f"CV folds actually used: {n_folds}")

    try:
        use_smote, smote_step = _safe_smote_from_fold(y, tscv_local)
    except Exception:
        use_smote, smote_step = (False, "passthrough")

    n_classes = int(y.nunique())
    is_binary = n_classes == 2

    xgb_common = dict(
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        tree_method="hist",
        use_label_encoder=False,
        scale_pos_weight=spw,
    )
    xgb_obj = (
        dict(objective="binary:logistic", eval_metric="logloss")
        if is_binary
        else dict(objective="multi:softprob", eval_metric="mlogloss", num_class=n_classes)
    )

    use_kbest = X.shape[1] >= 2
    if use_kbest:
        max_k = X.shape[1]
        k_choices = sorted(set([1, 2, 3, 5, 8, 10, 12, max(1, max_k // 2)]))
        k_choices = [k for k in k_choices if 1 <= k <= max_k]
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("varth", VarianceThreshold(threshold=0.0)),
            ("smote", smote_step),
            ("kbest", SelectKBest(score_func=f_classif)),
            ("clf", XGBClassifier(**xgb_common, **xgb_obj)),
        ]
        pipe = Pipeline(steps=steps)
        param_grid = {
            "kbest__k": k_choices,
            "clf__n_estimators": [200, 400],
            "clf__max_depth": [3, 5, 7],
            "clf__learning_rate": [0.03, 0.05],
            "clf__subsample": [0.7, 0.9, 1.0],
            "clf__colsample_bytree": [0.7, 0.9, 1.0],
            "clf__min_child_weight": [1, 3, 5],
            "clf__gamma": [0, 1],
        }
    else:
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("varth", VarianceThreshold(threshold=0.0)),
            ("smote", smote_step),
            ("clf", XGBClassifier(**xgb_common, **xgb_obj)),
        ]
        pipe = Pipeline(steps=steps)
        param_grid = {
            "clf__n_estimators": [200, 400],
            "clf__max_depth": [3, 5, 7],
            "clf__learning_rate": [0.03, 0.05],
            "clf__subsample": [0.7, 0.9, 1.0],
            "clf__colsample_bytree": [0.7, 0.9, 1.0],
            "clf__min_child_weight": [1, 3, 5],
            "clf__gamma": [0, 1],
        }

    print("\n🔍 Starting Grid Search (time-series CV)...")
    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=tscv_local,
        n_jobs=-1,
        verbose=2,
        error_score=0,
    )
    print(f"[{datetime.now():%H:%M:%S}] starting GridSearchCV...")
    grid_search.fit(X, y)
    print(f"[{datetime.now():%H:%M:%S}] gridsearch done.")

    print(f"\n✅ Best Params: {grid_search.best_params_}")
    print(f"🎯 Best Score (F1 Macro): {grid_search.best_score_:.4f}")

    best_model = grid_search.best_estimator_

    from sklearn.utils.class_weight import compute_sample_weight

    y_pred0 = best_model.predict(X)
    w_miss = 1.0 + 2.0 * (y_pred0 != y).astype(float)
    w_bal = compute_sample_weight(class_weight="balanced", y=y)
    w = w_miss * w_bal

    from copy import deepcopy

    best_model_wn = deepcopy(best_model)
    if hasattr(best_model_wn, "steps"):
        steps_map = dict(best_model_wn.steps)
        if "smote" in steps_map:
            best_model_wn.set_params(smote="passthrough")
    best_model_wn.fit(X, y, **{"clf__sample_weight": w})

    pipe_for_introspection = best_model_wn

    from sklearn.calibration import CalibratedClassifierCV

    try:
        cal = CalibratedClassifierCV(best_model_wn, cv=3, method="isotonic")
        cal.fit(X, y, sample_weight=w)
    except Exception as e:
        print(f"⚠️ Isotonic calibration failed ({e}) — falling back to sigmoid.")
        cal = CalibratedClassifierCV(best_model_wn, cv=3, method="sigmoid")
        cal.fit(X, y, sample_weight=w)

    best_model = cal

    payload = {"model": best_model, "features": list(X.columns)}
    joblib.dump(payload, model_path)

    try:
        import numpy as _np
        import shap

        shap_est = None
        shap_X = None
        base_est = getattr(best_model, "base_estimator", None) or best_model
        if hasattr(base_est, "named_steps"):
            kb = base_est.named_steps.get("kbest", None)
            clf = base_est.named_steps.get("clf", None)
            if kb is not None and hasattr(kb, "transform"):
                shap_X = kb.transform(X)
                try:
                    mask = kb.get_support()
                    feat_names = (
                        list(_np.array(list(X.columns))[mask])
                        if hasattr(mask, "__len__") and len(mask) == X.shape[1]
                        else list(X.columns)
                    )
                except Exception:
                    feat_names = list(X.columns)
            else:
                shap_X = X.values
                feat_names = list(X.columns)
            shap_est = clf or base_est
        else:
            shap_est = base_est
            shap_X = X.values
            feat_names = list(X.columns)

        sample_n = min(5000, shap_X.shape[0])
        if sample_n < shap_X.shape[0]:
            rs = _np.random.RandomState(42)
            take = rs.choice(shap_X.shape[0], size=sample_n, replace=False)
            shap_X_sample = shap_X[take]
        else:
            shap_X_sample = shap_X

        explainer = shap.TreeExplainer(shap_est)
        shap_vals = explainer.shap_values(shap_X_sample)

        if isinstance(shap_vals, list):
            shap_abs = _np.abs(_np.array(shap_vals)).max(axis=0)
        else:
            shap_abs = _np.abs(shap_vals)

        mean_abs = shap_abs.mean(axis=0)
        shap_df = pd.DataFrame({"feature": feat_names, "mean_abs_shap": mean_abs}).sort_values(
            "mean_abs_shap", ascending=False
        )
<<<<<<< HEAD
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        shap_df.to_csv(LOGS_DIR / "shap_importance.csv", index=False)
        print(f"💾 Wrote SHAP importances → {LOGS_DIR}/shap_importance.csv")
=======
        shap_df.to_csv("logs/shap_importance.csv", index=False)
        print("💾 Wrote SHAP importances → logs/shap_importance.csv")
>>>>>>> f1644007

    except ModuleNotFoundError:
        print("ℹ️ SHAP not installed — falling back to permutation importance.")
        try:
            from sklearn.inspection import permutation_importance

            pi = permutation_importance(best_model, X, y, n_repeats=5, random_state=42, n_jobs=-1)
            mean_abs = np.abs(pi.importances_mean)
            shap_df = pd.DataFrame(
                {"feature": list(X.columns), "mean_abs_shap": mean_abs}
            ).sort_values("mean_abs_shap", ascending=False)
<<<<<<< HEAD
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            shap_df.to_csv(LOGS_DIR / "shap_importance.csv", index=False)
            print(f"💾 Wrote permutation importance → {LOGS_DIR}/shap_importance.csv")
=======
            shap_df.to_csv("logs/shap_importance.csv", index=False)
            print("💾 Wrote permutation importance → logs/shap_importance.csv")
>>>>>>> f1644007
        except Exception as e2:
            print(f"⚠️ Permutation importance also failed: {e2}")
    except Exception as e:
        print(f"⚠️ Importance export skipped: {e}")

    selected_cols = list(X.columns)
    try:
        inspector = pipe_for_introspection if "pipe_for_introspection" in locals() else None
        if inspector is not None and hasattr(inspector, "named_steps"):
            kb = inspector.named_steps.get("kbest", None)
            if kb is not None and hasattr(kb, "get_support"):
                mask = kb.get_support()
                if hasattr(mask, "__len__") and len(mask) == X.shape[1]:
                    selected_cols = list(X.columns[mask])
                else:
                    print("⚠️ KBest mask shape mismatch; using all input columns.")
    except Exception as e:
        print(f"⚠️ Could not extract KBest-selected columns ({e}); using all input columns.")

<<<<<<< HEAD
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    pd.Series(selected_cols, dtype=str).to_csv(
        MODELS_DIR / "selected_features.txt", index=False, header=False
    )
    print(
        f"💾 Selected feature columns saved to {MODELS_DIR}/selected_features.txt ({len(selected_cols)} cols)"
=======
    os.makedirs("models", exist_ok=True)
    pd.Series(selected_cols, dtype=str).to_csv(
        "models/selected_features.txt", index=False, header=False
    )
    print(
        f"💾 Selected feature columns saved to models/selected_features.txt ({len(selected_cols)} cols)"
>>>>>>> f1644007
    )

    y_pred = best_model.predict(X)
    print("\n📈 Training Evaluation Metrics (encoded labels 0..K-1):")
    print(f"Accuracy:  {accuracy_score(y, y_pred):.4f}")
    print(f"Precision: {precision_score(y, y_pred, average='weighted'):.4f}")
    print(f"Recall:    {recall_score(y, y_pred, average='weighted'):.4f}")
    print(f"F1 Score:  {f1_score(y, y_pred, average='weighted'):.4f}")

    y_pred_orig = pd.Series(y_pred).map(inv_label_map)
    y_true_orig = pd.Series(y).map(inv_label_map)
    cls_order_enc = list(getattr(best_model, "classes_", sorted(pd.Series(y).unique())))
    cls_order_orig = [inv_label_map[c] for c in cls_order_enc]
    target_names = [str(c) for c in cls_order_orig]

    print("\n📊 Predicted class counts (original labels):")
    print(y_pred_orig.value_counts())
    print("\n🧾 Classification report (original labels):")
    print(
        classification_report(
            y_true_orig,
            y_pred_orig,
            labels=cls_order_orig,
            target_names=target_names,
            zero_division=0,
            digits=4,
        )
    )

    print("\n🧩 Confusion matrix (rows=true, cols=pred) — original labels order:")
    print(confusion_matrix(y_true_orig, y_pred_orig, labels=cls_order_orig))

    try:
        proba = best_model.predict_proba(X)
        print("\n🎯 Average Precision (PR-AUC) per class (original labels):")
        for i, _cls_enc in enumerate(cls_order_enc):
            cls_orig = cls_order_orig[i]
            ap = average_precision_score((y_true_orig == cls_orig).astype(int), proba[:, i])
            print(f"AP (class={cls_orig}): {ap:.4f}")

        pos_orig = 1
        pos_enc = {v: k for k, v in inv_label_map.items()}[pos_orig]
        col_idx = cls_order_enc.index(pos_enc)
        p_pos = proba[:, col_idx]
        y_pos = (y == pos_enc).astype(int)
        ts = np.linspace(0.05, 0.95, 19)
        best_t = 0.50
        best_f1 = -1.0
        best_prec = 0.0
        best_rec = 0.0
        for t_ in ts:
            y_hat = (p_pos >= t_).astype(int)
            f1 = f1_score(y_pos, y_hat, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t_
                best_prec = precision_score(y_pos, y_hat, zero_division=0)
                best_rec = recall_score(y_pos, y_hat, zero_division=0)

<<<<<<< HEAD
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with open(MODELS_DIR / "thresholds.json", "w") as f:
=======
        with open("models/thresholds.json", "w") as f:
>>>>>>> f1644007
            json.dump(
                {
                    "pos_orig": int(pos_orig),
                    "pos_enc": int(pos_enc),
                    "proba_col_index": int(col_idx),
                    "threshold": float(best_t),
                    "metric": "f1_crash_only",
                    "precision_on_train": float(best_prec),
                    "recall_on_train": float(best_rec),
                    "f1_on_train": float(best_f1),
                },
                f,
                indent=2,
            )
        print(
<<<<<<< HEAD
            f"💾 Saved decision threshold → {MODELS_DIR}/thresholds.json: "
=======
            f"💾 Saved decision threshold → models/thresholds.json: "
>>>>>>> f1644007
            f"t={best_t:.3f} (Crash: P={best_prec:.3f}, R={best_rec:.3f}, F1={best_f1:.3f})"
        )
    except Exception as e:
        print(f"⚠️ AP (per-class) skipped: {e}")

    grid_results = pd.DataFrame(grid_search.cv_results_)
<<<<<<< HEAD
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    grid_results.to_csv(LOGS_DIR / "gridsearch_xgb_results.csv", index=False)
    print(f"📊 Grid search results saved to {LOGS_DIR}/gridsearch_xgb_results.csv")
=======
    grid_results.to_csv("logs/gridsearch_xgb_results.csv", index=False)
    print("📊 Grid search results saved to logs/gridsearch_xgb_results.csv")
>>>>>>> f1644007
    return True


# ====================== CLI ======================
if __name__ == "__main__":
    print("📥 Loading SPY data...")
    df = load_SPY_data()
    try:
        success = train_best_xgboost_model(df)
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise SystemExit(f"[train] hard failure: {e}") from e

    if success:
        from predict import run_predictions

        run_predictions()  # predict.py
