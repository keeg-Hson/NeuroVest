# train.py
from dotenv import load_dotenv

load_dotenv(".env", override=True)

import json
import os
import socket
import warnings
from datetime import datetime

import joblib
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

socket.setdefaulttimeout(float(os.getenv("NET_TIMEOUT", 3)))

import numpy as np
from sklearn.model_selection import BaseCrossValidator

from config import TRAIN_CFG
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


# === Public entry points expected by run_all.py ======================================
def train_model(models=None, fast=False):
    df = load_SPY_data()
    return train_best_xgboost_model(df)


def run(models=None, fast=False):
    return train_model(models=models, fast=fast)


def main(models=None, fast=False):
    ok = train_model(models=models, fast=fast)
    return 0 if ok else 1


# =====================================================================================

# --- tolerate unknown CLI args when invoked like: python -m train --models xgb
import sys

if len(sys.argv) > 1:
    sys.argv = sys.argv[:1]


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
        start_test = self.min_train_size
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
        start_test = self.min_train_size
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


# Quiet warnings
warnings.filterwarnings(
    "ignore", message=r"\[.*\] WARNING: .*Parameters: { \"use_label_encoder\" } are not used\."
)
xgb.set_config(verbosity=0)


# Output dirs
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)


def _build_adaptive_cv(n_rows: int) -> BaseCrossValidator:
    min_train = max(200, int(0.2 * n_rows))
    n_splits = min(5, max(3, (n_rows - min_train) // 100))
    embargo = min(10, max(2, n_rows // 150))
    remainder = max(1, n_rows - min_train)
    test_size = max(25, remainder // max(1, n_splits))
    return PurgedWalkForwardSplit(
        n_splits=n_splits,
        min_train_size=min_train,
        test_size=test_size,
        embargo=embargo,
    )


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


from sklearn.base import clone


def pick_threshold_from_oof(pipe, X, y, cv, pos_label=1):
    """
    Time-series-safe OOF: iterate your CV splits, fit on train, predict_proba on test,
    fill a single out-of-fold vector (no sample predicted more than once), then pick t*.
    """
    n = len(X)
    proba_oof = np.full(n, np.nan, dtype=float)
    seen = np.zeros(n, dtype=bool)
    classes_seen = None
    col_idx = None

    # Use your existing generator
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

        # If a sample shows up twice (shouldn't), keep the first prediction
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

    ts = np.linspace(0.05, 0.95, 19)
    best_t, best_f1, best_prec, best_rec = 0.50, -1.0, 0.0, 0.0
    for t_ in ts:
        y_hat = (p >= t_).astype(int)
        f1 = f1_score(y_pos, y_hat, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t_
            best_prec = precision_score(y_pos, y_hat, zero_division=0)
            best_rec = recall_score(y_pos, y_hat, zero_division=0)

    return best_t, {
        "precision": float(best_prec),
        "recall": float(best_rec),
        "f1": float(best_f1),
        "proba_col_index": int(col_idx if col_idx is not None else 1),
        "pos_enc": int(pos_label),
    }


def train_best_xgboost_model(df: pd.DataFrame) -> bool:
    print("\n📊 Generating features...")
    df, all_feature_cols = add_features(df)

    # Read top signals (optional)
    try:
        with open("logs/top_signals.txt") as f:
            raw_top = [
                line.strip().split(",")[0]
                for line in f.readlines()
                if line.strip() and not line.startswith("Top")
            ]
        print(f"✅ Loaded top signals: {raw_top}")
    except FileNotFoundError:
        print("⚠️ logs/top_signals.txt not found. Will fall back to generated feature list.")
        raw_top = []

    # Coverage thresholds scale with data size
    N = len(df)
    MIN_VALID_ROWS = max(5, min(60, int(N * 0.40)))
    FORCE_MIN_ROWS = max(5, min(50, int(N * 0.30)))

    top_signals = [s for s in raw_top if s in df.columns and df[s].notna().sum() >= MIN_VALID_ROWS]
    print(f"✅ Filtered top signals with data: {top_signals}")

    feature_cols = [c for c in all_feature_cols if c in top_signals]
    print(f"🧪 Using top correlated signals only: {feature_cols}")

    forced_externals = [
        "Sector_MedianRet_20",
        "Sector_Dispersion_20",
        "Credit_Spread_20",
        "TNX_Change_20",
        "DXY_Change_20",
        "News_Sent_Z20",
        "Reddit_Sent_Z20",
    ]
    forced_available = [
        c for c in forced_externals if c in df.columns and df[c].notna().sum() >= FORCE_MIN_ROWS
    ]

    if not feature_cols:
        print("⚠️ No top_signals survived coverage filter — fallback to all_feature_cols + forced.")
        feature_cols = [
            c for c in all_feature_cols if c in df.columns and df[c].notna().sum() >= MIN_VALID_ROWS
        ]

    feature_cols = feature_cols + [c for c in forced_available if c not in feature_cols]

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
        raise RuntimeError(
            "No features available after dynamic fallback. "
            "Refresh data to increase history, or loosen coverage thresholds."
        )

    feature_cols = list(dict.fromkeys(feature_cols))
    print(f"➕ Force-available externals: {forced_available}")
    print(
        f"🧪 Final candidate features ({len(feature_cols)}): {feature_cols[:20]}{'...' if len(feature_cols) > 20 else ''}"
    )

    # CLEAN features BEFORE labeling/splitting
    df = finalize_features(df, feature_cols)

    # === Ensure Close exists for labeling ===
    import pandas as _pd

    try:
        _raw = load_SPY_data()
        _raw_idxed = _raw["Close"].astype(float)
        df.index = _pd.to_datetime(df.index, errors="coerce")
        _raw_idxed.index = _pd.to_datetime(_raw_idxed.index, errors="coerce")
        df["Close"] = _raw_idxed.reindex(df.index)
    except Exception as _e:
        if "Close" not in df.columns:
            raise RuntimeError(f"Could not attach Close for labeling: {_e}")
    df = df.dropna(subset=["Close"])
    # ========================================

    # Recompute usable cols after finalize
    feature_cols = [c for c in feature_cols if c in df.columns and df[c].notna().any()]

    # -------- Forward-returns branch --------
    if TRAIN_CFG.get("use_forward_returns", False):
        df = df.replace([np.inf, -np.inf], np.nan)

        df = add_forward_returns_and_labels(
            df,
            price_col=TRAIN_CFG["price_col"],  # "Close"
            horizon=TRAIN_CFG["horizon"],
            fee_bps=TRAIN_CFG["fee_bps"],
            slippage_bps=TRAIN_CFG["slippage_bps"],
            long_only=TRAIN_CFG["long_only"],
            pos_threshold=TRAIN_CFG["pos_threshold"],
        )

        # Build/align CLEAN input schema (no forward-looking columns)
        INPUT_SCHEMA_FPATH = "models/input_features_fwd.txt"

        def _clean_names(names):
            return [c for c in names if c not in FWD_BLACKLIST]

        try:
            with open(INPUT_SCHEMA_FPATH) as f:
                input_cols = _clean_names([line.strip() for line in f if line.strip()])
                print(f"📄 Loaded prior input schema (cleaned) with {len(input_cols)} cols.")
        except Exception:
            # First run: start from available feature cols + harmless basics
            base = list(dict.fromkeys(feature_cols + [c for c in ["Close"] if c in df.columns]))
            input_cols = _clean_names([c for c in base if c in df.columns])

        # Enforce cleanliness and materialize any missing cols
        input_cols = _clean_names(input_cols)
        for c in input_cols:
            if c not in df.columns:
                df[c] = np.nan

        # Hard fail if any leaky column still present anywhere
        leaked = sorted(set(df.columns) & FWD_BLACKLIST)
        if leaked:
            # we can keep them in df for labeling, but never in X
            pass  # no-op: just ensuring we don't include them in input_cols

        X = df[input_cols].astype(float).replace([np.inf, -np.inf], np.nan)
        y = df["y"].astype(int)
        mask_ok = pd.Series(y).notna()
        X, y = X.loc[mask_ok], y.loc[mask_ok]

        if any(c in FWD_BLACKLIST for c in X.columns):
            raise RuntimeError(f"Leaky features detected in X: {set(X.columns) & FWD_BLACKLIST}")

        if len(X) < 2:
            raise RuntimeError(
                f"Not enough rows to train after NaN-filtering (have {len(X)}). "
                f"Check SPY.csv length and NaNs in selected features."
            )

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
        )
        xgb_obj = dict(objective="binary:logistic", eval_metric="logloss")

        use_kbest = X.shape[1] >= 2
        if use_kbest:
            max_k = X.shape[1]
            # Floor at 5 to avoid underfitting to a single feature
            k_choices = sorted(set([5, 8, 10, 12, max(5, max_k // 2), max_k]))
            k_choices = [k for k in k_choices if 5 <= k <= max_k]
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
                "clf__n_estimators": [150],
                "clf__max_depth": [3, 5],
                "clf__learning_rate": [0.05],
                "clf__subsample": [0.8, 1.0],
                "clf__colsample_bytree": [0.8, 1.0],
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
                "clf__n_estimators": [150],
                "clf__max_depth": [3, 5],
                "clf__learning_rate": [0.05],
                "clf__subsample": [0.8, 1.0],
                "clf__colsample_bytree": [0.8, 1.0],
            }

        sample_weight_profit = compute_sample_weights(
            df,
            min_weight=TRAIN_CFG["min_weight"],
            max_weight=TRAIN_CFG["max_weight"],
            power=TRAIN_CFG["weight_power"],
            long_only=TRAIN_CFG["long_only"],
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
        grid_search.fit(X, y)
        print(f"[{datetime.now():%H:%M:%S}] starting GridSearchCV...")
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

        from sklearn.calibration import CalibratedClassifierCV

        try:
            cal = CalibratedClassifierCV(best_model_wn, cv=3, method="isotonic")
            cal.fit(X, y)
        except Exception as e:
            print(f"⚠️ [FWD] Isotonic calibration failed ({e}) — falling back to sigmoid.")
            cal = CalibratedClassifierCV(best_model_wn, cv=3, method="sigmoid")
            cal.fit(X, y)

        best_model = cal

        MODEL_DIR = "models"
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path_fwd = os.getenv(
            "MODEL_PATH_FWD", os.path.join(MODEL_DIR, "market_crash_model_fwd.pkl")
        )
        joblib.dump(best_model, model_path_fwd)
        print(f"💾 [FWD] Model saved to {model_path_fwd}")

        label_values = sorted(pd.Series(y).unique().tolist())
        label_map = {int(l): int(l) for l in label_values}
        inv_label_map = {int(l): int(l) for l in label_values}
        with open("models/label_map_fwd.json", "w") as f:
            json.dump(
                {
                    "label_map": {str(k): int(v) for k, v in label_map.items()},
                    "inv_label_map": {str(k): int(v) for k, v in inv_label_map.items()},
                },
                f,
                indent=2,
            )
        print("💾 [FWD] Label maps → models/label_map_fwd.json")

        try:
            # Use the best gridsearch pipeline (pre reweight/refit) to get OOF probabilities
            best_pipe_for_oof = grid_search.best_estimator_
            t_star, metr = pick_threshold_from_oof(best_pipe_for_oof, X, y, tscv_local, pos_label=1)
        except Exception as e:
            print(f"⚠️ [FWD] OOF threshold selection failed ({e}) — falling back to 0.50.")
            t_star, metr = 0.50, {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "proba_col_index": 1,
                "pos_enc": 1,
            }

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
        with open("models/thresholds_fwd.json", "w") as f:
            json.dump(thr_payload, f, indent=2)
        print(
            f"💾 [FWD] Thresholds → models/thresholds_fwd.json: "
            f"t={t_star:.3f} (P_oof={thr_payload['precision_oof']:.3f}, "
            f"R_oof={thr_payload['recall_oof']:.3f}, F1_oof={thr_payload['f1_oof']:.3f})"
        )

        print("✅ [FWD] Forward-returns training completed.")
        print(
            "ℹ️ To use this model in predict.py, set MODEL_PATH to 'models/market_crash_model_fwd.pkl' and load thresholds from 'models/thresholds_fwd.json'."
        )
        return True

    # -------- Triple-barrier branch (unchanged) --------
    from utils import label_events_triple_barrier

    df["Volatility"] = df["Close"].rolling(window=20).std()
    print("\n🧪 Sample volatility (tail):")
    print(df["Volatility"].dropna().tail(10))

    df = label_events_triple_barrier(df, vol_col="ATR_14", pt_mult=1.0, sl_mult=1.0, t_max=10)

    print(
        (df if "Date" in df.columns else df.reset_index().rename(columns={"index": "Date"}))[
            ["Date", "Close", "Event"]
        ].tail(15)
    )
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
        print("❌ No data left after dropping NaNs. Check signal columns or event labeling.")
        return False

    df = df[df["Event"] != 0].copy()

    try:
        with open("models/input_features.txt") as f:
            input_cols = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(
            f"⚠️ models/input_features.txt missing or unreadable ({e}); using available columns as-is."
        )
        input_cols = [c for c in df.columns if c not in ("Event",)]

    for c in input_cols:
        if c not in df.columns:
            df[c] = 0.0

    X = df[input_cols].astype(float).replace([np.inf, -np.inf], np.nan)
    y_orig = df["Event"]

    os.makedirs("models", exist_ok=True)
    pd.Series(list(X.columns), dtype=str).to_csv(
        "models/input_features.txt", index=False, header=False
    )
    print("💾 Input feature columns saved to models/input_features.txt")

    label_values = sorted(y_orig.unique())
    label_map = {lab: i for i, lab in enumerate(label_values)}
    inv_label_map = {v: k for k, v in label_map.items()}
    y = y_orig.map(label_map).astype(int)

    pos = int((y == 1).sum()) if 1 in y.unique() else 1
    neg = int((y == 0).sum()) if 0 in y.unique() else 1
    spw = max(1.0, neg / max(1, pos))
    print(f"🔧 scale_pos_weight for XGB: {spw:.2f}")

    print(f"🔤 Label encoding map: {label_map} (train on 0..{y.nunique()-1})")
    print("\n📊 Encoded class distribution (0..K-1):")
    print(y.value_counts())

    with open("models/label_map.json", "w") as f:
        json.dump(
            {
                "label_map": {str(k): int(v) for k, v in label_map.items()},
                "inv_label_map": {str(k): int(v) for k, v in inv_label_map.items()},
            },
            f,
            indent=2,
        )
    print("💾 Saved label maps to models/label_map.json")

    MODEL_DIR = "models"
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "market_crash_model.pkl")

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

    if is_binary:
        xgb_obj = dict(objective="binary:logistic", eval_metric="logloss")
    else:
        xgb_obj = dict(objective="multi:softprob", eval_metric="mlogloss", num_class=n_classes)

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
            "clf__n_estimators": [150],
            "clf__max_depth": [3, 5],
            "clf__learning_rate": [0.05],
            "clf__subsample": [0.8, 1.0],
            "clf__colsample_bytree": [0.8, 1.0],
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
            "clf__n_estimators": [150],
            "clf__max_depth": [3, 5],
            "clf__learning_rate": [0.05],
            "clf__subsample": [0.8, 1.0],
            "clf__colsample_bytree": [0.8, 1.0],
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
    grid_search.fit(X, y)
    print(f"[{datetime.now():%H:%M:%S}] starting GridSearchCV...")
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
        steps = dict(best_model_wn.steps)
        if "smote" in steps:
            best_model_wn.set_params(smote="passthrough")
    best_model_wn.fit(X, y, **{"clf__sample_weight": w})

    pipe_for_introspection = best_model_wn

    from sklearn.calibration import CalibratedClassifierCV

    try:
        cal = CalibratedClassifierCV(best_model_wn, cv=3, method="isotonic")
        cal.fit(X, y)
    except Exception as e:
        print(f"⚠️ Isotonic calibration failed ({e}) — falling back to sigmoid.")
        cal = CalibratedClassifierCV(best_model_wn, cv=3, method="sigmoid")
        cal.fit(X, y)

    best_model = cal
    joblib.dump(best_model, model_path)

    # Importance export (shap or permutation)
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
                        list(np.array(list(X.columns))[mask])
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
            rs = np.random.RandomState(42)
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
        shap_df.to_csv("logs/shap_importance.csv", index=False)
        print("💾 Wrote SHAP importances → logs/shap_importance.csv")

    except ModuleNotFoundError:
        print("ℹ️ SHAP not installed — falling back to permutation importance.")
        try:
            from sklearn.inspection import permutation_importance

            pi = permutation_importance(best_model, X, y, n_repeats=5, random_state=42, n_jobs=-1)
            mean_abs = np.abs(pi.importances_mean)
            shap_df = pd.DataFrame(
                {"feature": list(X.columns), "mean_abs_shap": mean_abs}
            ).sort_values("mean_abs_shap", ascending=False)
            shap_df.to_csv("logs/shap_importance.csv", index=False)
            print("💾 Wrote permutation importance → logs/shap_importance.csv")
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

    os.makedirs("models", exist_ok=True)
    pd.Series(selected_cols, dtype=str).to_csv(
        "models/selected_features.txt", index=False, header=False
    )
    print(
        f"💾 Selected feature columns saved to models/selected_features.txt ({len(selected_cols)} cols)"
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
        for i, cls_enc in enumerate(cls_order_enc):
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

        with open("models/thresholds.json", "w") as f:
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
            f"💾 Saved decision threshold → models/thresholds.json: t={best_t:.3f} (Crash: P={best_prec:.3f}, R={best_rec:.3f}, F1={best_f1:.3f})"
        )
    except Exception as e:
        print(f"⚠️ AP (per-class) skipped: {e}")

    grid_results = pd.DataFrame(grid_search.cv_results_)
    grid_results.to_csv("logs/gridsearch_xgb_results.csv", index=False)
    print("📊 Grid search results saved to logs/gridsearch_xgb_results.csv")
    return True


if __name__ == "__main__":
    print("📥 Loading SPY data...")
    df = load_SPY_data()
    try:
        success = train_best_xgboost_model(df)
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise SystemExit(f"[train] hard failure: {e}")

    if success:
        from predict import run_predictions

        run_predictions()
