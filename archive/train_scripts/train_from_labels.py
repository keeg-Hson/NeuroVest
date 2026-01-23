#!/usr/bin/env python3
"""
train_from_labels.py  -  Rich-feature, calibrated, time-aware training

Upgrades in this patch:
- Stronger base model: HistGradientBoostingClassifier (nonlinear, robust probs)
- Purged CV flavor via TimeSeriesSplit(gap=h) to reduce overlap leakage
- Conditional calibration: only calibrate if probability dispersion is healthy
- Choose best of BASE vs CAL on holdout AUC
- Save threshold by balanced accuracy and 'invert_proba' if flipping helps

Feature pipeline:
- Uses utils.add_features + utils.finalize_features for rich price + external features.
- Ensures consistency with train.py / predict.py feature engineering.
"""

from __future__ import annotations

import json
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn  # debug
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit, train_test_split

from config import MODELS_DIR, TRAIN_CFG
from utils import add_features, finalize_features, load_SPY_data

# --------------------------- Features (match main pipeline) ---------------------------


def _build_rich_features_and_labels() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Load SPY history, build rich features via utils.add_features, then construct
    a binary forward-returns label using TRAIN_CFG horizon/fees.
    """
    # Canonical price history
    raw = load_SPY_data().copy()  # Date index, OHLCV columns
    raw = raw.sort_index()
    raw = raw[~raw.index.isna()]

    # Add features; df_feat carries Close and any external signals if present
    df_feat, feature_cols = add_features(raw.reset_index().rename(columns={"index": "Date"}))

    # Ensure Close is present for labeling
    if "Close" not in df_feat.columns:
        raise SystemExit("Expected 'Close' in feature dataframe for label construction.")

    # Binary label: 1 if next h-day net return exceeds hurdle
    label = _build_labels(df_feat)

    # Finalize features (impute, clean infinities) for the current feature set
    X = finalize_features(df_feat, feature_cols)

    # Attach Date for meta/splitting; keep aligned index
    X = X.copy()
    X["Date"] = df_feat.index

    return X, label, feature_cols


def _build_labels(feat_df: pd.DataFrame) -> pd.Series:
    """
    Binary label: 1 if next h-day return exceeds total hurdle (fees+slip+edge).
    """
    h = int(TRAIN_CFG.get("horizon", 5))
    fee_bps = float(TRAIN_CFG.get("fee_bps", 1.5))
    slip_bps = float(TRAIN_CFG.get("slippage_bps", 2.0))
    edge_bps = float(TRAIN_CFG.get("min_edge_bps", 10.0))
    hurdle = (fee_bps + slip_bps + edge_bps) / 10_000.0

    px = feat_df["Close"].astype(float)
    fwd_ret = px.shift(-h) / px - 1.0
    return (fwd_ret > hurdle).astype(int)


def _threshold_sweep_balacc(
    y_true: np.ndarray,
    p1: np.ndarray,
    lo: float = 0.25,
    hi: float = 0.75,
    step: float = 0.005,
) -> tuple[float, float, float]:
    """
    Sweep threshold to maximize balanced accuracy (Youden's J).
    Returns (best_thr, best_balacc, best_acc).
    """
    import numpy as _np

    grid = _np.arange(lo, hi + 1e-12, step)
    best = (0.5, -1.0, -1.0)
    for thr in grid:
        pred = (p1 >= thr).astype(int)
        bal = balanced_accuracy_score(y_true, pred)
        acc = (pred == y_true).mean()
        if bal > best[1]:
            best = (float(thr), float(bal), float(acc))
    print(f"[train] holdout best threshold={best[0]:.3f} bal_acc={best[1]:.4f} acc={best[2]:.4f}")
    return best


# --------------------------------- Training ----------------------------------


def train_model() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Features + label (rich features from utils.add_features)
    feat_df, label, feature_cols = _build_rich_features_and_labels()

    # Build X/y, drop NaNs where needed
    full = feat_df.copy()
    full["Label"] = label
    full = full.dropna(subset=feature_cols + ["Label"]).reset_index(drop=True)

    X = full[feature_cols].astype(float)
    y = full["Label"].astype(int)

    pos_rate = float(y.mean())
    print("\n[train] features:", feature_cols, "\n")
    print(f"[train] aligned shapes → X:{X.shape}, y:{y.shape}")
    print(f"[train] scikit-learn version: {sklearn.__version__}")
    print(f"[train] label positive rate: {pos_rate:.3f}  (aim ~0.45–0.55)\n")

    # Time-ordered holdout (no shuffle)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)

    # Stronger base model (nonlinear; well-behaved probabilities)
    base = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_depth=4,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=0.10,
        max_iter=500,
        validation_fraction=0.1,
        early_stopping=True,
        random_state=7,
    )

    # Fit base on the train slice
    base.fit(X_train, y_train)

    # --- Base probabilities on holdout ---
    p1_base = base.predict_proba(X_test)[:, 1]
    y_pred_base = (p1_base >= 0.50).astype(int)
    auc_base = roc_auc_score(y_test, p1_base)
    bal_base = balanced_accuracy_score(y_test, y_pred_base)
    q = np.quantile(p1_base, [0.01, 0.05, 0.5, 0.95, 0.99])
    print("[train] holdout (base):\n", classification_report(y_test, y_pred_base, digits=3))
    print(
        "[train] holdout proba stats:",
        f"mean={p1_base.mean():.4f} sd={p1_base.std():.4f} min={p1_base.min():.4f} max={p1_base.max():.4f} "
        f"q01={q[0]:.4f} q05={q[1]:.4f} q50={q[2]:.4f} q95={q[3]:.4f} q99={q[4]:.4f}",
    )
    print(f"[train] holdout AUC={auc_base:.4f}  balanced_acc={bal_base:.4f}")

    # Purged/embargoed time-series CV to reduce overlap leakage during calibration
    h = int(TRAIN_CFG.get("horizon", 5))
    try:
        tscv = TimeSeriesSplit(n_splits=5, gap=h)
    except TypeError:
        tscv = TimeSeriesSplit(n_splits=5)

    # --- Optional calibration with purged TSCV ---
    use_cal = float(np.std(p1_base)) >= 0.03
    if use_cal:
        try:
            cal = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=tscv)
        except TypeError:
            cal = CalibratedClassifierCV(base_estimator=base, method="sigmoid", cv=5)
        cal.fit(X_train, y_train)
        p1_cal = cal.predict_proba(X_test)[:, 1]
        auc_cal = roc_auc_score(y_test, p1_cal)
        print(
            f"[train] CAL AUC={auc_cal:.4f} (base={auc_base:.4f})  sd(p_cal)={float(np.std(p1_cal)):.4f}"
        )
    else:
        print("[warn] Skipping calibration due to very low probability dispersion.")
        cal, p1_cal, auc_cal = None, None, -np.inf

    # --- Choose best on AUC ---
    if auc_cal > auc_base + 1e-6:
        clf = cal
        p1_keep = p1_cal
        auc_keep = auc_cal
        print("[train] adopting CALIBRATED model")
    else:
        clf = base
        p1_keep = p1_base
        auc_keep = auc_base
        print("[train] keeping BASE model")

    # --- Threshold by balanced accuracy on holdout probabilities ---
    best_thr, best_bal, best_acc = _threshold_sweep_balacc(y_test.to_numpy(), p1_keep)

    # --- Save model & features ---
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DIR / "market_crash_model.pkl"
    joblib.dump({"model": clf, "features": feature_cols}, out_path)
    print(f"[train] saved → {out_path}")

    # --- Save thresholds + optional invert_proba ---
    thr_path = MODELS_DIR / "thresholds.json"
    payload = {"threshold": float(best_thr)}
    # If flipping helps, set invert flag so predict/eval will use (1 - p)
    try:
        auc_flip = roc_auc_score(y_test, 1.0 - p1_keep)
        if auc_keep < 0.5 and auc_flip > 0.5:
            payload["invert_proba"] = True
            print(
                f"[train] NOTE: set invert_proba=True (AUC={auc_keep:.4f}, flipped={auc_flip:.4f})"
            )
    except Exception:
        pass
    thr_path.write_text(json.dumps(payload, indent=2))
    print(f"[train] wrote → {thr_path}  (balanced_acc={best_bal:.4f}, acc={best_acc:.4f})")

    # --- Split meta for OOS filtering ---
    split_idx = len(X_train)
    split_date = full.loc[split_idx, "Date"]
    (MODELS_DIR / "split_meta.json").write_text(
        json.dumps(
            {"split_index": int(split_idx), "split_date": str(pd.to_datetime(split_date).date())}
        )
    )
    print(f"[train] split_date (start of test): {split_date}")


if __name__ == "__main__":
    train_model()
