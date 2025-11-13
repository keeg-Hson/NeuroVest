#!/usr/bin/env python3
"""
train_improved.py - Enhanced training with accuracy optimizations

Key improvements over train.py:
1. Enhanced hyperparameter search with better regularization
2. Improved feature selection with tree-based pre-filtering
3. Better sample weighting with recency and volatility adjustments
4. Advanced cross-validation with proper embargo periods
5. Optimized threshold selection for precision-recall balance
"""

from dotenv import load_dotenv

load_dotenv(".env", override=True)

import json  # noqa: E402
import os  # noqa: E402
from copy import deepcopy  # noqa: E402
from datetime import datetime  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from imblearn.pipeline import Pipeline  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV, calibration_curve  # noqa: E402
from sklearn.ensemble import ExtraTreesClassifier  # noqa: E402
from sklearn.feature_selection import (  # noqa: E402
    SelectFromModel,
    SelectKBest,
    VarianceThreshold,
    mutual_info_classif,
)
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

# Import from existing modules
from config import MODELS_DIR, TRAIN_CFG  # noqa: E402
from train import (  # noqa: E402
    FWD_BLACKLIST,
    _cv_or_holdout,
    _iter_splits,
    _n_splits,
    _safe_smote_from_fold,
    _write_split_meta,
)
from utils import (  # noqa: E402
    add_features,
    add_forward_returns_and_labels,
    compute_sample_weights,
    ensure_no_future_leakage,
    finalize_features,
    load_SPY_data,
)

os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)


def enhanced_threshold_selection(pipe, X, y, cv, pos_label=1):
    """
    Improved threshold selection with:
    - Wider search range
    - Precision floor to avoid degenerate models
    - Balanced F1 optimization
    """
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
        raise RuntimeError("OOF builder produced no predictions.")

    y_pos = (np.asarray(y)[mask] == pos_label).astype(int)
    p = proba_oof[mask]

    # Enhanced threshold search with precision floor
    ts = np.linspace(0.30, 0.75, 46)  # Wider range, finer granularity
    best_t, best_f1, best_prec, best_rec = 0.50, -1.0, 0.0, 0.0
    min_precision = 0.45  # Higher precision floor to improve accuracy

    results = []
    for t_ in ts:
        y_hat = (p >= t_).astype(int)
        prec = precision_score(y_pos, y_hat, zero_division=0)
        rec = recall_score(y_pos, y_hat, zero_division=0)

        # Only consider thresholds with acceptable precision
        if prec < min_precision:
            continue

        f1 = f1_score(y_pos, y_hat, zero_division=0)

        # Prefer balanced precision/recall with slight precision bias
        balance_penalty = abs(prec - rec) / (prec + rec + 1e-9)
        # Add slight precision bias (better to miss trades than take bad ones)
        precision_bonus = 0.1 * (prec - 0.5) if prec > 0.5 else 0
        adjusted_f1 = f1 * (1.0 - 0.15 * balance_penalty) + precision_bonus

        results.append(
            {
                "threshold": t_,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "adjusted_f1": adjusted_f1,
            }
        )

        if adjusted_f1 > best_f1:
            best_f1 = f1  # Store original F1, not adjusted
            best_t = t_
            best_prec = prec
            best_rec = rec

    if results:
        print("\n[Threshold Search] Top 5 candidates:")
        results_df = pd.DataFrame(results).sort_values("adjusted_f1", ascending=False).head(5)
        print(results_df.to_string(index=False))

    return best_t, {
        "precision": float(best_prec),
        "recall": float(best_rec),
        "f1": float(best_f1),
        "proba_col_index": int(col_idx if col_idx is not None else 1),
        "pos_enc": int(pos_label),
    }


def train_improved_model(df: pd.DataFrame) -> bool:
    """
    Enhanced training with accuracy-focused improvements.
    """
    print("\n" + "=" * 80)
    print("ENHANCED TRAINING PIPELINE - ACCURACY OPTIMIZATION")
    print("=" * 80)

    print("\n📊 Generating enhanced features...")
    df, all_feature_cols = add_features(df)

    N = len(df)
    MIN_VALID_ROWS = max(5, min(60, int(N * 0.40)))

    feature_cols = [
        c for c in all_feature_cols if c in df.columns and df[c].notna().sum() >= MIN_VALID_ROWS
    ]

    if not feature_cols:
        raise RuntimeError("No features available after filtering.")

    feature_cols = list(dict.fromkeys(feature_cols))
    print(f"✅ Initial feature set: {len(feature_cols)} features")

    # Clean features BEFORE labeling
    df = finalize_features(df, feature_cols)

    # Ensure Close exists for labeling
    try:
        _raw = load_SPY_data()
        _raw_idxed = _raw["Close"].astype(float)
        df.index = pd.to_datetime(df.index, errors="coerce")
        _raw_idxed.index = pd.to_datetime(_raw_idxed.index, errors="coerce")
        df["Close"] = _raw_idxed.reindex(df.index)
    except Exception as e:
        if "Close" not in df.columns:
            raise RuntimeError(f"Could not attach Close for labeling: {e}") from e

    df = df.dropna(subset=["Close"])
    feature_cols = [c for c in feature_cols if c in df.columns and df[c].notna().any()]

    # Forward-returns labeling
    df = df.replace([np.inf, -np.inf], np.nan)

    print(
        f"\n[LABELING] horizon={TRAIN_CFG['horizon']}d, "
        f"pos_threshold={TRAIN_CFG['pos_threshold']:.4f}"
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

    INPUT_SCHEMA_FPATH = "models/input_features_fwd_improved.txt"

    def _clean_names(names):
        return [c for c in names if c not in FWD_BLACKLIST]

    input_cols = _clean_names([c for c in feature_cols if c in df.columns])

    for c in input_cols:
        if c not in df.columns:
            df[c] = np.nan

    X = df[input_cols].astype(float).replace([np.inf, -np.inf], np.nan)
    y = df["y"].astype(int)
    mask_ok = pd.Series(y).notna()
    X, y = X.loc[mask_ok], y.loc[mask_ok]

    if any(c in FWD_BLACKLIST for c in X.columns):
        raise RuntimeError(f"Leaky features detected: {set(X.columns) & FWD_BLACKLIST}")

    ensure_no_future_leakage(df, list(X.columns), ["y"], horizon_col="horizon_forward")

    # Save schema
    pd.Series(list(X.columns), dtype=str).to_csv(INPUT_SCHEMA_FPATH, index=False, header=False)
    print(f"💾 Saved input schema → {INPUT_SCHEMA_FPATH} ({len(X.columns)} cols)")

    # Class distribution analysis
    print("\n📊 Class Distribution:")
    print(f"  Total samples: {len(y)}")
    print(f"  Positive (y=1): {(y == 1).sum()} ({100 * (y == 1).sum() / len(y):.1f}%)")
    print(f"  Negative (y=0): {(y == 0).sum()} ({100 * (y == 0).sum() / len(y):.1f}%)")

    # Setup CV
    tscv_local = _cv_or_holdout(len(X), embargo=3, min_train_floor=30)  # Increased embargo
    n_folds = _n_splits(tscv_local, len(X))

    if n_folds == 0:
        if len(X) >= 2:
            tr = np.arange(0, len(X) - 1)
            te = np.array([len(X) - 1])
            tscv_local = [(tr, te)]
            n_folds = 1
        else:
            raise RuntimeError("Not enough rows to train.")

    print(f"\n✅ Using {n_folds}-fold time-series CV with embargo=3")
    _write_split_meta(X.index, tscv_local, out_path=MODELS_DIR / "split_meta_improved.json")

    # SMOTE check
    try:
        _, smote_step = _safe_smote_from_fold(y, tscv_local)
    except Exception:
        smote_step = "passthrough"

    # Enhanced XGBoost configuration
    xgb_common = dict(
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        tree_method="hist",
        use_label_encoder=False,
        early_stopping_rounds=75,  # Increased for better convergence
    )
    xgb_obj = dict(objective="binary:logistic", eval_metric="logloss")

    # Feature selection with tree-based pre-filtering
    use_kbest = X.shape[1] >= 2
    if use_kbest:
        max_k = X.shape[1]
        # More conservative feature selection
        k_choices = sorted(
            set([min(20, max_k), min(30, max_k), min(40, max_k), max(15, max_k // 2), max_k])
        )
        k_choices = [k for k in k_choices if 10 <= k <= max_k]
        print(f"🔧 Feature selection k_choices: {k_choices}")

        # Use tree-based pre-filtering for feature importance
        use_tree_prefilter = X.shape[1] > 50
        if use_tree_prefilter:
            print("🌲 Using ExtraTrees for feature pre-filtering")
            tree_selector = SelectFromModel(
                ExtraTreesClassifier(
                    n_estimators=150,  # More trees for better importance estimates
                    max_depth=6,
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced",  # Handle class imbalance
                ),
                threshold="median",
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

        # ENHANCED hyperparameter grid focused on accuracy
        param_grid = {
            "kbest__k": k_choices,
            # More estimators for better learning
            "clf__n_estimators": [400, 600, 800],
            # Controlled depth to prevent overfitting
            "clf__max_depth": [4, 5, 6, 7],
            # Lower learning rates for better generalization
            "clf__learning_rate": [0.01, 0.02, 0.03, 0.05],
            # Stronger regularization via subsampling
            "clf__subsample": [0.7, 0.8],
            "clf__colsample_bytree": [0.7, 0.8],
            # Increased min_child_weight to reduce overfitting
            "clf__min_child_weight": [5, 7, 10, 15],
            # Gamma for complexity penalty
            "clf__gamma": [0, 0.5, 1.0, 2.0],
            # L1 regularization (feature selection)
            "clf__reg_alpha": [0, 0.1, 0.5, 1.0],
            # L2 regularization (weight decay)
            "clf__reg_lambda": [1.0, 2.0, 3.0, 5.0],
            # Scale positive weight for class imbalance
            "clf__scale_pos_weight": [1.0, 1.5, 2.0],
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
            "clf__n_estimators": [400, 600, 800],
            "clf__max_depth": [4, 5, 6, 7],
            "clf__learning_rate": [0.01, 0.02, 0.03, 0.05],
            "clf__subsample": [0.7, 0.8],
            "clf__colsample_bytree": [0.7, 0.8],
            "clf__min_child_weight": [5, 7, 10, 15],
            "clf__gamma": [0, 0.5, 1.0, 2.0],
            "clf__reg_alpha": [0, 0.1, 0.5, 1.0],
            "clf__reg_lambda": [1.0, 2.0, 3.0, 5.0],
            "clf__scale_pos_weight": [1.0, 1.5, 2.0],
        }

    grid_size = np.prod([len(v) for v in param_grid.values()])
    print(f"🔧 Hyperparameter grid size: {grid_size:,} combinations")

    # Enhanced sample weights
    sample_weight_profit = compute_sample_weights(
        df.loc[X.index],
        min_weight=TRAIN_CFG["min_weight"],
        max_weight=TRAIN_CFG["max_weight"],
        power=TRAIN_CFG["weight_power"],
        long_only=TRAIN_CFG["long_only"],
    )

    print(
        f"\n🧮 Sample weight stats → "
        f"min={float(sample_weight_profit.min()):.3f}, "
        f"max={float(sample_weight_profit.max()):.3f}, "
        f"mean={float(sample_weight_profit.mean()):.3f}"
    )

    # Grid search with F1 optimization
    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="f1",  # Optimize F1 directly
        cv=tscv_local,
        n_jobs=-1,
        verbose=2,
        error_score=0,
        refit=True,
    )

    print(f"\n[{datetime.now():%H:%M:%S}] Starting GridSearchCV...")
    print(f"Searching through {grid_size:,} combinations for optimal hyperparameters...")
    print("This may take several hours depending on your system...")
    grid_search.fit(X, y)
    print(f"[{datetime.now():%H:%M:%S}] GridSearchCV completed!")

    print("\n✅ Best Parameters:")
    for k, v in grid_search.best_params_.items():
        print(f"  {k}: {v}")
    print(f"\n🎯 Best CV Score (F1): {grid_search.best_score_:.4f}")

    best_model = grid_search.best_estimator_

    # Re-fit with enhanced sample weighting
    from sklearn.utils.class_weight import compute_sample_weight as _csw

    y_pred0 = best_model.predict(X)
    # Weight errors more heavily + class balance + profit weights
    w_miss = 1.0 + 3.0 * (y_pred0 != y).astype(float)  # Increased error penalty
    w_bal = _csw(class_weight="balanced", y=y)
    w_final = w_miss * w_bal * sample_weight_profit

    # Re-fit without SMOTE (already balanced)
    best_model_wn = deepcopy(best_model)
    if hasattr(best_model_wn, "steps"):
        steps_map = dict(best_model_wn.steps)
        if "smote" in steps_map:
            best_model_wn.set_params(smote="passthrough")

    print("\n🔄 Re-fitting with enhanced sample weights...")
    best_model_wn.fit(X, y, **{"clf__sample_weight": w_final})

    # Check if calibration will actually help (quality gate)
    p_base = best_model_wn.predict_proba(X)[:, 1]
    prob_true, prob_pred = calibration_curve(y, p_base, n_bins=10, strategy="uniform")
    calibration_error = np.mean(np.abs(prob_true - prob_pred))

    print(f"📊 Calibration error: {calibration_error:.4f}")

    if calibration_error < 0.03:
        # Model is already well-calibrated, skip calibration to avoid degradation
        print("✅ Model already well-calibrated (error < 0.03), skipping calibration")
        best_model = best_model_wn
    else:
        # Probability calibration
        print(f"📈 Applying calibration (error = {calibration_error:.4f} >= 0.03)...")
        try:
            cal = CalibratedClassifierCV(best_model_wn, cv=3, method="isotonic")
            cal.fit(X, y, sample_weight=w_final)
            print("✅ Isotonic calibration successful")
        except Exception as e:
            print(f"⚠️ Isotonic calibration failed ({e}) — using sigmoid")
            cal = CalibratedClassifierCV(best_model_wn, cv=3, method="sigmoid")
            cal.fit(X, y, sample_weight=w_final)

        best_model = cal

    # Save model
    MODEL_DIR = "models"
    model_path_fwd = os.path.join(MODEL_DIR, "market_crash_model_fwd_improved.pkl")

    payload = {"model": best_model, "features": list(X.columns)}
    joblib.dump(payload, model_path_fwd)
    print(f"\n💾 Model saved → {model_path_fwd}")

    # Save label map
    label_values = sorted(pd.Series(y).unique().tolist())
    label_map = {int(v): int(v) for v in label_values}
    inv_label_map = {int(v): int(v) for v in label_values}
    with open("models/label_map_fwd_improved.json", "w") as f:
        json.dump(
            {
                "label_map": {str(k): int(v) for k, v in label_map.items()},
                "inv_label_map": {str(k): int(v) for k, v in inv_label_map.items()},
            },
            f,
            indent=2,
        )
    print("💾 Label maps saved")

    # Enhanced threshold selection
    print("\n🎯 Performing enhanced OOF threshold optimization...")
    try:
        best_pipe_for_oof = grid_search.best_estimator_
        t_star, metr = enhanced_threshold_selection(
            best_pipe_for_oof, X, y, tscv_local, pos_label=1
        )
    except Exception as e:
        print(f"⚠️ OOF threshold selection failed ({e}) — using fallback")
        t_star, metr = (
            0.55,
            {"precision": 0.0, "recall": 0.0, "f1": 0.0, "proba_col_index": 1, "pos_enc": 1},
        )

    thr_payload = {
        "pos_orig": 1,
        "pos_enc": metr.get("pos_enc", 1),
        "proba_col_index": metr.get("proba_col_index", 1),
        "threshold": float(t_star),
        "metric": "f1_positive_only_oof_enhanced",
        "precision_oof": float(metr.get("precision", 0.0)),
        "recall_oof": float(metr.get("recall", 0.0)),
        "f1_oof": float(metr.get("f1", 0.0)),
    }

    with open("models/thresholds_fwd_improved.json", "w") as f:
        json.dump(thr_payload, f, indent=2)

    print("\n💾 Optimized threshold saved → models/thresholds_fwd_improved.json")
    print(f"   Threshold: {t_star:.3f}")
    print(f"   OOF Precision: {thr_payload['precision_oof']:.3f}")
    print(f"   OOF Recall: {thr_payload['recall_oof']:.3f}")
    print(f"   OOF F1: {thr_payload['f1_oof']:.3f}")

    # Training set evaluation
    y_pred = best_model.predict(X)

    print("\n" + "=" * 80)
    print("TRAINING SET EVALUATION")
    print("=" * 80)
    print(f"Accuracy:  {accuracy_score(y, y_pred):.4f}")
    print(f"Precision: {precision_score(y, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y, y_pred, zero_division=0):.4f}")
    print(f"F1 Score:  {f1_score(y, y_pred, zero_division=0):.4f}")

    print("\n📊 Classification Report:")
    print(classification_report(y, y_pred, target_names=["No-Trade", "Trade"], zero_division=0))

    print("\n🧩 Confusion Matrix:")
    cm = confusion_matrix(y, y_pred)
    print("              Predicted")
    print("              0 (No)  1 (Yes)")
    print(f"Actual  0    {cm[0, 0]:6d}  {cm[0, 1]:6d}")
    print(f"        1    {cm[1, 0]:6d}  {cm[1, 1]:6d}")

    # Save grid search results
    grid_results = pd.DataFrame(grid_search.cv_results_)
    grid_results.to_csv("logs/gridsearch_improved_results.csv", index=False)
    print("\n📊 Grid search results → logs/gridsearch_improved_results.csv")

    print("\n" + "=" * 80)
    print("✅ ENHANCED TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 80)

    return True


if __name__ == "__main__":
    print("📥 Loading SPY data...")
    df = load_SPY_data()

    try:
        success = train_improved_model(df)
        if success:
            print("\n🎉 Model training completed successfully!")
            print("\nNext steps:")
            print("1. Review the improved metrics above")
            print("2. Run predictions with: python predict.py --backfill")
            print("3. Evaluate performance with: python evaluate.py")
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise SystemExit(f"Training failed: {e}") from e
