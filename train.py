# train.py

import os
import warnings
from datetime import datetime

import joblib
import pandas as pd

import xgboost as xgb
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, average_precision_score
)

from sklearn.impute import SimpleImputer




from utils import (
    load_SPY_data,
    add_features,
    finalize_features,
    label_events_volatility_adjusted,
    label_events_triple_barrier,
)


from sklearn.model_selection import BaseCrossValidator
import numpy as np

# === Public entry points expected by run_all.py ======================================
def train_model(models=None, fast=False):
    """
    Thin wrapper so run_all.py can call train.train_model(models=..., fast=...).
    We ignore 'models' (you only train XGB here) and 'fast' (unused).
    Returns True/False.
    """
    from utils import load_SPY_data
    df = load_SPY_data()
    return train_best_xgboost_model(df)


# Optional aliases — run_all.py will try these names too
def run(models=None, fast=False):
    return train_model(models=models, fast=fast)

def main(models=None, fast=False):
    ok = train_model(models=models, fast=fast)
    # match typical CLI convention
    return 0 if ok else 1
# =====================================================================================


# --- tolerate unknown CLI args when invoked like: python -m train --models xgb
import sys
if len(sys.argv) > 1:
    # Keep only the module name so argparse/unknown flags can't crash us
    sys.argv = sys.argv[:1]


class PurgedWalkForwardSplit(BaseCrossValidator):
    """
    Expanding-window, purged walk-forward CV.
    Guarantees: each fold has >= min_train_size train rows and >= 1 test row.
    get_n_splits dynamically matches what split() will yield for X.
    """
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
        # If X is provided, compute the *actual* number of folds we will yield
        if X is not None:
            return self._count_possible_splits(len(X))
        return self.n_splits



from sklearn.model_selection import BaseCrossValidator
import numpy as np

# --- tolerate unknown CLI args when invoked like: python -m train --models xgb
import sys
if len(sys.argv) > 1:
    # Make it behave like `python -m train` even if the runner passes flags
    sys.argv = sys.argv[:1]



# Quiet some noisy warnings
warnings.filterwarnings(
    "ignore",
    message=r"\[.*\] WARNING: .*Parameters: { \"use_label_encoder\" } are not used\."
)
xgb.set_config(verbosity=0)

# Create output folders
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)


def _build_adaptive_cv(n_rows: int) -> BaseCrossValidator:
    # 20% of data (>=200) for initial train, capped for tiny datasets
    min_train = max(200, int(0.2 * n_rows))
    # aim for 3–5 splits depending on size
    n_splits = min(5, max(3, (n_rows - min_train) // 100))
    # embargo ~ small gap to reduce bleed
    embargo = min(10, max(2, n_rows // 150))
    # test_size ~ chunk the remainder evenly
    remainder = max(1, n_rows - min_train)
    test_size = max(25, remainder // max(1, n_splits))

    return PurgedWalkForwardSplit(
        n_splits=n_splits,
        min_train_size=min_train,
        test_size=test_size,
        embargo=embargo
    )




def _min_minority_per_fold(y: pd.Series, cv: BaseCrossValidator) -> int:
    """Return the smallest minority-class count across all CV train folds."""
    min_count = np.inf
    idx = np.arange(len(y))
    for tr, _ in cv.split(idx):
        vc = y.iloc[tr].value_counts()
        if len(vc) < 2:
            # Only one class in this train fold → no SMOTE possible
            return 0
        min_count = min(min_count, int(vc.min()))
    return int(min_count if min_count != np.inf else 0)


def _safe_smote_from_fold(y: pd.Series, cv: BaseCrossValidator):
    """
    Decide whether to use SMOTE based on minority size in folds.
    Returns (use_smote: bool, smote_step_or_passthrough).
    """
    m = _min_minority_per_fold(y, cv)
    if m < 2:
        print(f"ℹ️ SMOTE disabled (min minority per fold = {m}).")
        return False, "passthrough"
    # SMOTE requires k_neighbors <= minority_count - 1
    k = max(1, min(5, m - 1))
    print(f"ℹ️ SMOTE enabled with k_neighbors={k} (min minority per fold={m}).")
    return True, SMOTE(random_state=42, k_neighbors=k)


def train_best_xgboost_model(df: pd.DataFrame) -> bool:
    print("\n📊 Generating features...")
    df, all_feature_cols = add_features(df)

    # --- Optional: read top_signals.txt produced by your selector ---
    try:
        with open("logs/top_signals.txt", "r") as f:
            raw_top = [
                line.strip().split(",")[0]
                for line in f.readlines()
                if line.strip() and not line.startswith("Top")
            ]
        print(f"✅ Loaded top signals: {raw_top}")
    except FileNotFoundError:
        print("⚠️ logs/top_signals.txt not found. Will fall back to generated feature list.")
        raw_top = []

    # --- Dynamic coverage thresholds (scale with data size) ---
    N = len(df)
    MIN_VALID_ROWS = max(5, min(60, int(N * 0.40)))
    FORCE_MIN_ROWS = max(5, min(50, int(N * 0.30)))

    top_signals = [
        s for s in raw_top
        if s in df.columns and df[s].notna().sum() >= MIN_VALID_ROWS
    ]
    print(f"✅ Filtered top signals with data: {top_signals}")

    feature_cols = [c for c in all_feature_cols if c in top_signals]
    print(f"🧪 Using top correlated signals only: {feature_cols}")

    # ---- Force-include a few external signals (if sufficiently covered) ----
    forced_externals = [
        "Sector_MedianRet_20", "Sector_Dispersion_20",
        "Credit_Spread_20", "TNX_Change_20", "DXY_Change_20",
        "News_Sent_Z20", "Reddit_Sent_Z20",
    ]
    forced_available = [
        c for c in forced_externals
        if c in df.columns and df[c].notna().sum() >= FORCE_MIN_ROWS
    ]

    if not feature_cols:
        print("⚠️ No top_signals survived coverage filter — fallback to all_feature_cols + forced.")
        feature_cols = [
            c for c in all_feature_cols
            if c in df.columns and df[c].notna().sum() >= MIN_VALID_ROWS
        ]

    feature_cols = feature_cols + [c for c in forced_available if c not in feature_cols]

    if not feature_cols:
        minimal_base = [
            "Daily_Return", "Return_Lag1", "Return_Lag3", "Return_Lag5",
            "ZMomentum", "RSI", "MACD", "MACD_Signal", "Stoch_K", "Stoch_D",
            "Gap_Pct", "Acceleration",
        ]
        feature_cols = [
            c for c in minimal_base
            if c in df.columns and df[c].notna().sum() >= 5
        ]

    if not feature_cols:
        raise RuntimeError(
            "No features available after dynamic fallback. "
            "Refresh data to increase history, or loosen coverage thresholds."
        )

    feature_cols = list(dict.fromkeys(feature_cols))
    print(f"➕ Force-available externals: {forced_available}")
    print(f"🧪 Final candidate features ({len(feature_cols)}): "
          f"{feature_cols[:20]}{'...' if len(feature_cols) > 20 else ''}")

    # --- CLEAN features BEFORE labeling/splitting ---
    df = finalize_features(df, feature_cols)

    # Volatility (for logging only)
    df["Volatility"] = df["Close"].rolling(window=20).std()
    print("\n🧪 Sample volatility (tail):")
    print(df["Volatility"].dropna().tail(10))

    # Label AFTER features are cleaned
    df = label_events_triple_barrier(df, vol_col="ATR_14", pt_mult=2.0, sl_mult=2.0, t_max=5)


    # Label sanity checks
    print(df[["Date", "Close", "Event"]].tail(15))
    print("\n📊 Distribution of Event labels (incl. NaNs):")
    print(df["Event"].value_counts(dropna=False))
    print("\n📊 Number of unique Events:")
    print(df["Event"].nunique(dropna=False))

    if df["Event"].nunique() <= 1:
        print("❌ Not enough class diversity in Event labels — training aborted.")
        return False

    # Drop rows missing Event or required features
    print("\n🧪 Missing values per feature column (post-clean):")
    print(df[feature_cols].isna().sum())
    print(f"\n🧪 Total rows before dropna: {len(df)}")

    valid_feature_cols = [c for c in feature_cols if df[c].notna().sum() > 0]
    required_cols = ["Event"] + valid_feature_cols
    df = df.dropna(subset=required_cols)

    print(f"\n🧹 Rows remaining after dropna: {len(df)}")
    if len(df) == 0:
        print("❌ No data left after dropping NaNs. Check signal columns or event labeling.")
        return False
    
    # Treat no-trade as a thresholding decision, not a class
    df = df[df["Event"] != 0].copy()

    # Features and ORIGINAL labels
    # ---- Align to train-time schema (pre-KBest) ----
    try:
        with open("models/input_features.txt", "r") as f:
            input_cols = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"⚠️ models/input_features.txt missing or unreadable ({e}); using available columns as-is.")
        # Fall back to whatever you have (not ideal; KBest may still complain)
        input_cols = [c for c in df.columns if c not in ("Event",)]

    # Add any missing columns as zeros (the safest neutral fill for tree models)
    for c in input_cols:
        if c not in df.columns:
            df[c] = 0.0

    # Strict column order + dtype
    X = df[input_cols].astype(float)
    # Replace inf/-inf (can arise in ratios) so the imputer can handle them
    X = X.replace([np.inf, -np.inf], np.nan)


    y_orig = df["Event"]                       # values now in {1, 2}

    # Persist the full pre-KBest input feature set (for predict-time alignment)
    os.makedirs("models", exist_ok=True)
    pd.Series(list(X.columns), dtype=str).to_csv(
        "models/input_features.txt", index=False, header=False
    )
    print("💾 Input feature columns saved to models/input_features.txt")


    # Encode labels to 0..K-1 for XGBoost
    label_values = sorted(y_orig.unique())     # e.g., [1, 2]
    label_map = {lab: i for i, lab in enumerate(label_values)}    # {1:0, 2:1}
    inv_label_map = {v: k for k, v in label_map.items()}          # {0:1, 1:2}
    y = y_orig.map(label_map).astype(int)      # use `y` (0/1) for all training below

    # Class weight hint for XGBoost (helps minority class without SMOTE artifacts)
    pos = int((y == 1).sum()) if 1 in y.unique() else 1
    neg = int((y == 0).sum()) if 0 in y.unique() else 1
    spw = max(1.0, neg / max(1, pos))  # e.g., if crash is rarer, this > 1
    print(f"🔧 scale_pos_weight for XGB: {spw:.2f}")


    print(f"🔤 Label encoding map: {label_map} (train on 0..{y.nunique()-1})")

    # show encoded distribution too
    print("\n📊 Encoded class distribution (0..K-1):")
    print(y.value_counts())

    # -- Persist label maps so predict/backtest can map probs ↔ labels --
    import json
    os.makedirs("models", exist_ok=True)
    with open("models/label_map.json", "w") as f:
        json.dump(
            {
                # original -> encoded (e.g., {1:0, 2:1})
                "label_map": {str(k): int(v) for k, v in label_map.items()},
                # encoded -> original (e.g., {0:1, 1:2})
                "inv_label_map": {str(k): int(v) for k, v in inv_label_map.items()}
            },
            f, indent=2
        )
    print("💾 Saved label maps to models/label_map.json")







    # ---- Model output path (define early, once) ----
    MODEL_DIR = "models"
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.getenv("MODEL_PATH", os.path.join(MODEL_DIR, "market_crash_model.pkl"))


    if X.shape[0] == 0:
        raise RuntimeError("Empty training matrix after cleaning.")
    if X.shape[1] == 0:
        raise RuntimeError("No features selected after filtering.")

    print("\n📊 Original label distribution (1/2):")
    print(y_orig.value_counts())
    print("\n📊 Encoded label distribution (0/1):")
    print(y.value_counts())


    # ---- Adaptive CV & SMOTE safety ----
    tscv_local = _build_adaptive_cv(len(X))
    print(f"CV folds actually used: {tscv_local.get_n_splits(X)}")

    use_smote, smote_step = _safe_smote_from_fold(y, tscv_local)

    # ---- Choose XGBoost objective dynamically  ----
    n_classes = int(y.nunique())
    is_binary = (n_classes == 2)

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




    # ---- Pipeline (skip KBest when too few features) ----
    use_kbest = X.shape[1] >= 2
    if use_kbest:
        max_k = X.shape[1]
        k_choices = sorted(set([1, 2, 3, 5, 8, 10, 12, max(1, max_k // 2)]))
        k_choices = [k for k in k_choices if 1 <= k <= max_k]

        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("smote", smote_step),
            ("kbest", SelectKBest(score_func=f_classif)),
            ("clf", XGBClassifier(**xgb_common, **xgb_obj)),
        ]
        pipe = Pipeline(steps=steps)


        param_grid = {
            "kbest__k": k_choices,
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [3, 5, 7],
            "clf__learning_rate": [0.01, 0.05, 0.1],
            "clf__subsample": [0.8, 1.0],
            "clf__colsample_bytree": [0.8, 1.0],
        }
    else:
        
        steps = [
            ("smote", smote_step),
            ("clf", XGBClassifier(**xgb_common, **xgb_obj)),

        ]
        pipe = Pipeline(steps=steps)


        param_grid = {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [3, 5, 7],
            "clf__learning_rate": [0.01, 0.05, 0.1],
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
        verbose=1,
        error_score=0,  # ignore fold errors instead of crashing
    )
    grid_search.fit(X, y)

    print(f"\n✅ Best Params: {grid_search.best_params_}")
    print(f"🎯 Best Score (F1 Macro): {grid_search.best_score_:.4f}")

    best_model = grid_search.best_estimator_

    

    from sklearn.utils.class_weight import compute_sample_weight

    y_pred0 = best_model.predict(X)                      # preds from CV-best pipe (on encoded y)
    w_miss = 1.0 + 2.0 * (y_pred0 != y).astype(float)    # 3x on mistakes
    w_bal  = compute_sample_weight(class_weight="balanced", y=y)

    w = w_miss * w_bal

    # Fit on a copy with SMOTE disabled so lengths match
    from copy import deepcopy
    best_model_wn = deepcopy(best_model)
    if hasattr(best_model_wn, "steps"):
        steps = dict(best_model_wn.steps)
        if "smote" in steps:
            best_model_wn.set_params(smote="passthrough")

    best_model_wn.fit(X, y, **{"clf__sample_weight": w})

    # For feature export later
    pipe_for_introspection = best_model_wn



    # Calibrate on the *weighted-refit* model
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


    # ---- SHAP export (post-fit, SHAP with fallback to permutation importance) ----
    try:
        import shap  # optional dependency
        import numpy as _np

        # Handle CalibratedClassifierCV(base_estimator=Pipeline(...))
        shap_est = None
        shap_X = None

        base_est = getattr(best_model, "base_estimator", None) or best_model  # CalibratedClassifierCV -> Pipeline or estimator
        if hasattr(base_est, "named_steps"):
            kb  = base_est.named_steps.get("kbest", None)
            clf = base_est.named_steps.get("clf", None)
            if kb is not None and hasattr(kb, "transform"):
                shap_X = kb.transform(X)  # numpy array post-KBest
                # Align feature names to selected set (if mask available)
                try:
                    mask = kb.get_support()
                    feat_names = list(np.array(list(X.columns))[mask]) if hasattr(mask, "__len__") and len(mask) == X.shape[1] else list(X.columns)
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

        # Sample to keep runtime reasonable
        sample_n = min(5000, shap_X.shape[0])
        if sample_n < shap_X.shape[0]:
            rs = np.random.RandomState(42)
            take = rs.choice(shap_X.shape[0], size=sample_n, replace=False)
            shap_X_sample = shap_X[take]
        else:
            shap_X_sample = shap_X

        explainer = shap.TreeExplainer(shap_est)
        shap_vals = explainer.shap_values(shap_X_sample)

        # Binary/multi-class unification: mean |SHAP| per feature
        if isinstance(shap_vals, list):
            shap_abs = _np.abs(_np.array(shap_vals)).max(axis=0)  # max across classes
        else:
            shap_abs = _np.abs(shap_vals)

        mean_abs = shap_abs.mean(axis=0)
        shap_df = pd.DataFrame({"feature": feat_names, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)
        shap_df.to_csv("logs/shap_importance.csv", index=False)
        print("💾 Wrote SHAP importances → logs/shap_importance.csv")

    except ModuleNotFoundError:
        # SHAP not installed — fallback to permutation importance with the same schema
        print("ℹ️ SHAP not installed — falling back to permutation importance.")
        try:
            from sklearn.inspection import permutation_importance
            pi = permutation_importance(best_model, X, y, n_repeats=5, random_state=42, n_jobs=-1)
            mean_abs = np.abs(pi.importances_mean)
            shap_df = pd.DataFrame({"feature": list(X.columns), "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)
            shap_df.to_csv("logs/shap_importance.csv", index=False)
            print("💾 Wrote permutation importance → logs/shap_importance.csv")
        except Exception as e2:
            print(f"⚠️ Permutation importance also failed: {e2}")
    except Exception as e:
        print(f"⚠️ Importance export skipped: {e}")





    

    # ---- ALWAYS define selected_cols safely (after best_model exists) ----
    

    selected_cols = list(X.columns)
    try:
        inspector = pipe_for_introspection if 'pipe_for_introspection' in locals() else None
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


    # Save *only* the selected features list (no index/header to avoid stray '0')
    os.makedirs("models", exist_ok=True)
    pd.Series(selected_cols, dtype=str).to_csv(
        "models/selected_features.txt", index=False, header=False
    )
    print(f"💾 Selected feature columns saved to models/selected_features.txt ({len(selected_cols)} cols)")

    # ---- Training-set metrics (sanity only) ----
    y_pred = best_model.predict(X)  # encoded preds in {0,1}
    print("\n📈 Training Evaluation Metrics (encoded labels 0..K-1):")
    print(f"Accuracy:  {accuracy_score(y, y_pred):.4f}")
    print(f"Precision: {precision_score(y, y_pred, average='weighted'):.4f}")
    print(f"Recall:    {recall_score(y, y_pred, average='weighted'):.4f}")
    print(f"F1 Score:  {f1_score(y, y_pred, average='weighted'):.4f}")

    # Human-readable reports in ORIGINAL label space
    y_pred_orig = pd.Series(y_pred).map(inv_label_map)
    y_true_orig = pd.Series(y).map(inv_label_map)
    cls_order_enc = list(getattr(best_model, "classes_", sorted(pd.Series(y).unique())))  # [0,1]
    cls_order_orig = [inv_label_map[c] for c in cls_order_enc]                             # [1,2]
    target_names = [str(c) for c in cls_order_orig]

    print("\n📊 Predicted class counts (original labels):")
    print(y_pred_orig.value_counts())

    print("\n🧾 Classification report (original labels):")
    print(classification_report(y_true_orig, y_pred_orig, labels=cls_order_orig,
                                target_names=target_names, zero_division=0, digits=4))

    print("\n🧩 Confusion matrix (rows=true, cols=pred) — original labels order:")
    print(confusion_matrix(y_true_orig, y_pred_orig, labels=cls_order_orig))

    # ---- Probability diagnostics: one-vs-rest AP per class (original labels) ----
    try:
        proba = best_model.predict_proba(X)  # shape (n, 2) for binary
        print("\n🎯 Average Precision (PR-AUC) per class (original labels):")
 
        for i, cls_enc in enumerate(cls_order_enc):
            cls_orig = inv_label_map[cls_enc]
            ap = average_precision_score((y_true_orig == cls_orig).astype(int), proba[:, i])
            print(f"AP (class={cls_orig}): {ap:.4f}")

        # -- Select a threshold for ORIGINAL class 1 (Crash) to maximize F1 for Crash only --
        # Determine which proba column corresponds to original label 1
        cls_order_enc = list(getattr(best_model, "classes_", sorted(pd.Series(y).unique())))  # e.g. [0,1]
        pos_orig = 1                                     # ORIGINAL class 1 = Crash in your codebase
        pos_enc = label_map[pos_orig]                    # encoded id (0 or 1) of original class 1
        col_idx = cls_order_enc.index(pos_enc)           # proba column index for class 1 (Crash)
        p_pos = proba[:, col_idx]                        # predicted prob for Crash

        # Optimize F1 for Crash-vs-NotCrash (instead of macro-F1)

        y_pos = (y == pos_enc).astype(int)               # 1 = Crash, 0 = NotCrash
        ts = np.linspace(0.05, 0.95, 19)

        best_t = 0.50
        best_f1 = -1.0
        best_prec = 0.0
        best_rec  = 0.0

        for t_ in ts:
            y_hat = (p_pos >= t_).astype(int)            # 1 = predict Crash
            f1  = f1_score(y_pos, y_hat, zero_division=0)
            if f1 > best_f1:
                best_f1  = f1
                best_t   = t_
                best_prec = precision_score(y_pos, y_hat, zero_division=0)
                best_rec  = recall_score(y_pos, y_hat, zero_division=0)

        thr_payload = {
            "pos_orig": int(pos_orig),
            "pos_enc": int(pos_enc),
            "proba_col_index": int(col_idx),
            "threshold": float(best_t),
            "metric": "f1_crash_only",
            "precision_on_train": float(best_prec),
            "recall_on_train": float(best_rec),
            "f1_on_train": float(best_f1),
        }
        with open("models/thresholds.json", "w") as f:
            json.dump(thr_payload, f, indent=2)
        print(
            f"💾 Saved decision threshold → models/thresholds.json: "
            f"t={best_t:.3f} (Crash: P={best_prec:.3f}, R={best_rec:.3f}, F1={best_f1:.3f})"
        )

    except Exception as e:
        print(f"⚠️ AP (per-class) skipped: {e}")





    # ---- Persist artifacts ----
    #model_path = "models/market_crash_model.pkl"
    #joblib.dump(best_model, model_path)
    #print(f"\n💾 Best model saved to {model_path}")

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
