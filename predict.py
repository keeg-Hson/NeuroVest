# predict.py

import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import json


from utils import (
    log_prediction_to_file,
    in_human_speak,
    load_SPY_data as load_data,
    add_features,
    send_telegram_alert,
    notify_user,
    finalize_features,
)

# ──────────────────────────────────────────────────────────────────────────────
# Env
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

MODEL_PATH = "models/market_crash_model.pkl"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _ensure_time_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure df has a DatetimeIndex for time-based interpolation."""
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        if "Date" in out.columns:
            out = out.set_index(pd.to_datetime(out["Date"], errors="coerce"))
        elif "Timestamp" in out.columns:
            out = out.set_index(pd.to_datetime(out["Timestamp"], errors="coerce"))
        else:
            out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[out.index.notna()]
    return out


def _class_index_map(classes_) -> dict:
    """Return a dict mapping class label -> column index in predict_proba output."""
    return {int(label): idx for idx, label in enumerate(classes_)}


def _read_feature_list(path: str) -> list[str]:
    if not os.path.exists(path):
        return []
    vals = pd.read_csv(path, header=None)[0].astype(str).str.strip().tolist()
    # drop obvious junk
    bad = {"", "0", "feature", "Unnamed: 0"}
    out, seen = [], set()
    for v in vals:
        if v in bad or v in seen:
            continue
        out.append(v); seen.add(v)
    return out

def _required_feature_names_for_pipeline(model) -> list[str]:
    """
    Return the exact pre-KBest input schema used at train time.
    Order of preference:
      1) Unwrap CalibratedClassifierCV → Pipeline → KBest.feature_names_in_
      2) models/input_features.txt (persisted at train time)
      3) Estimator.feature_names_in_ (if present and no KBest)
    We DO NOT use selected_features.txt here (that is post-KBest and will break KBest.transform).
    """
    # Unwrap calibrator if present
    base = getattr(model, "base_estimator", model)

    # 1) Pipeline with KBest → use the input names seen during fit
    try:
        if hasattr(base, "named_steps") and "kbest" in base.named_steps:
            kb = base.named_steps["kbest"]
            if hasattr(kb, "feature_names_in_"):
                return list(kb.feature_names_in_)
    except Exception:
        pass

    # 2) models/input_features.txt saved by train.py
    try:
        p = "models/input_features.txt"
        if os.path.exists(p):
            return pd.read_csv(p, header=None)[0].astype(str).str.strip().tolist()
    except Exception:
        pass

    # 3) If no KBest and estimator exposes names
    if hasattr(base, "feature_names_in_"):
        return list(base.feature_names_in_)

    return []





def _prepare_matrix_for_model(feature_df: pd.DataFrame, req_cols: list[str]) -> pd.DataFrame:
    """
    Ensure all required columns exist (creating missing ones as NaN) and
    return a view in the exact required order.
    """
    out = feature_df.copy()
    missing = [c for c in req_cols if c not in out.columns]
    for c in missing:
        out[c] = np.nan
    # time-interpolate + ffill/bfill on required set only
    out[req_cols] = (
        out[req_cols]
        .interpolate(method="time", limit_direction="both")
        .ffill()
        .bfill()
    )
    return out[req_cols]


# ──────────────────────────────────────────────────────────────────────────────
# LIVE PREDICT (single latest bar)
# ──────────────────────────────────────────────────────────────────────────────
def live_predict(feature_df: pd.DataFrame, raw_df: pd.DataFrame, model_path: str = MODEL_PATH):
    if not os.path.exists(model_path):
        print("❌ Model file not found.")
        return None, None, None

    model = joblib.load(model_path)

    try:
        # Ensure time index for interpolation safety if needed
        feature_df = _ensure_time_index(feature_df)
        raw_df = _ensure_time_index(raw_df)

        # Determine columns the pipeline expects
        req = _required_feature_names_for_pipeline(model)
        print(f"🧱 Required input schema (pre-KBest): {req}")

        if not req:
            print("⚠️ No usable feature columns found for the model.")
            return None, None, None

        # Build latest feature row (after ensuring columns)
        X_all = _prepare_matrix_for_model(feature_df, req)
        latest_features = X_all.iloc[[-1]]

        if latest_features.shape[0] == 0 or latest_features.shape[1] == 0:
            print("⚠️ No features available for latest row — skipping live prediction.")
            return None, None, None

        # Raw price refs for logging
        raw_latest = raw_df.iloc[-1]
        close_price = float(raw_latest.get("Close", float("nan")))
        open_price = float(raw_latest.get("Open", float("nan")))
        high = float(raw_latest.get("High", float("nan")))
        low = float(raw_latest.get("Low", float("nan")))

        
        # Predict with learned threshold
        class_probs = model.predict_proba(latest_features)[0]
        classes_enc = list(getattr(model, "classes_", [0, 1]))

        # Load maps/threshold
        maps = json.load(open("models/label_map.json"))
        inv_label_map = {int(k): int(v) for k, v in maps["inv_label_map"].items()}
        thr = json.load(open("models/thresholds.json"))
        col_idx = int(thr["proba_col_index"])
        t = float(thr["threshold"])
        pos_enc = int(thr["pos_enc"])

        # Decide encoded label by threshold on ORIGINAL class 1 (Crash)
        y_hat_enc = pos_enc if class_probs[col_idx] >= t else (1 - pos_enc)
        prediction = int(inv_label_map.get(y_hat_enc, y_hat_enc))  # ORIGINAL {1,2}

        # Map probs to ORIGINAL label space for reporting
        proba_by_orig = {}
        for j, enc_lab in enumerate(classes_enc):
            orig = inv_label_map.get(enc_lab, enc_lab)  # 0/1 -> 1/2
            proba_by_orig[int(orig)] = float(class_probs[j])

        # IMPORTANT: 1=Crash, 2=Spike in your code
        crash_confidence = proba_by_orig.get(1, float(class_probs[0]))
        spike_confidence = proba_by_orig.get(2, float(class_probs[1] if len(class_probs) > 1 else 0.0))


        # --- Decide winner consistently (argmax), then gate with thresholds ---
        # Probabilities already mapped to ORIGINAL labels:
        #   crash_confidence = P(orig=1), spike_confidence = P(orig=2)
        winner_is_crash = (crash_confidence >= spike_confidence)
        prediction = 1 if winner_is_crash else 2
        winner_prob = crash_confidence if winner_is_crash else spike_confidence

        # Optional quality gates
        GLOBAL_MIN_CONF = 0.65

        # Trend agreement (weekly MA filter)
        close = raw_df["Close"].astype(float)
        weekly = close.resample("W-FRI").last()
        weekly_ma = weekly.rolling(26).mean().reindex(close.index, method="ffill")
        in_uptrend = (close.iloc[-1] >= weekly_ma.iloc[-1]) if pd.notna(weekly_ma.iloc[-1]) else True
        trend_agrees = (not winner_is_crash and in_uptrend) or (winner_is_crash and not in_uptrend)

        # Extra sweep-based per-class thresholds (optional)
        from pathlib import Path as _Path
        best_cfg = {}
        cfg_path = _Path("configs/best_thresholds.json")
        if cfg_path.exists():
            try:
                import json as _json
                best_cfg = _json.load(open(cfg_path))
            except Exception:
                best_cfg = {}
        best_conf  = best_cfg.get("confidence_thresh", None)
        best_crash = best_cfg.get("crash_thresh", None)
        best_spike = best_cfg.get("spike_thresh", None)

        # Learned crash threshold t (from models/thresholds.json). If class-specific
        # thresholds exist, use those; else use t for both classes.
        class_t = float(
            best_crash if (winner_is_crash and best_crash is not None)
            else best_spike if ((not winner_is_crash) and best_spike is not None)
            else t
        )

        # Final gate: must clear (winner-specific threshold) AND global min AND optional best_conf
        min_required = max(class_t, (best_conf or 0.0), GLOBAL_MIN_CONF)
        passed = (winner_prob >= min_required) and trend_agrees

        if max(crash_confidence, spike_confidence) < 0.60:
            print("⚠️ Low-confidence prediction — consider ignoring this signal.")


        # Log + output
        timestamp = datetime.now()
        log_prediction_to_file(
            timestamp,
            prediction,
            crash_confidence,
            spike_confidence,
            close_price,
            open_price,
            high,
            low,
        )

        print(f"🔮 Prediction: {prediction}")
        print(f"📊 Class Probabilities (orig labels): Crash={crash_confidence:.4f}, Spike={spike_confidence:.4f}")
        print(f"Crash: {crash_confidence*100:.2f}%")
        print(f"Spike: {spike_confidence*100:.2f}%")

        try:
            forecast_text = in_human_speak(prediction)
        except TypeError:
            # Fallback if someone changes the signature again
            forecast_text = f"{'CRASH' if prediction == 1 else 'SPIKE'}"
        print(f"Prediction Forecast: {forecast_text}")



        notify_user(prediction, crash_confidence, spike_confidence)

        # Use the decision computed above
        label = "CRASH" if winner_is_crash else "SPIKE"




        if passed:
            msg = (
                f"🚨 *Market Alert* — {label} signal detected!\n\n"
                f"📉 Crash: `{crash_confidence:.2f}`\n"
                f"📈 Spike: `{spike_confidence:.2f}`\n"
                f"🔎 *Prediction*: `{label}`"
            )
            send_telegram_alert(msg, token=TELEGRAM_TOKEN, chat_id=TELEGRAM_CHAT_ID)

        return prediction, crash_confidence, spike_confidence


    except Exception as e:
        print(f"⚠️ Prediction error: {e}")
        return None, None, None


# ──────────────────────────────────────────────────────────────────────────────
# BATCH PREDICTION (full dataframe)
# ──────────────────────────────────────────────────────────────────────────────
def run_predictions(confidence_threshold: float = 0.80) -> pd.DataFrame | None:
    # Load and build features
    raw_df = load_data()
    feature_df, feature_cols = add_features(raw_df)

    # Clean & ensure DatetimeIndex for time interpolation
    feature_df = finalize_features(feature_df, feature_cols)
    feature_df = _ensure_time_index(feature_df)

    # Model + required columns
    if not os.path.exists(MODEL_PATH):
        print("❌ Model file not found.")
        return None
    model = joblib.load(MODEL_PATH)

    req = _required_feature_names_for_pipeline(model)
    if not req:
        print("⚠️ No usable feature columns for prediction — check training/pipeline alignment.")
        return feature_df

    # Prepare matrix (create missing cols, interpolate, order)
    X = _prepare_matrix_for_model(feature_df, req)

    # Guards
    if X.shape[0] == 0:
        print("⚠️ No rows to predict after cleaning — skipping.")
        return feature_df
    if X.shape[1] == 0:
        print("⚠️ No usable feature columns for prediction — check training/pipeline alignment.")
        return feature_df


    # Predict (use learned threshold from training)

    probs = model.predict_proba(X)
    classes_enc = list(getattr(model, "classes_", [0, 1]))

    # Load label maps + threshold
    with open("models/label_map.json", "r") as f:
        maps = json.load(f)
    inv_label_map = {int(k): int(v) for k, v in maps["inv_label_map"].items()}

    with open("models/thresholds.json", "r") as f:
        thr = json.load(f)

    col_idx = int(thr["proba_col_index"])  # which proba column corresponds to ORIGINAL class 1 (Crash)
    t = float(thr["threshold"])
    pos_enc = int(thr["pos_enc"])          # encoded id for original class 1

    # Thresholded predictions in ENCODED label space → ORIGINAL labels
    p_pos = probs[:, col_idx]
    y_hat_enc = np.where(p_pos >= t, pos_enc, 1 - pos_enc)
    y_hat_orig = np.vectorize(inv_label_map.get)(y_hat_enc).astype(int)  # {1,2}

    # Map probability columns to ORIGINAL labels
    proba_by_orig = {}
    for j, enc_lab in enumerate(classes_enc):
        orig = inv_label_map.get(enc_lab, enc_lab)  # 0/1 -> 1/2
        proba_by_orig[int(orig)] = probs[:, j]

    # Attach outputs (IMPORTANT: 1=Crash, 2=Spike in your code)
    feature_df["Prediction"] = y_hat_orig
    feature_df["Crash_Conf"] = proba_by_orig.get(1, probs[:, 0])  # orig=1
    feature_df["Spike_Conf"] = proba_by_orig.get(2, probs[:, 1])  # orig=2
    feature_df["Confidence"] = probs.max(axis=1) if probs.size else np.zeros(len(X))

    print(f"🔧 Using learned threshold t={t:.3f} on class=1 (Crash) [proba col={col_idx}]")
    print("Pred counts with learned threshold:\n", feature_df["Prediction"].value_counts())


    # Timestamp column (tz-naive)
    ts = feature_df.index
    try:
        feature_df["Timestamp"] = ts.tz_localize(None)
    except AttributeError:
        feature_df["Timestamp"] = pd.to_datetime(ts, errors="coerce")
        feature_df["Timestamp"] = feature_df["Timestamp"].dt.tz_localize(None)

    # Save atomically and return
    os.makedirs("logs", exist_ok=True)
    _tmp = "logs/.predictions_full.tmp.csv"
    feature_df.to_csv(_tmp, index=False)
    os.replace(_tmp, "logs/predictions_full.csv")
    return feature_df


# ──────────────────────────────────────────────────────────────────────────────
# Main (manual run)
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("📥 Loading SPY data...")
    raw_df = load_data()

    print("🧮 Building features...")
    feature_df, feature_cols = add_features(raw_df)

    print("🧹 Finalizing features...")
    feature_df = finalize_features(feature_df, feature_cols)
    feature_df = _ensure_time_index(feature_df)

    print("🔮 Running prediction on latest row...")
    live_predict(feature_df, raw_df)
