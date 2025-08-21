# predict.py

import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

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
    Prefer the fitted KBest's input names; then estimator.feature_names_in_;
    lastly models/selected_features.txt (safe when there's no KBest step).
    """
    # 1) If the model is a Pipeline with KBest, use the input columns seen at fit time
    try:
        if hasattr(model, "named_steps") and "kbest" in model.named_steps:
            kb = model.named_steps["kbest"]
            if hasattr(kb, "feature_names_in_"):
                return list(kb.feature_names_in_)
    except Exception:
        pass

    # 2) Otherwise, if the estimator exposes feature_names_in_, use that
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    # 3) Fallback: selected_features.txt (only safe if there was no KBest)
    p = "models/selected_features.txt"
    if os.path.exists(p):
        import pandas as pd  # ensure pandas is imported at top of file
        return pd.read_csv(p, header=None)[0].astype(str).str.strip().tolist()

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

        # Predict
        prediction = int(model.predict(latest_features)[0])
        class_probs = model.predict_proba(latest_features)[0]
        cmap = _class_index_map(model.classes_)

        crash_confidence = float(class_probs[cmap.get(1, 0)]) if len(class_probs) else 0.0
        spike_confidence = float(class_probs[cmap.get(2, 0)]) if len(class_probs) else 0.0

        if max(class_probs) < 0.6:
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
        print(f"📊 Class Probabilities: {class_probs}")
        print(f"Normal: {class_probs[cmap.get(0, 0)]*100:.2f}%"
              if 0 in cmap else "Normal: n/a")
        print(f"Crash: {crash_confidence*100:.2f}%")
        print(f"Spike: {spike_confidence*100:.2f}%")
        print(f"Prediction Forecast: {in_human_speak(prediction, crash_confidence, spike_confidence)}")

        notify_user(prediction, crash_confidence, spike_confidence)

        label = "NORMAL"
        if prediction == 1 and crash_confidence >= 0.7:
            label = "CRASH"
        elif prediction == 2 and spike_confidence >= 0.7:
            label = "SPIKE"

        if label != "NORMAL":
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

    # Predict
    preds = model.predict(X)
    probs = model.predict_proba(X)
    cmap = _class_index_map(model.classes_)

    crash_conf = probs[:, cmap.get(1, 0)] if probs.size else np.zeros(len(X))
    spike_conf = probs[:, cmap.get(2, 0)] if probs.size else np.zeros(len(X))

    # Attach outputs
    feature_df["Prediction"] = preds
    feature_df["Crash_Conf"] = crash_conf
    feature_df["Spike_Conf"] = spike_conf
    feature_df["Confidence"] = probs.max(axis=1) if probs.size else np.zeros(len(X))

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
