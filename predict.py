# predict.py

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

from config import PREDICT_CFG
from utils import expected_value  # already implemented in utils.py
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

# ──────────────────────────────────────────────────────────────────────────────
# Variant toggle (Crash/Spike vs Forward-Returns)
# ──────────────────────────────────────────────────────────────────────────────
PREDICT_VARIANT = os.getenv("PREDICT_VARIANT", "crash_spike").strip().lower()
# valid: "crash_spike" | "forward_returns"

ARTIFACTS = {
    "crash_spike": {
        "model":      "models/market_crash_model.pkl",
        "label_map":  "models/label_map.json",
        "thresholds": "models/thresholds.json",
    },
    "forward_returns": {
        "model":      "models/market_crash_model_fwd.pkl",
        "label_map":  "models/label_map_fwd.json",
        "thresholds": "models/thresholds_fwd.json",
    },
}
if PREDICT_VARIANT not in ARTIFACTS:
    raise ValueError(f"Unknown PREDICT_VARIANT={PREDICT_VARIANT}")

MODEL_PATH      = ARTIFACTS[PREDICT_VARIANT]["model"]
LABEL_MAP_PATH  = ARTIFACTS[PREDICT_VARIANT]["label_map"]
THRESH_PATH     = ARTIFACTS[PREDICT_VARIANT]["thresholds"]

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
FWD_BLACKLIST = {"y", "fwd_price", "fwd_ret_raw", "fwd_ret_net", "horizon_forward"}

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

def _attach_ohlc(pred_df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach Close (and Open/High/Low if present) by aligning to index.
    Keeps prediction frame's index and adds missing price cols.
    """
    out = pred_df.copy()
    raw = raw_df.copy()
    raw = _ensure_time_index(raw)

    # Build aligned series
    for col in ["Close", "Open", "High", "Low", "Volume"]:
        if col not in out.columns and col in raw.columns:
            ser = raw[col].astype(float)
            out[col] = ser.reindex(out.index)

    # Provide Date column for downstream scripts that expect it
    if "Date" not in out.columns:
        out["Date"] = out.index.tz_localize(None) if isinstance(out.index, pd.DatetimeIndex) else pd.NaT

    return out

def _class_index_map(classes_) -> dict:
    """Return a dict mapping class label -> column index in predict_proba output."""
    return {int(label): idx for idx, label in enumerate(classes_)}

def _read_feature_list(path: str) -> list[str]:
    if not os.path.exists(path):
        return []
    vals = pd.read_csv(path, header=None)[0].astype(str).str.strip().tolist()
    bad = {"", "0", "feature", "Unnamed: 0"}
    out, seen = [], set()
    for v in vals:
        if v in bad or v in seen:
            continue
        out.append(v); seen.add(v)
    return out

def _required_feature_names_for_pipeline(model) -> list[str]:
    base = getattr(model, "base_estimator", model)

    def _clean(names):
        return [c for c in names if c not in FWD_BLACKLIST]

    try:
        if hasattr(base, "named_steps") and "kbest" in base.named_steps:
            kb = base.named_steps["kbest"]
            if hasattr(kb, "feature_names_in_"):
                return _clean(list(kb.feature_names_in_))
    except Exception:
        pass

    try:
        p = "models/input_features_fwd.txt" if PREDICT_VARIANT == "forward_returns" else "models/input_features.txt"
        if os.path.exists(p):
            vals = pd.read_csv(p, header=None)[0].astype(str).str.strip().tolist()
            return _clean(vals)
    except Exception:
        pass

    if hasattr(base, "feature_names_in_"):
        return _clean(list(base.feature_names_in_))

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
    out[req_cols] = (
        out[req_cols]
        .interpolate(method="time", limit_direction="both")
        .ffill().bfill()
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
        feature_df = _ensure_time_index(feature_df)
        raw_df = _ensure_time_index(raw_df)

        req = _required_feature_names_for_pipeline(model)
        print(f"🧱 Required input schema (pre-KBest): {req}")
        if not req:
            print("⚠️ No usable feature columns found for the model.")
            return None, None, None

        X_all = _prepare_matrix_for_model(feature_df, req)
        latest_features = X_all.iloc[[-1]]
        if latest_features.shape[0] == 0 or latest_features.shape[1] == 0:
            print("⚠️ No features available for latest row — skipping live prediction.")
            return None, None, None

        # Prices for logging
        raw_latest = raw_df.iloc[-1]
        close_price = float(raw_latest.get("Close", float("nan")))
        open_price  = float(raw_latest.get("Open",  float("nan")))
        high        = float(raw_latest.get("High",  float("nan")))
        low         = float(raw_latest.get("Low",   float("nan")))

        # Predict
        class_probs = model.predict_proba(latest_features)[0]
        classes_enc = list(getattr(model, "classes_", [0, 1]))

        maps = json.load(open(LABEL_MAP_PATH))
        inv_label_map = {int(k): int(v) for k, v in maps["inv_label_map"].items()}
        thr  = json.load(open(THRESH_PATH))
        col_idx = int(thr.get("proba_col_index", 1))
        t       = float(thr.get("threshold", 0.5))
        pos_enc = int(thr.get("pos_enc", 1))

        # Variant-aware decoding
        if PREDICT_VARIANT == "crash_spike":
            # Original labels: {1=Crash, 2=Spike}
            proba_by_orig = {}
            for j, enc_lab in enumerate(classes_enc):
                orig = inv_label_map.get(enc_lab, enc_lab)  # 0/1 -> 1/2
                proba_by_orig[int(orig)] = float(class_probs[j])
            crash_confidence = proba_by_orig.get(1, float(class_probs[0]))
            spike_confidence = proba_by_orig.get(2, float(class_probs[1] if len(class_probs) > 1 else 0.0))
            winner_is_crash  = (crash_confidence >= spike_confidence)
            winner_prob      = crash_confidence if winner_is_crash else spike_confidence
            prediction       = 1 if winner_is_crash else 2
        else:
            # Forward-returns: {0=No-Trade, 1=Trade}
            p_trade = float(class_probs[col_idx]) if col_idx < len(class_probs) else float(class_probs[-1])
            crash_confidence = 0.0
            spike_confidence = p_trade  # reuse for uniform logging
            winner_prob = p_trade
            prediction  = 1 if p_trade >= t else 0
            winner_is_crash = False

        # EV gate
        avg_gain = PREDICT_CFG.get("avg_gain", 0.0040)
        avg_loss = PREDICT_CFG.get("avg_loss", 0.0030)
        fee_bps  = PREDICT_CFG.get("fee_bps", 1.5)
        slip_bps = PREDICT_CFG.get("slippage_bps", 2.0)
        ev = expected_value(
            prob_long=winner_prob,
            avg_gain=avg_gain,
            avg_loss=avg_loss,
            fee_bps=fee_bps,
            slippage_bps=slip_bps,
        )
        EV_MIN = PREDICT_CFG.get("ev_min", 0.0005)

        GLOBAL_MIN_CONF = 0.65
        close = raw_df["Close"].astype(float)
        weekly = close.resample("W-FRI").last()
        weekly_ma = weekly.rolling(26).mean().reindex(close.index, method="ffill")
        in_uptrend = (close.iloc[-1] >= weekly_ma.iloc[-1]) if pd.notna(weekly_ma.iloc[-1]) else True
        trend_agrees = (not winner_is_crash and in_uptrend) or (winner_is_crash and not in_uptrend)

        # Optional override thresholds file
        best_t = t
        try:
            from pathlib import Path
            cfg_path = Path("configs/best_thresholds.json")
            if cfg_path.exists():
                best_cfg = json.load(open(cfg_path))
                best_conf  = best_cfg.get("confidence_thresh", None)
                best_crash = best_cfg.get("crash_thresh", None)
                best_spike = best_cfg.get("spike_thresh", None)
                class_t = best_crash if (winner_is_crash and best_crash is not None) \
                          else best_spike if ((not winner_is_crash) and best_spike is not None) \
                          else t
                best_t = max(class_t, (best_conf or 0.0), GLOBAL_MIN_CONF)
            else:
                best_t = max(t, GLOBAL_MIN_CONF)
        except Exception:
            best_t = max(t, GLOBAL_MIN_CONF)

        passed = (winner_prob >= best_t) and trend_agrees and (ev >= EV_MIN)

        # Logging
        if PREDICT_VARIANT == "forward_returns":
            trade_conf = winner_prob
            if trade_conf < 0.60:
                print("⚠️ Low-confidence prediction — consider ignoring this signal.")
            print(f"🔮 Prediction (forward-returns): {prediction}  ({'TRADE' if prediction==1 else 'NO-TRADE'})")
            print(f"📊 Probabilities: Trade={trade_conf:.4f}, No-Trade={1.0-trade_conf:.4f}")
        else:
            if max(crash_confidence, spike_confidence) < 0.60:
                print("⚠️ Low-confidence prediction — consider ignoring this signal.")
            print(f"🔮 Prediction: {prediction}")
            print(f"📊 Class Probabilities (orig labels): Crash={crash_confidence:.4f}, Spike={spike_confidence:.4f}")
            print(f"Crash: {crash_confidence*100:.2f}%")
            print(f"Spike: {spike_confidence*100:.2f}%")

        timestamp = datetime.now()
        log_prediction_to_file(
            timestamp, prediction, crash_confidence, spike_confidence,
            close_price, open_price, high, low,
        )
        notify_user(prediction, crash_confidence, spike_confidence)

        label = ("TRADE" if prediction == 1 else "NO-TRADE") if PREDICT_VARIANT == "forward_returns" \
                else ("CRASH" if winner_is_crash else "SPIKE")

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
    # Load + build features
    raw_df = load_data()
    feature_df, feature_cols = add_features(raw_df)

    # Clean & ensure DatetimeIndex
    feature_df = finalize_features(feature_df, feature_cols)
    feature_df = _ensure_time_index(feature_df)
    raw_df = _ensure_time_index(raw_df)

    # Load model
    if not os.path.exists(MODEL_PATH):
        print("❌ Model file not found.")
        return None
    model = joblib.load(MODEL_PATH)

    # Schema
    req = _required_feature_names_for_pipeline(model)
    if not req:
        print("⚠️ No usable feature columns for prediction — check training/pipeline alignment.")
        return feature_df

    X = _prepare_matrix_for_model(feature_df, req)
    if X.shape[0] == 0:
        print("⚠️ No rows to predict after cleaning — skipping.")
        return feature_df
    if X.shape[1] == 0:
        print("⚠️ No usable feature columns for prediction — check training/pipeline alignment.")
        return feature_df

    # Predict
    probs = model.predict_proba(X)
    classes_enc = list(getattr(model, "classes_", [0, 1]))

    # Thresholds / label maps
    with open(LABEL_MAP_PATH, "r") as f:
        maps = json.load(f)
    inv_label_map = {int(k): int(v) for k, v in maps["inv_label_map"].items()}

    with open(THRESH_PATH, "r") as f:
        thr = json.load(f)

    # Defaults with safe fallbacks
    col_idx = int(thr.get("proba_col_index", 1))
    t       = float(thr.get("threshold", 0.5))
    pos_enc = int(thr.get("pos_enc", 1))

    if PREDICT_VARIANT == "crash_spike":
        # ORIGINAL labels {1=Crash, 2=Spike}
        p_pos = probs[:, col_idx]
        y_hat_enc  = np.where(p_pos >= t, pos_enc, 1 - pos_enc)
        y_hat_orig = np.vectorize(inv_label_map.get)(y_hat_enc).astype(int)  # {1,2}

        proba_by_orig = {}
        for j, enc_lab in enumerate(classes_enc):
            orig = inv_label_map.get(enc_lab, enc_lab)  # 0/1 -> 1/2
            proba_by_orig[int(orig)] = probs[:, j]

        feature_df["Prediction"]  = y_hat_orig
        feature_df["Crash_Conf"]  = proba_by_orig.get(1, probs[:, 0])
        feature_df["Spike_Conf"]  = proba_by_orig.get(2, probs[:, 1] if probs.shape[1] > 1 else np.zeros(len(X)))
        feature_df["Confidence"]  = probs.max(axis=1) if probs.size else np.zeros(len(X))
    else:
        # Forward-returns: {0=No-Trade, 1=Trade}
        p_trade = probs[:, col_idx] if probs.shape[1] > col_idx else probs[:, -1]
        y_hat_bin = (p_trade >= t).astype(int)
        feature_df["Prediction"]  = y_hat_bin
        feature_df["Trade_Conf"]  = p_trade
        feature_df["Confidence"]  = probs.max(axis=1) if probs.size else np.zeros(len(X))
        feature_df["Crash_Conf"]  = 0.0
        feature_df["Spike_Conf"]  = p_trade  # reuse for charts

    # Compute EV per row (winner-side prob)
    avg_gain = PREDICT_CFG.get("avg_gain", 0.0040)
    avg_loss = PREDICT_CFG.get("avg_loss", 0.0030)
    fee_bps  = PREDICT_CFG.get("fee_bps", 1.5)
    slip_bps = PREDICT_CFG.get("slippage_bps", 2.0)

    winner_prob = np.maximum(feature_df["Crash_Conf"].values, feature_df["Spike_Conf"].values)
    feature_df["EV"] = expected_value(winner_prob, avg_gain, avg_loss, fee_bps, slip_bps)

    print(f"🔧 Using learned threshold t={t:.3f} on class=1 (Crash) [proba col={col_idx}]")
    print("Pred counts with learned threshold:\n", feature_df["Prediction"].value_counts())

    # Timestamp & Date col
    ts = feature_df.index
    try:
        feature_df["Timestamp"] = ts.tz_localize(None)
    except AttributeError:
        feature_df["Timestamp"] = pd.to_datetime(ts, errors="coerce").dt.tz_localize(None)
    if "Date" not in feature_df.columns:
        feature_df["Date"] = feature_df["Timestamp"]

    # Attach OHLC so terminal checks (ret1 from Close) never KeyError
    feature_df = _attach_ohlc(feature_df, raw_df)

    # Save atomically & return
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
    raw_df = _ensure_time_index(raw_df)

    print("🔮 Running prediction on latest row...")
    live_predict(feature_df, raw_df)
