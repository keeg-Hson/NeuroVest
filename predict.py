# --- predict.py (clean forward_returns-first) ---
import os, json, joblib, numpy as np, pandas as pd
from datetime import datetime
from dotenv import load_dotenv

from config import PREDICT_CFG
from utils import (
    log_prediction_to_file,
    in_human_speak,
    load_SPY_data as load_data,
    add_features,
    finalize_features,
    expected_value,
)

load_dotenv()

PREDICT_VARIANT = os.getenv("PREDICT_VARIANT", "forward_returns").strip().lower()
if PREDICT_VARIANT not in {"forward_returns","crash_spike"}:
    raise ValueError(f"Unknown PREDICT_VARIANT={PREDICT_VARIANT}")

# Artifacts (env override allowed)
if PREDICT_VARIANT == "forward_returns":
    MODEL_PATH     = os.getenv("MODEL_PATH",     "models/market_crash_model_fwd.pkl")
    THRESH_PATH    = os.getenv("THRESH_PATH",    "models/thresholds_fwd.json")
    LABEL_MAP_PATH = os.getenv("LABEL_MAP_PATH", "models/label_map_fwd.json")
    PROBA_COL      = 1  # proba of class=1 (Trade)
else:
    MODEL_PATH     = os.getenv("MODEL_PATH",     "models/market_crash_model.pkl")
    THRESH_PATH    = os.getenv("THRESH_PATH",    "models/thresholds.json")
    LABEL_MAP_PATH = os.getenv("LABEL_MAP_PATH", "models/label_map.json")
    PROBA_COL      = 1  # adjust if your crash idx differs

FWD_BLACKLIST = {"y","fwd_price","fwd_ret_raw","fwd_ret_net","horizon_forward"}
GLOBAL_MIN_CONF = float(os.getenv("GLOBAL_MIN_CONF", "0.65"))

def _ensure_time_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        if "Date" in out.columns:
            out.index = pd.to_datetime(out["Date"], errors="coerce")
        elif "Timestamp" in out.columns:
            out.index = pd.to_datetime(out["Timestamp"], errors="coerce")
        else:
            out.index = pd.to_datetime(out.index, errors="coerce")
    return out[out.index.notna()]

def _attach_ohlc(pred_df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    out = pred_df.copy()
    raw = _ensure_time_index(raw_df.copy())
    for col in ["Close","Open","High","Low","Volume"]:
        if col not in out.columns and col in raw.columns:
            out[col] = raw[col].reindex(out.index)
    if "Date" not in out.columns:
        out["Date"] = out.index.tz_localize(None) if isinstance(out.index, pd.DatetimeIndex) else pd.NaT
    return out

def _required_feature_names_for_pipeline(model) -> list[str]:
    # Prefer recorded schema file if present
    try:
        path = "models/input_features_fwd.txt" if PREDICT_VARIANT=="forward_returns" else "models/input_features.txt"
        if os.path.exists(path):
            vals = pd.read_csv(path, header=None)[0].astype(str).str.strip().tolist()
            vals = [v for v in vals if v and v not in FWD_BLACKLIST and not v.startswith("fwd_")]
            # de-dup while preserving order
            seen=set(); clean=[]
            for v in vals:
                if v not in seen:
                    seen.add(v); clean.append(v)
            return clean
    except Exception:
        pass
    # Fall back to model introspection
    base = getattr(model, "base_estimator", model)
    if hasattr(base, "named_steps") and "kbest" in base.named_steps:
        kb = base.named_steps["kbest"]
        if hasattr(kb, "feature_names_in_"):
            return [c for c in kb.feature_names_in_ if c not in FWD_BLACKLIST and not str(c).startswith("fwd_")]
    if hasattr(base, "feature_names_in_"):
        return [c for c in base.feature_names_in_ if c not in FWD_BLACKLIST and not str(c).startswith("fwd_")]
    return []

def _prepare_matrix(feature_df: pd.DataFrame, req_cols: list[str]) -> pd.DataFrame:
    out = feature_df.copy()
    for c in req_cols:
        if c not in out.columns:
            out[c] = np.nan
    out = out[req_cols]
    out = out.replace([np.inf,-np.inf], np.nan)
    # time-aware fill if index is time
    try:
        out = out.interpolate(method="time", limit_direction="both")
    except Exception:
        pass
    out = out.fillna(out.median(numeric_only=True))
    return out[req_cols]

def live_predict(feature_df: pd.DataFrame, raw_df: pd.DataFrame):
    feature_df = _ensure_time_index(feature_df)
    raw_df     = _ensure_time_index(raw_df)

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
        return None

    model = joblib.load(MODEL_PATH)

    req = _required_feature_names_for_pipeline(model)
    print(f"🧱 Required input schema (pre-KBest): {req}")
    if not req:
        print("⚠️ No usable feature columns found for the model.")
        return None

    X_all  = _prepare_matrix(feature_df, req)
    X_last = X_all.iloc[[-1]]
    if X_last.empty:
        print("⚠️ No features for latest row.")
        return None

    # prices for log
    raw_last   = raw_df.iloc[-1] if len(raw_df) else pd.Series(dtype=float)
    close_px   = float(raw_last.get("Close", np.nan))
    open_px    = float(raw_last.get("Open",  np.nan))
    high_px    = float(raw_last.get("High",  np.nan))
    low_px     = float(raw_last.get("Low",   np.nan))

    # label map (optional)
    inv_label_map = {}
    try:
        with open(LABEL_MAP_PATH, "r") as fh:
            maps = json.load(fh)
        inv_label_map = {int(k): int(v) for k,v in maps.get("inv_label_map",{}).items()}
    except Exception:
        pass

    # thresholds
    t = 0.5; pos_enc = 1
    try:
        with open(THRESH_PATH, "r") as fh:
            thr = json.load(fh)
        t       = float(thr.get("threshold", 0.5))
        pos_enc = int(thr.get("pos_enc", 1))
    except Exception as e:
        print(f"⚠️ Thresholds load issue ({THRESH_PATH}): {e}")

    # predict proba on last row
    class_probs = model.predict_proba(X_last)[0]  # shape (n_classes,)
    classes_enc = list(getattr(model, "classes_", [0,1]))

    if PREDICT_VARIANT == "crash_spike":
        # map 2-class to {1=Crash, 2=Spike}
        proba_by_orig = {}
        for j, enc_lab in enumerate(classes_enc):
            orig = inv_label_map.get(enc_lab, enc_lab)
            proba_by_orig[int(orig)] = float(class_probs[j])
        crash_conf = proba_by_orig.get(1, float(class_probs[0]))
        spike_conf = proba_by_orig.get(2, float(class_probs[1] if len(class_probs)>1 else 0.0))
        winner_is_crash = crash_conf >= spike_conf
        winner_prob     = crash_conf if winner_is_crash else spike_conf
        prediction      = 1 if winner_is_crash else 2
    else:
        # forward_returns: {0=No-Trade, 1=Trade}
        col_idx   = PROBA_COL if PROBA_COL < len(class_probs) else -1
        p_trade   = float(class_probs[col_idx])
        t_eff     = max(t, GLOBAL_MIN_CONF)
        prediction= 1 if p_trade >= t_eff else 0
        winner_prob = p_trade
        crash_conf  = 0.0
        spike_conf  = p_trade

    # EV (use forward-returns params, harmless for crash/spike)
    avg_gain = PREDICT_CFG.get("avg_gain", 0.0040)
    avg_loss = PREDICT_CFG.get("avg_loss", 0.0030)
    fee_bps  = PREDICT_CFG.get("fee_bps", 1.5)
    slip_bps = PREDICT_CFG.get("slippage_bps", 2.0)
    EV = expected_value(winner_prob, avg_gain, avg_loss, fee_bps, slip_bps)

    # Pretty print + log
    if PREDICT_VARIANT == "forward_returns":
        human = "TRADE" if prediction==1 else "NO-TRADE"
        if winner_prob < GLOBAL_MIN_CONF:
            print("⚠️ Low-confidence prediction — consider ignoring this signal.")
        print(f"🔮 Prediction (forward-returns): {prediction}  ({human})")
        print(f"📊 Probabilities: Trade={winner_prob:.4f}, No-Trade={1.0-winner_prob:.4f}")
    else:
        print(f"🔮 Prediction (crash/spike): {prediction} ({in_human_speak(prediction)})")
        print(f"📊 Crash={crash_conf:.4f} | Spike={spike_conf:.4f}")

    # compose a 1-row frame for saving/returning
    last_idx = feature_df.index[-1]
    out = pd.DataFrame(index=[last_idx])
    out["Prediction"] = prediction
    out["Crash_Conf"] = crash_conf
    out["Spike_Conf"] = spike_conf
    out["Confidence"] = max(crash_conf, spike_conf, winner_prob)
    out = _attach_ohlc(out, raw_df)

    # timestamp for log writer signature
    ts = last_idx if isinstance(last_idx, pd.Timestamp) else pd.Timestamp.utcnow()
    log_prediction_to_file(
        timestamp=ts,
        prediction=prediction,
        crash_conf=crash_conf,
        spike_conf=spike_conf,
        close_price=close_px,
        open_price=open_px,
        high=high_px,
        low=low_px,
    )
    return out

if __name__ == "__main__":
    print("📥 Loading SPY data...")
    raw_df = load_data()

    print("🧮 Building features...")
    feat_df, feat_cols = add_features(raw_df)

    print("🧹 Finalizing features...")
    feat_df = finalize_features(feat_df, feat_cols)
    feat_df = _ensure_time_index(feat_df)
    raw_df  = _ensure_time_index(raw_df)

    print("🔮 Running prediction on latest row...")
    live_predict(feat_df, raw_df)
