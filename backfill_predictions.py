# backfill_predictions.py
import os, json, joblib, pandas as pd, numpy as np
from tqdm import tqdm
from utils import load_SPY_data, add_features, finalize_features, log_prediction_to_file

# Make sure we’re in forward-returns mode
os.environ["PREDICT_VARIANT"] = os.getenv("PREDICT_VARIANT","forward_returns")

MODEL_PATH  = os.getenv("MODEL_PATH",  "models/market_crash_model_fwd.pkl")
THRESH_PATH = os.getenv("THRESH_PATH", "models/thresholds_fwd.json")

def _required_feature_names():
    try:
        return pd.read_csv("models/input_features_fwd.txt", header=None)[0].astype(str).tolist()
    except Exception:
        return []

def _prep(df, req):
    X = df.copy()
    for c in req:
        if c not in X.columns:
            X[c] = np.nan
    X = X[req].replace([np.inf,-np.inf], np.nan)
    try:
        X = X.interpolate(method="time", limit_direction="both")
    except Exception:
        pass
    X = X.fillna(X.median(numeric_only=True))
    return X[req]

spy = load_SPY_data()
feat, feat_cols = add_features(spy)
feat = finalize_features(feat, feat_cols)
feat = feat.sort_index()
req  = _required_feature_names()

model = joblib.load(MODEL_PATH)
with open(THRESH_PATH, "r") as fh:
    thr = json.load(fh)
t = float(thr.get("threshold", 0.5))
# keep GLOBAL_MIN_CONF behavior consistent with predict.py
t = max(t, float(os.getenv("GLOBAL_MIN_CONF","0.65")))

rows = []
for dt in tqdm(feat.index, desc="backfill"):
    X = _prep(feat.loc[:dt], req)
    if X.empty: 
        continue
    proba = model.predict_proba(X.iloc[[-1]])[0]
    # forward-returns: class 1 = Trade
    p_trade = float(proba[1]) if len(proba)>1 else float(proba[0])
    pred = int(p_trade >= t)

    # Append to in-memory list; also log like live path so files stay in sync
    rows.append(dict(
        Date=dt.normalize(), Timestamp=dt, 
        Prediction=1 if pred==1 else 0,
        Crash_Conf=0.0, Spike_Conf=p_trade, Trade_Conf=p_trade, Confidence=max(p_trade, 1-p_trade)
    ))
    # optional: write through your existing logger to keep formats identical
    try:
        last_bar = spy.loc[dt]
        log_prediction_to_file(dt, pred, 0.0, p_trade,
                               float(last_bar["Close"]),
                               float(last_bar["Open"]),
                               float(last_bar["High"]),
                               float(last_bar["Low"]))
    except Exception:
        pass

if rows:
    df = pd.DataFrame(rows).sort_values(["Date","Timestamp"])
    df = df.drop_duplicates(subset=["Date"], keep="last")
    df.to_csv("logs/predictions_full.csv", index=False)
    print("Wrote logs/predictions_full.csv:", len(df), "rows")
else:
    print("No rows produced; check features/model.")
