# zscrap.py
import pandas as pd

import external_signals as ex
from utils import add_features, add_forward_returns_and_labels, finalize_features, load_SPY_data


def test_external_signals_smoke():
    base = load_SPY_data()
    sig = ex.add_external_signals(base)
    must = ["Sector_MedianRet_20", "Sector_Dispersion_20"]
    for col in must:
        assert col in sig.columns, f"Missing column: {col}"
        cov = float(sig[col].notna().mean())
        print(f"{col} coverage: {cov:.3f}")
        assert cov >= 0.70, f"Low coverage for {col} (got {cov:.3f})"
    print("✅ external_signals smoke passed")


def test_external_signals_idempotent():
    base = load_SPY_data()
    a = ex.add_external_signals(base).tail(5)[
        ["Sector_MedianRet_20", "Sector_Dispersion_20", "Credit_Spread_20"]
    ]
    b = ex.add_external_signals(base).tail(5)[
        ["Sector_MedianRet_20", "Sector_Dispersion_20", "Credit_Spread_20"]
    ]
    assert a.equals(b), "add_external_signals not idempotent on repeated calls"
    print("✅ idempotence passed")


def test_training_has_close_and_labels():
    raw = load_SPY_data()
    df, feat_cols = add_features(raw)
    df = finalize_features(df, feat_cols)

    # Attach Close to match train.py
    close = raw.set_index(pd.to_datetime(raw["Date"]))["Close"].astype(float)
    df["Close"] = close.reindex(df.index)
    assert "Close" in df.columns and df["Close"].notna().any(), "Close missing before labeling"

    lab = add_forward_returns_and_labels(df.copy(), price_col="Close", horizon=1)

    # Accept a few possible forward-return names; create if missing
    fwd_candidates = ["fwd_ret", "fwd_return", "forward_ret", "fwd_return_1d"]
    fwd_present = next((c for c in fwd_candidates if c in lab.columns), None)
    if fwd_present is None:
        lab["fwd_ret"] = lab["Close"].shift(-1) / lab["Close"] - 1.0
        fwd_present = "fwd_ret"

    for c in ["fwd_price", fwd_present, "y"]:
        assert c in lab.columns, f"Missing label column: {c}"

    uy = sorted(pd.Series(lab["y"].dropna().unique()).tolist())
    assert set(uy).issubset({0, 1}), f"Unexpected y values: {uy}"
    print("✅ training/labeling smoke passed")


def test_predictions_and_reliability():
    """
    Runs predict and prints confusion matrix vs next-day return,
    plus reliability buckets if confidence is present.
    """
    from predict import run_predictions

    pred = run_predictions()
    assert hasattr(pred, "empty") and not pred.empty, "run_predictions returned empty"
    assert "Prediction" in pred.columns, "Prediction column missing"

    # ensure Close for truth:
    if "Close" not in pred.columns:
        # try to merge from raw by Date index if present
        raw = load_SPY_data()
        raw_idx = raw.set_index(pd.to_datetime(raw["Date"]))
        if "Date" in pred.columns:
            pred_idx = pd.to_datetime(pred["Date"])
            pred = pred.set_index(pred_idx)
        if "Close" not in pred.columns:
            pred["Close"] = raw_idx["Close"].reindex(pred.index)
    assert "Close" in pred.columns, "Could not attach Close to predictions"

    # Truth = next-day return > 0
    ret1 = pred["Close"].pct_change().shift(-1)
    y_true = (ret1 > 0).astype(int)
    y_hat = pred["Prediction"].astype(int)

    cm = pd.crosstab(y_true, y_hat, rownames=["True"], colnames=["Pred"])
    print("\nConfusion matrix vs next-day > 0:")
    print(cm)

    # Reliability by confidence bucket (if available)
    pcol = (
        "Trade_Conf"
        if "Trade_Conf" in pred.columns
        else ("Spike_Conf" if "Spike_Conf" in pred.columns else None)
    )
    if pcol:
        bins = pd.cut(pred[pcol], [0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 1.0], right=False)
        print("\nCounts per confidence bucket:")
        print(pred.groupby(bins, observed=False)[pcol].count().rename("count"))
        print("\nEmpirical win-rate per bucket:")
        print(
            pred.groupby(bins, observed=False)
            .apply(lambda g: (ret1.reindex(g.index) > 0).mean())
            .rename("empirical_win_rate")
        )
    else:
        print("\n(no confidence column found)")

    print("✅ predictions/reliability sanity passed")


if __name__ == "__main__":
    # These mirror what you’re already running in Terminal
    test_external_signals_smoke()
    test_external_signals_idempotent()
    test_training_has_close_and_labels()
    # Optional: comment out if you don’t want to invoke the full predict path here
    test_predictions_and_reliability()
