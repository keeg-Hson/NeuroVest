from __future__ import annotations

from contextlib import suppress

import numpy as np
import pandas as pd


def get_required_features() -> list[str]:
    """Load the model's required feature list."""
    with suppress(Exception):
        return pd.read_csv("models/input_features_fwd.txt", header=None)[0].astype(str).tolist()
    with suppress(Exception):
        return pd.read_csv("models/input_features.txt", header=None)[0].astype(str).tolist()
    return []


def _prep(df: pd.DataFrame, req: list[str]) -> pd.DataFrame:
    """Ensure required columns exist, clean NaNs/inf, simple impute."""
    X = df.copy()
    for c in req:
        if c not in X.columns:
            X[c] = np.nan

    X = X.replace([np.inf, -np.inf], np.nan)

    with suppress(Exception):
        X = X.interpolate(method="time", limit_direction="both")

    X = X.fillna(X.median(numeric_only=True))
    return X[req]
