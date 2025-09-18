# walk_forward.py
import pandas as pd

def walk_forward_predict(df, retrain_every=60):
    """
    Retrain every N bars, append next-window predictions to logs/predictions_full.csv.
    """
    out = []
    for i in range(400, len(df)-1, retrain_every):
        train_df = df.iloc[:i].copy()
        test_df  = df.iloc[i:i+retrain_every].copy()
        # call existing training function on train_df -> model
        # run predict on test_df -> proba + class
        # append to 'out'
    pd.DataFrame(out).to_csv("logs/predictions_full.csv", index=False)
