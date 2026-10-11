import numpy as np
import pandas as pd

def build_features(X_raw):
    X_f = X_raw.copy()
    for i in range(1, 6):
        d = X_f[f"servo{i}_angulo_final"] - X_f[f"servo{i}_angulo_inicial"]
        X_f[f"servo{i}_delta_abs"] = d.abs()
        X_f[f"servo{i}_delta"] = d

    col_deltas = [f"servo{i}_delta_abs" for i in range(1, 6)]
    X_f["soma_angular_total"] = X_f[col_deltas].sum(axis=1)

    for s in [2, 4, 5]:
        rad = np.radians(X_f[f"servo{s}_angulo_final"])
        X_f[f"servo{s}_cos_fim"] = np.cos(rad)
        X_f[f"servo{s}_sin_fim"] = np.sin(rad)

    return X_f