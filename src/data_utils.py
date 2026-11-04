"""Utilidades de datos que NO dependen de torch (división por paciente y normalización)."""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def patient_split(meta: pd.DataFrame, val_frac: float, seed: int):
    """Divide en train/val SIN repartir a un mismo paciente entre ambos conjuntos.

    Un paciente aparece varias veces (vistas CC y MLO, varias lesiones); una
    división por filas filtraría información entre train y val.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    tr, va = next(gss.split(meta, groups=meta["patient_id"]))
    return np.sort(tr), np.sort(va)


def normalize_stack(X: np.ndarray) -> np.ndarray:
    """Normalización por recorte: estiramiento de contraste entre percentiles 1 y 99 -> [0,1].

    Quita las diferencias globales de brillantez entre escáneres/películas.
    """
    out = np.empty(X.shape, dtype=np.float32)
    for i, x in enumerate(X):
        lo, hi = np.percentile(x, [1, 99])
        out[i] = np.clip((x.astype(np.float32) - lo) / (hi - lo + 1e-6), 0, 1)
    return out
