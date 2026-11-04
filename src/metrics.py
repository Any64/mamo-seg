"""Métricas de segmentación (solo numpy/pandas)."""
import numpy as np
import pandas as pd

EPS = 1e-6


# --------------------------------------------------------------------------- métricas
def dice_np(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    return float((2 * (pred & gt).sum() + EPS) / (pred.sum() + gt.sum() + EPS))


def iou_np(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    return float(((pred & gt).sum() + EPS) / ((pred | gt).sum() + EPS))


def per_sample_metrics(pred: np.ndarray, gt: np.ndarray) -> pd.DataFrame:
    """Dice, IoU, precisión y sensibilidad (recall) por muestra. pred y gt: (N,H,W)."""
    rows = []
    for p, g in zip(pred.astype(bool), gt.astype(bool)):
        tp = (p & g).sum()
        rows.append({
            "dice": dice_np(p, g),
            "iou": iou_np(p, g),
            "precision": float((tp + EPS) / (p.sum() + EPS)),
            "recall": float((tp + EPS) / (g.sum() + EPS)),
        })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    """Media y desviación estándar de cada métrica."""
    out = {}
    for c in df.columns:
        out[f"{c}_mean"] = float(df[c].mean())
        out[f"{c}_std"] = float(df[c].std())
    return out
