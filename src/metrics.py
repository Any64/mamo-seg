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


# --------------------------------------------------------------------------- contorno y componentes
def _surface(mask: np.ndarray) -> np.ndarray:
    """Píxeles del borde (la máscara menos su erosión con conectividad 4)."""
    from scipy import ndimage as ndi
    return mask & ~ndi.binary_erosion(mask, border_value=0)


def hausdorff95(pred: np.ndarray, gt: np.ndarray, spacing: float = 1.0) -> float:
    """Distancia de Hausdorff al percentil 95 entre los bordes de ``pred`` y ``gt`` (en px · spacing).

    Convención de MedPy (``hd95``): percentil 95 del conjunto de las distancias de
    cada punto del borde de A al borde de B y de B a A. Devuelve NaN si alguna de las
    dos máscaras está vacía (las predicciones vacías se cuentan aparte).
    """
    from scipy import ndimage as ndi
    pred, gt = pred.astype(bool), gt.astype(bool)
    if not pred.any() or not gt.any():
        return float("nan")
    sp, sg = _surface(pred), _surface(gt)
    dist_to_g = ndi.distance_transform_edt(~sg, sampling=spacing)
    dist_to_p = ndi.distance_transform_edt(~sp, sampling=spacing)
    return float(np.percentile(np.concatenate([dist_to_g[sp], dist_to_p[sg]]), 95))


def n_components(mask: np.ndarray) -> int:
    """Número de componentes conexas (conectividad 8)."""
    from scipy import ndimage as ndi
    return int(ndi.label(mask.astype(bool), structure=np.ones((3, 3)))[1])


def extended_metrics(pred: np.ndarray, gt: np.ndarray) -> pd.DataFrame:
    """``per_sample_metrics`` más Hausdorff 95 (px del recorte; NaN si la predicción es vacía),
    número de componentes conexas y si la predicción es vacía."""
    df = per_sample_metrics(pred, gt)
    df["hd95"] = [hausdorff95(p, g) for p, g in zip(pred, gt)]
    df["n_componentes"] = [n_components(p) for p in pred]
    df["vacia"] = ~pred.reshape(len(pred), -1).any(axis=1)
    return df
