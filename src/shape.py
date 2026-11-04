"""Caracterización de la forma de la lesión a partir de su máscara.

Etapa de *interpretación geométrica* posterior a la segmentación: se calculan
descriptores del contorno (circularidad, solidez, excentricidad, concavidades,
variación radial, lóbulos) y un árbol de decisión poco profundo (interpretable)
asigna una de las 4 formas básicas del léxico BI-RADS (v2025):
ROUND / OVAL / LOBULATED / IRREGULAR.

No se entrena ninguna red adicional: el árbol trabaja sobre ~8 números por
máscara y sus reglas pueden leerse e incluirse en el informe.
"""
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.signal import find_peaks
from skimage import measure, morphology
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

SHAPE_CLASSES = ["ROUND", "OVAL", "LOBULATED", "IRREGULAR"]
FEATURE_NAMES = ["circularity", "solidity", "eccentricity", "axis_ratio",
                 "n_concavities", "roughness", "radial_cv", "n_lobes"]
MIN_AREA_PX = 30


def largest_component(mask: np.ndarray) -> np.ndarray:
    lab, n = ndi.label(mask)
    if n <= 1:
        return mask.astype(bool)
    areas = ndi.sum(mask, lab, index=np.arange(1, n + 1))
    return lab == (int(np.argmax(areas)) + 1)


def radial_signature(mask: np.ndarray, n_points: int = 128) -> np.ndarray:
    """Distancia del contorno al centroide, remuestreada a ``n_points`` por longitud de arco."""
    padded = np.pad(mask.astype(float), 1)
    contours = measure.find_contours(padded, 0.5)
    c = max(contours, key=len)
    seg = np.sqrt((np.diff(c, axis=0) ** 2).sum(axis=1))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.linspace(0, s[-1], n_points, endpoint=False)
    y, x = np.interp(t, s, c[:, 0]), np.interp(t, s, c[:, 1])
    cy, cx = ndi.center_of_mass(mask)
    return np.hypot(y - 1 - cy, x - 1 - cx)


def count_lobes(r: np.ndarray, prominence: float = 0.04, sigma: float = 2.0) -> int:
    """Número de máximos de la firma radial (normalizada y suavizada, periódica)."""
    r = ndi.gaussian_filter1d(r / r.mean(), sigma=sigma, mode="wrap")
    tiled = np.tile(r, 3)
    peaks, _ = find_peaks(tiled, prominence=prominence)
    n = len(r)
    return int(((peaks >= n) & (peaks < 2 * n)).sum())


def smooth_mask(mask: np.ndarray, sigma: float = 0.0) -> np.ndarray:
    """Suaviza el contorno: filtro gaussiano (``sigma`` en px) y binarización en 0.5.

    Atenúa el dentado de los contornos anotados de CBIS-DDSM. Con ``sigma=0`` no
    modifica la máscara.
    """
    m = np.asarray(mask).astype(bool)
    if sigma <= 0:
        return m
    return ndi.gaussian_filter(m.astype(np.float32), sigma=sigma) > 0.5


def shape_features(mask: np.ndarray, smooth_sigma: float = 0.0) -> dict:
    """Descriptores de forma de una máscara binaria. Devuelve NaN si la máscara es inválida.

    ``smooth_sigma`` > 0 suaviza el contorno antes de medir (ver ``smooth_mask``).
    """
    nan = {k: np.nan for k in FEATURE_NAMES}
    m = largest_component(smooth_mask(mask, smooth_sigma))
    area = int(m.sum())
    if area < MIN_AREA_PX:
        return nan
    try:
        props = measure.regionprops(m.astype(np.uint8))[0]
        perim = max(measure.perimeter(m), 1e-6)
        hull = morphology.convex_hull_image(m)
        hull_perim = max(measure.perimeter(hull), 1e-6)

        conc = morphology.opening(hull & ~m, morphology.disk(1))
        lab, n = ndi.label(conc)
        n_conc = 0
        if n:
            areas = ndi.sum(conc, lab, index=np.arange(1, n + 1))
            n_conc = int((areas >= 0.01 * area).sum())

        r = radial_signature(m)
        return {
            "circularity": float(min(1.0, 4 * np.pi * area / perim ** 2)),
            "solidity": float(props.solidity),
            "eccentricity": float(props.eccentricity),
            "axis_ratio": float(props.axis_minor_length / max(props.axis_major_length, 1e-6)),
            "n_concavities": n_conc,
            "roughness": float(perim / hull_perim),
            "radial_cv": float(r.std() / r.mean()),
            "n_lobes": count_lobes(r),
        }
    except Exception:
        return nan


def features_dataframe(masks: np.ndarray, smooth_sigma: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame([shape_features(m, smooth_sigma) for m in masks], columns=FEATURE_NAMES)


def fit_shape_tree(F: pd.DataFrame, y, max_depth: int = 3, seed: int = 42):
    """Árbol de decisión poco profundo (interpretable) sobre los descriptores."""
    ok = F.notna().all(axis=1).values
    clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=10,
                                 class_weight="balanced", random_state=seed)
    clf.fit(F.loc[ok, FEATURE_NAMES].values, np.asarray(y)[ok])
    return clf


def predict_shape(clf, F: pd.DataFrame) -> np.ndarray:
    """Predice la forma; 'NONE' si la máscara no permitió calcular descriptores."""
    pred = np.full(len(F), "NONE", dtype=object)
    ok = F.notna().all(axis=1).values
    if ok.any():
        pred[ok] = clf.predict(F.loc[ok, FEATURE_NAMES].values)
    return pred


def evaluate_shape(y_true, y_pred, labels=SHAPE_CLASSES) -> dict:
    """Exactitud, exactitud balanceada, F1 macro y matriz de confusión (filas=real,
    columnas=predicho + 'NONE')."""
    y_true, y_pred = np.asarray(y_true, dtype=object), np.asarray(y_pred, dtype=object)
    cm = confusion_matrix(y_true, y_pred, labels=labels + ["NONE"])[: len(labels)]
    return {
        "n": int(len(y_true)),
        "n_invalid": int((y_pred == "NONE").sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro",
                                   zero_division=0)),
        "confusion": cm,
    }


def tree_rules(clf) -> str:
    return export_text(clf, feature_names=FEATURE_NAMES)


# --------------------------------------------------------------------------- experimentos
TREE, LOGREG, FOREST, MAJORITY = ("árbol (prof. 3)", "regresión logística", "random forest",
                                  "clase mayoritaria")


def make_classifier(name: str, seed: int = 42):
    """Clasificadores usados en los experimentos de forma (todos con clases balanceadas)."""
    if name == TREE:
        return DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, class_weight="balanced",
                                      random_state=seed)
    if name == LOGREG:
        return make_pipeline(StandardScaler(),
                             LogisticRegression(class_weight="balanced", max_iter=5000))
    if name == FOREST:
        return RandomForestClassifier(n_estimators=500, min_samples_leaf=5, class_weight="balanced",
                                      n_jobs=-1, random_state=seed)
    if name == MAJORITY:
        return DummyClassifier(strategy="most_frequent")
    raise ValueError(f"Clasificador desconocido: {name}")


def fit_classifier(name: str, F: pd.DataFrame, y, seed: int = 42):
    """Ajusta ``name`` sobre las filas con descriptores válidos (sin NaN)."""
    ok = F.notna().all(axis=1).values
    return make_classifier(name, seed).fit(F.loc[ok, FEATURE_NAMES].values, np.asarray(y)[ok])


def binary_scores(y_true, prob) -> dict:
    """AUC-ROC y exactitud balanceada (umbral 0.5 sobre la probabilidad de la clase positiva)."""
    prob = np.asarray(prob)
    return {"AUC": float(roc_auc_score(y_true, prob)),
            "exactitud balanceada": float(balanced_accuracy_score(y_true, prob >= 0.5))}
