"""Métodos de referencia (sin aprendizaje profundo) para comparar con la U-Net.

1. ``center_prior``: un círculo centrado del tamaño medio de las lesiones de
   entrenamiento. Como los recortes están centrados en la lesión, este control
   mide cuánto del Dice se explica solo por la posición; la U-Net debe superarlo
   con holgura para que su resultado sea significativo.
2. ``classical_segment``: pipeline clásico de procesamiento de imágenes:
   suavizado gaussiano -> umbral de Otsu (histograma de la zona central) ->
   apertura/cierre morfológico -> rellenado de huecos -> componente conexa
   central.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage import filters, morphology


def fit_center_prior(Ytr: np.ndarray) -> float:
    """Radio (px) del círculo con el área media de las lesiones de entrenamiento."""
    mean_area = float(Ytr.reshape(len(Ytr), -1).sum(axis=1).mean())
    return float(np.sqrt(mean_area / np.pi))


def center_prior(shape: tuple, radius: float) -> np.ndarray:
    h, w = shape
    yy, xx = np.ogrid[:h, :w]
    return ((yy - (h - 1) / 2) ** 2 + (xx - (w - 1) / 2) ** 2) <= radius ** 2


def classical_segment(patch_u8: np.ndarray, sigma: float = 2.0, window_frac: float = 0.6,
                      open_radius: int = 3, close_radius: int = 3) -> np.ndarray:
    """Segmentación clásica de un recorte (uint8). Devuelve máscara booleana."""
    h, w = patch_u8.shape
    g = filters.gaussian(patch_u8.astype(np.float32) / 255.0, sigma=sigma)

    # Otsu calculado solo en la ventana central, donde está la lesión
    mh, mw = int(h * window_frac / 2), int(w * window_frac / 2)
    ch, cw = h // 2, w // 2
    window = g[ch - mh: ch + mh, cw - mw: cw + mw]
    if window.max() - window.min() < 1e-6:
        return np.zeros((h, w), dtype=bool)
    thr = filters.threshold_otsu(window)
    bw = g > thr

    bw = morphology.opening(bw, morphology.disk(open_radius))
    bw = morphology.closing(bw, morphology.disk(close_radius))
    bw = ndi.binary_fill_holes(bw)

    # quedarse con la componente conexa más cercana al centro del recorte
    lab, n = ndi.label(bw)
    if n == 0:
        return np.zeros((h, w), dtype=bool)
    if lab[ch, cw] > 0:
        return lab == lab[ch, cw]
    dist = ndi.distance_transform_edt(lab == 0, return_indices=True)[1]
    ny, nx = dist[0][ch, cw], dist[1][ch, cw]
    return lab == lab[ny, nx]


def classical_segment_stack(X: np.ndarray, **kw) -> np.ndarray:
    return np.stack([classical_segment(x, **kw) for x in X])
