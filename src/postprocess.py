"""Posprocesamiento morfológico FIJO de máscaras binarias (referencia sin entrenar).

Apertura y luego cierre con un disco de radio ``r`` y, después, solo la componente
conexa más grande. ``r = 0`` aplica únicamente el filtro de componente.
"""
import numpy as np
from scipy import ndimage as ndi


def disk(radius: int) -> np.ndarray:
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    return (xx ** 2 + yy ** 2) <= radius ** 2


def largest_component(mask: np.ndarray) -> np.ndarray:
    lab, n = ndi.label(mask, structure=np.ones((3, 3)))
    if n <= 1:
        return mask.astype(bool)
    sizes = np.bincount(lab.ravel())[1:]
    return lab == (1 + int(sizes.argmax()))


def postprocess_mask(mask: np.ndarray, radius: int = 0, keep_largest: bool = True) -> np.ndarray:
    m = mask.astype(bool)
    if radius > 0:
        se = disk(radius)
        # bordes rellenados con el valor más cercano para no erosionar desde fuera del recorte
        pad = radius
        mp = np.pad(m, pad, mode="edge")
        mp = ndi.binary_closing(ndi.binary_opening(mp, se), se)
        m = mp[pad:-pad, pad:-pad]
    return largest_component(m) if keep_largest else m


def postprocess_stack(masks: np.ndarray, radius: int = 0, keep_largest: bool = True) -> np.ndarray:
    return np.stack([postprocess_mask(m, radius, keep_largest) for m in masks]) if len(masks) else masks


DROP_THRESHOLD = 0.1   # "Dice casi 0"
CAUSE_EMPTY = "a) la apertura vacía la máscara"
CAUSE_COMPONENT = "b) componente mayor equivocada"
CAUSE_OTHER = "c) otra"


def drop_cause(dice_unet: float, dice_r0: float, dice_r: float, empty_r: bool,
               thr: float = DROP_THRESHOLD):
    """Causa por la que el posprocesamiento con radio r lleva un recorte a Dice < ``thr``.

    None si no hay caída (la U-Net ya tenía Dice < thr, o el posprocesado no baja de thr).
    b) si con r = 0 (solo la componente mayor) ya cae; si no, a) si con radio r queda
    vacía; c) en cualquier otro caso.
    """
    if dice_unet < thr or dice_r >= thr:
        return None
    if dice_r0 < thr:
        return CAUSE_COMPONENT
    if empty_r:
        return CAUSE_EMPTY
    return CAUSE_OTHER
