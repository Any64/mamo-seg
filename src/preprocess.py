"""Preprocesamiento: genera pares (recorte ROI, máscara) alineados y de tamaño fijo.

Idea: la máscara de CBIS-DDSM vive en las coordenadas de la mamografía completa.
Se calcula la caja envolvente de la lesión, se recorta un cuadrado centrado en
ella (``context`` veces el lado mayor de la caja) tanto de la mamografía como de
la máscara, y ambos se reescalan a ``img_size`` x ``img_size``. Así imagen y
máscara quedan registradas por construcción.
"""
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi

from .data_index import detect_columns, resolve_row

try:
    from tqdm import tqdm
except ImportError:  # tqdm es opcional
    def tqdm(it, **kw):
        return it

SHAPE_CLASSES = ["ROUND", "OVAL", "LOBULATED", "IRREGULAR"]


def clean_shape_label(label) -> str:
    """Devuelve la forma si es una de las 4 básicas de BI-RADS; si no, ''.

    Etiquetas combinadas (p. ej. 'IRREGULAR-ARCHITECTURAL_DISTORTION') o de otro
    tipo (lymph_node, asymmetric...) quedan fuera del análisis de forma.
    """
    s = str(label).strip().upper()
    return s if s in SHAPE_CLASSES else ""


def clean_mask(mask_bin: np.ndarray, min_area: int = 64) -> np.ndarray:
    """Quita motas pequeñas (ruido JPEG) conservando los componentes importantes."""
    lab, n = ndi.label(mask_bin)
    if n <= 1:
        return mask_bin
    areas = ndi.sum(mask_bin, lab, index=np.arange(1, n + 1))
    keep = np.where(areas >= max(min_area, 0.02 * areas.max()))[0] + 1
    if len(keep) == 0:
        return mask_bin
    return np.isin(lab, keep)


def bbox_of(mask_bin: np.ndarray):
    rows = np.flatnonzero(mask_bin.any(axis=1))
    cols = np.flatnonzero(mask_bin.any(axis=0))
    return rows[0], rows[-1], cols[0], cols[-1]


def crop_square(arr: np.ndarray, cy: float, cx: float, side: int) -> np.ndarray:
    """Recorta un cuadrado de lado ``side`` centrado en (cy, cx); rellena con 0 fuera de imagen."""
    y0 = int(round(cy - side / 2))
    x0 = int(round(cx - side / 2))
    out = np.zeros((side, side), dtype=arr.dtype)
    ys, xs = max(y0, 0), max(x0, 0)
    ye, xe = min(y0 + side, arr.shape[0]), min(x0 + side, arr.shape[1])
    if ye > ys and xe > xs:
        out[ys - y0: ye - y0, xs - x0: xe - x0] = arr[ys:ye, xs:xe]
    return out


def make_patch(img: np.ndarray, mask_bin: np.ndarray, img_size: int,
               context: float, min_side: int, jitter: float = 0.0, rng=None):
    """Recorta y reescala (imagen, máscara). Devuelve (img_u8, mask_u8, info).

    ``jitter`` desplaza al azar el centro del recorte (hasta ±jitter*lado) para que
    la lesión NO quede siempre en el centro: si lo estuviera, un círculo centrado
    ya obtendría un Dice altísimo y la tarea sería casi trivial.
    """
    y0, y1, x0, x1 = bbox_of(mask_bin)
    bh, bw = int(y1 - y0 + 1), int(x1 - x0 + 1)
    cy, cx = (y0 + y1) / 2.0, (x0 + x1) / 2.0
    side = int(round(max(max(bh, bw) * context, min_side)))
    sy = sx = 0.0
    if jitter > 0:
        rng = rng if rng is not None else np.random.default_rng(0)
        sy, sx = rng.uniform(-jitter, jitter, size=2)
        cy, cx = cy + sy * side, cx + sx * side

    ip = crop_square(img, cy, cx, side)
    mp = crop_square(mask_bin, cy, cx, side).astype(np.uint8) * 255

    ip = np.asarray(Image.fromarray(ip).resize((img_size, img_size), Image.BILINEAR))
    mp = np.asarray(Image.fromarray(mp).resize((img_size, img_size), Image.BILINEAR))
    mp = (mp > 127).astype(np.uint8)
    info = {"bbox_h": bh, "bbox_w": bw, "side_px": side,
            "shift_y": float(sy), "shift_x": float(sx),
            "lesion_area_frac": float(mp.mean())}
    return ip, mp, info


def patch_from_resolved(res: dict, img_size: int, context: float, min_side: int,
                        jitter: float = 0.0, rng=None):
    with Image.open(res["base"]) as im:
        base = np.asarray(im.convert("L"))
    with Image.open(res["mask"]) as im:
        if res["mode"] == "full_rescaled":
            im = im.resize((base.shape[1], base.shape[0]), Image.BILINEAR)
        mask_bin = np.asarray(im.convert("L")) > 127
    mask_bin = clean_mask(mask_bin)
    if mask_bin.sum() == 0:
        raise ValueError("máscara vacía")
    ip, mp, info = make_patch(base, mask_bin, img_size, context, min_side, jitter, rng)
    info.update(full_h=base.shape[0], full_w=base.shape[1], mode=res["mode"])
    return ip, mp, info


def _process_row(task):
    """Procesa una fila del CSV (función de módulo para poder usarla en procesos paralelos).

    Devuelve (estado, (img, máscara, info) o None).
    """
    idx, row, jpeg_dir, cols, img_size, context, min_side, jitter, seed = task
    res = resolve_row(jpeg_dir, row, cols)
    if res["status"] != "ok":
        return res["status"], None
    try:
        rng = np.random.default_rng([seed, int(idx)])  # desplazamiento reproducible por fila
        return "ok", patch_from_resolved(res, img_size, context, min_side, jitter, rng)
    except Exception as e:  # una fila rota no debe detener todo el proceso
        return f"error:{type(e).__name__}", None


def build_split(df: pd.DataFrame, jpeg_dir: Path, img_size: int, context: float,
                min_side: int, jitter: float = 0.0, seed: int = 0, progress: bool = True,
                workers: int = 1):
    """Procesa todas las filas de un CSV. Devuelve (X, Y, meta, conteo_de_estados).

    Con ``workers`` > 1 las filas se procesan en paralelo (procesos); el resultado
    es idéntico al secuencial porque el desplazamiento de cada fila depende solo
    de (seed, índice de la fila).
    """
    cols = detect_columns(df)
    X, Y, rows = [], [], []
    status = Counter()
    items = list(df.iterrows())
    tasks = [(idx, row, jpeg_dir, cols, img_size, context, min_side, jitter, seed) for idx, row in items]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as ex:
            outputs = list(tqdm(ex.map(_process_row, tasks, chunksize=4), total=len(tasks),
                                desc="recortes") if progress else ex.map(_process_row, tasks, chunksize=4))
    else:
        outputs = map(_process_row, tqdm(tasks, desc="recortes") if progress else tasks)
    for (idx, row), (st, out) in zip(items, outputs):
        if st != "ok":
            status[st] += 1
            continue
        ip, mp, info = out
        status["ok"] += 1
        X.append(ip)
        Y.append(mp)
        shape_raw = row.get("mass shape", "")
        rows.append({
            "row_idx": idx,
            "patient_id": row.get("patient_id", ""),
            "view": row.get("image view", ""),
            "side": row.get("left or right breast", ""),
            "abnormality_id": row.get("abnormality id", ""),
            "mass_shape_raw": shape_raw,
            "shape_label": clean_shape_label(shape_raw),
            "mass_margins": row.get("mass margins", ""),
            "pathology": str(row.get("pathology", "")).split("_")[0],  # BENIGN_WITHOUT_CALLBACK -> BENIGN
            "subtlety": row.get("subtlety", np.nan),
            "assessment": row.get("assessment", np.nan),
            "breast_density": row.get("breast_density", np.nan),
            **info,
        })
    X = np.stack(X) if X else np.zeros((0, img_size, img_size), np.uint8)
    Y = np.stack(Y) if Y else np.zeros((0, img_size, img_size), np.uint8)
    return X, Y, pd.DataFrame(rows), status


def save_split(processed_dir: Path, name: str, X, Y, meta: pd.DataFrame) -> None:
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(processed_dir / f"{name}.npz", X=X, Y=Y)
    meta.to_csv(processed_dir / f"{name}_meta.csv", index=False)


def load_split(processed_dir: Path, name: str):
    processed_dir = Path(processed_dir)
    d = np.load(processed_dir / f"{name}.npz")
    meta = pd.read_csv(processed_dir / f"{name}_meta.csv", keep_default_na=False)
    meta["subtlety"] = pd.to_numeric(meta["subtlety"], errors="coerce")
    return d["X"], d["Y"], meta


def alignment_contrast(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Contraste medio (dentro - fuera de la máscara) por muestra, en niveles de gris.

    Chequeo objetivo de que máscara e imagen están registradas: las masas suelen
    ser más densas (más brillantes) que el tejido que las rodea, así que la
    mayoría de los valores debería ser > 0.
    """
    out = np.zeros(len(X), dtype=np.float32)
    for i in range(len(X)):
        m = Y[i].astype(bool)
        if m.any() and (~m).any():
            out[i] = X[i][m].mean() - X[i][~m].mean()
    return out
