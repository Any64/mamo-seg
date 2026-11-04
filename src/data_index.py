"""Lectura de los CSV de CBIS-DDSM y resolución de las rutas reales de los archivos.

Particularidades de la versión de Kaggle (documentadas durante la exploración):

* Las rutas del CSV tienen la forma ``caso/UID_A/UID_B/archivo.dcm``. Solo el
  penúltimo segmento (UID_B) coincide con una carpeta real dentro de ``jpeg/``.
* Los archivos dentro de cada carpeta se llaman ``1-250.jpg``, ``2-249.jpg``...
  sin indicar si son recorte o máscara. A veces recorte y máscara comparten carpeta.
* La máscara es casi binaria (pocos valores de gris distintos) y el recorte tiene
  un rango de grises completo: así se distinguen.
* La máscara ROI NO está registrada con el recorte oficial: normalmente tiene el
  tamaño de la mamografía completa. Por eso la imagen base es la mamografía
  completa y los recortes ROI se generan en ``preprocess.py``.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # las mamografías completas superan el límite por defecto

MAX_MASK_UNIQUE = 100  # una máscara tiene muchos menos valores de gris distintos que una imagen


def load_case_csv(csv_dir: Path, split: str) -> pd.DataFrame:
    """Carga el CSV oficial de masas. ``split`` es 'train' o 'test'."""
    path = Path(csv_dir) / f"mass_case_description_{split}_set.csv"
    if not path.exists():
        raise FileNotFoundError(f"No se encontró {path}")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def detect_columns(df: pd.DataFrame):
    """Devuelve (col_full, col_crop, col_mask) buscando palabras clave en los nombres."""
    col_crop = next(c for c in df.columns if "cropped" in c.lower())
    col_mask = next(c for c in df.columns if "roi" in c.lower() or "mask" in c.lower())
    col_full = next(c for c in df.columns
                    if "image file path" in c.lower() and "cropped" not in c.lower())
    return col_full, col_crop, col_mask


def uid_of(path_str: str) -> str:
    """Penúltimo segmento de la ruta del CSV = nombre de la carpeta en jpeg/."""
    parts = str(path_str).strip().replace("\\", "/").split("/")
    return parts[-2]


def list_jpgs(jpeg_dir: Path, uid: str):
    folder = Path(jpeg_dir) / uid
    return sorted(folder.glob("*.jpg")) if folder.is_dir() else []


def image_size(path: Path):
    """(ancho, alto) leyendo solo la cabecera del archivo."""
    with Image.open(path) as im:
        return im.size


def n_unique_gray(path: Path, reduce: int = 8) -> int:
    """Número de niveles de gris distintos (en una versión reducida, para ir rápido)."""
    with Image.open(path) as im:
        w, h = im.size
        if im.format == "JPEG" and min(w, h) >= reduce * 64:
            im.draft("L", (w // reduce, h // reduce))  # decodificación JPEG reducida
        arr = np.asarray(im.convert("L"))
    return int(len(np.unique(arr)))


def resolve_row(jpeg_dir: Path, row: pd.Series, cols) -> dict:
    """Encuentra, para una fila del CSV, la imagen base y la máscara alineadas.

    Devuelve un diccionario con ``status`` ('ok' o el motivo de descarte),
    ``mode`` ('full', 'full_rescaled' o 'crop'), ``base`` y ``mask`` (rutas).
    """
    col_full, col_crop, col_mask = cols
    out = {"status": "ok", "mode": None, "base": None, "mask": None}

    full_files = list_jpgs(jpeg_dir, uid_of(row[col_full]))
    if not full_files:
        out["status"] = "missing_full"
        return out
    full = max(full_files, key=lambda p: int(np.prod(image_size(p))))

    cand = []
    for c in (col_mask, col_crop):
        for p in list_jpgs(jpeg_dir, uid_of(row[c])):
            if p not in cand:
                cand.append(p)
    if not cand:
        out["status"] = "missing_mask"
        return out

    uniq = {p: n_unique_gray(p) for p in cand}
    mask = min(cand, key=lambda p: uniq[p])
    if uniq[mask] > MAX_MASK_UNIQUE:
        out["status"] = "mask_ambiguous"
        return out
    others = [p for p in cand if p != mask]
    crop = max(others, key=lambda p: int(np.prod(image_size(p)))) if others else None

    fw, fh = image_size(full)
    mw, mh = image_size(mask)
    if (mw, mh) == (fw, fh):
        out.update(mode="full", base=full, mask=mask)
    elif abs((mw / mh) - (fw / fh)) / (fw / fh) < 0.02:
        out.update(mode="full_rescaled", base=full, mask=mask)
    elif crop is not None and image_size(crop) == (mw, mh):
        out.update(mode="crop", base=crop, mask=mask)
    else:
        out["status"] = "mask_unaligned"
    return out
