"""Genera un 'archive' sintético con la MISMA estructura que el dataset de Kaggle
(CBIS-DDSM), para probar el pipeline sin necesidad de los datos reales:

  archive/csv/mass_case_description_{train,test}_set.csv
  archive/jpeg/<UID>/<n>-<m>.jpg

Reproduce las particularidades reales: rutas de 3 niveles en el CSV, nombres de
archivo sin significado, recorte y máscara compartiendo carpeta (o en carpetas
separadas), y máscara del tamaño de la mamografía completa (con ruido JPEG).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from skimage.draw import polygon

SHAPES = ["ROUND", "OVAL", "LOBULATED", "IRREGULAR"]


def _uid(rng) -> str:
    return "1.3.6.1.4.1.9590.100.1.2." + "".join(str(d) for d in rng.integers(0, 10, size=38))


def _radius_profile(shape: str, theta: np.ndarray, rng) -> np.ndarray:
    if shape == "ROUND":
        return 1 + 0.02 * np.cos(3 * theta + rng.uniform(0, 6))
    if shape == "OVAL":
        ratio = rng.uniform(0.5, 0.7)
        return 1.0 / np.sqrt((np.cos(theta) / 1.0) ** 2 + (np.sin(theta) / ratio) ** 2)
    if shape == "LOBULATED":
        k = rng.integers(4, 7)
        return 1 + 0.22 * np.cos(k * theta + rng.uniform(0, 6))
    r = np.ones_like(theta)  # IRREGULAR
    for k in range(2, 9):
        r += rng.uniform(0.04, 0.16) * np.cos(k * theta + rng.uniform(0, 6))
    return np.clip(r, 0.4, None)


def _lesion_mask(shape_name, H, W, rng):
    base = int(rng.integers(45, 90))
    cy = int(rng.integers(int(0.3 * H), int(0.7 * H)))
    cx = int(rng.integers(int(0.3 * W), int(0.7 * W)))
    theta = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    r = base * _radius_profile(shape_name, theta, rng)
    rot = rng.uniform(0, np.pi)
    ys = cy + r * np.sin(theta + rot)
    xs = cx + r * np.cos(theta + rot)
    m = np.zeros((H, W), dtype=bool)
    rr, cc = polygon(ys, xs, shape=m.shape)
    m[rr, cc] = True
    return m


def _mammogram(mask, rng):
    H, W = mask.shape
    bg = ndi.gaussian_filter(rng.normal(0, 1, (H, W)), 18)
    bg = (bg - bg.min()) / (bg.max() - bg.min())
    img = 70 + 70 * bg + rng.normal(0, 6, (H, W))
    img += 55 * ndi.gaussian_filter(mask.astype(float), 3)  # la lesión es más brillante
    return np.clip(img, 0, 255).astype(np.uint8)


def _save_jpg(arr, path: Path, quality=92):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, quality=quality)


def _crop_of(img, mask):
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    side = int(max(y1 - y0, x1 - x0) * 1.1)
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    ya, xa = max(cy - side // 2, 0), max(cx - side // 2, 0)
    return img[ya: ya + side, xa: xa + side]


def make_archive(root: Path, n_train_patients=40, n_test_patients=12, seed=0,
                 H=1100, W=800) -> Path:
    rng = np.random.default_rng(seed)
    root = Path(root)
    jpeg, csv = root / "jpeg", root / "csv"
    csv.mkdir(parents=True, exist_ok=True)
    counter = {"n": 0}

    def next_name(k):
        counter["n"] += 1
        return f"{k}-{int(rng.integers(100, 999))}.jpg"

    def make_split(split, n_patients):
        rows = []
        prefix = "Mass-Training" if split == "train" else "Mass-Test"
        for p in range(n_patients):
            pid = f"P_{p + 1:05d}"
            for view in ("CC", "MLO"):
                shape = SHAPES[(p + (view == "MLO")) % 4] if rng.random() > 0.15 else SHAPES[int(rng.integers(0, 4))]
                label = shape
                if rng.random() < 0.08:
                    label = "IRREGULAR-ARCHITECTURAL_DISTORTION"
                mask = _lesion_mask(shape, H, W, rng)
                img = _mammogram(mask, rng)
                crop = _crop_of(img, mask)
                mask_u8 = mask.astype(np.uint8) * 255

                uid_full, uid_roi = _uid(rng), _uid(rng)
                uid_full_b, uid_roi_b = _uid(rng), _uid(rng)
                _save_jpg(img, jpeg / uid_full_b / next_name(1))
                if rng.random() < 0.7:   # recorte y máscara comparten carpeta
                    order = rng.random() < 0.5
                    _save_jpg(mask_u8, jpeg / uid_roi_b / next_name(1 if order else 2))
                    _save_jpg(crop, jpeg / uid_roi_b / next_name(2 if order else 1))
                    uid_mask_b = uid_roi_b
                else:                    # carpetas separadas
                    uid_mask_b = _uid(rng)
                    _save_jpg(mask_u8, jpeg / uid_mask_b / next_name(1))
                    _save_jpg(crop, jpeg / uid_roi_b / next_name(1))

                case = f"{prefix}_{pid}_LEFT_{view}"
                rows.append({
                    "patient_id": pid, "breast_density": 3, "left or right breast": "LEFT",
                    "image view": view, "abnormality id": 1, "abnormality type": "mass",
                    "mass shape": label, "mass margins": "CIRCUMSCRIBED", "assessment": 4,
                    "pathology": ["BENIGN", "MALIGNANT", "BENIGN_WITHOUT_CALLBACK"][int(rng.integers(0, 3))],
                    "subtlety": int(rng.integers(1, 6)),
                    "image file path": f"{case}/{uid_full}/{uid_full_b}/000000.dcm",
                    "cropped image file path": f"{case}_1/{uid_roi}/{uid_roi_b}/000000.dcm",
                    "ROI mask file path": f"{case}_1/{uid_roi}/{uid_mask_b}/000001.dcm\n",
                })
        pd.DataFrame(rows).to_csv(csv / f"mass_case_description_{split}_set.csv", index=False)

    make_split("train", n_train_patients)
    make_split("test", n_test_patients)
    return root


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("synthetic_archive")
    make_archive(out)
    print("Dataset sintético creado en", out.resolve())
