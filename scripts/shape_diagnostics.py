"""Diagnóstico de los descriptores: ¿error del pipeline o limitación de los datos?

1) Tamaño -> patología (regresión logística): side_px, lesion_area_frac y ambos.
2) Extremos de forma: ROUND vs IRREGULAR con los 8 descriptores (sigma=0).
3) Figura de comprobación visual: 8 ROUND y 8 IRREGULAR de train con sus descriptores.
4) Emparejamiento: lesion_area_frac del meta vs Y.mean(); patient_id/vista/lado/forma vs fila del CSV.

Ajuste solo con train; AUC en val y test (diagnóstico, sin selección entre opciones).
No usa la U-Net. Necesita el dataset (para releer el CSV original).

Salida: ``outputs/runs/<name>/shape_diagnostics/``.
Uso:    python scripts/shape_diagnostics.py --config configs/main.yaml
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402
import pandas as pd              # noqa: E402

from src import shape as shp                       # noqa: E402
from src.cli import base_parser, config_from_args  # noqa: E402
from src.data_index import load_case_csv           # noqa: E402
from src.pipeline import prepare_data              # noqa: E402
from src.preprocess import load_split              # noqa: E402
from src.viz import show_samples                   # noqa: E402

SPLITS = ("train", "val", "test")


def fit_eval(name, X, y, seed):
    """Ajusta en train y devuelve métricas en val y test."""
    clf = shp.make_classifier(name, seed).fit(X["train"], y["train"])
    return {s: shp.binary_scores(y[s], clf.predict_proba(X[s])[:, 1]) for s in ("val", "test")}


def check_pairing(cfg, split: str) -> dict:
    """Comprueba que cada recorte corresponde a su fila del meta y del CSV original."""
    X, Y, meta = load_split(cfg.processed_dir, f"{split}_{cfg.cache_tag}")
    area_err = np.abs(meta["lesion_area_frac"].values - Y.reshape(len(Y), -1).mean(axis=1))
    orig = load_case_csv(cfg.csv_dir, split).loc[meta["row_idx"].values]
    norm = lambda s: s.fillna("").astype(str).str.strip().values  # noqa: E731
    out = {"n": int(len(meta)), "area_frac_max_abs_err": float(area_err.max()),
           "row_idx_unicos": int(meta["row_idx"].nunique()),
           "X_Y_meta_misma_longitud": bool(len(X) == len(Y) == len(meta))}
    for col_meta, col_csv in (("patient_id", "patient_id"), ("view", "image view"),
                              ("side", "left or right breast"), ("mass_shape_raw", "mass shape")):
        out[f"{col_meta}_coinciden"] = int((norm(meta[col_meta]) == norm(orig[col_csv])).sum())
    return out


def main(argv=None):
    cfg = config_from_args(base_parser(__doc__.split("\n")[0]).parse_args(argv))
    out = cfg.run_dir / "shape_diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    data, _ = prepare_data(cfg)
    M = {k: data[k].meta for k in SPLITS}
    rows = []

    # 1) tamaño -> patología
    y = {k: (M[k]["pathology"] == "MALIGNANT").values.astype(int) for k in SPLITS}
    for label, cols in (("side_px", ["side_px"]), ("lesion_area_frac", ["lesion_area_frac"]),
                        ("side_px + lesion_area_frac", ["side_px", "lesion_area_frac"])):
        X = {k: np.log(M[k][cols].values.astype(float)) for k in SPLITS}   # tamaños en escala log
        for s, r in fit_eval(shp.LOGREG, X, y, cfg.seed).items():
            rows.append({"prueba": "1) tamaño -> patología", "entrada": label, "clasificador": shp.LOGREG,
                         "conjunto": s, "n": len(y[s]), **r})

    # 2) ROUND vs IRREGULAR
    sel = {k: M[k]["shape_label"].isin(["ROUND", "IRREGULAR"]).values for k in SPLITS}
    F = {k: shp.features_dataframe(data[k].Y[sel[k]], 0) for k in SPLITS}
    ok = {k: F[k].notna().all(axis=1).values for k in SPLITS}
    X = {k: F[k].loc[ok[k], shp.FEATURE_NAMES].values for k in SPLITS}
    y2 = {k: (M[k].loc[sel[k], "shape_label"].values[ok[k]] == "IRREGULAR").astype(int) for k in SPLITS}
    for name in (shp.LOGREG, shp.FOREST):
        for s, r in fit_eval(name, X, y2, cfg.seed).items():
            rows.append({"prueba": "2) ROUND vs IRREGULAR", "entrada": "8 descriptores (σ=0)",
                         "clasificador": name, "conjunto": s, "n": len(y2[s]), **r})
    medians = (F["train"].assign(forma=M["train"].loc[sel["train"], "shape_label"].values)
               .groupby("forma")[shp.FEATURE_NAMES].median().round(3))
    res = pd.DataFrame(rows)
    res.round(4).to_csv(out / "diagnostico_descriptores.csv", index=False)

    # 3) comprobación visual
    rng = np.random.default_rng(cfg.seed)
    mtr = M["train"]
    idx = np.concatenate([rng.choice(np.flatnonzero(mtr["shape_label"] == c), 8, replace=False)
                          for c in ("ROUND", "IRREGULAR")])
    titles = {}
    for i in idx:
        f = shp.shape_features(data["train"].Y[i])
        titles[i] = (f"{mtr.loc[i, 'shape_label']}  circ={f['circularity']:.2f}\n"
                     f"sol={f['solidity']:.2f}  lóbulos={f['n_lobes']}")
    show_samples(data["train"].X, data["train"].Y, idx=idx, ncols=4, titles=titles,
                 path=out / "diagnostico_mascaras.png")
    plt.close("all")

    # 4) emparejamiento
    pairing = {s: check_pairing(cfg, s) for s in ("train", "test")}
    with open(out / "diagnostico_descriptores.json", "w", encoding="utf-8") as fh:
        json.dump({"emparejamiento": pairing, "medianas_train_ROUND_vs_IRREGULAR": medians.to_dict()},
                  fh, indent=2, ensure_ascii=False)
    print(res.round(3).to_string())
    print("\nMedianas de los descriptores en train:\n", medians.to_string())
    print("\nEmparejamiento:", json.dumps(pairing, indent=2))


if __name__ == "__main__":
    main()
