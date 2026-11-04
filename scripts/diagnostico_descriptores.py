"""Diagnóstico final: ¿falta de señal en los descriptores = error del pipeline o limitación de los datos?

1) Tamaño -> patología (regresión logística): side_px, lesion_area_frac y ambos.
2) Extremos de forma: ROUND vs IRREGULAR con los 8 descriptores (sigma=0).
3) Figura de comprobación visual: 8 ROUND y 8 IRREGULAR de train con sus descriptores.
4) Emparejamiento: lesion_area_frac del meta vs Y.mean(); patient_id/view/mass shape vs fila del CSV.

Ajuste solo con train, AUC en val y test. No usa la U-Net ni cambia el diseño.
Ejecutar desde la raíz del repositorio:   python scripts/diagnostico_descriptores.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import matplotlib                                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402

from experimento_control_forma import CLASSIFIERS, make_classifier, scores  # noqa: E402
from src import shape as shp                                         # noqa: E402
from src.config import Config                                        # noqa: E402
from src.data_index import load_case_csv                             # noqa: E402
from src.data_utils import patient_split                             # noqa: E402
from src.preprocess import load_split                                # noqa: E402
from src.viz import show_samples                                     # noqa: E402


def fit_eval(name, X, y, seed):
    """Ajusta en train y devuelve métricas en val y test (diagnóstico: sin selección entre opciones)."""
    clf = make_classifier(name, seed).fit(X["train"], y["train"])
    return {split: scores(y[split], clf.predict_proba(X[split])[:, 1]) for split in ("val", "test")}


def check_pairing(cfg, split, X, Y, meta) -> dict:
    """Comprueba que cada recorte corresponde a su fila del meta y del CSV original."""
    area_err = np.abs(meta["lesion_area_frac"].values - Y.reshape(len(Y), -1).mean(axis=1))
    df = load_case_csv(cfg.csv_dir, split)
    orig = df.loc[meta["row_idx"].values]
    norm = lambda s: s.fillna("").astype(str).str.strip().values  # noqa: E731
    out = {"n": int(len(meta)), "area_frac_max_abs_err": float(area_err.max())}
    for col_meta, col_csv in (("patient_id", "patient_id"), ("view", "image view"),
                              ("side", "left or right breast"), ("mass_shape_raw", "mass shape")):
        out[f"{col_meta}_coinciden"] = int((norm(meta[col_meta]) == norm(orig[col_csv])).sum())
    out["row_idx_unicos"] = int(meta["row_idx"].nunique())
    out["X_e_Y_misma_longitud"] = bool(len(X) == len(Y) == len(meta))
    return out


def main():
    cfg = Config(archive_dir=REPO.parent / "archive", out_dir=REPO / "outputs")
    tag = cfg.cache_tag
    Xall, Yall, mall = load_split(cfg.processed_dir, f"train_{tag}")
    Xte, Yte, mte = load_split(cfg.processed_dir, f"test_{tag}")
    tr, va = patient_split(mall, cfg.val_frac, cfg.seed)
    Y = {"train": Yall[tr], "val": Yall[va], "test": Yte}
    M = {"train": mall.iloc[tr].reset_index(drop=True), "val": mall.iloc[va].reset_index(drop=True),
         "test": mte.reset_index(drop=True)}
    rows = []

    # ---- 1) tamaño -> patología ---------------------------------------------
    y = {k: (M[k]["pathology"] == "MALIGNANT").values.astype(int) for k in M}
    for label, cols in (("side_px", ["side_px"]), ("lesion_area_frac", ["lesion_area_frac"]),
                        ("side_px + lesion_area_frac", ["side_px", "lesion_area_frac"])):
        X = {k: np.log(M[k][cols].values.astype(float)) for k in M}   # tamaños: escala logarítmica
        for split, r in fit_eval("regresión logística", X, y, cfg.seed).items():
            rows.append({"prueba": "1) tamaño -> patología", "entrada": label,
                         "clasificador": "regresión logística", "conjunto": split, "n": len(y[split]), **r})

    # ---- 2) ROUND vs IRREGULAR ------------------------------------------------
    sel = {k: M[k]["shape_label"].isin(["ROUND", "IRREGULAR"]).values for k in M}
    F = {k: shp.features_dataframe(Y[k][sel[k]], 0) for k in M}
    ok = {k: F[k].notna().all(axis=1).values for k in M}
    X = {k: F[k].loc[ok[k], shp.FEATURE_NAMES].values for k in M}
    y2 = {k: (M[k].loc[sel[k], "shape_label"].values[ok[k]] == "IRREGULAR").astype(int) for k in M}
    for name in CLASSIFIERS:
        for split, r in fit_eval(name, X, y2, cfg.seed).items():
            rows.append({"prueba": "2) ROUND vs IRREGULAR", "entrada": "8 descriptores (σ=0)",
                         "clasificador": name, "conjunto": split, "n": len(y2[split]), **r})
    # medianas por clase en train, para leer junto a la figura
    Ftr = F["train"].assign(forma=M["train"].loc[sel["train"], "shape_label"].values)
    medians = Ftr.groupby("forma")[shp.FEATURE_NAMES].median().round(3)

    res = pd.DataFrame(rows)
    res.round(4).to_csv(cfg.tables_dir / "diagnostico_descriptores.csv", index=False)

    # ---- 3) figura de comprobación visual --------------------------------------
    rng = np.random.default_rng(cfg.seed)
    mtr, Xtr, Ytr = M["train"], Xall[tr], Y["train"]
    idx = np.concatenate([rng.choice(np.flatnonzero(mtr["shape_label"] == c), 8, replace=False)
                          for c in ("ROUND", "IRREGULAR")])
    titles = {}
    for i in idx:
        f = shp.shape_features(Ytr[i])
        titles[i] = (f"{mtr.loc[i, 'shape_label']}  circ={f['circularity']:.2f}\n"
                     f"sol={f['solidity']:.2f}  lóbulos={f['n_lobes']}")
    show_samples(Xtr, Ytr, idx=idx, ncols=4, titles=titles,
                 path=cfg.figures_dir / "diagnostico_mascaras.png")
    plt.close("all")

    # ---- 4) emparejamiento -----------------------------------------------------
    pairing = {"train": check_pairing(cfg, "train", Xall, Yall, mall),
               "test": check_pairing(cfg, "test", Xte, Yte, mte)}

    with open(cfg.tables_dir / "diagnostico_descriptores.json", "w", encoding="utf-8") as fh:
        json.dump({"emparejamiento": pairing, "medianas_train_ROUND_vs_IRREGULAR": medians.to_dict()},
                  fh, indent=2, ensure_ascii=False)
    print(res.round(3).to_string())
    print("\nMedianas de los descriptores en train:\n", medians.to_string())
    print("\nEmparejamiento:", json.dumps(pairing, indent=2))


if __name__ == "__main__":
    main()
