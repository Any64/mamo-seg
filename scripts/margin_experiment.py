"""Margen BI-RADS desde la geometría de la máscara anotada (mismo protocolo que shape_control.py).

Agrupación propuesta (``mass margins`` de CBIS-DDSM):
  - M1 circunscrito vs no circunscrito: CIRCUMSCRIBED (solo) frente a cualquier
    combinación de ILL_DEFINED, MICROLOBULATED o SPICULATED sin CIRCUMSCRIBED.
    Se excluyen OBSCURED solo (el borde está oculto por tejido: no es una propiedad
    del contorno), las combinaciones que incluyen CIRCUMSCRIBED y otro margen
    (contradictorias) y los vacíos.
  - M2 espiculado vs no espiculado: contiene SPICULATED frente al resto (sin vacíos).
Patología (subconjunto con margen anotado):
  - P1 solo descriptores, P2 solo etiqueta de margen (multi-hot de los 5 márgenes),
    P3 descriptores + margen.

Ajuste solo en train; σ ∈ {0, 4} y clasificador (regresión logística o random forest)
elegidos por AUC en validación; test una única vez por tarea. No usa la U-Net.

Salida: results/margen_conteos.csv, results/margen_experimento.csv, results/figures/margen_roc.png
Uso:    python scripts/margin_experiment.py
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402
import pandas as pd              # noqa: E402

from shape_control import CLASSIFIERS, SIGMAS, run_task  # noqa: E402
from src import shape as shp                             # noqa: E402
from src.config import load_config                       # noqa: E402
from src.pipeline import prepare_data                    # noqa: E402
from src.viz import plot_roc_panels                      # noqa: E402

MARGINS = ["CIRCUMSCRIBED", "OBSCURED", "ILL_DEFINED", "MICROLOBULATED", "SPICULATED"]
NOT_CIRC = {"ILL_DEFINED", "MICROLOBULATED", "SPICULATED"}
SPLITS = ("train", "val", "test")


def parts(margin: str) -> set:
    return set(margin.split("-")) if margin else set()


def group_circumscribed(margin: str):
    """1 = no circunscrito, 0 = circunscrito, None = excluido (ver docstring del módulo)."""
    p = parts(margin)
    if p == {"CIRCUMSCRIBED"}:
        return 0
    if p and "CIRCUMSCRIBED" not in p and p & NOT_CIRC:
        return 1
    return None


def group_spiculated(margin: str):
    p = parts(margin)
    return None if not p else int("SPICULATED" in p)


def counts_table(M: dict) -> pd.DataFrame:
    rows = []
    for k in SPLITS:
        for margin, n in M[k]["mass_margins"].fillna("").value_counts().items():
            rows.append({"mass_margins": margin or "(vacío)", "conjunto": k, "n": n})
    df = pd.DataFrame(rows).pivot_table(index="mass_margins", columns="conjunto", values="n", fill_value=0)
    df = df[list(SPLITS)].astype(int)
    df["total"] = df.sum(axis=1)
    df["M1_circunscrito_vs_no"] = [
        {0: "circunscrito", 1: "no circunscrito", None: "excluido"}[group_circumscribed("" if m == "(vacío)" else m)]
        for m in df.index]
    df["M2_espiculado"] = [
        {0: "no espiculado", 1: "espiculado", None: "excluido"}[group_spiculated("" if m == "(vacío)" else m)]
        for m in df.index]
    return df.sort_values("total", ascending=False).reset_index()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default=str(REPO / "configs" / "main.yaml"))
    p.add_argument("--data-dir")
    p.add_argument("--results-dir", default=str(REPO / "results"))
    a = p.parse_args(argv)
    res = Path(a.results_dir)
    (res / "figures").mkdir(parents=True, exist_ok=True)
    cfg = load_config(a.config, archive_dir=Path(a.data_dir).resolve() if a.data_dir else None)
    data, _ = prepare_data(cfg, progress=False, include_test=True)
    M = {k: data[k].meta.assign(mass_margins=data[k].meta["mass_margins"].fillna("")) for k in SPLITS}

    counts = counts_table(M)
    counts.to_csv(res / "margen_conteos.csv", index=False)
    print(counts.to_string(index=False), "\n")

    F = {sg: {k: shp.features_dataframe(data[k].Y, sg) for k in SPLITS} for sg in SIGMAS}
    ok = {k: np.logical_and.reduce([F[sg][k].notna().all(axis=1).values for sg in SIGMAS]) for k in SPLITS}
    opts = [(sg, c) for sg in SIGMAS for c in CLASSIFIERS]
    rows, roc = [], {}

    def feats(sel, sg):
        return {k: F[sg][k].loc[sel[k], shp.FEATURE_NAMES].values for k in SPLITS}

    for key, fn, task, positive in (
            ("M1", group_circumscribed, "M1: descriptores -> margen no circunscrito", "no circunscrito"),
            ("M2", group_spiculated, "M2: descriptores -> margen espiculado", "SPICULATED")):
        g = {k: M[k]["mass_margins"].map(fn) for k in SPLITS}
        sel = {k: ok[k] & g[k].notna().values for k in SPLITS}
        y = {k: g[k][sel[k]].astype(int).values for k in SPLITS}
        roc[key] = run_task(task, {sg: feats(sel, sg) for sg in SIGMAS}, y, opts, cfg.seed, rows, positive)

    has_margin = {k: ok[k] & (M[k]["mass_margins"] != "").values for k in SPLITS}
    yP = {k: (M[k].loc[has_margin[k], "pathology"] == "MALIGNANT").astype(int).values for k in SPLITS}
    multihot = {k: np.array([[int(m in parts(v)) for m in MARGINS]
                             for v in M[k].loc[has_margin[k], "mass_margins"]], dtype=float) for k in SPLITS}
    roc["P1"] = run_task("P: patología <- solo descriptores", {sg: feats(has_margin, sg) for sg in SIGMAS},
                         yP, opts, cfg.seed, rows, "MALIGNANT")
    roc["P2"] = run_task("P: patología <- solo etiqueta de margen", {"-": multihot}, yP,
                         [("-", c) for c in CLASSIFIERS], cfg.seed, rows, "MALIGNANT")
    both = {sg: {k: np.hstack([feats(has_margin, sg)[k], multihot[k]]) for k in SPLITS} for sg in SIGMAS}
    roc["P3"] = run_task("P: patología <- descriptores + margen", both, yP, opts, cfg.seed, rows, "MALIGNANT")

    out = pd.DataFrame(rows)
    out["elegida"] = out["elegida"].astype("boolean").fillna(False).astype(bool)
    out.round(4).to_csv(res / "margen_experimento.csv", index=False)

    curve = lambda key, label: (f"{label}; {roc[key][3]}", *roc[key][:3])  # noqa: E731
    plot_roc_panels([
        ("Margen desde descriptores, test", [curve("M1", "no circunscrito"), curve("M2", "espiculado")]),
        ("Patología (BENIGN vs MALIGNANT), test",
         [curve("P1", "solo descriptores"), curve("P2", "solo etiqueta de margen"),
          curve("P3", "descriptores + margen")]),
    ], "Margen BI-RADS: ¿lo recupera la geometría de la máscara?", res / "figures" / "margen_roc.png")
    plt.close("all")
    print(out[out["conjunto"] != "train"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
