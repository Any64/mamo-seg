"""Control de diagnóstico: ¿los descriptores geométricos contienen alguna señal?

Con los MISMOS descriptores de forma (máscaras anotadas, sigma 0 o 4):
  A) descriptores -> patología (BENIGN vs MALIGNANT), todas las lesiones.
  B) descriptores -> forma agrupada: regular (ROUND, OVAL) vs irregular (LOBULATED, IRREGULAR).
  C) patología a partir de: solo descriptores, solo etiqueta de forma (one-hot) y ambos
     (subconjunto con una de las 4 formas, igual para las tres variantes).

Ajuste solo con train; sigma y clasificador (regresión logística o random forest)
elegidos por AUC en validación; test se evalúa una única vez por tarea.
No usa la U-Net.

Salida: ``outputs/runs/<name>/shape_control/``.
Uso:    python scripts/shape_control.py --config configs/main.yaml
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402
import pandas as pd              # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from src import shape as shp                       # noqa: E402
from src.cli import base_parser, config_from_args  # noqa: E402
from src.pipeline import prepare_data              # noqa: E402
from src.viz import plot_roc_panels                # noqa: E402

SIGMAS = [0, 4]
CLASSIFIERS = [shp.LOGREG, shp.FOREST]
REGULAR = {"ROUND", "OVAL"}
SPLITS = ("train", "val", "test")


def run_task(task, X, y, options, seed, rows, positive):
    """Ajusta cada opción (variante, clasificador) en train, elige por AUC en val y evalúa test una vez.

    ``X[variante][split]`` son matrices de entrada; ``y[split]`` etiquetas binarias.
    Devuelve (etiqueta, fpr, tpr, auc) de test para la curva ROC.
    """
    val = []
    for v, name in options:
        clf = shp.make_classifier(name, seed).fit(X[v]["train"], y["train"])
        r = shp.binary_scores(y["val"], clf.predict_proba(X[v]["val"])[:, 1])
        val.append((r["AUC"], v, name, clf))
        rows.append({"tarea": task, "sigma": v, "clasificador": name, "conjunto": "val",
                     "n": len(y["val"]), "positivo": positive, **r})
    for split in ("val", "test"):   # clase mayoritaria: probabilidad constante -> AUC 0.5
        const = np.full(len(y[split]), float(np.mean(y["train"]) >= 0.5))
        rows.append({"tarea": task, "sigma": "-", "clasificador": shp.MAJORITY, "conjunto": split,
                     "n": len(y[split]), "positivo": positive, **shp.binary_scores(y[split], const)})
    auc_val, v, name, clf = max(val, key=lambda t: t[0])
    prob = clf.predict_proba(X[v]["test"])[:, 1]
    rows.append({"tarea": task, "sigma": v, "clasificador": name, "conjunto": "test", "n": len(y["test"]),
                 "positivo": positive, **shp.binary_scores(y["test"], prob), "elegida": True})
    print(f"{task}: elegido sigma={v}, {name} (AUC val={auc_val:.3f}) -> AUC test={rows[-1]['AUC']:.3f}")
    fpr, tpr, _ = roc_curve(y["test"], prob)
    return fpr, tpr, rows[-1]["AUC"], f"σ={v}, {name}"


def main(argv=None):
    cfg = config_from_args(base_parser(__doc__.split("\n")[0]).parse_args(argv))
    out = cfg.run_dir / "shape_control"
    out.mkdir(parents=True, exist_ok=True)
    data, _ = prepare_data(cfg)
    M = {k: data[k].meta for k in SPLITS}

    F = {sg: {k: shp.features_dataframe(data[k].Y, sg) for k in SPLITS} for sg in SIGMAS}
    ok = {k: np.logical_and.reduce([F[sg][k].notna().all(axis=1).values for sg in SIGMAS]) for k in SPLITS}
    malignant = {k: (M[k]["pathology"] == "MALIGNANT").values for k in SPLITS}
    has_shape = {k: (M[k]["shape_label"] != "").values for k in SPLITS}
    opts = [(sg, c) for sg in SIGMAS for c in CLASSIFIERS]
    rows, roc = [], {}

    def feats(sel, sg):
        return {k: F[sg][k].loc[sel[k], shp.FEATURE_NAMES].values for k in SPLITS}

    selA = ok
    roc["A"] = run_task("A: descriptores -> patología", {sg: feats(selA, sg) for sg in SIGMAS},
                        {k: malignant[k][selA[k]].astype(int) for k in SPLITS}, opts, cfg.seed, rows,
                        "MALIGNANT")
    selB = {k: ok[k] & has_shape[k] for k in SPLITS}
    irregular = {k: (~M[k]["shape_label"].isin(REGULAR)).values for k in SPLITS}
    roc["B"] = run_task("B: descriptores -> forma irregular", {sg: feats(selB, sg) for sg in SIGMAS},
                        {k: irregular[k][selB[k]].astype(int) for k in SPLITS}, opts, cfg.seed, rows,
                        "LOBULATED/IRREGULAR")
    yC = {k: malignant[k][selB[k]].astype(int) for k in SPLITS}
    onehot = {k: pd.get_dummies(pd.Categorical(M[k].loc[selB[k], "shape_label"],
                                               categories=shp.SHAPE_CLASSES)).values.astype(float)
              for k in SPLITS}
    roc["C1"] = run_task("C: patología <- solo descriptores", {sg: feats(selB, sg) for sg in SIGMAS},
                         yC, opts, cfg.seed, rows, "MALIGNANT")
    roc["C2"] = run_task("C: patología <- solo etiqueta de forma", {"-": onehot}, yC,
                         [("-", c) for c in CLASSIFIERS], cfg.seed, rows, "MALIGNANT")
    both = {sg: {k: np.hstack([feats(selB, sg)[k], onehot[k]]) for k in SPLITS} for sg in SIGMAS}
    roc["C3"] = run_task("C: patología <- descriptores + forma", both, yC, opts, cfg.seed, rows, "MALIGNANT")

    res = pd.DataFrame(rows)
    res["elegida"] = res["elegida"].astype("boolean").fillna(False).astype(bool)
    res.round(4).to_csv(out / "control_forma.csv", index=False)

    curve = lambda key, label: (f"{label}; {roc[key][3]}", *roc[key][:3])  # noqa: E731
    plot_roc_panels([
        ("Patología (BENIGN vs MALIGNANT), test",
         [curve("C1", "solo descriptores"), curve("C2", "solo etiqueta de forma"),
          curve("C3", "descriptores + forma")]),
        ("Forma regular vs irregular, test", [curve("B", "descriptores")]),
    ], "Control: ¿qué información contienen los descriptores geométricos?", out / "control_forma_roc.png")
    plt.close("all")
    print(res.round(3).to_string())


if __name__ == "__main__":
    main()
