"""Control de diagnóstico: ¿los descriptores geométricos contienen alguna señal?

Si la forma BI-RADS (4 clases) sale al nivel del azar, puede ser porque los
descriptores no capturan nada o porque las etiquetas de forma son poco
reproducibles. Se prueban tareas más simples con los MISMOS descriptores:

A) descriptores -> patología (BENIGN vs MALIGNANT), todas las lesiones.
B) descriptores -> forma agrupada: regular (ROUND, OVAL) vs irregular (LOBULATED, IRREGULAR).
C) patología a partir de: solo descriptores, solo etiqueta de forma (one-hot) y ambos
   (subconjunto con una de las 4 formas básicas, para comparar en igualdad de condiciones).

Reglas: máscaras ANOTADAS; ajuste solo con train; sigma (0 o 4) y clasificador
(regresión logística estandarizada o random forest) elegidos por AUC en validación;
test se evalúa una única vez por tarea, con la combinación elegida.
No usa la U-Net ni modifica nada del diseño.

Ejecutar desde la raíz del repositorio:   python scripts/experimento_control_forma.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import matplotlib                                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402
from sklearn.ensemble import RandomForestClassifier                  # noqa: E402
from sklearn.linear_model import LogisticRegression                  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve  # noqa: E402
from sklearn.pipeline import make_pipeline                           # noqa: E402
from sklearn.preprocessing import StandardScaler                     # noqa: E402

from src import shape as shp                                         # noqa: E402
from src.config import Config                                        # noqa: E402
from src.data_utils import patient_split                             # noqa: E402
from src.preprocess import load_split                                # noqa: E402

SIGMAS = [0, 4]
CLASSIFIERS = ["regresión logística", "random forest"]
REGULAR = {"ROUND", "OVAL"}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]   # ranuras categóricas 1-3 (validadas)


def make_classifier(name: str, seed: int):
    if name == "regresión logística":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(class_weight="balanced", max_iter=5000))
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=5, class_weight="balanced",
                                  n_jobs=-1, random_state=seed)


def scores(y, p) -> dict:
    """AUC-ROC y exactitud balanceada (umbral 0.5 sobre la probabilidad)."""
    return {"AUC": float(roc_auc_score(y, p)),
            "exactitud balanceada": float(balanced_accuracy_score(y, p >= 0.5))}


def run_task(task: str, X: dict, y: dict, options: list, seed: int, rows: list, positive: str):
    """Ajusta cada opción (sigma, clasificador) en train, elige por AUC en val y evalúa test una vez.

    ``X[s][split]`` son las matrices de entrada para la variante ``s`` (sigma o '-');
    ``y[split]`` son etiquetas binarias (1 = ``positive``). Devuelve (fpr, tpr, auc) de test.
    """
    val = []
    for s, name in options:
        clf = make_classifier(name, seed).fit(X[s]["train"], y["train"])
        r = scores(y["val"], clf.predict_proba(X[s]["val"])[:, 1])
        val.append((r["AUC"], s, name, clf))
        rows.append({"tarea": task, "sigma": s, "clasificador": name, "conjunto": "val",
                     "n": len(y["val"]), "positivo": positive, **r})
    # clase mayoritaria: probabilidad constante -> AUC 0.5, exactitud balanceada 0.5
    for split in ("val", "test"):
        maj = float(np.mean(y["train"]) >= 0.5)
        rows.append({"tarea": task, "sigma": "-", "clasificador": "clase mayoritaria", "conjunto": split,
                     "n": len(y[split]), "positivo": positive,
                     **scores(y[split], np.full(len(y[split]), maj))})
    auc_v, s, name, clf = max(val, key=lambda t: t[0])
    p = clf.predict_proba(X[s]["test"])[:, 1]
    rows.append({"tarea": task, "sigma": s, "clasificador": name, "conjunto": "test",
                 "n": len(y["test"]), "positivo": positive, **scores(y["test"], p), "elegida": True})
    print(f"{task}: elegido sigma={s}, {name} (AUC val={auc_v:.3f}) -> AUC test={rows[-1]['AUC']:.3f}")
    fpr, tpr, _ = roc_curve(y["test"], p)
    return fpr, tpr, rows[-1]["AUC"], f"σ={s}, {name}"


def main():
    cfg = Config(out_dir=REPO / "outputs")
    tag = cfg.cache_tag
    _, Yall, mall = load_split(cfg.processed_dir, f"train_{tag}")
    _, Yte, mte = load_split(cfg.processed_dir, f"test_{tag}")
    tr, va = patient_split(mall, cfg.val_frac, cfg.seed)          # misma división que el notebook
    Y = {"train": Yall[tr], "val": Yall[va], "test": Yte}
    M = {"train": mall.iloc[tr].reset_index(drop=True), "val": mall.iloc[va].reset_index(drop=True),
         "test": mte.reset_index(drop=True)}

    # descriptores sobre máscaras anotadas (las filas con NaN se excluyen)
    F = {s: {k: shp.features_dataframe(Y[k], s) for k in Y} for s in SIGMAS}
    ok = {k: np.logical_and.reduce([F[s][k].notna().all(axis=1).values for s in SIGMAS]) for k in Y}
    print("máscaras sin descriptores:", {k: int((~v).sum()) for k, v in ok.items()})

    malignant = {k: (M[k]["pathology"] == "MALIGNANT").values for k in Y}
    has_shape = {k: (M[k]["shape_label"] != "").values for k in Y}
    rows, roc = [], {}
    opts = [(s, c) for s in SIGMAS for c in CLASSIFIERS]

    def feats(sel, s):
        return {k: F[s][k].loc[sel[k], shp.FEATURE_NAMES].values for k in Y}

    # ---- A) descriptores -> patología (todas las lesiones) -------------------
    selA = {k: ok[k] for k in Y}
    roc["A"] = run_task("A: descriptores -> patología", {s: feats(selA, s) for s in SIGMAS},
                        {k: malignant[k][selA[k]].astype(int) for k in Y}, opts, cfg.seed, rows,
                        "MALIGNANT")

    # ---- B) descriptores -> forma regular vs irregular ------------------------
    selB = {k: ok[k] & has_shape[k] for k in Y}
    irregular = {k: (~M[k]["shape_label"].isin(REGULAR)).values for k in Y}
    roc["B"] = run_task("B: descriptores -> forma irregular", {s: feats(selB, s) for s in SIGMAS},
                        {k: irregular[k][selB[k]].astype(int) for k in Y}, opts, cfg.seed, rows,
                        "LOBULATED/IRREGULAR")

    # ---- C) patología desde descriptores / etiqueta de forma / ambos ---------
    selC = selB   # lesiones con una de las 4 formas: mismo subconjunto para las 3 variantes
    yC = {k: malignant[k][selC[k]].astype(int) for k in Y}
    onehot = {k: pd.get_dummies(pd.Categorical(M[k].loc[selC[k], "shape_label"],
                                               categories=shp.SHAPE_CLASSES)).values.astype(float)
              for k in Y}
    roc["C1"] = run_task("C: patología <- solo descriptores", {s: feats(selC, s) for s in SIGMAS},
                         yC, opts, cfg.seed, rows, "MALIGNANT")
    roc["C2"] = run_task("C: patología <- solo etiqueta de forma", {"-": onehot},
                         yC, [("-", c) for c in CLASSIFIERS], cfg.seed, rows, "MALIGNANT")
    both = {s: {k: np.hstack([feats(selC, s)[k], onehot[k]]) for k in Y} for s in SIGMAS}
    roc["C3"] = run_task("C: patología <- descriptores + forma", both, yC, opts, cfg.seed, rows,
                         "MALIGNANT")

    res = pd.DataFrame(rows)
    res["elegida"] = res["elegida"].fillna(False).astype(bool)
    res.round(4).to_csv(cfg.tables_dir / "control_forma.csv", index=False)
    with open(cfg.tables_dir / "control_forma_test.json", "w", encoding="utf-8") as f:
        json.dump(res[res["conjunto"] == "test"].round(4).to_dict("records"), f,
                  indent=2, ensure_ascii=False)
    plot_rocs(roc, cfg.figures_dir / "control_forma_roc.png")
    print("\n", res.round(3).to_string())


def plot_rocs(roc: dict, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    panels = [
        ("Patología (BENIGN vs MALIGNANT), test", [
            ("C1", "solo descriptores"),
            ("C2", "solo etiqueta de forma"),
            ("C3", "descriptores + forma")]),
        ("Forma regular vs irregular, test", [("B", "descriptores")]),
    ]
    for ax, (title, curves) in zip(axes, panels):
        for color, (key, label) in zip(COLORS, curves):
            fpr, tpr, auc, how = roc[key]
            ax.plot(fpr, tpr, color=color, lw=2, label=f"{label} (AUC {auc:.2f}; {how})")
        ax.plot([0, 1], [0, 1], color="#888", lw=1, ls="--", label="azar (AUC 0.50)")
        ax.set_xlabel("1 − especificidad")
        ax.set_ylabel("sensibilidad")
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")
        ax.grid(color="#e5e5e5", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.16), fontsize=7.5, frameon=False)
    fig.suptitle("Control: ¿qué información contienen los descriptores geométricos?", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
