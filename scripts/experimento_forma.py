"""Experimento controlado sobre la clasificación de la forma BI-RADS.

Pregunta: ¿la forma sale al nivel del azar por el ruido del contorno anotado,
por el clasificador, o porque los descriptores no contienen la señal?

* Reutiliza los recortes cacheados (outputs/processed) y los pesos ya entrenados
  (outputs/models/unet_best.pt). No reentrena la U-Net ni cambia el recorte.
* Suavizado del contorno: sigma en {0, 2, 4} px (src/shape.py, ``smooth_sigma``).
* Clasificadores: árbol de profundidad 3 (el del notebook), regresión logística
  estandarizada y random forest (solo como cota superior), más la clase mayoritaria.
* Todo se ajusta con las máscaras ANOTADAS de train. La combinación se elige
  solo con validación; test se evalúa una única vez para la combinación elegida.

Ejecutar desde la raíz del repositorio:   python scripts/experimento_forma.py
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
import torch                                                         # noqa: E402
from sklearn.dummy import DummyClassifier                            # noqa: E402
from sklearn.ensemble import RandomForestClassifier                  # noqa: E402
from sklearn.inspection import permutation_importance                # noqa: E402
from sklearn.linear_model import LogisticRegression                  # noqa: E402
from sklearn.pipeline import make_pipeline                           # noqa: E402
from sklearn.preprocessing import StandardScaler                     # noqa: E402

from src import shape as shp                                         # noqa: E402
from src.config import Config                                        # noqa: E402
from src.data_utils import patient_split                             # noqa: E402
from src.dataset import to_tensors                                   # noqa: E402
from src.metrics import per_sample_metrics                           # noqa: E402
from src.model import UNet                                           # noqa: E402
from src.preprocess import load_split                                # noqa: E402
from src.train import get_device, predict_probs, set_seed            # noqa: E402
from src.viz import plot_confusion                                   # noqa: E402

SIGMAS = [0, 2, 4]
CLASSIFIERS = ["árbol (prof. 3)", "regresión logística", "random forest"]
MAJORITY = "clase mayoritaria"
INTERPRETABLE = CLASSIFIERS[:2]   # candidatos a "combinación elegida"; el RF es solo cota superior
SIGNAL_THR = 0.35                 # exactitud balanceada en val (máscara anotada) para hablar de señal


def make_classifier(name: str, seed: int):
    if name == "árbol (prof. 3)":
        return None  # se ajusta con shp.fit_shape_tree (idéntico al notebook)
    if name == "regresión logística":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(class_weight="balanced", max_iter=5000))
    if name == "random forest":
        return RandomForestClassifier(n_estimators=500, min_samples_leaf=5, class_weight="balanced",
                                      n_jobs=-1, random_state=seed)
    if name == MAJORITY:
        return DummyClassifier(strategy="most_frequent")
    raise ValueError(name)


def fit(name: str, F: pd.DataFrame, y, seed: int):
    if name == "árbol (prof. 3)":
        return shp.fit_shape_tree(F, y, max_depth=3, seed=seed)
    ok = F.notna().all(axis=1).values
    clf = make_classifier(name, seed)
    clf.fit(F.loc[ok, shp.FEATURE_NAMES].values, np.asarray(y)[ok])
    return clf


def main():
    cfg = Config(out_dir=REPO / "outputs", base_channels=16)   # mismos parámetros que el notebook
    set_seed(cfg.seed)
    tag = cfg.cache_tag

    # ---- datos (misma división por paciente que el notebook) ----------------
    Xall, Yall, mall = load_split(cfg.processed_dir, f"train_{tag}")
    Xte, Yte, mte = load_split(cfg.processed_dir, f"test_{tag}")
    tr, va = patient_split(mall, cfg.val_frac, cfg.seed)
    Ytr, mtr = Yall[tr], mall.iloc[tr].reset_index(drop=True)
    Xva, Yva, mva = Xall[va], Yall[va], mall.iloc[va].reset_index(drop=True)
    print(f"train={len(tr)} val={len(va)} test={len(Xte)}")

    # ---- máscaras de la U-Net con los pesos ya entrenados ---------------------
    dev = get_device()
    model = UNet(base=cfg.base_channels).to(dev)
    model.load_state_dict(torch.load(cfg.models_dir / "unet_best.pt", map_location=dev))
    Pva = predict_probs(model, to_tensors(Xva, Yva, dev)[0]) > cfg.threshold
    Pte = predict_probs(model, to_tensors(Xte, Yte, dev)[0]) > cfg.threshold
    del model
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    # ---- (7) Dice por recorte en test vs. desplazamiento ---------------------
    dte = per_sample_metrics(Pte, Yte)
    dte.insert(0, "row_idx", mte["row_idx"].values)
    dte["shift_x"], dte["shift_y"] = mte["shift_x"].values, mte["shift_y"].values
    dte["abs_shift"] = np.hypot(dte["shift_x"], dte["shift_y"])
    dte["tercil_shift"] = pd.qcut(dte["abs_shift"], 3, labels=["bajo", "medio", "alto"])
    dte.to_csv(cfg.tables_dir / "dice_por_muestra.csv", index=False)
    by_shift = dte.groupby("tercil_shift", observed=True).agg(
        n=("dice", "size"), shift_min=("abs_shift", "min"), shift_max=("abs_shift", "max"),
        dice_media=("dice", "mean"), dice_mediana=("dice", "median"),
        frac_dice_lt_0_1=("dice", lambda d: float((d < 0.1).mean())))
    by_shift.to_csv(cfg.tables_dir / "dice_por_desplazamiento.csv")
    frac_fail = float((dte["dice"] < 0.1).mean())
    print("\nDice U-Net en test por tercil de |desplazamiento|:\n", by_shift.round(3).to_string())
    print(f"Fracción de recortes de test con Dice < 0.1: {frac_fail:.3f} "
          f"({int((dte['dice'] < 0.1).sum())} de {len(dte)})")

    # ---- descriptores por sigma ----------------------------------------------
    ktr = (mtr["shape_label"] != "").values
    kva = (mva["shape_label"] != "").values
    kte = (mte["shape_label"] != "").values
    ytr, yva, yte = (m.loc[k, "shape_label"].values for m, k in ((mtr, ktr), (mva, kva), (mte, kte)))
    feats = {}
    for s in SIGMAS:
        feats[s] = {
            "train": shp.features_dataframe(Ytr[ktr], s),
            ("val", "máscara anotada"): shp.features_dataframe(Yva[kva], s),
            ("val", "máscara U-Net"): shp.features_dataframe(Pva[kva], s),
        }
        print(f"descriptores sigma={s}: listos")

    # ---- (3,4) ajuste en train, evaluación en val ----------------------------
    rows, models = [], {}
    for s in SIGMAS:
        for name in CLASSIFIERS + [MAJORITY]:
            clf = fit(name, feats[s]["train"], ytr, cfg.seed)
            models[(s, name)] = clf
            for src_name in ("máscara anotada", "máscara U-Net"):
                pred = shp.predict_shape(clf, feats[s][("val", src_name)])
                r = shp.evaluate_shape(yva, pred)
                rows.append({"sigma": s, "clasificador": name, "conjunto": "val", "máscara": src_name,
                             "n": r["n"], "sin descriptores": r["n_invalid"],
                             "exactitud": r["accuracy"], "exactitud balanceada": r["balanced_accuracy"],
                             "F1 macro": r["macro_f1"]})
    res = pd.DataFrame(rows)

    # ---- elección SOLO con validación (máscara anotada, clasificadores interpretables)
    cand = res[(res["conjunto"] == "val") & (res["máscara"] == "máscara anotada")
               & res["clasificador"].isin(INTERPRETABLE)]
    best = cand.sort_values(["exactitud balanceada", "F1 macro"], ascending=False).iloc[0]
    s_best, c_best = int(best["sigma"]), best["clasificador"]
    print(f"\nCombinación elegida en validación: sigma={s_best}, {c_best} "
          f"(exactitud balanceada val = {best['exactitud balanceada']:.3f})")

    # ---- test: una sola vez, combinación elegida (+ clase mayoritaria como referencia)
    F_te = {"máscara anotada": shp.features_dataframe(Yte[kte], s_best),
            "máscara U-Net": shp.features_dataframe(Pte[kte], s_best)}
    test_res = {}
    for name in (c_best, MAJORITY):
        for src_name, F in F_te.items():
            r = shp.evaluate_shape(yte, shp.predict_shape(models[(s_best, name)], F))
            test_res[(name, src_name)] = r
            res.loc[len(res)] = {"sigma": s_best, "clasificador": name, "conjunto": "test",
                                 "máscara": src_name, "n": r["n"], "sin descriptores": r["n_invalid"],
                                 "exactitud": r["accuracy"],
                                 "exactitud balanceada": r["balanced_accuracy"], "F1 macro": r["macro_f1"]}
    res["elegida"] = (res["sigma"] == s_best) & (res["clasificador"] == c_best)
    res.round(4).to_csv(cfg.tables_dir / "forma_experimentos.csv", index=False)

    # ---- (5) figuras ----------------------------------------------------------
    for src_name, suf in (("máscara anotada", "anotada"), ("máscara U-Net", "unet")):
        plot_confusion(test_res[(c_best, src_name)]["confusion"], shp.SHAPE_CLASSES,
                       shp.SHAPE_CLASSES + ["NONE"],
                       f"Forma en test ({c_best}, sigma={s_best}), a partir de la {src_name}",
                       cfg.figures_dir / f"confusion_forma_mejor_{suf}.png")
        plt.close("all")
    # el árbol original (sigma=0) en test, para regenerar las figuras del notebook con el título corregido
    for src_name, suf in (("máscara anotada", "anotada"), ("máscara U-Net", "unet")):
        M = Yte if suf == "anotada" else Pte
        r = shp.evaluate_shape(yte, shp.predict_shape(models[(0, "árbol (prof. 3)")],
                                                      shp.features_dataframe(M[kte], 0)))
        plot_confusion(r["confusion"], shp.SHAPE_CLASSES, shp.SHAPE_CLASSES + ["NONE"],
                       f"Forma BI-RADS en test, a partir de la {src_name}",
                       cfg.figures_dir / f"confusion_forma_{suf}.png")
        plt.close("all")
    plot_balanced_accuracy(res[res["conjunto"] == "val"], cfg.figures_dir / "forma_exactitud_balanceada.png")

    # ---- (6) ¿qué descriptores aportan? ---------------------------------------
    va_ann = res[(res["conjunto"] == "val") & (res["máscara"] == "máscara anotada")]
    importance = {}
    if va_ann["exactitud balanceada"].max() > SIGNAL_THR:
        top = va_ann.sort_values("exactitud balanceada", ascending=False).iloc[0]
        s_top, c_top = int(top["sigma"]), top["clasificador"]
        F = feats[s_top][("val", "máscara anotada")]
        ok = F.notna().all(axis=1).values
        pi = permutation_importance(models[(s_top, c_top)], F.loc[ok, shp.FEATURE_NAMES].values,
                                    yva[ok], scoring="balanced_accuracy", n_repeats=30,
                                    random_state=cfg.seed)
        importance = {"sigma": s_top, "clasificador": c_top,
                      "importancia_permutacion_val": dict(sorted(
                          zip(shp.FEATURE_NAMES, pi.importances_mean.round(4).tolist()),
                          key=lambda kv: -kv[1]))}
        print("\nImportancia por permutación (val, máscara anotada):", importance)
    else:
        print(f"\nSin señal: ninguna combinación supera {SIGNAL_THR} de exactitud balanceada en val.")

    summary = {"eleccion": {"sigma": s_best, "clasificador": c_best},
               "frac_test_dice_lt_0_1": round(frac_fail, 4),
               "dice_test_por_tercil_shift": by_shift.round(4).reset_index().to_dict("records"),
               "descriptores_que_aportan": importance}
    with open(cfg.tables_dir / "forma_experimentos_resumen.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print("\n", res.round(3).to_string())


def plot_balanced_accuracy(val: pd.DataFrame, path: Path):
    """Exactitud balanceada en validación por sigma y clasificador (un panel por máscara)."""
    colors = dict(zip(CLASSIFIERS, ["#2a78d6", "#eb6834", "#1baf7a"]))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    w = 0.26
    for ax, src_name in zip(axes, ("máscara anotada", "máscara U-Net")):
        d = val[val["máscara"] == src_name]
        for j, name in enumerate(CLASSIFIERS):
            v = [d[(d["sigma"] == s) & (d["clasificador"] == name)]["exactitud balanceada"].iloc[0]
                 for s in SIGMAS]
            x = np.arange(len(SIGMAS)) + (j - 1) * w
            ax.bar(x, v, width=w - 0.03, color=colors[name], label=name)
            for xi, vi in zip(x, v):
                ax.text(xi, vi + 0.006, f"{vi:.2f}", ha="center", va="bottom", fontsize=7, color="#444")
        maj = d[d["clasificador"] == MAJORITY]["exactitud balanceada"].iloc[0]
        ax.axhline(maj, color="#888", lw=1.2, ls="--", label="clase mayoritaria")
        ax.axhline(SIGNAL_THR, color="#444", lw=0.8, ls=":", label=f"umbral de señal ({SIGNAL_THR})")
        ax.set_xticks(range(len(SIGMAS)), [f"σ = {s} px" for s in SIGMAS])
        ax.set_title(f"Validación, {src_name}", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e5e5e5", lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("exactitud balanceada")
    axes[0].set_ylim(0, 0.6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=8)
    fig.suptitle("Forma BI-RADS: suavizado del contorno vs. clasificador", fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
