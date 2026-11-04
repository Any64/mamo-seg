"""Ablación de la clasificación de forma BI-RADS (4 clases): suavizado del contorno x clasificador.

Pregunta: ¿la forma sale al nivel del azar por el ruido del contorno anotado o por
el clasificador? Se evalúan sigma en {0, 2, 4} px (``shape.smooth_mask``) y tres
clasificadores: árbol de profundidad 3 (interpretable), regresión logística
estandarizada y random forest (solo cota superior), más la clase mayoritaria.

* Ajuste solo con las máscaras ANOTADAS de train.
* Elección (sigma, clasificador interpretable) solo con validación, máscara anotada.
* Test se evalúa una única vez, para la combinación elegida (+ clase mayoritaria).
* Usa los pesos de ``outputs/runs/<name>/unet_best.pt`` (ejecutar antes run_experiment.py).

Salida: ``outputs/runs/<name>/shape_ablation/``.
Uso:    python scripts/shape_ablation.py --config configs/main.yaml
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
from sklearn.inspection import permutation_importance  # noqa: E402

from src import shape as shp                                        # noqa: E402
from src.cli import base_parser, config_from_args                   # noqa: E402
from src.pipeline import load_unet, predict_masks, prepare_data     # noqa: E402
from src.train import set_seed                                      # noqa: E402
from src.viz import plot_confusion, plot_grouped_bars               # noqa: E402

SIGMAS = [0, 2, 4]
CLASSIFIERS = [shp.TREE, shp.LOGREG, shp.FOREST]
INTERPRETABLE = [shp.TREE, shp.LOGREG]   # candidatos a la combinación elegida; el RF es cota superior
SIGNAL_THR = 0.35                        # exactitud balanceada en val (máscara anotada) para hablar de señal
MASKS = ("máscara anotada", "máscara U-Net")


def main(argv=None):
    p = base_parser(__doc__.split("\n")[0])
    p.add_argument("--weights", help="pesos de la U-Net (por defecto outputs/runs/<name>/unet_best.pt)")
    a = p.parse_args(argv)
    cfg = config_from_args(a)
    set_seed(cfg.seed)
    out = cfg.run_dir / "shape_ablation"
    out.mkdir(parents=True, exist_ok=True)

    data, _ = prepare_data(cfg)
    model = load_unet(cfg, a.weights or cfg.run_dir / "unet_best.pt")
    preds = {s: predict_masks(model, data[s].X, cfg.threshold) for s in ("val", "test")}
    del model

    keep = {s: (data[s].meta["shape_label"] != "").values for s in data}
    y = {s: data[s].meta.loc[keep[s], "shape_label"].values for s in data}
    masks = {s: {MASKS[0]: data[s].Y[keep[s]]} for s in data}
    for s in ("val", "test"):
        masks[s][MASKS[1]] = preds[s][keep[s]]

    # ---- ajuste en train y evaluación en val --------------------------------
    feats, models, rows = {}, {}, []
    for sg in SIGMAS:
        feats[sg] = {("train", MASKS[0]): shp.features_dataframe(masks["train"][MASKS[0]], sg)}
        for m in MASKS:
            feats[sg][("val", m)] = shp.features_dataframe(masks["val"][m], sg)
        for name in CLASSIFIERS + [shp.MAJORITY]:
            clf = shp.fit_classifier(name, feats[sg][("train", MASKS[0])], y["train"], cfg.seed)
            models[(sg, name)] = clf
            for m in MASKS:
                rows.append(row(sg, name, "val", m,
                                shp.evaluate_shape(y["val"], shp.predict_shape(clf, feats[sg][("val", m)]))))
        print(f"sigma={sg}: listo")
    res = pd.DataFrame(rows)

    # ---- elección solo con validación ---------------------------------------
    cand = res[(res["máscara"] == MASKS[0]) & res["clasificador"].isin(INTERPRETABLE)]
    best = cand.sort_values(["exactitud balanceada", "F1 macro"], ascending=False).iloc[0]
    s_best, c_best = int(best["sigma"]), best["clasificador"]
    print(f"Elegido en validación: sigma={s_best}, {c_best} "
          f"(exactitud balanceada = {best['exactitud balanceada']:.3f})")

    # ---- test: una sola vez --------------------------------------------------
    test_res = {}
    for m in MASKS:
        F = shp.features_dataframe(masks["test"][m], s_best)
        for name in (c_best, shp.MAJORITY):
            r = shp.evaluate_shape(y["test"], shp.predict_shape(models[(s_best, name)], F))
            test_res[(name, m)] = r
            res.loc[len(res)] = row(s_best, name, "test", m, r)
    res["elegida"] = (res["sigma"] == s_best) & (res["clasificador"] == c_best)
    res.round(4).to_csv(out / "forma_experimentos.csv", index=False)

    # ---- figuras ---------------------------------------------------------------
    for m, suf in zip(MASKS, ("anotada", "unet")):
        plot_confusion(test_res[(c_best, m)]["confusion"], shp.SHAPE_CLASSES, shp.SHAPE_CLASSES + ["NONE"],
                       f"Forma en test ({c_best}, sigma={s_best}), a partir de la {m}",
                       out / f"confusion_forma_mejor_{suf}.png")
    val = res[res["conjunto"] == "val"].assign(
        panel=lambda d: "Validación, " + d["máscara"], sigma_lbl=lambda d: "σ = " + d["sigma"].astype(str) + " px")
    majority = val[val["clasificador"] == shp.MAJORITY].groupby("panel")["exactitud balanceada"].first()
    plot_grouped_bars(val[val["clasificador"] != shp.MAJORITY], "sigma_lbl", "clasificador",
                      "exactitud balanceada", "panel", [f"σ = {sg} px" for sg in SIGMAS], CLASSIFIERS,
                      ref_lines={"clase mayoritaria": majority.to_dict(),
                                 f"umbral de señal ({SIGNAL_THR})": {k: SIGNAL_THR for k in majority.index}},
                      ylabel="exactitud balanceada",
                      title="Forma BI-RADS: suavizado del contorno vs. clasificador",
                      path=out / "forma_exactitud_balanceada.png")
    plt.close("all")

    # ---- ¿qué descriptores aportan? (solo si hay señal) ----------------------
    va_ann = res[(res["conjunto"] == "val") & (res["máscara"] == MASKS[0])]
    importance = {}
    if va_ann["exactitud balanceada"].max() > SIGNAL_THR:
        top = va_ann.sort_values("exactitud balanceada", ascending=False).iloc[0]
        sg, name = int(top["sigma"]), top["clasificador"]
        F = feats[sg][("val", MASKS[0])]
        ok = F.notna().all(axis=1).values
        pi = permutation_importance(models[(sg, name)], F.loc[ok, shp.FEATURE_NAMES].values, y["val"][ok],
                                    scoring="balanced_accuracy", n_repeats=30, random_state=cfg.seed)
        importance = {"sigma": sg, "clasificador": name, "importancia_permutacion_val": dict(
            sorted(zip(shp.FEATURE_NAMES, pi.importances_mean.round(4).tolist()), key=lambda kv: -kv[1]))}
    else:
        print(f"Sin señal: ninguna combinación supera {SIGNAL_THR} de exactitud balanceada en val.")
    with open(out / "resumen.json", "w", encoding="utf-8") as f:
        json.dump({"eleccion": {"sigma": s_best, "clasificador": c_best},
                   "descriptores_que_aportan": importance}, f, indent=2, ensure_ascii=False)
    print(res.round(3).to_string())


def row(sigma, name, split, mask, r) -> dict:
    return {"sigma": sigma, "clasificador": name, "conjunto": split, "máscara": mask, "n": r["n"],
            "sin descriptores": r["n_invalid"], "exactitud": r["accuracy"],
            "exactitud balanceada": r["balanced_accuracy"], "F1 macro": r["macro_f1"]}


if __name__ == "__main__":
    main()
