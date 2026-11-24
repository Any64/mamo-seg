"""Comparación U-Net / U-Net + posprocesamiento morfológico fijo / Morph-UNet (no entrena).

1) Ubicación del bloque (``--position-runs``): Dice/IoU en validación de cada ubicación.
2) Posprocesamiento fijo (apertura + cierre con disco de radio r y componente conexa
   más grande): r se elige SOLO con el Dice medio de validación de las U-Net.
3) Métricas en val y test (Dice, IoU, precisión, sensibilidad, componentes conexas,
   Hausdorff 95; las predicciones vacías se cuentan aparte): media ± DE entre semillas.
4) Diferencias pareadas por recorte (Dice promediado entre semillas): Wilcoxon e IC 95 %
   bootstrap (por recorte y por paciente). El modelo final se decide con validación.
5) Figuras: elementos estructurantes y p aprendidos; 8 ejemplos de validación donde
   cambia la predicción; comparación de métricas.
6) Forma BI-RADS con las máscaras del mejor modelo (mismo árbol que el experimento principal).

Las máscaras predichas se guardan en ``outputs/runs/<run>/pred_masks.npz`` (caché).
Uso:
    python scripts/compare_morphology.py --unet-runs main final_seed43 final_seed44 \\
        --morph-runs morph_unet_s42 morph_unet_s43 morph_unet_s44 \\
        --position-runs morph_unet_s42 morph_unet_skip_s42
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402
import pandas as pd              # noqa: E402
import yaml                      # noqa: E402

from src import shape as shp                                   # noqa: E402
from src.config import Config, load_config                     # noqa: E402
from src.metrics import extended_metrics                       # noqa: E402
from src.pipeline import load_unet, predict_masks, prepare_data  # noqa: E402
from src.postprocess import postprocess_stack                  # noqa: E402
from src.stats import bootstrap_mean_ci, paired_wilcoxon       # noqa: E402

UNET, POST, MORPH = "U-Net", "U-Net + posproc. fijo", "Morph-UNet"
COLORS = {UNET: "#2a78d6", POST: "#1baf7a", MORPH: "#eb6834"}
SPLITS = ("val", "test")
GT_COLOR = "#f2c200"   # anotación (amarillo) en las figuras de ejemplos
RADII = (0, 1, 2, 3, 4, 6, 8)
METRICS = ["dice", "iou", "precision", "recall", "hd95", "n_componentes"]


def run_config(runs_dir: Path, run: str) -> Config:
    raw = yaml.safe_load((runs_dir / run / "config.yaml").read_text(encoding="utf-8"))
    valid = Config().to_dict()
    return Config(**{k: v for k, v in raw.items() if k in valid})


def run_seed(cfg: Config) -> int:
    return cfg.seed if cfg.train_seed is None else cfg.train_seed


def get_masks(runs_dir: Path, run: str, data: dict) -> dict:
    """Máscaras binarias {split: (N,H,W)} del modelo ``run`` (con caché en pred_masks.npz)."""
    path = runs_dir / run / "pred_masks.npz"
    if path.exists():
        z = np.load(path)
        if all(s in z and len(z[s]) == len(data[s].X) for s in SPLITS):
            return {s: z[s] for s in SPLITS}
    cfg = run_config(runs_dir, run)
    model = load_unet(cfg, runs_dir / run / "unet_best.pt")
    masks = {s: predict_masks(model, data[s].X, cfg.threshold) for s in SPLITS}
    np.savez_compressed(path, **masks)
    return masks


def position_table(runs_dir: Path, runs: list, unet_run: str) -> pd.DataFrame:
    rows = []
    for r in [unet_run] + runs:
        m = json.loads((runs_dir / r / "metrics.json").read_text(encoding="utf-8"))
        cfg = run_config(runs_dir, r)
        seg = m["segmentacion"]["U-Net | val"]
        rows.append({"ejecucion": r, "arquitectura": cfg.arch,
                     "ubicacion": cfg.morph_position if cfg.arch == "morph_unet" else "—",
                     "dice_val": seg["dice"], "iou_val": seg["iou"], "precision_val": seg["precision"],
                     "recall_val": seg["recall"], "epocas": m["epocas_entrenadas"],
                     "mejor_epoca": m["mejor_epoca"], "segundos_por_epoca": m.get("segundos_por_epoca")})
    return pd.DataFrame(rows)


def select_radius(unet_masks: list, data: dict) -> pd.DataFrame:
    rows = []
    for r in RADII:
        dices = [extended_metrics(postprocess_stack(m["val"], r), data["val"].Y)["dice"].mean()
                 for m in unet_masks]
        rows.append({"radio": r, "dice_val_media": float(np.mean(dices)), "dice_val_de": float(np.std(dices, ddof=1)),
                     **{f"dice_val_semilla{i}": d for i, d in enumerate(dices)}})
    # sin posprocesamiento como referencia
    base = [extended_metrics(m["val"], data["val"].Y)["dice"].mean() for m in unet_masks]
    rows.append({"radio": "sin posproc.", "dice_val_media": float(np.mean(base)),
                 "dice_val_de": float(np.std(base, ddof=1)), **{f"dice_val_semilla{i}": d for i, d in enumerate(base)}})
    return pd.DataFrame(rows)


def per_crop(masks_by_method: dict, data: dict) -> pd.DataFrame:
    frames = []
    for (method, seed), masks in masks_by_method.items():
        for s in SPLITS:
            d = extended_metrics(masks[s], data[s].Y)
            meta = data[s].meta[["row_idx", "patient_id", "pathology", "shape_label"]].reset_index(drop=True)
            frames.append(pd.concat([pd.DataFrame({"método": method, "semilla": seed, "conjunto": s},
                                                  index=d.index), meta, d], axis=1))
    return pd.concat(frames, ignore_index=True)


def summarize(pc: pd.DataFrame) -> pd.DataFrame:
    g = pc.groupby(["método", "conjunto", "semilla"])
    per_seed = g.agg(dice=("dice", "mean"), iou=("iou", "mean"), precision=("precision", "mean"),
                     recall=("recall", "mean"), hd95=("hd95", "mean"), hd95_mediana=("hd95", "median"),
                     n_componentes=("n_componentes", "mean"),
                     frac_mas_de_1_componente=("n_componentes", lambda v: float((v > 1).mean())),
                     n_vacias=("vacia", "sum"), n=("dice", "size")).reset_index()
    rows = []
    for (m, s), d in per_seed.groupby(["método", "conjunto"], sort=False):
        row = {"método": m, "conjunto": s, "n_recortes": int(d["n"].iloc[0]), "n_semillas": len(d)}
        for c in ["dice", "iou", "precision", "recall", "hd95", "hd95_mediana", "n_componentes",
                  "frac_mas_de_1_componente", "n_vacias"]:
            row[f"{c}_media"] = float(d[c].mean())
            row[f"{c}_de"] = float(d[c].std(ddof=1)) if len(d) > 1 else float("nan")
        rows.append(row)
    order = {UNET: 0, POST: 1, MORPH: 2}
    out = pd.DataFrame(rows)
    return out.sort_values(["conjunto", "método"], key=lambda c: c.map(order) if c.name == "método"
                           else c.map({"val": 0, "test": 1})).reset_index(drop=True), per_seed


def paired_tests(pc: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    """Diferencias por recorte del Dice y del HD95 promediados entre semillas."""
    rows = []
    for s in SPLITS:
        d = pc[pc["conjunto"] == s]
        avg = d.groupby(["método", "row_idx"]).agg(dice=("dice", "mean"), hd95=("hd95", "mean"),
                                                   patient_id=("patient_id", "first")).reset_index()
        piv = {m: avg[avg["método"] == m].set_index("row_idx").sort_index() for m in (UNET, POST, MORPH)}
        for a, b in ((MORPH, UNET), (POST, UNET), (MORPH, POST)):
            for metric in ("dice", "hd95"):
                va, vb = piv[a][metric], piv[b][metric]
                ok = va.notna() & vb.notna()
                diff = (va - vb)[ok]
                groups = piv[a].loc[ok, "patient_id"].values
                w = paired_wilcoxon(va[ok].values, vb[ok].values) if (diff != 0).any() else {}
                ci_c = bootstrap_mean_ci(diff.values, n_boot, seed=seed)
                ci_p = bootstrap_mean_ci(diff.values, n_boot, seed=seed, groups=groups)
                rows.append({"conjunto": s, "comparación": f"{a} − {b}", "métrica": metric, "n": int(ok.sum()),
                             "diferencia_media": ci_c["media"], "ic95_inf_recortes": ci_c["ic_inf"],
                             "ic95_sup_recortes": ci_c["ic_sup"], "ic95_inf_pacientes": ci_p["ic_inf"],
                             "ic95_sup_pacientes": ci_p["ic_sup"],
                             "mediana_diferencia": float(np.median(diff)) if len(diff) else float("nan"),
                             "wilcoxon_p": w.get("p_valor", float("nan")),
                             "frac_a_mejor": float((diff > 0).mean() if metric == "dice" else (diff < 0).mean()),
                             "frac_iguales": float((diff == 0).mean())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- figuras
def plot_structuring_elements(models: dict, path: Path) -> pd.DataFrame:
    """Elementos estructurantes w (normalizados al máximo) y p aprendidos, por semilla."""
    rows = []
    for seed, model in models.items():
        for layer_name, layer in model.morph.layers().items():
            w = layer.weight.cpu().numpy()
            for c in range(w.shape[0]):
                rows.append({"semilla": seed, "capa": layer_name, "canal": c,
                             "p": float(layer.p[c].detach().cpu()),
                             "w_min_rel": float(w[c].min() / w[c].max()), "w": w[c] / w[c].max()})
    df = pd.DataFrame(rows)
    seed0 = sorted(models)[0]
    d0 = df[df["semilla"] == seed0]
    layers = list(dict.fromkeys(d0["capa"]))
    n_ch = int(d0["canal"].max()) + 1
    fig, axes = plt.subplots(n_ch, len(layers) + 1, figsize=(2.3 * (len(layers) + 1) + 1.5, 2.3 * n_ch),
                             gridspec_kw={"width_ratios": [1] * len(layers) + [1.6]})
    for j, layer_name in enumerate(layers):
        for c in range(n_ch):
            r = d0[(d0["capa"] == layer_name) & (d0["canal"] == c)].iloc[0]
            ax = axes[c, j]
            im = ax.imshow(r["w"], cmap="viridis", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"p = {r['p']:+.1f}", fontsize=9)
            if c == 0:
                ax.text(0.5, 1.32, layer_name.replace(": ", "\n"), transform=ax.transAxes, ha="center",
                        va="bottom", fontsize=9.5, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"canal {c}", fontsize=9)
    gs = axes[0, -1].get_gridspec()
    for a in axes[:, -1]:
        a.remove()
    ax = fig.add_subplot(gs[:, -1])
    for k, layer_name in enumerate(layers):
        v = df[df["capa"] == layer_name]
        for seed, vs in v.groupby("semilla"):
            ax.scatter(vs["p"], np.full(len(vs), k) + (seed - 43) * 0.12, s=22,
                       color={42: "#2a78d6", 43: "#eb6834", 44: "#1baf7a"}.get(seed, "#666"),
                       label=f"semilla {seed}" if k == 0 else None)
    ax.axvline(0, color="#999", lw=0.8)
    ax.set_yticks(range(len(layers)), [l.replace(": ", "\n") for l in layers], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("p aprendido\n(>0 ≈ dilatación, <0 ≈ erosión)", fontsize=9)
    ax.set_title("p por canal y semilla", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.colorbar(im, ax=axes[:, :-1].ravel().tolist(), shrink=0.6, label="peso w / máx (elemento estructurante)")
    fig.suptitle(f"Bloque morfológico aprendido (Morph-UNet, semilla {seed0}; filas = canales)", y=1.02)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return df.drop(columns="w")


def plot_examples(data, split, masks: dict, path: Path, n: int = 8) -> pd.DataFrame:
    """Los ``n`` recortes con mayor |ΔDice| entre U-Net y el mejor método morfológico."""
    gt = data[split].Y.astype(bool)
    dice = {m: extended_metrics(v, gt)["dice"].values for m, v in masks.items()}
    names = list(masks)
    delta = dice[names[-1]] - dice[names[0]]
    order = np.argsort(-np.abs(delta))[:n]
    fig, axes = plt.subplots(len(names) + 1, n, figsize=(2.05 * n, 2.15 * (len(names) + 1)))
    for j, i in enumerate(order):
        img = data[split].X[i]
        axes[0, j].imshow(img, cmap="gray")
        axes[0, j].contour(gt[i], levels=[0.5], colors=GT_COLOR, linewidths=1.2)
        axes[0, j].set_title(f"Δ Dice {delta[i]:+.2f}", fontsize=9)
        for k, m in enumerate(names, start=1):
            ax = axes[k, j]
            ax.imshow(img, cmap="gray")
            ax.contour(gt[i], levels=[0.5], colors=GT_COLOR, linewidths=0.9)
            if masks[m][i].any():
                ax.contour(masks[m][i], levels=[0.5], colors=COLORS.get(m, "#d62728"), linewidths=1.3)
            ax.set_title(f"Dice {dice[m][i]:.2f}", fontsize=8.5)
    for k, label in enumerate(["Imagen + anotación"] + names):
        axes[k, 0].set_ylabel(label.replace(" + ", "\n+ "), fontsize=9)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Recortes de {split} donde más cambia la predicción ({names[-1]} vs {names[0]}); "
                 "amarillo = anotación", y=1.0, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    meta = data[split].meta.iloc[order]
    return pd.DataFrame({"row_idx": meta["row_idx"].values, "patient_id": meta["patient_id"].values,
                         **{f"dice_{m}": dice[m][order] for m in names}, "delta": delta[order]})


def plot_comparison(summary: pd.DataFrame, path: Path) -> None:
    panels = [("dice", "Dice medio"), ("hd95", "Hausdorff 95 medio (px)"),
              ("frac_mas_de_1_componente", "Fracción con >1 componente")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    methods = list(dict.fromkeys(summary["método"]))
    for ax, (col, label) in zip(axes, panels):
        for k, m in enumerate(methods):
            d = summary[summary["método"] == m].set_index("conjunto")
            xs = np.arange(len(SPLITS)) + (k - 1) * 0.25
            vals = [d.loc[s, f"{col}_media"] for s in SPLITS]
            errs = [d.loc[s, f"{col}_de"] for s in SPLITS]
            ax.bar(xs, vals, 0.24, yerr=errs, capsize=3, color=COLORS[m], label=m)
            for x, v in zip(xs, vals):
                ax.text(x, v, f"{v:.3f}" if col != "hd95" else f"{v:.1f}", ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(range(len(SPLITS)), ["validación", "test"])
        ax.set_ylabel(label)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e5e5e5", lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylim(0, 1)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.08), frameon=False)
    fig.suptitle("U-Net vs posprocesamiento fijo vs Morph-UNet (media ± DE entre 3 semillas)", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- forma
def shape_rerun(data, masks: dict, seed: int) -> tuple:
    """Árbol de profundidad 3 ajustado con máscaras anotadas de train (como en el experimento
    principal), evaluado con la máscara anotada y con las de cada método (val y test)."""
    mtr = data["train"].meta
    ktr = (mtr["shape_label"] != "").values & mtr["shape_label"].notna().values
    clf = shp.fit_shape_tree(shp.features_dataframe(data["train"].Y[ktr]), mtr.loc[ktr, "shape_label"].values,
                             seed=seed)
    rows, dist = [], []
    for s in SPLITS:
        meta = data[s].meta
        k = (meta["shape_label"] != "").values & meta["shape_label"].notna().values
        y = meta.loc[k, "shape_label"].values
        sources = {"máscara anotada": data[s].Y, **{f"máscara {m}": v[s] for m, v in masks.items()}}
        for name, M in sources.items():
            pred = shp.predict_shape(clf, shp.features_dataframe(M[k]))
            r = shp.evaluate_shape(y, pred)
            rows.append({"conjunto": s, "máscara": name, "n": r["n"], "sin_descriptores": r["n_invalid"],
                         "exactitud": r["accuracy"], "exactitud_balanceada": r["balanced_accuracy"],
                         "f1_macro": r["macro_f1"]})
            vc = pd.Series(pred).value_counts(normalize=True)
            dist.append({"conjunto": s, "máscara": name,
                         **{f"frac_pred_{c}": float(vc.get(c, 0.0)) for c in shp.SHAPE_CLASSES + ["NONE"]}})
        vc = pd.Series(y).value_counts(normalize=True)
        dist.append({"conjunto": s, "máscara": "(etiqueta real)",
                     **{f"frac_pred_{c}": float(vc.get(c, 0.0)) for c in shp.SHAPE_CLASSES + ["NONE"]}})
    return pd.DataFrame(rows), pd.DataFrame(dist), shp.tree_rules(clf)


def main(argv=None):
    p = argparse.ArgumentParser(description="U-Net vs posprocesamiento fijo vs Morph-UNet.")
    p.add_argument("--unet-runs", nargs="+", required=True)
    p.add_argument("--morph-runs", nargs="+", required=True)
    p.add_argument("--position-runs", nargs="*", default=[])
    p.add_argument("--config", default=str(REPO / "configs" / "main.yaml"))
    p.add_argument("--data-dir")
    p.add_argument("--runs-dir", default=str(REPO / "outputs" / "runs"))
    p.add_argument("--results-dir", default=str(REPO / "results"))
    p.add_argument("--n-boot", type=int, default=1000)
    a = p.parse_args(argv)
    runs_dir, res = Path(a.runs_dir), Path(a.results_dir)
    (res / "figures").mkdir(parents=True, exist_ok=True)

    cfg = load_config(a.config, archive_dir=Path(a.data_dir).resolve() if a.data_dir else None)
    data, _ = prepare_data(cfg, progress=False, include_test=True)

    if a.position_runs:
        pos = position_table(runs_dir, a.position_runs, a.unet_runs[0])
        pos.to_csv(res / "morfologia_ubicacion.csv", index=False)
        print(pos.round(4).to_string(index=False), "\n")

    unet = {run_seed(run_config(runs_dir, r)): get_masks(runs_dir, r, data) for r in a.unet_runs}
    morph = {run_seed(run_config(runs_dir, r)): get_masks(runs_dir, r, data) for r in a.morph_runs}

    # (2) radio del posprocesamiento: solo validación
    sel = select_radius(list(unet.values()), data)
    sel.to_csv(res / "posproc_seleccion_radio.csv", index=False)
    cand = sel[sel["radio"] != "sin posproc."]
    best_r = int(cand.loc[cand["dice_val_media"].idxmax(), "radio"])
    print(sel.round(4).to_string(index=False), f"\nRadio elegido en validación: {best_r}\n")
    post = {sd: {s: postprocess_stack(m[s], best_r) for s in SPLITS} for sd, m in unet.items()}

    # (3) métricas por recorte y resumen
    by_method = {**{(UNET, sd): m for sd, m in unet.items()}, **{(POST, sd): m for sd, m in post.items()},
                 **{(MORPH, sd): m for sd, m in morph.items()}}
    pc = per_crop(by_method, data)
    pc.to_csv(REPO / "outputs" / "morfologia_por_recorte.csv", index=False)   # grande: fuera de git
    summary, per_seed = summarize(pc)
    summary.round(4).to_csv(res / "morfologia_comparacion.csv", index=False)
    per_seed.round(4).to_csv(res / "morfologia_por_semilla.csv", index=False)
    tests = paired_tests(pc, a.n_boot, cfg.seed)
    tests.to_csv(res / "morfologia_pruebas.csv", index=False)

    # decisión del modelo final: mayor Dice medio en validación
    val = summary[summary["conjunto"] == "val"].set_index("método")
    final = str(val["dice_media"].idxmax())
    decision = {"radio_posprocesamiento": best_r, "criterio": "mayor Dice medio en validación (3 semillas)",
                "dice_val": val["dice_media"].round(4).to_dict(), "modelo_final": final,
                "unet_runs": a.unet_runs, "morph_runs": a.morph_runs}
    (res / "morfologia_decision.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8")

    # (5) figuras
    models = {}
    for r in a.morph_runs:
        c = run_config(runs_dir, r)
        models[run_seed(c)] = load_unet(c, runs_dir / r / "unet_best.pt")
    se = plot_structuring_elements(models, res / "figures" / "morfologia_elementos_p.png")
    se.round(4).to_csv(res / "morfologia_p_aprendidos.csv", index=False)
    seed0 = min(unet)
    ex_masks = {UNET: unet[seed0]["val"], POST: post[seed0]["val"], MORPH: morph[min(morph)]["val"]}
    if final == POST:   # la última columna es la que se compara con la U-Net
        ex_masks = {UNET: ex_masks[UNET], MORPH: ex_masks[MORPH], POST: ex_masks[POST]}
    plot_examples(data, "val", ex_masks, res / "figures" / "morfologia_ejemplos.png") \
        .round(4).to_csv(res / "morfologia_ejemplos.csv", index=False)
    plot_comparison(summary, res / "figures" / "morfologia_comparacion.png")

    # (6) forma BI-RADS con las máscaras de cada método (semilla 42)
    shape_masks = {UNET: unet[seed0], POST: post[seed0], MORPH: morph[min(morph)]}
    shape_res, shape_dist, rules = shape_rerun(data, shape_masks, cfg.seed)
    shape_res.round(4).to_csv(res / "morfologia_forma.csv", index=False)
    shape_dist.round(4).to_csv(res / "morfologia_forma_distribucion.csv", index=False)

    cols = ["método", "conjunto", "dice_media", "dice_de", "iou_media", "precision_media", "recall_media",
            "hd95_media", "hd95_mediana_media", "n_componentes_media", "frac_mas_de_1_componente_media",
            "n_vacias_media"]
    print(summary[cols].round(4).to_string(index=False), "\n")
    print(tests.round(4).to_string(index=False), "\n")
    print(json.dumps(decision, indent=2, ensure_ascii=False), "\n")
    print(shape_res.round(4).to_string(index=False), "\n")
    print(shape_dist.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
