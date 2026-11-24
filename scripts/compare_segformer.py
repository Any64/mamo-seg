"""E13: U-Net vs SegFormer-B0, con y sin posprocesamiento fijo (r = 6). No entrena.

1) lr de SegFormer elegido SOLO en validación con la semilla 42 (``--lr-runs``).
2) Métricas en val y test (Dice, IoU, precisión, sensibilidad, HD95 medio y mediano,
   componentes conexas y predicciones vacías): media ± DE entre semillas.
3) Diferencias pareadas por recorte (Dice y HD95 promediados entre semillas): Wilcoxon e
   IC 95 % bootstrap por paciente (y por recorte).
4) Hipótesis de localización: fallos (Dice < 0.1) por modo, con el mismo criterio de
   ``failure_analysis.py`` (vacía / otra zona sin solape / sub-sobresegmentación), por
   modelo y semilla; McNemar exacto pareando semilla con semilla.
5) Forma BI-RADS con las máscaras de cada método (mismo árbol que el experimento principal).
6) Figuras: 8 recortes de test donde SegFormer difiere más de la U-Net (semilla 42) y
   comparación de métricas.

El test solo se informa: el radio (r = 6) y el lr vienen de validación.
Uso:
    python scripts/compare_segformer.py --unet-runs main final_seed43 final_seed44 \\
        --segformer-runs segformer_lr1e-4 segformer_s43 segformer_s44 \\
        --lr-runs segformer_lr1e-4 segformer_lr3e-4
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402
import pandas as pd              # noqa: E402
from scipy import stats          # noqa: E402

import compare_morphology as cm                                 # noqa: E402
from failure_analysis import MODES, THRESHOLD, failure_mode     # noqa: E402
from src.config import load_config                              # noqa: E402
from src.pipeline import cached_masks, prepare_data, run_config  # noqa: E402
from src.postprocess import postprocess_stack                   # noqa: E402
from src.stats import bootstrap_mean_ci, paired_wilcoxon        # noqa: E402

UNET, UNET_P = "U-Net", "U-Net + posproc."
SEG, SEG_P = "SegFormer-B0", "SegFormer-B0 + posproc."
METHODS = [UNET, UNET_P, SEG, SEG_P]
COLORS = {UNET: "#2a78d6", UNET_P: "#1baf7a", SEG: "#eb6834", SEG_P: "#8e44ad"}
SPLITS = ("val", "test")
RADIUS = 6   # elegido en validación en E12 (results/morfologia_decision.json); no se vuelve a elegir
PAIRS = [(SEG, UNET), (SEG, UNET_P), (SEG_P, UNET_P), (SEG_P, SEG)]


def seed_of(runs_dir: Path, run: str) -> int:
    c = run_config(runs_dir / run)
    return c.seed if c.train_seed is None else c.train_seed


def lr_table(runs_dir: Path, runs: list) -> pd.DataFrame:
    rows = []
    for r in runs:
        m = json.loads((runs_dir / r / "metrics.json").read_text(encoding="utf-8"))
        c = run_config(runs_dir / r)
        seg = m["segmentacion"]["U-Net | val"]          # la clave es la de la red entrenada
        rows.append({"ejecucion": r, "lr": c.lr, "semilla": seed_of(runs_dir, r), "dice_val": seg["dice"],
                     "iou_val": seg["iou"], "epocas": m["epocas_entrenadas"], "mejor_epoca": m["mejor_epoca"],
                     "segundos_por_epoca_mediana": float(pd.read_csv(runs_dir / r / "history.csv").iloc[:, -1].median())})
    return pd.DataFrame(rows)


def summarize(pc: pd.DataFrame):
    g = pc.groupby(["método", "conjunto", "semilla"])
    per_seed = g.agg(dice=("dice", "mean"), iou=("iou", "mean"), precision=("precision", "mean"),
                     recall=("recall", "mean"), hd95=("hd95", "mean"), hd95_mediana=("hd95", "median"),
                     n_componentes=("n_componentes", "mean"),
                     frac_mas_de_1_componente=("n_componentes", lambda v: float((v > 1).mean())),
                     n_vacias=("vacia", "sum"), n_dice_menor_01=("dice", lambda v: int((v < THRESHOLD).sum())),
                     n=("dice", "size")).reset_index()
    rows = []
    for (m, s), d in per_seed.groupby(["método", "conjunto"], sort=False):
        row = {"método": m, "conjunto": s, "n_recortes": int(d["n"].iloc[0]), "n_semillas": len(d)}
        for c in ["dice", "iou", "precision", "recall", "hd95", "hd95_mediana", "n_componentes",
                  "frac_mas_de_1_componente", "n_vacias", "n_dice_menor_01"]:
            row[f"{c}_media"] = float(d[c].mean())
            row[f"{c}_de"] = float(d[c].std(ddof=1)) if len(d) > 1 else float("nan")
        rows.append(row)
    out = pd.DataFrame(rows)
    out["_m"] = out["método"].map(METHODS.index)
    out["_s"] = out["conjunto"].map(SPLITS.index)
    return out.sort_values(["_s", "_m"]).drop(columns=["_m", "_s"]).reset_index(drop=True), per_seed


def paired_tests(pc: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    for s in SPLITS:
        avg = (pc[pc["conjunto"] == s].groupby(["método", "row_idx"])
               .agg(dice=("dice", "mean"), hd95=("hd95", "mean"), patient_id=("patient_id", "first")).reset_index())
        piv = {m: avg[avg["método"] == m].set_index("row_idx").sort_index() for m in METHODS}
        for a, b in PAIRS:
            for metric in ("dice", "hd95"):
                va, vb = piv[a][metric], piv[b][metric]
                ok = va.notna() & vb.notna()
                diff = (va - vb)[ok]
                groups = piv[a].loc[ok, "patient_id"].values
                w = paired_wilcoxon(va[ok].values, vb[ok].values) if (diff != 0).any() else {}
                ci_c = bootstrap_mean_ci(diff.values, n_boot, seed=seed)
                ci_p = bootstrap_mean_ci(diff.values, n_boot, seed=seed, groups=groups)
                rows.append({"conjunto": s, "comparación": f"{a} − {b}", "métrica": metric, "n": int(ok.sum()),
                             "diferencia_media": ci_c["media"], "ic95_inf_pacientes": ci_p["ic_inf"],
                             "ic95_sup_pacientes": ci_p["ic_sup"], "ic95_inf_recortes": ci_c["ic_inf"],
                             "ic95_sup_recortes": ci_c["ic_sup"],
                             "mediana_diferencia": float(np.median(diff)) if len(diff) else float("nan"),
                             "wilcoxon_p": w.get("p_valor", float("nan")),
                             "frac_a_mejor": float((diff > 0).mean() if metric == "dice" else (diff < 0).mean())})
    return pd.DataFrame(rows)


def failure_modes(by_method: dict, data: dict) -> pd.DataFrame:
    """Un registro por (método, semilla, conjunto, recorte) con su modo de fallo ("" si no falla)."""
    rows = []
    for (m, sd), masks in by_method.items():
        for s in SPLITS:
            gt = data[s].Y.astype(bool)
            for i, (p, g) in enumerate(zip(masks[s], gt)):
                inter = (p & g).sum()
                dice = 2 * inter / (p.sum() + g.sum()) if (p.sum() + g.sum()) else 1.0
                rows.append({"método": m, "semilla": sd, "conjunto": s, "i": i,
                             "modo": failure_mode(p, g) if dice < THRESHOLD else ""})
    return pd.DataFrame(rows)


def failure_tables(fm: pd.DataFrame):
    counts = []
    for (m, sd, s), d in fm.groupby(["método", "semilla", "conjunto"]):
        counts.append({"método": m, "semilla": sd, "conjunto": s, "fallos": int((d["modo"] != "").sum()),
                       **{mo: int((d["modo"] == mo).sum()) for mo in MODES}})
    counts = pd.DataFrame(counts)
    agg = counts.groupby(["método", "conjunto"]).agg(
        **{f"{c}_media": (c, "mean") for c in ["fallos"] + MODES},
        **{f"{c}_de": (c, "std") for c in ["fallos"] + MODES}).reset_index()
    # McNemar exacto pareado semilla con semilla (mismo recorte): ¿falla uno y el otro no?
    tests = []
    for a, b in PAIRS:
        for s in SPLITS:
            for what in ["fallos"] + MODES:
                n01 = n10 = 0
                for sd in sorted(fm["semilla"].unique()):
                    da = fm[(fm["método"] == a) & (fm["semilla"] == sd) & (fm["conjunto"] == s)].set_index("i")
                    db = fm[(fm["método"] == b) & (fm["semilla"] == sd) & (fm["conjunto"] == s)].set_index("i")
                    fa = (da["modo"] != "") if what == "fallos" else (da["modo"] == what)
                    fb = (db["modo"] != "") if what == "fallos" else (db["modo"] == what)
                    n10 += int((fa & ~fb).sum())
                    n01 += int((~fa & fb).sum())
                p = stats.binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else float("nan")
                tests.append({"conjunto": s, "comparación": f"{a} vs {b}", "tipo": what,
                              "solo_falla_a": n10, "solo_falla_b": n01, "mcnemar_exacto_p": p})
    agg["_m"] = agg["método"].map(METHODS.index)
    agg["_s"] = agg["conjunto"].map(SPLITS.index)
    agg = agg.sort_values(["_s", "_m"]).drop(columns=["_m", "_s"])
    return counts, agg, pd.DataFrame(tests)


def plot_comparison(summary: pd.DataFrame, path: Path) -> None:
    panels = [("dice", "Dice medio"), ("hd95_mediana", "HD95 mediano (px)"),
              ("n_dice_menor_01", "Recortes con Dice < 0.1")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    w = 0.2
    for ax, (col, label) in zip(axes, panels):
        for k, m in enumerate(METHODS):
            d = summary[summary["método"] == m].set_index("conjunto")
            xs = np.arange(len(SPLITS)) + (k - 1.5) * w
            vals = [d.loc[s, f"{col}_media"] for s in SPLITS]
            errs = [d.loc[s, f"{col}_de"] for s in SPLITS]
            ax.bar(xs, vals, w * 0.95, yerr=errs, capsize=3, color=COLORS[m], label=m)
            for x, v in zip(xs, vals):
                ax.text(x + w * 0.05, v * 0.5, f"{v:.3f}" if col == "dice" else f"{v:.1f}", ha="center",
                        va="center", fontsize=7, color="white", rotation=90)
        ax.set_xticks(range(len(SPLITS)), ["validación", "test"])
        ax.set_ylabel(label)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e5e5e5", lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylim(0, 1)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.08), frameon=False)
    fig.suptitle(f"U-Net vs SegFormer-B0, con y sin posprocesamiento fijo (r = {RADIUS}); media ± DE de 3 semillas",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description="E13: U-Net vs SegFormer-B0.")
    p.add_argument("--unet-runs", nargs="+", required=True)
    p.add_argument("--segformer-runs", nargs="+", required=True)
    p.add_argument("--lr-runs", nargs="*", default=[])
    p.add_argument("--config", default=str(REPO / "configs" / "main.yaml"))
    p.add_argument("--runs-dir", default=str(REPO / "outputs" / "runs"))
    p.add_argument("--results-dir", default=str(REPO / "results"))
    p.add_argument("--n-boot", type=int, default=1000)
    a = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    runs_dir, res = Path(a.runs_dir), Path(a.results_dir)
    (res / "figures").mkdir(parents=True, exist_ok=True)
    cfg = load_config(a.config)
    data, _ = prepare_data(cfg, progress=False, include_test=True)

    if a.lr_runs:
        lt = lr_table(runs_dir, a.lr_runs)
        lt.to_csv(res / "segformer_eleccion_lr.csv", index=False)
        print(lt.round(4).to_string(index=False), "\n")

    unet = {seed_of(runs_dir, r): cached_masks(runs_dir / r, data, SPLITS) for r in a.unet_runs}
    seg = {seed_of(runs_dir, r): cached_masks(runs_dir / r, data, SPLITS) for r in a.segformer_runs}
    post = lambda d: {sd: {s: postprocess_stack(m[s], RADIUS) for s in SPLITS} for sd, m in d.items()}  # noqa: E731
    by_method = {}
    for name, d in ((UNET, unet), (UNET_P, post(unet)), (SEG, seg), (SEG_P, post(seg))):
        by_method.update({(name, sd): m for sd, m in d.items()})

    pc = cm.per_crop(by_method, data)
    pc.to_csv(REPO / "outputs" / "segformer_por_recorte.csv", index=False)   # grande: fuera de git
    summary, per_seed = summarize(pc)
    summary.round(4).to_csv(res / "segformer_comparacion.csv", index=False)
    per_seed.round(4).to_csv(res / "segformer_por_semilla.csv", index=False)
    tests = paired_tests(pc, a.n_boot, cfg.seed)
    tests.to_csv(res / "segformer_pruebas.csv", index=False)

    fm = failure_modes(by_method, data)
    counts, agg, ftests = failure_tables(fm)
    counts.to_csv(res / "segformer_fallos_por_semilla.csv", index=False)
    agg.round(3).to_csv(res / "segformer_fallos.csv", index=False)
    ftests.to_csv(res / "segformer_fallos_pruebas.csv", index=False)

    seed0 = min(unet)
    ex = {UNET: unet[seed0]["test"], SEG: seg[min(seg)]["test"]}
    cm.COLORS.update(COLORS)
    cm.plot_examples(data, "test", ex, res / "figures" / "segformer_ejemplos.png") \
        .round(4).to_csv(res / "segformer_ejemplos.csv", index=False)
    plot_comparison(summary, res / "figures" / "segformer_comparacion.png")

    shape_masks = {UNET: unet[seed0], UNET_P: by_method[(UNET_P, seed0)], SEG: seg[min(seg)],
                   SEG_P: by_method[(SEG_P, min(seg))]}
    shape_res, shape_dist, _ = cm.shape_rerun(data, shape_masks, cfg.seed)
    shape_res.round(4).to_csv(res / "segformer_forma.csv", index=False)
    shape_dist.round(4).to_csv(res / "segformer_forma_distribucion.csv", index=False)

    cols = ["método", "conjunto", "dice_media", "dice_de", "iou_media", "precision_media", "recall_media",
            "hd95_media", "hd95_mediana_media", "n_componentes_media", "n_vacias_media", "n_dice_menor_01_media"]
    print(summary[cols].round(4).to_string(index=False), "\n")
    print(tests.round(4).to_string(index=False), "\n")
    print(agg.round(2).to_string(index=False), "\n")
    print(ftests[ftests["conjunto"] == "test"].round(4).to_string(index=False), "\n")
    print(shape_res.round(4).to_string(index=False), "\n")
    print(shape_dist.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
