"""Variabilidad e incertidumbre de la configuración final (tablas y figura en results/).

1) Semillas: media ± DE del Dice/IoU de la U-Net en val y test sobre las
   ejecuciones dadas (misma configuración, distinta semilla de entrenamiento).
2) Validación cruzada por paciente: resumen de ``outputs/runs/<cv>/cv_folds.csv``.
3) Incertidumbre en test (ejecución principal): IC 95 % por bootstrap (1000
   remuestreos) del Dice medio de U-Net, clásico y prior, remuestreando recortes y,
   como análisis de sensibilidad, pacientes; IC de la diferencia U-Net − clásico y
   prueba pareada de Wilcoxon sobre el Dice por recorte.

No entrena nada: solo lee ``outputs/runs/``.
Uso:
    python scripts/final_statistics.py --runs main final_seed43 final_seed44 --cv main_cv5
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

from src.pipeline import CLASSICAL, PRIOR, UNET                # noqa: E402
from src.stats import bootstrap_mean_ci, paired_wilcoxon      # noqa: E402

METHODS = [UNET, CLASSICAL, PRIOR]
COLORS = {UNET: "#2a78d6", CLASSICAL: "#eb6834", PRIOR: "#1baf7a"}


def seed_table(runs_dir: Path, runs: list) -> pd.DataFrame:
    rows = []
    for r in runs:
        m = json.loads((runs_dir / r / "metrics.json").read_text(encoding="utf-8"))
        cfg = yaml.safe_load((runs_dir / r / "config.yaml").read_text(encoding="utf-8"))
        seed = cfg.get("train_seed") if cfg.get("train_seed") is not None else cfg["seed"]
        seg = m["segmentacion"]
        rows.append({"ejecucion": r, "semilla_entrenamiento": int(seed),
                     "dice_val": seg[f"{UNET} | val"]["dice"], "iou_val": seg[f"{UNET} | val"]["iou"],
                     "dice_test": seg[f"{UNET} | test"]["dice"], "iou_test": seg[f"{UNET} | test"]["iou"],
                     "epocas": m["epocas_entrenadas"], "mejor_epoca": m["mejor_epoca"]})
    df = pd.DataFrame(rows)
    num = ["dice_val", "iou_val", "dice_test", "iou_test"]
    summary = {"ejecucion": "media ± DE", **{c: f"{df[c].mean():.4f} ± {df[c].std(ddof=1):.4f}" for c in num}}
    return pd.concat([df, pd.DataFrame([summary])], ignore_index=True)


def uncertainty(per_sample: pd.DataFrame, n_boot: int, seed: int):
    test = per_sample[per_sample["conjunto"] == "test"]
    dice = {m: test[test["método"] == m].sort_values("row_idx") for m in METHODS}
    rows = []
    for m in METHODS:
        d = dice[m]
        rec = bootstrap_mean_ci(d["dice"], n_boot, seed=seed)
        pac = bootstrap_mean_ci(d["dice"], n_boot, seed=seed, groups=d["patient_id"])
        rows.append({"método": m, "n_recortes": len(d), "n_pacientes": int(d["patient_id"].nunique()),
                     "dice_medio": rec["media"], "ic95_inf_recortes": rec["ic_inf"], "ic95_sup_recortes": rec["ic_sup"],
                     "ic95_inf_pacientes": pac["ic_inf"], "ic95_sup_pacientes": pac["ic_sup"]})
    u, c = dice[UNET], dice[CLASSICAL]
    assert (u["row_idx"].values == c["row_idx"].values).all()      # mismo recorte en cada par
    diff = u["dice"].values - c["dice"].values
    tests = {
        "bootstrap": {"n_boot": n_boot, "semilla": seed, "nivel": 0.95, "metodo": "percentil"},
        "diferencia_unet_menos_clasico": {
            "recortes": bootstrap_mean_ci(diff, n_boot, seed=seed),
            "pacientes": bootstrap_mean_ci(diff, n_boot, seed=seed, groups=u["patient_id"].values)},
        "wilcoxon_unet_vs_clasico": paired_wilcoxon(u["dice"].values, c["dice"].values),
    }
    return pd.DataFrame(rows), tests


def plot_summary(unc: pd.DataFrame, seeds: pd.DataFrame, cv: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    # (1) Dice en test con IC 95 %
    ax = axes[0]
    for i, r in unc.iterrows():
        m = r["método"]
        ax.errorbar(i - 0.1, r["dice_medio"], yerr=[[r["dice_medio"] - r["ic95_inf_recortes"]],
                    [r["ic95_sup_recortes"] - r["dice_medio"]]], fmt="o", color=COLORS[m], capsize=4, ms=7)
        ax.errorbar(i + 0.1, r["dice_medio"], yerr=[[r["dice_medio"] - r["ic95_inf_pacientes"]],
                    [r["ic95_sup_pacientes"] - r["dice_medio"]]], fmt="s", color=COLORS[m], capsize=4, ms=6,
                    mfc="white")
        ax.text(i + 0.22, r["dice_medio"], f"{r['dice_medio']:.3f}", va="center", fontsize=8.5)
    ax.set_xticks(range(len(unc)), ["U-Net", "Clásico\n(Otsu+morf.)", "Prior\ncentrado"], fontsize=9)
    ax.set_xlim(-0.5, len(unc) - 0.3)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Dice medio en test")
    ax.set_title("Test: IC 95 % bootstrap\n(● por recorte, □ por paciente)", fontsize=10)
    # (2) semillas
    ax = axes[1]
    s = seeds[seeds["ejecucion"] != "media ± DE"]
    for j, (col, label) in enumerate((("dice_val", "validación"), ("dice_test", "test"))):
        vals = s[col].astype(float).values
        ax.scatter(np.full(len(vals), j), vals, color="#2a78d6", s=40, zorder=3)
        ax.errorbar(j + 0.18, vals.mean(), yerr=vals.std(ddof=1), fmt="_", color="#333", capsize=5, ms=14)
        ax.text(j + 0.28, vals.mean(), f"{vals.mean():.3f}\n± {vals.std(ddof=1):.3f}", va="center", fontsize=8.5)
    ax.set_xticks([0, 1], ["validación", "test"])
    ax.set_xlim(-0.4, 1.8)
    ax.set_ylabel("Dice medio de la U-Net")
    ax.set_title(f"Semillas de entrenamiento (n={len(s)})", fontsize=10)
    # (3) validación cruzada por paciente
    ax = axes[2]
    for j, (col, label, color) in enumerate((("dice_unet", "U-Net", "#2a78d6"),
                                             ("dice_prior", "Prior", "#1baf7a"))):
        vals = cv[col].values
        ax.scatter(np.full(len(vals), j), vals, color=color, s=40, zorder=3)
        ax.errorbar(j + 0.18, vals.mean(), yerr=vals.std(ddof=1), fmt="_", color="#333", capsize=5, ms=14)
        ax.text(j + 0.28, vals.mean(), f"{vals.mean():.3f}\n± {vals.std(ddof=1):.3f}", va="center", fontsize=8.5)
    ax.set_xticks([0, 1], ["U-Net", "Prior centrado"])
    ax.set_xlim(-0.4, 1.8)
    ax.set_ylabel("Dice medio en la partición retenida")
    ax.set_title(f"Validación cruzada por paciente ({len(cv)} particiones)", fontsize=10)
    for ax in axes[1:]:
        ax.set_ylim(0, 1)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e5e5e5", lw=0.6)
        ax.set_axisbelow(True)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description="Variabilidad e incertidumbre de la configuración final.")
    p.add_argument("--runs", nargs="+", required=True, help="ejecuciones (semillas) en outputs/runs/")
    p.add_argument("--primary", help="ejecución para el análisis de test (por defecto la primera de --runs)")
    p.add_argument("--cv", required=True, help="carpeta de la validación cruzada en outputs/runs/")
    p.add_argument("--runs-dir", default=str(REPO / "outputs" / "runs"))
    p.add_argument("--results-dir", default=str(REPO / "results"))
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(argv)
    runs_dir, res = Path(a.runs_dir), Path(a.results_dir)
    res.mkdir(parents=True, exist_ok=True)

    seeds = seed_table(runs_dir, a.runs)
    seeds.to_csv(res / "variabilidad_semillas.csv", index=False)

    cv = pd.read_csv(runs_dir / a.cv / "cv_folds.csv")
    cv_out = pd.concat([cv, pd.DataFrame([{"fold": "media ± DE", **{
        c: f"{cv[c].mean():.4f} ± {cv[c].std(ddof=1):.4f}"
        for c in ("dice_unet", "iou_unet", "dice_prior", "iou_prior")}}])], ignore_index=True)
    cv_out.to_csv(res / "cv_paciente.csv", index=False)

    per_sample = pd.read_csv(runs_dir / (a.primary or a.runs[0]) / "per_sample.csv")
    unc, tests = uncertainty(per_sample, a.n_boot, a.seed)
    unc.round(4).to_csv(res / "incertidumbre_test.csv", index=False)
    tests["ejecucion"] = a.primary or a.runs[0]
    with open(res / "pruebas_test.json", "w", encoding="utf-8") as f:
        json.dump(tests, f, indent=2, ensure_ascii=False)
    plot_summary(unc, seeds, cv, res / "figures" / "variabilidad_incertidumbre.png")

    print(seeds.to_string(index=False), "\n")
    print(cv_out.round(4).to_string(index=False), "\n")
    print(unc.round(4).to_string(index=False), "\n")
    print(json.dumps(tests, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
