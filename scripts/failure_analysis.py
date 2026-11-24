"""Análisis de fallos (Dice < 0.1) de la U-Net en test.

Modos (por recorte con Dice < 0.1):
  - "predicción vacía": la red no marca ningún píxel.
  - "otra zona sin solape": marca algo, pero sin ningún píxel en común con la lesión.
  - "sub/sobresegmentación": hay solape, pero Dice < 0.1 (se distingue sub- de
    sobresegmentación por el cociente área predicha / área anotada).
Se compara el desplazamiento |shift| de la lesión (y su tamaño) entre fallos y no
fallos (Mann-Whitney; fracción de fallos en el tercil alto de desplazamiento).

Modelo principal (``--run``, semilla 42); se informa también cuántos fallos hay con las
otras semillas. No elige nada.
Uso:  python scripts/failure_analysis.py --run main --other-runs final_seed43 final_seed44
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402
import pandas as pd              # noqa: E402
from scipy import stats          # noqa: E402

from src.config import load_config                   # noqa: E402
from src.metrics import extended_metrics             # noqa: E402
from src.pipeline import cached_masks, prepare_data  # noqa: E402

THRESHOLD = 0.1
EMPTY, ELSEWHERE, PARTIAL = "predicción vacía", "otra zona sin solape", "sub/sobresegmentación"
MODES = [EMPTY, ELSEWHERE, PARTIAL]
GT_COLOR, PRED_COLOR = "#f2c200", "#d62728"


def failure_mode(pred: np.ndarray, gt: np.ndarray) -> str:
    if not pred.any():
        return EMPTY
    if not (pred & gt).any():
        return ELSEWHERE
    return PARTIAL


def classify(masks: np.ndarray, data_split) -> pd.DataFrame:
    gt = data_split.Y.astype(bool)
    df = extended_metrics(masks, gt)
    meta = data_split.meta.reset_index(drop=True)
    df = pd.concat([meta[["row_idx", "patient_id", "pathology", "shape_label", "mass_margins", "subtlety",
                          "breast_density", "lesion_area_frac", "shift_x", "shift_y"]], df], axis=1)
    df["abs_shift"] = np.hypot(df["shift_x"], df["shift_y"])
    df["tercil_shift"] = pd.qcut(df["abs_shift"], 3, labels=["bajo", "medio", "alto"])
    df["area_pred_sobre_anotada"] = masks.reshape(len(masks), -1).sum(1) / gt.reshape(len(gt), -1).sum(1)
    df["fallo"] = df["dice"] < THRESHOLD
    df["modo"] = [failure_mode(p, g) if f else "" for p, g, f in zip(masks, gt, df["fallo"])]
    df.loc[df["modo"] == PARTIAL, "detalle"] = np.where(
        df.loc[df["modo"] == PARTIAL, "area_pred_sobre_anotada"] < 1, "subsegmentación", "sobresegmentación")
    return df


def summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = [{"grupo": "todos los recortes", "n": len(df), "frac": 1.0,
             "abs_shift_mediana": df["abs_shift"].median(), "frac_tercil_alto": (df["tercil_shift"] == "alto").mean(),
             "area_lesion_mediana": df["lesion_area_frac"].median(), "sutileza_media": df["subtlety"].mean()}]
    for name, d in [("no fallo (Dice ≥ 0.1)", df[~df["fallo"]]), ("fallo (Dice < 0.1)", df[df["fallo"]])] + \
                   [(f"  {m}", df[df["modo"] == m]) for m in MODES]:
        rows.append({"grupo": name, "n": len(d), "frac": len(d) / len(df),
                     "abs_shift_mediana": d["abs_shift"].median() if len(d) else np.nan,
                     "frac_tercil_alto": (d["tercil_shift"] == "alto").mean() if len(d) else np.nan,
                     "area_lesion_mediana": d["lesion_area_frac"].median() if len(d) else np.nan,
                     "sutileza_media": d["subtlety"].mean() if len(d) else np.nan})
    return pd.DataFrame(rows)


def shift_tests(df: pd.DataFrame) -> dict:
    f, nf = df[df["fallo"]], df[~df["fallo"]]
    out = {}
    for col in ("abs_shift", "lesion_area_frac", "subtlety"):
        u = stats.mannwhitneyu(f[col], nf[col], alternative="two-sided")
        out[col] = {"mediana_fallo": float(f[col].median()), "mediana_no_fallo": float(nf[col].median()),
                    "U": float(u.statistic), "p": float(u.pvalue)}
    tab = pd.crosstab(df["tercil_shift"], df["fallo"])
    out["fallos_por_tercil_shift"] = {str(k): int(v) for k, v in tab.get(True, pd.Series(dtype=int)).items()}
    out["n_por_tercil_shift"] = {str(k): int(v) for k, v in tab.sum(axis=1).items()}
    if tab.shape[1] == 2:
        out["chi2_tercil"] = {"p": float(stats.chi2_contingency(tab)[1])}
    return out


def plot_modes(df: pd.DataFrame, data_split, masks: np.ndarray, path: Path, n: int = 8) -> None:
    fig, axes = plt.subplots(len(MODES), n, figsize=(2.05 * n, 2.35 * len(MODES)))
    X, gt = data_split.X, data_split.Y.astype(bool)
    for r, mode in enumerate(MODES):
        d = df[df["modo"] == mode].sort_values("abs_shift", ascending=False).head(n)
        for c in range(n):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            if c >= len(d):
                ax.axis("off")
                continue
            i = d.index[c]
            ax.imshow(X[i], cmap="gray")
            ax.contour(gt[i], levels=[0.5], colors=GT_COLOR, linewidths=1.1)
            if masks[i].any():
                ax.contour(masks[i], levels=[0.5], colors=PRED_COLOR, linewidths=1.1)
            ax.set_title(f"Dice {d.loc[i, 'dice']:.2f} | shift {d.loc[i, 'abs_shift']:.2f}", fontsize=8)
        axes[r, 0].set_ylabel(f"{mode}\n(n={int((df['modo'] == mode).sum())})", fontsize=9)
        if len(d) == 0:
            axes[r, 0].axis("on")
            axes[r, 0].set_xticks([])
            axes[r, 0].set_yticks([])
            axes[r, 0].text(0.5, 0.5, "sin casos", ha="center", va="center", transform=axes[r, 0].transAxes)
    fig.suptitle("Fallos de la U-Net en test (Dice < 0.1) por modo; amarillo = anotación, rojo = predicción; "
                 "shift = |desplazamiento| del centro (fracción del lado)", fontsize=10.5, y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_shift(df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    bins = np.linspace(0, df["abs_shift"].max(), 16)
    axes[0].hist(df.loc[~df["fallo"], "abs_shift"], bins=bins, density=True, alpha=0.6, color="#2a78d6",
                 label=f"Dice ≥ 0.1 (n={int((~df['fallo']).sum())})")
    axes[0].hist(df.loc[df["fallo"], "abs_shift"], bins=bins, density=True, alpha=0.7, color="#d62728",
                 label=f"Dice < 0.1 (n={int(df['fallo'].sum())})")
    axes[0].set_xlabel("|desplazamiento| del centro de la lesión (fracción del lado)")
    axes[0].set_ylabel("densidad")
    axes[0].legend(fontsize=8.5)
    axes[1].scatter(df["abs_shift"], df["dice"], s=10, alpha=0.5, color="#2a78d6")
    axes[1].axhline(THRESHOLD, color="#d62728", lw=0.8, ls="--")
    axes[1].set_xlabel("|desplazamiento| del centro de la lesión (fracción del lado)")
    axes[1].set_ylabel("Dice de la U-Net")
    rho = stats.spearmanr(df["abs_shift"], df["dice"])
    axes[1].set_title(f"Spearman ρ = {rho.statistic:.2f} (p = {rho.pvalue:.2g})", fontsize=10)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    import json
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run", default="main")
    p.add_argument("--other-runs", nargs="*", default=["final_seed43", "final_seed44"])
    p.add_argument("--config", default=str(REPO / "configs" / "main.yaml"))
    p.add_argument("--data-dir")
    p.add_argument("--runs-dir", default=str(REPO / "outputs" / "runs"))
    p.add_argument("--results-dir", default=str(REPO / "results"))
    a = p.parse_args(argv)
    res, runs_dir = Path(a.results_dir), Path(a.runs_dir)
    (res / "figures").mkdir(parents=True, exist_ok=True)
    cfg = load_config(a.config, archive_dir=Path(a.data_dir).resolve() if a.data_dir else None)
    data, _ = prepare_data(cfg, progress=False, include_test=True)

    masks = cached_masks(runs_dir / a.run, data, ("val", "test"))["test"]
    df = classify(masks, data["test"])
    fails = df[df["fallo"]].drop(columns=["fallo"])
    fails.round(4).to_csv(res / "fallos_test.csv", index=False)
    summ = summary(df)
    summ.round(4).to_csv(res / "fallos_resumen.csv", index=False)
    tests = shift_tests(df)
    tests["modelo"] = a.run
    tests["detalle_sub_sobre"] = fails["detalle"].value_counts().to_dict()
    tests["otras_semillas"] = {}
    for r in a.other_runs:
        d = classify(cached_masks(runs_dir / r, data, ("val", "test"))["test"], data["test"])
        tests["otras_semillas"][r] = {"n_fallos": int(d["fallo"].sum()),
                                      **{m: int((d["modo"] == m).sum()) for m in MODES}}
    (res / "fallos_pruebas.json").write_text(json.dumps(tests, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_modes(df, data["test"], masks, res / "figures" / "fallos_por_modo.png")
    plot_shift(df, res / "figures" / "fallos_desplazamiento.png")

    print(summ.round(3).to_string(index=False), "\n")
    print(fails[["row_idx", "modo", "detalle", "dice", "abs_shift", "tercil_shift", "lesion_area_frac",
                 "subtlety", "shape_label", "breast_density"]].round(3).to_string(index=False), "\n")
    print(json.dumps(tests, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
