"""Análisis estratificado en test del modelo principal (U-Net, 3 semillas promediadas por recorte).

Dice, IoU y Hausdorff 95 por forma (4 básicas + "otras"), margen, sutileza, patología,
densidad mamaria y vista CC/MLO. n por grupo (``n_pequeño`` si n < 15), Kruskal-Wallis
por variable y métrica con p ajustado por Holm (18 pruebas). No entrena ni elige nada:
solo describe el test de la configuración final.

Por recorte se promedia el Dice/IoU de las 3 semillas; el HD95 se promedia sobre las
semillas con predicción no vacía (NaN si las 3 están vacías: se cuentan aparte).

Uso:  python scripts/stratified_analysis.py --runs main final_seed43 final_seed44
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

from src.config import load_config                 # noqa: E402
from src.metrics import extended_metrics           # noqa: E402
from src.pipeline import cached_masks, prepare_data  # noqa: E402

MIN_N = 15
MARGINS = ["CIRCUMSCRIBED", "OBSCURED", "ILL_DEFINED", "MICROLOBULATED", "SPICULATED"]
SHAPES = ["ROUND", "OVAL", "LOBULATED", "IRREGULAR"]
VARIABLES = {   # columna -> (nombre, orden de los grupos)
    "forma": ("Forma BI-RADS", SHAPES + ["otras"]),
    "margen": ("Margen", MARGINS + ["mixto/sin dato"]),
    "sutileza": ("Sutileza (1 = sutil, 5 = evidente)", ["1", "2", "3", "4", "5"]),
    "patologia": ("Patología", ["BENIGN", "MALIGNANT"]),
    "densidad": ("Densidad mamaria (BI-RADS)", ["1", "2", "3", "4"]),
    "vista": ("Vista", ["CC", "MLO"]),
}
METRICS = {"dice": "Dice", "iou": "IoU", "hd95": "HD95 (px)"}


def strata(meta: pd.DataFrame) -> pd.DataFrame:
    """Variables de estratificación con los grupos de ``VARIABLES``."""
    shape = meta["shape_label"].fillna("").replace("", "otras")
    margin = meta["mass_margins"].fillna("").where(meta["mass_margins"].isin(MARGINS), "mixto/sin dato")
    return pd.DataFrame({
        "forma": shape.values, "margen": margin.values,
        "sutileza": meta["subtlety"].astype(int).clip(lower=1).astype(str).values,   # 0 (1 caso en val) -> 1
        "patologia": meta["pathology"].values,
        "densidad": meta["breast_density"].astype(int).astype(str).values,
        "vista": meta["view"].values,
    })


def per_crop_test(runs_dir: Path, runs: list, data: dict) -> pd.DataFrame:
    gt = data["test"].Y
    per_seed = [extended_metrics(cached_masks(runs_dir / r, data, ("val", "test"))["test"], gt) for r in runs]
    stack = lambda c: np.vstack([d[c].values for d in per_seed])  # noqa: E731
    meta = data["test"].meta.reset_index(drop=True)
    out = pd.DataFrame({"row_idx": meta["row_idx"], "patient_id": meta["patient_id"],
                        "dice": stack("dice").mean(0), "iou": stack("iou").mean(0),
                        "hd95": _nanmean0(stack("hd95")),
                        "semillas_vacias": stack("vacia").sum(0)})
    return pd.concat([out, strata(meta)], axis=1)


def _nanmean0(a: np.ndarray) -> np.ndarray:
    """Media por columna ignorando NaN; NaN si toda la columna es NaN (sin avisos)."""
    n = (~np.isnan(a)).sum(0)
    s = np.nansum(a, axis=0)
    return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def group_table(pc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for var, (label, order) in VARIABLES.items():
        for g in order:
            d = pc[pc[var] == g]
            if not len(d):
                continue
            rows.append({"variable": label, "grupo": g, "n": len(d), "n_pacientes": d["patient_id"].nunique(),
                         "n_pequeño": len(d) < MIN_N,
                         "dice_media": d["dice"].mean(), "dice_de": d["dice"].std(), "dice_mediana": d["dice"].median(),
                         "iou_media": d["iou"].mean(), "hd95_media": d["hd95"].mean(),
                         "hd95_mediana": d["hd95"].median(), "frac_dice_lt_0_1": (d["dice"] < 0.1).mean(),
                         "n_hd95_sin_dato": int(d["hd95"].isna().sum())})
    return pd.DataFrame(rows)


def kruskal_table(pc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for var, (label, order) in VARIABLES.items():
        for m, mlabel in METRICS.items():
            groups = [pc.loc[pc[var] == g, m].dropna().values for g in order]
            groups = [g for g in groups if len(g)]
            h, p = stats.kruskal(*groups)
            rows.append({"variable": label, "métrica": mlabel, "k_grupos": len(groups),
                         "n": int(sum(len(g) for g in groups)), "H": h, "p": p})
    df = pd.DataFrame(rows)
    # Holm-Bonferroni sobre las 18 pruebas
    order = np.argsort(df["p"].values)
    m = len(df)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * df["p"].iloc[i])
        adj[i] = min(1.0, running)
    df["p_holm"] = adj
    df["significativo_holm_0_05"] = df["p_holm"] < 0.05
    return df


def plot_strata(gt: pd.DataFrame, kw: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6))
    for ax, (var, (label, order)) in zip(axes.ravel(), VARIABLES.items()):
        d = gt[gt["variable"] == label].set_index("grupo").reindex([g for g in order if g in
                                                                     set(gt[gt["variable"] == label]["grupo"])])
        y = np.arange(len(d))
        colors = ["#bbbbbb" if small else "#2a78d6" for small in d["n_pequeño"]]
        ax.barh(y, d["dice_media"], xerr=d["dice_de"] / np.sqrt(d["n"]), color=colors, capsize=3, height=0.65)
        for yi, (g, r) in zip(y, d.iterrows()):
            ax.text(0.01, yi, f"n={r['n']}{' *' if r['n_pequeño'] else ''}", va="center", ha="left",
                    fontsize=8.5, color="white" if not r["n_pequeño"] else "black")
            ax.text(r["dice_media"] + r["dice_de"] / np.sqrt(r["n"]) + 0.01, yi,
                    f"{r['dice_media']:.2f} | HD95 {r['hd95_mediana']:.0f}", va="center", fontsize=8.5)
        ax.set_yticks(y, [g.replace("_", " ").lower() if var in ("forma", "margen", "patologia") else g
                          for g in d.index], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.15)
        k = kw[(kw["variable"] == label) & (kw["métrica"] == "Dice")].iloc[0]
        ax.set_title(f"{label}\nKruskal-Wallis Dice: p = {k['p']:.3g} (Holm {k['p_holm']:.3g})", fontsize=10)
        ax.set_xlabel("Dice medio en test (± EE)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#e5e5e5", lw=0.6)
        ax.set_axisbelow(True)
    fig.suptitle("U-Net en test por subgrupo (Dice por recorte promediado en 3 semillas; "
                 "texto: Dice medio | HD95 mediano en px; gris y * = n < 15)", y=1.0, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--runs", nargs="+", default=["main", "final_seed43", "final_seed44"])
    p.add_argument("--config", default=str(REPO / "configs" / "main.yaml"))
    p.add_argument("--data-dir")
    p.add_argument("--runs-dir", default=str(REPO / "outputs" / "runs"))
    p.add_argument("--results-dir", default=str(REPO / "results"))
    a = p.parse_args(argv)
    res = Path(a.results_dir)
    (res / "figures").mkdir(parents=True, exist_ok=True)
    cfg = load_config(a.config, archive_dir=Path(a.data_dir).resolve() if a.data_dir else None)
    data, _ = prepare_data(cfg, progress=False, include_test=True)

    pc = per_crop_test(Path(a.runs_dir), a.runs, data)
    gt, kw = group_table(pc), kruskal_table(pc)
    gt.round(4).to_csv(res / "estratificado_test.csv", index=False)
    kw.to_csv(res / "estratificado_kruskal.csv", index=False)
    plot_strata(gt, kw, res / "figures" / "estratificado_test.png")
    print(gt.round(3).to_string(index=False), "\n")
    print(kw.round(5).to_string(index=False))
    print(f"\nRecortes con las 3 semillas vacías: {(pc['semillas_vacias'] == len(a.runs)).sum()}"
          f" | con alguna vacía: {(pc['semillas_vacias'] > 0).sum()}")


if __name__ == "__main__":
    main()
