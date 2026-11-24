"""Sensibilidad del posprocesamiento fijo al radio (análisis posterior; no entrena ni elige).

El radio r = 6 se eligió en validación (``compare_morphology.py``). Este script solo
describe qué pasa con otros radios, en validación y en test. El test es descriptivo y
no se usa para elegir nada.

1) Por radio (sin posproc., 0, 1, 2, 3, 4, 6, 8), conjunto y semilla: Dice medio, HD95
   mediano, predicciones vacías y recortes con Dice < 0.1 -> results/posproc_sensibilidad.csv
2) Recortes donde r = 6 lleva el Dice a < 0.1 y la U-Net no (causas en
   ``src.postprocess.drop_cause``) -> results/posproc_caidas.csv,
   results/posproc_caidas_resumen.csv y results/figures/posproc_caidas_ejemplos.png

Uso:  python scripts/posproc_sensitivity.py --unet-runs main final_seed43 final_seed44
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

from src.config import load_config                                  # noqa: E402
from src.metrics import extended_metrics                            # noqa: E402
from src.pipeline import cached_masks, prepare_data, run_config     # noqa: E402
from src.postprocess import (CAUSE_COMPONENT, CAUSE_EMPTY, CAUSE_OTHER,  # noqa: E402
                             DROP_THRESHOLD, drop_cause, postprocess_stack)

SPLITS = ("val", "test")
RADII = (None, 0, 1, 2, 3, 4, 6, 8)   # None = sin posprocesamiento
CHOSEN = 6                            # elegido en validación (results/morfologia_decision.json)
CAUSES = (CAUSE_EMPTY, CAUSE_COMPONENT, CAUSE_OTHER)
GT_COLOR, UNET_COLOR, POST_COLOR = "#f2c200", "#2a78d6", "#1baf7a"


def label(r) -> str:
    return "sin posproc." if r is None else str(r)


def sensitivity(runs_dir: Path, runs: list, data: dict):
    rows, per_crop, masks = [], {}, {}
    for run in runs:
        rc = run_config(runs_dir / run)
        seed = rc.seed if rc.train_seed is None else rc.train_seed
        m = cached_masks(runs_dir / run, data, SPLITS)
        for s in SPLITS:
            for r in RADII:
                pred = m[s] if r is None else postprocess_stack(m[s], r)
                met = extended_metrics(pred, data[s].Y)
                masks[(seed, s, r)], per_crop[(seed, s, r)] = pred, met
                rows.append({"conjunto": s, "radio": label(r), "semilla": str(seed),
                             "uso": "elección (r = 6)" if s == "val" else "solo descriptivo",
                             "dice_media": met.dice.mean(), "hd95_mediana": met.hd95.median(),
                             "n_vacias": int(met.vacia.sum()),
                             "n_dice_menor_0.1": int((met.dice < DROP_THRESHOLD).sum()),
                             "n": len(met)})
        print(f"{run} (semilla {seed}) listo", flush=True)
    df = pd.DataFrame(rows)
    num = ["dice_media", "hd95_mediana", "n_vacias", "n_dice_menor_0.1"]
    g = df.groupby(["conjunto", "radio"], sort=False)
    avg = g[num].mean().join(g[num].std().add_suffix("_de")).reset_index()
    avg["semilla"] = "media"
    avg["uso"] = avg.conjunto.map({"val": "elección (r = 6)", "test": "solo descriptivo"})
    avg["n"] = avg.conjunto.map({s: len(data[s].Y) for s in SPLITS})
    out = pd.concat([df, avg], ignore_index=True)
    out["_s"] = out.conjunto.map({s: k for k, s in enumerate(SPLITS)})
    out["_r"] = out.radio.map({label(r): k for k, r in enumerate(RADII)})
    out = out.sort_values(["_s", "_r", "semilla"]).drop(columns=["_s", "_r"]).round(4)
    return out, per_crop, masks


def drops_table(per_crop: dict, data: dict, seeds: list) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        for s in SPLITS:
            u, r0, rr = (per_crop[(seed, s, k)] for k in (None, 0, CHOSEN))
            for i in range(len(u)):
                c = drop_cause(u.dice[i], r0.dice[i], rr.dice[i], bool(rr.vacia[i]))
                if c:
                    rows.append({"conjunto": s, "semilla": seed, "row_idx": i,
                                 "patient_id": data[s].meta.iloc[i].patient_id, "causa": c,
                                 "dice_unet": round(float(u.dice[i]), 3),
                                 "dice_r0": round(float(r0.dice[i]), 3),
                                 f"dice_r{CHOSEN}": round(float(rr.dice[i]), 3),
                                 "componentes_unet": int(u.n_componentes[i]),
                                 "area_anotada": int(data[s].Y[i].sum())})
    return pd.DataFrame(rows, columns=["conjunto", "semilla", "row_idx", "patient_id", "causa",
                                       "dice_unet", "dice_r0", f"dice_r{CHOSEN}",
                                       "componentes_unet", "area_anotada"])


def drops_summary(dr: pd.DataFrame, seeds: list) -> pd.DataFrame:
    rows = []
    for s in SPLITS:
        for c in CAUSES:
            sub = dr[(dr.conjunto == s) & (dr.causa == c)]
            n = [int((sub.semilla == x).sum()) for x in seeds]
            rows.append({"conjunto": s, "causa": c, **{f"semilla_{x}": k for x, k in zip(seeds, n)},
                         "total": sum(n), "media_por_semilla": round(float(np.mean(n)), 2),
                         "recortes_distintos": int(sub.row_idx.nunique())})
    return pd.DataFrame(rows)


def plot_drops(dr: pd.DataFrame, data: dict, masks: dict, path: Path) -> None:
    present = [c for c in CAUSES if (dr.causa == c).any()]
    if not present:
        return
    fig, axes = plt.subplots(len(present), 6, figsize=(16, 3.0 * len(present)), squeeze=False)
    for k, c in enumerate(present):
        sub = (dr[dr.causa == c].assign(_o=lambda d: d.conjunto.map({"val": 0, "test": 1}))
               .sort_values(["_o", "dice_unet"], ascending=[True, False])
               .drop_duplicates(["conjunto", "row_idx"]).head(6))
        for j in range(6):
            ax = axes[k, j]
            ax.set_xticks([]); ax.set_yticks([])
            if j >= len(sub):
                ax.axis("off")
                continue
            e = sub.iloc[j]
            s, i, seed = e.conjunto, int(e.row_idx), int(e.semilla)
            ax.imshow(data[s].X[i], cmap="gray")
            for msk, col, lw in ((data[s].Y[i], GT_COLOR, 1.4), (masks[(seed, s, None)][i], UNET_COLOR, 1.0),
                                 (masks[(seed, s, CHOSEN)][i], POST_COLOR, 1.4)):
                if msk.any():
                    ax.contour(msk, levels=[0.5], colors=col, linewidths=lw)
            ax.set_title(f"{s} #{i} (semilla {seed})\nDice U-Net {e.dice_unet:.2f} | r0 {e.dice_r0:.2f} | "
                         f"r{CHOSEN} {e[f'dice_r{CHOSEN}']:.2f}", fontsize=8)
        axes[k, 0].set_ylabel(c, fontsize=9)
    handles = [plt.Line2D([], [], color=col, lw=2) for col in (GT_COLOR, UNET_COLOR, POST_COLOR)]
    fig.legend(handles, ["anotación", "U-Net", f"U-Net + posproc. (r = {CHOSEN})"],
               loc="lower center", ncol=3, frameon=False)
    fig.suptitle(f"Recortes donde el posprocesamiento (r = {CHOSEN}) baja el Dice a < 0.1, por causa")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(REPO / "configs" / "main.yaml"))
    ap.add_argument("--unet-runs", nargs="+", default=["main", "final_seed43", "final_seed44"])
    ap.add_argument("--runs-dir", default=str(REPO / "outputs" / "runs"))
    ap.add_argument("--results", default=str(REPO / "results"))
    a = ap.parse_args(argv)
    runs_dir, res = Path(a.runs_dir), Path(a.results)
    (res / "figures").mkdir(parents=True, exist_ok=True)

    data, _ = prepare_data(load_config(a.config), progress=False, include_test=True)
    sens, per_crop, masks = sensitivity(runs_dir, a.unet_runs, data)
    sens.to_csv(res / "posproc_sensibilidad.csv", index=False)
    print(sens[sens.semilla == "media"].drop(columns=["uso", "n", "semilla"]).to_string(index=False))

    seeds = sorted({k[0] for k in per_crop})
    dr = drops_table(per_crop, data, seeds)
    dr.to_csv(res / "posproc_caidas.csv", index=False)
    summ = drops_summary(dr, seeds)
    summ.to_csv(res / "posproc_caidas_resumen.csv", index=False)
    print(summ.to_string(index=False))
    plot_drops(dr, data, masks, res / "figures" / "posproc_caidas_ejemplos.png")


if __name__ == "__main__":
    main()
