"""Controles de E13 (no entrena): preentrenamiento de SegFormer-B0 y lr de la U-Net.

1) Control de lr de la U-Net (``--unet-lr-runs``): Dice de VALIDACIÓN con semilla 42 para
   cada lr (solo validación; no se evalúa test). La U-Net de referencia no cambia.
2) U-Net vs SegFormer-B0 preentrenado vs SegFormer-B0 sin preentrenar (pesos aleatorios,
   mismo lr 3e-4): métricas en val y test (media ± DE de 3 semillas), diferencias
   pareadas (Wilcoxon e IC bootstrap por paciente) de (sin preentrenar − U-Net) y
   (preentrenado − sin preentrenar), y fallos por modo (criterio de E11).
El test solo se informa.
Uso:
    python scripts/compare_segformer_controls.py --unet-runs main final_seed43 final_seed44 \\
        --segformer-runs segformer_lr3e-4 segformer_s43 segformer_s44 \\
        --scratch-runs segformer_scratch_s42 segformer_scratch_s43 segformer_scratch_s44 \\
        --unet-lr-runs main unet_lr3e-4
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

import compare_morphology as cm                                 # noqa: E402
import compare_segformer as cs                                  # noqa: E402
from src.config import load_config                              # noqa: E402
from src.pipeline import cached_masks, prepare_data, run_config  # noqa: E402

UNET, SEG = cs.UNET, cs.SEG
SEG_S = "SegFormer-B0 sin preentrenar"
COLORS = {UNET: cs.COLORS[UNET], SEG: cs.COLORS[SEG], SEG_S: "#9a6b2f"}
SPLITS = cs.SPLITS


def history_row(runs_dir: Path, run: str) -> dict:
    m = json.loads((runs_dir / run / "metrics.json").read_text(encoding="utf-8"))
    c = run_config(runs_dir / run)
    h = pd.read_csv(runs_dir / run / "history.csv")
    return {"ejecucion": run, "arch": c.arch, "pretrained": c.pretrained if c.arch == "segformer_b0" else "—",
            "lr": c.lr, "semilla": cs.seed_of(runs_dir, run), "dice_val": m["segmentacion"]["U-Net | val"]["dice"],
            "epocas": m["epocas_entrenadas"], "mejor_epoca": m["mejor_epoca"],
            "segundos_por_epoca_mediana": float(h["seconds"].median()),
            "hay_nan": bool(h[["train_loss", "val_dice"]].isna().any().any())}


def decomposition(summary: pd.DataFrame, tests: pd.DataFrame) -> pd.DataFrame:
    """Desglose en tres escalones del Dice: U-Net → sin preentrenar (arquitectura) → preentrenado.

    Las dos diferencias pareadas suman la ventaja total; la fracción es la de la estimación puntual.
    """
    rows = []
    for split in SPLITS:
        s = summary[summary["conjunto"] == split].set_index("método")["dice_media"]
        t = tests[(tests["conjunto"] == split) & (tests["métrica"] == "dice")].set_index("comparación")
        total = s[SEG] - s[UNET]
        rows.append({"conjunto": split, "escalón": "U-Net", "dice_media": s[UNET]})
        for label, comp in (("arquitectura (sin preentrenar − U-Net)", f"{SEG_S} − {UNET}"),
                            ("preentrenamiento (preentrenado − sin preentrenar)", f"{SEG} − {SEG_S}")):
            r = t.loc[comp]
            rows.append({"conjunto": split, "escalón": label,
                         "dice_media": s[SEG_S] if label.startswith("arq") else s[SEG],
                         "delta": r["diferencia_media"], "ic95_inf_pacientes": r["ic95_inf_pacientes"],
                         "ic95_sup_pacientes": r["ic95_sup_pacientes"], "wilcoxon_p": r["wilcoxon_p"],
                         "fraccion_de_la_ventaja": r["diferencia_media"] / total})
    return pd.DataFrame(rows)


def plot_curves(runs_dir: Path, groups: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for name, runs in groups.items():
        for k, r in enumerate(runs):
            h = pd.read_csv(runs_dir / r / "history.csv")
            ax.plot(h["epoch"], h["val_dice"], color=COLORS[name], lw=1.3, alpha=0.9,
                    label=name if k == 0 else None)
    ax.set_xlabel("época")
    ax.set_ylabel("Dice de validación")
    ax.set_ylim(0.4, 0.85)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#e5e5e5", lw=0.6)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title("Curvas de validación (3 semillas por modelo)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description="Controles de E13: preentrenamiento y lr.")
    p.add_argument("--unet-runs", nargs="+", required=True)
    p.add_argument("--segformer-runs", nargs="+", required=True)
    p.add_argument("--scratch-runs", nargs="+", required=True)
    p.add_argument("--unet-lr-runs", nargs="*", default=[])
    p.add_argument("--config", default=str(REPO / "configs" / "main.yaml"))
    p.add_argument("--runs-dir", default=str(REPO / "outputs" / "runs"))
    p.add_argument("--results-dir", default=str(REPO / "results"))
    p.add_argument("--n-boot", type=int, default=1000)
    a = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    runs_dir, res = Path(a.runs_dir), Path(a.results_dir)
    (res / "figures").mkdir(parents=True, exist_ok=True)

    # (1) control de lr de la U-Net: solo validación
    if a.unet_lr_runs:
        lr = pd.DataFrame([history_row(runs_dir, r) for r in a.unet_lr_runs])
        lr.to_csv(res / "unet_control_lr.csv", index=False)
        print(lr.round(4).to_string(index=False), "\n")
    runs_info = pd.DataFrame([history_row(runs_dir, r) for r in a.segformer_runs + a.scratch_runs])
    runs_info.to_csv(res / "segformer_controles_entrenamiento.csv", index=False)
    print(runs_info.round(4).to_string(index=False), "\n")

    # (2) comparación (reutiliza las funciones de compare_segformer con otros métodos y pares)
    cs.METHODS = [UNET, SEG, SEG_S]
    cs.PAIRS = [(SEG_S, UNET), (SEG, SEG_S)]
    cfg = load_config(a.config)
    data, _ = prepare_data(cfg, progress=False, include_test=True)
    by_method = {}
    for name, runs in ((UNET, a.unet_runs), (SEG, a.segformer_runs), (SEG_S, a.scratch_runs)):
        for r in runs:
            by_method[(name, cs.seed_of(runs_dir, r))] = cached_masks(runs_dir / r, data, SPLITS)
    pc = cm.per_crop(by_method, data)
    summary, per_seed = cs.summarize(pc)
    summary.round(4).to_csv(res / "segformer_controles_comparacion.csv", index=False)
    per_seed.round(4).to_csv(res / "segformer_controles_por_semilla.csv", index=False)
    tests = cs.paired_tests(pc, a.n_boot, cfg.seed)
    tests.to_csv(res / "segformer_controles_pruebas.csv", index=False)
    steps = decomposition(summary, tests)
    steps.to_csv(res / "segformer_controles_desglose.csv", index=False)
    print(steps.round(4).to_string(index=False), "\n")
    counts, agg, ftests = cs.failure_tables(cs.failure_modes(by_method, data))
    agg.round(3).to_csv(res / "segformer_controles_fallos.csv", index=False)
    ftests.to_csv(res / "segformer_controles_fallos_pruebas.csv", index=False)
    plot_curves(runs_dir, {UNET: a.unet_runs, SEG: a.segformer_runs, SEG_S: a.scratch_runs},
                res / "figures" / "segformer_controles_curvas.png")

    cols = ["método", "conjunto", "dice_media", "dice_de", "iou_media", "precision_media", "recall_media",
            "hd95_media", "hd95_mediana_media", "n_vacias_media", "n_dice_menor_01_media"]
    print(summary[cols].round(4).to_string(index=False), "\n")
    print(tests.round(4).to_string(index=False), "\n")
    print(agg.round(2).to_string(index=False), "\n")
    print(ftests.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
