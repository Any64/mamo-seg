"""Ablaciones del experimento principal, evaluadas SOLO en validación.

Lee ``configs/ablations.yaml`` (config base + variantes), entrena cada variante con
``pipeline.run_experiment(evaluate_test=False)`` en ``outputs/runs/abl_<variante>/``
(reanudable: una variante con metrics.json no se repite) y reúne Dice/IoU de la
U-Net, del prior centrado y del método clásico en validación en
``results/ablaciones.csv`` y ``results/figures/ablaciones.png``.

Uso:
    python scripts/run_ablations.py --config configs/ablations.yaml
    python scripts/run_ablations.py --only context2 jitter0     # algunas variantes
    python scripts/run_ablations.py --collect-only              # solo reunir resultados
"""
import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import pandas as pd  # noqa: E402
import yaml          # noqa: E402

from src.config import Config, load_config            # noqa: E402
from src.pipeline import CLASSICAL, PRIOR, UNET, run_experiment  # noqa: E402
from src.viz import plot_ablations                     # noqa: E402

# parámetros que deben coincidir para reutilizar una ejecución existente
MATCH_KEYS = ("img_size", "context", "jitter", "min_side", "val_frac", "seed", "base_channels", "loss",
              "batch_size", "epochs", "lr", "weight_decay", "patience", "threshold", "amp")


def same_config(run_dir: Path, cfg: Config) -> bool:
    path = run_dir / "config.yaml"
    if not (run_dir / "metrics.json").exists() or not path.exists():
        return False
    saved = {**Config().to_dict(), **yaml.safe_load(path.read_text(encoding="utf-8"))}  # claves nuevas -> defecto
    if saved.get("train_seed") not in (None, cfg.seed):
        return False
    return all(saved[k] == getattr(cfg, k) for k in MATCH_KEYS)


def collect(spec: dict, run_names: dict, out_dir: Path) -> pd.DataFrame:
    rows = []
    for group, g in spec["groups"].items():
        for v in g["variants"]:
            run_dir = Path(load_config(REPO / spec["base"]).out_dir) / "runs" / run_names[v]
            m = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
            seg = m["segmentacion"]
            rows.append({
                "grupo": group, "variante": v, "parametro": g["param"],
                "valor": cfg.get(g["param"], Config().to_dict()[g["param"]]),
                "es_referencia": v == "ref", "ejecucion": run_names[v],
                "dice_unet_val": seg[f"{UNET} | val"]["dice"], "dice_unet_val_de": seg[f"{UNET} | val"]["dice_std"],
                "iou_unet_val": seg[f"{UNET} | val"]["iou"],
                "dice_prior_val": seg[f"{PRIOR} | val"]["dice"], "iou_prior_val": seg[f"{PRIOR} | val"]["iou"],
                "dice_clasico_val": seg[f"{CLASSICAL} | val"]["dice"],
                "frac_dice_lt_0_1_val": m.get("unet_val_frac_dice_lt_0_1"),
                "epocas": m["epocas_entrenadas"], "mejor_epoca": m["mejor_epoca"],
                "segundos_por_epoca": m.get("segundos_por_epoca",
                                            round(m["segundos_entrenamiento"] / m["epocas_entrenadas"], 1)),
            })
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "ablaciones.csv", index=False)
    plot_ablations(df, out_dir / "figures" / "ablaciones.png")
    return df


def main(argv=None):
    p = argparse.ArgumentParser(description="Ablaciones evaluadas solo en validación.")
    p.add_argument("--config", default="configs/ablations.yaml")
    p.add_argument("--only", nargs="*", help="variantes a ejecutar (por defecto todas)")
    p.add_argument("--collect-only", action="store_true")
    p.add_argument("--data-dir", help="carpeta del dataset; prioridad sobre MAMO_DATA_DIR")
    p.add_argument("--results-dir", default=str(REPO / "results"))
    a = p.parse_args(argv)
    spec = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    data_dir = Path(a.data_dir).resolve() if a.data_dir else None

    run_names = {}
    for v, over in spec["variants"].items():
        cfg = load_config(REPO / spec["base"], archive_dir=data_dir, name=f"abl_{v}", **over)
        reuse = spec.get("reuse", {}).get(v)
        if reuse and same_config(cfg.out_dir / "runs" / reuse, cfg):
            run_names[v] = reuse
            continue
        run_names[v] = cfg.name
        if a.collect_only or (a.only and v not in a.only) or (cfg.run_dir / "metrics.json").exists():
            continue
        t0 = time.time()
        print(f"\n=== Variante {v}: {over or 'referencia'} -> {cfg.run_dir} ===", flush=True)
        run_experiment(cfg, evaluate_test=False)
        print(f"=== {v} terminada en {(time.time() - t0) / 60:.1f} min ===", flush=True)

    missing = [v for v, n in run_names.items()
               if not (load_config(REPO / spec["base"]).out_dir / "runs" / n / "metrics.json").exists()]
    if missing:
        print("Variantes sin resultados todavía:", missing)
        return None
    df = collect(spec, run_names, Path(a.results_dir))
    print(df[["variante", "valor", "dice_unet_val", "iou_unet_val", "dice_prior_val", "epocas"]].to_string(index=False))
    return df


if __name__ == "__main__":
    main()
