"""Punto de entrada reproducible del experimento principal.

Preprocesa (con caché), entrena la U-Net, evalúa la U-Net y los métodos de
referencia en validación y test, analiza la forma BI-RADS y guarda todo en
``outputs/runs/<name>/`` (config.yaml, metrics.json, per_sample.csv,
history.csv, pesos, tablas y figuras).

Ejemplos (desde la raíz del repositorio):
    python scripts/run_experiment.py --config configs/main.yaml
    python scripts/run_experiment.py --config configs/quick.yaml
    python scripts/run_experiment.py --config configs/main.yaml --seed 7 --name main_seed7
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # sin ventanas: las figuras solo se guardan

from src.cli import base_parser, config_from_args  # noqa: E402
from src.pipeline import run_experiment            # noqa: E402


def parse_args(argv=None):
    p = base_parser("Entrena y evalúa la U-Net y los métodos de referencia.")
    p.add_argument("--seed", type=int, help="sobrescribe la semilla")
    p.add_argument("--epochs", type=int, help="sobrescribe el número máximo de épocas")
    p.add_argument("--no-figures", action="store_true", help="no generar figuras")
    return p.parse_args(argv)


def main(argv=None) -> dict:
    a = parse_args(argv)
    cfg = config_from_args(a, seed=a.seed, epochs=a.epochs)
    print(f"Experimento '{cfg.name}' | dataset: {cfg.archive_dir} | salida: {cfg.run_dir}")
    return run_experiment(cfg, figures=not a.no_figures)


if __name__ == "__main__":
    main()
