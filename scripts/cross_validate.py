"""Validación cruzada AGRUPADA POR PACIENTE (GroupKFold) sobre el train oficial.

Entrena la U-Net en cada partición (con la configuración dada) y mide Dice/IoU de
la U-Net y del prior centrado en la partición retenida. El test oficial no se usa.
Reanudable: las particiones terminadas no se repiten.

Salida: ``outputs/runs/<name>_cv<k>/`` (cv_folds.csv, cv_summary.json, train.log).
Uso:    python scripts/cross_validate.py --config configs/main.yaml --folds 5
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cli import base_parser, config_from_args  # noqa: E402
from src.pipeline import cross_validate            # noqa: E402


def main(argv=None):
    p = base_parser(__doc__.split("\n")[0])
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, help="sobrescribe el número máximo de épocas")
    a = p.parse_args(argv)
    cfg = config_from_args(a, epochs=a.epochs)
    df = cross_validate(cfg, n_splits=a.folds)
    print(df.round(4).to_string(index=False))
    print(f"\nDice U-Net: {df.dice_unet.mean():.4f} ± {df.dice_unet.std(ddof=1):.4f} | "
          f"prior: {df.dice_prior.mean():.4f} ± {df.dice_prior.std(ddof=1):.4f}")
    return df


if __name__ == "__main__":
    main()
