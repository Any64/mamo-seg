"""Argumentos de línea de comandos comunes a los scripts de ``scripts/``."""
import argparse
from pathlib import Path

from .config import Config, load_config


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default="configs/main.yaml", help="archivo YAML de configs/")
    p.add_argument("--name", help="nombre del experimento (carpeta en outputs/runs/)")
    p.add_argument("--data-dir", help="carpeta del dataset (csv/ y jpeg/); prioridad sobre MAMO_DATA_DIR")
    p.add_argument("--out-dir", help="carpeta de salida (por defecto la de la config)")
    return p


def config_from_args(args, **overrides) -> Config:
    """Config del YAML con los argumentos de la CLI aplicados (rutas de la CLI relativas al cwd)."""
    cwd_path = lambda p: Path(p).resolve() if p else None  # noqa: E731
    return load_config(args.config, name=args.name, archive_dir=cwd_path(args.data_dir),
                       out_dir=cwd_path(args.out_dir), **overrides)
