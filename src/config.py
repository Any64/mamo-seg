"""Configuración central del proyecto.

Todos los parámetros que afectan a los experimentos viven aquí, de modo que un
experimento queda completamente descrito por un objeto ``Config``. Los
experimentos se describen en archivos YAML (``configs/``) que se cargan con
``load_config``.
"""
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR_ENV = "MAMO_DATA_DIR"   # variable de entorno con la ruta del dataset


@dataclass
class Config:
    # --- identificación y rutas -------------------------------------------
    name: str = "main"                    # nombre del experimento -> outputs/runs/<name>/
    archive_dir: Path = Path("archive")   # carpeta descomprimida del dataset de Kaggle (csv/, jpeg/)
    out_dir: Path = Path("outputs")       # aquí se guarda todo lo generado

    # --- preprocesamiento --------------------------------------------------
    img_size: int = 256     # lado (px) de los recortes ROI que ve la red
    context: float = 3.0    # lado del recorte = context * lado mayor de la caja de la lesión
    jitter: float = 0.2     # desplazamiento aleatorio (fracción del lado) del centro del recorte
    min_side: int = 96      # lado mínimo (px de la mamografía original) del recorte
    val_frac: float = 0.2   # fracción de PACIENTES de train oficial usada como validación
    seed: int = 42          # semilla de los DATOS (desplazamientos y división train/val) y, por defecto, del entrenamiento
    workers: int = 1        # procesos para generar los recortes (no cambia el resultado)

    # --- modelo / entrenamiento -------------------------------------------
    arch: str = "unet"        # "unet" o "morph_unet" (U-Net + bloque morfológico aprendible, src/morph.py)
    base_channels: int = 32   # canales de la 1ª capa de la U-Net (16 si solo hay CPU)
    morph_channels: int = 4   # morph_unet: canales del bloque morfológico
    morph_kernel: int = 5     # morph_unet: lado del elemento estructurante
    morph_position: str = "head"  # morph_unet: "head" (antes de la cabeza 1x1) o "skip" (salto de máxima resolución)
    loss: str = "bce_dice"    # "bce_dice" (BCE + Dice) o "dice" (solo Dice suave)
    train_seed: int | None = None  # semilla del entrenamiento (pesos, lotes, aumento); None = ``seed``
    batch_size: int = 16
    epochs: int = 40
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10        # early stopping sobre el Dice de validación
    threshold: float = 0.5    # umbral para binarizar la probabilidad predicha
    amp: bool = False         # precisión mixta (fp16). Dejar en False en GPUs GTX 16xx (dan NaN)

    def __post_init__(self):
        self.archive_dir = Path(self.archive_dir)
        self.out_dir = Path(self.out_dir)

    def to_dict(self) -> dict:
        return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()}

    # --- rutas derivadas ----------------------------------------------------
    @property
    def cache_tag(self) -> str:
        """Identifica los parámetros de recorte: si cambian, se regeneran los recortes."""
        return f"s{self.img_size}_c{self.context:g}_j{self.jitter:g}"

    @property
    def csv_dir(self) -> Path:
        return self.archive_dir / "csv"

    @property
    def jpeg_dir(self) -> Path:
        return self.archive_dir / "jpeg"

    @property
    def processed_dir(self) -> Path:
        """Caché de recortes, compartida entre experimentos con los mismos parámetros de recorte."""
        return self.out_dir / "processed"

    @property
    def run_dir(self) -> Path:
        return self.out_dir / "runs" / self.name

    @property
    def models_dir(self) -> Path:
        return self.out_dir / "models"

    @property
    def figures_dir(self) -> Path:
        return self.out_dir / "figures"

    @property
    def tables_dir(self) -> Path:
        return self.out_dir / "tables"

    def make_dirs(self) -> None:
        for d in (self.processed_dir, self.models_dir, self.figures_dir, self.tables_dir):
            d.mkdir(parents=True, exist_ok=True)


def _resolve(path, base: Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def load_config(path, **overrides) -> Config:
    """Carga un YAML de ``configs/`` y aplica ``overrides`` (los valores ``None`` se ignoran).

    El YAML puede agrupar las claves en secciones (``data:``, ``model:``...); se
    aplanan, y cada clave debe ser un campo de ``Config``. Ruta del dataset, de
    mayor a menor prioridad: ``overrides['archive_dir']`` > variable de entorno
    ``MAMO_DATA_DIR`` > ``archive_dir`` del YAML. Las rutas relativas se
    interpretan respecto de la raíz del repositorio.
    """
    import yaml  # importación local: el resto del paquete no depende de PyYAML

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    flat = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            flat.update(value)
        else:
            flat[key] = value
    if os.environ.get(DATA_DIR_ENV):
        flat["archive_dir"] = os.environ[DATA_DIR_ENV]
    flat.update({k: v for k, v in overrides.items() if v is not None})

    valid = {f.name for f in fields(Config)}
    unknown = sorted(set(flat) - valid)
    if unknown:
        raise ValueError(f"Claves desconocidas en {path}: {unknown}. Válidas: {sorted(valid)}")
    for key in ("archive_dir", "out_dir"):
        if key in flat:
            flat[key] = _resolve(flat[key], REPO_ROOT)
    return Config(**flat)
