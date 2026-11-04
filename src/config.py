"""Configuración central del proyecto.

Todos los parámetros que afectan a los experimentos viven aquí, de modo que un
experimento queda completamente descrito por un objeto ``Config``.
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # --- rutas -------------------------------------------------------------
    archive_dir: Path = Path("archive")   # carpeta descomprimida del dataset de Kaggle
    out_dir: Path = Path("outputs")       # aquí se guarda todo lo generado

    # --- preprocesamiento --------------------------------------------------
    img_size: int = 256     # lado (px) de los recortes ROI que ve la red
    context: float = 3.0    # lado del recorte = context * lado mayor de la caja de la lesión
    jitter: float = 0.2     # desplazamiento aleatorio (fracción del lado) del centro del recorte
    min_side: int = 96      # lado mínimo (px de la mamografía original) del recorte
    val_frac: float = 0.2   # fracción de PACIENTES de train oficial usada como validación
    seed: int = 42

    # --- modelo / entrenamiento -------------------------------------------
    base_channels: int = 32   # canales de la 1ª capa de la U-Net (16 si solo hay CPU)
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
        return self.out_dir / "processed"

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
