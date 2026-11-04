"""Descarga y verifica el dataset CBIS-DDSM (versión de Kaggle de awsaf49).

* Si el dataset ya está en la carpeta de destino, solo verifica su estructura.
* Si no, y hay credenciales de Kaggle (``~/.kaggle/kaggle.json`` o las variables
  ``KAGGLE_USERNAME``/``KAGGLE_KEY``) y el paquete ``kaggle`` está instalado,
  lo descarga y descomprime (~5 GB).
* En otro caso, imprime instrucciones para la descarga manual.

Destino (de mayor a menor prioridad): ``--dest`` > variable de entorno
``MAMO_DATA_DIR`` > ``data/CBIS-DDSM`` dentro del repositorio.

Uso:
    python scripts/download_data.py                 # descarga si hace falta
    python scripts/download_data.py --check-only    # solo verifica
"""
import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.config import DATA_DIR_ENV  # noqa: E402

KAGGLE_DATASET = "awsaf49/cbis-ddsm-breast-cancer-image-dataset"
KAGGLE_URL = f"https://www.kaggle.com/datasets/{KAGGLE_DATASET}"
DEFAULT_DEST = REPO / "data" / "CBIS-DDSM"
REQUIRED_FILES = ["csv/mass_case_description_train_set.csv", "csv/mass_case_description_test_set.csv"]

INSTRUCTIONS = f"""
No se pudo descargar automáticamente. Opciones:

  A) Con la API de Kaggle:
       1. pip install kaggle
       2. En https://www.kaggle.com/settings -> "Create New Token" (descarga kaggle.json)
       3. Copia kaggle.json a  %USERPROFILE%\\.kaggle\\kaggle.json  (Windows)
                               ~/.kaggle/kaggle.json             (Linux/Mac)
       4. Vuelve a ejecutar:  python scripts/download_data.py

  B) Manualmente:
       1. Descarga el ZIP desde {KAGGLE_URL}
       2. Descomprímelo en  {{dest}}
          (debe quedar {{dest}}/csv/ y {{dest}}/jpeg/)
       3. Si lo dejas en otra carpeta, indica la ruta con la variable de entorno {DATA_DIR_ENV}:
            PowerShell:  $env:{DATA_DIR_ENV} = "D:\\datos\\CBIS-DDSM"     (sesión actual)
                         setx {DATA_DIR_ENV} "D:\\datos\\CBIS-DDSM"       (permanente)
            bash:        export {DATA_DIR_ENV}=/ruta/a/CBIS-DDSM

Más detalles en docs/DATA.md.
"""


def default_dest() -> Path:
    env = os.environ.get(DATA_DIR_ENV)
    return Path(env).expanduser() if env else DEFAULT_DEST


def check_structure(dest: Path) -> list:
    """Lista de problemas encontrados (vacía si la estructura es la esperada)."""
    dest = Path(dest)
    problems = [f"falta {f}" for f in REQUIRED_FILES if not (dest / f).is_file()]
    jpeg = dest / "jpeg"
    if not jpeg.is_dir():
        problems.append("falta la carpeta jpeg/")
    elif not any(p.is_dir() for p in jpeg.iterdir()):
        problems.append("la carpeta jpeg/ está vacía")
    return problems


def kaggle_credentials_available(env=None, home=None) -> bool:
    env = os.environ if env is None else env
    if env.get("KAGGLE_USERNAME") and env.get("KAGGLE_KEY"):
        return True
    config_dir = Path(env["KAGGLE_CONFIG_DIR"]) if env.get("KAGGLE_CONFIG_DIR") else \
        Path(home if home is not None else Path.home()) / ".kaggle"
    return (config_dir / "kaggle.json").is_file()


def download_with_kaggle(dest: Path) -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi  # import diferido: paquete opcional
    api = KaggleApi()
    api.authenticate()
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Descargando {KAGGLE_DATASET} en {dest} (~5 GB, puede tardar)...")
    api.dataset_download_files(KAGGLE_DATASET, path=str(dest), unzip=True, quiet=False)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Descarga y verifica el dataset CBIS-DDSM.")
    p.add_argument("--dest", type=Path, help=f"carpeta destino (por defecto ${DATA_DIR_ENV} o data/CBIS-DDSM)")
    p.add_argument("--check-only", action="store_true", help="solo verifica, no descarga")
    a = p.parse_args(argv)
    dest = (a.dest or default_dest()).resolve()

    problems = check_structure(dest)
    if not problems:
        print(f"Dataset OK en {dest}")
        return 0
    print(f"Dataset no disponible en {dest}: " + "; ".join(problems))
    if a.check_only:
        return 1

    try:
        import kaggle  # noqa: F401  (solo para saber si está instalado)
        has_pkg = True
    except Exception:  # ImportError, o error de autenticación al importar
        has_pkg = False
    if has_pkg and kaggle_credentials_available():
        try:
            download_with_kaggle(dest)
        except Exception as e:  # red, credenciales inválidas, términos no aceptados...
            print(f"Error al descargar con la API de Kaggle: {e}")
        else:
            problems = check_structure(dest)
            if not problems:
                print(f"Dataset OK en {dest}")
                return 0
            print("La descarga terminó pero la estructura no es la esperada: " + "; ".join(problems))
    elif not has_pkg:
        print("El paquete 'kaggle' no está instalado.")
    else:
        print("No se encontraron credenciales de Kaggle.")
    print(INSTRUCTIONS.format(dest=dest))
    return 1


if __name__ == "__main__":
    sys.exit(main())
