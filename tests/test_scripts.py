"""Pruebas de los puntos de entrada (scripts/) y de la configuración, con un dataset sintético y sin red.

Ejecutar desde la raíz del repositorio:   python -m pytest -q tests
"""
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DATA_DIR_ENV, load_config               # noqa: E402
from tests.make_synthetic_archive import make_archive         # noqa: E402


def load_script(name: str):
    """Importa scripts/<name>.py como módulo (scripts/ no es un paquete)."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def archive(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "archive"
    make_archive(path, n_train_patients=30, n_test_patients=10)
    return path


@pytest.fixture
def tiny_config(tmp_path):
    """Config pequeña (recortes de 64 px, U-Net mínima) para que la prueba corra en segundos."""
    cfg = {"name": "tiny", "data": {"archive_dir": "no/existe", "out_dir": "outputs"},
           "preprocessing": {"img_size": 64, "context": 3.0, "jitter": 0.2},
           "model": {"base_channels": 8},
           "training": {"seed": 0, "epochs": 2, "batch_size": 8, "patience": 5, "amp": False}}
    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- configuración
def test_main_config_reproduces_reported_setup(monkeypatch):
    monkeypatch.delenv(DATA_DIR_ENV, raising=False)
    cfg = load_config(ROOT / "configs" / "main.yaml")
    assert (cfg.img_size, cfg.context, cfg.jitter) == (256, 3.0, 0.2)
    assert (cfg.base_channels, cfg.batch_size, cfg.epochs, cfg.seed) == (16, 8, 40, 42)
    assert cfg.amp is False and cfg.val_frac == 0.2
    assert cfg.archive_dir.is_absolute() and cfg.archive_dir.is_relative_to(ROOT)


def test_data_dir_priority(monkeypatch, tmp_path):
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path / "env"))
    assert load_config(ROOT / "configs" / "main.yaml").archive_dir == tmp_path / "env"
    cfg = load_config(ROOT / "configs" / "main.yaml", archive_dir=tmp_path / "cli", epochs=3)
    assert cfg.archive_dir == tmp_path / "cli" and cfg.epochs == 3


def test_unknown_config_key_is_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("training:\n  epocs: 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="epocs"):
        load_config(bad)


# --------------------------------------------------------------------------- run_experiment
def test_run_experiment_end_to_end(archive, tiny_config, tmp_path):
    pytest.importorskip("torch")
    run = load_script("run_experiment")
    out = tmp_path / "out"
    metrics = run.main(["--config", str(tiny_config), "--data-dir", str(archive),
                        "--out-dir", str(out), "--epochs", "1", "--name", "prueba"])
    run_dir = out / "runs" / "prueba"
    for f in ("config.yaml", "metrics.json", "per_sample.csv", "history.csv", "train.log", "unet_best.pt",
              "metricas_segmentacion.csv", "metricas_forma.csv", "dice_por_desplazamiento.csv",
              "figures/curvas_entrenamiento.png", "figures/predicciones_test.png",
              "figures/confusion_forma_unet.png"):
        assert (run_dir / f).exists(), f
    # la config guardada refleja los argumentos de la CLI
    saved = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    assert saved["epochs"] == 1 and saved["name"] == "prueba" and saved["img_size"] == 64
    # métricas de los 3 métodos en val y test, y caché de recortes reutilizable
    on_disk = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert on_disk == json.loads(json.dumps(metrics))
    assert metrics["epocas_entrenadas"] == 1
    for m in ("Prior centrado", "Clásico (Otsu+morfología)", "U-Net"):
        for s in ("val", "test"):
            assert 0 <= metrics["segmentacion"][f"{m} | {s}"]["dice"] <= 1
    ps = pd.read_csv(run_dir / "per_sample.csv")
    assert set(ps["método"]) == {"Prior centrado", "Clásico (Otsu+morfología)", "U-Net"}
    assert len(ps) == 3 * (metrics["n"]["val"] + metrics["n"]["test"])
    assert (out / "processed" / "train_s64_c3_j0.2.npz").exists()


def test_run_experiment_without_data_gives_clear_error(tiny_config, tmp_path, monkeypatch):
    monkeypatch.delenv(DATA_DIR_ENV, raising=False)
    run = load_script("run_experiment")
    with pytest.raises(FileNotFoundError, match="MAMO_DATA_DIR"):
        run.main(["--config", str(tiny_config), "--out-dir", str(tmp_path / "out"), "--no-figures"])


# --------------------------------------------------------------------------- download_data (sin red)
@pytest.fixture
def dl(monkeypatch, tmp_path):
    mod = load_script("download_data")
    for var in ("KAGGLE_USERNAME", "KAGGLE_KEY", DATA_DIR_ENV):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path / "sin_credenciales"))

    def no_network(dest):
        raise AssertionError("no debe intentar descargar en las pruebas")
    monkeypatch.setattr(mod, "download_with_kaggle", no_network)
    return mod


def test_download_check_structure(dl, archive, tmp_path):
    assert dl.check_structure(archive) == []
    problems = dl.check_structure(tmp_path / "vacio")
    assert any("mass_case_description_train_set" in p for p in problems)
    assert any("jpeg" in p for p in problems)


def test_download_skips_when_dataset_present(dl, archive, capsys):
    assert dl.main(["--dest", str(archive)]) == 0
    assert "Dataset OK" in capsys.readouterr().out


def test_download_without_credentials_prints_instructions(dl, tmp_path, capsys):
    assert dl.main(["--dest", str(tmp_path / "destino")]) == 1
    out = capsys.readouterr().out
    assert "kaggle.com/datasets/awsaf49" in out and DATA_DIR_ENV in out
    assert dl.main(["--check-only", "--dest", str(tmp_path / "destino")]) == 1


def test_kaggle_credentials_detection(dl, tmp_path):
    assert dl.kaggle_credentials_available({"KAGGLE_USERNAME": "u", "KAGGLE_KEY": "k"}, home=tmp_path)
    assert not dl.kaggle_credentials_available({}, home=tmp_path)
    (tmp_path / ".kaggle").mkdir()
    (tmp_path / ".kaggle" / "kaggle.json").write_text("{}", encoding="utf-8")
    assert dl.kaggle_credentials_available({}, home=tmp_path)


def test_default_destination_uses_env(dl, monkeypatch, tmp_path):
    assert dl.default_dest() == ROOT / "data" / "CBIS-DDSM"
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path))
    assert dl.default_dest() == tmp_path


def test_run_experiment_without_test_split(archive, tiny_config, tmp_path):
    """Modo de selección de modelo: el test oficial no se genera ni se evalúa."""
    pytest.importorskip("torch")
    run = load_script("run_experiment")
    out = tmp_path / "out"
    metrics = run.main(["--config", str(tiny_config), "--data-dir", str(archive), "--out-dir", str(out),
                        "--epochs", "1", "--no-test", "--name", "sin_test"])
    assert metrics["test_evaluado"] is False
    assert not any("test" in k for k in metrics["segmentacion"])
    assert not list((out / "processed").glob("test_*"))
    assert (out / "runs" / "sin_test" / "figures" / "predicciones_val.png").exists()


def test_parallel_preprocessing_is_identical(archive):
    """Los recortes generados en paralelo son idénticos a los secuenciales."""
    import numpy as np
    from src.data_index import load_case_csv
    from src.preprocess import build_split
    df = load_case_csv(archive / "csv", "test")
    a = build_split(df, archive / "jpeg", 64, 3.0, 96, jitter=0.2, seed=1, progress=False, workers=1)
    b = build_split(df, archive / "jpeg", 64, 3.0, 96, jitter=0.2, seed=1, progress=False, workers=2)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and a[2].equals(b[2])


def test_loss_selection():
    torch = pytest.importorskip("torch")
    from src.losses import get_loss
    logits, target = torch.zeros(2, 1, 8, 8), torch.ones(2, 1, 8, 8)
    assert get_loss("dice")(logits, target) < get_loss("bce_dice")(logits, target)
    with pytest.raises(ValueError):
        get_loss("focal")


def test_cross_validation_groups_by_patient(archive, tmp_path):
    pytest.importorskip("torch")
    from src.config import Config
    from src.pipeline import cross_validate
    cfg = Config(archive_dir=archive, out_dir=tmp_path / "out", img_size=64, epochs=1,
                 base_channels=8, batch_size=8, name="cv")
    df = cross_validate(cfg, n_splits=3, verbose=False)
    from src.data_index import load_case_csv
    n_rows = len(load_case_csv(archive / "csv", "train"))
    assert len(df) == 3 and df["n_val"].sum() == n_rows      # cada recorte se valida una vez
    assert df[["dice_unet", "dice_prior"]].apply(lambda c: c.between(0, 1)).all().all()
    assert (tmp_path / "out" / "runs" / "cv_cv3" / "cv_summary.json").exists()


def test_bootstrap_and_wilcoxon():
    import numpy as np
    from src.stats import bootstrap_mean_ci, paired_wilcoxon
    rng = np.random.default_rng(0)
    a = rng.uniform(0.6, 0.9, 200)
    b = a - 0.3 + rng.normal(0, 0.05, 200)
    ci = bootstrap_mean_ci(a, n_boot=500, seed=1)
    assert ci["ic_inf"] < a.mean() < ci["ic_sup"] and ci["ic_sup"] - ci["ic_inf"] < 0.05
    groups = np.repeat(np.arange(100), 2)                 # 2 recortes por paciente
    ci_g = bootstrap_mean_ci(a, n_boot=500, seed=1, groups=groups)
    assert ci_g["ic_inf"] < a.mean() < ci_g["ic_sup"]
    assert bootstrap_mean_ci(a, n_boot=500, seed=1) == ci      # reproducible
    w = paired_wilcoxon(a, b)
    assert w["p_valor"] < 1e-10 and w["frac_a_mejor"] > 0.95 and abs(w["media_diferencia"] - 0.3) < 0.02
