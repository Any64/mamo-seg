"""Pruebas del pipeline con un dataset sintético.

Ejecutar desde la raíz del repositorio:   python -m pytest -q tests
(o simplemente:                            python tests/test_pipeline.py)
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import Config                                   # noqa: E402
from src.data_index import load_case_csv                        # noqa: E402
from src.preprocess import alignment_contrast, build_split      # noqa: E402
from src.classical import (center_prior, classical_segment_stack,  # noqa: E402
                           fit_center_prior)
from src.metrics import per_sample_metrics                      # noqa: E402
from src import shape as shp                                    # noqa: E402
from tests.make_synthetic_archive import make_archive           # noqa: E402

IMG = 64


@pytest.fixture(scope="module")
def data():
    tmp = Path(tempfile.mkdtemp())
    make_archive(tmp / "archive", n_train_patients=30, n_test_patients=10)
    cfg = Config(archive_dir=tmp / "archive", out_dir=tmp / "out", img_size=IMG, epochs=2,
                 base_channels=8, batch_size=8, patience=5)
    out = {}
    for split in ("train", "test"):
        df = load_case_csv(cfg.csv_dir, split)
        X, Y, meta, status = build_split(df, cfg.jpeg_dir, cfg.img_size, cfg.context,
                                         cfg.min_side, jitter=cfg.jitter, seed=cfg.seed,
                                         progress=False)
        out[split] = (df, X, Y, meta, status)
    return cfg, out


def test_all_rows_resolved(data):
    cfg, d = data
    for split in ("train", "test"):
        df, X, Y, meta, status = d[split]
        assert status["ok"] == len(df), status
        assert X.shape == (len(df), IMG, IMG) and Y.shape == X.shape
        assert (meta["mode"] == "full").all()      # la máscara es del tamaño de la mamografía


def test_masks_are_aligned_with_images(data):
    _, d = data
    _, X, Y, _, _ = d["train"]
    contrast = alignment_contrast(X, Y)
    assert (contrast > 0).mean() > 0.9             # el interior de la máscara es más brillante
    assert 0.02 < Y.mean() < 0.6                   # la lesión ocupa una fracción razonable


def test_jitter_moves_lesion_off_center(data):
    cfg, d = data
    _, X, Y, meta, _ = d["train"]
    # el centroide de la lesión ya no coincide siempre con el centro del recorte
    cy = np.array([np.argwhere(y).mean(axis=0)[0] for y in Y]) / IMG - 0.5
    assert cy.std() > 0.03
    assert meta["shift_y"].abs().max() <= cfg.jitter + 1e-9
    # y es reproducible: misma semilla -> mismos desplazamientos
    df = load_case_csv(cfg.csv_dir, "train")
    _, _, meta2, _ = build_split(df, cfg.jpeg_dir, cfg.img_size, cfg.context, cfg.min_side,
                                 jitter=cfg.jitter, seed=cfg.seed, progress=False)
    assert np.allclose(meta["shift_y"], meta2["shift_y"])


def test_baselines_beat_empty_prediction(data):
    _, d = data
    _, Xtr, Ytr, _, _ = d["train"]
    _, Xte, Yte, _, _ = d["test"]
    r = fit_center_prior(Ytr)
    prior = np.stack([center_prior((IMG, IMG), r)] * len(Yte))
    classical = classical_segment_stack(Xte)
    assert per_sample_metrics(prior, Yte)["dice"].mean() > 0.2
    assert per_sample_metrics(classical, Yte)["dice"].mean() > 0.4


def test_shape_pipeline(data):
    _, d = data
    _, _, Ytr, mtr, _ = d["train"]
    _, _, Yte, mte, _ = d["test"]
    ktr, kte = (mtr["shape_label"] != "").values, (mte["shape_label"] != "").values
    Ftr, Fte = shp.features_dataframe(Ytr[ktr]), shp.features_dataframe(Yte[kte])
    clf = shp.fit_shape_tree(Ftr, mtr.loc[ktr, "shape_label"].values)
    pred = shp.predict_shape(clf, Fte)
    res = shp.evaluate_shape(mte.loc[kte, "shape_label"].values, pred)
    assert res["accuracy"] > 0.45                  # azar = 0.25
    assert res["confusion"].shape == (4, 5)
    assert any(f in shp.tree_rules(clf) for f in shp.FEATURE_NAMES)


def test_unet_trains_and_predicts(data):
    torch = pytest.importorskip("torch")
    from src.dataset import to_tensors
    from src.data_utils import patient_split
    from src.train import predict_probs, train_unet
    cfg, d = data
    _, X, Y, meta, _ = d["train"]
    tr, va = patient_split(meta, 0.25, 0)
    assert set(meta.loc[tr, "patient_id"]).isdisjoint(set(meta.loc[va, "patient_id"]))
    model, hist = train_unet(cfg, X[tr], Y[tr], X[va], Y[va], verbose=False)
    assert len(hist) >= 1 and np.isfinite(hist["val_dice"]).all()
    xt, _ = to_tensors(X[va], Y[va], torch.device("cpu"))
    probs = predict_probs(model, xt)
    assert probs.shape == (len(va), IMG, IMG) and 0 <= probs.min() and probs.max() <= 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
