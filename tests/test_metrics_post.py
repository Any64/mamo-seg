"""Pruebas de Hausdorff 95, componentes conexas y posprocesamiento fijo."""
import numpy as np
from scipy import ndimage as ndi

from src.metrics import extended_metrics, hausdorff95, n_components
from src.postprocess import largest_component, postprocess_mask


def _square(y0, x0, side, size=64):
    m = np.zeros((size, size), bool)
    m[y0:y0 + side, x0:x0 + side] = True
    return m


def test_hd95_identical_is_zero():
    m = _square(10, 10, 20)
    assert hausdorff95(m, m) == 0.0


def test_hd95_shifted_square():
    a, b = _square(10, 10, 20), _square(10, 15, 20)   # desplazado 5 px en x
    hd = hausdorff95(a, b)
    assert 4.0 <= hd <= 5.0 + 1e-9
    assert hausdorff95(a, b) == hausdorff95(b, a)      # simétrica
    assert np.isclose(hausdorff95(a, b, spacing=0.5), hd / 2)


def test_hd95_robust_to_small_outlier():
    # un píxel aislado lejos afecta a la distancia de Hausdorff máxima, no a la del percentil 95
    gt = _square(10, 10, 30)
    pred = gt.copy()
    pred[60, 60] = True
    assert hausdorff95(pred, gt) < 1.0


def test_hd95_empty_is_nan():
    assert np.isnan(hausdorff95(np.zeros((32, 32), bool), _square(5, 5, 10, 32)))


def test_n_components_and_extended_metrics():
    m = _square(5, 5, 10) | _square(40, 40, 5)
    m2 = m.copy()
    m2[15, 15] = True   # toca en diagonal al primer cuadrado: conectividad 8
    assert n_components(m) == 2 and n_components(m2) == 2
    pred = np.stack([m, np.zeros_like(m)])
    gt = np.stack([_square(5, 5, 10), _square(5, 5, 10)])
    df = extended_metrics(pred, gt)
    assert list(df.n_componentes) == [2, 0] and list(df.vacia) == [False, True]
    assert np.isnan(df.hd95[1]) and df.hd95[0] > 0


def test_postprocess_keeps_largest_and_removes_specks():
    m = _square(10, 10, 30)
    m[50, 50] = True
    out = postprocess_mask(m, radius=2)
    assert n_components(out) == 1 and not out[50, 50]
    assert np.array_equal(largest_component(np.zeros((8, 8), bool)), np.zeros((8, 8), bool))
    # sin radio solo filtra componentes; el cuadrado no cambia
    assert np.array_equal(postprocess_mask(_square(10, 10, 30), radius=0), _square(10, 10, 30))


def test_postprocess_closing_fills_small_hole_and_border_safe():
    m = _square(0, 0, 40)          # toca el borde del recorte
    m[20, 20] = False
    out = postprocess_mask(m, radius=2)
    assert out[20, 20] and out[0, 0]   # agujero cerrado; el borde del recorte no se erosiona
    # la apertura solo redondea la esquina convexa interior (39, 39): pocos píxeles
    assert (out ^ ndi.binary_fill_holes(m)).sum() <= 6


def test_drop_cause():
    from src.postprocess import CAUSE_COMPONENT, CAUSE_EMPTY, CAUSE_OTHER, drop_cause
    assert drop_cause(0.05, 0.0, 0.0, True) is None            # la U-Net ya fallaba
    assert drop_cause(0.6, 0.6, 0.5, False) is None            # no cae
    assert drop_cause(0.6, 0.0, 0.0, False) == CAUSE_COMPONENT
    assert drop_cause(0.6, 0.0, 0.0, True) == CAUSE_COMPONENT  # con r = 0 ya fallaba
    assert drop_cause(0.6, 0.6, 0.0, True) == CAUSE_EMPTY
    assert drop_cause(0.6, 0.6, 0.02, False) == CAUSE_OTHER
