"""Pruebas de las capas morfológicas aprendibles (src/morph.py) y de la Morph-UNet."""
import numpy as np
import pytest
import torch
from scipy import ndimage as ndi

from src.config import Config
from src.model import MorphUNet, UNet, build_model, count_parameters
from src.morph import MorphBlock, PConv2d

# Tolerancia frente a scipy con |p| = 200 y entrada en [0.1, 1]: la media contra-armónica
# solo se acerca al máximo/mínimo cuando |p| -> inf. Medido: error máximo ~0.003 y medio
# ~3e-4 (con |p| = 50 el error máximo sube a ~0.016).
P_LARGE, TOL_MAX, TOL_MEAN = 200.0, 0.01, 1e-3


def _image(seed=0, size=48):
    rng = np.random.default_rng(seed)
    img = ndi.gaussian_filter(rng.random((size, size)), 1.5)
    return (img - img.min()) / (img.max() - img.min()) * 0.9 + 0.1


def _apply(layer, img):
    with torch.no_grad():
        return layer(torch.tensor(img, dtype=torch.float32)[None, None])[0, 0].numpy()


@pytest.mark.parametrize("sign,scipy_fn", [(1, ndi.grey_dilation), (-1, ndi.grey_erosion)])
def test_pconv_flat_square_matches_scipy(sign, scipy_fn):
    img = _image()
    y = _apply(PConv2d(1, 5, sign * P_LARGE), img)
    ref = scipy_fn(img, footprint=np.ones((5, 5)), mode="nearest")
    err = np.abs(y - ref)
    assert err.max() < TOL_MAX and err.mean() < TOL_MEAN


@pytest.mark.parametrize("sign,scipy_fn", [(1, ndi.grey_dilation), (-1, ndi.grey_erosion)])
def test_pconv_disk_footprint_matches_scipy(sign, scipy_fn):
    img = _image(seed=1)
    yy, xx = np.mgrid[-2:3, -2:3]
    disk = (xx ** 2 + yy ** 2) <= 4
    layer = PConv2d(1, 5, sign * P_LARGE, learn_se=False)
    with torch.no_grad():   # elemento estructurante fijo: w = 1 en el disco, ~0 fuera
        layer.log_w.copy_(torch.tensor(np.where(disk, 0.0, -1e4), dtype=torch.float32).view(1, -1))
    err = np.abs(_apply(layer, img) - scipy_fn(img, footprint=disk, mode="nearest"))
    assert err.max() < TOL_MAX and err.mean() < TOL_MEAN


def test_pconv_p_zero_is_local_mean():
    img = _image(seed=2)
    y = _apply(PConv2d(1, 3, 0.0), img)
    assert np.allclose(y, ndi.uniform_filter(img, 3, mode="nearest"), atol=1e-5)


def test_pconv_no_nan_extreme_values():
    x = torch.cat([torch.zeros(1, 2, 16, 16), torch.ones(1, 2, 16, 16), torch.full((1, 2, 16, 16), 1e-9)])
    for p in (-500.0, -5.0, 0.0, 5.0, 500.0):
        layer = PConv2d(2, 5, p)
        y = layer(x)
        y.sum().backward()
        assert torch.isfinite(y).all() and torch.isfinite(layer.p.grad).all()
        assert torch.isfinite(layer.log_w.grad).all()


def test_morph_block_shape_and_gradients():
    torch.manual_seed(0)
    block = MorphBlock(8, morph_channels=3, kernel_size=5).train()
    f = torch.randn(2, 8, 32, 32, requires_grad=True)
    out = block(f)
    assert out.shape == f.shape
    out.pow(2).mean().backward()
    for name, p in block.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    for name, layer in block.layers().items():   # p y el elemento estructurante aprenden
        assert layer.p.grad.abs().sum() > 0, name
        assert layer.log_w.grad.abs().sum() > 0, name
    assert f.grad.abs().sum() > 0


def test_morph_block_eval_equals_train_forward():
    # el checkpointing solo cambia cómo se calcula el gradiente, no la salida
    torch.manual_seed(0)
    block = MorphBlock(4, 2, 3)
    f = torch.randn(1, 4, 16, 16, requires_grad=True)
    a = block.train()(f)
    b = block.eval()(f)
    assert torch.allclose(a, b, atol=1e-6)


@pytest.mark.parametrize("position", ["head", "skip"])
def test_morph_unet_shape_and_no_nan(position):
    torch.manual_seed(0)
    net = MorphUNet(base=8, morph_channels=2, kernel_size=3, position=position)
    x = torch.rand(2, 1, 64, 64)
    y = net(x)
    assert y.shape == (2, 1, 64, 64) and torch.isfinite(y).all()
    y.mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
    assert net.morph.open_ero.p.grad is not None
    # el bloque añade muy pocos parámetros
    assert 0 < count_parameters(net) - count_parameters(UNet(base=8)) < 1000


def test_build_model_from_config():
    assert type(build_model(Config(base_channels=8))) is UNet
    m = build_model(Config(arch="morph_unet", base_channels=8, morph_channels=2, morph_kernel=3,
                           morph_position="skip"))
    assert isinstance(m, MorphUNet) and m.position == "skip"
    with pytest.raises(ValueError):
        build_model(Config(arch="resnet"))
    with pytest.raises(ValueError):
        MorphUNet(base=8, position="bottleneck")
