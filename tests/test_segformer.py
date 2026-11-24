"""Pruebas de SegFormer-B0 con pesos aleatorios (sin descargar nada de Hugging Face)."""
import pytest
import torch

pytest.importorskip("transformers")

from src.config import Config          # noqa: E402
from src.losses import get_loss        # noqa: E402
from src.model import build_model      # noqa: E402


def _model():
    torch.manual_seed(0)
    return build_model(Config(arch="segformer_b0", pretrained=""))


def test_output_shape_and_finite():
    m = _model().eval()
    with torch.no_grad():
        y = m(torch.rand(2, 1, 64, 64))
    assert y.shape == (2, 1, 64, 64)
    assert torch.isfinite(y).all()


def test_gradients_reach_encoder_and_decoder():
    m = _model().train()
    x = torch.rand(2, 1, 64, 64)
    t = (torch.rand(2, 1, 64, 64) > 0.5).float()
    loss = get_loss("bce_dice")(m(x), t)
    loss.backward()
    assert torch.isfinite(loss)
    named = dict(m.named_parameters())
    enc = [p for n, p in named.items() if ".segformer." in n]   # codificador MiT
    dec = [p for n, p in named.items() if "decode_head" in n]
    assert enc and dec
    for group in (enc, dec):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in group)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in group)


def test_parameter_count_matches_mit_b0():
    n = sum(p.numel() for p in _model().parameters())
    assert 3.5e6 < n < 4.0e6     # SegFormer-B0 con cabeza de 1 clase: ~3.7 M
