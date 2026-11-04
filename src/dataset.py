"""Datos para el entrenamiento (torch): tensores, lotes y aumento de datos
(se aplica sobre el lote, directamente en el dispositivo de cómputo)."""
import math

import numpy as np
import torch
import torch.nn.functional as F

from .data_utils import normalize_stack


def to_tensors(X: np.ndarray, Y: np.ndarray, device: torch.device):
    xt = torch.from_numpy(normalize_stack(X))[:, None].to(device)
    yt = torch.from_numpy(Y.astype(np.float32))[:, None].to(device)
    return xt, yt


def iterate_batches(x: torch.Tensor, y: torch.Tensor, batch_size: int,
                    shuffle: bool, generator: torch.Generator | None = None):
    n = x.size(0)
    if shuffle:
        idx = torch.randperm(n, generator=generator).to(x.device)
    else:
        idx = torch.arange(n, device=x.device)
    for s in range(0, n, batch_size):
        b = idx[s: s + batch_size]
        yield x[b], y[b]


def augment_batch(x: torch.Tensor, y: torch.Tensor, rot_deg: float = 20.0,
                  scale: tuple = (0.85, 1.15), shift: float = 0.08,
                  contrast: float = 0.15, brightness: float = 0.08):
    """Aumento de datos geométrico (volteos, rotación, escala, traslación) y de
    intensidad (contraste/brillo). La misma transformación geométrica se aplica a
    imagen y máscara (a la máscara con interpolación de vecino más cercano)."""
    B, dev = x.size(0), x.device
    ang = (torch.rand(B, device=dev) * 2 - 1) * rot_deg * math.pi / 180
    sc = torch.empty(B, device=dev).uniform_(*scale)
    tx = (torch.rand(B, device=dev) * 2 - 1) * shift * 2   # coords normalizadas en [-1,1]
    ty = (torch.rand(B, device=dev) * 2 - 1) * shift * 2
    fx = (torch.rand(B, device=dev) < 0.5).float() * 2 - 1  # +1 / -1 (volteo horizontal)
    fy = (torch.rand(B, device=dev) < 0.5).float() * 2 - 1  # +1 / -1 (volteo vertical)
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([
        torch.stack([cos * fx, -sin * fy, tx], dim=1),
        torch.stack([sin * fx, cos * fy, ty], dim=1),
    ], dim=1)  # (B,2,3)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    xa = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    ya = F.grid_sample(y, grid, mode="nearest", padding_mode="zeros", align_corners=False)
    c = torch.empty(B, 1, 1, 1, device=dev).uniform_(1 - contrast, 1 + contrast)
    b = torch.empty(B, 1, 1, 1, device=dev).uniform_(-brightness, brightness)
    xa = (xa * c + b).clamp(0, 1)
    return xa, (ya > 0.5).float()
