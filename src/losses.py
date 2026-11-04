"""Funciones de pérdida para el entrenamiento (torch)."""
import torch
import torch.nn.functional as F

EPS = 1e-6


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(logits)
    dims = (1, 2, 3)
    inter = (p * target).sum(dims)
    denom = p.sum(dims) + target.sum(dims)
    return 1.0 - ((2 * inter + EPS) / (denom + EPS)).mean()


def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """BCE + Dice: la BCE da gradientes estables y el Dice compensa el desbalance
    entre píxeles de lesión y de fondo."""
    return F.binary_cross_entropy_with_logits(logits, target) + soft_dice_loss(logits, target)
