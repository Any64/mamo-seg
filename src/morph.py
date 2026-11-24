"""Capas morfológicas aprendibles por media contra-armónica (p-convolución).

Masci et al. (2013), "A learning framework for morphological operators using
counter-harmonic mean":

    y = conv(x^(p+1), w) / conv(x^p, w),   x > 0,  w >= 0

Con p -> +inf, y tiende al máximo de x en el soporte de w (dilatación con
elemento estructurante plano); con p -> -inf, al mínimo (erosión); con p = 0 es
una media ponderada (convolución lineal normalizada).

Implementación estable: escribiendo x^p = exp(p·log x), el cociente es una media
de x ponderada por softmax(p·log x + log w) sobre la ventana, así que no hay
desbordamiento aunque |p| sea grande. Cada canal tiene su propio elemento
estructurante (log w aprendible, k x k) y su propio p aprendible.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

EPS = 1e-4   # la entrada se lleva a [EPS, 1] para que log x sea finito


class PConv2d(nn.Module):
    """p-convolución por canal (depthwise) con borde replicado; la salida tiene el tamaño de la entrada.

    La entrada debe ser positiva (se recorta a ``[EPS, inf)``). ``p_init`` > 0
    inicia como dilatación suave y < 0 como erosión suave.
    """

    def __init__(self, channels: int, kernel_size: int = 5, p_init: float = 5.0, learn_se: bool = True):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size debe ser impar")
        self.k = kernel_size
        self.p = nn.Parameter(torch.full((channels,), float(p_init)))
        # log del elemento estructurante; 0 = cuadrado plano (w = 1 en toda la ventana)
        self.log_w = nn.Parameter(torch.zeros(channels, kernel_size * kernel_size), requires_grad=learn_se)

    @property
    def weight(self) -> torch.Tensor:
        """Elemento estructurante w = exp(log w), forma (C, k, k)."""
        return self.log_w.detach().exp().view(-1, self.k, self.k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = x.clamp_min(EPS)
        r = self.k // 2
        patches = F.unfold(F.pad(x, (r, r, r, r), mode="replicate"), self.k)   # (B, C*k*k, H*W)
        patches = patches.view(b, c, self.k * self.k, h * w)
        logits = self.p.view(1, c, 1, 1) * patches.log() + self.log_w.view(1, c, -1, 1)
        y = (torch.softmax(logits, dim=2) * patches).sum(dim=2)
        return y.view(b, c, h, w)


class MorphBlock(nn.Module):
    """Apertura (erosión→dilatación) y cierre (dilatación→erosión) aprendibles con conexión residual.

    f (C canales) -> conv 1x1 a ``m`` canales -> sigmoide (entrada positiva en (0, 1))
    -> [apertura, cierre] -> conv 1x1 a C canales -> f + salida.
    Se usan pocos canales (``m``) porque cada p-convolución despliega k·k vecinos por píxel.
    """

    def __init__(self, channels: int, morph_channels: int = 4, kernel_size: int = 5, p0: float = 5.0):
        super().__init__()
        self.reduce = nn.Conv2d(channels, morph_channels, 1)
        self.open_ero = PConv2d(morph_channels, kernel_size, -p0)
        self.open_dil = PConv2d(morph_channels, kernel_size, +p0)
        self.close_dil = PConv2d(morph_channels, kernel_size, +p0)
        self.close_ero = PConv2d(morph_channels, kernel_size, -p0)
        self.expand = nn.Conv2d(2 * morph_channels, channels, 1)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        x = torch.sigmoid(self.reduce(f))
        if self.training and x.requires_grad:
            # checkpointing: no guarda los k·k vecinos desplegados; se recalculan en backward (cabe en 4 GB)
            opened = checkpoint(self._open, x, use_reentrant=False)
            closed = checkpoint(self._close, x, use_reentrant=False)
        else:
            opened, closed = self._open(x), self._close(x)
        return f + self.expand(torch.cat([opened, closed], dim=1))

    def _open(self, x):
        return self.open_dil(self.open_ero(x))

    def _close(self, x):
        return self.close_ero(self.close_dil(x))

    def layers(self) -> dict:
        """Las cuatro p-convoluciones por nombre (para inspeccionar p y los elementos estructurantes)."""
        return {"apertura: erosión": self.open_ero, "apertura: dilatación": self.open_dil,
                "cierre: dilatación": self.close_dil, "cierre: erosión": self.close_ero}
