"""U-Net (Ronneberger et al., 2015) para segmentación binaria de recortes ROI."""
import torch
import torch.nn as nn

from .morph import MorphBlock


class DoubleConv(nn.Module):
    """(Conv3x3 -> BatchNorm -> ReLU) x 2"""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """U-Net de ``depth`` niveles. Entrada (B,1,H,W) con H y W múltiplos de 2**depth.

    Devuelve logits (B,1,H,W); aplicar sigmoide para obtener probabilidades.
    """

    def __init__(self, in_ch: int = 1, out_ch: int = 1, base: int = 32, depth: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        chs = [base * 2 ** i for i in range(depth + 1)]
        self.down = nn.ModuleList()
        prev = in_ch
        for i, c in enumerate(chs[:-1]):
            self.down.append(DoubleConv(prev, c, dropout=0.0))
            prev = c
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(chs[-2], chs[-1], dropout=dropout)

        self.up = nn.ModuleList()
        self.up_conv = nn.ModuleList()
        for i in range(depth, 0, -1):
            self.up.append(nn.ConvTranspose2d(chs[i], chs[i - 1], 2, stride=2))
            self.up_conv.append(DoubleConv(chs[i - 1] * 2, chs[i - 1]))
        self.head = nn.Conv2d(chs[0], out_ch, 1)

    def forward(self, x):
        skips = []
        for block in self.down:
            x = block(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        return self.head(self.decode(x, skips))

    def decode(self, x, skips):
        for up, conv, skip in zip(self.up, self.up_conv, reversed(skips)):
            x = up(x)
            x = torch.cat([self.process_skip(skip), x], dim=1)  # conexión de salto
            x = conv(x)
        return x

    def process_skip(self, skip):
        return skip


class MorphUNet(UNet):
    """U-Net con un bloque morfológico aprendible (``src.morph.MorphBlock``).

    ``position="head"``: sobre los mapas finales, antes de la cabeza 1x1.
    ``position="skip"``: sobre la conexión de salto de máxima resolución.
    """

    def __init__(self, base: int = 32, morph_channels: int = 4, kernel_size: int = 5,
                 position: str = "head", **kw):
        super().__init__(base=base, **kw)
        if position not in ("head", "skip"):
            raise ValueError(f"position debe ser 'head' o 'skip', no {position!r}")
        self.position = position
        self.morph = MorphBlock(base, morph_channels, kernel_size)

    def forward(self, x):
        skips = []
        for block in self.down:
            x = block(x)
            skips.append(x)
            x = self.pool(x)
        x = self.decode(self.bottleneck(x), skips)
        if self.position == "head":
            x = self.morph(x)
        return self.head(x)

    def process_skip(self, skip):
        # solo la conexión de máxima resolución tiene ``base`` canales
        if self.position == "skip" and skip.size(1) == self.morph.reduce.in_channels:
            return self.morph(skip)
        return skip


ARCHS = ("unet", "morph_unet")


def build_model(cfg) -> nn.Module:
    """Red indicada por ``cfg.arch`` (``unet`` o ``morph_unet``)."""
    if cfg.arch == "unet":
        return UNet(base=cfg.base_channels)
    if cfg.arch == "morph_unet":
        return MorphUNet(base=cfg.base_channels, morph_channels=cfg.morph_channels,
                         kernel_size=cfg.morph_kernel, position=cfg.morph_position)
    raise ValueError(f"arch desconocida: {cfg.arch!r}. Opciones: {ARCHS}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
