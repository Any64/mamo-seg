"""U-Net (Ronneberger et al., 2015) para segmentación binaria de recortes ROI."""
import torch
import torch.nn as nn


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
        for up, conv, skip in zip(self.up, self.up_conv, reversed(skips)):
            x = up(x)
            x = torch.cat([skip, x], dim=1)  # conexión de salto
            x = conv(x)
        return self.head(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
