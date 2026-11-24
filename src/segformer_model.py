"""SegFormer-B0 (Xie et al., 2021) para segmentación binaria de recortes de 1 canal.

Transformer jerárquico (Mix Transformer, MiT-b0) con atención sobre toda la imagen y un
decodificador MLP liviano. Requiere el paquete opcional ``transformers``.

- Entrada (B, 1, H, W) replicada a 3 canales.
- La salida del decodificador (1 canal, H/4 × W/4) se escala a H × W con interpolación
  bilineal y se devuelven logits (B, 1, H, W), igual que la U-Net.
- ``pretrained=True`` (o el nombre de un repositorio, por defecto "nvidia/mit-b0") carga el
  codificador preentrenado en ImageNet-1k desde Hugging Face (se descarga una vez a la caché
  de HF y no se versiona); el decodificador siempre se inicializa al azar.
- ``pretrained=False`` (o "") usa la misma arquitectura con pesos aleatorios: control sin
  preentrenamiento (E13b) y pruebas sin red.
"""
import torch.nn as nn
import torch.nn.functional as F

MIT_B0 = "nvidia/mit-b0"


def _import_transformers():
    try:
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
    except ImportError as e:  # pragma: no cover - depende del entorno
        raise ImportError("arch 'segformer_b0' requiere el paquete opcional 'transformers' "
                          "(pip install 'transformers>=4.40'; ver requirements.txt)") from e
    return SegformerConfig, SegformerForSemanticSegmentation


class SegFormerBinary(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        if pretrained is True:
            pretrained = MIT_B0
        SegformerConfig, SegformerForSemanticSegmentation = _import_transformers()
        if pretrained:
            self.net = SegformerForSemanticSegmentation.from_pretrained(
                pretrained, num_labels=1, ignore_mismatched_sizes=True)
        else:
            # misma arquitectura que nvidia/mit-b0, pesos aleatorios
            cfg = SegformerConfig(num_labels=1, depths=[2, 2, 2, 2], hidden_sizes=[32, 64, 160, 256],
                                  decoder_hidden_size=256, num_attention_heads=[1, 2, 5, 8])
            self.net = SegformerForSemanticSegmentation(cfg)

    def forward(self, x):
        h, w = x.shape[-2:]
        logits = self.net(pixel_values=x.expand(-1, 3, -1, -1)).logits      # (B, 1, H/4, W/4)
        return F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
