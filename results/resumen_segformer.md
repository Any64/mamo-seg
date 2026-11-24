# Resumen E13: U-Net frente a SegFormer-B0

El profesor pidió probar una familia de modelos distinta a la U-Net. Se eligió **SegFormer-B0** (Xie et al., 2021), un
transformer pequeño con atención global y un decodificador MLP.

**Hipótesis:** los fallos de la U-Net son de localización, porque elige otra estructura densa (E11). Un modelo con
contexto global debería fallar menos en ese modo.

**Protocolo:** el mismo de `configs/main.yaml`:
- recortes propios de 256 × 256, división por paciente, BCE + Dice, AdamW, early stopping con paciencia 10, 40
  épocas como máximo, batch 8 y `amp: false`;
- el lr se eligió solo en validación, con la semilla 42;
- después se entrenaron las semillas 43 y 44;
- **el test solo se informa.**

Configuración en `configs/segformer_b0.yaml`. Comparación con `scripts/compare_segformer.py`.

## Implementación y costo

- **Modelo** (`src/segformer_model.py`):
  - Entrada de 1 canal replicada a 3.
  - Codificador MiT-b0 preentrenado en ImageNet-1k (`nvidia/mit-b0`, Hugging Face; se descarga una vez y no se
    versiona). El decodificador se inicializa al azar.
  - Salida de 1 canal a 64 × 64, escalada a 256 × 256 con interpolación bilineal.
  - Requiere el paquete opcional `transformers` (probado con 5.24.1).
- **Pruebas** (`tests/test_segformer.py`): forma de la salida, ausencia de NaN, gradientes en el codificador y el
  decodificador, y número de parámetros. Usan pesos aleatorios y no descargan nada.
- **Costo:**
  - 3 714 401 parámetros, frente a 1 942 289 de la U-Net.
  - ~29–34 s por época, frente a 16 s; ~1.1 GB de GPU.
  - Ningún NaN.
  - Cada entrenamiento duró 11–19 min, gracias al early stopping. Los 4 entrenamientos sumaron ~1 h.

### Elección del lr (validación, semilla 42; `segformer_eleccion_lr.csv`)

| lr | Dice val | Épocas (mejor) |
|---|---|---|
| 1e-4 | 0.806 | 17 (7) |
| **3e-4** | **0.809** | 24 (14) |

Elegido: **3e-4**. La diferencia es pequeña, de 0.003.

## 1. Comparación (media ± DE de 3 semillas; `segformer_comparacion.csv`, `segformer_por_semilla.csv`)

Posprocesamiento: el fijo de E12 (r = 6), sin volver a elegirlo.

| Método | Conj. | Dice | IoU | Precisión | Sensib. | HD95 medio (px) | HD95 mediano (px) | Componentes | Vacías | Dice < 0.1 |
|---|---|---|---|---|---|---|---|---|---|---|
| U-Net | val | 0.777 ± 0.003 | 0.666 | 0.769 | 0.826 | 26.6 | 13.7 | 1.80 | 3.0 | 6.7 |
| U-Net + posproc. | val | 0.791 ± 0.006 | 0.687 | 0.810 | 0.812 | 16.6 | 12.4 | 0.98 | 4.3 | 10.0 |
| **SegFormer-B0** | val | **0.810 ± 0.001** | 0.701 | 0.753 | 0.888 | 16.4 | 12.5 | 1.25 | 0 | 5.0 |
| SegFormer-B0 + posproc. | val | 0.810 ± 0.003 | 0.703 | 0.760 | 0.879 | 15.8 | 12.5 | 1.00 | 0 | 6.3 |
| U-Net | test | 0.762 ± 0.009 | 0.652 | 0.758 | 0.804 | 25.6 | 15.4 | 1.75 | 2.3 | 13.3 |
| U-Net + posproc. | test | 0.767 ± 0.007 | 0.662 | 0.789 | 0.789 | 18.3 | 13.1 | 0.98 | 7.3 | 21.3 |
| **SegFormer-B0** | test | **0.792 ± 0.003** | 0.683 | 0.738 | 0.867 | 17.4 | 12.8 | 1.23 | 0 | 10.0 |
| SegFormer-B0 + posproc. | test | 0.793 ± 0.003 | 0.686 | 0.746 | 0.860 | 16.9 | 12.8 | 1.00 | 0 | 11.7 |

### Diferencias pareadas (`segformer_pruebas.csv`)

Se comparan el Dice y el HD95 de cada recorte, promediados entre semillas. Se informa el IC 95 % por bootstrap,
remuestreando pacientes, y el Wilcoxon pareado.

| Diferencia | Conj. | Δ Dice [IC 95 %] | p | Δ HD95 medio px [IC 95 %] | p |
|---|---|---|---|---|---|
| SegFormer − U-Net | val | +0.033 [0.020, 0.046] | 7e-4 | −10.3 [−13.2, −7.6] | < 1e-4 |
| SegFormer − U-Net + posproc. | val | +0.019 [0.006, 0.032] | 0.54 | −0.6 [−1.7, 0.6] | 0.81 |
| SegFormer − U-Net | test | **+0.030 [0.019, 0.041]** | < 1e-4 | **−8.5 [−10.4, −6.7]** | < 1e-4 |
| SegFormer − U-Net + posproc. | test | +0.026 [0.014, 0.038] | 0.019 | −1.7 [−2.7, −0.6] | 0.001 |
| SegFormer + posproc. − SegFormer | test | +0.001 [0.000, 0.003] | < 1e-4 | −0.5 [−1.0, −0.1] | 0.27 |

- **SegFormer-B0 mejora a la U-Net de forma clara y consistente**: cerca de +0.03 de Dice en validación y en test, con
  las 3 semillas por encima. También mejora el HD95.
- **Frente a la U-Net + posprocesamiento:**
  - En test, la ventaja es de +0.026 en Dice y es significativa.
  - En validación, el IC del Dice excluye el 0, pero el Wilcoxon no es significativo (p = 0.54). La mediana de las
    diferencias es casi 0, así que la ventaja media viene de pocos recortes con mucha diferencia.
- **Dónde gana:** la mejora viene de la **sensibilidad** (0.87–0.89 frente a 0.80–0.83). La precisión es algo menor
  (0.74–0.75 frente a 0.76–0.77). SegFormer marca regiones más grandes y nunca deja una predicción vacía.
- **Posprocesamiento sobre SegFormer:** casi no cambia nada (+0.001). SegFormer ya produce en general una sola
  componente (1.23 de media, frente a 1.75 de la U-Net).

## 2. ¿Falla menos en "otra zona"? (`segformer_fallos.csv`, `segformer_fallos_pruebas.csv`)

Se usa el mismo criterio que E11: Dice < 0.1, clasificado por modo. Las cifras son la media de fallos por semilla.
La prueba es un McNemar exacto que empareja cada semilla con la misma semilla del otro modelo, sobre el mismo recorte;
la columna "solo falla" cuenta los recortes donde falla SegFormer y no la U-Net, y al revés.

| Modo | U-Net test | SegFormer test | Solo falla SegFormer / solo U-Net (test) | p | U-Net val | SegFormer val | p val |
|---|---|---|---|---|---|---|---|
| Fallos (total) | 13.3 | 10.0 | 14 / 24 | 0.14 | 6.7 | 5.0 | 0.38 |
| Predicción vacía | 2.3 | **0** | 0 / 7 | 0.016 | 3.0 | **0** | 0.004 |
| **Otra zona sin solape** | 5.3 | 4.0 | 7 / 11 | **0.48** | 1.0 | **4.3** | **0.013** |
| Sub/sobresegmentación | 5.7 | 6.0 | 15 / 14 | 1.0 | 2.7 | 0.7 | 0.11 |

**Respuesta honesta: no, SegFormer no falla menos en "otra zona".**
- **Test:** pasa de 5.3 a 4.0 fallos por semilla, pero la diferencia no es significativa (p = 0.48). Además, en 7 de
  los 18 recortes discordantes falla SegFormer y no la U-Net.
- **Validación:** va en contra de la hipótesis. SegFormer falla en otra zona más veces que la U-Net (4.3 frente a 1.0
  por semilla; 12 frente a 2 discordantes, p = 0.013).
- **Qué cambia de verdad:** SegFormer elimina las predicciones vacías y tiene algo menos de fallos totales (no
  significativo).
- **De dónde sale la mejora media:** de los recortes donde la U-Net marca solo un fragmento pequeño o disperso de la
  lesión y SegFormer la cubre entera (`figures/segformer_ejemplos.png`; 8 de los 8 casos con mayor diferencia son de
  este tipo). Es un problema de detección de lesiones de bajo contraste, no de elegir la estructura equivocada.
- **Conclusión:** la confusión con otra estructura densa sigue presente con atención global. Es coherente con E11:
  dentro del recorte puede haber estructuras tan compatibles con una masa como la lesión misma.

## 3. Forma BI-RADS con las máscaras de SegFormer (`segformer_forma*.csv`)

Se usa el árbol de E1 (profundidad 3, ajustado en train con máscaras anotadas), con la semilla 42.

| Máscara | LOBULATED val | LOBULATED test | Exactitud balanceada val | Exactitud balanceada test |
|---|---|---|---|---|
| Anotada | 29 % | 36 % | 0.30 | 0.22 |
| U-Net | 72 % | 75 % | 0.30 | 0.26 |
| U-Net + posproc. | 76 % | 82 % | 0.26 | 0.28 |
| SegFormer-B0 | 71 % | 70 % | 0.29 | 0.30 |
| SegFormer-B0 + posproc. | 71 % | 76 % | 0.27 | 0.31 |
| (etiqueta real) | 28 % | 24 % | — | — |

**No cambia.**
- Las máscaras de SegFormer también son suaves, y el árbol las clasifica como LOBULATED en el 70 % de los casos.
- La exactitud balanceada sigue cerca del azar (0.25). Los valores de 0.30–0.31 en test no se repiten en validación.
- El límite está en las máscaras de CBIS-DDSM (E2–E4, E10), no en la red.

## 4. Interpretación y cautelas

- **SegFormer-B0 es el mejor segmentador probado:** Dice en test 0.792 ± 0.003, frente a 0.762 ± 0.009 de la U-Net.
  Se elige en validación (0.810 frente a 0.777). En validación, con y sin posprocesamiento empatan (0.8103 frente a
  0.8097), así que el posprocesamiento no aporta.
- **La hipótesis de localización no se confirma.** La mejora viene de una mayor sensibilidad (cubre lesiones enteras,
  sin predicciones vacías), no de evitar otras estructuras densas.
- **Cautelas:**
  - **Preentrenamiento:** SegFormer usa un codificador preentrenado en ImageNet y la U-Net se entrena desde cero. Por
    eso la mejora no puede atribuirse solo a la atención global; parte puede venir del preentrenamiento.
  - **Tamaño:** SegFormer tiene el doble de parámetros.
  - **lr:** se ajustó para SegFormer (2 valores); el de la U-Net viene de la entrega parcial.
  - **Muestra:** son 3 semillas, y varios recortes son del mismo paciente; el IC por paciente lo tiene en cuenta.
- **Lo que no cambia:** las cifras de referencia ya publicadas (U-Net 0.762 ± 0.009) y la decisión de E12. E13 se
  informa como un experimento adicional.
