# Segmentación de masas mamarias (CBIS-DDSM) y caracterización de su forma según BI-RADS 2025

Proyecto del curso **CBM423 – Procesamiento Digital de Imágenes Médicas** (Universidad de Valparaíso).

## Descripción

1. **Segmentación:** una U-Net segmenta masas mamarias en recortes ROI de mamografías del dataset CBIS-DDSM.
2. **Comparación:** su desempeño (Dice, IoU) se compara con un método clásico de procesamiento de imágenes
   (suavizado + umbral de Otsu + morfología matemática) y con un control "prior centrado".
3. **Forma BI-RADS:** a partir de la máscara se calculan descriptores geométricos del contorno y un árbol de
   decisión poco profundo (interpretable, sin red neuronal adicional) asigna la forma
   `ROUND / OVAL / LOBULATED / IRREGULAR`, léxico restablecido en BI-RADS v2025. Se valida contra el campo
   `mass shape` del dataset, usando máscaras anotadas (cota superior) y máscaras predichas por la U-Net.

> **Decisión de preprocesamiento relevante.** En CBIS-DDSM la máscara ROI está en las coordenadas de la mamografía
> *completa* y no está registrada con el recorte oficial. Por eso se generan recortes propios a partir de la
> mamografía completa y la máscara (ver `src/preprocess.py`), lo que garantiza imagen y máscara alineadas.

## Estructura

```
src/
  config.py       parámetros del proyecto
  data_index.py   lectura de CSV y resolución de rutas (particularidades de la versión de Kaggle)
  preprocess.py   recortes ROI alineados, etiquetas de forma, chequeo de alineación
  dataset.py      división por paciente, normalización, aumento de datos
  model.py        U-Net
  losses.py       BCE + Dice
  metrics.py      Dice, IoU, precisión, sensibilidad
  train.py        entrenamiento con early stopping
  classical.py    métodos de referencia (prior centrado; Otsu + morfología)
  shape.py        descriptores geométricos y árbol de decisión para la forma BI-RADS
  viz.py          figuras
notebooks/proyecto_mamografia.ipynb   ejecuta todo el pipeline paso a paso
tests/                                pruebas con un dataset sintético
report/                               informes (LaTeX)
```

## Datos

Dataset: [CBIS-DDSM en Kaggle (awsaf49)](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset)
(versión organizada del [CBIS-DDSM de TCIA](https://www.cancerimagingarchive.net/collection/cbis-ddsm/)).
Descargar, descomprimir y apuntar `archive_dir` (en el notebook) a la carpeta que contiene `csv/` y `jpeg/`.
**Los datos no se incluyen en el repositorio.**

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
```

Para usar GPU NVIDIA, instalar PyTorch con CUDA siguiendo https://pytorch.org/get-started/locally/
(el código usa la GPU automáticamente si está disponible; sin GPU funciona en CPU, más lento).

## Ejecución

1. Abrir `notebooks/proyecto_mamografia.ipynb`.
2. Editar `archive_dir` en la celda de configuración.
3. Ejecutar las celdas en orden. La primera ejecución del preprocesamiento tarda ~10–20 min y guarda su resultado
   en `outputs/processed/`.

Salidas en `outputs/`: `figures/` (figuras), `tables/` (métricas en CSV/JSON), `models/` (pesos de la U-Net).

## Pruebas

```bash
python -m pytest -q tests
```

Las pruebas generan un dataset sintético con la misma estructura que el de Kaggle y verifican el pipeline completo
(resolución de rutas, alineación, métodos de referencia, forma y entrenamiento de la U-Net).

## Reproducibilidad

Semilla fija (`Config.seed = 42`); la división train/validación es **por paciente**; el conjunto de test es el
oficial de CBIS-DDSM y solo se usa para la evaluación final.

## Limitaciones conocidas

* CBIS-DDSM proviene de mamografías de película digitalizada (años 90), no de mamografía digital moderna (FFDM).
* El recorte se construye a partir de la caja de la lesión (con un desplazamiento aleatorio `jitter`), por lo que la posición aproximada es conocida; el control "prior centrado" mide cuánto se explica solo por eso. Sin el desplazamiento, un círculo centrado alcanzaba Dice 0.857.
* En GPUs GTX 16xx la precisión mixta (`amp=True`) produjo pérdida NaN; por eso `amp=False` por defecto.
* La exactitud de la forma depende de la calidad de la máscara y de la variabilidad entre lectores de las etiquetas originales.

## Referencias

* Lee, R.S. et al. (2017). A curated mammography data set for use in computer-aided detection and diagnosis research. *Scientific Data*, 4, 170177.
* Ronneberger, O., Fischer, P., Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI. arXiv:1505.04597.
* American College of Radiology (2025). ACR BI-RADS® Manual, v2025.
