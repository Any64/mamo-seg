# Segmentación de masas mamarias (CBIS-DDSM) y caracterización de su forma según BI-RADS 2025

Proyecto del curso **CBM423 – Procesamiento Digital de Imágenes Médicas** (Universidad de Valparaíso).

## Descripción

1. **Segmentación:** una U-Net segmenta masas mamarias en recortes ROI de mamografías del dataset CBIS-DDSM.
2. **Comparación:** su desempeño (Dice, IoU) se compara con un método clásico de procesamiento de imágenes
   (suavizado + umbral de Otsu + morfología matemática) y con un control "prior centrado" (círculo fijo en el centro).
3. **Forma BI-RADS:** a partir de la máscara se calculan descriptores geométricos del contorno y un árbol de
   decisión de profundidad 3 (interpretable) asigna la forma `ROUND / OVAL / LOBULATED / IRREGULAR` (léxico BI-RADS
   v2025). Se valida contra el campo `mass shape` del dataset, con máscaras anotadas y con máscaras de la U-Net.

**Resultado principal:** la U-Net alcanza Dice 0.771 en el test oficial (vs. 0.35 de las referencias). La forma
BI-RADS, en cambio, queda al nivel del azar incluso con las máscaras anotadas; los experimentos de control muestran
que es una limitación de las máscaras de CBIS-DDSM y no un error del pipeline (ver [Resultados](#resultados)).

> **Decisiones de diseño.** (1) En CBIS-DDSM la máscara ROI está en las coordenadas de la mamografía *completa* y no
> está registrada con el recorte oficial: por eso se generan recortes propios desde la mamografía completa y la
> máscara (`src/preprocess.py`). (2) El centro del recorte se desplaza al azar (`jitter=0.2`): sin ese
> desplazamiento, un círculo centrado lograba Dice 0.857 y la tarea era casi trivial. (3) La división
> train/validación es por paciente y el test oficial solo se usa en la evaluación final.

## Estructura del repositorio

```
configs/        configuraciones YAML de los experimentos (main.yaml = resultados del informe; quick.yaml = demo)
src/            librería
  config.py       parámetros (dataclass Config) y carga de YAML
  data_index.py   lectura de CSV y resolución de rutas (particularidades de la versión de Kaggle)
  preprocess.py   recortes ROI alineados, etiquetas de forma, chequeo de alineación
  data_utils.py   división por paciente, normalización
  dataset.py      tensores y aumento de datos (torch)
  model.py        U-Net
  losses.py       BCE + Dice
  metrics.py      Dice, IoU, precisión, sensibilidad
  train.py        entrenamiento con early stopping
  classical.py    métodos de referencia (prior centrado; Otsu + morfología)
  shape.py        descriptores geométricos, árbol de decisión y clasificadores de los experimentos de forma
  pipeline.py     pipeline completo (lo comparten el script principal y el notebook)
  viz.py, cli.py  figuras; argumentos comunes de los scripts
scripts/        ejecutables
  download_data.py      descarga/verifica el dataset
  run_experiment.py     experimento principal (preprocesa, entrena, evalúa, guarda)
  shape_ablation.py     forma: suavizado del contorno × clasificador
  shape_control.py      forma: controles (patología, 2 clases, etiqueta vs geometría)
  shape_diagnostics.py  forma: tamaño, extremos ROUND/IRREGULAR, emparejamiento de datos
notebooks/      demostración paso a paso (proyecto_mamografia.ipynb; build_notebook.py lo genera)
tests/          pruebas con un dataset sintético (sin datos reales ni red)
results/        tablas y figuras finales usadas en el informe (versionadas; ver results/README.md)
docs/DATA.md    origen, estructura y particularidades del dataset
report/         informe (LaTeX)
outputs/        todo lo generado: caché de recortes, pesos, ejecuciones (NO versionado)
```

## Requisitos

* Python 3.10 o superior (probado con 3.11.9 en Windows 11).
* Paquetes de `requirements.txt` (numpy, scipy, pandas, scikit-image, scikit-learn, matplotlib, PyTorch, PyYAML,
  Jupyter, pytest).
* GPU NVIDIA recomendada (probado con una GTX 1650 de 4 GB: ~16 s por época). Sin GPU funciona en CPU, mucho más
  lento.
* ~6 GB de disco para el dataset y ~100 MB para `outputs/`.

## Instalación

```powershell
git clone <url-del-repositorio> mamo-seg
cd mamo-seg
python -m venv .venv
.venv\Scripts\Activate.ps1          # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

**GPU (NVIDIA):** `pip install -r requirements.txt` puede instalar la versión de PyTorch solo para CPU. Para usar la
GPU, instalar PyTorch con CUDA con el comando que indica https://pytorch.org/get-started/locally/ para tu sistema
(el proyecto se probó con `torch 2.14.1+cu132`) y comprobar:

```powershell
python -c "import torch; print(torch.cuda.is_available())"   # debe imprimir True
```

**Problemas conocidos en Windows:**
* `OSError: [WinError 4551] Una directiva de Control de aplicaciones bloqueó este archivo` al importar torch:
  *Smart App Control* de Windows 11 bloquea DLL de PyTorch no firmadas. Se resuelve desactivándolo en
  Seguridad de Windows → Control de aplicaciones y navegador (decisión del usuario; puede no ser reversible sin
  reinstalar Windows).
* En GPUs GTX 16xx la precisión mixta produce pérdida NaN: mantener `amp: false` (valor por defecto).

## Datos

Dataset: [CBIS-DDSM en Kaggle (awsaf49)](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset),
versión en JPEG del [CBIS-DDSM de TCIA](https://www.cancerimagingarchive.net/collection/cbis-ddsm/).
**Los datos no se incluyen en el repositorio.**

```powershell
python scripts/download_data.py      # con credenciales de Kaggle descarga (~5 GB); si no, muestra instrucciones
```

Por defecto el dataset se busca en `data/CBIS-DDSM/` (carpetas `csv/` y `jpeg/`). Si está en otra carpeta:

```powershell
$env:MAMO_DATA_DIR = "D:\datos\CBIS-DDSM"     # o: setx MAMO_DATA_DIR "D:\datos\CBIS-DDSM" (permanente)
python scripts/download_data.py --check-only  # verifica la estructura
```

Detalles y particularidades del dataset en [docs/DATA.md](docs/DATA.md).

## Ejecución

### Demo rápida (3 épocas)

```powershell
python scripts/run_experiment.py --config configs/quick.yaml
```

La primera ejecución genera los recortes (~11 min, se guardan en `outputs/processed/` y se reutilizan); luego
entrena 3 épocas (~1 min en GPU). Resultados en `outputs/runs/quick/`. Con 3 épocas la U-Net ya supera a las
referencias (Dice test ≈ 0.6).

Alternativa interactiva: abrir `notebooks/proyecto_mamografia.ipynb` (usa las mismas funciones de `src/`;
`MAMO_EPOCHS=3` para la versión rápida).

### Reproducir los resultados principales

```powershell
python scripts/run_experiment.py --config configs/main.yaml   # E1: ~13 min en GTX 1650 (con recortes en caché)
python scripts/shape_ablation.py                              # E2: necesita los pesos de E1 (~3 min)
python scripts/shape_control.py                               # E3 (~2 min)
python scripts/shape_diagnostics.py                           # E4 (~1 min)
```

Opciones de `run_experiment.py`: `--seed N`, `--epochs N`, `--name <carpeta>`, `--data-dir <ruta>`,
`--no-figures`. Cada ejecución guarda en `outputs/runs/<name>/`: `config.yaml` (configuración usada),
`metrics.json`, `per_sample.csv` (métricas por recorte de los 3 métodos), `history.csv`, `train.log`,
`unet_best.pt`, tablas CSV y `figures/`. La correspondencia con `results/` está en
[results/README.md](results/README.md).

### Pruebas

```powershell
python -m pytest -q tests
```

Generan un dataset sintético con la misma estructura que el de Kaggle y verifican el pipeline completo (resolución de
rutas, alineación, referencias, forma, entrenamiento, `run_experiment.py` de punta a punta y `download_data.py` sin
red). Tardan ~1 min.

## Resultados

Test oficial de CBIS-DDSM (378 masas); validación: 257 masas de pacientes no vistos en entrenamiento
(entrenamiento: 1061). Media ± desviación estándar por recorte.

**Segmentación (E1)**

| Método | Dice val | Dice test | IoU test |
|---|---|---|---|
| Prior centrado | 0.344 | 0.352 ± 0.211 | 0.236 |
| Clásico (Otsu + morfología) | 0.348 | 0.324 ± 0.225 | 0.220 |
| **U-Net** | **0.784** | **0.771 ± 0.204** | **0.662** |

La U-Net se entrenó 40 épocas (mejor Dice de validación 0.784 en la época 37). El 3.4 % de los recortes de test
tiene Dice < 0.1: son fallos de localización (segmenta otra estructura densa), concentrados en las lesiones más
desplazadas del centro.

**Forma BI-RADS (4 clases; azar = 0.25)**

| Experimento | Máscara | Exactitud | Exactitud balanceada | F1 macro |
|---|---|---|---|---|
| E1: árbol de profundidad 3, test | anotada | 0.244 | 0.220 | 0.215 |
| E1: árbol de profundidad 3, test | U-Net | 0.247 | 0.249 | 0.172 |
| E2: mejor combinación en val (σ=4, árbol), test | anotada | 0.269 | 0.263 | 0.235 |
| E2: ídem | U-Net | 0.269 | 0.294 | 0.219 |
| Clase mayoritaria, test | – | 0.281 | 0.250 | 0.110 |

**Controles (E3, E4; AUC en test)**

| Pregunta | AUC test |
|---|---|
| Descriptores → patología (benigna / maligna) | 0.506 |
| Descriptores → forma regular vs irregular | 0.493 |
| Descriptores → ROUND vs IRREGULAR | 0.58 |
| Tamaño de la lesión → patología | 0.60 |
| **Etiqueta de forma del radiólogo → patología** | **0.780** |

Interpretación: la etiqueta de forma sí contiene información (predice la patología con AUC 0.78), pero la geometría
de las máscaras de CBIS-DDSM no la refleja; las máscaras son contornos toscos o figuras trazadas con herramienta,
independientes de la forma anotada. El emparejamiento recorte↔máscara↔fila del CSV se verificó al 100 % (E4).

## Limitaciones

* CBIS-DDSM proviene de mamografías de película digitalizada (años 90), no de mamografía digital moderna (FFDM).
* El recorte se construye a partir de la caja de la lesión (con un desplazamiento aleatorio), por lo que la posición
  aproximada es conocida; el control "prior centrado" mide cuánto se explica solo por eso. No es un sistema de
  detección.
* Las máscaras ROI de CBIS-DDSM no representan con fidelidad el contorno de la lesión, lo que impide derivar la forma
  BI-RADS desde la geometría (objetivo 3) en este dataset, con independencia de la calidad de la segmentación.
* Las etiquetas de forma tienen variabilidad entre lectores y no se dispone de una segunda lectura.
* Los resultados en GPU pueden variar ligeramente entre ejecuciones (cuDNN no es completamente determinista).

## Referencias

* Lee, R.S., Gimenez, F., Hoogi, A., Miyake, K.K., Gorovoy, M., Rubin, D.L. (2017). A curated mammography data set
  for use in computer-aided detection and diagnosis research. *Scientific Data*, 4, 170177.
* Ronneberger, O., Fischer, P., Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation.
  MICCAI. arXiv:1505.04597.
* Otsu, N. (1979). A threshold selection method from gray-level histograms. *IEEE Trans. Syst. Man Cybern.*, 9(1),
  62–66.
* American College of Radiology (2025). ACR BI-RADS® Atlas, v2025.
