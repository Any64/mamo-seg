# Segmentación de masas mamarias (CBIS-DDSM) y caracterización de su forma según BI-RADS 2025

Proyecto del curso **CBM423 – Procesamiento Digital de Imágenes Médicas** (Universidad de Valparaíso).

## Descripción

1. **Segmentación:** una U-Net segmenta masas mamarias en recortes ROI de mamografías del dataset CBIS-DDSM.
2. **Comparación:** su desempeño (Dice, IoU, Hausdorff 95) se compara con un método clásico de procesamiento de
   imágenes (suavizado + umbral de Otsu + morfología matemática) y con un control "prior centrado" (círculo fijo en
   el centro).
3. **Forma y margen BI-RADS:** a partir de la máscara se calculan descriptores geométricos del contorno y un árbol de
   decisión de profundidad 3 (interpretable) asigna la forma `ROUND / OVAL / LOBULATED / IRREGULAR` (léxico BI-RADS
   v2025). Se valida contra los campos `mass shape` y `mass margins` del dataset.

**Resultado principal:** la U-Net alcanza un Dice de **0.762 ± 0.009 en el test oficial** (media ± DE de 3 semillas;
validación cruzada por paciente: 0.771 ± 0.014), frente a 0.32–0.35 de las referencias. La forma y el margen BI-RADS,
en cambio, no se recuperan desde la geometría de las máscaras, ni siquiera con las máscaras anotadas: los controles
muestran que es una limitación de las máscaras de CBIS-DDSM y no un error del pipeline (ver [Resultados](#resultados)).

**Informe final:** [report/main.pdf](report/main.pdf). Ya está compilado. Para recompilarlo desde `report/main.tex`, ejecutar
`pdflatex main`, `bibtex main` y dos veces más `pdflatex main` dentro de `report/` (ver
[report/README.md](report/README.md)).

> **Decisiones de diseño.**
> 1. En CBIS-DDSM la máscara ROI está en las coordenadas de la mamografía *completa* y no está registrada con el
>    recorte oficial. Por eso se generan recortes propios desde la mamografía completa y la máscara
>    (`src/preprocess.py`).
> 2. El centro del recorte se desplaza al azar (`jitter=0.2`). Sin ese desplazamiento, un círculo centrado logra un
>    Dice de 0.857 y supera a la U-Net (ablación E5), así que la tarea sería casi trivial.
> 3. La división train/validación es por paciente. Todas las comparaciones y selecciones se hacen con validación (o
>    con validación cruzada sobre el train oficial); el test oficial solo se reporta para la configuración final.

## Estructura del repositorio

```
configs/        configuraciones YAML (main.yaml = configuración final; quick.yaml = demo; ablations.yaml;
                morph_unet*.yaml = U-Net con bloque morfológico)
src/            librería
  config.py       parámetros (dataclass Config) y carga de YAML
  data_index.py   lectura de CSV y resolución de rutas (particularidades de la versión de Kaggle)
  preprocess.py   recortes ROI alineados (en paralelo), etiquetas de forma, chequeo de alineación
  data_utils.py   división por paciente, normalización
  dataset.py      tensores y aumento de datos (torch)
  model.py        U-Net y Morph-UNet (build_model según cfg.arch)
  morph.py        capas morfológicas aprendibles (p-convolución / media contra-armónica)
  losses.py       BCE + Dice y solo Dice
  metrics.py      Dice, IoU, precisión, sensibilidad, Hausdorff 95, componentes conexas
  postprocess.py  posprocesamiento morfológico fijo (apertura/cierre + componente mayor)
  stats.py        IC bootstrap (por recorte y por paciente) y Wilcoxon pareado
  train.py        entrenamiento con early stopping
  classical.py    métodos de referencia (prior centrado; Otsu + morfología)
  shape.py        descriptores geométricos, árbol de decisión y clasificadores
  pipeline.py     pipeline completo, validación cruzada por paciente, caché de predicciones
  viz.py, cli.py  figuras; argumentos comunes de los scripts
scripts/        ejecutables (ver la tabla de experimentos)
notebooks/      proyecto_mamografia.ipynb (paso a paso, entrena) y demo_resultados.ipynb (sin entrenar);
                build_notebook.py y build_demo.py los generan
tests/          pruebas con un dataset sintético (sin datos reales ni red)
results/        tablas y figuras finales (versionadas; ver results/README.md)
report/         informe final: main.pdf (compilado), main.tex, refs.bib y figuras
docs/           DATA.md (dataset) y GUIA_REPOSITORIO.md (archivo por archivo)
outputs/        todo lo generado: caché de recortes, pesos, ejecuciones, logs (NO versionado)
```

Guía detallada en [docs/GUIA_REPOSITORIO.md](docs/GUIA_REPOSITORIO.md).

## Requisitos

* Python 3.10 o superior (probado con 3.11.9 en Windows 11).
* Paquetes de `requirements.txt` (numpy, scipy, pandas, scikit-image, scikit-learn, matplotlib, PyTorch, PyYAML,
  Jupyter, pytest).
* GPU NVIDIA recomendada (probado con una GTX 1650 de 4 GB: ~16 s por época con la U-Net, ~29 s con SegFormer-B0 y ~60 s con la Morph-UNet).
  Sin GPU funciona en CPU, mucho más lento.
* ~6 GB de disco para el dataset y ~1 GB para `outputs/`.

## Instalación

```powershell
git clone <url-del-repositorio> mamo-seg
cd mamo-seg
python -m venv .venv
.venv\Scripts\Activate.ps1          # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

**Opcional, solo para SegFormer-B0 (E13):** `pip install "transformers>=4.40"` (probado con 5.24.1). La primera
ejecución descarga una vez los pesos preentrenados `nvidia/mit-b0` desde Hugging Face a la caché del usuario; no se
versionan. Sin `transformers`, todo lo demás funciona y sus pruebas se omiten.

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
* `ImportError: DLL load failed ... El nombre del archivo o la extensión es demasiado largo`: la ruta del
  repositorio (y del `.venv` dentro de él) supera el límite de 260 caracteres de Windows. Clonar en una ruta más
  corta (p. ej. `C:\proyectos\mamo-seg`) o habilitar las rutas largas de Windows.

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

### Demo rápida

```powershell
python scripts/run_experiment.py --config configs/quick.yaml     # entrena 3 épocas (~1 min en GPU)
```

La primera ejecución genera los recortes (~3 min con 8 procesos; se guardan en `outputs/processed/` y se
reutilizan). Para ver los resultados del modelo final **sin entrenar** (necesita los pesos de E1), abrir
`notebooks/demo_resultados.ipynb`: métricas, galería de aciertos y fallos, forma BI-RADS y `mostrar_caso(i)`
(< 1 min). El notebook `proyecto_mamografia.ipynb` recorre el experimento principal paso a paso (`MAMO_EPOCHS=3`
para la versión rápida).

### Experimentos

Tiempos medidos en una GTX 1650 de 4 GB con los recortes en caché. Cada entrenamiento guarda en
`outputs/runs/<name>/`: `config.yaml`, `metrics.json`, `per_sample.csv`, `history.csv`, `train.log`,
`unet_best.pt`, tablas y `figures/`. La correspondencia con `results/` está en [results/README.md](results/README.md).

| Exp. | Qué hace | Comando | Salida | Tiempo |
|---|---|---|---|---|
| E1 | Principal: U-Net + referencias + forma (semilla 42) | `python scripts/run_experiment.py --config configs/main.yaml` | `outputs/runs/main/` | ~13 min |
| E2 | Forma: suavizado × clasificador (necesita E1) | `python scripts/shape_ablation.py` | `outputs/runs/main/shape_ablation/` | ~2 min |
| E3 | Forma: controles (patología, 2 clases, etiqueta) | `python scripts/shape_control.py` | `outputs/runs/main/shape_control/` | ~1 min |
| E4 | Forma: tamaño, ROUND vs IRREGULAR, emparejamiento | `python scripts/shape_diagnostics.py` | `outputs/runs/main/shape_diagnostics/` | < 1 min |
| E5 | Ablaciones en validación: context, jitter, canales, pérdida | `python scripts/run_ablations.py` | `outputs/runs/abl_*/`, `results/ablaciones.*` | ~1 h 45 min |
| E6 | Semillas 43 y 44 (42 = E1) | `python scripts/run_experiment.py --config configs/main.yaml --train-seed 43 --name final_seed43` (ídem 44) | `outputs/runs/final_seed4*/` | ~12 min c/u |
| E7 | Validación cruzada de 5 particiones por paciente (train oficial) | `python scripts/cross_validate.py --config configs/main.yaml --folds 5` | `outputs/runs/main_cv5/` | ~55 min |
| E8 | Variabilidad e incertidumbre (bootstrap, Wilcoxon) | `python scripts/final_statistics.py --runs main final_seed43 final_seed44 --cv main_cv5` | `results/` | < 1 min |
| E9 | Análisis estratificado en test | `python scripts/stratified_analysis.py` | `results/estratificado_*` | < 1 min* |
| E10 | Margen BI-RADS desde la geometría | `python scripts/margin_experiment.py` | `results/margen_*` | ~1 min |
| E11 | Análisis de fallos (Dice < 0.1) | `python scripts/failure_analysis.py` | `results/fallos_*` | < 1 min* |
| E12 | Morph-UNet y posprocesamiento fijo | `python scripts/run_experiment.py --config configs/morph_unet.yaml --no-test --name morph_unet_s42` (y `morph_unet_skip.yaml`; semillas con `--train-seed`), luego `python scripts/compare_morphology.py ...` (ver su docstring) | `outputs/runs/morph_*`, `results/morfologia_*` | ~35–40 min por entrenamiento (59 s/época), comparación ~5 min |
| E13 | SegFormer-B0 (transformer, codificador preentrenado; requiere `transformers`) | `python scripts/run_experiment.py --config configs/segformer_b0.yaml --no-test --lr 1e-4 --name segformer_lr1e-4` (ídem 3e-4; semillas 43/44 con `--train-seed`), luego `python scripts/compare_segformer.py ...` (ver su docstring) | `outputs/runs/segformer_*`, `results/segformer_*` | ~11–19 min por entrenamiento (29–34 s/época), comparación ~5 min |
| E13b | Controles de E13: SegFormer-B0 sin preentrenar (3 semillas) y U-Net con lr 3e-4 (semilla 42, solo val) | `python scripts/run_experiment.py --config configs/segformer_b0_scratch.yaml --no-test --train-seed 42 --name segformer_scratch_s42` (ídem 43/44) y `python scripts/run_experiment.py --config configs/main.yaml --no-test --lr 3e-4 --name unet_lr3e-4`, luego `python scripts/compare_segformer_controls.py ...` (ver su docstring) | `outputs/runs/segformer_scratch_*`, `outputs/runs/unet_lr3e-4`, `results/segformer_controles_*`, `results/unet_control_lr.csv` | ~20–23 min por SegFormer sin preentrenar (40 épocas), ~15 min la U-Net, comparación ~5 min |

\* E9 y E11 predicen una vez con los modelos de E1/E6 y guardan las máscaras en `outputs/runs/<run>/pred_masks.npz`.

Opciones de `run_experiment.py`: `--seed N` (datos), `--train-seed N` (solo pesos, lotes y aumento), `--epochs N`,
`--name <carpeta>`, `--data-dir <ruta>`, `--no-test` (solo validación), `--no-figures`.

### Pruebas

```powershell
python -m pytest -q tests
```

Generan un dataset sintético con la misma estructura que el de Kaggle y verifican el pipeline completo (rutas,
alineación, referencias, forma, entrenamiento, `run_experiment.py` de punta a punta, validación cruzada por
paciente, estadística, capas morfológicas frente a scipy, SegFormer-B0 sin descarga, Hausdorff 95, posprocesamiento, notebook de demostración y
`download_data.py` sin red). 47 pruebas (4 se omiten sin `transformers`), ~1 min.

## Resultados

Test oficial de CBIS-DDSM: 378 masas de 201 pacientes. Validación: 257 masas de pacientes no vistos en el
entrenamiento (1061 masas).

### Tabla final (cifra de referencia: media ± DE de 3 semillas de entrenamiento)

| Método | Dice val | Dice test | IoU test | HD95 test (px, mediana) | Dice < 0.1 (test) |
|---|---|---|---|---|---|
| Prior centrado | 0.344 | 0.352 | 0.236 | – | – |
| Clásico (Otsu + morfología) | 0.348 | 0.324 | 0.220 | – | – |
| **U-Net** (configuración final) | **0.777 ± 0.003** | **0.762 ± 0.009** | **0.652 ± 0.011** | **15.4 ± 1.4** | 3.5 % ± 0.8 % |

- **Validación cruzada de 5 particiones agrupada por paciente** (train oficial, E7): U-Net 0.771 ± 0.014 y prior
  0.348 ± 0.021.
- **Incertidumbre en test** (semilla 42, E8): IC 95 % por bootstrap de 1000 remuestreos para U-Net 0.769 [0.746, 0.791]
  y clásico 0.324 [0.302, 0.348]. La diferencia U-Net − clásico es +0.445 [0.420, 0.469] (también remuestreando
  pacientes); Wilcoxon pareado p ≈ 5·10⁻⁶⁰, con la U-Net mejor en el 94 % de los recortes.
- Las referencias son deterministas, así que no tienen DE entre semillas.
- El HD95 está en píxeles del recorte de 256 × 256; las predicciones vacías (2–3 por semilla) se cuentan aparte.

### Ablaciones (E5, solo validación, semilla 42)

| Variante | Dice U-Net | Dice prior | Lectura |
|---|---|---|---|
| Referencia (context 3, jitter 0.2, base 16, BCE+Dice) | 0.781 | 0.345 | — |
| context 2 / 4 | 0.802 / 0.762 | 0.531 / 0.207 | la tarea se hace más fácil o más difícil; no se usa para elegir |
| jitter 0 / 0.1 / 0.3 | 0.839 / 0.825 / 0.753 | **0.857** / 0.631 / 0.161 | sin jitter, el prior supera a la U-Net (atajo de posición) |
| base 32 | 0.767 | 0.345 | no mejora, con 2,3 veces más tiempo por época |
| solo Dice | 0.767 | 0.345 | no mejora |

### Dónde falla (E9, E11)

- **Subgrupos de test:** el Dice baja con la sutileza (de 0.82 con sutileza 5 a 0.58 con sutileza 1) y con la
  densidad mamaria (de 0.85 con densidad 1 a 0.68 con densidad 4); ambas son significativas tras Holm.
- **No influyen:** la patología, la vista ni la forma básica.
- **Fallos (Dice < 0.1, 16 de 378 recortes):** casi todos son de localización, porque la red segmenta otra
  estructura densa. Se asocian a lesiones más desplazadas, más pequeñas y más sutiles; el desplazamiento no
  los explica por sí solo.

### Forma y margen BI-RADS (E1–E4, E10; azar: exactitud 0.25, AUC 0.5)

| Pregunta (test) | Resultado |
|---|---|
| Forma en 4 clases, árbol de profundidad 3, máscara anotada / U-Net (E1) | exactitud balanceada 0.220 / 0.249 |
| Mejor combinación en validación (σ = 4, árbol), máscara anotada / U-Net (E2) | 0.263 / 0.294 |
| Descriptores → patología / forma regular vs irregular (E3) | AUC 0.506 / 0.493 |
| Descriptores → margen no circunscrito / espiculado (E10) | AUC 0.646 / 0.500 |
| **Etiqueta de forma del radiólogo → patología** (E3) | **AUC 0.780** |
| **Etiqueta de margen del radiólogo → patología** (E10) | **AUC 0.782** |

**Interpretación:** las etiquetas de forma y de margen sí contienen información, ya que predicen la patología con AUC
≈ 0.78. Pero la geometría de las máscaras de CBIS-DDSM no la refleja: son contornos toscos, sin espículas e
independientes de la forma anotada. El emparejamiento recorte ↔ máscara ↔ fila del CSV se verificó al 100 % (E4).

### Morfología (E12; media ± DE de 3 semillas; detalle en `results/resumen_morfologia.md`)

| Método | Dice val | Dice test | HD95 test (px, media) | > 1 componente (test) |
|---|---|---|---|---|
| U-Net | 0.777 ± 0.003 | 0.762 ± 0.009 | 25.6 ± 1.9 | 32 % |
| **U-Net + posproc. fijo** (r = 6, elegido en val) | **0.791 ± 0.006** | **0.767 ± 0.007** | **18.3 ± 0.9** | 0 % |
| Morph-UNet (bloque en el salto, +628 parámetros) | 0.777 ± 0.000 | 0.762 ± 0.001 | 26.6 ± 0.9 | 35 % |

- **Morph-UNet:** la morfología aprendible dentro de la red no mejora; empata con la U-Net y tarda 3,7 veces más
  por época.
- **Posprocesamiento fijo:** baja el HD95 de forma significativa en val y en test. El Dice solo mejora en val (en test,
  IC [−0.000, 0.009]). La ganancia viene casi toda de quedarse con la componente conexa mayor.
- **Forma:** las máscaras de las redes se clasifican como LOBULATED en el 63–82 % de los casos, frente al 24 % de las
  etiquetas reales.

### SegFormer-B0 (E13; media ± DE de 3 semillas; detalle en `results/resumen_segformer.md`)

| Método | Dice val | Dice test | HD95 test (px, mediana) | Vacías (test) | Dice < 0.1 (test) |
|---|---|---|---|---|---|
| U-Net | 0.777 ± 0.003 | 0.762 ± 0.009 | 15.4 | 2.3 | 13.3 |
| **SegFormer-B0** (3.7 M parámetros, lr 3e-4 elegido en val) | **0.810 ± 0.001** | **0.792 ± 0.003** | **12.8** | 0 | 10.0 |
| SegFormer-B0 + posproc. fijo (r = 6) | 0.810 ± 0.003 | 0.793 ± 0.003 | 12.8 | 0 | 11.7 |

- **Mejora:** SegFormer − U-Net en test: Dice +0.030 [IC 95 % por paciente 0.019, 0.041], Wilcoxon p < 1e-4. Viene de
  la sensibilidad (0.87 frente a 0.80) y de no dejar predicciones vacías.
- **La hipótesis de localización no se confirma:** los fallos en "otra zona" no bajan de forma significativa en test
  (5.3 frente a 4.0 por semilla, McNemar p = 0.48) y en validación suben (1.0 frente a 4.3).
- **Cautela:** el codificador está preentrenado en ImageNet y la U-Net no, así que la mejora no se atribuye solo a la
  atención global.
- **Forma BI-RADS:** no cambia.

**Controles (E13b; detalle en la sección 5 de `results/resumen_segformer.md`):**

| Método | Dice val | Dice test | HD95 test (px, mediana) | Vacías (test) | Dice < 0.1 (test) |
|---|---|---|---|---|---|
| SegFormer-B0 sin preentrenar (pesos aleatorios, lr 3e-4) | 0.801 ± 0.005 | 0.783 ± 0.001 | 12.9 | 0 | 11.7 |
| U-Net con lr 3e-4 (solo semilla 42, solo val) | 0.784 (frente a 0.781 con lr 1e-3) | — | — | — | — |

- **Desglose del Dice de validación:** U-Net 0.777 → SegFormer sin preentrenar 0.801 → SegFormer preentrenado 0.810.
  - Arquitectura: +0.024 [IC 95 % por paciente 0.012, 0.038], p = 0.002 (~74 % de la ventaja).
  - Preentrenamiento: +0.008 [−0.002, 0.019], p = 0.86 (~26 %; no concluyente).
  - En test (solo se informa): +0.020 [0.010, 0.031] y +0.010 [−0.001, 0.020].
- **Conclusión:** la mayor parte de la mejora se atribuye a la arquitectura (que también difiere en tamaño). El
  preentrenamiento sobre todo acelera la convergencia. El lr de la U-Net no explica la diferencia (+0.003).

## Limitaciones

* CBIS-DDSM proviene de mamografías de película digitalizada (años 90), no de mamografía digital moderna (FFDM).
* El recorte se construye a partir de la caja de la lesión, con un desplazamiento aleatorio, así que la posición
  aproximada es conocida. El control "prior centrado" mide cuánto se explica solo por eso. No es un sistema de
  detección.
* Las máscaras ROI de CBIS-DDSM no representan con fidelidad el contorno. Eso impide derivar la forma y el margen
  BI-RADS desde la geometría en este dataset, con independencia de la calidad de la segmentación.
* Las etiquetas de forma y margen tienen variabilidad entre lectores, y no hay una segunda lectura.
* Las ablaciones usan una sola semilla y la variabilidad entre semillas se mide con solo 3. Por eso las diferencias
  de alrededor de 0.01 entre variantes no se distinguen del ruido.
* En el análisis estratificado, varios recortes son del mismo paciente y las variables están correlacionadas (por
  ejemplo, densidad y sutileza). Por eso los p-valores son optimistas.
* Los resultados en GPU varían ligeramente entre ejecuciones, porque cuDNN no es completamente determinista. Una
  nueva ejecución de E1 dio un Dice en test de 0.769, frente a 0.771 en la entrega parcial.
* E13 compara una red preentrenada (SegFormer-B0, ImageNet) con una U-Net entrenada desde cero, con distinto número
  de parámetros; el lr solo se ajustó para SegFormer (2 valores).
* Controles de E13 (E13b):
  * Los 4 modelos nuevos (3 SegFormer-B0 sin preentrenar y la U-Net con lr 3e-4) llegaron al tope de 40 épocas, con la
    mejor época en la 39–40. La comparación vale "con un presupuesto de 40 épocas" y no dice dónde convergerían.
  * La época 1 de cada SegFormer fue más lenta (42–48 s) por el calentamiento de CUDA; desde la época 2, 28–34 s.
  * El control de lr de la U-Net es de una sola semilla (42) y solo en validación.
* E12 compara la Morph-UNet con solo 3 semillas y 40 épocas; la semilla 43 alcanzó su mejor época en la última.

## Referencias

* Lee, R.S., Gimenez, F., Hoogi, A., Miyake, K.K., Gorovoy, M., Rubin, D.L. (2017). A curated mammography data set
  for use in computer-aided detection and diagnosis research. *Scientific Data*, 4, 170177.
* Ronneberger, O., Fischer, P., Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation.
  MICCAI. arXiv:1505.04597.
* Otsu, N. (1979). A threshold selection method from gray-level histograms. *IEEE Trans. Syst. Man Cybern.*, 9(1),
  62–66.
* Masci, J., Angulo, J., Schmidhuber, J. (2013). A learning framework for morphological operators using
  counter-harmonic mean. *ISMM 2013*, LNCS 7883, 329–340.
* American College of Radiology (2025). ACR BI-RADS® Atlas, v2025.
