# Guía del repositorio

Qué hace cada archivo y dónde buscar cada cosa. Instalación, ejecución y resultados están en el
[README principal](../README.md); el detalle de las tablas y figuras está en [results/README.md](../results/README.md).

## Flujo general

```
dataset (CSV + JPEG)  ->  recortes ROI en caché  ->  entrenamiento  ->  evaluación  ->  results/
  src/data_index.py        src/preprocess.py         src/train.py      src/pipeline.py
                           outputs/processed/        outputs/runs/     src/metrics.py, src/stats.py
```

Todo lo pesado o regenerable (recortes `.npz`, pesos `.pt`, máscaras predichas, logs, zips) va a `outputs/`, que no se
versiona. `results/` guarda solo las tablas y figuras finales.

## Configuración (`configs/`)

| Archivo | Para qué |
|---|---|
| `main.yaml` | Configuración final: img 256, context 3, jitter 0.2, U-Net base 16, BCE+Dice, 40 épocas, batch 8, `amp: false` |
| `quick.yaml` | Igual que `main` con 3 épocas (demo y verificación rápida) |
| `ablations.yaml` | Variantes de E5 (context, jitter, canales base, pérdida) sobre `main.yaml` |
| `morph_unet.yaml`, `morph_unet_skip.yaml` | E12: `main.yaml` + bloque morfológico antes de la cabeza 1x1 o en el salto de máxima resolución |

Los YAML se agrupan por secciones (`data`, `preprocessing`, `model`, `training`), que `load_config` aplana; toda
clave tiene que ser un campo de `src/config.py:Config`.

## Librería (`src/`)

| Archivo | Contenido principal |
|---|---|
| `config.py` | `Config` (todos los parámetros), `load_config` (YAML + `MAMO_DATA_DIR` + sobrescrituras) |
| `data_index.py` | Lectura de los CSV de CBIS-DDSM y resolución de rutas de la versión de Kaggle |
| `preprocess.py` | Recortes ROI desde la mamografía completa y la máscara, desplazamiento aleatorio (`jitter`), en paralelo |
| `data_utils.py` | División por paciente (`patient_split`) y normalización |
| `dataset.py` | Tensores, lotes y aumento de datos en GPU |
| `model.py` | `UNet`, `MorphUNet` y `build_model(cfg)` (según `cfg.arch`) |
| `morph.py` | `PConv2d` (p-convolución estable vía softmax) y `MorphBlock` (apertura/cierre + residual) |
| `losses.py` | BCE + Dice y solo Dice (`get_loss`) |
| `train.py` | `train_unet`: AdamW, ReduceLROnPlateau, early stopping por Dice de validación |
| `metrics.py` | Dice, IoU, precisión, sensibilidad, `hausdorff95`, `n_components`, `extended_metrics` |
| `postprocess.py` | Posprocesamiento fijo: apertura + cierre con disco y componente conexa mayor |
| `stats.py` | `bootstrap_mean_ci` (por recorte o por paciente) y `paired_wilcoxon` |
| `classical.py` | Referencias: prior centrado y Otsu + morfología |
| `shape.py` | Descriptores geométricos, árbol de profundidad 3 y clasificadores de los experimentos de forma |
| `pipeline.py` | `prepare_data`, `run_experiment`, `cross_validate`, `cached_masks`, tablas por recorte |
| `viz.py`, `cli.py` | Figuras comunes; argumentos comunes de los scripts |

## Scripts (`scripts/`)

| Script | Experimento | Entrena |
|---|---|---|
| `download_data.py` | Descarga o verifica el dataset (`--check-only`) | — |
| `run_experiment.py` | E1, E6 (`--train-seed`), E12 (configs morph) | sí |
| `shape_ablation.py`, `shape_control.py`, `shape_diagnostics.py` | E2, E3, E4 | no |
| `run_ablations.py` | E5 (reanudable; reutiliza `main` como referencia) | sí |
| `cross_validate.py` | E7 (GroupKFold por paciente; reanudable por partición) | sí |
| `final_statistics.py` | E8 (semillas, CV, bootstrap, Wilcoxon) | no |
| `stratified_analysis.py` | E9 (subgrupos de test, Kruskal-Wallis + Holm) | no |
| `margin_experiment.py` | E10 (margen desde la geometría; reutiliza `run_task` de `shape_control.py`) | no |
| `failure_analysis.py` | E11 (modos de fallo y desplazamiento) | no |
| `compare_morphology.py` | E12 (ubicación del bloque, radio del posprocesamiento elegido en validación, comparación) | no |

## Notebooks (`notebooks/`)

| Archivo | Para qué |
|---|---|
| `proyecto_mamografia.ipynb` (de `build_notebook.py`) | Experimento principal paso a paso; entrena (`MAMO_EPOCHS=3` para la versión rápida) |
| `demo_resultados.ipynb` (de `build_demo.py`) | Sin entrenar: métricas, galería, forma BI-RADS y `mostrar_caso(i)` en menos de 1 min |

Los notebooks se versionan sin salidas; para cambiarlos se edita el `build_*.py` y se vuelve a generar.

## Pruebas (`tests/`)

| Archivo | Qué verifica |
|---|---|
| `make_synthetic_archive.py` | Crea un dataset sintético con la estructura de Kaggle (no es una prueba) |
| `test_pipeline.py` | Rutas, alineación, referencias, forma, entrenamiento corto |
| `test_scripts.py` | Config, `run_experiment.py` de punta a punta, modo sin test, errores claros, descarga sin red, preprocesamiento paralelo idéntico, pérdidas, CV por paciente, bootstrap y Wilcoxon |
| `test_morph.py` | p-convolución frente a `scipy.ndimage.grey_dilation/erosion`, gradientes, NaN, formas, `build_model` |
| `test_metrics_post.py` | Hausdorff 95, componentes conexas, posprocesamiento fijo |
| `test_demo.py` | Estructura del notebook de demostración y que esté al día con `build_demo.py` |

## Resultados, informe y documentación

| Ruta | Contenido |
|---|---|
| `results/` | Tablas y figuras finales; `README.md` describe cada archivo; `resumen_fase1.md`, `resumen_analisis.md` y `resumen_morfologia.md` son los resúmenes con interpretación |
| `report/` | Esqueleto LaTeX del informe (`main.tex`, `refs.bib`) y copia de las figuras finales en `figures/` |
| `docs/DATA.md` | Origen, estructura y particularidades del dataset |
| `outputs/` (no versionado) | `processed/` (recortes), `runs/<name>/` (cada ejecución), `logs/`, zips de entrega |
