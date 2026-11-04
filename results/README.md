# Resultados finales (versionados)

Tablas y figuras que usa el informe (`report/`). Provienen de la ejecución de la entrega parcial
(octubre de 2026, GTX 1650, `configs/main.yaml`: img 256, context 3, jitter 0.2, U-Net base 16, batch 8,
40 épocas, semilla 42). Todo lo regenerable y pesado (recortes, pesos) queda en `outputs/`, fuera de git.

Para regenerarlos (ver README principal):

| Experimento | Comando | Salida |
|---|---|---|
| **E1** Principal: U-Net + referencias + forma (árbol) | `python scripts/run_experiment.py --config configs/main.yaml` | `outputs/runs/main/` |
| **E2** Ablación de la forma (suavizado × clasificador) | `python scripts/shape_ablation.py` | `outputs/runs/main/shape_ablation/` |
| **E3** Control de los descriptores (patología, 2 clases, etiqueta) | `python scripts/shape_control.py` | `outputs/runs/main/shape_control/` |
| **E4** Diagnóstico (tamaño, ROUND vs IRREGULAR, emparejamiento) | `python scripts/shape_diagnostics.py` | `outputs/runs/main/shape_diagnostics/` |

E2 necesita los pesos de E1. E3 y E4 solo necesitan los recortes en caché (los genera E1). Los resultados de E1
pueden variar ligeramente entre ejecuciones en GPU (cuDNN no es completamente determinista).

> Los nombres de archivo de esta carpeta son los de la entrega parcial; cuando el nombre de la salida actual es
> distinto, se indica entre paréntesis.

## Tablas (`tables/`)

| Archivo | Exp. | Contenido |
|---|---|---|
| `metricas_segmentacion.csv` | E1 | Dice, IoU, precisión y sensibilidad (media ± DE) de prior centrado, Otsu+morfología y U-Net en val y test |
| `historial_entrenamiento.csv` | E1 | Pérdida de train, Dice de val, lr y segundos por época (`history.csv`) |
| `dice_por_muestra.csv` | E1 | Dice/IoU de la U-Net por recorte de test con su desplazamiento (`per_sample.csv`, que incluye los 3 métodos) |
| `dice_por_desplazamiento.csv` | E1 | Dice de la U-Net en test por tercil de \|desplazamiento\| y fracción con Dice < 0.1 |
| `metricas_forma.csv` | E1 | Forma BI-RADS (árbol, profundidad 3): exactitud, exactitud balanceada y F1 macro con máscara anotada y U-Net |
| `resumen.json` | E1 | Resumen numérico de E1 y configuración usada (`metrics.json`) |
| `estado_preprocesamiento.json` | E1 | Filas procesadas/descartadas por split (1318 train, 378 test, sin descartes) |
| `forma_experimentos.csv` | E2 | Exactitud, exactitud balanceada y F1 macro por σ ∈ {0,2,4} y clasificador (val); test solo para la combinación elegida |
| `forma_experimentos_resumen.json` | E2 | Combinación elegida en validación (σ=4, árbol) e importancia de descriptores (vacía: sin señal) |
| `control_forma.csv` | E3 | AUC y exactitud balanceada de las tareas A, B y C (val por opción, test para la elegida) |
| `control_forma_test.json` | E3 | Solo las filas de test de `control_forma.csv` |
| `diagnostico_descriptores.csv` | E4 | AUC del tamaño -> patología y de ROUND vs IRREGULAR (val y test) |
| `diagnostico_descriptores.json` | E4 | Comprobación de emparejamiento meta↔CSV↔máscara (100 % de coincidencias) y medianas de descriptores |

## Figuras (`figures/`)

| Archivo | Exp. | Contenido |
|---|---|---|
| `muestras_recortes.png` | E1 | 12 recortes de train con el contorno anotado (verde) |
| `hist_area_lesion.png` | E1 | Distribución de la fracción del recorte ocupada por la lesión (notebook) |
| `curvas_entrenamiento.png` | E1 | Pérdida de train y Dice de validación por época |
| `predicciones_test.png` | E1 | 12 recortes de test: anotación (verde) y U-Net (rojo) |
| `peores_casos_test.png` | E1 | Los 8 recortes de test con menor Dice (fallos de localización) |
| `descriptores_por_forma.png` | E1 | Boxplots de los 8 descriptores geométricos por forma (train) |
| `confusion_forma_anotada.png` | E1 | Matriz de confusión de la forma en test desde la máscara anotada |
| `confusion_forma_unet.png` | E1 | Ídem desde la máscara de la U-Net |
| `forma_exactitud_balanceada.png` | E2 | Exactitud balanceada en val por σ y clasificador |
| `confusion_forma_mejor_anotada.png` | E2 | Matriz de confusión en test de la combinación elegida (máscara anotada) |
| `confusion_forma_mejor_unet.png` | E2 | Ídem con la máscara de la U-Net |
| `control_forma_roc.png` | E3 | Curvas ROC en test: patología (descriptores / etiqueta / ambos) y forma regular vs irregular |
| `diagnostico_mascaras.png` | E4 | 8 máscaras ROUND y 8 IRREGULAR de train con circularidad, solidez y lóbulos |
