"""Genera notebooks/proyecto_mamografia.ipynb (ejecutar tras modificar este archivo).

El notebook es una demostración paso a paso: toda la lógica vive en ``src/``
(principalmente ``src/pipeline.py``), la misma que usa ``scripts/run_experiment.py``.
"""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip("\n")))  # noqa: E731
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip("\n")))  # noqa: E731

md("""
# Segmentación de masas mamarias (CBIS-DDSM) y caracterización de su forma (BI-RADS 2025)

**CBM423 – Procesamiento Digital de Imágenes Médicas · Universidad de Valparaíso**

Demostración paso a paso del experimento principal. Toda la lógica vive en `src/` (este notebook solo la orquesta y
muestra resultados); `scripts/run_experiment.py` ejecuta exactamente los mismos pasos de forma no interactiva.

| Paso | Qué hace | Función |
|---|---|---|
| 1 | Recortes ROI alineados (mamografía completa + máscara), con caché | `pipeline.prepare_data` |
| 2 | Verifica la alineación imagen↔máscara y explora los datos | `preprocess.alignment_contrast` |
| 3 | División train/validación por paciente | `pipeline.prepare_data` |
| 4 | Métodos de referencia (prior centrado y Otsu+morfología) | `pipeline.run_baselines` |
| 5 | Entrena la U-Net | `train.train_unet` |
| 6 | Evalúa (Dice, IoU, precisión, sensibilidad) | `pipeline.evaluate_unet` |
| 7 | Deriva la forma BI-RADS desde la máscara | `pipeline.shape_analysis` |

Los resultados se guardan en `outputs/runs/notebook/`.
""")

md("""
## 0. Configuración

Se carga `configs/main.yaml`. La ruta del dataset se toma de la variable de entorno `MAMO_DATA_DIR` (o de
`archive_dir` en la config); ver `docs/DATA.md`. Para una prueba rápida, define `MAMO_EPOCHS=3` antes de abrir
Jupyter (o cambia `epochs` abajo).
""")
code(r"""
import os, sys, json
from pathlib import Path

REPO = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import load_config
from src import pipeline as pl

epochs = int(os.environ["MAMO_EPOCHS"]) if os.environ.get("MAMO_EPOCHS") else None  # None = la de la config
cfg = load_config(REPO / "configs" / "main.yaml", name="notebook", epochs=epochs)
fig_dir = cfg.run_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)
print("Dataset:", cfg.archive_dir, "| existe:", cfg.archive_dir.exists())
print(f"img_size={cfg.img_size} context={cfg.context} jitter={cfg.jitter} | épocas={cfg.epochs} | salida: {cfg.run_dir}")
""")

md("""
## 1. Preprocesamiento: recortes ROI alineados

**Por qué no usamos directamente los recortes oficiales:** en CBIS-DDSM la máscara está en las coordenadas de la
*mamografía completa*, no del recorte oficial; ambos no están registrados píxel a píxel. Por eso generamos nuestros
propios recortes: se calcula la caja de la lesión en la máscara, se recorta un cuadrado de
`context × lado mayor` tanto en la mamografía como en la máscara, y se reescala a `img_size × img_size`.

**Desplazamiento aleatorio (`jitter`):** si el recorte se centrara exactamente en la lesión, la posición quedaría
"regalada" y un simple círculo en el centro ya lograba un Dice de 0.857 (medido en una primera versión). Por eso el
centro del recorte se desplaza al azar hasta ±`jitter` × lado (con semilla fija, de modo reproducible).

La primera vez tarda ~10–15 min (lee ~1700 mamografías grandes) y guarda el resultado en `outputs/processed/`;
las siguientes ejecuciones lo cargan al instante. La división train/validación es **por paciente**.
""")
code(r"""
data, status = pl.prepare_data(cfg)
if status:
    print("Filas procesadas / descartadas:", status)
tr, va, te = data["train"], data["val"], data["test"]
assert set(tr.meta["patient_id"]).isdisjoint(set(va.meta["patient_id"]))   # sin fuga entre pacientes
print(f"train={len(tr.X)} | validación={len(va.X)} | test oficial={len(te.X)}")
print("Modo de alineación de las máscaras (train):", tr.meta["mode"].value_counts().to_dict())
""")

md("""
## 2. Verificación de la alineación y exploración de datos

Las masas suelen ser más densas (más brillantes) que el tejido que las rodea, por lo que el contraste
(interior − exterior de la máscara) debería ser positivo en la gran mayoría de los casos.
""")
code(r"""
from src.preprocess import alignment_contrast
from src.viz import show_samples, plot_hist

c = alignment_contrast(tr.X, tr.Y)
print(f"Recortes con el interior de la máscara más brillante que el exterior: {(c > 0).mean()*100:.1f}%")
print(f"Contraste medio (niveles de gris 0-255): {c.mean():.1f}")
show_samples(tr.X, tr.Y, n=12, path=fig_dir / "muestras_recortes.png")
plt.show()
""")
code(r"""
print("Etiquetas de forma (campo 'mass shape') en train:")
print(tr.meta["mass_shape_raw"].value_counts().head(10), "\n")
usable = tr.meta["shape_label"] != ""
print(f"Con una de las 4 formas BI-RADS básicas: {usable.sum()} de {len(tr.meta)}")
print("Patología:", tr.meta["pathology"].value_counts().to_dict())
plot_hist(tr.meta["lesion_area_frac"], "fracción del recorte ocupada por la lesión",
          "Tamaño relativo de las lesiones", fig_dir / "hist_area_lesion.png")
plt.show()
""")

md("""
## 3–4. Métodos de referencia (sin aprendizaje profundo)

* **Prior centrado:** círculo centrado del área media de las lesiones. Mide cuánto del Dice se explica solo por la
  posición (sin mirar la imagen). *La U-Net debe superarlo con holgura.*
* **Clásico (Otsu + morfología):** suavizado gaussiano → umbral de Otsu en la zona central → apertura/cierre →
  rellenado de huecos → componente conexa central.
""")
code(r"""
results = pl.run_baselines(data)
pl.summary_table(results)
""")

md("""
## 5. Entrenamiento de la U-Net

Pérdida BCE + Dice, AdamW, `ReduceLROnPlateau`, *early stopping* sobre el Dice de validación y aumento de datos
(volteos, rotación, escala, traslación, contraste/brillo). `amp=False`: en GPUs GTX 16xx la precisión mixta da NaN.
""")
code(r"""
from src.train import train_unet
from src.viz import plot_history

model, hist = train_unet(cfg, tr.X, tr.Y, va.X, va.Y, ckpt_path=cfg.run_dir / "unet_best.pt",
                         log_path=cfg.run_dir / "train.log")
hist.to_csv(cfg.run_dir / "history.csv", index=False)
plot_history(hist, fig_dir / "curvas_entrenamiento.png")
plt.show()
""")

md("## 6. Evaluación en validación y test")
code(r"""
preds, unet_res = pl.evaluate_unet(model, data, cfg)
results.update(unet_res)
table = pl.summary_table(results)
table.to_csv(cfg.run_dir / "metricas_segmentacion.csv", index=False)
table
""")
code(r"""
# Predicciones: contorno verde = anotación, rojo = U-Net (12 recortes de test al azar)
Pte = preds["test"]
show_samples(te.X, te.Y, Pte, n=12, path=fig_dir / "predicciones_test.png")
plt.show()

# Los 8 peores casos de test
dte = results[(pl.UNET, "test")]
worst = dte["dice"].sort_values().index[:8].values
show_samples(te.X, te.Y, Pte, idx=worst, path=fig_dir / "peores_casos_test.png",
             titles={i: f"#{i}  Dice={dte.dice[i]:.2f}" for i in worst})
plt.show()
""")
code(r"""
# ¿Dónde falla? Dice de la U-Net en test por sutileza, patología y desplazamiento de la lesión
per_sample = pl.per_sample_table(results, data)
per_sample.to_csv(cfg.run_dir / "per_sample.csv", index=False)
d = per_sample[(per_sample["método"] == pl.UNET) & (per_sample["conjunto"] == "test")]
print("Por sutileza (1 = muy sutil ... 5 = obvia):")
print(d.groupby("subtlety")["dice"].agg(["mean", "std", "count"]).round(3), "\n")
print("Por patología:")
print(d.groupby("pathology")["dice"].agg(["mean", "std", "count"]).round(3), "\n")
print("Por tercil de |desplazamiento| de la lesión:")
print(pl.dice_by_shift(per_sample).round(3))
""")

md("""
## 7. De la segmentación a la forma BI-RADS (v2025)

Descriptores geométricos del contorno + **árbol de decisión de profundidad 3** (reglas legibles) que asigna
ROUND / OVAL / LOBULATED / IRREGULAR. Se ajusta con las máscaras **anotadas** de train y se evalúa con (a) máscaras
anotadas — cota superior — y (b) máscaras **predichas por la U-Net**. Resultado: al nivel del azar incluso con las
máscaras anotadas; los experimentos de control (`scripts/shape_*.py`) muestran que es una limitación de las
máscaras de CBIS-DDSM, no un error del pipeline.
""")
code(r"""
from src import shape as shp
from src.viz import plot_confusion, plot_feature_boxplots

shape_res, clf, F_plot = pl.shape_analysis(data, preds, cfg)
print("Reglas aprendidas por el árbol:\n")
print(shp.tree_rules(clf))
plot_feature_boxplots(F_plot, "forma", shp.FEATURE_NAMES, shp.SHAPE_CLASSES, fig_dir / "descriptores_por_forma.png")
plt.show()
st = pl.shape_table(shape_res)
st.to_csv(cfg.run_dir / "metricas_forma.csv", index=False)
print("Azar (4 clases) = 0.25")
st
""")
code(r"""
for name, suf in [("máscara anotada", "anotada"), ("máscara U-Net", "unet")]:
    plot_confusion(shape_res[("test", name)]["confusion"], shp.SHAPE_CLASSES, shp.SHAPE_CLASSES + ["NONE"],
                   f"Forma BI-RADS en test, a partir de la {name}", fig_dir / f"confusion_forma_{suf}.png")
    plt.show()
""")

md("## 8. Resumen")
code(r"""
resumen = {
    "n": {s: int(len(data[s].X)) for s in data},
    "segmentacion": {f"{m} | {s}": {"dice": round(float(d.dice.mean()), 4), "iou": round(float(d.iou.mean()), 4)}
                     for (m, s), d in results.items()},
    "forma": {f"{s} | {m}": {k: round(float(r[k]), 4) for k in ("accuracy", "balanced_accuracy", "macro_f1")}
              for (s, m), r in shape_res.items()},
    "epocas_entrenadas": int(len(hist)), "mejor_dice_val": round(float(hist.val_dice.max()), 4),
}
with open(cfg.run_dir / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(resumen, f, indent=2, ensure_ascii=False)
print(json.dumps(resumen["segmentacion"], indent=2, ensure_ascii=False))
print("\nTodo guardado en:", cfg.run_dir.resolve())
""")

nb["cells"] = C
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
nbf.write(nb, str(Path(__file__).with_name("proyecto_mamografia.ipynb")))
print("Notebook creado")
