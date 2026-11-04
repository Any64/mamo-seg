"""Genera notebooks/proyecto_mamografia.ipynb (se ejecuta una sola vez al desarrollar)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip("\n")))

md("""
# Segmentación de masas mamarias (CBIS-DDSM) y caracterización de su forma (BI-RADS 2025)

**CBM423 – Procesamiento Digital de Imágenes Médicas · Universidad de Valparaíso**

Este notebook ejecuta el pipeline completo, de arriba hacia abajo. Toda la lógica vive en la carpeta `src/`
(el notebook solo la orquesta y muestra resultados):

| Paso | Qué hace | Módulo |
|---|---|---|
| 1 | Genera recortes ROI alineados (mamografía completa + máscara) | `src/preprocess.py` |
| 2 | Verifica la alineación imagen↔máscara y explora los datos | `src/preprocess.py`, `src/viz.py` |
| 3 | Divide train/validación por paciente | `src/dataset.py` |
| 4 | Métodos de referencia (prior centrado y Otsu+morfología) | `src/classical.py` |
| 5 | Entrena la U-Net | `src/model.py`, `src/train.py` |
| 6 | Evalúa (Dice, IoU, precisión, sensibilidad) | `src/metrics.py` |
| 7 | Deriva la forma BI-RADS desde la máscara | `src/shape.py` |

Todo lo generado (figuras, tablas, modelo) se guarda en `outputs/`.
""")

md("## 0. Configuración")
code(r"""
import os, sys, json
from pathlib import Path

REPO = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import Config

cfg = Config(
    archive_dir=os.environ.get("MAMO_DATA_DIR", REPO.parent / "archive"),  # carpeta con csv/ y jpeg/
    out_dir=REPO / "outputs",
    img_size=256,        # lado de los recortes
    context=3.0,         # el recorte mide 3 veces el lado mayor de la lesión
    jitter=0.2,          # la lesión se desplaza al azar hasta ±20% (no queda siempre al centro)
    base_channels=16,    # 16 = rápido (GTX 1650); 32 = más capacidad, ~4 veces más lento
    batch_size=8,
    epochs=int(os.environ.get("MAMO_EPOCHS", 40)),  # prueba rápida: MAMO_EPOCHS=3
    amp=False,           # NO activar en GPUs GTX 16xx (produce NaN)
)
cfg.make_dirs()
print("Carpeta del dataset existe:", cfg.archive_dir.exists())
""")

md("""
## 1. Preprocesamiento: recortes ROI alineados

**Por qué no usamos directamente los recortes oficiales:** en CBIS-DDSM la máscara está en las coordenadas de la
*mamografía completa*, no del recorte oficial; ambos no están registrados píxel a píxel. Por eso generamos nuestros
propios recortes: se calcula la caja de la lesión en la máscara, se recorta un cuadrado de
`context × lado mayor` tanto en la mamografía como en la máscara, y se reescala a `img_size × img_size`.

**Desplazamiento aleatorio (`jitter`):** si el recorte se centrara exactamente en la lesión, la posición quedaría
"regalada" y un simple círculo en el centro ya lograba un Dice de 0.857 (medido en una primera versión). Por eso el
centro del recorte se desplaza al azar hasta ±`jitter` × lado (con semilla fija, de modo reproducible). Así la red debe
*localizar y delinear* la lesión, como ocurriría con una región de interés aproximada entregada por un detector.

Esta celda tarda ~10–20 min la primera vez (lee ~1700 mamografías grandes) y guarda el resultado en `outputs/processed/`;
las siguientes ejecuciones lo cargan al instante.
""")
code(r"""
from src.data_index import load_case_csv
from src.preprocess import build_split, save_split, load_split

tag = cfg.cache_tag   # cambia si cambian img_size / context / jitter -> se regeneran los recortes
status_report = {}
for split in ["train", "test"]:
    if not (cfg.processed_dir / f"{split}_{tag}.npz").exists():
        df = load_case_csv(cfg.csv_dir, split)
        X, Y, meta, status = build_split(df, cfg.jpeg_dir, cfg.img_size, cfg.context, cfg.min_side,
                                         jitter=cfg.jitter, seed=cfg.seed)
        save_split(cfg.processed_dir, f"{split}_{tag}", X, Y, meta)
        status_report[split] = dict(status)
        print(f"{split}: {len(df)} filas del CSV -> {dict(status)}")
if status_report:
    with open(cfg.tables_dir / "estado_preprocesamiento.json", "w", encoding="utf-8") as f:
        json.dump(status_report, f, indent=2)

Xtr_all, Ytr_all, mtr_all = load_split(cfg.processed_dir, f"train_{tag}")
Xte, Yte, mte = load_split(cfg.processed_dir, f"test_{tag}")
print("train (oficial):", Xtr_all.shape, "| test (oficial):", Xte.shape)
print("\nModo de alineación de las máscaras (train):")
print(mtr_all["mode"].value_counts())
""")

md("""
## 2. Verificación de la alineación y exploración de datos

Chequeo **objetivo** de que imagen y máscara están registradas: las masas suelen ser más densas (más brillantes) que
el tejido que las rodea, por lo que el contraste (interior − exterior de la máscara) debería ser positivo en la gran
mayoría de los casos. Además se inspeccionan visualmente muestras aleatorias (contorno verde = máscara anotada).
""")
code(r"""
from src.preprocess import alignment_contrast
from src.viz import show_samples, plot_hist

c = alignment_contrast(Xtr_all, Ytr_all)
print(f"Recortes donde el interior de la máscara es más brillante que el exterior: {(c > 0).mean()*100:.1f}%")
print(f"Contraste medio (niveles de gris 0-255): {c.mean():.1f}")
show_samples(Xtr_all, Ytr_all, n=12, path=cfg.figures_dir / "muestras_recortes.png")
plt.show()
""")
code(r"""
print("Etiquetas de forma (campo 'mass shape') en train:")
print(mtr_all["mass_shape_raw"].value_counts().head(12), "\n")

usable = mtr_all["shape_label"] != ""
print(f"Con una de las 4 formas BI-RADS básicas: {usable.sum()} de {len(mtr_all)}")
print(mtr_all.loc[usable, "shape_label"].value_counts(), "\n")
print("Patología:", mtr_all["pathology"].value_counts().to_dict())
print("Número de pacientes distintos (train):", mtr_all["patient_id"].nunique())
plot_hist(mtr_all["lesion_area_frac"], "fracción del recorte ocupada por la lesión",
          "Tamaño relativo de las lesiones", cfg.figures_dir / "hist_area_lesion.png")
plt.show()
""")

md("## 3. División entrenamiento / validación (por paciente)")
code(r"""
from src.data_utils import patient_split

tr, va = patient_split(mtr_all, cfg.val_frac, cfg.seed)
Xtr, Ytr, mtr = Xtr_all[tr], Ytr_all[tr], mtr_all.iloc[tr].reset_index(drop=True)
Xva, Yva, mva = Xtr_all[va], Ytr_all[va], mtr_all.iloc[va].reset_index(drop=True)
assert set(mtr["patient_id"]).isdisjoint(set(mva["patient_id"]))   # sin fuga entre pacientes
print(f"train={len(Xtr)} | validación={len(Xva)} | test oficial={len(Xte)}")
""")

md("""
## 4. Métodos de referencia (sin aprendizaje profundo)

* **Prior centrado:** círculo centrado del área media de las lesiones. Mide cuánto del Dice se explica solo por la
  posición (sin mirar la imagen). *La U-Net debe superarlo con holgura.*
* **Clásico (Otsu + morfología):** suavizado gaussiano → umbral de Otsu en la zona central → apertura/cierre →
  rellenado de huecos → componente conexa central.
""")
code(r"""
from src.classical import fit_center_prior, center_prior, classical_segment_stack
from src.metrics import per_sample_metrics

radius = fit_center_prior(Ytr)
print(f"Radio del prior centrado: {radius:.1f} px")
prior = lambda n: np.stack([center_prior(Xtr.shape[1:], radius)] * n)

results = {}
for name, X_, Y_ in [("val", Xva, Yva), ("test", Xte, Yte)]:
    results[("Prior centrado", name)] = per_sample_metrics(prior(len(X_)), Y_)
    results[("Clásico (Otsu+morfología)", name)] = per_sample_metrics(classical_segment_stack(X_), Y_)

def summary_table(results):
    rows = []
    for (m, s), d in results.items():
        rows.append({"método": m, "conjunto": s, "n": len(d),
                     "Dice": f"{d.dice.mean():.3f} ± {d.dice.std():.3f}",
                     "IoU": f"{d.iou.mean():.3f} ± {d.iou.std():.3f}",
                     "Precisión": f"{d.precision.mean():.3f}", "Sensibilidad": f"{d.recall.mean():.3f}"})
    return pd.DataFrame(rows)

summary_table(results)
""")

md("""
## 5. Entrenamiento de la U-Net

Pérdida BCE + Dice, AdamW, `ReduceLROnPlateau`, *early stopping* sobre el Dice de validación y aumento de datos
(volteos, rotación, escala, traslación, contraste/brillo). Con una GPU modesta usa `base_channels=16`; para una prueba rápida, `epochs=3`.
""")
code(r"""
from src.train import train_unet, get_device
from src.viz import plot_history

model, hist = train_unet(cfg, Xtr, Ytr, Xva, Yva, log_path=cfg.out_dir / "logs" / "entrenamiento.log")
hist.to_csv(cfg.tables_dir / "historial_entrenamiento.csv", index=False)
plot_history(hist, cfg.figures_dir / "curvas_entrenamiento.png")
plt.show()
""")

md("## 6. Evaluación")
code(r"""
from src.dataset import to_tensors
from src.train import predict_probs

dev = get_device()
Pva = predict_probs(model, to_tensors(Xva, Yva, dev)[0]) > cfg.threshold
Pte = predict_probs(model, to_tensors(Xte, Yte, dev)[0]) > cfg.threshold
results[("U-Net", "val")] = per_sample_metrics(Pva, Yva)
results[("U-Net", "test")] = per_sample_metrics(Pte, Yte)

table = summary_table(results)
table.to_csv(cfg.tables_dir / "metricas_segmentacion.csv", index=False)
table
""")
code(r"""
# Predicciones: contorno verde = anotación, rojo = U-Net (12 recortes de test al azar)
show_samples(Xte, Yte, Pte, n=12, path=cfg.figures_dir / "predicciones_test.png")
plt.show()

# Los 8 peores casos de test
dte = results[("U-Net", "test")]
worst = dte["dice"].sort_values().index[:8].values
show_samples(Xte, Yte, Pte, idx=worst, path=cfg.figures_dir / "peores_casos_test.png",
             titles={i: f"#{i}  Dice={dte.dice[i]:.2f}" for i in worst})
plt.show()
""")
code(r"""
# ¿Dónde falla? Dice de la U-Net en test según dificultad visual (subtlety), patología y tamaño
d = dte.copy()
d["subtlety"], d["pathology"], d["area_frac"] = mte["subtlety"].values, mte["pathology"].values, mte["lesion_area_frac"].values
print("Por sutileza (1 = muy sutil ... 5 = obvia):")
print(d.groupby("subtlety")["dice"].agg(["mean", "std", "count"]).round(3), "\n")
print("Por patología:")
print(d.groupby("pathology")["dice"].agg(["mean", "std", "count"]).round(3), "\n")
print(f"Correlación Dice vs. tamaño relativo de la lesión: {d['dice'].corr(d['area_frac']):.3f}")
""")

md("""
## 7. De la segmentación a la forma BI-RADS (v2025)

A partir de la máscara se calculan descriptores geométricos del contorno y un **árbol de decisión de profundidad 3**
(reglas legibles; no es una red neuronal) asigna ROUND / OVAL / LOBULATED / IRREGULAR. El árbol se ajusta con las
máscaras **anotadas** del conjunto de entrenamiento y se evalúa sobre (a) máscaras anotadas — cota superior — y
(b) máscaras **predichas por la U-Net** — lo que ocurriría en la práctica. La diferencia mide cuánto se propaga el
error de segmentación a la forma. Solo se usan lesiones con una de las 4 formas básicas.
""")
code(r"""
from src import shape as shp
from src.viz import plot_confusion, plot_feature_boxplots

ktr = (mtr["shape_label"] != "").values
F_tr = shp.features_dataframe(Ytr[ktr])
clf = shp.fit_shape_tree(F_tr, mtr.loc[ktr, "shape_label"].values, seed=cfg.seed)
print("Reglas aprendidas por el árbol:\n")
print(shp.tree_rules(clf))

F_plot = F_tr.assign(forma=mtr.loc[ktr, "shape_label"].values)
plot_feature_boxplots(F_plot, "forma", shp.FEATURE_NAMES, shp.SHAPE_CLASSES,
                      cfg.figures_dir / "descriptores_por_forma.png")
plt.show()
""")
code(r"""
shape_res = {}
for name, Y_, P_, m_ in [("val", Yva, Pva, mva), ("test", Yte, Pte, mte)]:
    k = (m_["shape_label"] != "").values
    for src_name, M in [("máscara anotada", Y_), ("máscara U-Net", P_)]:
        pred = shp.predict_shape(clf, shp.features_dataframe(M[k]))
        shape_res[(name, src_name)] = shp.evaluate_shape(m_.loc[k, "shape_label"].values, pred)

shape_table = pd.DataFrame([
    {"conjunto": s, "máscara usada": m, "n": r["n"], "sin descriptores": r["n_invalid"],
     "exactitud": round(r["accuracy"], 3), "exactitud balanceada": round(r["balanced_accuracy"], 3),
     "F1 macro": round(r["macro_f1"], 3)} for (s, m), r in shape_res.items()])
shape_table.to_csv(cfg.tables_dir / "metricas_forma.csv", index=False)
print("Azar (4 clases) = 0.25")
shape_table
""")
code(r"""
for src_name in ["máscara anotada", "máscara U-Net"]:
    r = shape_res[("test", src_name)]
    plot_confusion(r["confusion"], shp.SHAPE_CLASSES, shp.SHAPE_CLASSES + ["NONE"],
                   f"Forma BI-RADS en test, a partir de la {src_name}",
                   cfg.figures_dir / f"confusion_forma_{'anotada' if 'anotada' in src_name else 'unet'}.png")
    plt.show()
""")

md("## 8. Resumen para el informe")
code(r"""
resumen = {
    "n_train": int(len(Xtr)), "n_val": int(len(Xva)), "n_test": int(len(Xte)),
    "segmentacion": {f"{m} | {s}": {k: round(float(v), 4) for k, v in
                     {"dice": d.dice.mean(), "dice_std": d.dice.std(), "iou": d.iou.mean()}.items()}
                     for (m, s), d in results.items()},
    "forma": {f"{s} | {m}": {k: round(float(v), 4) for k, v in r.items() if k in
              ("accuracy", "balanced_accuracy", "macro_f1")} for (s, m), r in shape_res.items()},
    "epocas_entrenadas": int(len(hist)), "mejor_dice_val": round(float(hist.val_dice.max()), 4),
    "config": {k: str(v) for k, v in vars(cfg).items()},
}
with open(cfg.tables_dir / "resumen.json", "w", encoding="utf-8") as f:
    json.dump(resumen, f, indent=2, ensure_ascii=False)
print(json.dumps(resumen["segmentacion"], indent=2, ensure_ascii=False))
print("\nTodo guardado en:", cfg.out_dir.resolve())
""")

nb["cells"] = C
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
nbf.write(nb, str(__import__("pathlib").Path(__file__).with_name("proyecto_mamografia.ipynb")))
print("Notebook creado")
