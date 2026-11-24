"""Genera notebooks/demo_resultados.ipynb (ejecutar tras modificar este archivo).

Demostración SIN entrenar: carga el modelo ya entrenado (``outputs/runs/main/unet_best.pt``)
y los recortes en caché, y muestra métricas, una galería de aciertos y fallos, la forma
BI-RADS anotada vs predicha y ``mostrar_caso(i)``. Tarda menos de 2 minutos si los
recortes ya están en ``outputs/processed/`` (si no, los genera la primera vez).
"""
from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).with_name("demo_resultados.ipynb")


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    C = []
    md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip("\n")))  # noqa: E731
    code = lambda s: C.append(nbf.v4.new_code_cell(s.strip("\n")))  # noqa: E731

    md("""
# Demostración de resultados (sin reentrenar)

**CBM423 – Procesamiento Digital de Imágenes Médicas · Universidad de Valparaíso**

Carga la U-Net ya entrenada del experimento principal (`outputs/runs/main/unet_best.pt`, configuración
`configs/main.yaml`) y la evalúa en el **test oficial** de CBIS-DDSM. No entrena nada.

Requisitos: el dataset (variable `MAMO_DATA_DIR`, ver `docs/DATA.md`) y los pesos de `main`
(`python scripts/run_experiment.py --config configs/main.yaml`). Si los recortes ya están en
`outputs/processed/`, todo el notebook corre en menos de 2 minutos.
""")
    code(r"""
import sys, time
from pathlib import Path

REPO = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import pipeline as pl
from src import shape as shp
from src.config import load_config
from src.metrics import extended_metrics
from src.viz import plot_confusion

T0 = time.time()
RUN = REPO / "outputs" / "runs" / "main"
cfg = load_config(REPO / "configs" / "main.yaml")
data, _ = pl.prepare_data(cfg, progress=False, include_test=True)
test = data["test"]
print(f"Recortes: train={len(data['train'].X)} val={len(data['val'].X)} test={len(test.X)} "
      f"| {time.time() - T0:.0f} s")
""")

    md("""
## 1. Modelo entrenado y métricas en test

Dice, IoU, precisión, sensibilidad y Hausdorff 95 (en píxeles del recorte de 256 × 256; las predicciones
vacías no tienen HD95 y se cuentan aparte), junto a los dos métodos de referencia.
""")
    code(r"""
model = pl.load_unet(pl.run_config(RUN), RUN / "unet_best.pt")
pred = pl.predict_masks(model, test.X, cfg.threshold)
met = extended_metrics(pred, test.Y)

ref = pl.run_baselines(data, splits=("test",))
rows = [{"método": m, "Dice": d.dice.mean(), "IoU": d.iou.mean(), "Precisión": d.precision.mean(),
         "Sensibilidad": d.recall.mean()} for (m, s), d in ref.items()]
rows.append({"método": "U-Net", "Dice": met.dice.mean(), "IoU": met.iou.mean(),
             "Precisión": met.precision.mean(), "Sensibilidad": met.recall.mean(),
             "HD95 mediano (px)": met.hd95.median(), "predicciones vacías": int(met.vacia.sum()),
             "Dice < 0.1": int((met.dice < 0.1).sum())})
display(pd.DataFrame(rows).round(3))
print(f"{time.time() - T0:.0f} s")
""")

    md("""
## 2. Galería: mejores y peores recortes

Amarillo = anotación del radiólogo; rojo = U-Net.
""")
    code(r"""
def _contornos(ax, i):
    ax.imshow(test.X[i], cmap="gray")
    ax.contour(test.Y[i], levels=[0.5], colors="#f2c200", linewidths=1.1)
    if pred[i].any():
        ax.contour(pred[i], levels=[0.5], colors="#d62728", linewidths=1.1)
    ax.set_xticks([]); ax.set_yticks([])

orden = np.argsort(met.dice.values)
fig, axes = plt.subplots(2, 6, figsize=(15, 5.4))
for ax, i in zip(axes[0], orden[::-1][:6]):
    _contornos(ax, i); ax.set_title(f"#{i}  Dice {met.dice[i]:.2f}", fontsize=9)
for ax, i in zip(axes[1], orden[:6]):
    _contornos(ax, i); ax.set_title(f"#{i}  Dice {met.dice[i]:.2f}", fontsize=9)
axes[0, 0].set_ylabel("aciertos", fontsize=11)
axes[1, 0].set_ylabel("fallos", fontsize=11)
fig.suptitle("U-Net en test: 6 mejores y 6 peores recortes (amarillo = anotación, rojo = U-Net)")
plt.tight_layout(); plt.show()
""")

    md("""
## 3. Forma BI-RADS: anotada vs predicha

Árbol de decisión de profundidad 3 ajustado con los descriptores de las máscaras **anotadas** de train
(igual que en el experimento principal) y aplicado en test a la máscara anotada y a la de la U-Net.
Filas: forma anotada por el radiólogo; columnas: forma predicha (`NONE` = máscara sin descriptores).
""")
    code(r"""
forma_res, arbol, _ = pl.shape_analysis(data, {"test": pred}, cfg, splits=("test",))
display(pl.shape_table(forma_res))
for (s, nombre), r in forma_res.items():
    plot_confusion(r["confusion"], shp.SHAPE_CLASSES, shp.SHAPE_CLASSES + ["NONE"],
                   title=f"Forma en test desde la {nombre}")
    plt.show()
print(f"{time.time() - T0:.0f} s")
""")

    md("""
## 4. Inspeccionar un caso: `mostrar_caso(i)`

Muestra el recorte `i` del test con la anotación y la predicción, sus métricas y sus metadatos
(forma, margen, patología, sutileza, densidad), además de la forma que asigna el árbol a cada máscara.
""")
    code(r"""
def mostrar_caso(i: int):
    # Muestra el recorte i del test oficial: imagen, anotación, U-Net y metadatos.
    if not 0 <= i < len(test.X):
        raise IndexError(f"i debe estar entre 0 y {len(test.X) - 1}")
    m = test.meta.iloc[i]
    f_anot = shp.predict_shape(arbol, shp.features_dataframe(test.Y[i:i + 1]))[0]
    f_pred = shp.predict_shape(arbol, shp.features_dataframe(pred[i:i + 1]))[0]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9))
    axes[0].imshow(test.X[i], cmap="gray"); axes[0].set_title("recorte")
    _contornos(axes[1], i); axes[1].set_title("amarillo = anotación, rojo = U-Net", fontsize=9)
    axes[2].imshow(test.Y[i].astype(int) * 2 + pred[i].astype(int), cmap="viridis", vmin=0, vmax=3)
    axes[2].set_title("VP (amarillo) · FN (verde) · FP (azul)", fontsize=9)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Caso #{i} | paciente {m.patient_id} {m.view} | Dice {met.dice[i]:.3f} | "
                 f"HD95 {met.hd95[i]:.1f} px", fontsize=11)
    plt.tight_layout(); plt.show()
    return pd.Series({"patología": m.pathology, "forma anotada": m.shape_label or "(otra)",
                      "forma (árbol, máscara anotada)": f_anot, "forma (árbol, máscara U-Net)": f_pred,
                      "margen": m.mass_margins, "sutileza": m.subtlety, "densidad": m.breast_density,
                      "Dice": round(met.dice[i], 3), "IoU": round(met.iou[i], 3),
                      "HD95 (px)": round(met.hd95[i], 1), "componentes": met.n_componentes[i]})

mostrar_caso(int(orden[::-1][0]))   # el mejor caso
""")
    code(r"""
mostrar_caso(int(orden[0]))   # el peor caso
""")
    code(r"""
print(f"Tiempo total: {time.time() - T0:.0f} s")
""")
    nb["cells"] = C
    nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    return nb


if __name__ == "__main__":
    nbf.write(build(), OUT)
    print(f"Escrito {OUT}")
