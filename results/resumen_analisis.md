# Resumen: análisis estratificado, margen, fallos y demostración

Análisis sobre los modelos ya entrenados (`configs/main.yaml`; semillas 42, 43 y 44). No se entrenó ni se
eligió nada con el test: las tareas 1 y 3 solo describen el test de la configuración final, y en la tarea 2
la combinación se eligió en validación y el test se evaluó una sola vez.

## 1. Análisis estratificado en test (`estratificado_test.csv`, `estratificado_kruskal.csv`)

El Dice de cada recorte es el promedio de las 3 semillas; el HD95 está en px del recorte de 256 × 256. 378 recortes
y 201 pacientes. "Otras" formas: distorsión arquitectural, ganglio linfático y asimetrías (54 casos).

| Variable | Grupos (Dice medio) | Kruskal-Wallis p (Holm) |
|---|---|---|
| Forma | ROUND 0.82 · OVAL 0.79 · LOBULATED 0.78 · IRREGULAR 0.78 · **otras 0.61** | 4e-5 (5e-4) |
| Margen | CIRCUMSCRIBED 0.81 · MICROLOB. 0.82 · OBSCURED 0.77 · ILL_DEF. 0.76 · SPICULATED 0.72 · mixto 0.72 | 0.014 (0.12, n.s.) |
| Sutileza | 1: 0.58 (n = 14 < 15) · 2: 0.62 · 3: 0.75 · 4: 0.78 · 5: 0.82 | 1e-9 (2e-8) |
| Densidad | 1: 0.85 · 2: 0.79 · 3: 0.71 · 4: 0.68 | 8e-7 (1e-5) |
| Patología | BENIGN 0.75 · MALIGNANT 0.79 | 0.30 (n.s.) |
| Vista | CC 0.77 · MLO 0.76 | 0.84 (n.s.) |

- **Lo que pesa es la dificultad de la imagen, no el tipo de lesión.** Las lesiones sutiles y las mamas densas
  (BI-RADS 3–4) bajan el Dice unos 0.15–0.25 y doblan el HD95 mediano, de 12 a 24 px.
- **Forma:** la diferencia significativa la produce solo el grupo "otras", que no son masas propiamente dichas.
  Entre las 4 formas básicas el Dice va de 0.78 a 0.82.
- **Margen:** las espiculadas segmentan peor (0.72), pero la diferencia no resiste la corrección por 18 pruebas.
- **Cautelas:** varios recortes son del mismo paciente y las variables están correlacionadas (por ejemplo, densidad
  y sutileza), así que los p-valores son optimistas y no hay que leerlos como efectos causales independientes.

## 2. Margen BI-RADS desde la geometría (`margen_conteos.csv`, `margen_experimento.csv`)

Agrupación (los conteos están en `margen_conteos.csv`):
- **M1, circunscrito vs no circunscrito:** CIRCUMSCRIBED solo frente a ILL_DEFINED, MICROLOBULATED o SPICULATED,
  solos o combinados, siempre sin CIRCUMSCRIBED. Se excluyen tres grupos:
  - OBSCURED solo: el borde está tapado por tejido, así que no es una propiedad del contorno.
  - Las mezclas con CIRCUMSCRIBED, que son contradictorias.
  - Los casos sin dato.
- **M2, espiculado vs no espiculado.**

Se ajustó en train, se eligió σ ∈ {0, 4} × {regresión logística, random forest} por AUC en validación y el
test se evaluó una sola vez.

| Tarea | AUC val (elegida) | AUC test |
|---|---|---|
| M1 descriptores → no circunscrito | 0.587 | 0.646 |
| M2 descriptores → espiculado | 0.589 | 0.500 |
| Patología ← solo descriptores | 0.554 | 0.470 |
| Patología ← solo etiqueta de margen | 0.868 | 0.782 |
| Patología ← descriptores + margen | 0.860 | 0.807 |

- **La geometría casi no recupera el margen.** Circunscrito vs no queda apenas sobre el azar, y la espiculación no se
  detecta (AUC 0.50). Las máscaras de CBIS-DDSM no dibujan las espículas.
- **La etiqueta de margen sí predice la patología (AUC 0.78)**, pero esa información viene de la lectura del
  radiólogo, no del contorno.
- **No mejora:** sumar los descriptores al margen no ayuda en validación (0.860 frente a 0.868). La diferencia en
  test (0.807 frente a 0.782) no es evidencia de mejora.
- Esto confirma lo que ya mostraba la forma (E2–E4): el límite está en lo gruesas que son las máscaras.

## 3. Análisis de fallos (`fallos_test.csv`, `fallos_resumen.csv`, `fallos_pruebas.json`)

El modelo principal (semilla 42) tiene Dice < 0.1 en 16 de 378 recortes (4,2 %); las semillas 43 y 44 tienen 14 y 10.

| Modo | n | Comentario |
|---|---|---|
| Predicción vacía | 2 | una lesión cortada por el borde de la mamografía |
| Otra zona sin solape | 7 | segmenta otra estructura densa o deja manchas pequeñas lejos de la lesión |
| Sub/sobresegmentación | 7 (5 sub, 2 sobre) | en la figura, la mayoría son también localizaciones en otra estructura que apenas tocan la lesión |

**¿Los explica el desplazamiento alto? Solo en parte.**
- Los fallos tienen un desplazamiento algo mayor: mediana 0.19 frente a 0.15 (Mann-Whitney p = 0.044). El 56 % cae en
  el tercil alto, cuando se esperaría un 33 % (2, 5 y 9 fallos por tercil; χ² p = 0.09).
- Pero 7 de 16 fallos tienen desplazamiento bajo o medio.
- Los fallos también son lesiones más pequeñas (p = 0.0006) y más sutiles (p = 0.048), y 12 de 16 están en mamas
  de densidad 3–4.
- **Interpretación:** el modo dominante es la localización. Cuando la lesión es pequeña o sutil, o queda lejos del
  centro, la red elige otra estructura densa del recorte.

## 4. Demostración (`notebooks/demo_resultados.ipynb`)

- **Generación:** lo crea `notebooks/build_demo.py`.
- **Qué hace:** carga `outputs/runs/main/unet_best.pt` sin reentrenar y muestra:
  - métricas en test con HD95 y los métodos de referencia;
  - una galería de 6 aciertos y 6 fallos;
  - la forma BI-RADS anotada frente a la predicha, con matrices de confusión;
  - `mostrar_caso(i)`.
- **Tiempo:** de punta a punta, 53 s en CPU. La copia ejecutada está en `outputs/demo_resultados_ejecutado.ipynb`.
- **Verificación:** reproduce el Dice en test de `main` (0.769).
