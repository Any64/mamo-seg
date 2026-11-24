# Resumen E12: morfología en capas convolucionales y posprocesamiento morfológico

El profesor sugirió "agregar morfología en capas convolucionales". Este experimento compara tres métodos con el
mismo protocolo que la configuración final (`configs/main.yaml`: recortes propios, división por paciente,
`amp: false` y semillas 42, 43 y 44):

- **U-Net:** E1 y E6, sin cambios.
- **U-Net + posprocesamiento fijo:** apertura y cierre con un disco, seguidos de la componente conexa mayor. Se aplica
  sobre las mismas máscaras de la U-Net.
- **Morph-UNet:** U-Net con un bloque morfológico aprendible.

**Todo se eligió en validación**: la ubicación del bloque, el radio del posprocesamiento y el modelo final. El test
se evaluó una sola vez, para informar.

## Implementación

- **Archivos:** `src/morph.py`, `src/model.py`, `src/postprocess.py` y `src/metrics.py`.
- **Capa:** p-convolución con media contraarmónica (Masci et al., 2013). Se calcula de forma estable como
  softmax(p · log x + log w) sobre ventanas de 5 × 5.
  - Si p > 0 se aproxima a una dilatación y si p < 0 a una erosión.
  - p y el elemento estructurante w se aprenden por canal.
  - Las pruebas la comparan con `scipy.ndimage` para |p| grande.
- **Bloque:**
  1. conv 1x1 de reducción a 4 canales y sigmoide;
  2. dos ramas en paralelo: apertura (erosión → dilatación) y cierre (dilatación → erosión);
  3. conv 1x1 de expansión;
  4. conexión residual.
- **Costo:**
  - Agrega 628 parámetros (1 942 917 frente a 1 942 289).
  - Una época tarda ~59 s, frente a ~16 s de la U-Net. Para entrar en los 4 GB de la GTX 1650 se usa gradient
    checkpointing y los datos de train se mantienen en CPU.
  - Cada entrenamiento duró unos 35–40 min.

## 1. Ubicación del bloque (validación, semilla 42; `morfologia_ubicacion.csv`)

| Ubicación | Dice val | Épocas (mejor) |
|---|---|---|
| U-Net sin bloque | 0.781 | 40 (36) |
| Antes de la cabeza 1x1 (`head`) | 0.771 | 32 (22) |
| **En el salto de máxima resolución (`skip`)** | **0.777** | 35 (25) |

Se eligió `skip` por su Dice en validación. Ninguna de las dos ubicaciones supera a la U-Net con la misma semilla.

## 2. Radio del posprocesamiento (validación, 3 semillas de la U-Net; `posproc_seleccion_radio.csv`)

| Radio (px) | sin posproc. | 0 | 1 | 2 | 3 | 4 | **6** | 8 |
|---|---|---|---|---|---|---|---|---|
| Dice val | 0.777 | 0.791 | 0.791 | 0.791 | 0.790 | 0.791 | **0.791** | 0.790 |

- **Elegido: r = 6**, que tiene el mayor Dice medio (0.7912).
- **Prácticamente toda la ganancia viene de quedarse con la componente conexa mayor.** Con r = 0 ya se obtiene 0.7909;
  la apertura y el cierre no aportan nada medible.

## 3. Comparación (media ± DE entre 3 semillas; `morfologia_comparacion.csv`, `morfologia_por_semilla.csv`)

| Método | Conjunto | Dice | IoU | Precisión | Sensib. | HD95 medio (px) | HD95 mediano (px) | Recortes con > 1 componente | Predicciones vacías |
|---|---|---|---|---|---|---|---|---|---|
| U-Net | val | 0.777 ± 0.003 | 0.666 | 0.769 | 0.826 | 26.6 ± 1.7 | 13.7 | 34 % | 3.0 |
| **U-Net + posproc.** | val | **0.791 ± 0.006** | 0.687 | 0.810 | 0.812 | **16.6 ± 0.2** | 12.4 | 0 % | 4.3 |
| Morph-UNet | val | 0.777 ± 0.000 | 0.664 | 0.770 | 0.821 | 27.4 ± 0.8 | 13.9 | 37 % | 2.3 |
| U-Net | test | 0.762 ± 0.009 | 0.652 | 0.758 | 0.804 | 25.6 ± 1.9 | 15.4 | 32 % | 2.3 |
| **U-Net + posproc.** | test | **0.767 ± 0.007** | 0.662 | 0.789 | 0.789 | **18.3 ± 0.9** | 13.1 | 0 % | 7.3 |
| Morph-UNet | test | 0.762 ± 0.001 | 0.650 | 0.754 | 0.801 | 26.6 ± 0.9 | 15.2 | 35 % | 1.3 |

Las predicciones vacías no tienen HD95 y se cuentan aparte, en la última columna.

### Pruebas pareadas (`morfologia_pruebas.csv`)

Las diferencias se calculan sobre el Dice y el HD95 de cada recorte, promediados entre las 3 semillas. Se informa el
IC 95 % por bootstrap, remuestreando pacientes, y el Wilcoxon pareado.

| Diferencia | Conjunto | Δ Dice [IC 95 %] | p | Δ HD95 px [IC 95 %] | p |
|---|---|---|---|---|---|
| Morph-UNet − U-Net | val | −0.001 [−0.006, 0.006] | 0.10 | +0.8 [−0.2, 2.0] | 0.40 |
| Posproc. − U-Net | val | **+0.014 [0.008, 0.021]** | 0.007 | **−9.8 [−12.9, −7.3]** | 0.0002 |
| Morph-UNet − U-Net | test | −0.001 [−0.006, 0.005] | 0.65 | +0.8 [−0.1, 1.8] | 0.96 |
| Posproc. − U-Net | test | +0.004 [−0.000, 0.009] | 0.15 | **−6.6 [−8.4, −4.9]** | 0.004 |

**Modelo final elegido en validación: U-Net + posprocesamiento fijo** (`morfologia_decision.json`).

## 4. Interpretación

**La morfología aprendida dentro de la red no mejora nada.**
- La Morph-UNet empata con la U-Net en Dice, tanto en validación como en test, y su HD95 es algo peor (sin
  significación). Además, tarda 3,7 veces más por época.
- La única diferencia observable es una variabilidad menor entre semillas (DE 0.000–0.001 frente a 0.003–0.009). Con
  3 semillas esto no es concluyente.
- **Por qué, probablemente:**
  - Los p aprendidos apenas se movieron de su valor inicial ±5 (rango 4.2–5.5; `morfologia_p_aprendidos.csv`), así
    que las capas siguen siendo erosiones y dilataciones suaves de 5 × 5. Lo que cambió son los elementos
    estructurantes (`morfologia_elementos_p.png`).
  - Con la conexión residual, la red puede ignorar el bloque.
  - Los errores dominantes de la U-Net son de localización (E11): segmentar otra estructura densa. Una operación local
    de 5 × 5 no corrige eso.
- **Cautela:** la semilla 43 llegó a su mejor época justo en la 40, la última. Con más épocas podría mejorar un poco.

**El posprocesamiento fijo sí ayuda, sobre todo en el contorno.**
- **HD95:** baja de 26.6 a 16.6 px en validación y de 25.6 a 18.3 px en test, significativo en ambos. Elimina las
  manchas falsas lejanas: el 32–34 % de las predicciones tenía más de una componente.
- **Dice:** sube 0.014 en validación, de forma significativa. En test sube 0.004, con un IC que toca el 0, así que
  la mejora del Dice no se confirma en test.
- **Costo:** cuando la componente mayor es la equivocada, el recorte pasa a Dice 0, como en los 3 primeros casos de
  `morfologia_ejemplos.png`. Por eso las predicciones vacías o sin solape en test suben de 2.3 a 7.3.
- **Qué hace en realidad:** "quedarse con la componente mayor", no apertura y cierre (sección 2).

## 5. Forma BI-RADS con las máscaras de cada método (`morfologia_forma*.csv`)

Se usa el árbol de E1 (profundidad 3, ajustado en train con máscaras anotadas). Fracción de predicciones LOBULATED:

| Máscara | val | test |
|---|---|---|
| Anotada | 29 % | 36 % |
| U-Net | 72 % | 75 % |
| U-Net + posproc. | 76 % | 82 % |
| Morph-UNet | 63 % | 73 % |
| (etiqueta real del radiólogo) | 28 % | 24 % |

Exactitud balanceada en test (azar 0.25): anotada 0.22, U-Net 0.26, posproc. 0.28, Morph-UNet 0.27.

- **Se confirma la hipótesis.** Las máscaras de la red son más suaves y redondeadas que las anotadas, y el árbol las
  manda en su mayoría a LOBULATED. El posprocesamiento, que suaviza todavía más, acentúa el sesgo; la Morph-UNet lo
  reduce un poco.
- **Ninguna máscara recupera la forma:** todas quedan en el nivel del azar. Esto es coherente con E2–E4 y E10.

## 6. Conclusión

- **Para el informe:** agregar morfología aprendible en las capas convolucionales no mejora la segmentación en este
  problema. El cuello de botella es la localización, no la regularidad del borde.
- **Morfología clásica fija:** en su forma mínima, quedarse con la componente conexa mayor, sí mejora el HD95 de
  manera robusta y el Dice de forma marginal.
- **Modelo final:** U-Net + posprocesamiento fijo (r = 6), elegido en validación. En test: Dice 0.767 ± 0.007 y HD95
  mediano 13.1 px.
- **La cifra de referencia del informe no cambia:** la U-Net, con 0.762 ± 0.009, sigue siendo la referencia; el
  posprocesamiento se informa como mejora.
