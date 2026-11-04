# Resumen: ablaciones, variabilidad e incertidumbre

Protocolo: toda comparación o selección se hizo con validación (partición por paciente) o con validación
cruzada sobre el train oficial. El test oficial (378 recortes, 201 pacientes) se evaluó solo con la
configuración final: `configs/main.yaml` (context 3, jitter 0.2, U-Net base 16, BCE+Dice, 40 épocas).

## Tabla resumen

| Análisis | Datos | U-Net (Dice) | Prior centrado | Clásico |
|---|---|---|---|---|
| Referencia, semilla 42 | val | 0.781 | 0.345 | 0.348 |
| Ablación base 32 | val | 0.767 (−0.014) | — | — |
| Ablación solo Dice | val | 0.767 (−0.014) | — | — |
| Ablación jitter 0 | val | 0.839 | **0.857** | 0.344 |
| 3 semillas (42, 43, 44) | val | 0.777 ± 0.003 | — | — |
| 3 semillas (42, 43, 44) | test | 0.762 ± 0.009 | — | — |
| CV 5 particiones por paciente | train oficial | 0.771 ± 0.014 | 0.348 ± 0.021 | — |
| Test, semilla 42, IC 95 % por recorte | test | 0.769 [0.746, 0.791] | 0.352 [0.330, 0.376] | 0.324 [0.302, 0.348] |
| Test, IC 95 % por paciente | test | [0.746, 0.795] | [0.333, 0.374] | [0.299, 0.354] |
| Diferencia U-Net − clásico | test | +0.445 [0.420, 0.469]; Wilcoxon p = 4.6e-60; U-Net mejor en 93.9 % de los recortes | | |

## Interpretación

- **Ventaja de la U-Net:** supera al método clásico y al prior por unos 0.42–0.45 de Dice. Esa ventaja es mucho
  mayor que cualquier fuente de variación medida: entre semillas DE ≈ 0.01, entre particiones DE ≈ 0.014, y el
  IC por bootstrap tiene un ancho de ≈ 0.05.
- **Generalización:** el test (0.762 ± 0.009) queda 0.015 por debajo de la validación (0.777) y dentro del rango
  de la CV (0.756–0.787). No hay indicio de sobreajuste a la partición de validación.
- **Atajo de posición:** sin jitter, un círculo fijo en el centro alcanza Dice 0.857 y supera a la U-Net. Con
  jitter 0, el Dice mide sobre todo la posición de la lesión en el recorte, no la calidad de la segmentación.
  Al subir el jitter, el prior cae de 0.86 a 0.16, mientras que la U-Net baja solo de 0.84 a 0.75. Por eso se
  mantiene jitter 0.2 aunque dé un Dice menor.
- **Context:** un context menor sube el Dice (0.80 con context 2), pero también sube el prior (0.53). La tarea se
  vuelve más fácil, no mejor resuelta. Context y jitter se reportan de forma descriptiva y no se usaron para
  elegir.
- **Lo que no mejora:** duplicar los canales (base 32: 2,3 veces más tiempo por época) y usar solo Dice como
  pérdida no superan a la referencia en validación (−0.014 cada uno). Con una sola semilla y DE entre semillas
  ≈ 0.003–0.009, no hay evidencia de que alguno sea mejor, así que se mantiene la configuración más simple.
- **Limitaciones:** solo 3 semillas; las ablaciones usan una sola semilla. En el 6 % de los recortes de test el
  clásico supera a la U-Net; esos casos no se analizaron por separado. La clasificación de forma BI-RADS sigue sin señal útil (E2–E4); no se reabrió aquí.
