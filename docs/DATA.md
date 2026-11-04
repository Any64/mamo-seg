# Datos: CBIS-DDSM (versión de Kaggle)

## Origen

* Dataset usado: [CBIS-DDSM Breast Cancer Image Dataset (awsaf49, Kaggle)](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset),
  una conversión a JPEG del [CBIS-DDSM de TCIA](https://www.cancerimagingarchive.net/collection/cbis-ddsm/)
  (Lee et al., 2017). Tamaño aproximado: 5 GB comprimido.
* Licencia y condiciones: las del CBIS-DDSM original (TCIA, CC BY 3.0). **Los datos no se incluyen en este
  repositorio.**
* Solo se usan las **masas** (`mass_case_description_*`); las calcificaciones no se usan.

## Obtención

```bash
python scripts/download_data.py               # usa la API de Kaggle si hay credenciales
python scripts/download_data.py --check-only  # solo verifica la estructura
```

Sin credenciales de Kaggle, el script imprime las instrucciones para la descarga manual (descargar el ZIP desde la
página del dataset y descomprimirlo).

## Dónde ponerlo

El código busca la carpeta del dataset en este orden:

1. `--data-dir` en la línea de comandos de los scripts;
2. la variable de entorno `MAMO_DATA_DIR`;
3. `archive_dir` en la config (`configs/main.yaml`): por defecto `data/CBIS-DDSM` dentro del repositorio
   (la carpeta `data/` está en `.gitignore`).

```powershell
# Windows (PowerShell): solo la sesión actual / de forma permanente
$env:MAMO_DATA_DIR = "D:\datos\CBIS-DDSM"
setx MAMO_DATA_DIR "D:\datos\CBIS-DDSM"
```

## Estructura esperada

```
CBIS-DDSM/
├── csv/
│   ├── mass_case_description_train_set.csv    <- usado (1318 masas)
│   ├── mass_case_description_test_set.csv     <- usado (378 masas, test oficial)
│   ├── calc_case_description_{train,test}_set.csv
│   ├── dicom_info.csv
│   └── meta.csv
└── jpeg/
    └── 1.3.6.1.4.1.9590.100.1.2.<UID>/        <- una carpeta por serie DICOM
        ├── 1-263.jpg
        └── 2-241.jpg ...
```

## Particularidades (documentadas durante la exploración)

Implementadas en `src/data_index.py` y `src/preprocess.py`:

1. **Rutas del CSV.** Las columnas `image file path`, `cropped image file path` y `ROI mask file path` tienen la forma
   `caso/UID_A/UID_B/archivo.dcm`. Solo el **penúltimo segmento (UID_B)** coincide con una carpeta real de `jpeg/`;
   el nombre del archivo `.dcm` no existe como tal.
2. **Nombres de archivo sin significado.** Dentro de cada carpeta los archivos se llaman `1-263.jpg`, `2-241.jpg`…, sin
   indicar si son recorte o máscara, y a veces el recorte y la máscara comparten carpeta. Se distinguen por su
   contenido: la máscara es casi binaria (≤ 100 niveles de gris distintos) y el recorte tiene el rango completo.
3. **La máscara está a escala de la mamografía completa.** La máscara ROI tiene el tamaño de la mamografía completa
   (o su misma proporción, en 65 casos de train: se reescala), **no** el del recorte oficial, y no está registrada
   con él. Por eso los recortes se generan desde la mamografía completa y la máscara (`src/preprocess.py`), lo que
   garantiza imagen y máscara alineadas por construcción. Usar el recorte oficial con la máscara desalinea ambos.
4. **Etiqueta de forma.** El campo `mass shape` incluye, además de ROUND / OVAL / LOBULATED / IRREGULAR, etiquetas
   combinadas (p. ej. `IRREGULAR-ARCHITECTURAL_DISTORTION`) u otras (`LYMPH_NODE`, `ASYMMETRIC_BREAST_TISSUE`…);
   el análisis de forma usa solo las 4 básicas (1100 de 1318 masas de train).
5. **Patología.** `BENIGN_WITHOUT_CALLBACK` se agrupa con `BENIGN`.
6. **Calidad de las máscaras.** Muchas máscaras son contornos toscos o figuras geométricas trazadas con herramienta
   (círculos, elipses), con independencia de la forma anotada. Ver `scripts/shape_diagnostics.py` y la limitación
   correspondiente en el README.

## Datos derivados (caché)

El primer `run_experiment.py` (o el notebook) genera los recortes en `outputs/processed/`
(`{train,test}_s256_c3_j0.2.npz` + `*_meta.csv`, ~60 MB, ~11 min). El nombre incluye `img_size`, `context` y
`jitter`: si cambian, se regeneran. Resultado esperado: 1318 recortes de train y 378 de test, sin descartes.
