# ECG Arrhythmia Explorer (Streamlit + PhysioNet)

Aplicación en **Streamlit** para visualizar ECGs (estilo papel electrocardiográfico) y analizar **frecuencia cardiaca** detectando picos **R** con **NeuroKit2**. Dataset: **PhysioNet ECG-Arrhythmia v1.0.0**.

## Requisitos

- Python 3.11+
- El dataset en formato **WFDB**, con la carpeta `WFDBRecords/` bajo `data/raw/`.

## Instalación

```bash
pip install -r requirements.txt
```

## Dataset

Coloca el material del ZIP en **`data/raw/`** (carpeta en la raíz del repo). **`src/`** es solo código; no hace falta copiar señales dentro de `src/data/`.

1. Se descarga el ZIP desde PhysioNet:
   https://physionet.org/content/ecg-arrhythmia/get-zip/1.0.0/
2. Se extrae el contenido en `data/raw/` de forma que queden, como mínimo:
   - `data/raw/WFDBRecords/` (señales WFDB)
   - `data/raw/ConditionNames_SNOMED-CT.csv` (mapeo de diagnósticos SNOMED)
   - `data/raw/RECORDS` (lista de rutas de registros)
   - Opcional: `data/raw/LICENSE.txt`, `data/raw/SHA256SUMS.txt`

Las rutas están centralizadas en `src/utils/paths.py` (`condition_names_snomed_csv`, etc.).

## Configuración

Opcional: crea un archivo `.env` en la raíz del proyecto.

Variables soportadas:

- `WFDB_RECORDS_DIR` (por defecto: `data/raw/WFDBRecords`)
- `SAMPLE_RECORD_COUNT` (por defecto: `20`)

## Ejecución

```bash
streamlit run app.py
```

## Estructura del proyecto

`src/` contiene módulos separados por responsabilidad: config, data ingestion, procesamiento (HR) y visualización.

