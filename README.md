# ECG Arrhythmia Lab (Streamlit + PhysioNet)

Aplicación para el curso de **redes neuronales / series temporales**: visualización **tipo papel ECG** (25 mm/s, 10 mm = 1 mV), **frecuencia cardiaca** con **NeuroKit2** (picos R, bpm, alertas 60–100 lpm) y **clasificación opcional** (MLP en 4 clases alineadas a SNOMED CT).

**Dataset:** [PhysioNet ECG-Arrhythmia v1.0.0](https://physionet.org/content/ecg-arrhythmia/1.0.0/)

## Requisitos

- Python 3.11+
- Datos WFDB bajo `data/raw/WFDBRecords/` (y opcionalmente `ConditionNames_SNOMED-CT.csv`, `RECORDS`, etc.)

## Instalación

```bash
pip install -r requirements.txt
```

Incluye dependencias para **Jupyter** si quieres usar el notebook local de referencia (`jupyter`, `ipykernel`).

## Dataset

Coloca el material del ZIP en **`data/raw/`** (raíz del repo). **`src/`** es solo código.

1. Descarga: https://physionet.org/content/ecg-arrhythmia/get-zip/1.0.0/
2. Extrae de modo que exista `data/raw/WFDBRecords/` y, si puedes, `data/raw/ConditionNames_SNOMED-CT.csv`.

## Entrenar el clasificador (Google Colab)

El entrenamiento del MLP se hace en **Google Colab**:

**[Notebook Colab — entrenamiento ECG / MLP](https://colab.research.google.com/drive/1FqXI-l_JdPgBclxsA88uJzln3_7oirYB)**

1. Abre el enlace, sube o monta los datos que use el notebook (según las celdas del Colab).
2. Ejecuta las celdas hasta generar el pipeline entrenado.
3. Descarga el artefacto y colócalo en el repositorio como:

   **`artifacts/models/ecg_mlp_pipeline.joblib`**

4. La app lo detecta por defecto (`src/config/settings.py` prioriza ese nombre).


## Configuración (`.env` opcional)

| Variable | Descripción |
|----------|-------------|
| `WFDB_RECORDS_DIR` | Ruta a `WFDBRecords` (por defecto `data/raw/WFDBRecords`) |
| `SAMPLE_RECORD_COUNT` | Cuántos registros listar en la UI |
| `CLASSIFIER_MODEL_PATH` | Ruta al `.joblib` del MLP (por defecto `artifacts/models/ecg_mlp_pipeline.joblib`, si no existe se prueba `ecg_mlp_4class.joblib`) |

## Ejecutar la aplicación

```bash
streamlit run app.py
```

Tema visual por defecto en `.streamlit/config.toml`.

## Asistente y modelo

El artefacto **`ecg_mlp_pipeline.joblib`** se usa en la página de clasificación y en el chat del asistente vía `src/modeling/infer.py` (`load_model_and_infer`).

## Estructura relevante

- `src/` — configuración, datos WFDB, procesamiento (HR), visualización, modelo.
- `pages/` — pantallas Streamlit (exploración ECG, frecuencia, clasificación, asistente).
- `artifacts/models/ecg_mlp_pipeline.joblib` — pipeline MLP generado desde Colab (o entrenamiento local equivalente).
