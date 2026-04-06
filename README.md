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

Incluye dependencias para ejecutar el **notebook** de entrenamiento (`jupyter`, `ipykernel`).

## Dataset

Coloca el material del ZIP en **`data/raw/`** (raíz del repo). **`src/`** es solo código.

1. Descarga: https://physionet.org/content/ecg-arrhythmia/get-zip/1.0.0/
2. Extrae de modo que exista `data/raw/WFDBRecords/` y, si puedes, `data/raw/ConditionNames_SNOMED-CT.csv`.

## Entrenar el clasificador (notebook)

1. Abre Jupyter desde la **raíz del proyecto** (donde están `src/` y `notebooks/`).  
   Si la terminal está en `src/`, ejecuta antes `cd ..`. Si lanzas `jupyter notebook notebooks/...` **desde `src/`**, buscará `src/notebooks/...` y dará *No such file or directory*.

   ```bash
   jupyter notebook notebooks/train_ecg_classifier.ipynb
   ```

2. La primera celda fija `ROOT`, hace `os.chdir(ROOT)` y añade el repo a `sys.path` (coherente con Streamlit).

3. Ejecuta todas las celdas. Se genera `artifacts/models/ecg_mlp_4class.joblib` (MLP baseline; límites editables en el notebook).

4. La app carga ese archivo vía `CLASSIFIER_MODEL_PATH` en `.env` (opcional).

**Si ves `Registros .hea encontrados: 0`:** las rutas relativas del `.env` (p. ej. `data/raw/WFDBRecords`) se resuelven ahora respecto a la **raíz del repo**, no al directorio desde el que abriste Jupyter. Comprueba que exista esa carpeta y que dentro haya `WFDBRecords/.../*.hea`.

Para **regenerar** el `.ipynb` desde el script del repo:

```bash
python scripts/create_training_notebook.py
```

## Configuración (`.env` opcional)

| Variable | Descripción |
|----------|-------------|
| `WFDB_RECORDS_DIR` | Ruta a `WFDBRecords` (por defecto `data/raw/WFDBRecords`) |
| `SAMPLE_RECORD_COUNT` | Cuántos registros listar en la UI |
| `CLASSIFIER_MODEL_PATH` | Ruta al `.joblib` del MLP (por defecto `artifacts/models/ecg_mlp_4class.joblib`) |

## Ejecutar la aplicación

```bash
streamlit run app.py
```

Tema visual por defecto en `.streamlit/config.toml`.

## Integración futura (chatbot)

El mismo artefacto `ecg_mlp_4class.joblib` puede cargarse desde otro servicio o un agente: la interfaz común es `src/modeling/infer.py` (`load_model_and_infer`).

## Estructura relevante

- `src/` — configuración, datos WFDB, procesamiento (HR), visualización, modelo.
- `pages/` — pantallas Streamlit (dataset, ECG, frecuencia, clasificación).
- `notebooks/train_ecg_classifier.ipynb` — entrenamiento offline del MLP.
