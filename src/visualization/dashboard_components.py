from __future__ import annotations

import streamlit as st


def dataset_status_message(has_dataset: bool, wfdb_dir: str) -> None:
    if has_dataset:
        st.success(f"Dataset detectado en: {wfdb_dir}")
    else:
        st.warning(f"Dataset no detectado. Esperado en: {wfdb_dir}")

