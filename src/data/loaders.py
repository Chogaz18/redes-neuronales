from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import wfdb

from src.config.settings import Settings
from src.data.exceptions import ECGReadError, LeadNotFoundError
from src.data.metadata import RecordRef
from src.data.parsers import choose_rpeak_lead, header_to_metadata
from src.data.wfdb_paths import wfdb_local_record_name
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ECGRecord:
    record_id: str
    fs: float
    lead_names: List[str]
    signals: np.ndarray  # shape: [n_leads, n_samples]
    snomed_ct_codes: List[str]


def _slice_samples(signals: np.ndarray, start_s: float, duration_s: Optional[float], fs: float) -> np.ndarray:
    start_idx = int(round(start_s * fs))
    if duration_s is None:
        return signals[:, start_idx:]
    end_idx = int(round((start_s + duration_s) * fs))
    return signals[:, start_idx:end_idx]


def load_ecg_record(
    settings: Settings,
    record_ref: RecordRef,
    *,
    leads: Optional[Sequence[str]] = None,
    start_s: float = 0.0,
    duration_s: Optional[float] = None,
) -> ECGRecord:
    """
    Carga señales desde WFDB y retorna una estructura homogénea.
    """
    try:
        rn = wfdb_local_record_name(record_ref)
        header = wfdb.rdheader(rn, pn_dir=None)
        fs = float(header.fs)
        full = wfdb.rdrecord(rn, pn_dir=None)
        # full.p_signal shape: [n_samples, n_sig]
        p_signal = np.asarray(full.p_signal, dtype=float)
        signals = p_signal.T
        lead_names = list(full.sig_name) if getattr(full, "sig_name", None) is not None else list(header.sig_name)
        snomed_codes = []
        try:
            # Heurística: SNOMED desde `.hea`
            snomed_codes = header_to_metadata(record_ref).get("snomed_ct_codes", [])
        except Exception:
            snomed_codes = []

        if leads is not None:
            indices = []
            for lead in leads:
                try:
                    indices.append(lead_names.index(lead))
                except ValueError as e:
                    raise LeadNotFoundError(f"Lead '{lead}' no encontrado en {record_ref.record_id}") from e
            signals = signals[indices, :]
            lead_names = list(leads)

        signals = _slice_samples(signals, start_s=start_s, duration_s=duration_s, fs=fs)
        return ECGRecord(
            record_id=record_ref.record_id,
            fs=fs,
            lead_names=lead_names,
            signals=signals,
            snomed_ct_codes=snomed_codes,
        )
    except LeadNotFoundError:
        raise
    except Exception as e:
        logger.exception("Error cargando registro %s", record_ref.record_id)
        raise ECGReadError(f"No se pudo leer el registro: {record_ref.record_id}.") from e

