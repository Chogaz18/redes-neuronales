from __future__ import annotations


class DatasetNotFoundError(FileNotFoundError):
    pass


class RecordNotFoundError(FileNotFoundError):
    pass


class ECGReadError(RuntimeError):
    pass


class LeadNotFoundError(KeyError):
    pass

