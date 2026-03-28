from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, TypeVar

T = TypeVar("T")


def first_or_none(items: Iterable[T]) -> Optional[T]:
    for item in items:
        return item
    return None


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def safe_int(value: object, default: int) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except Exception:
        return default


def safe_float(value: object, default: float) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return default


def unique_preserve_order(items: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

