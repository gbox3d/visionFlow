from __future__ import annotations

import os
from typing import Optional, Sequence


_TRUTHY_VALUES = ("1", "true", "yes", "y", "on")


def env_value(*names: str) -> Optional[str]:
    for name in names:
        if not name:
            continue
        value = os.getenv(name)
        if value is None:
            continue
        value = value.strip()
        if value == "":
            continue
        return value
    return None


def env_str(name: str, default: str) -> str:
    value = env_value(name)
    return value if value is not None else default


def env_int(name: str, default: int) -> int:
    value = env_value(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    value = env_value(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def env_bool(name: str, default: bool) -> bool:
    value = env_value(name)
    if value is None:
        return default
    return value.lower() in _TRUTHY_VALUES


def env_lang(name: str, default: str = "auto") -> str | None:
    value = env_str(name, default).strip().lower()
    return None if value in ("", "none", "auto") else value


def env_str_any(names: Sequence[str], default: str) -> str:
    value = env_value(*names)
    return value if value is not None else default


def env_int_any(names: Sequence[str], default: int) -> int:
    value = env_value(*names)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def env_float_any(names: Sequence[str], default: float) -> float:
    value = env_value(*names)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def env_bool_any(names: Sequence[str], default: bool) -> bool:
    value = env_value(*names)
    if value is None:
        return default
    return value.lower() in _TRUTHY_VALUES


def env_lang_any(names: Sequence[str], default: str = "auto") -> str | None:
    value = env_str_any(names, default).strip().lower()
    return None if value in ("", "none", "auto") else value
