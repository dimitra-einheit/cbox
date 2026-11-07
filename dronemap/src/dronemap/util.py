"""Small shared helpers: console output, formatting, errors."""
from __future__ import annotations

import os
import sys


class UserError(Exception):
    """An error caused by input the user can fix. Printed without a traceback."""


_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


def info(msg: str) -> None:
    print(msg, flush=True)


def step(msg: str) -> None:
    print(_c("1;36", "==> ") + _c("1", msg), flush=True)


def ok(msg: str) -> None:
    print(_c("32", "OK  ") + msg, flush=True)


def warn(msg: str) -> None:
    print(_c("33", "WARNING: ") + msg, flush=True)


def fail(msg: str) -> None:
    print(_c("31", "ERROR: ") + msg, file=sys.stderr, flush=True)


def fmt_area(m2: float) -> str:
    acres = m2 / 4046.8564224
    return f"{m2:,.0f} m² ({acres:,.2f} acres / {m2 / 10000:,.2f} ha)"


def fmt_minutes(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    return f"{m} min {s:02d} s"


def fmt_size(nbytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if nbytes < 1024 or unit == "TB":
            return f"{nbytes:,.1f} {unit}" if unit != "B" else f"{int(nbytes)} B"
        nbytes /= 1024.0
    return f"{nbytes:.1f} TB"


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def google_maps_link(lat: float, lon: float) -> str:
    return f"https://www.google.com/maps/search/?api=1&query={lat:.6f},{lon:.6f}"
