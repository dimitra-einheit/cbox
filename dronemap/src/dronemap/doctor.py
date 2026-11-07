"""`dronemap doctor`: check the machine is ready."""
from __future__ import annotations

import importlib
import os
import platform
import shutil
import subprocess
import sys

from .util import fmt_size, info

OKS, WARNS, FAILS = "OK  ", "WARN", "FAIL"


def _line(status: str, what: str, detail: str = "") -> None:
    info(f"[{status}] {what}" + (f" - {detail}" if detail else ""))


def run_doctor() -> int:
    bad = 0
    info(f"dronemap on {platform.system()} {platform.machine()}, Python {platform.python_version()}")
    if sys.version_info < (3, 9):
        _line(FAILS, "Python 3.9+ needed")
        bad += 1
    for mod, pip in (("numpy", "numpy"), ("scipy", "scipy"), ("rasterio", "rasterio"),
                     ("matplotlib", "matplotlib"), ("PIL", "pillow")):
        try:
            m = importlib.import_module(mod)
            _line(OKS, f"Python package {pip}", getattr(m, "__version__", ""))
        except ImportError:
            _line(FAILS, f"Python package {pip} missing", "run the setup script")
            bad += 1

    from .odm import docker_status, image_present

    running, msg, dinfo = docker_status()
    if running:
        mem = (dinfo.get("MemTotal") or 0) / 1024 ** 3
        _line(OKS if mem >= 24 else WARNS, "Docker running",
              f"{dinfo.get('NCPU', '?')} CPUs, {mem:.0f} GB memory"
              + ("" if mem >= 24 else " (raise Memory in Docker Desktop > Settings > Resources, e.g. 48 GB)"))
        if image_present("opendronemap/odm:latest"):
            _line(OKS, "OpenDroneMap image present")
        else:
            _line(WARNS, "OpenDroneMap image not downloaded yet", "docker pull opendronemap/odm:latest")
    else:
        _line(FAILS, "Docker", msg)
        bad += 1

    from .splat import find_opensplat

    b = find_opensplat()
    if b:
        metal = os.path.isfile(os.path.join(os.path.dirname(b), "default.metallib"))
        try:
            v = subprocess.run([b, "--version"], capture_output=True, text=True, timeout=60).stdout.strip()
        except (OSError, subprocess.TimeoutExpired) as e:
            v = f"does not start: {e}"
        if platform.system() == "Darwin":
            _line(OKS if metal else WARNS, "OpenSplat", f"{b} {v}" + ("" if metal else
                  " (CPU-only build: install Xcode and run setup/setup_mac.sh --opensplat)"))
        else:
            _line(OKS, "OpenSplat", f"{b} {v}")
    else:
        _line(WARNS, "OpenSplat not installed", "only needed for Gaussian splats (setup/setup_mac.sh --opensplat)")

    free = shutil.disk_usage(os.getcwd()).free
    _line(OKS if free > 100 * 1024 ** 3 else WARNS, "Free disk here", fmt_size(free) +
          ("" if free > 100 * 1024 ** 3 else " (a big ODM job can need 50-150 GB while running)"))
    info("")
    info("Ready." if not bad else f"{bad} problem(s) to fix (see above).")
    return 0 if not bad else 1
