"""Train a 3D Gaussian splat with OpenSplat from an ODM-processed project.

ODM writes opensfm/image_list.txt with paths as seen inside its Docker container
(/datasets/<project>/images/...). OpenSplat reads those paths literally, so we
build a small input folder with the same reconstruction and host paths.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time

from .util import UserError, ensure_dir, fmt_minutes, info, ok, step, warn

TOOLKIT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def find_opensplat(explicit: str | None = None) -> str | None:
    cands = []
    if explicit:
        cands.append(explicit)
    if os.environ.get("OPENSPLAT_BIN"):
        cands.append(os.environ["OPENSPLAT_BIN"])
    exe = "opensplat.exe" if os.name == "nt" else "opensplat"
    cands.append(os.path.join(TOOLKIT_DIR, "tools", "OpenSplat", "build", exe))
    cands.append(os.path.join(TOOLKIT_DIR, "tools", "OpenSplat", "build", "Release", exe))
    w = shutil.which("opensplat")
    if w:
        cands.append(w)
    for c in cands:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def total_ram_bytes() -> int:
    try:
        if sys.platform == "darwin":
            return int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout)
        if os.path.isfile("/proc/meminfo"):
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return 16 * 1024 ** 3


def prepare_input(project: str, out_dir: str) -> dict:
    """Create out_dir/{reconstruction.json,image_list.txt} with host image paths."""
    rec_path = os.path.join(project, "opensfm", "reconstruction.json")
    if not os.path.isfile(rec_path):
        raise UserError(f"{rec_path} not found. Run 'dronemap process {project} --preset detail' first.")
    with open(rec_path, encoding="utf-8") as f:
        recs = json.load(f)
    if not recs:
        raise UserError("ODM reconstruction is empty - the photos could not be aligned.")
    rec = max(recs, key=lambda r: len(r.get("shots", {})))
    bad = {c.get("projection_type") for c in rec.get("cameras", {}).values()} - {"perspective", "brown"}
    if bad:
        raise UserError(f"Camera model {', '.join(sorted(bad))} is not supported by OpenSplat. Re-run "
                        f"'dronemap process' (it uses --camera-lens brown) with --rerun-from opensfm.")
    images_dir = os.path.join(project, "images")
    by_name = {fn: os.path.join(images_dir, fn) for fn in os.listdir(images_dir)}
    by_lower = {fn.lower(): p for fn, p in by_name.items()}
    ensure_dir(out_dir)
    lines, missing = [], []
    for shot in rec["shots"]:
        p = by_name.get(shot) or by_lower.get(shot.lower())
        if p is None:
            missing.append(shot)
        else:
            lines.append(os.path.abspath(p))
    if missing:
        raise UserError(f"{len(missing)} aligned photos are missing from {images_dir} (e.g. {missing[0]}).")
    dst = os.path.join(out_dir, "reconstruction.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump([rec], f)
    with open(os.path.join(out_dir, "image_list.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    cam = next(iter(rec["cameras"].values()))
    return {"n_images": len(lines), "width": int(cam.get("width", 0)), "height": int(cam.get("height", 0)),
            "n_points": len(rec.get("points", {})), "n_total": sum(len(r.get("shots", {})) for r in recs)}


def choose_downscale(n: int, w: int, h: int, ram: int) -> float:
    """Smallest downscale (>= 2 for 12 MP+ photos) whose decoded images fit in ~30% of RAM."""
    budget = 0.30 * ram
    start = 2.0 if w * h >= 12e6 else 1.0
    for d in (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0):
        if d < start:
            continue
        if n * (w / d) * (h / d) * 3 <= budget:
            return d
    return 8.0


def run_splat(a) -> int:
    project = os.path.abspath(a.project)
    work = ensure_dir(os.path.join(project, "dronemap", "splat"))
    meta = prepare_input(project, os.path.join(work, "input"))
    if meta["n_total"] > meta["n_images"]:
        warn(f"ODM split the photos into several groups; using the largest ({meta['n_images']} of "
             f"{meta['n_total']} photos). Gaps in coverage usually cause this.")
    ram = total_ram_bytes()
    if str(a.downscale).lower() == "auto":
        ds = choose_downscale(meta["n_images"], meta["width"], meta["height"], ram)
    else:
        ds = float(a.downscale)
    w_eff, h_eff = int(meta["width"] / ds), int(meta["height"] / ds)
    # ~2 KB of GPU memory per gaussian (OpenSplat docs); keep it under ~35% of unified memory
    mg = a.max_gaussians or int(min(5_000_000, max(500_000, 0.35 * ram / 2048 / 1_000_000) * 1_000_000))
    out_ply = os.path.join(work, f"{a.name}.ply")
    binp = find_opensplat(a.opensplat)
    step(f"Gaussian splat: {meta['n_images']} photos at {w_eff}x{h_eff} (downscale {ds:g}), "
         f"{a.iters} steps, up to {mg / 1e6:.1f} M splats")
    args = [os.path.join(work, "input"), "-n", str(a.iters), "-d", f"{ds:g}", "--max-gaussians", str(mg),
            "-o", out_ply, "--save-every", "5000"]
    if a.resume:
        ck = _latest_checkpoint(work, a.name)
        if ck:
            info(f"Resuming from {ck}")
            args += ["--resume", ck]
        else:
            warn("No checkpoint found to resume from; starting fresh.")
    if binp is None:
        msg = ("OpenSplat is not installed. On the Mac run:  setup/setup_mac.sh --opensplat\n"
               "(needs Xcode for Apple-GPU support). Inputs are ready in " + os.path.join(work, "input"))
        if a.dry_run:
            warn(msg)
            info("Command: opensplat " + " ".join(shlex.quote(x) for x in args))
            return 0
        raise UserError(msg)
    cmd_base = [binp] + args
    if a.dry_run:
        info("Command (dry run): " + " ".join(shlex.quote(c) for c in cmd_base))
        return 0
    log_path = os.path.join(work, f"opensplat_{_dt.datetime.now():%Y%m%d_%H%M%S}.log")
    info(f"Log: {log_path}")
    info("Stop any time with Ctrl+C; a checkpoint is saved every 5000 steps (resume with --resume).")
    rc = _stream(cmd_base, log_path, a.iters)
    if rc != 0:
        raise UserError(f"OpenSplat failed (exit {rc}). See {log_path}")
    ok(f"Splat saved: {out_ply}")
    info("View/clean it: drag the .ply into https://playcanvas.com/supersplat/editor (runs in your browser;")
    info("the file is not uploaded unless you choose to publish). Crop stray splats, then export.")
    return 0


def _latest_checkpoint(work: str, name: str) -> str | None:
    best = None
    for fn in os.listdir(work):
        m = re.match(re.escape(name) + r"_(\d+)\.ply$", fn)
        if m and (best is None or int(m.group(1)) > best[0]):
            best = (int(m.group(1)), os.path.join(work, fn))
    if best:
        return best[1]
    p = os.path.join(work, f"{name}.ply")
    return p if os.path.isfile(p) else None


def _stream(cmd: list, log_path: str, total_steps: int) -> int:
    step_re = re.compile(r"Step (\d+):")
    t0 = time.time()
    first = None
    last_print = 0.0
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        log.write("$ " + " ".join(shlex.quote(c) for c in cmd) + "\n")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                errors="replace", bufsize=1)
        try:
            for line in proc.stdout:
                log.write(line)
                m = step_re.match(line)
                if not m:
                    sys.stdout.write(line)
                    if line.startswith("Using CPU") and platform.system() == "Darwin":
                        warn("OpenSplat is running on the CPU (about 100x slower). Rebuild with Metal: "
                             "setup/setup_mac.sh --opensplat")
                    continue
                s = int(m.group(1))
                now = time.time()
                if first is None:
                    first = (s, now)
                    continue
                if now - last_print >= 30 or s >= total_steps:
                    rate = (now - first[1]) / max(1, s - first[0])
                    eta = rate * max(0, total_steps - s)
                    sys.stdout.write(f"  step {s}/{total_steps}  {rate:.2f} s/step  ETA {fmt_minutes(eta)}  "
                                     f"(elapsed {fmt_minutes(now - t0)})\n")
                    sys.stdout.flush()
                    last_print = now
            return proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()
            raise
