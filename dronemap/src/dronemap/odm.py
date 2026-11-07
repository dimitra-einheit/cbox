"""Run OpenDroneMap (ODM) in Docker with presets tuned for a DJI Mavic 2 Pro."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys

from .util import UserError, fmt_size, info, ok, step, warn

IMAGE_EXT = {".jpg", ".jpeg", ".tif", ".tiff", ".dng"}
VIDEO_EXT = {".mp4", ".mov", ".lrv", ".ts", ".avi", ".mkv"}

# ODM options per preset and quality. Values are strings; None = flag without value.
COMMON = {
    "--camera-lens": "brown",  # explicit lens model; also what OpenSplat needs
    "--auto-boundary": None,
    "--cog": None,
    "--build-overviews": None,
    "--gltf": None,  # single-file textured 3D model (.glb)
}

PRESETS = {
    "survey": {
        "base": {"--dsm": None, "--dtm": None, "--orthophoto-kmz": None},
        "fast": {"--pc-quality": "medium", "--feature-quality": "medium", "--dem-resolution": "10",
                 "--orthophoto-resolution": "5", "--mesh-size": "150000"},
        "standard": {"--pc-quality": "high", "--feature-quality": "high", "--dem-resolution": "5",
                     "--orthophoto-resolution": "2", "--mesh-size": "300000"},
        "max": {"--pc-quality": "ultra", "--feature-quality": "ultra", "--dem-resolution": "3",
                "--orthophoto-resolution": "1.5", "--mesh-size": "500000"},
        "gb_per_image": {"fast": 0.03, "standard": 0.06, "max": 0.14},
    },
    "detail": {
        "base": {"--dsm": None, "--orthophoto-kmz": None},
        "fast": {"--pc-quality": "medium", "--feature-quality": "high", "--mesh-octree-depth": "11",
                 "--mesh-size": "300000", "--dem-resolution": "5", "--orthophoto-resolution": "2"},
        "standard": {"--pc-quality": "high", "--feature-quality": "ultra", "--mesh-octree-depth": "12",
                     "--mesh-size": "1000000", "--dem-resolution": "2", "--orthophoto-resolution": "1"},
        "max": {"--pc-quality": "ultra", "--feature-quality": "ultra", "--mesh-octree-depth": "13",
                "--mesh-size": "2000000", "--dem-resolution": "2", "--orthophoto-resolution": "1"},
        "gb_per_image": {"fast": 0.04, "standard": 0.07, "max": 0.16},
    },
}

KEY_OUTPUTS = [
    ("odm_orthophoto/odm_orthophoto.tif", "Orthophoto (GeoTIFF) - open in QGIS"),
    ("odm_orthophoto/odm_orthophoto.kmz", "Orthophoto for Google Earth"),
    ("odm_dem/dtm.tif", "Terrain model, ground only (DTM) - used by 'dronemap site/pond'"),
    ("odm_dem/dsm.tif", "Surface model incl. trees/buildings (DSM)"),
    ("odm_texturing/odm_textured_model_geo.glb", "Textured 3D model (GLB) - Blender, glTF viewers"),
    ("odm_texturing/odm_textured_model_geo.obj", "Textured 3D model (OBJ + textures)"),
    ("odm_georeferencing/odm_georeferenced_model.laz", "Point cloud (LAZ) - CloudCompare"),
    ("odm_report/report.pdf", "Quality report (overlap, accuracy, GSD)"),
]


def list_images(images_dir: str) -> tuple[list, list]:
    imgs, vids = [], []
    for fn in sorted(os.listdir(images_dir)):
        p = os.path.join(images_dir, fn)
        if not os.path.isfile(p) or fn.startswith("."):
            continue
        ext = os.path.splitext(fn)[1].lower()
        if ext in IMAGE_EXT:
            imgs.append(p)
        elif ext in VIDEO_EXT:
            vids.append(p)
    return imgs, vids


def _docker(*args, capture=True, timeout=60):
    return subprocess.run(["docker", *args], capture_output=capture, text=True, timeout=timeout)


def docker_status() -> tuple[bool, str, dict]:
    """(running, message, info) about the Docker daemon."""
    if not shutil.which("docker"):
        return False, "Docker is not installed (install Docker Desktop - see setup/setup_mac.sh)", {}
    try:
        r = _docker("info", "--format", "{{json .}}")
    except subprocess.TimeoutExpired:
        return False, "Docker did not answer. Is Docker Desktop still starting?", {}
    if r.returncode != 0:
        return False, "Docker is installed but not running. Open Docker Desktop and wait for 'Engine running'.", {}
    try:
        data = json.loads(r.stdout)
    except json.JSONDecodeError:
        data = {}
    return True, "Docker is running", data


def image_present(image: str) -> bool:
    try:
        return _docker("image", "inspect", image).returncode == 0
    except subprocess.TimeoutExpired:
        return False


def build_odm_args(preset: str, quality: str, rolling_shutter: bool = True, sky_removal: bool = False,
                   gcp_container_path: str | None = None, boundary_container_path: str | None = None,
                   extra: str = "") -> list:
    P = PRESETS[preset]
    opts: dict = {}
    opts.update(COMMON)
    opts.update(P["base"])
    opts.update(P[quality])
    if rolling_shutter:
        opts["--rolling-shutter"] = None  # Mavic 2 Pro: 56 ms readout (in ODM's database)
    if sky_removal:
        opts["--sky-removal"] = None
    if gcp_container_path:
        opts["--gcp"] = gcp_container_path
    if boundary_container_path:
        opts["--boundary"] = boundary_container_path
        opts.pop("--auto-boundary", None)
    args = []
    for k, v in opts.items():
        args.append(k)
        if v is not None:
            args.append(v)
    if extra:
        args.extend(shlex.split(extra))
    return args


def _signature(args: list) -> str:
    clean = []
    skip = False
    for a in args:
        if skip:
            skip = False
            continue
        if a in ("--rerun-from", "--rerun", "--end-with"):
            skip = True
            continue
        if a == "--rerun-all":
            continue
        clean.append(a)
    return hashlib.sha1(" ".join(clean).encode()).hexdigest()[:12]


def _boundary_to_geojson(path: str, out_path: str) -> str:
    from .geo import read_polygon

    _, ring = read_polygon(path)
    coords = [[lon, lat] for lat, lon in ring] + [[ring[0][1], ring[0][0]]]
    gj = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {},
                                                      "geometry": {"type": "Polygon", "coordinates": [coords]}}]}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(gj, f)
    return out_path


def run_process(a) -> int:
    project = os.path.abspath(a.project)
    name = os.path.basename(project.rstrip(os.sep))
    parent = os.path.dirname(project.rstrip(os.sep))
    images_dir = os.path.join(project, "images")
    if not os.path.isdir(images_dir):
        raise UserError(f"No 'images' folder in {project}. Put the photos in {images_dir}")
    if " " in name:
        raise UserError(f"Project folder name must not contain spaces: '{name}'")
    imgs, vids = list_images(images_dir)
    if vids:
        raise UserError(
            f"{len(vids)} video file(s) in {images_dir} (e.g. {os.path.basename(vids[0])}). ODM would turn them "
            "into thousands of frames. Move videos out of the images folder and run again."
        )
    if len(imgs) < 5:
        raise UserError(f"Only {len(imgs)} photos in {images_dir}; need at least 5 (ideally hundreds).")
    total = sum(os.path.getsize(p) for p in imgs)
    step(f"Project '{name}': {len(imgs)} photos ({fmt_size(total)}), preset {a.preset}, quality {a.quality}")

    # GPS sanity check on a sample of photos (needs Pillow; skip quietly otherwise)
    try:
        from .photos import read_photo_meta

        sample = imgs[:: max(1, len(imgs) // 10)][:10]
        with_gps = sum(1 for p in sample if read_photo_meta(p).get("lat") is not None)
        if with_gps == 0:
            warn("Sampled photos have no GPS tags. ODM can still build a 3D model, but it will not be "
                 "georeferenced (no real-world coordinates or elevations).")
    except ImportError:
        pass

    gcp_host = None
    if a.gcp:
        if not os.path.isfile(a.gcp):
            raise UserError(f"GCP file not found: {a.gcp}")
        gcp_host = os.path.join(project, "gcp_list.txt")
        if os.path.abspath(a.gcp) != gcp_host:
            shutil.copyfile(a.gcp, gcp_host)
    elif os.path.isfile(os.path.join(project, "gcp_list.txt")):
        gcp_host = os.path.join(project, "gcp_list.txt")
        info(f"Using ground control points from {gcp_host}")
    boundary_host = None
    if a.boundary:
        boundary_host = _boundary_to_geojson(a.boundary, os.path.join(project, "dronemap_boundary.geojson"))

    ctr = f"/datasets/{name}"
    odm_args = build_odm_args(
        a.preset, a.quality, rolling_shutter=not a.no_rolling_shutter, sky_removal=a.sky_removal,
        gcp_container_path=f"{ctr}/gcp_list.txt" if gcp_host else None,
        boundary_container_path=f"{ctr}/dronemap_boundary.geojson" if boundary_host else None,
        extra=a.odm_args,
    )
    sig = _signature(odm_args)
    state_path = os.path.join(project, "dronemap_last_run.json")
    prev = None
    if os.path.isfile(state_path):
        try:
            with open(state_path, encoding="utf-8") as f:
                prev = json.load(f)
        except (OSError, json.JSONDecodeError):
            prev = None
    if a.rerun_from:
        odm_args += ["--rerun-from", a.rerun_from]
    elif prev and prev.get("signature") != sig and os.path.isdir(os.path.join(project, "opensfm")):
        warn("Settings differ from the previous run of this project - reprocessing everything (--rerun-all).")
        odm_args.append("--rerun-all")
    elif prev and prev.get("signature") == sig:
        info("Same settings as last run: ODM will skip stages that already finished (safe to resume).")

    container = "dronemap-" + re.sub(r"[^a-z0-9_.-]", "-", name.lower())
    mount = parent.replace("\\", "/") if os.name == "nt" else parent  # Docker Desktop on Windows wants C:/...
    cmd = ["docker", "run", "--rm", "--name", container,
           "-v", f"{mount}:/datasets", a.image, "--project-path", "/datasets", name] + odm_args

    if a.dry_run:
        info("Docker command (dry run):")
        info("  " + " ".join(shlex.quote(c) for c in cmd))
        return 0

    running, msg, dinfo = docker_status()
    if not running:
        raise UserError(msg)
    mem_gb = (dinfo.get("MemTotal") or 0) / 1024 ** 3
    ncpu = dinfo.get("NCPU") or 0
    need = 4 + len(imgs) * PRESETS[a.preset]["gb_per_image"][a.quality]
    info(f"Docker resources: {ncpu} CPUs, {mem_gb:.0f} GB memory. Rough need for this job: ~{need:.0f} GB.")
    if mem_gb and need > mem_gb:
        warn("This job may run out of memory. In Docker Desktop > Settings > Resources raise Memory "
             "(e.g. 48 GB on a 64 GB Mac), or use --quality fast, or split it: "
             "--odm-args \"--split 300 --split-overlap 60\".")
    if not image_present(a.image):
        step(f"Downloading ODM image {a.image} (about 2 GB, one time)")
        if subprocess.run(["docker", "pull", a.image]).returncode != 0:
            raise UserError(f"Could not download {a.image}. Check the internet connection and try again.")

    # Remove a leftover container with the same name (e.g. after a crash)
    subprocess.run(["docker", "rm", "-f", container], capture_output=True)

    log_path = os.path.join(project, f"dronemap_odm_{_dt.datetime.now():%Y%m%d_%H%M%S}.log")
    step("Running OpenDroneMap. This can take from 30 minutes to many hours.")
    info(f"Full log: {log_path}")
    info("You can stop with Ctrl+C and run the same command later to resume.")
    # remember these settings now, so an interrupted run resumes only with the same settings
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump({"signature": sig, "preset": a.preset, "quality": a.quality, "args": odm_args,
                   "status": "started", "started": _dt.datetime.now().isoformat(timespec="seconds")}, f, indent=2)
    t0 = _dt.datetime.now()
    rc = _stream(cmd, log_path, container)
    took = _dt.datetime.now() - t0
    if rc != 0:
        if rc == 137:
            warn("ODM was killed, most likely out of memory (exit 137). Raise Docker's memory limit, use "
                 "--quality fast, or add --odm-args \"--split 300 --split-overlap 60\", then run again.")
        raise UserError(f"ODM failed (exit code {rc}) after {str(took).split('.')[0]}. "
                        f"The end of {log_path} says why.")
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump({"signature": sig, "preset": a.preset, "quality": a.quality, "args": odm_args,
                   "status": "finished", "finished": _dt.datetime.now().isoformat(timespec="seconds")}, f,
                  indent=2)
    ok(f"ODM finished in {str(took).split('.')[0]}")
    _report_outputs(project)
    return 0


def _stream(cmd: list, log_path: str, container: str) -> int:
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        log.write("$ " + " ".join(shlex.quote(c) for c in cmd) + "\n")
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                    errors="replace", bufsize=1)
        except FileNotFoundError:
            raise UserError("The 'docker' command was not found.") from None
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
            return proc.wait()
        except KeyboardInterrupt:
            warn("Stopping ODM container...")
            subprocess.run(["docker", "stop", container], capture_output=True)
            proc.wait()
            raise


def _report_outputs(project: str) -> None:
    lines = []
    for rel, desc in KEY_OUTPUTS:
        p = os.path.join(project, rel)
        if os.path.isfile(p):
            lines.append(f"  {rel:50s} {fmt_size(os.path.getsize(p)):>10s}  {desc}")
    step("Results")
    for l in lines:
        info(l)
    with open(os.path.join(project, "dronemap_outputs.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    info("Next: 'dronemap site <project>' for terrain analysis, 'dronemap pond ...' for pond design,")
    info("      'dronemap splat <project>' for a Gaussian splat (detail preset).")
