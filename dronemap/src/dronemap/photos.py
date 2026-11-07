"""`dronemap photos`: read drone photo GPS/gimbal tags and check ground coverage in the field."""
from __future__ import annotations

import datetime as _dt
import json
import math
import os
import re

from .util import UserError, ensure_dir, fmt_area, info, ok, step, warn

_XMP_KEYS = {
    "rel_alt": rb'drone-dji:RelativeAltitude="([-+0-9.]+)"',
    "abs_alt": rb'drone-dji:AbsoluteAltitude="([-+0-9.]+)"',
    "gimbal_pitch": rb'drone-dji:GimbalPitchDegree="([-+0-9.]+)"',
    "gimbal_yaw": rb'drone-dji:GimbalYawDegree="([-+0-9.]+)"',
    "flight_yaw": rb'drone-dji:FlightYawDegree="([-+0-9.]+)"',
}


def _rat(v) -> float:
    try:
        return float(v)
    except TypeError:
        return v[0] / v[1] if v[1] else 0.0


def _dms(t) -> float:
    return _rat(t[0]) + _rat(t[1]) / 60.0 + _rat(t[2]) / 3600.0


def read_photo_meta(path: str) -> dict:
    """GPS position, altitude, gimbal angles and time of a photo (DJI XMP + EXIF)."""
    from PIL import Image

    meta = {"path": path, "name": os.path.basename(path), "lat": None, "lon": None}
    try:
        with Image.open(path) as im:
            meta["width"], meta["height"] = im.size
            ex = im.getexif()
            meta["make"] = str(ex.get(271, "")).strip("\x00 ")
            meta["model"] = str(ex.get(272, "")).strip("\x00 ")
            gps = ex.get_ifd(0x8825)
            if gps and 2 in gps and 4 in gps:
                lat = _dms(gps[2])
                lon = _dms(gps[4])
                if str(gps.get(1, "N")).upper().startswith("S"):
                    lat = -lat
                if str(gps.get(3, "E")).upper().startswith("W"):
                    lon = -lon
                meta["lat"], meta["lon"] = lat, lon
                if 6 in gps:
                    meta["gps_alt"] = _rat(gps[6])
            exif = ex.get_ifd(0x8769)
            dt = exif.get(36867) or ex.get(306)
            if dt:
                try:
                    meta["time"] = _dt.datetime.strptime(str(dt).strip("\x00 "), "%Y:%m:%d %H:%M:%S")
                except ValueError:
                    pass
    except OSError as e:
        meta["error"] = str(e)
        return meta
    try:
        with open(path, "rb") as f:
            head = f.read(256 * 1024)
        for k, rx in _XMP_KEYS.items():
            m = re.search(rx, head)
            if m:
                meta[k] = float(m.group(1))
    except OSError:
        pass
    return meta


def _groups_by_time(metas, gap_s=120):
    flights = []
    cur = []
    for m in metas:
        if cur and "time" in m and "time" in cur[-1] and (m["time"] - cur[-1]["time"]).total_seconds() > gap_s:
            flights.append(cur)
            cur = []
        cur.append(m)
    if cur:
        flights.append(cur)
    return flights


def check_photos(images: str, plan_json: str | None, out: str | None) -> int:
    import numpy as np

    from .camera import CAMERAS, DEFAULT_CAMERA
    from .geo import LocalProjection
    from .kml import PALETTE, KmlDoc

    if not os.path.isdir(images):
        raise UserError(f"Folder not found: {images}")
    files = []
    for root, _, fns in os.walk(images):
        for fn in fns:
            if fn.lower().endswith((".jpg", ".jpeg", ".dng")) and not fn.startswith("."):
                files.append(os.path.join(root, fn))
    files.sort()
    if not files:
        raise UserError(f"No photos in {images}")
    out = ensure_dir(out or os.path.join(os.path.dirname(os.path.abspath(images.rstrip(os.sep))), "photo_check"))
    step(f"Reading {len(files)} photos")
    metas = [read_photo_meta(p) for p in files]
    geo = [m for m in metas if m.get("lat") is not None]
    if not geo:
        raise UserError("None of the photos have GPS tags.")
    geo.sort(key=lambda m: (m.get("time") or _dt.datetime.min, m["name"]))
    plan = None
    if plan_json:
        with open(plan_json, encoding="utf-8") as f:
            plan = json.load(f)
    cam = CAMERAS.get(plan["camera"] if plan else DEFAULT_CAMERA, CAMERAS[DEFAULT_CAMERA])
    lat0 = sum(m["lat"] for m in geo) / len(geo)
    lon0 = sum(m["lon"] for m in geo) / len(geo)
    proj = LocalProjection(lat0, lon0)
    for m in geo:
        m["x"], m["y"] = proj.to_xy(m["lat"], m["lon"])
    flights = _groups_by_time(geo)
    rel = [m["rel_alt"] for m in geo if "rel_alt" in m]
    nadir = [m for m in geo if m.get("gimbal_pitch", -90) <= -80]
    oblique = [m for m in geo if m.get("gimbal_pitch", -90) > -80]

    # ---- coverage: how many nadir photos see each 1 m cell (approximate footprints)
    poly_xy = [proj.to_xy(lat, lon) for lat, lon in plan["polygon"]] if plan else None
    xs = [m["x"] for m in geo] + ([p[0] for p in poly_xy] if poly_xy else [])
    ys = [m["y"] for m in geo] + ([p[1] for p in poly_xy] if poly_xy else [])
    pad = 60.0
    x0, x1, y0, y1 = min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad
    cs = 1.0
    nx, ny = int((x1 - x0) / cs) + 1, int((y1 - y0) / cs) + 1
    count = np.zeros((ny, nx), dtype=np.int32)
    gx = x0 + (np.arange(nx) + 0.5) * cs
    gy = y1 - (np.arange(ny) + 0.5) * cs
    for m in nadir:
        alt = m.get("rel_alt") or (plan["altitude"] if plan else None)
        if not alt or alt < 5:
            continue
        across, along = cam.footprint_m(alt)
        yaw = math.radians(m.get("gimbal_yaw", m.get("flight_yaw", 0.0)))
        fx, fy = math.sin(yaw), math.cos(yaw)  # camera forward (along) unit vector
        rx, ry = fy, -fx  # right (across)
        r = math.hypot(across, along) / 2
        c0 = max(0, int((m["x"] - r - x0) / cs))
        c1 = min(nx, int((m["x"] + r - x0) / cs) + 1)
        r0 = max(0, int((y1 - (m["y"] + r)) / cs))
        r1 = min(ny, int((y1 - (m["y"] - r)) / cs) + 1)
        if c0 >= c1 or r0 >= r1:
            continue
        X, Y = np.meshgrid(gx[c0:c1] - m["x"], gy[r0:r1] - m["y"])
        a_ = np.abs(X * fx + Y * fy) <= along / 2
        b_ = np.abs(X * rx + Y * ry) <= across / 2
        count[r0:r1, c0:c1] += (a_ & b_)

    in_plot = None
    stats = {}
    if poly_xy:
        from matplotlib.path import Path

        PX, PY = np.meshgrid(gx, gy)
        in_plot = Path(poly_xy).contains_points(np.stack([PX.ravel(), PY.ravel()], 1)).reshape(PX.shape)
        tot = in_plot.sum() * cs * cs
        for k in (1, 3, 5, 9):
            stats[k] = float(((count >= k) & in_plot).sum() * cs * cs / max(tot, 1))

    # ---- gaps (missed photos): along a line, or across a line change
    gaps = []
    plan_line = max((g["line_spacing"] for g in plan["grids"]), default=None) if plan else None

    def same_dir(h1, h2):
        return abs((h1 - h2 + 180) % 360 - 180) < 30

    for fl in flights:
        seq = [m for m in fl if m.get("gimbal_pitch", -90) <= -80] or fl
        if len(seq) < 5:
            continue
        steps = []
        for a, b in zip(seq, seq[1:]):
            steps.append((a, b, math.hypot(b["x"] - a["x"], b["y"] - a["y"]),
                          math.degrees(math.atan2(b["x"] - a["x"], b["y"] - a["y"]))))
        med = sorted(s[2] for s in steps)[len(steps) // 2]
        turns = []
        for k, (a, b, dd, hd) in enumerate(steps):
            straight = (k > 0 and same_dir(hd, steps[k - 1][3])) or \
                       (k + 1 < len(steps) and same_dir(hd, steps[k + 1][3]))
            if straight:
                if dd > max(2.2 * med, med + 8):
                    gaps.append((a, b, dd, med))
            else:
                turns.append((a, b, dd))
        if turns:
            base = plan_line or sorted(t[2] for t in turns)[len(turns) // 2]
            for a, b, dd in turns:
                if dd > base + 2.5 * med:  # line change that also skipped photos
                    gaps.append((a, b, dd, med))

    # ---- outputs: map + KML
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 9), dpi=140)
    ext = (0, x1 - x0, 0, y1 - y0)
    cnt = np.where(count > 0, count, np.nan)
    im = ax.imshow(cnt, extent=ext, origin="upper", cmap="viridis", vmin=0, vmax=max(12, np.nanmax(cnt) if
                   np.isfinite(cnt).any() else 12))
    cb = plt.colorbar(im, ax=ax, fraction=0.035)
    cb.set_label("nadir photos seeing each spot (aim for 9+, minimum 5)")
    if in_plot is not None:
        low = np.where(in_plot & (count < 5), 1.0, np.nan)
        ax.imshow(low, extent=ext, origin="upper", cmap="autumn", alpha=0.8, vmin=0, vmax=1)
        px = [p[0] - x0 for p in poly_xy] + [poly_xy[0][0] - x0]
        py = [p[1] - y0 for p in poly_xy] + [poly_xy[0][1] - y0]
        ax.plot(px, py, color="yellow", lw=2, label="plot boundary")
    for k, fl in enumerate(flights):
        ax.scatter([m["x"] - x0 for m in fl], [m["y"] - y0 for m in fl], s=6, label=f"flight {k + 1} ({len(fl)})")
    for a, b, dd, med in gaps:
        ax.plot([a["x"] - x0, b["x"] - x0], [a["y"] - y0, b["y"] - y0], color="red", lw=3)
    ax.set_aspect("equal")
    ax.set_xlabel("metres east")
    ax.set_ylabel("metres north")
    ax.set_title("Photo coverage check (red patches inside the plot = re-fly)", loc="left")
    ax.legend(loc="lower right", fontsize=7, markerscale=2)
    fig.tight_layout()
    png = os.path.join(out, "photo_coverage.png")
    fig.savefig(png)
    plt.close(fig)

    doc = KmlDoc("Photo positions")
    for i in range(len(PALETTE)):
        doc.style_point(f"f{i}", PALETTE[i], 0.5)
    doc.style_line("gap", "ff0000ff", 4)
    for k, fl in enumerate(flights):
        doc.folder(f"Flight {k + 1}: {len(fl)} photos")
        for m in fl:
            t = m["time"].strftime("%H:%M:%S") if m.get("time") else ""
            desc = f"{t} alt {m.get('rel_alt', '?')} m, gimbal {m.get('gimbal_pitch', '?')}°"
            doc.point(m["name"], m["lat"], m["lon"], f"f{k % len(PALETTE)}", desc)
        doc.end_folder()
    if gaps:
        doc.folder("Gaps (missed photos)")
        for a, b, dd, med in gaps:
            doc.line(f"gap {dd:.0f} m", [(a["lat"], a["lon"]), (b["lat"], b["lon"])], "gap")
        doc.end_folder()
    kml = doc.save(os.path.join(out, "photo_positions.kml"))

    # ---- summary
    info("")
    info(f"Photos: {len(files)} total, {len(geo)} with GPS, {len(nadir)} nadir, {len(oblique)} oblique, "
         f"{len(flights)} flight(s)")
    for k, fl in enumerate(flights, 1):
        t0 = fl[0].get("time")
        t1 = fl[-1].get("time")
        span = f"{t0:%H:%M}-{t1:%H:%M}" if t0 and t1 else ""
        info(f"  flight {k}: {len(fl)} photos {span}")
    if rel:
        info(f"Height above take-off: {min(rel):.0f}-{max(rel):.0f} m (median {sorted(rel)[len(rel) // 2]:.0f} m)")
    no_gps = len(files) - len(geo)
    if no_gps:
        warn(f"{no_gps} photos have no GPS - they will be harder to place.")
    if stats:
        tot = fmt_area(in_plot.sum() * cs * cs)
        info(f"Plot ({tot}) seen by >=5 nadir photos: {stats[5] * 100:.0f}%, >=9: {stats[9] * 100:.0f}%")
        if stats[5] < 0.97:
            warn("Parts of the plot have fewer than 5 nadir photos (orange on the map). Re-fly those lines "
                 "before leaving the site.")
        else:
            ok("Nadir coverage of the plot looks complete.")
    if plan:
        exp = sum(g["photos"] for g in plan["grids"])
        info(f"Plan expected ~{exp} grid photos; found {len(geo)} photos in total.")
    if gaps:
        warn(f"{len(gaps)} gap(s) along flight lines (red on the map), e.g. {gaps[0][2]:.0f} m between "
             f"{gaps[0][0]['name']} and {gaps[0][1]['name']} (normal spacing ~{gaps[0][3]:.0f} m).")
    elif len(geo) > 10:
        ok("No gaps along flight lines.")
    info(f"Coverage map: {png}")
    info(f"Google Earth: {kml}")
    return 0
