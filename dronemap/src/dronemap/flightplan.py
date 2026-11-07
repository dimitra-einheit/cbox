"""Mapping-flight planner.

Generates lawn-mower grids (nadir + oblique cross-hatch) and orbits around
points of interest, splits them into battery-sized missions, and estimates
flight time. Output is written by litchi.py (CSV) and plan_outputs.py (KML,
summary).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .camera import Camera
from .geo import (
    LocalProjection,
    best_line_heading,
    clip_to_band,
    heading_to_vector,
    polygon_area,
    polygon_centroid,
    polygon_perimeter,
    rotate,
)

# Litchi waypoint action codes
ACT_NONE = -1
ACT_STAY = 0  # param: milliseconds
ACT_PHOTO = 1
ACT_ROTATE = 4
ACT_TILT = 5

# Timing model used for estimates (deliberately conservative).
CLIMB_MS = 4.0  # m/s ascent
DESCENT_MS = 3.0  # m/s descent
RTH_MS = 10.0  # m/s return-to-home cruise
WAYPOINT_STOP_S = 4.0  # brake + turn + accelerate at each waypoint ("straight lines" path mode)
PHOTO_ACTION_S = 1.5  # time for a 'take photo' action
FIXED_OVERHEAD_S = 45.0  # take-off, mission start, landing

PRESETS = {
    "survey": {
        "description": "Terrain survey (earthworks, contours, volumes). Nadir grid plus a perpendicular, "
                       "slightly tilted grid: the crossing, tilted views stop the 3D model from bending "
                       "('doming'), which matters for pond levels and drainage.",
        "altitude": 80.0,
        "front": 0.75,
        "side": 0.70,
        "speed": 4.0,
        "grids": [("A", 0.0, -90.0, "nadir"), ("B", 90.0, -70.0, "cross-hatch, camera tilted 20°")],
    },
    "detail": {
        "description": "High-detail 3D model / Gaussian splat. Nadir grid plus two oblique grids so walls, "
                       "verandas and trees are seen from all four sides. Add --orbit for key buildings.",
        "altitude": 45.0,
        "front": 0.80,
        "side": 0.75,
        "speed": 3.0,
        "grids": [("A", 0.0, -90.0, "nadir"), ("B", 90.0, -60.0, "oblique N/S facades"),
                  ("C", 0.0, -60.0, "oblique E/W facades")],
    },
    "quick": {
        "description": "Fast single-grid overview (scouting, rough orthophoto). Not for volumes.",
        "altitude": 100.0,
        "front": 0.70,
        "side": 0.65,
        "speed": 6.0,
        "grids": [("A", 0.0, -90.0, "nadir")],
    },
}


@dataclass
class Waypoint:
    x: float
    y: float
    alt: float
    heading: float
    pitch: float
    speed: float
    dist_interval: float = -1.0
    actions: list = field(default_factory=list)  # list of (action_code, param)
    lat: float = 0.0
    lon: float = 0.0


@dataclass
class Mission:
    name: str
    kind: str  # 'grid' or 'orbit'
    label: str
    waypoints: list
    est_time_s: float = 0.0
    est_photos: int = 0
    lines: int = 0
    distance_m: float = 0.0


@dataclass
class GridInfo:
    name: str
    label: str
    heading: float
    pitch: float
    line_spacing: float
    photo_spacing: float
    n_lines: int
    photos: int
    missions: list


@dataclass
class OrbitInfo:
    name: str
    lat: float
    lon: float
    radius: float
    rings: list  # list of (alt, pitch, photos)
    missions: list


@dataclass
class Plan:
    name: str
    preset: str
    camera: Camera
    polygon: list  # [(lat, lon)]
    proj: LocalProjection
    home: tuple  # (lat, lon)
    home_xy: tuple
    altitude: float
    altitude_mode: str  # 'relative' or 'agl'
    front: float
    side: float
    speed: float
    base_heading: float
    buffer_side: float
    buffer_along: float
    battery_s: float
    grids: list = field(default_factory=list)
    orbits: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def area_m2(self) -> float:
        return polygon_area([self.proj.to_xy(*p) for p in self.polygon])

    @property
    def perimeter_m(self) -> float:
        return polygon_perimeter([self.proj.to_xy(*p) for p in self.polygon])

    @property
    def missions(self) -> list:
        out = []
        for g in self.grids:
            out.extend(g.missions)
        for o in self.orbits:
            out.extend(o.missions)
        return out


# ------------------------------------------------------------------ timing


def _action_time(wp: Waypoint) -> float:
    t = 0.0
    for code, param in wp.actions:
        if code == ACT_STAY:
            t += param / 1000.0
        elif code == ACT_PHOTO:
            t += PHOTO_ACTION_S
        elif code in (ACT_ROTATE, ACT_TILT):
            t += 2.0
    return t


def estimate_mission(home_xy, wps: list) -> tuple[float, float]:
    """(seconds, metres flown) for a mission flown from and back to home_xy."""
    first, last = wps[0], wps[-1]
    d_in = math.hypot(first.x - home_xy[0], first.y - home_xy[1])
    d_out = math.hypot(last.x - home_xy[0], last.y - home_xy[1])
    t = FIXED_OVERHEAD_S + first.alt / CLIMB_MS + d_in / max(first.speed, 1.0) + _action_time(first)
    dist = d_in + d_out
    for a, b in zip(wps, wps[1:]):
        d = math.hypot(b.x - a.x, b.y - a.y)
        dz = abs(b.alt - a.alt)
        seg = math.hypot(d, dz)
        dist += seg
        t += seg / max(a.speed, 0.5) + WAYPOINT_STOP_S + _action_time(b)
    t += d_out / RTH_MS + last.alt / DESCENT_MS
    return t, dist


# ------------------------------------------------------------------ grids


def _balanced_split(items: list, cost, budget: float, max_lines: int) -> list:
    """Split `items` (kept in order) into the fewest contiguous groups whose cost fits the
    budget, then rebalance so the groups take similar time (no 1-line leftover missions)."""
    n = len(items)
    if n == 0:
        return []
    # fewest groups (greedy is optimal for contiguous splitting with monotone costs)
    k, i = 0, 0
    while i < n:
        j = i + 1
        while j < n and (j + 1 - i) <= max_lines and cost(items[i:j + 1]) <= budget:
            j += 1
        k += 1
        i = j
    if k == 1:
        return [items]
    # minimise the longest group with exactly k contiguous groups (small n -> simple DP)
    T = [[0.0] * (n + 1) for _ in range(n + 1)]
    for a in range(n):
        for b in range(a + 1, min(n, a + max_lines) + 1):
            T[a][b] = cost(items[a:b])
    INF = float("inf")
    best = [[INF] * (n + 1) for _ in range(k + 1)]
    cut = [[0] * (n + 1) for _ in range(k + 1)]
    best[0][0] = 0.0
    for g in range(1, k + 1):
        for b in range(1, n + 1):
            for a in range(max(0, b - max_lines), b):
                if best[g - 1][a] == INF:
                    continue
                v = max(best[g - 1][a], T[a][b])
                if v < best[g][b]:
                    best[g][b], cut[g][b] = v, a
    groups, b = [], n
    for g in range(k, 0, -1):
        a = cut[g][b]
        groups.append(items[a:b])
        b = a
    return list(reversed(groups))


def _grid_lines(poly_xy, heading, line_spacing, strip_width, buffer_side, buffer_along):
    """Return list of ((x1,y1),(x2,y2)) flight lines, all pointing along `heading`."""
    ex, ey = heading_to_vector(heading)
    phi = math.atan2(ey, ex)
    rp = rotate(poly_xy, -phi)
    ys = [p[1] for p in rp]
    ymin, ymax = min(ys), max(ys)
    span = (ymax - ymin) + 2.0 * buffer_side
    n = max(2, int(math.ceil(span / line_spacing - 1e-9)) + 1)
    y0 = (ymin + ymax) / 2.0 - (n - 1) * line_spacing / 2.0
    lines = []
    for i in range(n):
        y = y0 + i * line_spacing
        band = clip_to_band(rp, y - strip_width / 2.0, y + strip_width / 2.0)
        if not band:
            continue
        xs = min(p[0] for p in band) - buffer_along
        xe = max(p[0] for p in band) + buffer_along
        a, b = rotate([(xs, y), (xe, y)], phi)
        lines.append((a, b))
    return lines


def _order_lines(lines, home_xy):
    """Serpentine order starting from the corner nearest home. Returns [(start, end), ...]."""
    best = None
    for rev_order in (False, True):
        seq = list(reversed(lines)) if rev_order else list(lines)
        for flip_first in (False, True):
            ordered = []
            for i, (a, b) in enumerate(seq):
                fwd = (i % 2 == 0) != flip_first
                ordered.append((a, b) if fwd else (b, a))
            s = ordered[0][0]
            d = math.hypot(s[0] - home_xy[0], s[1] - home_xy[1])
            if best is None or d < best[0]:
                best = (d, ordered)
    return best[1]


def plan_grid(plan: Plan, name: str, heading_offset: float, pitch: float, label: str,
              poly_xy) -> GridInfo:
    cam = plan.camera
    across, along = cam.footprint_m(plan.altitude)
    line_spacing = across * (1.0 - plan.side)
    photo_spacing = along * (1.0 - plan.front)
    heading = (plan.base_heading + heading_offset) % 360.0
    raw = _grid_lines(poly_xy, heading, line_spacing, across, plan.buffer_side, plan.buffer_along)
    ordered = _order_lines(raw, plan.home_xy)

    def line_wps(a, b):
        hd = math.degrees(math.atan2(b[0] - a[0], b[1] - a[1])) % 360.0
        w1 = Waypoint(a[0], a[1], plan.altitude, hd, pitch, plan.speed, dist_interval=photo_spacing)
        w2 = Waypoint(b[0], b[1], plan.altitude, hd, pitch, plan.speed, dist_interval=-1.0)
        return [w1, w2]

    def photos_on(a, b):
        return int(math.hypot(b[0] - a[0], b[1] - a[1]) // photo_spacing) + 1

    missions = _balanced_split(ordered, lambda segs: estimate_mission(
        plan.home_xy, [w for s in segs for w in line_wps(*s)])[0], plan.battery_s, max_lines=49)

    out_missions = []
    total_photos = 0
    for k, segs in enumerate(missions, 1):
        wps = [w for s in segs for w in line_wps(*s)]
        t, dist = estimate_mission(plan.home_xy, wps)
        ph = sum(photos_on(*s) for s in segs)
        total_photos += ph
        m = Mission(f"{name}{k}", "grid", f"Grid {name} ({label}) part {k}/{len(missions)}", wps,
                    est_time_s=t, est_photos=ph, lines=len(segs), distance_m=dist)
        if t > plan.battery_s:
            plan.warnings.append(
                f"Mission {m.name} needs about {t / 60:.1f} min, more than the {plan.battery_s / 60:.0f} min "
                "battery budget, because a single flight line is that long. Split the plot or raise --battery-minutes."
            )
        out_missions.append(m)
    return GridInfo(name, label, heading, pitch, line_spacing, photo_spacing, len(ordered), total_photos,
                    out_missions)


# ------------------------------------------------------------------ orbits

def orbit_pitch(alt: float, radius: float, target_height: float) -> float:
    return -math.degrees(math.atan2(max(alt - target_height, 0.5), radius))


def plan_orbit(plan: Plan, name: str, lat: float, lon: float, radius: float, alts: list,
               photos_per_ring: int, target_height: float) -> OrbitInfo:
    cx, cy = plan.proj.to_xy(lat, lon)
    hx, hy = plan.home_xy
    start = math.degrees(math.atan2(hx - cx, hy - cy)) % 360.0 if (hx, hy) != (cx, cy) else 0.0
    cam = plan.camera
    vfov_half = math.degrees(math.atan2(cam.sensor_h_mm / 2.0, cam.focal_mm))
    rings = []
    wps: list[Waypoint] = []
    for alt in alts:
        pitch = orbit_pitch(alt, radius, target_height)
        if pitch + vfov_half > -2.0:
            plan.warnings.append(
                f"Orbit {name} at {alt:.0f} m: camera at {pitch:.0f}° will show sky at the top of frames. "
                "Fly higher or closer, or process with --sky-removal."
            )
        rings.append((alt, pitch, photos_per_ring))
        for j in range(photos_per_ring):
            b = (start + j * 360.0 / photos_per_ring) % 360.0  # clockwise around the point
            ex, ey = heading_to_vector(b)
            wps.append(Waypoint(cx + radius * ex, cy + radius * ey, alt, (b + 180.0) % 360.0, pitch,
                                min(plan.speed, 3.0), actions=[(ACT_STAY, 1000), (ACT_PHOTO, 0)]))
    # split if more than 99 waypoints or too long
    missions = []
    chunk: list = []
    for wp in wps:
        trial = chunk + [wp]
        t, _ = estimate_mission(plan.home_xy, trial)
        if chunk and (len(trial) > 99 or t > plan.battery_s):
            missions.append(chunk)
            chunk = [wp]
        else:
            chunk = trial
    if chunk:
        missions.append(chunk)
    out = []
    for k, ch in enumerate(missions, 1):
        t, dist = estimate_mission(plan.home_xy, ch)
        suffix = f" part {k}/{len(missions)}" if len(missions) > 1 else ""
        out.append(Mission(f"{name}{'' if len(missions) == 1 else k}", "orbit",
                           f"Orbit {name} r={radius:.0f} m{suffix}", ch, est_time_s=t, est_photos=len(ch),
                           distance_m=dist))
    return OrbitInfo(name, lat, lon, radius, rings, out)


# ------------------------------------------------------------------ top level


def build_plan(name: str, polygon_latlon, camera: Camera, preset: str = "survey", altitude: float | None = None,
               front: float | None = None, side: float | None = None, speed: float | None = None,
               heading: float | None = None, buffer: float | None = None, battery_minutes: float = 12.0,
               home=None, altitude_mode: str = "relative", grids: str | None = None,
               oblique_pitch: float | None = None, orbits=None, orbit_alts=(20.0, 35.0),
               orbit_photos: int = 36, orbit_target_height: float = 2.0) -> Plan:
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset}")
    P = PRESETS[preset]
    altitude = float(altitude if altitude is not None else P["altitude"])
    front = float(front if front is not None else P["front"])
    side = float(side if side is not None else P["side"])
    speed = float(speed if speed is not None else P["speed"])
    for v, nm in ((front, "front overlap"), (side, "side overlap")):
        if not 0.3 <= v < 0.95:
            raise ValueError(f"{nm} must be between 0.30 and 0.95 (got {v})")

    lat0 = sum(p[0] for p in polygon_latlon) / len(polygon_latlon)
    lon0 = sum(p[1] for p in polygon_latlon) / len(polygon_latlon)
    proj = LocalProjection(lat0, lon0)
    poly_xy = [proj.to_xy(*p) for p in polygon_latlon]
    if home is None:
        cx, cy = polygon_centroid(poly_xy)
        home = proj.to_latlon(cx, cy)
    home_xy = proj.to_xy(*home)
    base_heading = best_line_heading(poly_xy) if heading is None else float(heading) % 360.0

    across, along = camera.footprint_m(altitude)
    line_spacing = across * (1 - side)
    photo_spacing = along * (1 - front)
    buffer_side = float(buffer) if buffer is not None else line_spacing
    buffer_along = float(buffer) if buffer is not None else along / 2.0

    plan = Plan(name=name, preset=preset, camera=camera, polygon=list(polygon_latlon), proj=proj, home=tuple(home),
                home_xy=home_xy, altitude=altitude, altitude_mode=altitude_mode, front=front, side=side, speed=speed,
                base_heading=base_heading, buffer_side=buffer_side, buffer_along=buffer_along,
                battery_s=battery_minutes * 60.0)

    # Photo interval check: the camera needs ~2.5 s between JPEGs.
    interval = photo_spacing / speed
    if interval < camera.min_photo_interval_s:
        new_speed = math.floor(photo_spacing / camera.min_photo_interval_s * 10) / 10.0
        plan.warnings.append(
            f"At {speed:.1f} m/s the camera would need a photo every {interval:.1f} s; the Mavic 2 Pro needs "
            f"about {camera.min_photo_interval_s:.1f} s. Speed lowered to {new_speed:.1f} m/s."
        )
        plan.speed = speed = new_speed
    blur_px = speed * camera.rolling_shutter_ms / 1000.0 / camera.gsd_m(altitude)
    if blur_px > 25:
        plan.warnings.append(
            f"Rolling-shutter skew about {blur_px:.0f} px at {speed:.1f} m/s. ODM corrects this "
            "(dronemap process turns on --rolling-shutter), but slower is sharper."
        )
    if altitude > 120:
        plan.warnings.append("Altitude above 120 m AGL (400 ft): above the Indian green-zone limit.")
    if altitude_mode == "relative":
        plan.warnings.append(
            "Altitudes are relative to the take-off point. Take off from the highest part of the plot (or near it), "
            "or re-run with --agl so Litchi follows the ground."
        )

    grid_defs = P["grids"]
    if grids:
        wanted = [g.strip().upper() for g in grids.split(",") if g.strip()]
        grid_defs = [g for g in grid_defs if g[0] in wanted]
        if not grid_defs:
            raise ValueError(f"--grids {grids}: preset '{preset}' only has grids "
                             f"{', '.join(g[0] for g in P['grids'])}")
    for gname, off, pitch, label in grid_defs:
        if oblique_pitch is not None and pitch > -90:
            pitch = float(oblique_pitch)
        plan.grids.append(plan_grid(plan, gname, off, pitch, label, poly_xy))

    for i, spec in enumerate(orbits or [], 1):
        olat, olon, radius, tgt = spec
        plan.orbits.append(plan_orbit(plan, f"O{i}", olat, olon, radius, list(orbit_alts), orbit_photos,
                                      orbit_target_height if tgt is None else tgt))

    # Fill in lat/lon for every waypoint
    for m in plan.missions:
        for w in m.waypoints:
            w.lat, w.lon = proj.to_latlon(w.x, w.y)
    return plan
