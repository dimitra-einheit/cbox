"""Geodesy and geometry helpers.

- LocalProjection: metres east/north around a reference point (accurate to
  millimetres over a farm-sized area).
- Polygon maths: area, centroid, rotation, band clipping.
- Readers for plot boundaries drawn in Google Earth (KML/KMZ), GeoJSON, or a
  plain text list of "lat, lon" lines (e.g. copied from Google Maps).
"""
from __future__ import annotations

import json
import math
import os
import re
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

from .util import UserError

WGS84_A = 6378137.0
WGS84_F = 1 / 298.257223563
WGS84_E2 = WGS84_F * (2 - WGS84_F)


class LocalProjection:
    """Local east/north metres around (lat0, lon0) on the WGS84 ellipsoid."""

    def __init__(self, lat0: float, lon0: float):
        self.lat0 = lat0
        self.lon0 = lon0
        phi = math.radians(lat0)
        s = math.sin(phi)
        w = math.sqrt(1.0 - WGS84_E2 * s * s)
        self._m_lat = WGS84_A * (1.0 - WGS84_E2) / (w ** 3)  # metres per radian of latitude
        self._m_lon = WGS84_A / w * math.cos(phi)  # metres per radian of longitude

    def to_xy(self, lat: float, lon: float) -> tuple[float, float]:
        return (math.radians(lon - self.lon0) * self._m_lon, math.radians(lat - self.lat0) * self._m_lat)

    def to_latlon(self, x: float, y: float) -> tuple[float, float]:
        return (self.lat0 + math.degrees(y / self._m_lat), self.lon0 + math.degrees(x / self._m_lon))


# ---------------------------------------------------------------- polygon maths

Point = tuple  # (x, y)


def polygon_area(pts: list[Point]) -> float:
    a = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0


def polygon_perimeter(pts: list[Point]) -> float:
    n = len(pts)
    return sum(math.dist(pts[i], pts[(i + 1) % n]) for i in range(n))


def polygon_centroid(pts: list[Point]) -> Point:
    a = cx = cy = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        cr = x1 * y2 - x2 * y1
        a += cr
        cx += (x1 + x2) * cr
        cy += (y1 + y2) * cr
    if abs(a) < 1e-9:
        return (sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n)
    a *= 0.5
    return (cx / (6 * a), cy / (6 * a))


def rotate(pts: list[Point], angle_rad: float) -> list[Point]:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return [(x * c - y * s, x * s + y * c) for x, y in pts]


def _clip_half_plane(poly: list[Point], keep, intersect) -> list[Point]:
    out: list[Point] = []
    n = len(poly)
    for i in range(n):
        cur, nxt = poly[i], poly[(i + 1) % n]
        cin, nin = keep(cur), keep(nxt)
        if cin:
            out.append(cur)
        if cin != nin:
            out.append(intersect(cur, nxt))
    return out


def clip_to_band(poly: list[Point], ymin: float, ymax: float) -> list[Point]:
    """Clip a polygon to the horizontal band ymin <= y <= ymax (Sutherland-Hodgman)."""

    def ix(p, q, yv):
        t = (yv - p[1]) / (q[1] - p[1])
        return (p[0] + t * (q[0] - p[0]), yv)

    out = _clip_half_plane(poly, lambda p: p[1] >= ymin, lambda p, q: ix(p, q, ymin))
    if not out:
        return out
    return _clip_half_plane(out, lambda p: p[1] <= ymax, lambda p, q: ix(p, q, ymax))


def heading_to_vector(heading_deg: float) -> tuple[float, float]:
    """Compass heading (0 = north, 90 = east) to a unit (east, north) vector."""
    h = math.radians(heading_deg)
    return (math.sin(h), math.cos(h))


def bearing_deg(x1: float, y1: float, x2: float, y2: float) -> float:
    """Compass bearing from point 1 to point 2 in local metres."""
    return math.degrees(math.atan2(x2 - x1, y2 - y1)) % 360.0


def best_line_heading(pts: list[Point]) -> float:
    """Flight-line heading (0-180) that gives the fewest lines: lines run along the
    polygon's longest dimension, i.e. the direction minimising the perpendicular width."""
    best_h, best_w = 0.0, float("inf")
    for h10 in range(0, 1800, 5):  # 0.5 degree steps
        h = h10 / 10.0
        ex, ey = heading_to_vector(h)
        # perpendicular unit vector
        px, py = -ey, ex
        proj = [x * px + y * py for x, y in pts]
        w = max(proj) - min(proj)
        if w < best_w - 1e-6:
            best_w, best_h = w, h
    return best_h


# ---------------------------------------------------------------- boundary files


@dataclass
class Shape:
    name: str
    kind: str  # 'polygon', 'line' or 'point'
    coords: list = field(default_factory=list)  # list of (lat, lon)


def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[1] if "}" in tag else tag


def _parse_kml_coords(text: str) -> list[tuple[float, float]]:
    out = []
    for tok in text.replace("\n", " ").replace("\t", " ").split():
        parts = tok.split(",")
        if len(parts) >= 2:
            lon, lat = float(parts[0]), float(parts[1])
            out.append((lat, lon))
    return out


def _read_kml_text(text: str) -> list[Shape]:
    try:
        root = ET.fromstring(text)
    except ET.ParseError as e:
        raise UserError(f"Could not read KML: {e}") from None
    shapes: list[Shape] = []
    for pm in root.iter():
        if _strip_ns(pm.tag) != "Placemark":
            continue
        name = ""
        for ch in pm:
            if _strip_ns(ch.tag) == "name" and ch.text:
                name = ch.text.strip()
        for el in pm.iter():
            tag = _strip_ns(el.tag)
            if tag == "Polygon":
                for sub in el.iter():
                    if _strip_ns(sub.tag) == "outerBoundaryIs":
                        for c in sub.iter():
                            if _strip_ns(c.tag) == "coordinates" and c.text:
                                shapes.append(Shape(name, "polygon", _parse_kml_coords(c.text)))
                                break
                        break
            elif tag == "LineString":
                for c in el:
                    if _strip_ns(c.tag) == "coordinates" and c.text:
                        shapes.append(Shape(name, "line", _parse_kml_coords(c.text)))
            elif tag == "Point":
                for c in el:
                    if _strip_ns(c.tag) == "coordinates" and c.text:
                        shapes.append(Shape(name, "point", _parse_kml_coords(c.text)[:1]))
    return shapes


def _read_geojson(data) -> list[Shape]:
    shapes: list[Shape] = []

    def handle(geom, name):
        if not geom:
            return
        t = geom.get("type")
        c = geom.get("coordinates")
        if t == "Polygon":
            shapes.append(Shape(name, "polygon", [(p[1], p[0]) for p in c[0]]))
        elif t == "MultiPolygon":
            for poly in c:
                shapes.append(Shape(name, "polygon", [(p[1], p[0]) for p in poly[0]]))
        elif t == "LineString":
            shapes.append(Shape(name, "line", [(p[1], p[0]) for p in c]))
        elif t == "Point":
            shapes.append(Shape(name, "point", [(c[1], c[0])]))
        elif t == "GeometryCollection":
            for g in geom.get("geometries", []):
                handle(g, name)

    if data.get("type") == "FeatureCollection":
        for f in data.get("features", []):
            props = f.get("properties") or {}
            handle(f.get("geometry"), str(props.get("name", "")))
    elif data.get("type") == "Feature":
        props = data.get("properties") or {}
        handle(data.get("geometry"), str(props.get("name", "")))
    else:
        handle(data, "")
    return shapes


_NUM = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _read_text_points(text: str) -> list[Shape]:
    pts = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        nums = _NUM.findall(line)
        if len(nums) < 2:
            continue
        a, b = float(nums[0]), float(nums[1])
        pts.append((a, b))
    if not pts:
        return []
    # Lines are "lat, lon". If the first number can't be a latitude, the file is lon, lat.
    if any(abs(a) > 90 for a, _ in pts):
        pts = [(b, a) for a, b in pts]
    if len(pts) >= 3:
        return [Shape("", "polygon", pts)]
    return [Shape(f"P{i + 1}", "point", [p]) for i, p in enumerate(pts)]


def read_shapes(path: str) -> list[Shape]:
    """Read all shapes from a KML/KMZ/GeoJSON/text file (coordinates as (lat, lon))."""
    if not os.path.isfile(path):
        raise UserError(f"File not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".kmz":
        with zipfile.ZipFile(path) as z:
            kmls = [n for n in z.namelist() if n.lower().endswith(".kml")]
            if not kmls:
                raise UserError(f"No KML inside {path}")
            kmls.sort(key=lambda n: (n.lower() != "doc.kml", n))
            shapes = _read_kml_text(z.read(kmls[0]).decode("utf-8", "replace"))
    elif ext == ".kml":
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            shapes = _read_kml_text(f.read())
    elif ext in (".geojson", ".json"):
        with open(path, "r", encoding="utf-8") as f:
            shapes = _read_geojson(json.load(f))
    else:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            shapes = _read_text_points(f.read())
    for s in shapes:
        for lat, lon in s.coords:
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                raise UserError(f"Coordinate out of range in {path}: {lat}, {lon}")
    return shapes


def _clean_ring(coords: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out = []
    for c in coords:
        if not out or (abs(c[0] - out[-1][0]) > 1e-10 or abs(c[1] - out[-1][1]) > 1e-10):
            out.append(c)
    if len(out) > 1 and abs(out[0][0] - out[-1][0]) < 1e-10 and abs(out[0][1] - out[-1][1]) < 1e-10:
        out.pop()
    return out


def read_polygon(path: str, name: str | None = None) -> tuple[str, list[tuple[float, float]]]:
    """Return (name, [(lat, lon), ...]) of the polygon in a boundary file.

    Accepts a drawn polygon, a closed path, or (as a last resort) a set of
    point placemarks taken in order (e.g. corners walked with a phone).
    """
    shapes = read_shapes(path)
    polys = [s for s in shapes if s.kind == "polygon"]
    if name:
        polys = [s for s in polys if s.name.lower() == name.lower()]
        if not polys:
            raise UserError(f"No polygon called '{name}' in {path}")
    if not polys:
        lines = [s for s in shapes if s.kind == "line" and len(s.coords) >= 3]
        if lines:
            polys = lines
    if not polys:
        pts = [s.coords[0] for s in shapes if s.kind == "point"]
        if len(pts) >= 3:
            polys = [Shape("points", "polygon", pts)]
    if not polys:
        raise UserError(
            f"No polygon found in {path}. Draw the plot boundary as a polygon in Google Earth "
            "(Add > Polygon) and save it as KML, or list 'lat, lon' corner points one per line."
        )
    s = polys[0]
    ring = _clean_ring(s.coords)
    if len(ring) < 3:
        raise UserError(f"Polygon in {path} has fewer than 3 distinct corners")
    return s.name, ring


def read_points(path: str) -> list[tuple[str, float, float]]:
    """Point placemarks as (name, lat, lon). A plain text file gives one point per line."""
    shapes = read_shapes(path)
    pts = [(s.name, s.coords[0][0], s.coords[0][1]) for s in shapes if s.kind == "point"]
    if not pts and os.path.splitext(path)[1].lower() not in (".kml", ".kmz", ".geojson", ".json"):
        for s in shapes:
            pts.extend((f"P{i + 1}", lat, lon) for i, (lat, lon) in enumerate(s.coords))
    return pts


def parse_latlon(text: str) -> tuple[float, float]:
    nums = _NUM.findall(text)
    if len(nums) < 2:
        raise UserError(f"Expected 'lat,lon' but got '{text}'")
    lat, lon = float(nums[0]), float(nums[1])
    if abs(lat) > 90:
        lat, lon = lon, lat
    return lat, lon
