"""Raster helpers: load DTM/orthophoto onto a common metric grid, hillshade, slope,
polygon masks, contours, and Google Earth (KMZ) / GeoTIFF / DXF writers."""
from __future__ import annotations

import io
import math
import os
import zipfile
from dataclasses import dataclass

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import Affine, from_origin
from rasterio.warp import calculate_default_transform, reproject, transform as warp_transform, transform_bounds

from .kml import KmlDoc
from .util import UserError

WGS84 = CRS.from_epsg(4326)


@dataclass
class Grid:
    z: np.ndarray  # float64 elevations, NaN = no data
    transform: Affine  # north-up, square cells
    crs: CRS
    res: float

    @property
    def shape(self):
        return self.z.shape

    @property
    def bounds(self):
        h, w = self.z.shape
        left, top = self.transform.c, self.transform.f
        return (left, top - h * self.res, left + w * self.res, top)

    @property
    def cell_area(self) -> float:
        return self.res * self.res

    def xy_of(self, rows, cols):
        x = self.transform.c + (np.asarray(cols) + 0.5) * self.res
        y = self.transform.f - (np.asarray(rows) + 0.5) * self.res
        return x, y

    def rc_of(self, x, y):
        c = np.floor((np.asarray(x) - self.transform.c) / self.res).astype(int)
        r = np.floor((self.transform.f - np.asarray(y)) / self.res).astype(int)
        return r, c

    def to_lonlat(self, x, y):
        lon, lat = warp_transform(self.crs, WGS84, list(np.atleast_1d(x)), list(np.atleast_1d(y)))
        return np.array(lon), np.array(lat)

    def from_lonlat(self, lon, lat):
        x, y = warp_transform(WGS84, self.crs, list(np.atleast_1d(lon)), list(np.atleast_1d(lat)))
        return np.array(x), np.array(y)


# ------------------------------------------------------------------ project files


def resolve_dtm(source: str) -> str:
    if os.path.isfile(source):
        return source
    for rel in ("odm_dem/dtm.tif", "odm_dem/dsm.tif"):
        p = os.path.join(source, rel)
        if os.path.isfile(p):
            if rel.endswith("dsm.tif"):
                print("NOTE: no DTM (ground model) found - using the DSM, which includes trees and buildings. "
                      "Process with '--preset survey' to get a DTM.")
            return p
    raise UserError(f"No terrain model in {source}. Expected odm_dem/dtm.tif - run "
                    f"'dronemap process {source} --preset survey' first, or pass a DTM .tif directly.")


def resolve_ortho(source: str) -> str | None:
    base = source if os.path.isdir(source) else os.path.dirname(os.path.dirname(os.path.abspath(source)))
    p = os.path.join(base, "odm_orthophoto", "odm_orthophoto.tif")
    return p if os.path.isfile(p) else None


def default_out(source: str, sub: str) -> str:
    if os.path.isdir(source):
        return os.path.join(source, "dronemap", sub)
    return os.path.join(os.path.dirname(os.path.abspath(source)), "dronemap", sub)


def utm_crs_for(lon: float, lat: float) -> CRS:
    zone = int((lon + 180.0) // 6.0) + 1
    return CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)


# ------------------------------------------------------------------ loading


def dataset_info(path: str) -> dict:
    with rasterio.open(path) as src:
        if src.crs is None:
            raise UserError(f"{path} has no coordinate system (not a GeoTIFF from ODM?)")
        crs = src.crs
        if crs.is_geographic:
            lon = (src.bounds.left + src.bounds.right) / 2
            lat = (src.bounds.top + src.bounds.bottom) / 2
            crs = utm_crs_for(lon, lat)
        b = transform_bounds(src.crs, crs, *src.bounds)
        native = abs(src.transform.a) if not src.crs.is_geographic else abs(src.transform.a) * 111000
        return {"crs": crs, "bounds": b, "native_res": native, "nodata": src.nodata}


def load_grid(path: str, res: float, bounds=None, crs: CRS | None = None) -> Grid:
    """Resample band 1 onto a north-up grid with square `res`-metre cells."""
    inf = dataset_info(path)
    crs = crs or inf["crs"]
    if bounds is None:
        bounds = inf["bounds"]
    left, bottom, right, top = bounds
    left = math.floor(left / res) * res
    top = math.ceil(top / res) * res
    w = max(1, int(math.ceil((right - left) / res)))
    h = max(1, int(math.ceil((top - bottom) / res)))
    if w * h > 60_000_000:
        raise UserError(f"Grid of {w}x{h} cells is too large; use a coarser --res than {res} m.")
    tr = from_origin(left, top, res, res)
    z = np.full((h, w), np.nan, dtype=np.float64)
    resampling = Resampling.average if res >= inf["native_res"] * 0.99 else Resampling.bilinear
    with rasterio.open(path) as src:
        reproject(source=rasterio.band(src, 1), destination=z, src_transform=src.transform, src_crs=src.crs,
                  src_nodata=src.nodata, dst_transform=tr, dst_crs=crs, dst_nodata=np.nan, resampling=resampling)
    z[z < -1000] = np.nan  # stray nodata values
    if not np.isfinite(z).any():
        raise UserError(f"No elevation data inside the requested area in {path}")
    return Grid(z, tr, crs, res)


def read_rgb(path: str, grid_like: Grid, max_px: int = 4096) -> tuple[np.ndarray, Affine] | tuple[None, None]:
    """Orthophoto resampled over the grid's extent, at up to max_px on the long side.
    Returns (rgba uint8 HxWx4, transform)."""
    if not path or not os.path.isfile(path):
        return None, None
    left, bottom, right, top = grid_like.bounds
    span = max(right - left, top - bottom)
    res = max(span / max_px, 0.01)
    w = int(math.ceil((right - left) / res))
    h = int(math.ceil((top - bottom) / res))
    tr = from_origin(left, top, res, res)
    out = np.zeros((4, h, w), dtype=np.uint8)
    with rasterio.open(path) as src:
        nb = src.count
        for i in range(min(nb, 4)):
            reproject(source=rasterio.band(src, i + 1), destination=out[i], src_transform=src.transform,
                      src_crs=src.crs, dst_transform=tr, dst_crs=grid_like.crs, resampling=Resampling.average)
        if nb < 4:
            out[3] = np.where(out[:3].sum(axis=0) > 0, 255, 0)
    return np.moveaxis(out, 0, -1), tr


# ------------------------------------------------------------------ analysis helpers


def fill_nan_nearest(z: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    mask = ~np.isfinite(z)
    if not mask.any():
        return z.copy()
    idx = ndimage.distance_transform_edt(mask, return_distances=False, return_indices=True)
    return z[tuple(idx)]


def hillshade(z: np.ndarray, res: float, azimuth: float = 315.0, altitude: float = 45.0) -> np.ndarray:
    from matplotlib.colors import LightSource

    zf = fill_nan_nearest(z)
    hs = LightSource(azdeg=azimuth, altdeg=altitude).hillshade(zf, vert_exag=1.0, dx=res, dy=res)
    hs[~np.isfinite(z)] = np.nan
    return hs


def slope_percent(z: np.ndarray, res: float) -> np.ndarray:
    zf = fill_nan_nearest(z)
    gy, gx = np.gradient(zf, res)
    s = np.hypot(gx, gy) * 100.0
    s[~np.isfinite(z)] = np.nan
    return s


def polygon_mask(grid: Grid, ring_latlon) -> np.ndarray:
    lon = [p[1] for p in ring_latlon]
    lat = [p[0] for p in ring_latlon]
    xs, ys = warp_transform(WGS84, grid.crs, lon, lat)
    coords = list(zip(xs, ys))
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    geom = {"type": "Polygon", "coordinates": [coords]}
    return rasterize([(geom, 1)], out_shape=grid.shape, transform=grid.transform, fill=0,
                     all_touched=False, dtype="uint8").astype(bool)


def ring_to_crs(grid: Grid, ring_latlon) -> list:
    lon = [p[1] for p in ring_latlon]
    lat = [p[0] for p in ring_latlon]
    xs, ys = warp_transform(WGS84, grid.crs, lon, lat)
    return list(zip(xs, ys))


def bounds_for_ring(crs: CRS, ring_latlon, margin: float):
    xs, ys = warp_transform(WGS84, crs, [p[1] for p in ring_latlon], [p[0] for p in ring_latlon])
    return (min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin)


def contour_lines(grid: Grid, z: np.ndarray, interval: float, smooth_cells: float = 1.0):
    """[(level, [array Nx2 of map x,y]), ...]"""
    import contourpy
    from scipy import ndimage

    valid = np.isfinite(z)
    zf = fill_nan_nearest(z)
    if smooth_cells > 0:
        zf = ndimage.gaussian_filter(zf, smooth_cells)
    zm = np.ma.array(zf, mask=~valid)
    h, w = z.shape
    xs = grid.transform.c + (np.arange(w) + 0.5) * grid.res
    ys = grid.transform.f - (np.arange(h) + 0.5) * grid.res
    gen = contourpy.contour_generator(x=xs, y=ys, z=zm, line_type="Separate")
    lo = math.ceil(np.nanmin(z) / interval) * interval
    hi = math.floor(np.nanmax(z) / interval) * interval
    out = []
    lvl = lo
    while lvl <= hi + 1e-9:
        lines = [simplify(l, grid.res * 0.5) for l in gen.lines(lvl) if len(l) >= 2]
        lines = [l for l in lines if len(l) >= 2]
        if lines:
            out.append((round(lvl, 3), lines))
        lvl += interval
    return out


def simplify(pts: np.ndarray, tol: float) -> np.ndarray:
    """Douglas-Peucker line simplification (iterative)."""
    n = len(pts)
    if n < 3:
        return pts
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        p, q = pts[a], pts[b]
        seg = pts[a + 1:b]
        d = q - p
        L = math.hypot(d[0], d[1])
        if L == 0:
            dist = np.hypot(seg[:, 0] - p[0], seg[:, 1] - p[1])
        else:
            dist = np.abs(d[0] * (seg[:, 1] - p[1]) - d[1] * (seg[:, 0] - p[0])) / L
        i = int(np.argmax(dist))
        if dist[i] > tol:
            m = a + 1 + i
            keep[m] = True
            stack.append((a, m))
            stack.append((m, b))
    return pts[keep]


# ------------------------------------------------------------------ writers


def write_geotiff(path: str, arr: np.ndarray, grid: Grid, nodata: float = -9999.0, dtype="float32") -> str:
    data = np.where(np.isfinite(arr), arr, nodata).astype(dtype)
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype=dtype, crs=grid.crs, transform=grid.transform, nodata=nodata,
                       compress="deflate", tiled=True) as dst:
        dst.write(data, 1)
    return path


def _png_bytes(rgba: np.ndarray) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def to_wgs84_rgba(rgba: np.ndarray, transform: Affine, crs: CRS):
    """Reproject an RGBA image to lat/lon. Returns (rgba, north, south, east, west)."""
    h, w = rgba.shape[:2]
    left, top = transform.c, transform.f
    right, bottom = left + w * transform.a, top + h * transform.e
    dst_tr, dw, dh = calculate_default_transform(crs, WGS84, w, h, left, bottom, right, top)
    out = np.zeros((4, dh, dw), dtype=np.uint8)
    for i in range(4):
        reproject(source=np.ascontiguousarray(rgba[:, :, i]), destination=out[i], src_transform=transform,
                  src_crs=crs, dst_transform=dst_tr, dst_crs=WGS84, resampling=Resampling.nearest)
    west, north = dst_tr.c, dst_tr.f
    east, south = west + dw * dst_tr.a, north + dh * dst_tr.e
    return np.moveaxis(out, 0, -1), north, south, east, west


class Kmz:
    """Google Earth package: raster overlays + vector placemarks."""

    def __init__(self, name: str):
        self.doc = KmlDoc(name)
        self.files: dict[str, bytes] = {}

    def overlay(self, name: str, rgba: np.ndarray, transform: Affine, crs: CRS, visible: bool = True) -> None:
        img, n, s, e, w = to_wgs84_rgba(rgba, transform, crs)
        fn = f"files/overlay_{len(self.files) + 1}.png"
        self.files[fn] = _png_bytes(img)
        self.doc.ground_overlay(name, fn, n, s, e, w, visible=visible)

    def save(self, path: str) -> str:
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("doc.kml", self.doc.to_string())
            for fn, data in self.files.items():
                z.writestr(fn, data)
        return path


def colorize(values: np.ndarray, cmap: str, vmin: float, vmax: float, alpha: float = 1.0,
             shade: np.ndarray | None = None) -> np.ndarray:
    """Map a float array to RGBA uint8 (NaN -> transparent), optionally multiplied by a hillshade."""
    import matplotlib

    cm = matplotlib.colormaps[cmap]
    norm = np.clip((values - vmin) / max(vmax - vmin, 1e-9), 0, 1)
    rgba = cm(np.nan_to_num(norm))
    if shade is not None:
        s = np.nan_to_num(shade, nan=1.0)
        rgba[..., :3] *= (0.55 + 0.45 * s)[..., None]
    rgba[..., 3] = np.where(np.isfinite(values), alpha, 0.0)
    return (rgba * 255).astype(np.uint8)


def write_dxf_contours(path: str, contours) -> str:
    """AutoCAD R12 DXF with one 3D polyline per contour (map coordinates, Z = elevation)."""
    out = ["0", "SECTION", "2", "ENTITIES"]
    for level, lines in contours:
        major = abs(level / 1.0 - round(level / 1.0)) < 1e-6
        layer = "CONTOUR_MAJOR" if major else "CONTOUR_MINOR"
        for line in lines:
            out += ["0", "POLYLINE", "8", layer, "66", "1", "70", "8", "10", "0.0", "20", "0.0", "30", f"{level:.3f}"]
            for x, y in line:
                out += ["0", "VERTEX", "8", layer, "10", f"{x:.3f}", "20", f"{y:.3f}", "30", f"{level:.3f}",
                        "70", "32"]
            out += ["0", "SEQEND", "8", layer]
    out += ["0", "ENDSEC", "0", "EOF"]
    with open(path, "w", encoding="ascii") as f:
        f.write("\n".join(out) + "\n")
    return path
