"""Hand-sculpt earthworks in Blender and measure them.

to-blender  : terrain (DTM) -> textured OBJ in local metres + terrain_georef.json
from-blender: your edited OBJ -> cut/fill volumes, maps and a designed-terrain GeoTIFF
"""
from __future__ import annotations

import datetime as _dt
import json
import os

import numpy as np

from .util import UserError, ensure_dir, fmt_area, info, ok, step, warn

BLENDER_STEPS = """
In Blender (free, blender.org):
  1. File > Import > Wavefront (.obj) > pick terrain.obj (keep defaults: Forward -Z, Up Y).
     Press Z > Material Preview to see the drone photo on it.
  2. Do NOT move, rotate or scale the whole object (that would shift it off its real position).
  3. Shape the ground: Sculpt Mode (Draw / Clay Strips / Flatten / Grab), or Edit Mode with
     Proportional Editing (O) and G then Z to move vertices up/down. 1 Blender unit = 1 metre.
  4. File > Export > Wavefront (.obj), keep defaults (Forward -Z, Up Y), tick 'Selected Only',
     save as terrain_edited.obj next to terrain.obj.
  5. Run:  dronemap from-blender <same project> <path>/terrain_edited.obj
"""


def to_blender(a) -> int:
    from .geo import read_polygon
    from .mesh import heightfield, uv_for, write_obj
    from .raster import bounds_for_ring, dataset_info, default_out, load_grid, read_rgb, resolve_dtm, resolve_ortho

    dtm = resolve_dtm(a.source)
    ortho = resolve_ortho(a.source)
    out = ensure_dir(a.out or default_out(a.source, "blender"))
    inf = dataset_info(dtm)
    bounds = None
    if a.area:
        _, ring = read_polygon(a.area)
        bounds = bounds_for_ring(inf["crs"], ring, a.margin)
    grid = load_grid(dtm, a.res, bounds)
    left, bottom, right, top = grid.bounds
    origin = (round((left + right) / 2), round((bottom + top) / 2), float(np.floor(np.nanmin(grid.z))))
    x0, y0 = left + grid.res / 2, top - grid.res / 2
    v, f, mx, my = heightfield(grid.z, grid.res, x0, y0, origin)
    if len(f) > 4_000_000:
        warn(f"{len(f):,} triangles - Blender will be slow. Use --res 1 or --area to limit the region.")
    rgba, rtr = read_rgb(ortho, grid, 8192) if ortho else (None, None)
    step(f"Writing {len(v):,} vertices / {len(f):,} triangles")
    obj = os.path.join(out, "terrain.obj")
    write_obj(obj, v, f, uv_for(mx, my, rtr, rgba.shape) if rgba is not None else None,
              rgba[..., :3] if rgba is not None else None)
    geo = {"dtm": os.path.abspath(dtm), "crs_wkt": grid.crs.to_wkt(), "origin": origin, "res": grid.res,
           "bounds": grid.bounds, "created": _dt.datetime.now().isoformat(timespec="seconds"),
           "note": "OBJ coordinates are Y-up: X=east, Y=up, Z=-north, metres relative to origin."}
    with open(os.path.join(out, "terrain_georef.json"), "w", encoding="utf-8") as fp:
        json.dump(geo, fp, indent=2)
    ok(f"Blender terrain: {obj}")
    info(BLENDER_STEPS)
    return 0


def from_blender(a) -> int:
    from rasterio.crs import CRS
    from scipy.interpolate import griddata

    from .maps import MapFig, colorbar
    from .mesh import read_obj_vertices
    from .raster import Grid, hillshade, load_grid, resolve_dtm, write_geotiff

    georef = a.georef or os.path.join(os.path.dirname(os.path.abspath(a.edited)), "terrain_georef.json")
    if not os.path.isfile(georef):
        raise UserError(f"{georef} not found (it is written by 'dronemap to-blender' next to terrain.obj).")
    with open(georef, encoding="utf-8") as fp:
        g = json.load(fp)
    dtm = resolve_dtm(a.source)
    crs = CRS.from_wkt(g["crs_wkt"])
    grid = load_grid(dtm, g["res"], tuple(g["bounds"]), crs)
    ox, oy, oz = g["origin"]
    step(f"Reading {a.edited}")
    v = read_obj_vertices(a.edited)  # local metres relative to the origin
    left, bottom, right, top = grid.bounds
    x, y = v[:, 0], v[:, 1]
    if x.min() + ox > right or x.max() + ox < left or y.min() + oy > top or y.max() + oy < bottom:
        raise UserError("The edited mesh does not overlap the original terrain - was it moved, or exported "
                        "with different axis settings (use Forward -Z, Up Y)?")
    h, w = grid.shape
    # interpolate in local coordinates (large UTM numbers cost precision in the triangulation)
    gx, gy = np.meshgrid(left - ox + (np.arange(w) + 0.5) * grid.res, top - oy - (np.arange(h) + 0.5) * grid.res)
    step("Comparing with the original terrain")
    ze = griddata((x, y), v[:, 2], (gx, gy), method="linear") + oz
    z0 = grid.z
    diff = ze - z0
    diff[~np.isfinite(diff)] = 0.0
    diff[np.abs(diff) < 0.02] = 0.0
    a_ = grid.cell_area
    cut = float(np.clip(-diff, 0, None).sum() * a_)
    fill = float(np.clip(diff, 0, None).sum() * a_)
    changed = diff != 0
    out = ensure_dir(a.out or os.path.dirname(os.path.abspath(a.edited)))
    edited_grid = Grid(np.where(np.isfinite(ze), ze, z0), grid.transform, grid.crs, grid.res)
    write_geotiff(os.path.join(out, "edited_terrain.tif"), edited_grid.z, grid)
    write_geotiff(os.path.join(out, "edited_cut_fill_m.tif"), np.where(changed, -diff, np.nan), grid)
    mf = MapFig(grid, "Blender edit: cut (red) / fill (blue)")
    mf.background(hillshade(edited_grid.z, grid.res))
    vmax = max(float(np.abs(diff).max()), 0.5)
    im = mf.image(np.where(changed, -diff, np.nan), cmap="RdBu_r", vmin=-vmax, vmax=vmax, alpha=0.75)
    colorbar(im, mf.ax, "cut depth (+) / fill height (-), m")
    png = mf.finish(os.path.join(out, "edited_cut_fill.png"))
    surplus = cut - fill / a.compaction
    lines = [
        "# Blender earthwork",
        f"- Changed area: {fmt_area(changed.sum() * a_)}",
        f"- Cut (dig): {cut:,.0f} m³",
        f"- Fill (raise): {fill:,.0f} m³ compacted (needs {fill / a.compaction:,.0f} m³ of dug soil)",
        (f"- Surplus soil: {surplus:,.0f} m³ in the ground (~{surplus * a.swell:,.0f} m³ loose)" if surplus >= 0
         else f"- Shortfall: {-surplus:,.0f} m³ of soil must be brought in"),
        f"- Max dig {float(np.clip(-diff, 0, None).max()):.2f} m, max raise {float(np.clip(diff, 0, None).max()):.2f} m",
        f"- Map: {png}",
        f"- Edited terrain GeoTIFF (use with 'dronemap site' to check new drainage): "
        f"{os.path.join(out, 'edited_terrain.tif')}",
    ]
    rep = "\n".join(lines) + "\n"
    with open(os.path.join(out, "edited_report.md"), "w", encoding="utf-8") as fp:
        fp.write(rep)
    info("")
    info(rep)
    if not changed.any():
        warn("No height changes found. Did you export the edited object (and save the .blend edits)?")
    return 0
