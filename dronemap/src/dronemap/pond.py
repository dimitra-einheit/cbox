"""`dronemap pond`: carve a pond into the terrain model and quantify it.

Geometry (all on a regular grid):
- The outline you draw is the TOP EDGE of the pond.
- Inner sides slope down from the edge at `side_slope` (horizontal : 1 vertical) to a flat floor.
- Floor level = lowest natural ground on the edge minus `depth` (or an absolute --floor-level).
- Dug-out pond: water can stand up to the lowest edge point minus freeboard.
- With --bund-height, an earth bank (crest width, outer slope) is built all round where the ground is
  below the crest; the water level rises with it (crest minus freeboard).
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from .geo import LocalProjection, read_polygon
from .raster import (Kmz, bounds_for_ring, colorize, contour_lines, dataset_info, default_out, fill_nan_nearest,
                     hillshade, load_grid, polygon_mask, read_rgb, resolve_dtm, resolve_ortho, ring_to_crs,
                     write_geotiff)
from .util import UserError, ensure_dir, fmt_area, info, ok, step, warn

EARTH = np.array([150, 112, 72], dtype=float)  # excavated soil colour for 3D views


# ------------------------------------------------------------------ core maths (pure, testable)


def design_pond(z: np.ndarray, res: float, inside: np.ndarray, depth: float | None = 3.0,
                floor_level: float | None = None, side_slope: float = 2.0, freeboard: float = 0.5,
                bund_height: float = 0.0, bund_crest: float = 2.0, bund_outer_slope: float = 2.0) -> dict:
    from scipy import ndimage

    if side_slope <= 0 or bund_outer_slope <= 0:
        raise UserError("Slopes must be positive (e.g. 2 for 2:1).")
    if inside.sum() < 4:
        raise UserError("Pond outline covers fewer than 4 grid cells - draw it bigger or use a finer --res.")
    zf = fill_nan_nearest(z)
    # distance (m) from each inside cell to the edge, and terrain just outside the nearest edge point
    d_in, idx = ndimage.distance_transform_edt(inside, sampling=res, return_indices=True)
    d_in = np.maximum(d_in - res / 2.0, 0.0)
    t_rim = zf[idx[0], idx[1]]
    rim = ndimage.binary_dilation(inside, structure=np.ones((3, 3))) & ~inside
    if not rim.any():
        raise UserError("Pond outline touches the edge of the terrain data; add margin or move it.")
    lowest_rim = float(zf[rim].min())
    highest_rim = float(zf[rim].max())
    crest = lowest_rim + bund_height if bund_height > 0 else None
    fsl = (crest if crest is not None else lowest_rim) - freeboard
    floor = float(floor_level) if floor_level is not None else lowest_rim - float(depth)
    if floor >= fsl:
        raise UserError(f"Floor ({floor:.2f} m) would be above the full water level ({fsl:.2f} m). Dig deeper.")

    ref = t_rim if crest is None else np.maximum(t_rim, crest)
    slope_surf = np.maximum(floor, ref - d_in / side_slope)
    if crest is None:
        design_in = np.minimum(zf, slope_surf)
    else:
        low_side = t_rim < crest
        design_in = np.where(low_side, slope_surf, np.minimum(zf, slope_surf))
    design = np.where(inside, design_in, zf)
    if crest is not None:
        d_out = ndimage.distance_transform_edt(~inside, sampling=res)
        d_out = np.maximum(d_out - res / 2.0, 0.0)
        bank = crest - np.maximum(0.0, d_out - bund_crest) / bund_outer_slope
        design = np.where(~inside, np.maximum(zf, bank), design)
    cut = np.maximum(zf - design, 0.0)
    fill = np.maximum(design - zf, 0.0)
    a = res * res
    levels = list(np.arange(floor, fsl, 0.25)) + [fsl]
    stage = []
    for h in levels:
        wd = np.where(inside, np.maximum(h - design, 0.0), 0.0)
        stage.append({"level": float(h), "depth": float(h - floor), "area_m2": float((wd > 0).sum() * a),
                      "volume_m3": float(wd.sum() * a)})
    water = np.where(inside, np.maximum(fsl - design, 0.0), 0.0)
    return {
        "design": design, "cut": cut, "fill": fill, "water_depth": water, "zf": zf,
        "lowest_rim": lowest_rim, "highest_rim": highest_rim, "crest": crest, "fsl": fsl, "floor": floor,
        "cut_m3": float(cut.sum() * a), "fill_m3": float(fill.sum() * a),
        "capacity_m3": float(water.sum() * a), "water_area_m2": float((water > 0).sum() * a),
        "top_area_m2": float(inside.sum() * a), "floor_area_m2": float((inside & (np.abs(design - floor) < 1e-6)).sum() * a),
        "max_cut_m": float(cut.max()), "max_fill_m": float(fill.max()), "water_depth_m": float(fsl - floor),
        "dig_depth_below_lowest_edge_m": float(lowest_rim - floor), "stage": stage,
        "edge_touch": bool(crest is not None and (fill[0, :].any() or fill[-1, :].any() or fill[:, 0].any()
                                                   or fill[:, -1].any())),
    }


def earth_balance(r: dict, swell: float, compaction: float, trolley_m3: float, excavator_rate: float,
                  cost_per_hour: float | None) -> dict:
    fill_bank = r["fill_m3"] / compaction if compaction > 0 else r["fill_m3"]
    spoil_bank = r["cut_m3"] - fill_bank
    out = {
        "fill_needs_bank_m3": fill_bank,
        "spoil_bank_m3": spoil_bank,
        "spoil_loose_m3": max(spoil_bank, 0.0) * swell,
        "borrow_bank_m3": max(-spoil_bank, 0.0),
        "trolley_loads": max(spoil_bank, 0.0) * swell / trolley_m3 if trolley_m3 > 0 else None,
        "machine_hours": r["cut_m3"] / excavator_rate if excavator_rate > 0 else None,
    }
    out["cost"] = out["machine_hours"] * cost_per_hour if (cost_per_hour and out["machine_hours"]) else None
    return out


# ------------------------------------------------------------------ CLI entry


def _rect_ring(center, length, width, rotation_deg):
    lat0, lon0 = center
    proj = LocalProjection(lat0, lon0)
    t = math.radians(rotation_deg)
    ax = (math.sin(t), math.cos(t))  # long axis (east, north)
    pe = (ax[1], -ax[0])
    pts = []
    for sl, sw in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
        x = sl * length / 2 * ax[0] + sw * width / 2 * pe[0]
        y = sl * length / 2 * ax[1] + sw * width / 2 * pe[1]
        pts.append(proj.to_latlon(x, y))
    return pts


def run_pond(a) -> int:
    from .geo import parse_latlon

    dtm = resolve_dtm(a.source)
    ortho = resolve_ortho(a.source)
    if a.outline:
        oname, ring = read_polygon(a.outline)
    else:
        try:
            L, W = [float(v) for v in a.size.lower().replace("×", "x").split("x")]
        except ValueError:
            raise UserError("--size must look like 40x30") from None
        ring = _rect_ring(parse_latlon(a.center), L, W, a.rotation)
        oname = f"{L:g} x {W:g} m rectangle"
    depths = [float(d) for d in str(a.depth).split(",") if d.strip()]
    if a.floor_level is not None:
        depths = [None]
    base_out = a.out or default_out(a.source, a.name)
    inf = dataset_info(dtm)
    margin = max(15.0, a.bund_crest + (a.bund_height + 4.0) * a.bund_outer_slope + 8.0)
    results = []
    for dep in depths:
        label = f"floor {a.floor_level:g} m" if dep is None else f"depth {dep:g} m"
        out = ensure_dir(base_out if len(depths) == 1 else os.path.join(base_out, f"depth_{dep:g}m"))
        step(f"Pond '{a.name}' ({oname}), {label}")
        for attempt in range(3):
            grid = load_grid(dtm, a.res, bounds_for_ring(inf["crs"], ring, margin))
            inside = polygon_mask(grid, ring)
            nodata_frac = float((~np.isfinite(grid.z) & inside).sum()) / max(inside.sum(), 1)
            r = design_pond(grid.z, grid.res, inside, depth=dep, floor_level=a.floor_level,
                            side_slope=a.side_slope, freeboard=a.freeboard, bund_height=a.bund_height,
                            bund_crest=a.bund_crest, bund_outer_slope=a.bund_outer_slope)
            if not r["edge_touch"]:
                break
            margin *= 2
        if nodata_frac > 0.02:
            warn(f"{nodata_frac * 100:.0f}% of the pond area has no terrain data (outside the survey?) - "
                 "volumes there are guessed from nearby ground.")
        eb = earth_balance(r, a.swell, a.compaction, a.trolley_m3, a.excavator_rate, a.cost_per_hour)
        catch = _catchment(dtm, inf, ring, a)
        files = _outputs(grid, inside, ring, r, eb, catch, ortho, a, out)
        rep = _report(a, oname, label, r, eb, catch, files)
        with open(os.path.join(out, "pond_report.md"), "w", encoding="utf-8") as f:
            f.write(rep)
        numbers = {k: v for k, v in r.items() if not isinstance(v, np.ndarray)}
        numbers.update({"earth": eb, "catchment": catch})
        with open(os.path.join(out, "pond_numbers.json"), "w", encoding="utf-8") as f:
            json.dump(numbers, f, indent=2, default=float)
        info("")
        info(rep)
        results.append((label, r, eb, catch, out))
    if len(results) > 1:
        info(_comparison(results, a))
        with open(os.path.join(base_out, "comparison.md"), "w", encoding="utf-8") as f:
            f.write(_comparison(results, a))
    ok(f"Pond design written to {os.path.abspath(base_out)}")
    return 0


def _catchment(dtm, inf, ring, a) -> dict | None:
    """Upstream area (within the surveyed DTM) that drains into the pond, on a 1-2 m grid."""
    from . import hydro

    try:
        b = inf["bounds"]
        span = max(b[2] - b[0], b[3] - b[1])
        hres = max(1.0, span / 700.0)
        g = load_grid(dtm, hres)
        target = polygon_mask(g, ring) & np.isfinite(g.z)
        if target.sum() == 0:
            return None
        filled = hydro.priority_flood(g.z)
        rec = hydro.d8_receivers(filled, g.res)
        _, order = hydro.accumulate(filled, rec, g.cell_area)
        c = hydro.catchment(target, rec, order)
        area = float(c.sum() * g.cell_area)
        return {"area_m2": area, "grid_m": hres,
                "inflow_m3_per_year": area * a.rainfall_mm / 1000.0 * a.runoff,
                "rainfall_mm": a.rainfall_mm, "runoff_coefficient": a.runoff}
    except Exception as e:  # catchment is a bonus; never block the design on it
        warn(f"Catchment estimate skipped: {e}")
        return None


# ------------------------------------------------------------------ outputs


def _outputs(grid, inside, ring, r, eb, catch, ortho, a, out) -> dict:
    from .maps import MapFig, colorbar
    from .mesh import GlbBuilder, heightfield, uv_for, write_stl_terrain

    files = {}
    z, design, cut, fill = r["zf"], r["design"], r["cut"], r["fill"]
    cutfill = np.where(cut > 0, cut, -fill)
    files["design_tif"] = write_geotiff(os.path.join(out, "designed_terrain.tif"), design, grid)
    files["cutfill_tif"] = write_geotiff(os.path.join(out, "cut_fill_m.tif"), np.where(np.abs(cutfill) > 1e-6,
                                                                                         cutfill, np.nan), grid)
    hs_d = hillshade(design, grid.res)
    rgba, rtr = read_rgb(ortho, grid, 2048) if ortho else (None, None)
    ring_xy = ring_to_crs(grid, ring)
    changed = (cut > 0.02) | (fill > 0.02)

    # ---- plan map
    mf = MapFig(grid, f"Pond plan: cut (red) / fill (blue); full water level {r['fsl']:.2f} m")
    mf.background(hs_d, rgba, rtr, dim=0.25)
    vmax = max(r["max_cut_m"], r["max_fill_m"], 0.5)
    shown = np.where(changed, cutfill, np.nan)
    im = mf.image(shown, cmap="RdBu_r", vmin=-vmax, vmax=vmax, alpha=0.75)
    colorbar(im, mf.ax, "cut depth (+) / fill height (-), m")
    dc = contour_lines(grid, np.where(changed | inside, design, np.nan), 0.5, smooth_cells=0)
    mf.contours(dc, major_every=1.0, color="k")
    water = np.where(r["water_depth"] > 0, 1.0, np.nan)
    mf.image(water, cmap="Blues", vmin=0, vmax=1.5, alpha=0.25)
    mf.polygon(ring_xy, "yellow", 2.0, label="pond top edge")
    mf.finish(os.path.join(out, "pond_plan.png"), legend=True,
              note="Black lines: designed ground every 0.5 m. Light blue: water surface at full level.")
    files["plan_png"] = os.path.join(out, "pond_plan.png")

    # ---- cross sections along the pond's long and short axes
    files["sections_png"] = _sections(grid, inside, r, os.path.join(out, "pond_sections.png"))

    # ---- 3D models (before / after) as GLB
    step("Building 3D before/after models")
    left, bottom, right, top = grid.bounds
    origin = ((left + right) / 2, (bottom + top) / 2, float(np.nanmin(design)) - 1.0)
    x0 = left + grid.res / 2
    y0 = top - grid.res / 2
    stepn = max(1, int(round(0.5 / grid.res)))  # ~0.5 m mesh
    for tag, surf in (("before", z), ("after", design)):
        v, f, mx, my = heightfield(surf, grid.res, x0, y0, origin, stepn)
        g = GlbBuilder()
        if rgba is not None:
            tex = rgba[..., :3].astype(float)
            if tag == "after":
                tex = _paint_earth(tex, rtr, grid, changed, hs_d)
            mat = g.material("terrain", texture=g.texture(tex.astype(np.uint8)))
            g.mesh("terrain", v, f, mat, uv=uv_for(mx, my, rtr, rgba.shape))
        else:
            col = _vertex_colors(surf, changed if tag == "after" else None, stepn)
            g.mesh("terrain", v, f, g.material("terrain"), colors=col)
        if tag == "after":
            wmask = r["water_depth"] > 0
            wz = np.where(wmask, r["fsl"], np.nan)
            wv, wf, _, _ = heightfield(wz, grid.res, x0, y0, origin, stepn)
            g.mesh("water", wv, wf, g.material("water", (0.15, 0.45, 0.85, 0.55), blend=True))
        files[f"glb_{tag}"] = g.save(os.path.join(out, f"pond_3d_{tag}.glb"))

    if a.stl:
        info_stl = write_stl_terrain(os.path.join(out, "pond_after_print.stl"), design, grid.res,
                                     exaggeration=a.exaggeration)
        files["stl"] = os.path.join(out, "pond_after_print.stl")
        files["stl_info"] = info_stl

    # ---- Google Earth overlay
    kmz = Kmz(f"Pond {a.name}")
    kmz.overlay("Cut (red) / fill (blue)", colorize(shown, "RdBu_r", -vmax, vmax, 0.8), grid.transform, grid.crs)
    kmz.overlay("Water at full level", colorize(np.where(r["water_depth"] > 0, r["water_depth"], np.nan), "Blues",
                                                0, max(r["water_depth_m"], 0.5), 0.7), grid.transform, grid.crs,
                False)
    kmz.doc.style_poly("edge", "ff00ffff", "00000000", 2.5)
    kmz.doc.polygon("Pond top edge", ring, "edge", f"Water level {r['fsl']:.2f} m, floor {r['floor']:.2f} m")
    files["kmz"] = kmz.save(os.path.join(out, "pond.kmz"))
    return files


def _paint_earth(tex, tex_tr, grid, changed, hs_design):
    """Recolour excavated/filled areas on the orthophoto texture with shaded soil."""
    h, w = tex.shape[:2]
    cols = np.arange(w)
    rows = np.arange(h)
    tx = tex_tr.c + (cols + 0.5) * tex_tr.a
    ty = tex_tr.f + (rows + 0.5) * tex_tr.e
    gc = np.clip(((tx - grid.transform.c) / grid.res).astype(int), 0, grid.shape[1] - 1)
    gr = np.clip(((grid.transform.f - ty) / grid.res).astype(int), 0, grid.shape[0] - 1)
    ch = changed[gr[:, None], gc[None, :]]
    shade = np.nan_to_num(hs_design[gr[:, None], gc[None, :]], nan=0.7)
    soil = EARTH[None, None, :] * (0.55 + 0.6 * shade[..., None])
    out = tex.copy()
    out[ch] = np.clip(soil[ch], 0, 255)
    return out


def _vertex_colors(surf, changed, stepn):
    s = surf[::stepn, ::stepn]
    ok_ = np.isfinite(s)
    zmin, zmax = np.nanmin(s), np.nanmax(s)
    t = (s[ok_] - zmin) / max(zmax - zmin, 1e-6)
    col = np.stack([0.35 + 0.4 * t, 0.55 + 0.25 * t, 0.25 + 0.2 * t], axis=1)
    if changed is not None:
        ch = changed[::stepn, ::stepn][ok_]
        col[ch] = EARTH / 255.0
    return col


def _sections(grid, inside, r, path):
    import matplotlib.pyplot as plt

    rows, cols = np.nonzero(inside)
    x, y = grid.xy_of(rows, cols)
    cx, cy = x.mean(), y.mean()
    cov = np.cov(np.stack([x - cx, y - cy]))
    evals, evecs = np.linalg.eigh(cov)
    axes = [evecs[:, 1], evecs[:, 0]]  # long, short
    fig, axs = plt.subplots(2, 1, figsize=(11, 8), dpi=150)
    names = ["Long section", "Cross section"]
    for ax, d, nm in zip(axs, axes, names):
        proj = (x - cx) * d[0] + (y - cy) * d[1]
        half = (proj.max() - proj.min()) / 2 + 15.0
        s = np.arange(-half, half, grid.res / 2)
        px, py = cx + s * d[0], cy + s * d[1]
        rr, cc = grid.rc_of(px, py)
        okk = (rr >= 0) & (rr < grid.shape[0]) & (cc >= 0) & (cc < grid.shape[1])
        s, rr, cc = s[okk], rr[okk], cc[okk]
        nat = r["zf"][rr, cc]
        des = r["design"][rr, cc]
        ax.fill_between(s, des, nat, where=nat > des, color="#e8a07a", alpha=0.6, label="cut (soil removed)",
                        zorder=2)
        ax.fill_between(s, nat, des, where=des > nat, color="#8c6d46", alpha=0.85, label="fill (bund)", zorder=2)
        wet = r["water_depth"][rr, cc] > 0
        ax.fill_between(s, des, r["fsl"], where=wet, color="#4a90d9", alpha=0.85, label="water at full level",
                        zorder=3)
        ax.plot(s, nat, color="#7f5f3f", lw=1.5, label="natural ground")
        ax.plot(s, des, color="k", lw=1.5, label="designed ground")
        ax.axhline(r["fsl"], color="#2166ac", lw=0.8, ls="--")
        ax.text(s[0], r["fsl"], f" FWL {r['fsl']:.2f}", color="#2166ac", fontsize=8, va="bottom")
        ax.axhline(r["floor"], color="k", lw=0.5, ls=":")
        ax.text(s[0], r["floor"], f" floor {r['floor']:.2f}", fontsize=8, va="bottom")
        bearing = (math.degrees(math.atan2(d[0], d[1])) + 360) % 180
        ax.set_title(f"{nm} through the pond centre (direction {bearing:.0f}°)", loc="left", fontsize=11)
        ax.set_xlabel("distance from pond centre (m)")
        ax.set_ylabel("elevation (m)")
        ax.grid(alpha=0.3)
    axs[0].legend(loc="lower right", fontsize=8, ncol=3)
    fig.text(0.01, 0.005, "Vertical scale exaggerated. FWL = full water level.", fontsize=7, color="#444")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _report(a, oname, label, r, eb, catch, files) -> str:
    L = [f"# Pond design: {a.name} ({oname}, {label})", ""]
    L.append("## Levels")
    L.append(f"- Natural ground along the edge: {r['lowest_rim']:.2f} m (lowest) to {r['highest_rim']:.2f} m (highest)")
    if r["crest"] is not None:
        L.append(f"- Bund crest: {r['crest']:.2f} m ({a.bund_height:g} m above the lowest edge point), "
                 f"crest width {a.bund_crest:g} m, outer slope {a.bund_outer_slope:g}:1")
    L.append(f"- Full water level: {r['fsl']:.2f} m (freeboard {a.freeboard:g} m)")
    L.append(f"- Floor: {r['floor']:.2f} m -> water depth at full level {r['water_depth_m']:.2f} m; "
             f"deepest dig {r['max_cut_m']:.2f} m")
    L.append(f"- Inner side slope {a.side_slope:g}:1 (horizontal:vertical)")
    L.append("")
    L.append("## Size and capacity")
    L.append(f"- Top edge area: {fmt_area(r['top_area_m2'])}; flat floor area: {fmt_area(r['floor_area_m2'])}")
    L.append(f"- Water surface at full level: {fmt_area(r['water_area_m2'])}")
    cap = r["capacity_m3"]
    L.append(f"- Storage at full level: {cap:,.0f} m³ = {cap * 1000:,.0f} litres = {cap / 1233.5:.2f} acre-feet")
    L.append(f"- That is about {cap / (4046.86 * 0.06):,.0f} acre-irrigations of 6 cm water depth, before "
             "evaporation and seepage losses (unlined ponds can lose a lot in sandy soil).")
    if catch:
        n_fill = catch["inflow_m3_per_year"] / cap if cap > 0 else 0
        L.append(f"- Catchment draining into the pond (surveyed area only): {fmt_area(catch['area_m2'])}; rough "
                 f"runoff {catch['inflow_m3_per_year']:,.0f} m³/year at {catch['rainfall_mm']:g} mm rain x "
                 f"{catch['runoff_coefficient']:g} -> fills the pond ~{n_fill:.1f}x per year")
        if n_fill < 1:
            L.append("  The pond's own catchment won't fill it: put it lower on the drainage line, add a "
                     "diversion channel, or plan for well/canal top-up.")
    L.append("")
    L.append("## Earthwork")
    L.append(f"- Excavation (cut): {r['cut_m3']:,.0f} m³ in the ground")
    L.append(f"- Bund fill: {r['fill_m3']:,.0f} m³ compacted (needs {eb['fill_needs_bank_m3']:,.0f} m³ of dug soil)")
    if eb["spoil_bank_m3"] >= 0:
        L.append(f"- Surplus soil: {eb['spoil_bank_m3']:,.0f} m³ in the ground = ~{eb['spoil_loose_m3']:,.0f} m³ loose "
                 f"(x{a.swell:g} swell) = ~{eb['trolley_loads']:,.0f} tractor-trolley loads of {a.trolley_m3:g} m³. "
                 "Use it to raise building pads, paths or field bunds.")
    else:
        L.append(f"- Shortfall: {eb['borrow_bank_m3']:,.0f} m³ of soil must be brought in for the bund.")
    L.append(f"- Excavator digging time: ~{eb['machine_hours']:,.0f} hours at {a.excavator_rate:g} m³/hour "
             "(check this rate with your operator)"
             + (f" -> ~₹{eb['cost']:,.0f} at ₹{a.cost_per_hour:,.0f}/hour" if eb["cost"] else
                " (add --cost-per-hour for a cost estimate)"))
    L.append("")
    L.append("## Stage-capacity (water level vs stored volume)")
    L.append("| level (m) | depth (m) | surface area (m²) | volume (m³) |")
    L.append("|---:|---:|---:|---:|")
    for s in r["stage"]:
        L.append(f"| {s['level']:.2f} | {s['depth']:.2f} | {s['area_m2']:,.0f} | {s['volume_m3']:,.0f} |")
    L.append("")
    L.append("## Files")
    for k in ("plan_png", "sections_png", "glb_before", "glb_after", "kmz", "design_tif", "cutfill_tif", "stl"):
        if files.get(k):
            L.append(f"- {files[k]}")
    if files.get("stl_info"):
        si = files["stl_info"]
        L.append(f"  STL: {si['size_mm'][0]} x {si['size_mm'][1]} mm, scale {si['scale']}, vertical x{a.exaggeration:g}")
    L.append("")
    L.append("Open the .glb files in Blender, or drag them into https://gltf-viewer.donmccurdy.com (local, "
             "nothing uploaded). Volumes are only as good as the terrain model: survey bare ground, and "
             "check a few levels with a dumpy level before you dig.")
    return "\n".join(L) + "\n"


def _comparison(results, a) -> str:
    L = ["", "# Options compared", "",
         "| option | water depth | capacity m³ | cut m³ | fill m³ | surplus soil m³ (bank) | trolley loads | machine h |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for label, r, eb, catch, out in results:
        L.append(f"| {label} | {r['water_depth_m']:.2f} | {r['capacity_m3']:,.0f} | {r['cut_m3']:,.0f} | "
                 f"{r['fill_m3']:,.0f} | {eb['spoil_bank_m3']:,.0f} | {eb['trolley_loads'] or 0:,.0f} | "
                 f"{eb['machine_hours']:,.0f} |")
    L.append("")
    L.append("Capacity per m³ dug: " + ", ".join(f"{lab}: {r['capacity_m3'] / max(r['cut_m3'], 1):.2f}"
                                                for lab, r, *_ in results))
    return "\n".join(L) + "\n"
