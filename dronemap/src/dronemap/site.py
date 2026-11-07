"""`dronemap site`: terrain analysis for planning earthworks and water harvesting."""
from __future__ import annotations

import json
import math
import os

import numpy as np

from . import hydro
from .geo import read_polygon
from .maps import MapFig, colorbar
from .raster import (Kmz, bounds_for_ring, colorize, contour_lines, dataset_info, default_out, hillshade,
                     load_grid, polygon_mask, read_rgb, resolve_dtm, resolve_ortho, ring_to_crs, slope_percent,
                     write_dxf_contours, write_geotiff)
from .util import ensure_dir, fmt_area, google_maps_link, info, ok, step, warn

SLOPE_CLASSES = [(0, 2, "flat 0-2%"), (2, 5, "gentle 2-5%"), (5, 10, "moderate 5-10%"),
                 (10, 20, "steep 10-20%"), (20, 1e9, "very steep >20%")]


def _latlon(grid, r, c):
    x, y = grid.xy_of([r], [c])
    lon, lat = grid.to_lonlat(x, y)
    return float(lat[0]), float(lon[0])


def run_site(a) -> int:
    dtm = resolve_dtm(a.source)
    ortho = resolve_ortho(a.source)
    out = ensure_dir(a.out or default_out(a.source, "site"))
    step(f"Terrain analysis of {dtm}")
    inf = dataset_info(dtm)
    ring = None
    bounds = None
    if a.boundary:
        _, ring = read_polygon(a.boundary)
        bounds = bounds_for_ring(inf["crs"], ring, 30.0)
    grid = load_grid(dtm, a.res, bounds)
    z = grid.z
    valid = np.isfinite(z)
    plot = polygon_mask(grid, ring) & valid if ring else valid
    if plot.sum() < 10:
        warn("Boundary barely overlaps the terrain model - check the KML is for this project.")
        plot = valid
    hs = hillshade(z, grid.res)
    slope = slope_percent(z, grid.res)

    # ---------------- statistics
    zp = z[plot]
    area = plot.sum() * grid.cell_area
    hi_rc = np.unravel_index(np.nanargmax(np.where(plot, z, -np.inf)), z.shape)
    lo_rc = np.unravel_index(np.nanargmin(np.where(plot, z, np.inf)), z.shape)
    stats = {
        "dtm": dtm, "crs": str(grid.crs), "grid_m": grid.res, "area_m2": float(area),
        "elev_min": float(np.nanmin(zp)), "elev_max": float(np.nanmax(zp)), "elev_mean": float(np.nanmean(zp)),
        "relief_m": float(np.nanmax(zp) - np.nanmin(zp)), "slope_mean_pct": float(np.nanmean(slope[plot])),
        "highest": _latlon(grid, *hi_rc), "lowest": _latlon(grid, *lo_rc),
        "slope_classes": [],
    }
    for lo, hi, label in SLOPE_CLASSES:
        m = plot & (slope >= lo) & (slope < hi)
        stats["slope_classes"].append({"class": label, "area_m2": float(m.sum() * grid.cell_area)})

    # ---------------- hydrology (coarser grid keeps it fast)
    hres = max(a.res, 1.0)
    hgrid = grid if hres == grid.res else load_grid(dtm, hres, grid.bounds)
    step(f"Drainage analysis on a {hres:g} m grid")
    filled = hydro.priority_flood(hgrid.z)
    rec = hydro.d8_receivers(filled, hgrid.res)
    acc, order = hydro.accumulate(filled, rec, hgrid.cell_area)
    dep, dep_list = hydro.depressions(hgrid.z, filled, hgrid.res)
    hvalid = np.isfinite(hgrid.z)
    hplot = (polygon_mask(hgrid, ring) & hvalid) if ring else hvalid
    if hplot.sum() < 10:
        hplot = hvalid
    streams = (acc >= a.stream_area) & hvalid
    lines = hydro.stream_polylines(streams, rec)

    # outlets: plot cells whose flow leaves the plot (or the data)
    r = rec.ravel()
    flat_plot = hplot.ravel()
    leaving = np.zeros(hplot.size, dtype=bool)
    ids = np.nonzero(flat_plot)[0]
    leaving[ids] = (r[ids] < 0) | ~flat_plot[np.maximum(r[ids], 0)]
    cand = sorted(np.nonzero(leaving)[0], key=lambda i: -acc.ravel()[i])
    outlets = []
    for i in cand:
        rr, cc = divmod(int(i), hgrid.shape[1])
        if any(math.hypot(rr - o["row"], cc - o["col"]) * hres < 25 for o in outlets):
            continue
        A = float(acc.ravel()[i])
        if A < max(a.stream_area / 4, 200):
            break
        lat, lon = _latlon(hgrid, rr, cc)
        outlets.append({"row": rr, "col": cc, "lat": lat, "lon": lon, "catchment_m2": A,
                        "runoff_m3_per_year": A * a.rainfall_mm / 1000.0 * a.runoff})
        if len(outlets) >= 6:
            break
    for d in dep_list:
        d["lat"], d["lon"] = _latlon(hgrid, int(round(d["row"])), int(round(d["col"])))
    plot_dep = [d for d in dep_list if hplot[int(round(d["row"])), int(round(d["col"]))]][:10]
    stats["outlets"] = [{k: v for k, v in o.items() if k not in ("row", "col")} for o in outlets]
    stats["depressions"] = [{k: v for k, v in d.items() if k not in ("row", "col", "label")} for d in plot_dep]

    # ---------------- contours
    step(f"Contours every {a.contour_interval:g} m")
    cont = contour_lines(grid, np.where(valid, z, np.nan), a.contour_interval)
    _write_contour_geojson(grid, cont, os.path.join(out, "contours.geojson"))
    write_dxf_contours(os.path.join(out, "contours.dxf"), cont)

    # ---------------- rasters
    write_geotiff(os.path.join(out, "slope_percent.tif"), slope, grid)
    write_geotiff(os.path.join(out, "flow_accumulation_m2.tif"), np.where(hvalid, acc, np.nan), hgrid)
    write_geotiff(os.path.join(out, "ponding_depth_m.tif"), np.where(hvalid, dep, np.nan), hgrid)

    # ---------------- maps
    step("Drawing maps")
    rgba, rgba_tr = read_rgb(ortho, grid, 3000) if ortho else (None, None)
    ring_xy = ring_to_crs(grid, ring) if ring else None
    major = 1.0 if a.contour_interval < 1 else a.contour_interval * 5
    zmin, zmax = np.nanpercentile(zp, 1), np.nanpercentile(zp, 99)

    # overview
    mf = MapFig(grid, "Site overview: contours, drainage lines, low spots, water exits")
    mf.background(hs, rgba, rgba_tr, dim=0.15)
    mf.contours(cont, major_every=major, color="#5a3e1b")
    _draw_streams(mf, hgrid, lines, acc)
    _draw_points(mf, hgrid, plot_dep, outlets)
    if ring_xy:
        mf.polygon(ring_xy, "yellow", 2.0, label="plot boundary")
    mf.finish(os.path.join(out, "site_overview.png"), legend=True,
              note=f"Contours every {a.contour_interval:g} m (labels every {major:g} m). Blue lines: drainage "
                   f"(catchment > {a.stream_area:g} m²).")
    # elevation
    mf = MapFig(grid, "Elevation (metres above sea level) with contours")
    im = mf.image(colorize(z, "terrain", zmin, zmax, 1.0, hs))
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    sm = cm.ScalarMappable(norm=mcolors.Normalize(zmin, zmax), cmap="terrain")
    colorbar(sm, mf.ax, "elevation (m)")
    mf.contours(cont, major_every=major)
    if ring_xy:
        mf.polygon(ring_xy, "red", 1.5)
    mf.finish(os.path.join(out, "site_elevation.png"))
    # slope classes
    mf = MapFig(grid, "Slope classes")
    cls = np.full(z.shape, np.nan)
    for k, (lo, hi, _) in enumerate(SLOPE_CLASSES):
        cls[(slope >= lo) & (slope < hi)] = k
    cmap = mcolors.ListedColormap(["#1a9850", "#91cf60", "#fee08b", "#fc8d59", "#d73027"])
    mf.background(hs)
    im = mf.image(cls, cmap=cmap, vmin=-0.5, vmax=4.5, alpha=0.65)
    cb = colorbar(im, mf.ax, "")
    cb.set_ticks(range(5))
    cb.set_ticklabels([c[2] for c in SLOPE_CLASSES])
    if ring_xy:
        mf.polygon(ring_xy, "k", 1.5)
    mf.finish(os.path.join(out, "site_slope.png"))
    # drainage
    mf = MapFig(hgrid, "Where water flows (upstream area) and where it collects")
    mf.background(hillshade(hgrid.z, hgrid.res))
    la = np.log10(np.where(hvalid, acc, np.nan))
    im = mf.image(la, cmap="Blues", vmin=math.log10(hgrid.cell_area), vmax=np.nanmax(la), alpha=0.75)
    colorbar(im, mf.ax, "log10 upstream area (m²)")
    dmask = np.where(dep > 0.05, dep, np.nan)
    mf.image(dmask, cmap="cool", alpha=0.8)
    _draw_points(mf, hgrid, plot_dep, outlets)
    if ring_xy:
        mf.polygon(ring_xy, "orange", 1.5)
    mf.finish(os.path.join(out, "site_drainage.png"), legend=True,
              note="Magenta: natural low spots that pond after rain. Triangles: where water leaves the plot.")

    # ---------------- Google Earth package
    kmz = Kmz("Site analysis")
    if rgba is not None:
        kmz.overlay("Orthophoto", rgba, rgba_tr, grid.crs, visible=False)
    kmz.overlay("Elevation", colorize(z, "terrain", zmin, zmax, 0.85, hs), grid.transform, grid.crs, False)
    slope_rgba = colorize(cls, "RdYlGn_r", 0, 4, 0.6)
    kmz.overlay("Slope classes", slope_rgba, grid.transform, grid.crs, False)
    pond_rgba = colorize(np.where(dep > 0.05, dep, np.nan), "cool", 0, max(0.3, float(np.nanmax(dep))), 0.8)
    kmz.overlay("Natural ponding (rain collects here)", pond_rgba, hgrid.transform, hgrid.crs, True)
    d = kmz.doc
    d.style_line("cmaj", "ff1b3e5a", 1.6)
    d.style_line("cmin", "991b3e5a", 0.8)
    d.style_line("stream", "ffff9933", 3.0)
    d.style_point("outlet", "ff0000ff", 1.1, "http://maps.google.com/mapfiles/kml/shapes/triangle.png")
    d.style_point("pondpt", "ffff00ff", 0.9, "http://maps.google.com/mapfiles/kml/shapes/water.png")
    d.folder("Contours")
    for level, ls in cont:
        is_major = abs(level / major - round(level / major)) < 1e-6
        for ln in ls:
            lon, lat = grid.to_lonlat(ln[:, 0], ln[:, 1])
            d.line(f"{level:g} m", list(zip(lat, lon)), "cmaj" if is_major else "cmin")
    d.end_folder()
    d.folder("Drainage lines")
    for ln in lines:
        rows = [p[0] for p in ln]
        cols = [p[1] for p in ln]
        x, y = hgrid.xy_of(rows, cols)
        lon, lat = hgrid.to_lonlat(x, y)
        d.line("drainage", list(zip(lat, lon)), "stream")
    d.end_folder()
    d.folder("Water exits")
    for k, o in enumerate(outlets, 1):
        d.point(f"Exit {k}: {o['catchment_m2'] / 4046.86:.1f} ac drains here", o["lat"], o["lon"], "outlet",
                f"Catchment {fmt_area(o['catchment_m2'])} (within surveyed area). Rough annual runoff "
                f"{o['runoff_m3_per_year']:,.0f} m³ at {a.rainfall_mm:g} mm rain, coefficient {a.runoff:g}.")
    d.end_folder()
    d.folder("Natural low spots")
    for k, p in enumerate(plot_dep, 1):
        d.point(f"Low spot {k}", p["lat"], p["lon"], "pondpt",
                f"Area {p['area_m2']:,.0f} m², holds ~{p['volume_m3']:,.0f} m³, max depth {p['max_depth_m']:.2f} m")
    d.end_folder()
    if ring:
        d.style_poly("plotb", "ff00ffff", "0000ffff", 2.5)
        d.polygon("Plot boundary", ring, "plotb")
    kmz.save(os.path.join(out, "site.kmz"))

    # ---------------- report
    with open(os.path.join(out, "site_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    report = _report(stats, a, out)
    with open(os.path.join(out, "site_report.md"), "w", encoding="utf-8") as f:
        f.write(report)
    info("")
    info(report)
    ok(f"Site analysis written to {out}")
    return 0


def _draw_streams(mf, hgrid, lines, acc):
    first = True
    amax = max(float(np.nanmax(acc)), 1.0)
    for ln in lines:
        rows = [p[0] for p in ln]
        cols = [p[1] for p in ln]
        x, y = hgrid.xy_of(rows, cols)
        a = acc[rows[-1], cols[-1]]
        lw = 0.8 + 3.0 * math.sqrt(a / amax)
        mf.ax.plot(mf.lx(x), mf.ly(y), color="#1f78ff", lw=lw, alpha=0.9, label="drainage line" if first else None)
        first = False


def _draw_points(mf, hgrid, deps, outlets):
    for k, p in enumerate(deps, 1):
        x, y = hgrid.xy_of([p["row"]], [p["col"]])
        mf.ax.scatter(mf.lx(x), mf.ly(y), s=40 + p["volume_m3"] ** 0.5 * 4, facecolor="none", edgecolor="magenta",
                      lw=1.5, label="natural low spot" if k == 1 else None, zorder=5)
        mf.ax.text(mf.lx(x)[0] + 3, mf.ly(y)[0] + 3, f"L{k}", color="magenta", fontsize=8, zorder=6)
    for k, o in enumerate(outlets, 1):
        x, y = hgrid.xy_of([o["row"]], [o["col"]])
        mf.ax.scatter(mf.lx(x), mf.ly(y), marker="v", s=90, color="red", edgecolor="k",
                      label="water leaves plot" if k == 1 else None, zorder=6)
        mf.ax.text(mf.lx(x)[0] + 3, mf.ly(y)[0] - 8, f"E{k}", color="red", fontsize=9, fontweight="bold", zorder=7)


def _write_contour_geojson(grid, cont, path):
    feats = []
    for level, lines in cont:
        for ln in lines:
            lon, lat = grid.to_lonlat(ln[:, 0], ln[:, 1])
            feats.append({"type": "Feature", "properties": {"elevation": level},
                          "geometry": {"type": "LineString",
                                       "coordinates": [[round(float(x), 8), round(float(y), 8)]
                                                       for x, y in zip(lon, lat)]}})
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f)


def _report(s, a, out) -> str:
    L = ["# Site analysis", ""]
    L.append(f"- Area analysed: {fmt_area(s['area_m2'])}")
    L.append(f"- Elevation: {s['elev_min']:.2f} to {s['elev_max']:.2f} m (relief {s['relief_m']:.2f} m), "
             f"average slope {s['slope_mean_pct']:.1f}%")
    hl, ll = s["highest"], s["lowest"]
    L.append(f"- Highest point: {hl[0]:.6f}, {hl[1]:.6f}  ({google_maps_link(*hl)})")
    L.append(f"- Lowest point:  {ll[0]:.6f}, {ll[1]:.6f}  ({google_maps_link(*ll)})")
    L.append("")
    L.append("## Slope")
    for c in s["slope_classes"]:
        pct = 100 * c["area_m2"] / max(s["area_m2"], 1)
        L.append(f"- {c['class']:16s} {fmt_area(c['area_m2'])}  ({pct:.0f}%)")
    L.append("")
    L.append("## Where surface water leaves the plot (E-points on the maps)")
    if not s["outlets"]:
        L.append("- No significant exit found (flat site, or the boundary encloses the whole catchment).")
    for k, o in enumerate(s["outlets"], 1):
        L.append(f"- E{k}: {o['lat']:.6f}, {o['lon']:.6f} - drains {fmt_area(o['catchment_m2'])}; "
                 f"~{o['runoff_m3_per_year']:,.0f} m³ runoff/year "
                 f"({a.rainfall_mm:g} mm rain x {a.runoff:g} runoff coefficient)")
    L.append("")
    L.append("A pond just upstream of the biggest exit catches the most runoff. Catchments only count the "
             "surveyed area; water arriving from neighbouring land adds to them.")
    L.append("")
    L.append("## Natural low spots (L-points: rain ponds here today)")
    if not s["depressions"]:
        L.append("- None deeper than 5 cm.")
    for k, d in enumerate(s["depressions"], 1):
        L.append(f"- L{k}: {d['lat']:.6f}, {d['lon']:.6f} - {d['area_m2']:,.0f} m², holds ~{d['volume_m3']:,.0f} m³, "
                 f"max {d['max_depth_m']:.2f} m deep")
    L.append("")
    L.append("## Files")
    for fn in ("site_overview.png", "site_elevation.png", "site_slope.png", "site_drainage.png", "site.kmz",
               "contours.dxf", "contours.geojson", "slope_percent.tif", "flow_accumulation_m2.tif",
               "ponding_depth_m.tif", "site_stats.json"):
        L.append(f"- {os.path.join(out, fn)}")
    L.append("")
    L.append("Open site.kmz in Google Earth Pro; contours.dxf opens in AutoCAD/QCAD for a contractor.")
    return "\n".join(L) + "\n"
