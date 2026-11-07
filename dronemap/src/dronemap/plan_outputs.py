"""Write a flight plan to disk: Litchi CSVs, Google Earth KML preview, summary, plan.json."""
from __future__ import annotations

import json
import math
import os

from .flightplan import PRESETS, Plan
from .kml import PALETTE, KmlDoc
from .litchi import write_mission_csv
from .util import ensure_dir, fmt_area, fmt_minutes, google_maps_link


def write_plan(plan: Plan, out_dir: str) -> dict:
    ensure_dir(out_dir)
    files = {"csv": []}
    for m in plan.missions:
        p = os.path.join(out_dir, f"{plan.name}_{m.name}.csv")
        write_mission_csv(p, m, plan.altitude_mode)
        files["csv"].append(p)
    files["kml"] = write_kml(plan, os.path.join(out_dir, f"{plan.name}_preview.kml"))
    files["json"] = write_json(plan, os.path.join(out_dir, f"{plan.name}_plan.json"))
    files["summary"] = os.path.join(out_dir, f"{plan.name}_summary.txt")
    with open(files["summary"], "w", encoding="utf-8") as f:
        f.write(summary_text(plan, files))
    return files


def write_kml(plan: Plan, path: str) -> str:
    doc = KmlDoc(f"{plan.name} flight plan")
    doc.style_poly("plot", "ff00ffff", "3300ffff", 3)
    doc.style_point("home", "ff0000ff", 1.1, "http://maps.google.com/mapfiles/kml/paddle/H.png")
    doc.style_point("start", "ff00ff00", 0.9, "http://maps.google.com/mapfiles/kml/paddle/grn-circle.png")
    doc.style_point("photo", "ffffffff", 0.35)
    for i in range(len(PALETTE)):
        doc.style_line(f"m{i}", PALETTE[i], 2.5)

    doc.polygon("Plot boundary", plan.polygon, "plot", fmt_area(plan.area_m2))
    doc.point("Take-off / home (planned)", plan.home[0], plan.home[1], "home")

    ci = 0
    for g in plan.grids:
        doc.folder(f"Grid {g.name}: {g.label} (heading {g.heading:.0f}°, camera {g.pitch:.0f}°)")
        for m in g.missions:
            pts = [(w.lat, w.lon) for w in m.waypoints]
            desc = (f"{m.label}\n{m.lines} lines, ~{m.est_photos} photos, ~{fmt_minutes(m.est_time_s)}, "
                    f"{m.distance_m / 1000:.2f} km")
            doc.line(f"{plan.name}_{m.name}", pts, f"m{ci % len(PALETTE)}", desc)
            doc.point(f"{m.name} start", pts[0][0], pts[0][1], "start", desc)
            ci += 1
        doc.end_folder()
        # approximate photo positions (hidden by default)
        doc.folder(f"Grid {g.name} photo positions (approx.)", visible=False)
        for m in g.missions:
            for a, b in m.segments:
                length = math.hypot(b[0] - a[0], b[1] - a[1])
                n = int(length / g.photo_spacing + 1e-6) + 1
                for k in range(n):
                    t = (k * g.photo_spacing) / length if length > 0 else 0.0
                    lat, lon = plan.proj.to_latlon(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
                    doc.point("", lat, lon, "photo", visible=False)
        doc.end_folder()
    for o in plan.orbits:
        doc.folder(f"Orbit {o.name} (r={o.radius:.0f} m)")
        doc.point(f"{o.name} centre", o.lat, o.lon, "start")
        for m in o.missions:
            pts = [(w.lat, w.lon) for w in m.waypoints]
            doc.line(f"{plan.name}_{m.name}", pts, f"m{ci % len(PALETTE)}",
                     f"{m.label}: {len(pts)} photos, ~{fmt_minutes(m.est_time_s)}")
            ci += 1
        doc.end_folder()
    return doc.save(path)


def write_json(plan: Plan, path: str) -> str:
    data = {
        "name": plan.name,
        "preset": plan.preset,
        "camera": plan.camera.key,
        "polygon": plan.polygon,
        "home": plan.home,
        "altitude": plan.altitude,
        "altitude_mode": plan.altitude_mode,
        "front": plan.front,
        "side": plan.side,
        "speed": plan.speed,
        "gsd_cm": plan.camera.gsd_m(plan.altitude) * 100,
        "grids": [
            {"name": g.name, "label": g.label, "heading": g.heading, "pitch": g.pitch,
             "line_spacing": g.line_spacing, "photo_spacing": g.photo_spacing, "lines": g.n_lines,
             "photos": g.photos, "missions": [m.name for m in g.missions]}
            for g in plan.grids
        ],
        "orbits": [
            {"name": o.name, "lat": o.lat, "lon": o.lon, "radius": o.radius,
             "rings": [{"alt": a, "pitch": p, "photos": n} for a, p, n in o.rings]}
            for o in plan.orbits
        ],
        "missions": [
            {"name": m.name, "kind": m.kind, "label": m.label, "waypoints": len(m.waypoints),
             "est_time_s": round(m.est_time_s), "est_photos": m.est_photos, "distance_m": round(m.distance_m)}
            for m in plan.missions
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path


def summary_text(plan: Plan, files: dict | None = None) -> str:
    cam = plan.camera
    across, along = cam.footprint_m(plan.altitude)
    lat_c = sum(p[0] for p in plan.polygon) / len(plan.polygon)
    lon_c = sum(p[1] for p in plan.polygon) / len(plan.polygon)
    L = []
    add = L.append
    add(f"FLIGHT PLAN: {plan.name}   (preset: {plan.preset})")
    add("=" * 72)
    add(PRESETS[plan.preset]["description"])
    add("")
    add(f"Plot area        : {fmt_area(plan.area_m2)}, perimeter {plan.perimeter_m:,.0f} m")
    add(f"Plot centre      : {lat_c:.6f}, {lon_c:.6f}  ({google_maps_link(lat_c, lon_c)})")
    add(f"Planned take-off : {plan.home[0]:.6f}, {plan.home[1]:.6f}")
    add(f"Camera           : {cam.name}")
    alt_txt = "above ground (Litchi terrain follow)" if plan.altitude_mode == "agl" else "above take-off point"
    add(f"Altitude         : {plan.altitude:.0f} m {alt_txt}")
    add(f"Ground detail    : {cam.gsd_m(plan.altitude) * 100:.2f} cm per pixel; photo covers "
        f"{across:.0f} m x {along:.0f} m")
    add(f"Overlap          : {plan.front * 100:.0f}% front, {plan.side * 100:.0f}% side")
    add(f"Speed            : {plan.speed:.1f} m/s"
        + ("  (stop-and-shoot: hover for every photo)" if plan.stop_and_shoot else
           "  (photos taken on the move, every N metres)"))
    add(f"Battery budget   : {plan.battery_s / 60:.0f} min flying per mission (change with --battery-minutes)")
    add("")
    total_photos = 0
    total_time = 0.0
    for g in plan.grids:
        add(f"Grid {g.name} - {g.label}: heading {g.heading:.0f}°, camera pitch {g.pitch:.0f}°, "
            f"{g.n_lines} lines {g.line_spacing:.1f} m apart, photo every {g.photo_spacing:.1f} m "
            f"({g.photo_spacing / plan.speed:.1f} s)")
        for m in g.missions:
            add(f"   {plan.name}_{m.name}.csv  {m.lines:2d} lines  {len(m.waypoints):3d} waypoints  "
                f"~{m.est_photos:4d} photos  ~{fmt_minutes(m.est_time_s)}  {m.distance_m / 1000:.2f} km")
            total_photos += m.est_photos
            total_time += m.est_time_s
    for o in plan.orbits:
        rings = ", ".join(f"{a:.0f} m @ {p:.0f}°" for a, p, _ in o.rings)
        add(f"Orbit {o.name} around {o.lat:.6f}, {o.lon:.6f}: radius {o.radius:.0f} m, rings {rings}")
        for m in o.missions:
            add(f"   {plan.name}_{m.name}.csv  {len(m.waypoints):3d} waypoints (stop + photo each)  "
                f"~{fmt_minutes(m.est_time_s)}")
            total_photos += m.est_photos
            total_time += m.est_time_s
    n_m = len(plan.missions)
    add("")
    add(f"TOTAL: {n_m} missions (= battery swaps), ~{total_photos} photos "
        f"(~{total_photos * 9 / 1024:.1f} GB on the SD card), ~{fmt_minutes(total_time)} of flying")
    if plan.warnings:
        add("")
        add("WARNINGS")
        for w in plan.warnings:
            add(f" - {w}")
    add("")
    add("LOAD INTO LITCHI (once, at home with internet)")
    add(" 1. Open https://flylitchi.com/hub on the Mac and log in with your Litchi account.")
    add(" 2. Missions > Import > pick one CSV. Check the path on the map, then open the mission")
    add("    settings and set: Heading Mode = Custom, Path Mode = Straight Lines,")
    add(f"    Finish Action = RTH, Cruising Speed = {plan.speed:.0f} m/s. Missions > Save (use the CSV name).")
    add(" 3. Repeat for each CSV. Open the preview KML in Google Earth Pro to see everything together.")
    add(" 4. In the Litchi app (Waypoints mode) open the saved missions once while online so they sync.")
    if not plan.stop_and_shoot:
        add(" 5. FIRST TIME ONLY: fly the first mission alone, then run 'dronemap photos' on its photos. If it")
        add("    did not take a photo every few metres along each line, re-plan with --stop-and-shoot.")
    add("")
    add("CAMERA SETTINGS (Litchi camera menu, before the first mission)")
    add(" - Photo: JPEG (not RAW), aspect 3:2, single shot.")
    add(" - Exposure: shutter 1/1000 s (1/800 if overcast); ISO auto capped at 400 (or fixed 100-200 in sun).")
    add(" - White balance: fixed (Sunny or Cloudy), not Auto. Same for all missions of one job.")
    add(" - Focus: hover at mission altitude, tap-focus on the ground once, then switch to MF so it stays put.")
    add("")
    add("FIELD CHECKLIST")
    add(" - Wind under ~20 km/h. Best light: overcast, or 10:00-14:00 for short shadows.")
    add(" - For terrain: bare ground beats standing crops (the model sees crop tops, not soil).")
    add(" - Failsafe (signal lost) = Return-to-Home; RTH altitude above the tallest tree.")
    add(" - Propellers: marked on marked motors, no chips. Batteries: check cell voltages.")
    add(" - Obstacle sensing does not protect you during missions - check the path in the KML.")
    if plan.orbits:
        add(" - Orbits fly LOW: walk around each orbit circle first; it must be clear of trees/wires")
        add("   at that height. Climb above the trees manually before starting an orbit mission.")
    add(" - After landing, before leaving the site: run  dronemap photos <SD card DCIM folder>")
    add("   to check coverage and spot gaps while you can still re-fly.")
    if files:
        add("")
        add("FILES")
        for p in files.get("csv", []):
            add(f"  {p}")
        add(f"  {files.get('kml')}   (Google Earth preview)")
        add(f"  {files.get('json')}   (used by 'dronemap photos' to check coverage)")
    return "\n".join(L) + "\n"
