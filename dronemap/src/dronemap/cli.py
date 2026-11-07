"""Command line interface: `dronemap <command> ...`  (run `dronemap -h`)."""
from __future__ import annotations

import argparse
import os
import sys

from . import __version__
from .util import UserError, fail, info, ok, step, warn

EPILOG = """
Typical workflow
  1. dronemap plan bagrod.kml --preset survey      -> ~/drone/plans/bagrod: Litchi missions + preview
  2. fly the missions, copy photos to  ~/drone/projects/bagrod/images/
  3. dronemap photos ~/drone/projects/bagrod/images --plan ~/drone/plans/bagrod/bagrod_plan.json
  4. dronemap process ~/drone/projects/bagrod --preset survey  -> orthophoto, DSM/DTM, 3D model (ODM)
  5. dronemap site ~/drone/projects/bagrod                     -> contours, drainage, low spots
  6. dronemap pond ~/drone/projects/bagrod --outline pond.kml  -> volumes, bunds, sections, 3D before/after

  High-detail model + Gaussian splat (Tigadda):
     dronemap plan tigadda.kml --preset detail --orbit LAT,LON
     dronemap process ~/drone/projects/tigadda --preset detail --splat
"""


def _add_plan(sub):
    p = sub.add_parser("plan", help="Create Litchi mapping missions for a plot boundary",
                       description="Create battery-sized Litchi waypoint missions (CSV) for mapping a plot, "
                                   "plus a Google Earth preview and a summary with camera settings.")
    p.add_argument("boundary", help="Plot boundary: KML/KMZ from Google Earth, GeoJSON, or a text file of "
                                    "'lat, lon' corner lines")
    p.add_argument("--preset", choices=["survey", "detail", "quick"], default="survey",
                   help="survey = terrain/earthworks (default); detail = high-res 3D/splat; quick = scouting")
    p.add_argument("--name", help="Job name (default: boundary file name)")
    p.add_argument("--out", help="Output folder (default: ~/drone/plans/<name>)")
    p.add_argument("--polygon-name", help="Use the polygon with this name if the file has several")
    p.add_argument("--altitude", type=float, help="Flight height in metres (preset default: survey 80, detail 45)")
    p.add_argument("--front", type=float, help="Front overlap 0-1 (survey 0.75, detail 0.80)")
    p.add_argument("--side", type=float, help="Side overlap 0-1 (survey 0.70, detail 0.75)")
    p.add_argument("--speed", type=float, help="Speed in m/s (survey 4, detail 3)")
    p.add_argument("--heading", type=float, help="Flight-line direction in degrees (default: auto, fewest lines)")
    p.add_argument("--buffer", type=float, help="Extra margin flown outside the boundary, metres (default: auto)")
    p.add_argument("--battery-minutes", type=float, default=12.0,
                   help="Max flying minutes per mission (default 12, conservative for older batteries)")
    p.add_argument("--home", help="Planned take-off point 'lat,lon' (default: plot centre)")
    p.add_argument("--agl", action="store_true",
                   help="Altitude above ground (Litchi terrain follow) instead of above take-off point")
    p.add_argument("--grids", help="Only these grids, e.g. 'A' or 'A,B'")
    p.add_argument("--oblique-pitch", type=float, help="Gimbal pitch for oblique grids (survey -70, detail -60)")
    p.add_argument("--stop-and-shoot", action="store_true",
                   help="Hover for every photo (one waypoint per photo): sharpest and most robust, 2-3x more flying")
    p.add_argument("--orbit", action="append", default=[], metavar="LAT,LON[,RADIUS[,TARGET_H]]",
                   help="Add an orbit (circle of photos) around a building/tree. Repeatable.")
    p.add_argument("--orbit-file", help="KML/text file of points to orbit")
    p.add_argument("--orbit-radius", type=float, default=25.0, help="Default orbit radius in metres (25)")
    p.add_argument("--orbit-alts", default="20,35", help="Orbit heights in metres, comma separated (20,35)")
    p.add_argument("--orbit-photos", type=int, default=36, help="Photos per orbit ring (36 = every 10°)")
    p.add_argument("--orbit-target-height", type=float, default=2.0,
                   help="Height the camera aims at, above take-off level (2 m)")
    p.set_defaults(func=cmd_plan)


def cmd_plan(a) -> int:
    from .camera import get_camera, DEFAULT_CAMERA
    from .flightplan import build_plan
    from .geo import parse_latlon, read_points, read_polygon
    from .plan_outputs import summary_text, write_plan

    pname, poly = read_polygon(a.boundary, a.polygon_name)
    name = a.name or os.path.splitext(os.path.basename(a.boundary))[0]
    name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
    out = a.out or os.path.join(os.path.expanduser("~"), "drone", "plans", name)
    home = parse_latlon(a.home) if a.home else None
    orbits = []
    for spec in a.orbit:
        parts = [s.strip() for s in spec.split(",") if s.strip()]
        if len(parts) < 2:
            raise UserError(f"--orbit needs at least 'lat,lon' (got '{spec}')")
        lat, lon = parse_latlon(",".join(parts[:2]))
        radius = float(parts[2]) if len(parts) > 2 else a.orbit_radius
        tgt = float(parts[3]) if len(parts) > 3 else None
        orbits.append((lat, lon, radius, tgt))
    if a.orbit_file:
        for _, lat, lon in read_points(a.orbit_file):
            orbits.append((lat, lon, a.orbit_radius, None))
    alts = [float(x) for x in a.orbit_alts.split(",") if x.strip()]
    step(f"Planning '{name}' from {a.boundary}" + (f" (polygon '{pname}')" if pname else ""))
    plan = build_plan(name, poly, get_camera(DEFAULT_CAMERA), preset=a.preset, altitude=a.altitude,
                      front=a.front, side=a.side, speed=a.speed, heading=a.heading, buffer=a.buffer,
                      battery_minutes=a.battery_minutes, home=home,
                      altitude_mode="agl" if a.agl else "relative", grids=a.grids,
                      oblique_pitch=a.oblique_pitch, orbits=orbits, orbit_alts=alts,
                      orbit_photos=a.orbit_photos, orbit_target_height=a.orbit_target_height,
                      stop_and_shoot=a.stop_and_shoot)
    files = write_plan(plan, out)
    info("")
    info(summary_text(plan, files))
    ok(f"Plan written to {os.path.abspath(out)}")
    return 0


def _add_photos(sub):
    p = sub.add_parser("photos", help="Check photo coverage after flying (run in the field)",
                       description="Read GPS from the photos, map where they were taken, flag gaps, "
                                   "and write a coverage map (PNG) and KML.")
    p.add_argument("images", help="Folder with the drone photos (e.g. the SD card DCIM folder)")
    p.add_argument("--plan", help="The *_plan.json from 'dronemap plan' to compare against")
    p.add_argument("--out", help="Output folder (default: <images>/../photo_check)")
    p.set_defaults(func=cmd_photos)


def cmd_photos(a) -> int:
    from .photos import check_photos

    return check_photos(a.images, a.plan, a.out)


def _add_process(sub):
    p = sub.add_parser("process", help="Run OpenDroneMap on a project folder (Docker)",
                       description="Process PROJECT/images with OpenDroneMap in Docker. Outputs stay in PROJECT.")
    p.add_argument("project", help="Project folder containing an 'images' sub-folder")
    p.add_argument("--preset", choices=["survey", "detail"], default="survey",
                   help="survey = orthophoto + DSM/DTM for earthworks; detail = max 3D detail + splat-ready")
    p.add_argument("--quality", choices=["fast", "standard", "max"], default="standard",
                   help="fast = quick check, standard (default), max = slowest / most detail")
    p.add_argument("--splat", action="store_true", help="Train a Gaussian splat with OpenSplat afterwards")
    p.add_argument("--gcp", help="Ground control file (ODM gcp_list.txt format). Auto-used if PROJECT/gcp_list.txt exists")
    p.add_argument("--boundary", help="Crop outputs to this KML/GeoJSON polygon")
    p.add_argument("--no-rolling-shutter", action="store_true",
                   help="Turn off rolling-shutter correction (only if every photo was taken while hovering)")
    p.add_argument("--sky-removal", action="store_true", help="Mask sky in oblique photos (needs internet once)")
    p.add_argument("--rerun-from", help="Restart from an ODM stage (e.g. openmvs, odm_meshing, odm_dem)")
    p.add_argument("--odm-args", default="", help="Extra raw ODM arguments, quoted")
    p.add_argument("--image", default="opendronemap/odm:latest", help="ODM Docker image")
    p.add_argument("--dry-run", action="store_true", help="Print the Docker command without running it")
    p.set_defaults(func=cmd_process)


def cmd_process(a) -> int:
    from .odm import run_process

    rc = run_process(a)
    if rc == 0 and a.splat and not a.dry_run:
        from .splat import run_splat

        ns = argparse.Namespace(project=a.project, iters=30000, downscale="auto", max_gaussians=None,
                                name="splat", resume=False, opensplat=None, dry_run=False)
        rc = run_splat(ns)
    return rc


def _add_splat(sub):
    p = sub.add_parser("splat", help="Train a Gaussian splat (OpenSplat) from a processed project",
                       description="Train a 3D Gaussian splat from an ODM-processed project using OpenSplat "
                                   "(Apple Metal GPU on the Mac).")
    p.add_argument("project", help="Project folder already processed with 'dronemap process'")
    p.add_argument("--iters", type=int, default=30000, help="Training steps (30000 default; 7000 for a preview)")
    p.add_argument("--downscale", default="auto", help="Image downscale factor (auto, 1 = full 20 MP, 2, 4)")
    p.add_argument("--max-gaussians", type=int, help="Cap on splat count (auto from memory)")
    p.add_argument("--name", default="splat", help="Output name (splat -> PROJECT/dronemap/splat/splat.ply)")
    p.add_argument("--resume", action="store_true", help="Continue from the last saved checkpoint")
    p.add_argument("--opensplat", help="Path to the opensplat binary (default: auto-detect)")
    p.add_argument("--dry-run", action="store_true", help="Prepare inputs and print the command only")
    p.set_defaults(func=cmd_splat)


def cmd_splat(a) -> int:
    from .splat import run_splat

    return run_splat(a)


def _add_site(sub):
    p = sub.add_parser("site", help="Terrain analysis: contours, slope, drainage, low spots",
                       description="Analyse the terrain model (DTM): hillshade, slope, contours, where water "
                                   "flows and collects, and where it leaves the plot.")
    p.add_argument("source", help="Processed project folder, or a DTM GeoTIFF")
    p.add_argument("--boundary", help="Plot boundary (KML/GeoJSON/text) to clip the analysis and stats")
    p.add_argument("--res", type=float, default=0.5, help="Analysis grid in metres (0.5)")
    p.add_argument("--contour-interval", type=float, default=0.5, help="Contour spacing in metres (0.5)")
    p.add_argument("--stream-area", type=float, default=2000.0,
                   help="Catchment (m²) above which a flow path is drawn as a drainage line (2000)")
    p.add_argument("--rainfall-mm", type=float, default=1100.0, help="Annual rainfall for runoff estimates (1100)")
    p.add_argument("--runoff", type=float, default=0.3, help="Runoff coefficient (0.3 = farmland)")
    p.add_argument("--out", help="Output folder (default: PROJECT/dronemap/site)")
    p.set_defaults(func=cmd_site)


def cmd_site(a) -> int:
    from .site import run_site

    return run_site(a)


def _add_pond(sub):
    p = sub.add_parser("pond", help="Design a pond: cut/fill, bunds, capacity, sections, 3D before/after",
                       description="Carve a pond into the terrain model and report excavation, bund fill, "
                                   "water capacity, earth balance, and cost. Produces maps, cross-sections, "
                                   "3D models (GLB), a Google Earth overlay and a designed-terrain GeoTIFF.")
    p.add_argument("source", help="Processed project folder, or a DTM GeoTIFF")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--outline", help="Pond top edge drawn as a polygon (KML/GeoJSON/text)")
    g.add_argument("--center", help="Pond centre 'lat,lon' (rectangular pond, use with --size)")
    p.add_argument("--size", default="40x30", help="Rectangle LENGTHxWIDTH in metres with --center (40x30)")
    p.add_argument("--rotation", type=float, default=0.0, help="Rectangle long-axis bearing in degrees (0 = north)")
    p.add_argument("--depth", default="3",
                   help="Floor depth (m) below the LOWEST point of the pond edge. Comma list compares options: 2.5,3,3.5")
    p.add_argument("--floor-level", type=float, help="Absolute floor elevation (overrides --depth)")
    p.add_argument("--side-slope", type=float, default=2.0, help="Inner side slope, horizontal per 1 vertical (2)")
    p.add_argument("--freeboard", type=float, default=0.5, help="Gap between full water level and the rim (0.5 m)")
    p.add_argument("--bund-height", type=float, default=0.0,
                   help="Raise the rim with an earth bund this much above the lowest edge point (0 = dug-out only)")
    p.add_argument("--bund-crest", type=float, default=2.0, help="Bund top width in metres (2)")
    p.add_argument("--bund-outer-slope", type=float, default=2.0, help="Bund outer slope H:V (2)")
    p.add_argument("--res", type=float, default=0.25, help="Calculation grid in metres (0.25)")
    p.add_argument("--swell", type=float, default=1.25, help="Loose/bank volume ratio of dug soil (1.25)")
    p.add_argument("--compaction", type=float, default=0.90, help="Compacted/bank ratio for bund fill (0.90)")
    p.add_argument("--trolley-m3", type=float, default=3.0, help="Tractor-trolley load, loose m³ (3)")
    p.add_argument("--excavator-rate", type=float, default=60.0, help="Excavator output, bank m³ per hour (60)")
    p.add_argument("--cost-per-hour", type=float, help="Excavator hire cost per hour (₹) to estimate cost")
    p.add_argument("--rainfall-mm", type=float, default=1100.0, help="Annual rainfall for inflow estimate (1100)")
    p.add_argument("--runoff", type=float, default=0.3, help="Runoff coefficient for the catchment (0.3)")
    p.add_argument("--stl", action="store_true", help="Also write a 3D-printable STL of the designed terrain")
    p.add_argument("--exaggeration", type=float, default=2.0, help="Vertical exaggeration for the STL (2)")
    p.add_argument("--name", default="pond", help="Design name (output folder name)")
    p.add_argument("--out", help="Output folder (default: PROJECT/dronemap/<name>)")
    p.set_defaults(func=cmd_pond)


def cmd_pond(a) -> int:
    from .pond import run_pond

    return run_pond(a)


def _add_blender(sub):
    p = sub.add_parser("to-blender", help="Export terrain as a textured OBJ for hand-sculpting in Blender")
    p.add_argument("source", help="Processed project folder, or a DTM GeoTIFF")
    p.add_argument("--area", help="Polygon (KML/GeoJSON/text) limiting the exported area (default: whole DTM)")
    p.add_argument("--res", type=float, default=0.5, help="Mesh grid spacing in metres (0.5)")
    p.add_argument("--margin", type=float, default=10.0, help="Margin around --area in metres (10)")
    p.add_argument("--out", help="Output folder (default: PROJECT/dronemap/blender)")
    p.set_defaults(func=cmd_to_blender)

    q = sub.add_parser("from-blender", help="Measure cut/fill of a terrain you sculpted in Blender")
    q.add_argument("source", help="Same project folder / DTM used for to-blender")
    q.add_argument("edited", help="OBJ exported from Blender (default export settings)")
    q.add_argument("--georef", help="terrain_georef.json written by to-blender (default: next to the OBJ)")
    q.add_argument("--swell", type=float, default=1.25)
    q.add_argument("--compaction", type=float, default=0.90)
    q.add_argument("--out", help="Output folder (default: next to the edited OBJ)")
    q.set_defaults(func=cmd_from_blender)


def cmd_to_blender(a) -> int:
    from .blender import to_blender

    return to_blender(a)


def cmd_from_blender(a) -> int:
    from .blender import from_blender

    return from_blender(a)


def _add_doctor(sub):
    p = sub.add_parser("doctor", help="Check that Docker, ODM, OpenSplat and Python packages are ready")
    p.set_defaults(func=cmd_doctor)


def cmd_doctor(a) -> int:
    from .doctor import run_doctor

    return run_doctor()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="dronemap", description="Local drone mapping toolkit: plan Litchi missions, "
                                 "process with OpenDroneMap, train Gaussian splats, design ponds.",
                                 epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"dronemap {__version__}")
    sub = ap.add_subparsers(dest="command", metavar="<command>")
    _add_plan(sub)
    _add_photos(sub)
    _add_process(sub)
    _add_splat(sub)
    _add_site(sub)
    _add_pond(sub)
    _add_blender(sub)
    _add_doctor(sub)
    return ap


def main(argv=None) -> int:
    ap = build_parser()
    a = ap.parse_args(argv)
    if not getattr(a, "func", None):
        ap.print_help()
        return 1
    try:
        return int(a.func(a) or 0)
    except UserError as e:
        fail(str(e))
        return 2
    except ValueError as e:
        fail(str(e))
        return 2
    except KeyboardInterrupt:
        warn("Stopped by user.")
        return 130
    except ImportError as e:
        fail(f"{e}. Run the setup script (setup/setup_mac.sh) to install the Python packages.")
        return 3


if __name__ == "__main__":
    sys.exit(main())
