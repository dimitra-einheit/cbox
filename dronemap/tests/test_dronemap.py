"""Tests for dronemap. Run from the dronemap folder:

    PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import math
import os
import re
import struct
import tempfile
import unittest

import numpy as np

from dronemap.camera import get_camera
from dronemap.flightplan import build_plan, _balanced_split
from dronemap.geo import LocalProjection, read_polygon, read_points
from dronemap.litchi import HEADER, N_COLUMNS, waypoint_row
from dronemap.plan_outputs import write_plan

# Row format check taken from the litchi-wp package (validates Litchi Mission Hub CSV lines)
LITCHI_RX = re.compile(
    r"^[-]?\d+(\.\d+)?,[-]?\d+(\.\d+)?,\d+(\.\d+)?,[-]?\d+(\.\d+)?,[-]?\d+(\.\d+)?,[0-1]+,([-1]|[0-2]),"
    r"[-]?\d+(\.\d+)?,((-1|[0-5]),[-]?\d+(\.\d+)?,){15}[0-1],\d+(\.\d+)?,[-]?\d+(\.\d+)?,[-]?\d+(\.\d+)?,"
    r"\d+(\.\d+)?,[-]?[0-1],((-1|\d+)(\.\d+)?),((-1|\d+)(\.\d+)?)$"
)

# ~14-acre quadrilateral near Rahatgarh (synthetic test plot)
PLOT = [(23.78000, 78.37800), (23.78040, 78.38050), (23.78230, 78.38080), (23.78210, 78.37820)]
CAM = get_camera("mavic2pro")


class TestFlightPlan(unittest.TestCase):
    def test_gsd_mavic2pro(self):
        self.assertAlmostEqual(CAM.gsd_m(80) * 100, 1.881, places=2)
        across, along = CAM.footprint_m(80)
        self.assertAlmostEqual(across, 102.9, delta=0.2)
        self.assertAlmostEqual(along, 68.6, delta=0.2)

    def test_survey_plan_csv_is_valid_litchi(self):
        plan = build_plan("t", PLOT, CAM, preset="survey")
        with tempfile.TemporaryDirectory() as d:
            files = write_plan(plan, d)
            self.assertTrue(files["csv"])
            for p in files["csv"]:
                with open(p) as f:
                    lines = f.read().splitlines()
                self.assertEqual(lines[0], HEADER)
                self.assertEqual(len(lines[0].split(",")), N_COLUMNS)
                for ln in lines[1:]:
                    self.assertEqual(len(ln.split(",")), N_COLUMNS)
                    self.assertRegex(ln, LITCHI_RX)
                self.assertLessEqual(len(lines) - 1, 99)
            with open(files["json"]) as f:
                self.assertEqual(json.load(f)["preset"], "survey")

    def test_missions_fit_battery_and_cover_lines(self):
        plan = build_plan("t", PLOT, CAM, preset="survey", battery_minutes=12)
        for g in plan.grids:
            self.assertEqual(sum(m.lines for m in g.missions), g.n_lines)
            for m in g.missions:
                self.assertLessEqual(m.est_time_s, 12 * 60 + 1)
                starts = m.waypoints[::2]
                self.assertTrue(all(w.dist_interval > 0 for w in starts))
                self.assertTrue(all(w.dist_interval < 0 for w in m.waypoints[1::2]))

    def test_grid_covers_polygon(self):
        plan = build_plan("t", PLOT, CAM, preset="survey")
        across, _ = CAM.footprint_m(plan.altitude)
        g = plan.grids[0]
        segs = [(m.waypoints[i], m.waypoints[i + 1]) for m in g.missions for i in range(0, len(m.waypoints), 2)]
        for lat, lon in PLOT:
            px, py = plan.proj.to_xy(lat, lon)
            covered = 0
            for a, b in segs:
                dx, dy = b.x - a.x, b.y - a.y
                L = math.hypot(dx, dy)
                t = ((px - a.x) * dx + (py - a.y) * dy) / L / L
                dist = abs((px - a.x) * dy - (py - a.y) * dx) / L
                if 0 <= t <= 1 and dist <= across / 2:
                    covered += 1
            self.assertGreaterEqual(covered, 2, "each corner should be seen from at least two lines")

    def test_balanced_split_avoids_tiny_last_mission(self):
        items = list(range(11))
        groups = _balanced_split(items, lambda g: 60 + 100 * len(g), 500, 49)
        self.assertEqual(sum(len(g) for g in groups), 11)
        sizes = [len(g) for g in groups]
        self.assertLessEqual(max(sizes) - min(sizes), 1)

    def test_orbit_points_at_centre(self):
        c = (23.7810, 78.3790)
        plan = build_plan("t", PLOT, CAM, preset="detail", orbits=[(c[0], c[1], 25.0, None)], grids="A")
        o = plan.orbits[0]
        self.assertEqual(len(o.missions[0].waypoints), 72)
        cx, cy = plan.proj.to_xy(*c)
        for w in o.missions[0].waypoints:
            self.assertAlmostEqual(math.hypot(w.x - cx, w.y - cy), 25.0, delta=0.01)
            to_c = math.degrees(math.atan2(cx - w.x, cy - w.y)) % 360
            self.assertAlmostEqual((w.heading - to_c + 180) % 360 - 180, 0, delta=0.1)
            self.assertLess(w.pitch, -20)
            self.assertRegex(waypoint_row(w), LITCHI_RX)

    def test_stop_and_shoot_one_waypoint_per_photo(self):
        moving = build_plan("t", PLOT, CAM, preset="detail", grids="A")
        hover = build_plan("t", PLOT, CAM, preset="detail", grids="A", stop_and_shoot=True)
        self.assertEqual(hover.grids[0].n_lines, moving.grids[0].n_lines)
        n_wp = sum(len(m.waypoints) for m in hover.grids[0].missions)
        self.assertEqual(n_wp, hover.grids[0].photos)
        self.assertEqual(hover.grids[0].photos, moving.grids[0].photos)
        for m in hover.grids[0].missions:
            self.assertLessEqual(len(m.waypoints), 99)
            for w in m.waypoints:
                self.assertIn((1, 0), w.actions)  # take-photo action at every waypoint
                self.assertEqual(w.dist_interval, -1.0)
                self.assertRegex(waypoint_row(w), LITCHI_RX)
        self.assertGreater(sum(m.est_time_s for m in hover.missions), 1.5 * sum(m.est_time_s for m in moving.missions))

    def test_speed_capped_by_photo_interval(self):
        plan = build_plan("t", PLOT, CAM, preset="detail", speed=8.0, grids="A")
        spacing = plan.grids[0].photo_spacing
        self.assertGreaterEqual(spacing / plan.speed, CAM.min_photo_interval_s - 1e-6)


class TestBoundaryFiles(unittest.TestCase):
    def test_kml_geojson_text(self):
        with tempfile.TemporaryDirectory() as d:
            kml = os.path.join(d, "b.kml")
            coords = " ".join(f"{lon},{lat},0" for lat, lon in PLOT + [PLOT[0]])
            with open(kml, "w") as f:
                f.write('<?xml version="1.0"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document><Placemark>'
                        f"<name>Bagrod</name><Polygon><outerBoundaryIs><LinearRing><coordinates>{coords}"
                        "</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark></Document></kml>")
            name, ring = read_polygon(kml)
            self.assertEqual(name, "Bagrod")
            self.assertEqual(len(ring), 4)
            self.assertAlmostEqual(ring[0][0], PLOT[0][0])
            gj = os.path.join(d, "b.geojson")
            with open(gj, "w") as f:
                json.dump({"type": "Feature", "properties": {}, "geometry": {
                    "type": "Polygon", "coordinates": [[[lon, lat] for lat, lon in PLOT + [PLOT[0]]]]}}, f)
            self.assertEqual(len(read_polygon(gj)[1]), 4)
            txt = os.path.join(d, "b.txt")
            with open(txt, "w") as f:
                f.write("# corners\n" + "\n".join(f"{lat}, {lon}" for lat, lon in PLOT))
            self.assertAlmostEqual(read_polygon(txt)[1][2][1], PLOT[2][1])
            pts = os.path.join(d, "p.txt")
            with open(pts, "w") as f:
                f.write("23.7810, 78.3790\n23.7812, 78.3792\n")
            self.assertEqual(len(read_points(pts)), 2)

    def test_projection_roundtrip(self):
        p = LocalProjection(23.78, 78.38)
        x, y = p.to_xy(23.781, 78.381)
        self.assertAlmostEqual(y, 110.75, delta=0.2)  # 0.001 deg latitude ~ 110.75 m here
        lat, lon = p.to_latlon(x, y)
        self.assertAlmostEqual(lat, 23.781, places=9)
        self.assertAlmostEqual(lon, 78.381, places=9)


class TestPond(unittest.TestCase):
    def setUp(self):
        self.res = 0.25
        h, w = int(70 / self.res), int(80 / self.res)
        self.z = np.full((h, w), 100.0)
        ys, xs = np.mgrid[0:h, 0:w]
        cx, cy = (xs + 0.5) * self.res, (ys + 0.5) * self.res
        self.inside = (np.abs(cx - 40) < 20) & (np.abs(cy - 35) < 15)  # 40 x 30 m

    def test_frustum_volumes_match_prismoidal_formula(self):
        from dronemap.pond import design_pond

        r = design_pond(self.z, self.res, self.inside, depth=3.0, side_slope=2.0, freeboard=0.5)
        L, W, s, D = 40, 30, 2, 3
        Ab = (L - 2 * s * D) * (W - 2 * s * D)
        V = D / 6 * (L * W + Ab + 4 * (L - s * D) * (W - s * D))
        self.assertAlmostEqual(r["cut_m3"], V, delta=V * 0.002)
        d = 2.5
        Lb, Wb = L - 2 * s * D, W - 2 * s * D
        Vw = d / 6 * (Ab + (Lb + 2 * s * d) * (Wb + 2 * s * d) + 4 * (Lb + s * d) * (Wb + s * d))
        self.assertAlmostEqual(r["capacity_m3"], Vw, delta=Vw * 0.002)
        self.assertAlmostEqual(r["floor_area_m2"], Ab, delta=1)
        self.assertEqual(r["fill_m3"], 0.0)
        self.assertAlmostEqual(r["fsl"], 99.5)

    def test_bund_raises_level_and_needs_fill(self):
        from dronemap.pond import design_pond, earth_balance

        r0 = design_pond(self.z, self.res, self.inside, depth=3.0)
        r1 = design_pond(self.z, self.res, self.inside, depth=3.0, bund_height=1.0)
        self.assertAlmostEqual(r1["fsl"], r0["fsl"] + 1.0)
        self.assertGreater(r1["fill_m3"], 400)
        self.assertGreater(r1["capacity_m3"], r0["capacity_m3"])
        eb = earth_balance(r1, 1.25, 0.9, 3.0, 60.0, 1800.0)
        self.assertAlmostEqual(eb["spoil_bank_m3"], r1["cut_m3"] - r1["fill_m3"] / 0.9)
        self.assertAlmostEqual(eb["cost"], r1["cut_m3"] / 60.0 * 1800.0)

    def test_sloping_ground_digs_deeper_on_high_side(self):
        from dronemap.pond import design_pond

        h, w = self.z.shape
        z = 100.0 + np.linspace(0, 2.0, w)[None, :].repeat(h, axis=0)  # rises 2 m to the east
        r = design_pond(z, self.res, self.inside, depth=3.0)
        self.assertGreater(r["max_cut_m"], 3.5)
        self.assertLess(r["lowest_rim"], r["highest_rim"])


class TestHydro(unittest.TestCase):
    def test_flood_accumulation_catchment(self):
        from dronemap import hydro

        h, w = 40, 30
        rows = np.arange(h)[:, None]
        z = 50.0 - 0.1 * rows + np.zeros((h, w)) + 0.02 * np.abs(np.arange(w) - 15)[None, :]  # valley sloping south
        z[20, 15] -= 1.0  # a 1 m pit; it spills over the lower downstream neighbour (row 21)
        filled = hydro.priority_flood(z)
        self.assertAlmostEqual(filled[20, 15], z[21, 15], delta=1e-3)
        rec = hydro.d8_receivers(filled, 1.0)
        acc, order = hydro.accumulate(filled, rec, 1.0)
        self.assertAlmostEqual(acc.max(), acc[-1, :].max())
        self.assertGreater(acc[-1, 15], 300)  # valley floor collects most of the area
        target = np.zeros_like(z, dtype=bool)
        target[-1, :] = True
        c = hydro.catchment(target, rec, order)
        self.assertGreater(c.sum(), 0.6 * h * w)
        dep, lst = hydro.depressions(z, filled, 1.0, min_depth=0.05, min_area=0.5)
        self.assertTrue(any(abs(d["row"] - 20) < 1 and abs(d["col"] - 15) < 1 for d in lst))


class TestMeshes(unittest.TestCase):
    def test_glb_structure(self):
        from dronemap.mesh import GlbBuilder, heightfield

        z = np.random.default_rng(0).normal(0, 0.2, (20, 30)) + 100
        v, f, mx, my = heightfield(z, 0.5, 0.0, 10.0, (0, 0, 100))
        g = GlbBuilder()
        g.mesh("t", v, f, g.material("m", texture=g.texture(np.zeros((8, 8, 3), np.uint8))),
               uv=np.zeros((len(v), 2)))
        with tempfile.TemporaryDirectory() as d:
            p = g.save(os.path.join(d, "t.glb"))
            with open(p, "rb") as fh:
                data = fh.read()
        magic, ver, length = struct.unpack("<III", data[:12])
        self.assertEqual((magic, ver, length), (0x46546C67, 2, len(data)))
        jlen, jtype = struct.unpack("<II", data[12:20])
        js = json.loads(data[20:20 + jlen])
        self.assertEqual(js["accessors"][0]["count"], 600)
        self.assertEqual(js["accessors"][-1]["count"], len(f) * 3)
        self.assertEqual(len(f), 2 * 19 * 29)

    def test_stl_is_watertight(self):
        from dronemap.mesh import write_stl_terrain

        z = np.random.default_rng(1).normal(0, 0.3, (25, 35)).cumsum(0) / 5 + 100
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.stl")
            write_stl_terrain(p, z, 0.5)
            with open(p, "rb") as fh:
                data = fh.read()
        n = struct.unpack("<I", data[80:84])[0]
        tris = np.frombuffer(data[84:], dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])["v"]
        self.assertEqual(len(tris), n)
        edges = {}
        for t in np.round(tris.astype(np.float64), 4):
            pts = [tuple(p) for p in t]
            for a, b in ((0, 1), (1, 2), (2, 0)):
                key = tuple(sorted((pts[a], pts[b])))
                edges[key] = edges.get(key, 0) + 1
        self.assertTrue(all(c == 2 for c in edges.values()), "every edge must be shared by two triangles")
        # positive signed volume = outward-facing normals
        vol = np.einsum("ij,ij->i", tris[:, 0], np.cross(tris[:, 1], tris[:, 2])).sum() / 6
        self.assertGreater(vol, 0)


class TestPhotos(unittest.TestCase):
    def test_reads_dji_gps_and_xmp(self):
        from PIL import Image

        from dronemap.photos import read_photo_meta

        img = Image.new("RGB", (64, 48), (90, 120, 60))
        ex = Image.Exif()
        gps = {1: "N", 2: (23.0, 46.0, 48.0), 3: "E", 4: (78.0, 22.0, 30.0), 6: 512.5}
        ex[0x8825] = gps
        ex[271] = "Hasselblad"
        ex[272] = "L1D-20c"
        xmp = (b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
               b'<rdf:Description xmlns:drone-dji="http://www.dji.com/drone-dji/1.0/" '
               b'drone-dji:RelativeAltitude="+80.10" drone-dji:GimbalPitchDegree="-90.00" '
               b'drone-dji:GimbalYawDegree="+45.30"/></rdf:RDF></x:xmpmeta>')
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "DJI_0001.JPG")
            img.save(p, exif=ex, xmp=xmp)
            m = read_photo_meta(p)
        self.assertAlmostEqual(m["lat"], 23 + 46 / 60 + 48 / 3600, places=6)
        self.assertAlmostEqual(m["lon"], 78 + 22 / 60 + 30 / 3600, places=6)
        self.assertAlmostEqual(m["rel_alt"], 80.1)
        self.assertAlmostEqual(m["gimbal_pitch"], -90.0)
        self.assertAlmostEqual(m["gimbal_yaw"], 45.3)
        self.assertEqual(m["model"], "L1D-20c")


if __name__ == "__main__":
    unittest.main()
