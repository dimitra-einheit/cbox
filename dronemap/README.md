# dronemap: drone survey toolkit for Bagrod and Tigadda

Everything runs on your own Mac or PC. After setup, nothing needs internet, apart from loading missions into
Litchi and the one-time downloads.

| Step | Command | What you get |
|---|---|---|
| Plan | `dronemap plan` | Litchi waypoint missions (CSV), sized to your batteries; a Google Earth preview; camera settings |
| Check in the field | `dronemap photos` | Coverage map that shows gaps *before* you leave the site |
| Process | `dronemap process` | OpenDroneMap: orthophoto, terrain model (DTM), surface model (DSM), point cloud, textured 3D model (GLB/OBJ), quality report |
| Splat | `dronemap splat` | Gaussian splat (photoreal 3D) trained on the Mac's GPU with OpenSplat |
| Site | `dronemap site` | Contours (DXF for CAD, Google Earth, GeoJSON), slope classes, drainage lines, natural low spots, where water leaves the plot |
| Pond | `dronemap pond` | Excavation and bund volumes, water capacity, stage-volume table, earth balance, cost, cross-sections, 3D before/after, Google Earth overlay, optional 3D-print STL |
| Sculpt | `dronemap to-blender` / `from-blender` | Shape the ground by hand in Blender, then get the cut/fill volumes |

**Gaussian splat vs 3D model.** The splat is the photoreal "walk-through" view, best for showing the
Sanctuary. The *measurable* 3D model (mesh, terrain, volumes) comes straight from photogrammetry, which
is more accurate than converting a splat into a mesh. Both come from the same camera solution, so they line up.

---

## 1. Setup (once)

### Mac (main machine: M1 Max, 64 GB)
1. Get the code: on GitHub use **Code > Download ZIP** and unzip it to `~/cbox` (or `git clone` the repository).
   Then in **Terminal**: `cd ~/cbox/dronemap`.
2. Run `setup/setup_mac.sh`. It installs Homebrew (if missing), a private Python environment,
   **Docker Desktop**, and the **OpenDroneMap** engine (~2 GB). It also adds the `dronemap` command.
3. In **Docker Desktop > Settings > Resources** set **Memory 48 GB, CPUs 10, Swap 4 GB**, then *Apply & restart*.
4. For Gaussian splats: install **Xcode** from the App Store (free, needed for the Apple-GPU build), open it once,
   then run `setup/setup_mac.sh --opensplat` (~20 min build).
5. Check: `dronemap doctor`. Everything should say OK.
6. Try it: `dronemap plan examples/example_plot.kml`. It plans a made-up 14-acre plot and prints the summary.

Docker Desktop is free for personal use and for small businesses (under 250 people and under $10 M revenue).

### Windows PC
See [setup/SETUP_WINDOWS.md](setup/SETUP_WINDOWS.md). The PC can plan, process and run the terrain/pond tools.
Splats are Mac-only: the Vega 64 isn't supported by the splat engine.

### Apps you'll also use (free unless noted)
- **Litchi** for DJI, on your phone (paid once). It flies the missions; the Mavic 2 Pro has no built-in mapping mode.
- **Google Earth Pro** (Mac) to draw boundaries and pond outlines, and to view results.
- Optional: **QGIS** (maps/GeoTIFFs), **Blender** (3D/sculpting), **CloudCompare** (point clouds),
  **Bambu Studio** (prints the STL terrain model on the A1).

---

## 2. Draw the boundary

In **Google Earth Pro**: *Add > Polygon*, click around the plot corners, name it (e.g. *Bagrod*), OK.
Right-click it in *Places* > **Save Place As... > KML**. You can also use a text file with one
`lat, lon` corner per line (copy them from Google Maps).

---

## 3. Bagrod: 15-acre terrain survey, then pond design

### Plan
```bash
dronemap plan ~/maps/bagrod.kml --preset survey --home 23.xxxxx,78.xxxxx
```
`--home` is where you'll take off. Use the **highest practical point**: altitudes are relative to take-off.
The survey preset flies **80 m** (1.9 cm/pixel) in two crossing grids. The second grid is tilted 20°, which keeps
the terrain model from bending ("doming"). That matters when you design water levels. Expect about **5 missions
of ≤12 min** and ~440 photos for 15 acres. Short of batteries? `--battery-minutes 15` or `--altitude 100`
(fewer photos). Use `--grids A` only for a quick look; it isn't accurate enough for pond levels.

Plans are saved in `~/drone/plans/bagrod/`. Open `bagrod_preview.kml` in Google Earth to check the paths, then follow
`bagrod_summary.txt` to import each CSV into **Litchi Mission Hub** (flylitchi.com/hub) and set
Heading Mode *Custom*, Path Mode *Straight Lines*, Finish Action *RTH*.

**First time:** fly only mission A1, then run `dronemap photos` on those photos. Litchi should take a photo every
~17 m along each line while the drone keeps flying. If it doesn't, re-plan with `--stop-and-shoot`. The drone
then hovers for every photo: more batteries, but foolproof. Process those photos with `--no-rolling-shutter`.

**When to fly:** bare fields give the best ground model (crops and long grass become "ground").
Overcast days or 10:00-14:00 are best. Avoid strong wind.

### Fly, then check before leaving the site
Copy the SD card's `DCIM` photos into `~/drone/projects/bagrod/images/` (photos only, no videos), then:
```bash
dronemap photos ~/drone/projects/bagrod/images --plan ~/drone/plans/bagrod/bagrod_plan.json
```
Orange patches inside the plot, or red gap lines, mean re-fly that mission.

### Process (about 1-3 hours on the Mac)
```bash
dronemap process ~/drone/projects/bagrod --preset survey
```
The command can be stopped (Ctrl+C) and resumed by running it again. Results land in the project folder:
`odm_orthophoto/odm_orthophoto.tif` (+ `.kmz` for Google Earth), `odm_dem/dtm.tif` (ground) and
`dsm.tif` (surface), `odm_texturing/odm_textured_model_geo.glb` (3D), and `odm_report/report.pdf`
(check the overlap and GSD pages).

### Understand the land
```bash
dronemap site ~/drone/projects/bagrod --boundary ~/maps/bagrod.kml
```
Outputs are in `~/drone/projects/bagrod/dronemap/site/`:
- `site_overview.png` shows contours, drainage lines, natural low spots (L1, L2, ...) and **where water leaves the plot** (E1, E2, ...).
  A pond just upstream of the biggest exit catches the most runoff.
- `site.kmz` opens in Google Earth with all layers. `contours.dxf` is for an engineer/contractor (AutoCAD/QCAD).
- `site_report.md` lists elevations, slope classes, catchment areas and rough annual runoff. Set your rainfall with
  `--rainfall-mm`; the default of 1100 is a placeholder, so use the local figure.

### Design the pond
Draw the pond's **top edge** as a polygon in Google Earth over the orthophoto KMZ, save it as `pond.kml`, then:
```bash
dronemap pond ~/drone/projects/bagrod --outline ~/maps/pond.kml --depth 2.5,3,3.5 --side-slope 2 --cost-per-hour 1800
```
Or try a rectangle quickly: `--center 23.xxxxx,78.xxxxx --size 60x40 --rotation 30`.

| Option | Meaning |
|---|---|
| `--depth` | Floor depth below the **lowest** point of the edge. A comma list compares options side by side |
| `--side-slope 2` | Inner slope 2 horizontal : 1 vertical. Use 2.5-3 for loose or sandy soil, 1.5 for firm clay |
| `--freeboard 0.5` | Safety gap between full water level and the rim |
| `--bund-height 1` | Build an earth bank 1 m above the lowest edge point. It raises the water level and uses dug soil |
| `--swell 1.25`, `--compaction 0.9`, `--trolley-m3 3` | Soil bulking, bund compaction, tractor-trolley size |
| `--excavator-rate 60`, `--cost-per-hour` | Machine output (m³/h) and hire rate for time and cost |
| `--stl --exaggeration 2` | 3D-printable terrain block (fits the A1 bed) |

Outputs (in `~/drone/projects/bagrod/dronemap/pond/`):
- `pond_report.md` with levels, capacity (m³, litres, acre-feet), stage-capacity table, cut, fill, surplus soil
  (and how many trolley loads), excavator hours and cost. It also says whether the pond's own catchment will fill it.
- `pond_plan.png` (cut/fill map) and `pond_sections.png` (long and cross sections).
- `pond_3d_before.glb` and `pond_3d_after.glb` (drag into Blender or https://gltf-viewer.donmccurdy.com).
- `pond.kmz` for Google Earth. `designed_terrain.tif` can be fed back into `dronemap site` to see the new drainage.

**Freeform shaping:** `dronemap to-blender ~/drone/projects/bagrod --area ~/maps/pond_area.kml` writes a textured
terrain to sculpt in Blender (steps are printed). Then
`dronemap from-blender ~/drone/projects/bagrod <folder>/terrain_edited.obj` gives the volumes of your edit.

**Accuracy.** Drone-only heights are good *relative* to each other. With the crossing grids they are typically
within about 5-10 cm across the plot, which is what volumes and slopes need. Absolute height above sea level can
be several metres off.
Before construction, check 4-5 points with a dumpy level. Or lay out 5-8 marked targets (60x60 cm
black/white squares), have a local surveyor measure them with DGPS, put them in `~/drone/projects/bagrod/gcp_list.txt`
([ODM format](https://docs.opendronemap.org/gcp/)) and reprocess. `dronemap process` uses that file automatically.

---

## 4. Tigadda: high-detail model and Gaussian splat of the Sanctuary

### Plan
```bash
dronemap plan ~/maps/tigadda.kml --preset detail --home 23.xxxxx,78.xxxxx \
    --orbit 23.xxxxx,78.xxxxx --orbit 23.xxxxx,78.xxxxx,30
```
- Grids: nadir + two oblique grids at **45 m** (≈1.1 cm/pixel), so roofs, walls and trees are seen from all sides.
- `--orbit LAT,LON[,RADIUS]` adds a circle of 36 photos at 20 m and 35 m around a building (default radius 25 m).
  Add one per building or feature tree. **Walk each orbit circle first: it must be clear of trees and wires at
  that height.**
- Expect ~8 missions and ~900+ photos for 4 acres with two orbits. If any tree or mast is taller than ~35 m, use `--altitude 55`.

### Process + splat (overnight)
```bash
dronemap process ~/drone/projects/tigadda --preset detail --splat
```
- ODM builds the detailed mesh (`odm_texturing/odm_textured_model_geo.glb`).
- OpenSplat then trains the splat on the Mac GPU: `~/drone/projects/tigadda/dronemap/splat/splat.ply`. It saves progress
  every 5000 steps. If stopped, run `dronemap splat ~/drone/projects/tigadda --resume`.
- Quick preview first: `dronemap splat ~/drone/projects/tigadda --iters 7000`.
- **View and clean the splat** at https://playcanvas.com/supersplat/editor. Drag in `splat.ply`, crop floaters, and
  export a compressed file for the Sanctuary website. The editor runs in your browser; nothing is uploaded unless
  you publish.
- `--quality max` squeezes out more detail. It is slower and may run out of memory on big photo sets; if ODM
  stops with exit code 137, raise Docker's memory or go back to standard.

---

## 5. Commands at a glance
```
dronemap plan BOUNDARY [--preset survey|detail|quick] [--altitude M] [--home LAT,LON] [--orbit LAT,LON[,R]] [--battery-minutes 12] [--agl] [--stop-and-shoot]
dronemap photos IMAGES_DIR [--plan NAME_plan.json]
dronemap process PROJECT [--preset survey|detail] [--quality fast|standard|max] [--splat] [--gcp FILE] [--boundary KML] [--dry-run]
dronemap splat PROJECT [--iters 30000] [--downscale auto|1|2|4] [--resume]
dronemap site PROJECT|DTM.tif [--boundary KML] [--contour-interval 0.5] [--rainfall-mm 1100]
dronemap pond PROJECT|DTM.tif (--outline KML | --center LAT,LON --size LxW [--rotation DEG]) [--depth 3] [--bund-height 0] [--stl]
dronemap to-blender PROJECT [--area KML] [--res 0.5]
dronemap from-blender PROJECT EDITED.obj
dronemap doctor
```
Every command has `-h` for all options. A project folder is any folder with an `images/` sub-folder.

## 6. Troubleshooting
- **"Docker is not running"**: open Docker Desktop and wait for *Engine running*.
- **ODM exit code 137**: out of memory. Raise Docker's memory, use `--quality fast`, or
  `--odm-args "--split 300 --split-overlap 60"`.
- **Few photos aligned / holes in the model**: not enough overlap (check `dronemap photos`), motion blur
  (use 1/1000 s), or water/uniform crops. Re-fly the weak area and process again with all photos.
- **Splat says "Using CPU"**: rebuild with Xcode installed: `setup/setup_mac.sh --opensplat`.
- **"libc10.dylib can't be opened"** (macOS security): System Settings > Privacy & Security > *Allow*, then run again.
- **Litchi import fails**: import the CSV exactly as written (don't open and re-save it in Excel).

## 7. Safety and rules (India)
- Your Mavic 2 Pro (907 g) is a **micro** drone under India's Drone Rules. It needs a UIN on the Digital Sky
  platform, and flying it for business needs a remote pilot certificate. Check the current rules and the
  **airspace map** (green zone) before each job. Stay below 120 m, and fly in daylight within line of sight.
- Missions don't use obstacle avoidance. Check every path in the preview KML, keep RTH altitude above the tallest
  tree, and keep your thumbs near the sticks so you can take over.

---
Built on [OpenDroneMap](https://github.com/OpenDroneMap/ODM) (AGPL) and [OpenSplat](https://github.com/pierotofy/OpenSplat)
(AGPL). Both run as separate programs. This toolkit only prepares inputs and reads their outputs.
