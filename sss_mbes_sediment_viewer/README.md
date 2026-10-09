# Costa Head SSS · MBES · sediment viewer

Independent local research inspector: the native SSS mosaic, MBES elevation,
an elevation-geometry roughness diagnostic and the official SBF sediment
polygons in one EPSG:32630 frame. It leaves `sss_contact_inspector` and all
source ZIPs untouched and only writes its own `derived/` cache.

## Evidence-backed data model

* CRS: EPSG:32630, from the official GIS `.prj` files and the Costa ERS
  (`WGS84`, `NUTM30`).
* SSS: the existing native ECW service (`sss_contact_inspector/serve.py`) is
  the source of truth. The viewer draws its `/api/tile` pyramid (z = 10 is the
  0.1 m native level) through the same world→screen transform as every vector,
  so SSS and polygons cannot drift apart with zoom or pan.
* MBES: `C13007_Costa_AllData_XYZ_2m_UTM30N_LAT_wip00.pts` (2 m nodes,
  14.5 M points, LAT elevation −77 … −2.7 m, negative below LAT).
  **The 0.5 m member is not elevation.** Its third column is an integer 1–35
  that stripes along survey lines and is uncorrelated with the 2 m elevations
  (r = −0.18 over a shared 300 m box); it behaves like a per-cell sounding
  count. The package README lists the genuine 0.5 m LAT grid as "held
  offline". `build.py` refuses any member whose values are all integers.
* SSS contacts: `13_86_GIS_Shapefiles.zip → SSS Listing.zip →
  C13007a_CostaHead_SSSListing.shp/.dbf` (3825 points; geometry equals the
  Easting/Northing attributes to < 1 cm). They are **not** taken from
  `CES-86_SSS_target_listings.zip`: in `C13007a_Costa_Afl_Single_Target_Listing.csv`
  rows 1394 onward (S2352–S3823, 1472 targets) the Easting/Northing columns are
  an arithmetic series (+2.8 m E, +0.5 m N per row, i.e. a spreadsheet
  auto-fill), placing "boulders" on a 4 km straight line through blank
  mosaic. For S961–S2351 the two sources agree (median 0.05 m). Only the 46
  linear contacts exist solely in the CSV.
* SBF polygons are official geometry. `Gravelly SAND - AfL area - North Up`
  is normalised to `Gravelly SAND` for styling only; original attributes
  remain and are shown on click.
* Roughness: absolute residual from an 11 × 11-cell (22 m at 2 m) local mean,
  needing at least 61/121 valid cells; holes and edges are transparent. It is
  a geometry diagnostic, not sediment ground truth and not backscatter.
* Elevation tiles are hillshaded (azimuth 315°, altitude 45°, 8× vertical
  exaggeration, because the shelf here is nearly flat) on top of a fixed global
  colour ramp; the ramp limits are in the sidebar and in `audit.json`.

## Reproducible workflow

```sh
cd /Users/haonanzhang/IRP-external/sss_mbes_sediment_viewer
python3 build.py                 # full 2 m extent: ~30 s, ~1 GB peak RSS, 61 MB of tiles
python3 build.py --region 478900 6548100 479300 6548500   # optional sub-box
python3 -m unittest discover -s tests
python3 serve.py --port 8091     # starts the ECW service on 8090 if it is not running
```

Open `http://127.0.0.1:8091`. The builder streams the ZIP member twice (extent
scan, then grid fill), computes roughness in row bands, and writes north-up
256 px PNG tile pyramids to `derived/data/mbes/{elevation,roughness}/z/x_y.png`
plus `derived/data/audit.json` (bounds, ranges, per-level tile lists, RSS,
timings) and a compact `derived/data/vectors.json`.

## Checking alignment yourself

* **Check SSS**: at any view centre, compares the tiles the page actually drew
  with the contact inspector's validated `/api/crop` window (20 m, 0.1 m) and
  reports the best-matching shift. Expect ≤ 0.1 m (the crop path resamples its
  window by up to one pixel).
* **Split**: SSS on the left, the chosen MBES layer on the right, one shared
  frame; drag the handle to swipe across features such as sand-wave crests.
* Click any polygon, contact, sand wave or SBF polyline for its official
  attributes; nested polygons resolve to the smallest containing one. The
  search box centres the view on the first matching attribute.

## Limitations

The 2 m grid is the finest genuine elevation available offline, so bedforms
with 3–7 m wavelengths are only marginally resolved. Roughness and elevation
colours are global-range diagnostics; local contrast comes from the hillshade
only. The viewer needs the ECW service on port 8090 for SSS; without it the
MBES layers and vectors still work.
