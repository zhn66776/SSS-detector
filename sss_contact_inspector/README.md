# SSS Contact Inspector

A local, browser-based reviewer for Costa Head CES-86 and Moray Firth CES-371
side-scan-sonar contacts. The map uses projected EPSG:32630 coordinates, loads
only visible tiles, clusters contacts at small scales, and reads the right-hand
inspection crop directly from native raster rows.

## Current capability

- Moray is fully usable: all six 1 m GeoTIFF strips, OWF Site and ECR contacts,
  native-pixel PNG crops, map pan/zoom, search, category filters, coordinates,
  scale bar, and crop pan/zoom.
- `Magnetic Anomaly` is excluded from the default SSS view. A separate checkbox displays magnetic anomalies alongside the SSS contacts.
- Costa is fully enabled through a project-local arm64 ECW 3.3 read-only
  decoder: AFL, CR1, CR2 and CR3 contacts, overview tiles and 0.1 m native-pixel
  crops are available without installing a system-wide GDAL.
- The decoder reads the ECW grey band. Pixels beyond useful image coverage appear near-white because the appended opacity band is unavailable. The alignment report includes a sample from this area.

## Evidence used for CRS and georeferencing

- Costa ERS: `Datum = WGS84`, `Projection = NUTM30`, 0.1 m cells,
  129132 columns by 201580 rows, registration coordinate
  E 477323.46875 / N 6568269.5. This resolves to EPSG:32630. The MEDIN file's
  `EPSG::4326` describes its geographic discovery bounding box. The raster uses EPSG:32630. The ERS pixel-centre versus pixel-edge convention has a 0.05 m ambiguity.
- Moray GeoTIFF key 3072 is EPSG:32630 in every TIFF. GeoTIFF pixel scale is
  1.0 m and agrees with all six TFWs; each TFW centre origin is half a pixel
  inside the GeoTIFF tie point.

The generated `data/metadata.json` records every raster extent, source ZIP size
and mtime, member CRC, counts, rejected rows, preparation performance and the
contact-in-raster coverage audit.

## Run

Prerequisites: Python 3.9+ and the packages in `requirements.txt`.

```bash
cd /Users/haonanzhang/IRP-external/sss_contact_inspector
python3 prepare.py --copy-costa-ecw
python3 serve.py --port 8766
```

Open <http://127.0.0.1:8766>. The service binds to localhost by default.

Preparation is resumable: existing derived TIFFs with the exact expected byte
size are reused. Original ZIPs are opened read-only.

## Tests and alignment evidence

```bash
python3 -m unittest discover -s tests -v
python3 validate_alignment.py
```

`validation/alignment_report.json` and the native/annotated PNG pairs provide
inspectable Moray OWF/ECR samples. Annotated copies are nearest-neighbour 10x
enlargements; the unannotated 50×50 files are direct native 1 m windows.

## I/O and resource limits

- preparation uses one worker and 8 MiB streaming copies;
- raster requests are limited to four concurrent workers;
- TIFF reads decode only requested PackBits scanlines (`RowsPerStrip=1`);
- the in-process tile/crop cache is capped at 96 items;
- raster windows are decoded on demand;
- maximum zoom uses native-resolution PNG tiles; lower levels use JPEG only for
  display efficiency;
- the right crop is always lossless PNG generated from source raster rows.

This build produced about 1.1 GiB of derived data (six Moray TIFFs, a read-only
copy of the 1.006 GB Costa ECW, contacts and metadata). Cold extraction,
resumable rerun, service RSS, tile and crop
measurements are recorded in `data/performance.json`. The latest preparation is
also in `data/metadata.json`; live cumulative latency is shown below every crop
and at `/api/stats`.

## Rebuilding the Costa decoder

The project-local decoder is built from the public ECW JPEG 2000 SDK 3.3 source
preserved under `vendor-src/libecw`, including its original `License.txt` and
the full corresponding source. Three small compatibility patches are present:
modern macOS supplies `wcsdup`, while `getlogin` and `gettimeofday` require
their current POSIX headers.

To rebuild both the static library and the narrow window helper:

```bash
chmod +x native/rebuild_ecw_stack.sh native/build_ecw_window.sh
native/rebuild_ecw_stack.sh
local/bin/ecw_window info data/derived_sources/costa/C13007a_Costa_AllAreas_Comments_Rev1.ecw
```

Runtime uses one persistent decoder process with a 128 MiB SDK cache and reads requested windows at native resolution.
