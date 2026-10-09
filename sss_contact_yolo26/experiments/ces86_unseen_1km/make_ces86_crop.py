#!/usr/bin/env python3
"""Cut a 1 km x 1 km native 0.1 m/px GeoTIFF from the CES-86 (Costa Head) ECW mosaic for inference.

Same source and decoder as the training patches: the grey band (band 0) of
C13007a_Costa_AllAreas_Comments_Rev1.ecw, decoded 1:1 (no resampling, no stretching) by the project-local
decoder sss_contact_inspector/local/bin/ecw_window.

The decoder's output depends on the requested window size: a 150x150 window reproduces the training PNGs
exactly, while larger windows differ by ~4 grey levels on average (max 20). So the GeoTIFF is built the way the
training patches were made: every 150x150 window on the 100 px grid is decoded on its own, and each pixel is
taken from the window whose centre is nearest (outer rim from whole windows, then each window's central
100x100 px). A patch the map app cuts from this GeoTIFF on the same grid therefore has its central 100x100 px
identical to a training-style decode; its 25 px rim comes from the neighbouring windows.

The window was chosen to be fully inside the survey
coverage, at least 100 m away from every patch of the spatial split bundle (train/val/test/buffer), and to
contain many official CES-86 point contacts (reference only; they were not used to pick the model).

    python3 make_ces86_crop.py                                   (the 1 km window below, into this folder)
    python3 make_ces86_crop.py --x0 74200 --y0 50000 --size 20000 --out ../ces86_unseen_2km

Writes only into the output folder (default: this folder):
    ces86_x080200_y048800_0p1m.tif   8-bit gray GeoTIFF, 10000 x 10000, 0.1 m/px, EPSG:32630
    reference_contacts.csv           official CES-86 point contacts inside the window
    manifest.json                    source, window, georeference, checks
"""
import argparse
import csv
import json
import os
import re
import subprocess

import numpy as np
import tifffile
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))                  # .../IRP-external
INSPECTOR = os.path.join(REPO, "sss_contact_inspector")
DECODER = os.path.join(INSPECTOR, "local/bin/ecw_window")
ECW = os.path.join(INSPECTOR, "data/derived_sources/costa/C13007a_Costa_AllAreas_Comments_Rev1.ecw")
ORIGINAL_ZIP = "/Volumes/HP P900/MDE_SSS_Contacts_Costa_Head_Moray_Firth_2026-09-20/Costa_Head_CES-86/CES-86_Mosaic.zip"
BUNDLE = os.path.join(REPO, "sss_contact_yolov8_spatial_split_bundle")
X0, Y0, SIZE = 80200, 48800, 10000          # ECW pixel window (multiples of 100, like the bundle patch ids)
PATCH, STRIDE = 150, 100
E_REG, N_REG, RES = 477323.46875, 6568269.5, 0.1
BAND = 0


def grid_starts(n):
    """Same rule as the map app: every STRIDE px, plus one flush with the far edge."""
    s = list(range(0, n - PATCH + 1, STRIDE))
    if s[-1] != n - PATCH:
        s.append(n - PATCH)
    return s


class Worker:
    """Persistent decoder process (ecw_window serve), one request per 150 px window, as the training export."""

    def __init__(self):
        self.p = subprocess.Popen([DECODER, "serve", ECW], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        json.loads(self.p.stdout.readline())
        self.n = 0

    def decode(self, x0, y0, w=PATCH, h=PATCH):
        self.n += 1
        self.p.stdin.write(b"%d %d %d %d %d %d %d %d\n" % (self.n, x0, y0, x0 + w - 1, y0 + h - 1, w, h, BAND))
        self.p.stdin.flush()
        status, rid, size = self.p.stdout.readline().decode().strip().split(" ", 2)
        assert status == "OK" and int(rid) == self.n, (status, rid, size)
        out = self.p.stdout.read(int(size))
        header = b"P5\n%d %d\n255\n" % (w, h)
        assert out.startswith(header) and len(out) == len(header) + w * h
        return np.frombuffer(out, np.uint8, offset=len(header)).reshape(h, w)

    def close(self):
        self.p.stdin.close()
        self.p.wait()


def decode(x0, y0, w, h):
    """Native 1:1 window (inclusive end indices) as a uint8 array."""
    out = subprocess.run([DECODER, "window", ECW, str(x0), str(y0), str(x0 + w - 1), str(y0 + h - 1), str(w), str(h),
                          str(BAND)], capture_output=True, check=True).stdout
    header = b"P5\n%d %d\n255\n" % (w, h)
    assert out.startswith(header) and len(out) == len(header) + w * h, out[:40]
    return np.frombuffer(out, np.uint8, offset=len(header)).reshape(h, w)


def check_unseen():
    """Every bundle patch (train/val/test/buffer) must be at least 1000 px (100 m) from the window."""
    for r in csv.DictReader(open(os.path.join(BUNDLE, "metadata/split_manifest.csv"))):
        m = re.search(r"_x(\d+)_y(\d+)$", r["patch_id"])
        x, y = int(m.group(1)), int(m.group(2))
        dx = max(X0 - (x + PATCH), x - (X0 + SIZE), 0)
        dy = max(Y0 - (y + PATCH), y - (Y0 + SIZE), 0)
        assert max(dx, dy) >= 1000, "window is within 100 m of bundle patch %s" % r["patch_id"]


def main():
    global HERE, X0, Y0, SIZE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--x0", type=int, default=X0)
    ap.add_argument("--y0", type=int, default=Y0)
    ap.add_argument("--size", type=int, default=SIZE)
    ap.add_argument("--out", default=HERE)
    a = ap.parse_args()
    X0, Y0, SIZE, HERE = a.x0, a.y0, a.size, os.path.abspath(a.out)
    os.makedirs(HERE, exist_ok=True)
    NAME = "ces86_x%06d_y%06d_0p1m" % (X0, Y0)
    check_unseen()
    info = json.loads(subprocess.run([DECODER, "info", ECW], capture_output=True, check=True, text=True).stdout)
    assert (info["width"], info["height"], info["projection"], info["datum"]) == (129132, 201580, "NUTM30", "WGS84"), info
    assert abs(info["origin_x"] - E_REG) < 1e-6 and abs(info["origin_y"] - N_REG) < 1e-6, info

    worker = Worker()
    # check first: a 150 px window decode reproduces the training PNGs (same decoder, band and window rule)
    check = []
    for pid in ("costa_val_x063200_y053100", "costa_val_x064200_y053500"):
        m = re.search(r"_x(\d+)_y(\d+)$", pid)
        png = np.asarray(Image.open(os.path.join(BUNDLE, "data/images/test", pid + ".png")))
        check.append({"patch": pid, "identical_to_training_png": bool(np.array_equal(
            worker.decode(int(m.group(1)), int(m.group(2))), png))})
    assert all(c["identical_to_training_png"] for c in check), check

    starts = grid_starts(SIZE)
    tiles = {(x, y): worker.decode(X0 + x, Y0 + y) for y in starts for x in starts}
    worker.close()
    img = np.empty((SIZE, SIZE), np.uint8)
    for (x, y), t in tiles.items():                  # pass 1: whole windows (fills the outer rim)
        img[y:y + PATCH, x:x + PATCH] = t
    for (x, y), t in tiles.items():                  # pass 2: central 100x100 px of every window
        img[y + 25:y + 125, x + 25:x + 125] = t[25:125, 25:125]
    rim = [float(np.abs(img[y:y + PATCH, x:x + PATCH].astype(int) - t).mean()) for (x, y), t in list(tiles.items())[::97]]

    e0, n0 = E_REG + RES * X0, N_REG - RES * Y0          # top-left corner, same convention as the bundle manifest
    geokeys = (1, 1, 0, 4, 1024, 0, 1, 1, 1025, 0, 1, 1, 3072, 0, 1, 32630, 3076, 0, 1, 9001)
    tif = os.path.join(HERE, NAME + ".tif")
    tifffile.imwrite(tif, img, photometric="minisblack", compression="zlib",
                     extratags=[(33550, "d", 3, (RES, RES, 0.0)), (33922, "d", 6, (0.0, 0.0, 0.0, e0, n0, 0.0)),
                                (34735, "H", len(geokeys), geokeys)])

    contacts = [r for r in json.load(open(os.path.join(INSPECTOR, "data/contacts.json")))
                if r["dataset"] == "costa" and r.get("geometry") != "line"]
    rows = []
    for r in contacts:
        px, py = (r["e"] - e0) / RES, (n0 - r["n"]) / RES
        if 0 <= px < SIZE and 0 <= py < SIZE:
            rows.append({"id": r["id"], "area": r["area"], "description": r.get("description"), "easting": r["e"],
                         "northing": r["n"], "px": round(px, 1), "py": round(py, 1)})
    with open(os.path.join(HERE, "reference_contacts.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    white = float((img == 255).mean())
    json.dump({"original_source_zip": ORIGINAL_ZIP, "ecw": ECW, "decoder": DECODER, "decoder_info": info,
               "band": BAND, "window_ecw_px_inclusive": [X0, Y0, X0 + SIZE - 1, Y0 + SIZE - 1],
               "size_px": [SIZE, SIZE], "m_per_px": RES, "resampling": "none (1:1 decode)",
               "decode_method": "%d separate 150x150 window decodes on the 100 px grid, nearest-centre stitching; "
                                "a single large-window decode differs from the training PNGs by ~4 grey levels" % len(tiles),
               "page_patch_vs_own_window_mean_abs_diff": round(float(np.mean(rim)), 3),
               "crs": "EPSG:32630 (WGS 84 / UTM 30N)", "utm_top_left": [e0, n0],
               "georef_note": "top-left = registration coordinate + 0.1 m * pixel index, the convention of the bundle "
                              "manifest; ERS does not state edge vs centre (0.05 m)",
               "no_opacity_band": "the decoder exposes only the grey band; no-data is white (255)",
               "fraction_pixels_255": round(white, 5), "decoder_consistency_check": check,
               "distance_to_bundle_patches": ">= 100 m from every train/val/test/buffer patch (checked per patch)",
               "script": os.path.abspath(__file__),
               "reference_contacts": len(rows), "geotiff": os.path.basename(tif)},
              open(os.path.join(HERE, "manifest.json"), "w"), indent=1)
    print("%s | %d reference contacts | 255-pixels %.3f%% | check %s" % (tif, len(rows), 100 * white, check))


if __name__ == "__main__":
    main()
