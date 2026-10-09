#!/usr/bin/env python3
"""Resample a 0.5 m/px MMT 270 HF side-scan crop to 0.1 m/px and cut it into 150x150 patches (experiment).

The YOLO26 contact model was trained on 0.1 m/px patches of the CES-86 mosaic. This survey only has 0.5 m/px
mosaics, so this experiment upsamples a 1 km x 1 km crop 5x (bicubic). Upsampling does not add detail: treat any
result on these patches as experimental.

    ../../apps/sss_contact_yolo26_test_inference_map/.venv/bin/python make_resampled_patches.py

Reads one tile from the external drive (read-only) and writes only into this folder:
    B05_HF_R0001_C0010_0p1m.tif     8-bit gray GeoTIFF, 0.1 m/px, 10000 x 10000, original CRS keys copied
    patches/                        150x150 L-mode PNGs, 150 px window every 100 px (same grid as the map app)
    reference_contacts.csv          RVO SSS contacts inside the crop (Cable/Pipeline points excluded)
    manifest.json                   source, crop, resampling, georeference, counts
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import tifffile
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DRIVE = "/Volumes/HP P900/irp_data_explore"
TILE = os.path.join(DRIVE, "sss_mosaics_only/02_MOSAICS/02_0.5mx0.5m_HF/B05/MMT_270_FR_SSS_B05_HF_SW070304R0001_C0010.tif")
CONTACTS = os.path.join(DRIVE, "06_INTEGRATED_MBES_SSS_MAG/01_CORRELATED_TARGET_LIST/01_XLSX/"
                               "MMT_270_RVO_Integrated_SSS_MAG_Contact_Listing.xlsx")
X0, Y0, SIZE = 1600, 1000, 2000        # crop in source pixels (0.5 m): 1 km x 1 km, chosen for contact density
SRC_RES, DST_RES = 0.5, 0.1
K = int(round(SRC_RES / DST_RES))      # 5
PATCH, STRIDE = 150, 100
NAME = "B05_HF_R0001_C0010"


def grid_starts(n):
    """Same rule as the map app: every STRIDE px, plus one flush with the far edge."""
    s = list(range(0, n - PATCH + 1, STRIDE))
    if s[-1] != n - PATCH:
        s.append(n - PATCH)
    return s


def main():
    Image.MAX_IMAGE_PIXELS = None
    with tifffile.TiffFile(TILE) as tf:
        page = tf.pages[0]
        scale, tie = page.tags[33550].value, page.tags[33922].value
        geokeys = {c: page.tags[c].value for c in (34735, 34736, 34737) if c in page.tags}
    assert abs(scale[0] - SRC_RES) < 1e-9 and abs(scale[1] - SRC_RES) < 1e-9, scale
    rgba = np.asarray(Image.open(TILE))[Y0:Y0 + SIZE, X0:X0 + SIZE]        # Pillow decodes any compression
    assert rgba.shape == (SIZE, SIZE, 4), rgba.shape
    valid = rgba[..., 3] > 0
    gray = rgba[..., 0]
    if not (np.array_equal(gray[valid], rgba[..., 1][valid]) and np.array_equal(gray[valid], rgba[..., 2][valid])):
        sys.exit("R, G, B differ inside the valid area: not a grayscale mosaic, stopping")

    # 5x bicubic upsampling; Pillow maps pixel areas, so the crop's outer edges stay where they were
    n = SIZE * K
    up = np.asarray(Image.fromarray(gray).resize((n, n), Image.BICUBIC))
    up_valid = np.asarray(Image.fromarray(valid.astype(np.uint8) * 255).resize((n, n), Image.NEAREST)) > 0
    e0 = tie[3] + X0 * SRC_RES                  # top-left corner of the crop (PixelIsArea)
    n0 = tie[4] - Y0 * SRC_RES

    os.makedirs(os.path.join(HERE, "patches"), exist_ok=True)
    tags = [(33550, "d", 3, (DST_RES, DST_RES, 0.0)), (33922, "d", 6, (0.0, 0.0, 0.0, e0, n0, 0.0))]
    for code, v in geokeys.items():             # keep the original CRS definition (ETRS89 / UTM 31N, user-defined)
        tags.append((code, "H" if code == 34735 else "d" if code == 34736 else "s",
                     len(v) if code != 34737 else 0, v))
    tif = os.path.join(HERE, NAME + "_0p1m.tif")
    if up_valid.all():
        tifffile.imwrite(tif, up, photometric="minisblack", compression="zlib", extratags=tags)
    else:                                       # second band = opacity (0 = no data), as the map app expects
        tifffile.imwrite(tif, np.dstack([up, up_valid.astype(np.uint8) * 255]), photometric="minisblack",
                         extrasamples=["unassalpha"], compression="zlib", extratags=tags)

    written = skipped = 0
    for y in grid_starts(n):
        for x in grid_starts(n):
            if not up_valid[y:y + PATCH, x:x + PATCH].any():
                skipped += 1
                continue
            Image.fromarray(np.ascontiguousarray(up[y:y + PATCH, x:x + PATCH])).save(
                os.path.join(HERE, "patches", "%s_x%06d_y%06d.png" % (NAME, x, y)))
            written += 1

    c = pd.read_excel(CONTACTS)
    c = c[~c["CLASSIFICATION"].isin(["Cable", "Pipeline"])].copy()
    c["px"] = (c["EASTING"] - e0) / DST_RES
    c["py"] = (n0 - c["NORTHING"]) / DST_RES
    c = c[(c.px >= 0) & (c.px < n) & (c.py >= 0) & (c.py < n)]
    c = c[up_valid[c.py.astype(int), c.px.astype(int)]]
    c[["ID", "EASTING", "NORTHING", "px", "py", "LENGTH (m)", "WIDTH (m)", "HEIGHT (m)", "CLASSIFICATION"]].round(3) \
        .to_csv(os.path.join(HERE, "reference_contacts.csv"), index=False)

    json.dump({"source_tile": TILE, "source_m_per_px": SRC_RES, "crop_source_px": [X0, Y0, X0 + SIZE, Y0 + SIZE],
               "resampling": "Pillow bicubic x%d (gray); opacity nearest" % K, "m_per_px": DST_RES,
               "size_px": [n, n], "utm_top_left": [e0, n0], "crs": "ETRS89 / UTM zone 31N (keys copied from source)",
               "valid_fraction": round(float(up_valid.mean()), 4), "geotiff": os.path.basename(tif),
               "patch_px": PATCH, "stride_px": STRIDE, "patches_written": written, "patches_skipped_nodata": skipped,
               "patch_name": NAME + "_x{X}_y{Y}.png: X, Y = top-left pixel in the 0.1 m GeoTIFF; "
                             "easting = E0 + 0.1*X, northing = N0 - 0.1*Y (pixel corner)",
               "reference_contacts": int(len(c)),
               "reference_contact_classes": c["CLASSIFICATION"].value_counts().to_dict()},
              open(os.path.join(HERE, "manifest.json"), "w"), indent=1)
    print("geotiff %s | patches %d (skipped %d) | reference contacts %d" % (tif, written, skipped, len(c)))


if __name__ == "__main__":
    main()
