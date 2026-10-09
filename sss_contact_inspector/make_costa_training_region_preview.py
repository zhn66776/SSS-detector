#!/usr/bin/env python3
"""Create a reproducible Costa Head training-region preview and five native-pixel samples.

The four region vertices are the centres of the black outline in the user's
overview screenshot.  They were mapped linearly from the screenshot's inner
map rectangle (x=4..984, y=3..1499) to the complete Costa ECW extent.  This is
an intentionally approximate selection region, not a surveyed boundary.

No source archive or source ECW is modified.  The ECW decoder is asked only
for one overview and fifteen small windows.
"""
from __future__ import annotations

import csv
import io
import json
import random
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "training_region_preview"
CONTACTS = ROOT / "data" / "contacts.json"
METADATA = ROOT / "data" / "metadata.json"
ECW = ROOT / "data" / "derived_sources" / "costa" / "C13007a_Costa_AllAreas_Comments_Rev1.ecw"
DECODER = ROOT / "local" / "bin" / "ecw_window"

# Screenshot pixel positions at the centre of the 17 px black outline, in
# clockwise order: lower-left, upper-left, upper-right, lower-right.
SCREEN_POLYGON = [(481, 401), (683, 56), (976, 230), (774, 574)]
SCREEN_MAP_BOX = (4, 3, 984, 1499)  # left, top, right, bottom pixels
OVERVIEW_SIZE = (980, 1496)
PATCH_SIZES = (100, 50, 25)
MARKED_PATCH_SIZE = 150
SEED = 20260921


def point_in_polygon(e: float, n: float, polygon: list[tuple[float, float]]) -> bool:
    """Strict ray-casting membership; boundary points are not counted."""
    inside = False
    j = len(polygon) - 1
    for i, (ei, ni) in enumerate(polygon):
        ej, nj = polygon[j]
        if (ni > n) != (nj > n) and e < (ej - ei) * (n - ni) / (nj - ni) + ei:
            inside = not inside
        j = i
    return inside


def decode(x0: int, y0: int, x1: int, y1: int, width: int, height: int) -> Image.Image:
    """Decode grey band 0, preserving requested output pixels exactly."""
    run = subprocess.run(
        [str(DECODER), "window", str(ECW), str(x0), str(y0), str(x1), str(y1), str(width), str(height), "0"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # ecw_window writes a binary PGM (P5), including its short header.
    image = Image.open(io.BytesIO(run.stdout)).convert("L")
    if image.size != (width, height):
        raise RuntimeError(f"decoder returned {image.size}; expected {(width, height)}: {run.stderr.decode(errors='replace')}")
    return image


def font(size: int):
    for candidate in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def main() -> None:
    if not ECW.exists() or not DECODER.exists():
        raise FileNotFoundError("Costa ECW or the project-local ECW decoder is missing")
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "patches").mkdir(parents=True)
    (OUT / "marked_150px_patches").mkdir(parents=True)

    meta = json.loads(METADATA.read_text())
    raster = meta["datasets"]["costa"]["rasters"][0]
    left, right = raster["left"], raster["right"]
    top, bottom = raster["top"], raster["bottom"]
    width, height = raster["width"], raster["height"]
    sx0, sy0, sx1, sy1 = SCREEN_MAP_BOX

    def screenshot_to_world(x: float, y: float) -> tuple[float, float]:
        return (
            left + (x - sx0) / (sx1 - sx0) * (right - left),
            top - (y - sy0) / (sy1 - sy0) * (top - bottom),
        )

    polygon = [screenshot_to_world(x, y) for x, y in SCREEN_POLYGON]
    contacts = json.loads(CONTACTS.read_text())
    inside = [
        row for row in contacts
        if row["dataset"] == "costa" and row["geometry"] == "point" and point_in_polygon(row["e"], row["n"], polygon)
    ]
    if len(inside) < 5:
        raise RuntimeError(f"only {len(inside)} point contacts were inside the approximate polygon")
    samples = random.Random(SEED).sample(inside, 5)

    geojson = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {
                "name": "Costa Head approximate training region from user screenshot",
                "crs": "EPSG:32630",
                "selection_method": "linear screenshot-to-ECW mapping; black outline centreline",
                "screenshot_map_box_px": SCREEN_MAP_BOX,
            },
            "geometry": {"type": "Polygon", "coordinates": [[*polygon, polygon[0]]]},
        }],
    }
    (OUT / "approximate_training_region.geojson").write_text(json.dumps(geojson, indent=2) + "\n")

    fields = ["uid", "id", "area", "description", "e", "n", "length_m", "width_m", "height_m", "source_file", "source_row"]
    with (OUT / "contacts_inside_approximate_region.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fields} for row in inside])

    overview = decode(0, 0, width - 1, height - 1, *OVERVIEW_SIZE).convert("RGB")
    draw = ImageDraw.Draw(overview)
    draw.line([*SCREEN_POLYGON, SCREEN_POLYGON[0]], fill=(0, 0, 0), width=7, joint="curve")
    for index, sample in enumerate(samples, start=1):
        x = (sample["e"] - left) / (right - left) * OVERVIEW_SIZE[0]
        y = (top - sample["n"]) / (top - bottom) * OVERVIEW_SIZE[1]
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=(255, 32, 32), outline=(255, 255, 255), width=2)
        draw.text((x + 9, y - 14), str(index), fill=(0, 0, 0), stroke_width=2, stroke_fill=(255, 255, 255), font=font(22))
    overview.save(OUT / "overview_polygon_and_five_samples.png")

    panel_w, panel_h, heading_h, row_h = 330, 330, 80, 380
    grid = Image.new("RGB", (210 + panel_w * len(PATCH_SIZES), heading_h + row_h * len(samples)), "white")
    gd = ImageDraw.Draw(grid)
    heading = font(20)
    small = font(15)
    bold = font(18)
    gd.text((12, 22), "Contact", fill="black", font=heading)
    for column, pixels in enumerate(PATCH_SIZES):
        x = 210 + column * panel_w
        gd.text((x + 82, 22), f"{pixels} x {pixels} px", fill="black", font=heading)
        gd.text((x + 88, 48), f"{pixels / 10:g} m x {pixels / 10:g} m", fill="black", font=small)

    manifest = {
        "crs": "EPSG:32630",
        "source": {"ecw": str(ECW), "native_resolution_m": 0.1, "band": 0},
        "polygon": {"screen_vertices_px": SCREEN_POLYGON, "screen_map_box_px": SCREEN_MAP_BOX, "world_vertices_m": polygon},
        "inside_point_contacts": len(inside),
        "sampling": {"count": 5, "seed": SEED, "point_contacts_only": True},
        "samples": [],
    }
    marked_patches: list[tuple[dict, Image.Image]] = []
    for row_index, sample in enumerate(samples):
        y = heading_h + row_index * row_h
        details = f"{row_index + 1}. {sample['id']}\n{sample['area']} | {sample['description']}\nE {sample['e']:.1f}\nN {sample['n']:.1f}\n{sample.get('length_m', '')} x {sample.get('width_m', '')} m"
        gd.multiline_text((12, y + 20), details, fill="black", font=bold, spacing=6)
        item = {key: sample.get(key) for key in fields}
        item["patches"] = []
        # Keep the list coordinate before rounding so the marker represents the
        # supplied contact location, rather than merely the nearest source cell.
        cx_float = (sample["e"] - left) / raster["resolution_x"]
        cy_float = (top - sample["n"]) / raster["resolution_y"]
        cx, cy = round(cx_float), round(cy_float)
        for col, pixels in enumerate(PATCH_SIZES):
            half = pixels // 2
            # Odd/even convention: native source pixel [cx-half, cy-half] is the
            # upper-left corner; centre deviation is no more than 0.05 m.
            x0, y0 = cx - half, cy - half
            patch = decode(x0, y0, x0 + pixels - 1, y0 + pixels - 1, pixels, pixels)
            name = f"{row_index + 1:02d}_{sample['id']}_{pixels}px.png"
            patch.save(OUT / "patches" / name)
            scaled = patch.resize((300, 300), Image.Resampling.NEAREST).convert("RGB")
            px = 210 + col * panel_w + 15
            grid.paste(scaled, (px, y + 35))
            border = ImageDraw.Draw(grid)
            border.rectangle((px, y + 35, px + 299, y + 334), outline="black", width=1)
            item["patches"].append({"pixels": pixels, "metres": pixels / 10, "file": f"patches/{name}", "source_window_px": [x0, y0, x0 + pixels - 1, y0 + pixels - 1]})

        marked_half = MARKED_PATCH_SIZE // 2
        mx0, my0 = cx - marked_half, cy - marked_half
        marked = decode(mx0, my0, mx0 + MARKED_PATCH_SIZE - 1, my0 + MARKED_PATCH_SIZE - 1, MARKED_PATCH_SIZE, MARKED_PATCH_SIZE).convert("RGB")
        marker_x, marker_y = cx_float - mx0, cy_float - my0
        marker = ImageDraw.Draw(marked)
        # The 7-pixel-radius ring marks the contact-list coordinate only; it is
        # not an inferred object footprint.
        marker.ellipse((marker_x - 7, marker_y - 7, marker_x + 7, marker_y + 7), outline=(255, 0, 0), width=2)
        marked_name = f"{row_index + 1:02d}_{sample['id']}_150px_contact_marked.png"
        marked.save(OUT / "marked_150px_patches" / marked_name)
        item["marked_150px"] = {
            "pixels": MARKED_PATCH_SIZE,
            "metres": MARKED_PATCH_SIZE / 10,
            "file": f"marked_150px_patches/{marked_name}",
            "source_window_px": [mx0, my0, mx0 + MARKED_PATCH_SIZE - 1, my0 + MARKED_PATCH_SIZE - 1],
            "contact_list_location_px": [marker_x, marker_y],
            "circle_radius_px": 7,
        }
        marked_patches.append((sample, marked))
        manifest["samples"].append(item)
    grid.save(OUT / "five_contacts_native_pixel_patch_grid.png")

    marked_grid = Image.new("RGB", (720, 110 + 500 * len(marked_patches)), "white")
    mg = ImageDraw.Draw(marked_grid)
    mg.text((14, 22), "Costa Head contacts: 150 x 150 native px (15 x 15 m)", fill="black", font=font(24))
    mg.text((14, 56), "Red ring = coordinate recorded in the official contact list; it is not an object-size label.", fill="black", font=font(16))
    for index, (sample, marked) in enumerate(marked_patches, start=1):
        y = 100 + (index - 1) * 500
        mg.multiline_text((14, y + 24), f"{index}. {sample['id']}\n{sample['area']} | {sample['description']}\nE {sample['e']:.1f}\nN {sample['n']:.1f}", fill="black", font=font(18), spacing=8)
        enlarged = marked.resize((450, 450), Image.Resampling.NEAREST)
        marked_grid.paste(enlarged, (250, y + 20))
        mg.rectangle((250, y + 20, 699, y + 469), outline="black", width=1)
    marked_grid.save(OUT / "five_contacts_150px_contact_locations_marked.png")
    (OUT / "sample_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(OUT), "inside_point_contacts": len(inside), "sample_ids": [x["id"] for x in samples]}, indent=2))


if __name__ == "__main__":
    main()
