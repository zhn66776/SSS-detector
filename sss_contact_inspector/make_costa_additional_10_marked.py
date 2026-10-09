#!/usr/bin/env python3
"""Render ten additional, non-overlapping Costa contact-list samples at 150 px."""
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageStat

import make_costa_training_region_preview as base


ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT / "training_region_preview" / "sample_manifest.json"
OUT = ROOT / "training_region_preview" / "additional_10_150px_marked"
COUNT = 10
SEED = 20260922
PIXELS = 150
# Reject the near-white windows where the ECW has no useful grey coverage.
# Measured before the red location ring is painted.
MIN_GREY_VARIANCE = 450.0


def main() -> None:
    previous = json.loads(PREVIOUS.read_text())
    polygon = [tuple(vertex) for vertex in previous["polygon"]["world_vertices_m"]]
    used = {sample["uid"] for sample in previous["samples"]}
    metadata = json.loads(base.METADATA.read_text())
    raster = metadata["datasets"]["costa"]["rasters"][0]
    left, top = raster["left"], raster["top"]
    res_x, res_y = raster["resolution_x"], raster["resolution_y"]
    candidates = [
        item for item in json.loads(base.CONTACTS.read_text())
        if item["dataset"] == "costa" and item["geometry"] == "point"
        and item["uid"] not in used and base.point_in_polygon(item["e"], item["n"], polygon)
    ]
    shuffled = candidates[:]
    random.Random(SEED).shuffle(shuffled)
    half = PIXELS // 2
    samples = []
    for sample in shuffled:
        cx_float = (sample["e"] - left) / res_x
        cy_float = (top - sample["n"]) / res_y
        cx, cy = round(cx_float), round(cy_float)
        x0, y0 = cx - half, cy - half
        patch = base.decode(x0, y0, x0 + PIXELS - 1, y0 + PIXELS - 1, PIXELS, PIXELS)
        variance = ImageStat.Stat(patch).var[0]
        if variance >= MIN_GREY_VARIANCE:
            samples.append((sample, patch.convert("RGB"), x0, y0, (cx_float - x0, cy_float - y0), variance))
        if len(samples) == COUNT:
            break
    if len(samples) != COUNT:
        raise RuntimeError(f"only found {len(samples)} textured windows from {len(candidates)} eligible contacts")
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    columns, cell_width, row_height, title_height = 2, 700, 500, 105
    rows = (COUNT + columns - 1) // columns
    grid = Image.new("RGB", (columns * cell_width, title_height + rows * row_height), "white")
    draw = ImageDraw.Draw(grid)
    draw.text((14, 20), "Additional Costa contacts: 150 x 150 native px (15 x 15 m)", fill="black", font=base.font(24))
    draw.text((14, 56), "Red ring = official contact-list coordinate; excludes the previous five samples.", fill="black", font=base.font(16))

    manifest = {
        "crs": "EPSG:32630",
        "source": {"ecw": str(base.ECW), "native_resolution_m": 0.1, "band": 0},
        "selection": {"count": COUNT, "seed": SEED, "excluded_previous_uids": sorted(used), "eligible_remaining_point_contacts": len(candidates), "minimum_grey_variance": MIN_GREY_VARIANCE},
        "samples": [],
    }
    for index, (sample, patch, x0, y0, location, variance) in enumerate(samples, start=1):
        marker = ImageDraw.Draw(patch)
        marker.ellipse((location[0] - 7, location[1] - 7, location[0] + 7, location[1] + 7), outline=(255, 0, 0), width=2)
        name = f"{index:02d}_{sample['id']}_150px_contact_marked.png"
        patch.save(OUT / name)

        column, row = (index - 1) % columns, (index - 1) // columns
        origin_x, origin_y = column * cell_width, title_height + row * row_height
        draw.multiline_text((origin_x + 14, origin_y + 24), f"{index}. {sample['id']}\n{sample['area']} | {sample['description']}\nE {sample['e']:.1f}\nN {sample['n']:.1f}", fill="black", font=base.font(18), spacing=8)
        enlarged = patch.resize((450, 450), Image.Resampling.NEAREST)
        grid.paste(enlarged, (origin_x + 230, origin_y + 20))
        draw.rectangle((origin_x + 230, origin_y + 20, origin_x + 679, origin_y + 469), outline="black", width=1)
        manifest["samples"].append({
            "uid": sample["uid"], "id": sample["id"], "area": sample["area"], "description": sample["description"],
            "e": sample["e"], "n": sample["n"], "length_m": sample.get("length_m"), "width_m": sample.get("width_m"),
            "file": name, "source_window_px": [x0, y0, x0 + PIXELS - 1, y0 + PIXELS - 1],
            "contact_list_location_px": location, "circle_radius_px": 7, "grey_variance_before_marker": variance,
        })
    grid.save(OUT / "additional_ten_150px_contact_locations_marked.png")
    (OUT / "additional_sample_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(OUT), "sample_ids": [sample[0]["id"] for sample in samples]}, indent=2))


if __name__ == "__main__":
    main()
