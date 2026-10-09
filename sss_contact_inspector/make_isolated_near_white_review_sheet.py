#!/usr/bin/env python3
"""Make a visual review sheet for the 21 isolated near-white candidates."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "yolo_manual_annotation_grid_150px_stride100_v1_cleaned"
QUARANTINE = DATASET / "quarantine_near_white.csv"
TRAIN_POLYGON = ROOT / "training_split_design" / "train_patch_centres_150px.geojson"
REVIEW_DIR = DATASET / "review"
OUTPUT = REVIEW_DIR / "isolated_near_white_review_sheet.png"


def load_font(size: int):
    for candidate in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                pass
    return ImageFont.load_default()


def world_to_map(easting: float, northing: float, bounds: tuple[float, float, float, float], area: tuple[int, int, int, int]) -> tuple[float, float]:
    min_e, min_n, max_e, max_n = bounds
    left, top, width, height = area
    scale = min(width / (max_e - min_e), height / (max_n - min_n))
    used_w = (max_e - min_e) * scale
    used_h = (max_n - min_n) * scale
    offset_x = left + (width - used_w) / 2
    offset_y = top + (height - used_h) / 2
    return offset_x + (easting - min_e) * scale, offset_y + (max_n - northing) * scale


def main() -> None:
    with QUARANTINE.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if int(row["component_size"]) == 1]
    rows.sort(key=lambda row: (int(row["y0"]), int(row["x0"])))
    if len(rows) != 21:
        raise RuntimeError(f"expected 21 isolated candidates, found {len(rows)}")

    polygon_doc = json.loads(TRAIN_POLYGON.read_text(encoding="utf-8"))
    geometry = polygon_doc["features"][0]["geometry"]
    if geometry["type"] != "Polygon":
        raise RuntimeError("train patch-centre geometry is not a Polygon")
    ring = [(float(e), float(n)) for e, n in geometry["coordinates"][0]]
    min_e = min(point[0] for point in ring)
    max_e = max(point[0] for point in ring)
    min_n = min(point[1] for point in ring)
    max_n = max(point[1] for point in ring)
    bounds = (min_e, min_n, max_e, max_n)

    cards_per_row = 3
    card_width = 260
    card_height = 245
    title_height = 92
    map_width = 430
    gap = 22
    map_left = cards_per_row * card_width + gap
    width = map_left + map_width + 28
    rows_of_cards = (len(rows) + cards_per_row - 1) // cards_per_row
    height = title_height + rows_of_cards * card_height + 24
    canvas = Image.new("RGB", (width, height), (248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(28)
    subtitle_font = load_font(16)
    label_font = load_font(15)
    small_font = load_font(12)

    draw.text((20, 14), "21 isolated near-white train patches", fill=(15, 15, 15), font=title_font)
    draw.text(
        (20, 52),
        "Left: raw 150×150 PNGs (display-enlarged). Right: train patch-centre polygon; numbers match cards.",
        fill=(65, 65, 65),
        font=subtitle_font,
    )

    map_area = (map_left + 18, title_height + 12, map_width - 36, height - title_height - 36)
    map_box = (map_area[0] - 12, map_area[1] - 12, map_area[0] + map_area[2] + 12, map_area[1] + map_area[3] + 12)
    draw.rounded_rectangle(map_box, radius=12, fill=(232, 244, 235), outline=(40, 90, 50), width=2)
    polygon_pixels = [world_to_map(e, n, bounds, map_area) for e, n in ring]
    draw.polygon(polygon_pixels, fill=(205, 239, 214), outline=(20, 60, 25))
    draw.line(polygon_pixels, fill=(20, 60, 25), width=3, joint="curve")
    draw.text((map_box[0] + 12, map_box[1] + 10), "train patch-centre area", fill=(20, 65, 25), font=label_font)
    draw.text((map_box[0] + 12, map_box[3] - 25), "E →   N ↑", fill=(55, 75, 55), font=small_font)

    for index, row in enumerate(rows, start=1):
        easting = float(row["centre_easting"])
        northing = float(row["centre_northing"])
        px, py = world_to_map(easting, northing, bounds, map_area)
        draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=(165, 165, 165), outline=(30, 30, 30))
        draw.text((px + 6, py - 9), str(index), fill=(40, 40, 40), font=small_font)

    for index, row in enumerate(rows, start=1):
        column = (index - 1) % cards_per_row
        card_row = (index - 1) // cards_per_row
        card_x = column * card_width + 12
        card_y = title_height + card_row * card_height
        draw.rounded_rectangle(
            (card_x, card_y, card_x + card_width - 10, card_y + card_height - 10),
            radius=10,
            fill=(255, 255, 255),
            outline=(185, 185, 185),
            width=1,
        )
        source = DATASET / row["quarantine_file_relative_path"]
        with Image.open(source) as image:
            image.load()
            display = image.resize((190, 190), Image.Resampling.NEAREST).convert("RGB")
        image_x = card_x + 30
        image_y = card_y + 8
        canvas.paste(display, (image_x, image_y))
        draw.rectangle((image_x, image_y, image_x + 189, image_y + 189), outline=(70, 70, 70), width=1)
        label = f"#{index}  x{int(row['x0'])} y{int(row['y0'])}"
        draw.text((card_x + 10, card_y + 202), label, fill=(15, 15, 15), font=label_font)
        coord = f"E {float(row['centre_easting']):.1f}  N {float(row['centre_northing']):.1f}"
        draw.text((card_x + 10, card_y + 221), coord, fill=(70, 70, 70), font=small_font)
        px, py = world_to_map(float(row["centre_easting"]), float(row["centre_northing"]), bounds, map_area)
        draw.ellipse((px - 7, py - 7, px + 7, py + 7), fill=(215, 40, 35), outline=(90, 0, 0), width=2)
        draw.text((px + 8, py - 10), str(index), fill=(130, 0, 0), font=label_font)

    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT, format="PNG", optimize=True)
    print(OUTPUT)


if __name__ == "__main__":
    main()
