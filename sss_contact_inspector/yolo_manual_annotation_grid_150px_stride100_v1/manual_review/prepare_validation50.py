#!/usr/bin/env python3
from __future__ import annotations

import csv
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "manual_review" / "validation_50_yolo_demo"
INDEX = ROOT / "patch_index.csv"
SEED = 20260921
COUNT = 50


def font(size: int):
    for candidate in ("/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"):
        path = Path(candidate)
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size)
            except OSError:
                pass
    return ImageFont.load_default()


def main() -> None:
    rows = [row for row in csv.DictReader(INDEX.open(newline="")) if row["split"] == "val"]
    selected = random.Random(SEED).sample(rows, COUNT)
    selected.sort(key=lambda row: row["patch_id"])
    with (OUT / "selected_validation_50.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader(); writer.writerows(selected)

    columns, thumb, cell, title = 10, 240, 260, 48
    sheet = Image.new("RGB", (columns * cell, title + ((COUNT + columns - 1) // columns) * cell), (246, 247, 248))
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 10), f"Validation blind review sample: {COUNT} patches; seed={SEED}; raw only", fill=(25, 30, 35), font=font(16))
    for index, row in enumerate(selected, start=1):
        image_path = ROOT / row["file_relative_path"]
        with Image.open(image_path) as image:
            thumb_image = image.convert("RGB").resize((thumb, thumb), Image.Resampling.NEAREST)
        col, row_index = (index - 1) % columns, (index - 1) // columns
        x, y = col * cell + 10, title + row_index * cell + 10
        sheet.paste(thumb_image, (x, y))
        draw.rectangle((x, y, x + thumb - 1, y + thumb - 1), outline=(80, 85, 90), width=1)
        draw.text((x + 3, y + 3), str(index), fill=(255, 0, 0), font=font(12))
    sheet.save(OUT / "validation_50_raw_inspection_sheet.png", format="PNG", optimize=True)
    print({"count": len(selected), "seed": SEED, "sheet": str(OUT / "validation_50_raw_inspection_sheet.png")})


if __name__ == "__main__":
    main()
