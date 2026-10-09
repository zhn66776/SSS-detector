#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
OUT = ROOT / "manual_review" / "validation_50_contact_overlap_yolo_demo"
INDEX = ROOT / "patch_index.csv"
CONTACTS = PROJECT / "data" / "contacts.json"
METADATA = PROJECT / "data" / "metadata.json"
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
    index_rows = list(csv.DictReader(INDEX.open(newline="")))
    val = {(int(r["x0"]), int(r["y0"])): r for r in index_rows if r["split"] == "val"}
    raster = json.loads(METADATA.read_text())["datasets"]["costa"]["rasters"][0]
    contacts = json.loads(CONTACTS.read_text())
    overlap = defaultdict(list)
    for contact in contacts:
        if contact.get("dataset") != "costa" or contact.get("geometry") != "point":
            continue
        px = (float(contact["e"]) - raster["left"]) / raster["resolution_x"]
        py = (raster["top"] - float(contact["n"])) / raster["resolution_y"]
        key = (int(px // 100) * 100, int(py // 100) * 100)
        if key in val:
            overlap[key].append(contact)
    candidates = [val[key] for key in sorted(overlap)]
    if len(candidates) < COUNT:
        raise RuntimeError(f"only {len(candidates)} validation patches overlap a contact")
    selected = random.Random(SEED).sample(candidates, COUNT)
    selected.sort(key=lambda r: r["patch_id"])
    OUT.mkdir(parents=True, exist_ok=True)
    fields = list(selected[0]) + ["selection_contact_id", "selection_contact_uid"]
    with (OUT / "selected_validation_50_contact_overlap.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in selected:
            key = (int(row["x0"]), int(row["y0"]))
            contact = overlap[key][0]
            output = dict(row)
            output["selection_contact_id"] = contact["id"]
            output["selection_contact_uid"] = contact["uid"]
            writer.writerow(output)

    columns, thumb, cell, title = 10, 240, 260, 48
    sheet = Image.new("RGB", (columns * cell, title + ((COUNT + columns - 1) // columns) * cell), (246, 247, 248))
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 10), f"Validation contact-overlap sample: {COUNT} raw patches; seed={SEED}; no circles/IDs shown", fill=(25, 30, 35), font=font(16))
    for index, row in enumerate(selected, start=1):
        with Image.open(ROOT / row["file_relative_path"]) as image:
            thumb_image = image.convert("RGB").resize((thumb, thumb), Image.Resampling.NEAREST)
        col, row_index = (index - 1) % columns, (index - 1) // columns
        x, y = col * cell + 10, title + row_index * cell + 10
        sheet.paste(thumb_image, (x, y))
        draw.rectangle((x, y, x + thumb - 1, y + thumb - 1), outline=(80, 85, 90), width=1)
        draw.text((x + 3, y + 3), str(index), fill=(255, 0, 0), font=font(12))
    sheet.save(OUT / "validation_50_contact_overlap_raw_sheet.png", format="PNG", optimize=True)
    print({"count": len(selected), "candidate_pool": len(candidates), "seed": SEED, "sheet": str(OUT / "validation_50_contact_overlap_raw_sheet.png")})


if __name__ == "__main__":
    main()
