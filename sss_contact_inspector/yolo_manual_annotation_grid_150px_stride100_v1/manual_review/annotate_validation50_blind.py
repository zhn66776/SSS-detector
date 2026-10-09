#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "manual_review" / "validation_50_contact_overlap_yolo_demo"
OUT = REVIEW / "blind_yolo_annotations"
SELECTED = REVIEW / "selected_validation_50_contact_overlap.csv"
SHEET = REVIEW / "blind_yolo_50_sheet.png"

# Pixel edges [xmin, ymin, xmax, ymax). These are visual candidates based on
# raw images only; the selection contact coordinate is not used to place them.
BOXES = {
    "costa_val_x068100_y048600": (58, 42, 99, 103),
    "costa_val_x068600_y050500": (35, 38, 92, 103),
    "costa_val_x068800_y054000": (30, 26, 92, 88),
    "costa_val_x068900_y051500": (35, 18, 92, 70),
    "costa_val_x069300_y047500": (70, 44, 132, 112),
    "costa_val_x071200_y044200": (60, 18, 121, 75),
    "costa_val_x071400_y047300": (35, 42, 98, 105),
    "costa_val_x072300_y049400": (24, 20, 82, 78),
    "costa_val_x072700_y043300": (8, 22, 63, 80),
    "costa_val_x072900_y043200": (38, 8, 99, 67),
}
CLASS_ID = 0


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
    rows = list(csv.DictReader(SELECTED.open(newline="")))
    OUT.mkdir(parents=True, exist_ok=True)
    raw_out = OUT / "images_raw"
    boxed_out = OUT / "images_boxed_review"
    labels_out = OUT / "labels"
    raw_out.mkdir(exist_ok=True)
    boxed_out.mkdir(exist_ok=True)
    labels_out.mkdir(exist_ok=True)
    output_rows = []
    cards = []
    for row in rows:
        patch_id = row["patch_id"]
        source = ROOT / row["file_relative_path"]
        raw_image = Image.open(source).convert("L")
        raw_image.save(raw_out / f"{patch_id}.png", format="PNG", optimize=True)
        image = raw_image.convert("RGB")
        box = BOXES.get(patch_id)
        status = "candidate_context" if box else "no_clear_context_or_line_like_background"
        confidence = "medium" if box else "none"
        if box:
            ImageDraw.Draw(image).rectangle(box, outline=(35, 220, 80), width=2)
            xmin, ymin, xmax, ymax = box
            yolo = (CLASS_ID, (xmin + xmax) / 2 / 150, (ymin + ymax) / 2 / 150, (xmax - xmin) / 150, (ymax - ymin) / 150)
            label_text = " ".join(f"{value:.6f}" if isinstance(value, float) else str(value) for value in yolo) + "\n"
        else:
            label_text = ""
        image_path = boxed_out / f"{patch_id}.png"
        label_path = labels_out / f"{patch_id}.txt"
        image.save(image_path, format="PNG", optimize=True)
        label_path.write_text(label_text, encoding="utf-8")
        output_rows.append({
            "patch_id": patch_id,
            "split": "val",
            "selection_contact_id": row["selection_contact_id"],
            "status": status,
            "confidence": confidence,
            "pixel_box": "" if not box else ",".join(map(str, box)),
            "label_file": label_path.name,
            "image_file": image_path.name,
            "note": "candidate box includes visible dark response and adjacent bright shadow/tail; box drawn without using contact coordinate",
        })
        cards.append((patch_id, image, status))

    with (OUT / "annotation_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader(); writer.writerows(output_rows)

    cell_w, cell_h = 260, 290
    sheet = Image.new("RGB", (cell_w * 10, cell_h * 5), (246, 247, 248))
    draw = ImageDraw.Draw(sheet)
    for index, (patch_id, image, status) in enumerate(cards, start=1):
        col, row_index = (index - 1) % 10, (index - 1) // 10
        x, y = col * cell_w + 5, row_index * cell_h + 5
        thumb = image.resize((240, 240), Image.Resampling.NEAREST)
        sheet.paste(thumb, (x, y + 22))
        color = (35, 160, 70) if status == "candidate_context" else (120, 125, 130)
        draw.rectangle((x, y + 22, x + 239, y + 261), outline=color, width=2)
        draw.text((x + 3, y + 3), f"{index} {patch_id[-16:]}", fill=(25, 30, 35), font=font(11))
    sheet.save(SHEET, format="PNG", optimize=True)
    print({"total": len(rows), "boxed": len(BOXES), "empty": len(rows) - len(BOXES), "output": str(OUT), "sheet": str(SHEET)})


if __name__ == "__main__":
    main()
