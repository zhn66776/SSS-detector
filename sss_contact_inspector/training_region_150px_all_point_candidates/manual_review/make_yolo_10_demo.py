#!/usr/bin/env python3
"""Create ten review-only YOLO box examples using the stone-plus-shadow rule."""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REVIEW_DIR = Path(__file__).resolve().parent
DATASET_DIR = REVIEW_DIR.parent
OUT_DIR = REVIEW_DIR / "yolo_demo_10"
SHEET = REVIEW_DIR / "yolo_demo_10_sheet.png"

# Continuous pixel edges [xmin, ymin, xmax, ymax).  The boxes include the
# compact stone response and the bright upper-left acoustic shadow/tail.
BOXES = {
    "S2932": (40, 50, 85, 85),
    "S3629": (48, 47, 86, 90),
    "S2537": (39, 45, 85, 90),
    "S2594": (45, 47, 86, 91),
    "S3685": (45, 50, 86, 92),
    "S3454": (52, 52, 88, 92),
    "S3049": (48, 50, 88, 93),
    "S3652": (45, 50, 90, 94),
    "S2234": (50, 52, 88, 95),
    "S182": (48, 50, 90, 92),
}

CONFIDENCE = {
    "S2932": "high",
    "S3629": "medium",
    "S2537": "high",
    "S2594": "high",
    "S3685": "medium",
    "S3454": "medium",
    "S3049": "medium",
    "S3652": "medium",
    "S2234": "medium",
    "S182": "low_line_like_background",
}


def font(size: int):
    for candidate in ("/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"):
        path = Path(candidate)
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size)
            except OSError:
                pass
    return ImageFont.load_default()


def draw_box(path: Path, box: tuple[int, int, int, int], size: int) -> Image.Image:
    image = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.NEAREST)
    scale = size / 150.0
    draw = ImageDraw.Draw(image)
    draw.rectangle(tuple(round(value * scale) for value in box), outline=(35, 220, 80), width=max(2, round(2 * scale / 4)))
    return image


def main() -> None:
    rows = {row["id"]: row for row in csv.DictReader((DATASET_DIR / "accepted_contacts.csv").open(newline=""))}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cards: list[tuple[str, Image.Image, Image.Image]] = []
    manifest_rows = []
    for id_, box in BOXES.items():
        row = rows[id_]
        raw = draw_box(DATASET_DIR / row["raw_patch_file"], box, 260)
        preview = draw_box(DATASET_DIR / row["preview_file"], box, 260)
        raw.save(OUT_DIR / f"{id_}_raw_boxed.png", format="PNG", optimize=True)
        preview.save(OUT_DIR / f"{id_}_preview_boxed.png", format="PNG", optimize=True)
        xmin, ymin, xmax, ymax = box
        yolo = (0, (xmin + xmax) / 2 / 150, (ymin + ymax) / 2 / 150, (xmax - xmin) / 150, (ymax - ymin) / 150)
        (OUT_DIR / f"{id_}.txt").write_text(" ".join(f"{value:.6f}" if isinstance(value, float) else str(value) for value in yolo) + "\n", encoding="utf-8")
        manifest_rows.append({
            "id": id_, "uid": row["uid"], "pixel_box": f"{xmin},{ymin},{xmax},{ymax}",
            "yolo": " ".join(f"{value:.6f}" if isinstance(value, float) else str(value) for value in yolo),
            "confidence": CONFIDENCE[id_], "note": "stone response plus upper-left bright acoustic shadow/tail; review-only candidate",
        })
        cards.append((id_, raw, preview))

    with (OUT_DIR / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader(); writer.writerows(manifest_rows)

    cell_w, cell_h = 620, 350
    sheet = Image.new("RGB", (cell_w * 2, cell_h * 5), (246, 247, 248))
    draw = ImageDraw.Draw(sheet)
    for index, (id_, raw, preview) in enumerate(cards):
        col, row_index = index % 2, index // 2
        ox, oy = col * cell_w, row_index * cell_h
        draw.text((ox + 10, oy + 8), f"{id_}  confidence={CONFIDENCE[id_]}", fill=(25, 30, 35), font=font(16))
        sheet.paste(raw, (ox + 10, oy + 40)); sheet.paste(preview, (ox + 300, oy + 40))
        draw.rectangle((ox + 10, oy + 40, ox + 269, oy + 299), outline=(80, 85, 90), width=1)
        draw.rectangle((ox + 300, oy + 40, ox + 559, oy + 299), outline=(80, 85, 90), width=1)
        draw.text((ox + 14, oy + 305), "raw + green box", fill=(60, 65, 70), font=font(12))
        draw.text((ox + 304, oy + 305), "preview + red coordinate", fill=(60, 65, 70), font=font(12))
    sheet.save(SHEET, format="PNG", optimize=True)
    print({"sheet": str(SHEET), "output_dir": str(OUT_DIR), "count": len(cards)})


if __name__ == "__main__":
    main()
