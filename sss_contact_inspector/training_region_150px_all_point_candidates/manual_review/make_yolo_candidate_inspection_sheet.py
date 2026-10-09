#!/usr/bin/env python3
from pathlib import Path
import csv
from PIL import Image, ImageDraw, ImageFont

REVIEW = Path(__file__).resolve().parent
DATASET = REVIEW.parent
IDS = ["S3629", "S2537", "S2594", "S3685", "S3454", "S3049", "S3652", "S190", "S2234", "S182"]
OUT = REVIEW / "yolo_candidate_inspection_sheet.png"

def font(size):
    for candidate in ("/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"):
        path = Path(candidate)
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size)
            except OSError:
                pass
    return ImageFont.load_default()

def main():
    rows = {r["id"]: r for r in csv.DictReader((DATASET / "accepted_contacts.csv").open(newline=""))}
    cell_w, cell_h = 650, 370
    sheet = Image.new("RGB", (cell_w * 2, cell_h * 5), (246, 247, 248))
    draw = ImageDraw.Draw(sheet)
    for i, id_ in enumerate(IDS):
        row = rows[id_]
        col, line = i % 2, i // 2
        ox, oy = col * cell_w, line * cell_h
        draw.text((ox + 10, oy + 8), f"{id_}  variance={float(row['grey_variance']):.1f}", fill=(25, 30, 35), font=font(17))
        for panel, key in enumerate(("raw_patch_file", "preview_file")):
            path = DATASET / row[key]
            image = Image.open(path).convert("RGB").resize((300, 300), Image.Resampling.NEAREST)
            x, y = ox + 10 + panel * 315, oy + 40
            sheet.paste(image, (x, y))
            d = ImageDraw.Draw(sheet)
            for value in range(0, 151, 10):
                px = x + value * 2
                py = y + value * 2
                d.line((px, y, px, y + 300), fill=(120, 120, 120), width=1)
                d.line((x, py, x + 300, py), fill=(120, 120, 120), width=1)
            d.rectangle((x, y, x + 299, y + 299), outline=(80, 85, 90), width=1)
            d.text((x + 3, y + 3), "raw" if panel == 0 else "preview", fill=(255, 0, 0), font=font(12))
    sheet.save(OUT, format="PNG", optimize=True)
    print(OUT)

if __name__ == "__main__":
    main()
