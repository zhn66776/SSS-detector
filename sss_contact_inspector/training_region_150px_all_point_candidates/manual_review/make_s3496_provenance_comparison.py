#!/usr/bin/env python3
"""Compare the two conflicting official S3496 listing coordinates."""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REVIEW_DIR = Path(__file__).resolve().parent
DATASET_DIR = REVIEW_DIR.parent
sys.path.insert(0, str(DATASET_DIR.parent / "training_region_150px_dataset"))
import export_dataset as base  # noqa: E402


OUTPUT = REVIEW_DIR / "S3496_provenance_comparison.png"


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
    _, raster = base.load_raster_metadata()
    sources = [
        {
            "label": "Current contacts.json / CSV source row 2538",
            "source": "C13007a_Costa_Afl_Single_Target_Listing.csv",
            "e": 488779.1,
            "n": 6564020.9,
            "size": "3.0 x 0.8 x 0.6 m",
        },
        {
            "label": "GIS SSS Listing DBF record 3496",
            "source": "C13007a_CostaHead_SSSListing.dbf",
            "e": 486899.41,
            "n": 6563943.41,
            "size": "2.4 x 0.5 x 0.5 m",
        },
    ]
    canvas = Image.new("RGB", (980, 500), (246, 247, 248))
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 16), "S3496: two official listing coordinates", fill=(25, 30, 35), font=font(25))
    draw.text((20, 52), "Both use UTM zone 30N; they are different listing records/versions, not a red-circle rendering issue.", fill=(75, 82, 90), font=font(15))
    for index, source in enumerate(sources):
        y = 95 + index * 190
        window = base.window_for(source["e"], source["n"], raster)
        pixels = base.decode_window_pixels(window["x0"], window["y0"], window["x1"], window["y1"])
        raw, preview = base.make_preview(pixels, window)
        size = 150
        canvas.paste(raw.convert("RGB").resize((size, size), Image.Resampling.NEAREST), (20, y))
        canvas.paste(preview.resize((size, size), Image.Resampling.NEAREST), (190, y))
        draw.rectangle((20, y, 169, y + 149), outline=(90, 95, 100), width=1)
        draw.rectangle((190, y, 339, y + 149), outline=(90, 95, 100), width=1)
        draw.text((20, y + 153), "raw", fill=(70, 75, 80), font=font(13))
        draw.text((190, y + 153), "preview", fill=(70, 75, 80), font=font(13))
        details = [
            source["label"],
            source["source"],
            f"E/N: {source['e']}, {source['n']}",
            f"size: {source['size']}",
            f"window: {window['x0']},{window['y0']} to {window['x1']},{window['y1']}",
        ]
        for line_index, line in enumerate(details):
            draw.text((390, y + 10 + line_index * 25), line, fill=(40, 45, 50), font=font(15 if line_index == 0 else 13))
    canvas.save(OUTPUT, format="PNG", optimize=True)
    if base._ECW_WORKER:
        base._ECW_WORKER.close()
    print(OUTPUT)


if __name__ == "__main__":
    main()
