#!/usr/bin/env python3
"""Create a non-training YOLO box demo for S2932."""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


REVIEW_DIR = Path(__file__).resolve().parent
DATASET_DIR = REVIEW_DIR.parent
RAW = DATASET_DIR / "patches_raw" / "costa_afl_S2932_1974.png"
PREVIEW = DATASET_DIR / "patches_preview_marked" / "costa_afl_S2932_1974.png"
OUTPUT = REVIEW_DIR / "S2932_yolo_box_demo.png"
LABEL = REVIEW_DIR / "S2932_yolo_candidate.txt"

# Candidate visual box: includes the compact dark response and its adjacent
# acoustic shadow, while avoiding the larger diagonal background stripes.
XMIN, YMIN, XMAX, YMAX = 40, 50, 85, 85
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


def add_box(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((XMIN, YMIN, XMAX, YMAX), outline=(40, 220, 80), width=2)
    return image


def main() -> None:
    raw = add_box(RAW).resize((600, 600), Image.Resampling.NEAREST)
    preview = add_box(PREVIEW).resize((600, 600), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (1260, 760), (246, 247, 248))
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 16), "S2932 YOLO candidate box: stone response + acoustic shadow", fill=(25, 30, 35), font=font(24))
    draw.text((20, 52), "Green = proposed object box; red ring in the preview = official contact coordinate only.", fill=(75, 82, 90), font=font(15))
    canvas.paste(raw, (20, 90))
    canvas.paste(preview, (640, 90))
    draw.rectangle((20, 90, 619, 689), outline=(80, 85, 90), width=1)
    draw.rectangle((640, 90, 1239, 689), outline=(80, 85, 90), width=1)
    draw.text((24, 700), "raw, unmarked", fill=(55, 60, 65), font=font(14))
    draw.text((644, 700), "preview, red coordinate ring", fill=(55, 60, 65), font=font(14))
    canvas.save(OUTPUT, format="PNG", optimize=True)

    # YOLO uses the continuous box edges [xmin, xmax) and [ymin, ymax).
    width = XMAX - XMIN
    height = YMAX - YMIN
    x_center = (XMIN + XMAX) / 2 / 150
    y_center = (YMIN + YMAX) / 2 / 150
    norm_width = width / 150
    norm_height = height / 150
    LABEL.write_text(f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n", encoding="utf-8")
    print({"image": str(OUTPUT), "label": str(LABEL), "pixel_box": [XMIN, YMIN, XMAX, YMAX], "yolo": [CLASS_ID, x_center, y_center, norm_width, norm_height]})


if __name__ == "__main__":
    main()
