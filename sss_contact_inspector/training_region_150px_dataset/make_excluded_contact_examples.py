#!/usr/bin/env python3
"""Make a small visual audit gallery for representative rejected windows."""

from __future__ import annotations

import base64
import csv
import html
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT = Path(__file__).resolve().parent
VIZ_DIR = Path(
    "/Users/haonanzhang/.codex/visualizations/2026/09/21/01a0c180-63ff-7ba3-bbb1-7f87669b5860"
)
PNG_PATH = OUT / "excluded_contact_examples.png"
HTML_PATH = VIZ_DIR / "costa-excluded-contact-examples.html"
SKIPPED_PATH = OUT / "skipped_contacts.csv"


def load_font(size: int):
    for path in (
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        candidate = Path(path)
        if candidate.exists():
            try:
                return ImageFont.truetype(str(candidate), size)
            except OSError:
                pass
    return ImageFont.load_default()


def choose_examples(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups = {
        "grey_variance_below_450": [
            row for row in rows if row["skip_reason"] == "grey_variance_below_450"
        ],
        "grey_variance_below_450_and_mean_at_or_above_250": [
            row
            for row in rows
            if row["skip_reason"] == "grey_variance_below_450_and_mean_at_or_above_250"
        ],
    }
    selected: list[dict[str, str]] = []
    for reason, group in groups.items():
        ordered = sorted(group, key=lambda row: (float(row["grey_variance"]), float(row["grey_mean"])))
        positions = [0, len(ordered) // 2, len(ordered) - 1]
        for position in positions:
            row = dict(ordered[position])
            row["gallery_reason"] = reason
            selected.append(row)
    return selected


def make_patch_images(row: dict[str, str], exporter):
    _, raster = exporter.load_raster_metadata()
    e = float(row["e"])
    n = float(row["n"])
    window = exporter.window_for(e, n, raster)
    pixels = exporter.decode_window_pixels(window["x0"], window["y0"], window["x1"], window["y1"])
    raw = Image.frombytes("L", (exporter.PATCH_SIZE, exporter.PATCH_SIZE), pixels)
    preview = raw.convert("RGB")
    marker = ImageDraw.Draw(preview)
    marker.ellipse(
        (
            window["patch_x"] - exporter.MARKER_RADIUS_PX,
            window["patch_y"] - exporter.MARKER_RADIUS_PX,
            window["patch_x"] + exporter.MARKER_RADIUS_PX,
            window["patch_y"] + exporter.MARKER_RADIUS_PX,
        ),
        outline=(255, 0, 0),
        width=exporter.MARKER_WIDTH_PX,
    )
    return raw, preview, window


def draw_gallery(rows: list[dict[str, str]], images: list[tuple[Image.Image, Image.Image, dict]]) -> None:
    width = 1300
    card_width = 625
    card_height = 270
    left_margin = 20
    gap = 20
    top = 175
    height = top + math.ceil(len(rows) / 2) * (card_height + 18) + 30
    canvas = Image.new("RGB", (width, height), (246, 247, 248))
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(28)
    subtitle_font = load_font(17)
    label_font = load_font(16)
    body_font = load_font(15)
    small_font = load_font(13)

    draw.text((20, 18), "Costa Head: representative examples of 1,011 rejected contacts", fill=(25, 30, 35), font=title_font)
    draw.text(
        (20, 60),
        "Each card: unmarked raw grayscale window on the left; review preview with coordinate ring on the right.",
        fill=(75, 82, 90),
        font=subtitle_font,
    )

    bars = [
        ("Variance < 450 only", 139, (210, 128, 40)),
        ("Variance < 450 and mean >= 250", 872, (165, 75, 72)),
    ]
    max_count = max(value for _, value, _ in bars)
    bar_x = 20
    bar_y = 108
    for label, value, color in bars:
        draw.text((bar_x, bar_y), f"{label}: {value}", fill=(45, 50, 55), font=small_font)
        draw.rectangle((bar_x + 290, bar_y + 2, bar_x + 290 + int(700 * value / max_count), bar_y + 19), fill=color)
        bar_y += 27

    for index, (row, (raw, preview, window)) in enumerate(zip(rows, images), start=1):
        column = (index - 1) % 2
        row_number = (index - 1) // 2
        x = left_margin + column * (card_width + gap)
        y = top + row_number * (card_height + 18)
        reason = row["gallery_reason"]
        reason_label = "A · variance < 450" if reason == "grey_variance_below_450" else "B · variance < 450 + near-white mean"
        accent = (210, 128, 40) if reason == "grey_variance_below_450" else (165, 75, 72)
        draw.rectangle((x, y, x + card_width, y + card_height), outline=(190, 195, 200), width=1)
        draw.rectangle((x, y, x + 7, y + card_height), fill=accent)
        draw.text((x + 18, y + 12), f"Example {index}: {reason_label}", fill=(25, 30, 35), font=label_font)
        draw.text((x + 18, y + 38), f"{row['uid']}  |  {row['id']}", fill=(75, 82, 90), font=small_font)

        image_size = 158
        raw_rgb = raw.convert("RGB").resize((image_size, image_size), Image.Resampling.NEAREST)
        preview_rgb = preview.resize((image_size, image_size), Image.Resampling.NEAREST)
        image_y = y + 67
        raw_x = x + 18
        preview_x = x + 192
        canvas.paste(raw_rgb, (raw_x, image_y))
        canvas.paste(preview_rgb, (preview_x, image_y))
        draw.rectangle((raw_x, image_y, raw_x + image_size - 1, image_y + image_size - 1), outline=(100, 105, 110), width=1)
        draw.rectangle((preview_x, image_y, preview_x + image_size - 1, image_y + image_size - 1), outline=(100, 105, 110), width=1)
        draw.text((raw_x, image_y + image_size + 4), "raw, unmarked", fill=(65, 70, 75), font=small_font)
        draw.text((preview_x, image_y + image_size + 4), "preview, red ring", fill=(65, 70, 75), font=small_font)

        detail_x = x + 370
        detail_y = y + 76
        details = [
            f"grey mean     {float(row['grey_mean']):.3f}",
            f"grey variance {float(row['grey_variance']):.3f}",
            f"contact px    ({window['patch_x']:.3f}, {window['patch_y']:.3f})",
            f"source row    {row['source_row']}",
            f"area          {row['area']}",
        ]
        for line in details:
            draw.text((detail_x, detail_y), line, fill=(45, 50, 55), font=body_font)
            detail_y += 25

    draw.text(
        (20, height - 24),
        "These are explanatory examples only; rejected windows were not written to patches_raw/.",
        fill=(85, 90, 95),
        font=small_font,
    )
    canvas.save(PNG_PATH, format="PNG", optimize=True)


def make_html(rows: list[dict[str, str]], images: list[tuple[Image.Image, Image.Image, dict]]) -> None:
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    image_data = base64.b64encode(PNG_PATH.read_bytes()).decode("ascii")
    reason_text = "；".join(
        [
            "Variance < 450: 139 contacts",
            "Variance < 450 and mean >= 250: 872 contacts",
        ]
    )
    markup = f'''<div id="costa-excluded-contact-examples">
<style>
#costa-excluded-contact-examples {{
  color: var(--foreground);
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  max-width: 1024px;
}}
#costa-excluded-contact-examples h2 {{
  margin: 0 0 6px;
  font-size: 18px;
  font-weight: 500;
}}
#costa-excluded-contact-examples p {{
  margin: 0 0 12px;
  color: var(--muted-foreground);
  font-size: 13px;
}}
#costa-excluded-contact-examples img {{
  display: block;
  max-width: 100%;
  height: auto;
  border: 1px solid var(--border);
}}
</style>
<h2>Representative windows for excluded contacts</h2>
<p>{html.escape(reason_text)}. Left: unmarked source image. Right: red circles indicate survey coordinates only.</p>
<img src="data:image/png;base64,{image_data}" alt="Six representative rejected Costa Head contact windows, each showing an unmarked raw grayscale patch beside a red-ring review preview, with grey mean and variance." />
</div>'''
    HTML_PATH.write_text(markup, encoding="utf-8")


def main() -> None:
    rows = list(csv.DictReader(SKIPPED_PATH.open(encoding="utf-8", newline="")))
    selected = choose_examples(rows)
    sys.path.insert(0, str(OUT))
    import export_dataset as exporter

    images = [make_patch_images(row, exporter) for row in selected]
    draw_gallery(selected, images)
    make_html(selected, images)
    if exporter._ECW_WORKER:
        exporter._ECW_WORKER.close()
    print(
        {
            "png": str(PNG_PATH),
            "html": str(HTML_PATH),
            "examples": [row["uid"] for row in selected],
            "html_bytes": HTML_PATH.stat().st_size,
        }
    )


if __name__ == "__main__":
    main()
