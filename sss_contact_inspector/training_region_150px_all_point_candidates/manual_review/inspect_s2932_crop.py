from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

root = Path(__file__).resolve().parents[1]
source = root / "patches_raw" / "costa_afl_S2932_1974.png"
out = Path(__file__).resolve().parent / "S2932_zoom_grid.png"
image = Image.open(source).convert("RGB")
left, top, right, bottom = 35, 35, 115, 115
crop = image.crop((left, top, right, bottom)).resize((800, 800), Image.Resampling.NEAREST)
draw = ImageDraw.Draw(crop)
font = ImageFont.load_default()
for value in range(0, 81, 10):
    x = value * 10
    y = value * 10
    draw.line((x, 0, x, 800), fill=(120, 120, 120), width=1)
    draw.line((0, y, 800, y), fill=(120, 120, 120), width=1)
    draw.text((x + 2, 2), str(left + value), fill=(255, 0, 0), font=font)
    draw.text((2, y + 2), str(top + value), fill=(255, 0, 0), font=font)
crop.save(out, format="PNG", optimize=True)
print(out)
