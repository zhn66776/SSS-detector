"""Bounded-memory native-pixel renderer for the Galapagos tiled float GeoTIFF."""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
from PIL import Image
import tifffile


SOURCE = Path('/Users/haonanzhang/Downloads/S0611_IguanasW_Horizontal_i3cm_all_SANDY.tiff')
ROOT = Path(__file__).resolve().parent
LOW, HIGH = -35.0, 35.0


def tile(page: tifffile.TiffPage, file, index: int) -> np.ndarray:
    file.seek(page.dataoffsets[index])
    data = file.read(page.databytecounts[index])
    decoded, _, _ = page.decode(data, index)
    return np.asarray(decoded).reshape(page.tilelength, page.tilewidth)


def to_gray(a: np.ndarray) -> np.ndarray:
    valid = np.isfinite(a) & (a < 1e30)
    result = np.zeros(a.shape, dtype=np.uint8)
    result[valid] = np.clip((a[valid] - LOW) * 255 / (HIGH - LOW), 0, 255).astype(np.uint8)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--overview', action='store_true')
    parser.add_argument('--tile', type=int)
    args = parser.parse_args()
    ROOT.mkdir(exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*GDAL_NODATA.*')
        with tifffile.TiffFile(SOURCE) as tif, SOURCE.open('rb') as f:
            page = tif.pages[0]
            cols = (page.imagewidth + page.tilewidth - 1) // page.tilewidth
            rows = (page.imagelength + page.tilelength - 1) // page.tilelength
            if args.overview:
                scale = 16
                canvas = Image.new('L', (cols * 128, rows * 128))
                for i in range(len(page.dataoffsets)):
                    if page.databytecounts[i] < 50000:
                        continue
                    a = tile(page, f, i)[::scale, ::scale]
                    canvas.paste(Image.fromarray(to_gray(a)), ((i % cols) * 128, (i // cols) * 128))
                canvas = canvas.crop((0, 0, (page.imagewidth + scale - 1) // scale, (page.imagelength + scale - 1) // scale))
                out = ROOT / 'overview_16x.png'
            elif args.tile is not None:
                i = args.tile
                if not 0 <= i < len(page.dataoffsets):
                    raise SystemExit('Invalid tile index')
                canvas = Image.fromarray(to_gray(tile(page, f, i)))
                canvas = canvas.crop((0, 0, min(page.tilewidth, page.imagewidth - (i % cols) * page.tilewidth), min(page.tilelength, page.imagelength - (i // cols) * page.tilelength)))
                out = ROOT / f'native_tile_{i}.png'
            else:
                parser.error('Choose --overview or --tile INDEX')
            canvas.save(out, optimize=True)
            print(out)


if __name__ == '__main__':
    main()
