#!/usr/bin/env python3
"""Build lossless native-resolution SSS PNG tiles from source GeoTIFFs."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from tifffile import TiffFile


TILE_SIZE = 256
VERSION = "sss-lossless-native-v1"


def log(message: str) -> None:
    print(message, flush=True)


def safe_id(path: Path) -> str:
    block = re.search(r"blk(\d+)", path.name, re.IGNORECASE)
    date = re.search(r"_(20\d{6})_blk", path.name, re.IGNORECASE)
    short = "plain" if "Seabed_and_Sub_Seabed" not in path.name else "mapping"
    return f"blk{block.group(1).zfill(2)}-{date.group(1) if date else 'unknown'}-{short}"


def signature(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def save_png(array: np.ndarray, mode: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    Image.fromarray(array, mode=mode).save(temp, format="PNG", compress_level=6, optimize=False)
    os.replace(temp, path)


def build_one(tif_path: Path, output: Path) -> dict:
    dataset_id = safe_id(tif_path)
    target = output / dataset_id
    complete = target / "complete.json"
    expected = {"source": signature(tif_path), "version": VERSION, "tile_size": TILE_SIZE}
    if complete.exists():
        state = json.loads(complete.read_text())
        if state.get("expected") == expected:
            log(f"reuse {dataset_id}: {state['tile_count']} lossless tiles")
            return state
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    tile_count = 0
    with TiffFile(tif_path) as tif:
        page = tif.pages[0]
        width, height = int(page.imagewidth), int(page.imagelength)
        samples = int(page.samplesperpixel)
        max_level = math.ceil(math.log2(max(width, height)))
        level_dir = target / str(max_level)
        level_dir.mkdir(parents=True)
        palette = page.colormap
        palette8 = (palette // 257).astype(np.uint8) if palette is not None else np.vstack([np.arange(256, dtype=np.uint8)] * 3)
        color = not (np.array_equal(palette8[0], palette8[1]) and np.array_equal(palette8[1], palette8[2]))
        has_alpha = samples > 1
        current_group = -1
        index_rows: np.ndarray | None = None
        alpha_rows: np.ndarray | None = None
        valid_rows = 0

        def flush(group: int, rows: int) -> None:
            nonlocal tile_count
            if group < 0 or index_rows is None or rows <= 0:
                return
            for x_tile, x0 in enumerate(range(0, width, TILE_SIZE)):
                x1 = min(width, x0 + TILE_SIZE)
                indices = index_rows[:rows, x0:x1]
                alpha = alpha_rows[:rows, x0:x1] if alpha_rows is not None else None
                if alpha is not None and not np.any(alpha):
                    continue
                if color:
                    rgb = np.moveaxis(palette8[:, indices], 0, -1)
                    if alpha is not None:
                        pixels = np.dstack([rgb, alpha])
                        mode = "RGBA"
                    else:
                        pixels = rgb
                        mode = "RGB"
                else:
                    gray = palette8[0][indices]
                    if alpha is not None:
                        pixels = np.dstack([gray, alpha])
                        mode = "LA"
                    else:
                        pixels = gray
                        mode = "L"
                save_png(pixels, mode, level_dir / f"{x_tile}_{group}.png")
                tile_count += 1

        total_groups = math.ceil(height / TILE_SIZE)
        for segment, index, _shape in page.segments(sort=True, maxworkers=1):
            if segment is None:
                continue
            y0, x0 = int(index[2]), int(index[3])
            rows = segment[0]
            if rows.ndim == 2:
                rows = rows[..., None]
            group = y0 // TILE_SIZE
            if group != current_group:
                flush(current_group, valid_rows)
                if current_group >= 0 and (current_group + 1) % 8 == 0:
                    log(f"  {dataset_id}: rows {current_group + 1}/{total_groups}")
                current_group = group
                group_height = min(TILE_SIZE, height - group * TILE_SIZE)
                index_rows = np.zeros((group_height, width), dtype=np.uint8)
                alpha_rows = np.zeros((group_height, width), dtype=np.uint8) if has_alpha else None
                valid_rows = group_height
            local_y = y0 - group * TILE_SIZE
            row_count = min(rows.shape[0], valid_rows - local_y)
            col_count = min(rows.shape[1], width - x0)
            index_rows[local_y : local_y + row_count, x0 : x0 + col_count] = rows[:row_count, :col_count, 0]
            if alpha_rows is not None:
                alpha_rows[local_y : local_y + row_count, x0 : x0 + col_count] = rows[:row_count, :col_count, 1]
        flush(current_group, valid_rows)
    state = {
        "expected": expected,
        "dataset": dataset_id,
        "width": width,
        "height": height,
        "max_level": max_level,
        "tile_count": tile_count,
        "tile_root": f"lossless/{dataset_id}",
        "extension": "png",
    }
    complete.write_text(json.dumps(state, indent=2) + "\n")
    log(f"complete {dataset_id}: {tile_count} lossless tiles")
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    states = [build_one(path, args.output) for path in sorted(args.source.rglob("*.tif"))]
    (args.output / "manifest.json").write_text(json.dumps(states, indent=2) + "\n")
    log(f"complete: {args.output / 'manifest.json'}")


if __name__ == "__main__":
    main()
