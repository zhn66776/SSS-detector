#!/usr/bin/env python3
"""Compare native SSS pixels with the generated quality-84 JPEG tiles."""

from __future__ import annotations

import io
import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image
from tifffile import TiffFile


SOURCE = Path("/Volumes/HP P900/Dudgeon_2013_MDE/02_extracted/SSS/imagery")
TILES = Path("/Volumes/HP P900/Dudgeon_2013_MDE/03_sss_inspector/tiles")
TILE_SIZE = 256


def safe_id(path: Path) -> str:
    block = re.search(r"blk(\d+)", path.name, re.IGNORECASE)
    date = re.search(r"_(20\d{6})_blk", path.name, re.IGNORECASE)
    short = "plain" if "Seabed_and_Sub_Seabed" not in path.name else "mapping"
    return f"blk{block.group(1).zfill(2)}-{date.group(1) if date else 'unknown'}-{short}"


def encode_png(array: np.ndarray, mode: str) -> int:
    output = io.BytesIO()
    Image.fromarray(array).save(output, format="PNG", compress_level=6, optimize=False)
    return output.tell()


def compare(path: Path, sample_count: int = 3) -> dict:
    dataset_id = safe_id(path)
    state = json.loads((TILES / dataset_id / "complete.json").read_text())
    max_level = int(state["max_level"])
    base = TILES / dataset_id / str(max_level)
    choices = sorted(base.glob("*.jpg"), key=lambda item: item.stat().st_size, reverse=True)[:sample_count]
    selected: dict[int, list[tuple[int, Path]]] = {}
    for jpeg in choices:
        x, y = map(int, jpeg.stem.split("_"))
        selected.setdefault(y, []).append((x, jpeg))

    records = []
    with TiffFile(path) as tif:
        page = tif.pages[0]
        width, height = int(page.imagewidth), int(page.imagelength)
        samples = int(page.samplesperpixel)
        palette = page.colormap
        palette8 = (palette // 257).astype(np.uint8) if palette is not None else np.vstack([np.arange(256, dtype=np.uint8)] * 3)
        color_mode = "RGB" if not (
            np.array_equal(palette8[0], palette8[1]) and np.array_equal(palette8[1], palette8[2])
        ) else "L"
        buffers: dict[tuple[int, int], tuple[np.ndarray, np.ndarray | None]] = {}
        for y_tile, items in selected.items():
            rows = min(TILE_SIZE, height - y_tile * TILE_SIZE)
            for x_tile, _jpeg in items:
                cols = min(TILE_SIZE, width - x_tile * TILE_SIZE)
                buffers[(x_tile, y_tile)] = (
                    np.zeros((rows, cols), dtype=np.uint8),
                    np.zeros((rows, cols), dtype=np.uint8) if samples > 1 else None,
                )
        for segment, index, _shape in page.segments(sort=True, maxworkers=1):
            if segment is None:
                continue
            y0, x0 = int(index[2]), int(index[3])
            y_tile = y0 // TILE_SIZE
            if y_tile not in selected:
                continue
            rows = segment[0]
            if rows.ndim == 2:
                rows = rows[..., None]
            local_y = y0 - y_tile * TILE_SIZE
            for x_tile, _jpeg in selected[y_tile]:
                tile_x0 = x_tile * TILE_SIZE
                tile_x1 = min(width, tile_x0 + TILE_SIZE)
                overlap0 = max(tile_x0, x0)
                overlap1 = min(tile_x1, x0 + rows.shape[1])
                if overlap0 >= overlap1:
                    continue
                indices, alpha = buffers[(x_tile, y_tile)]
                src0, src1 = overlap0 - x0, overlap1 - x0
                dst0, dst1 = overlap0 - tile_x0, overlap1 - tile_x0
                row_count = min(rows.shape[0], indices.shape[0] - local_y)
                indices[local_y : local_y + row_count, dst0:dst1] = rows[:row_count, src0:src1, 0]
                if alpha is not None:
                    alpha[local_y : local_y + row_count, dst0:dst1] = rows[:row_count, src0:src1, 1]

        for y_tile, items in selected.items():
            for x_tile, jpeg_path in items:
                indices, alpha = buffers[(x_tile, y_tile)]
                if color_mode == "L":
                    exact = palette8[0][indices]
                    if alpha is not None:
                        a16 = alpha.astype(np.uint16)
                        exact = ((exact.astype(np.uint16) * a16 + 255 * (255 - a16) + 127) // 255).astype(np.uint8)
                else:
                    exact = np.moveaxis(palette8[:, indices], 0, -1)
                    if alpha is not None:
                        a16 = alpha.astype(np.uint16)[..., None]
                        exact = ((exact.astype(np.uint16) * a16 + 255 * (255 - a16) + 127) // 255).astype(np.uint8)
                with Image.open(jpeg_path) as image:
                    decoded = np.asarray(image.convert(color_mode))
                difference = np.abs(exact.astype(np.int16) - decoded.astype(np.int16))
                mse = float(np.mean((exact.astype(np.float32) - decoded.astype(np.float32)) ** 2))
                records.append(
                    {
                        "tile": jpeg_path.name,
                        "pixels": int(difference.size),
                        "mae": float(np.mean(difference)),
                        "rmse": math.sqrt(mse),
                        "psnr": float("inf") if mse == 0 else 20 * math.log10(255 / math.sqrt(mse)),
                        "gt1": float(np.mean(difference > 1)),
                        "gt3": float(np.mean(difference > 3)),
                        "gt5": float(np.mean(difference > 5)),
                        "max": int(np.max(difference)),
                        "jpeg_bytes": jpeg_path.stat().st_size,
                        "png_bytes": encode_png(exact, color_mode),
                    }
                )
    total_pixels = sum(item["pixels"] for item in records)
    weighted = lambda key: sum(item[key] * item["pixels"] for item in records) / total_pixels
    return {
        "dataset": dataset_id,
        "samples": len(records),
        "mae": weighted("mae"),
        "rmse": weighted("rmse"),
        "psnr": weighted("psnr"),
        "gt1": weighted("gt1"),
        "gt3": weighted("gt3"),
        "gt5": weighted("gt5"),
        "max": max(item["max"] for item in records),
        "jpeg_bytes": sum(item["jpeg_bytes"] for item in records),
        "png_bytes": sum(item["png_bytes"] for item in records),
    }


def main() -> None:
    results = []
    for path in sorted(SOURCE.rglob("*.tif")):
        result = compare(path)
        results.append(result)
        print(json.dumps(result), flush=True)
    total_samples = sum(item["samples"] for item in results)
    summary = {
        "datasets": len(results),
        "sample_tiles": total_samples,
        "mean_mae": sum(item["mae"] * item["samples"] for item in results) / total_samples,
        "mean_rmse": sum(item["rmse"] * item["samples"] for item in results) / total_samples,
        "mean_psnr_db": sum(item["psnr"] * item["samples"] for item in results) / total_samples,
        "mean_gt1": sum(item["gt1"] * item["samples"] for item in results) / total_samples,
        "mean_gt3": sum(item["gt3"] * item["samples"] for item in results) / total_samples,
        "mean_gt5": sum(item["gt5"] * item["samples"] for item in results) / total_samples,
        "max_abs_difference": max(item["max"] for item in results),
        "jpeg_bytes": sum(item["jpeg_bytes"] for item in results),
        "png_bytes": sum(item["png_bytes"] for item in results),
    }
    summary["png_to_jpeg_size_ratio"] = summary["png_bytes"] / summary["jpeg_bytes"]
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
