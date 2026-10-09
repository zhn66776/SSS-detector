#!/usr/bin/env python3
"""Convert streamed geophysical grids into transparent web tile pyramids."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


TILE_SIZE = 256


def log(message: str) -> None:
    print(message, flush=True)


def signature(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def sample_values(sums: np.memmap, counts: np.memmap) -> np.ndarray:
    sampled_sums = np.asarray(sums[::8, ::8])
    sampled_counts = np.asarray(counts[::8, ::8])
    valid = sampled_counts > 0
    if not np.any(valid):
        raise ValueError("Grid contains no valid cells")
    return sampled_sums[valid] / sampled_counts[valid]


def interpolate_stops(normalized: np.ndarray, stops: np.ndarray) -> np.ndarray:
    position = np.clip(normalized, 0.0, 1.0) * (len(stops) - 1)
    lower = np.floor(position).astype(np.int16)
    upper = np.minimum(lower + 1, len(stops) - 1)
    fraction = (position - lower)[..., None]
    return np.rint(stops[lower] * (1 - fraction) + stops[upper] * fraction).astype(np.uint8)


def colorize(values: np.ndarray, valid: np.ndarray, layer_id: str, limits: tuple[float, float]) -> np.ndarray:
    low, high = limits
    normalized = (values - low) / max(high - low, 1e-9)
    if layer_id == "mbes":
        stops = np.array(
            [[48, 18, 59], [50, 64, 159], [32, 144, 140], [94, 201, 98], [225, 220, 55], [244, 110, 36], [180, 4, 38]],
            dtype=np.float32,
        )
    else:
        stops = np.array(
            [[34, 74, 168], [92, 151, 208], [238, 239, 235], [226, 132, 105], [170, 35, 55]],
            dtype=np.float32,
        )
    rgb = interpolate_stops(normalized, stops)
    alpha = np.where(valid, 225, 0).astype(np.uint8)
    return np.dstack([rgb, alpha])


def save_png(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    image.save(temp, format="PNG", compress_level=6, optimize=False)
    os.replace(temp, path)


def build_base(
    sums: np.memmap,
    counts: np.memmap,
    layer_id: str,
    target: Path,
    max_level: int,
    limits: tuple[float, float],
) -> None:
    height, width = counts.shape
    level_dir = target / str(max_level)
    level_dir.mkdir(parents=True, exist_ok=True)
    y_tiles = math.ceil(height / TILE_SIZE)
    x_tiles = math.ceil(width / TILE_SIZE)
    for y_tile in range(y_tiles):
        y0 = y_tile * TILE_SIZE
        y1 = min(height, y0 + TILE_SIZE)
        count_rows = np.asarray(counts[y0:y1])
        sum_rows = np.asarray(sums[y0:y1])
        for x_tile in range(x_tiles):
            x0 = x_tile * TILE_SIZE
            x1 = min(width, x0 + TILE_SIZE)
            tile_counts = count_rows[:, x0:x1]
            valid = tile_counts > 0
            if not np.any(valid):
                continue
            tile_values = np.zeros(tile_counts.shape, dtype=np.float32)
            np.divide(sum_rows[:, x0:x1], tile_counts, out=tile_values, where=valid)
            rgba = colorize(tile_values, valid, layer_id, limits)
            save_png(Image.fromarray(rgba, mode="RGBA"), level_dir / f"{x_tile}_{y_tile}.png")
        if y_tile % 8 == 0 or y_tile + 1 == y_tiles:
            log(f"  base rows: {y_tile + 1}/{y_tiles}")


def build_lower(target: Path, width: int, height: int, max_level: int) -> None:
    previous_width, previous_height = width, height
    for level in range(max_level - 1, -1, -1):
        previous_dir = target / str(level + 1)
        level_dir = target / str(level)
        level_dir.mkdir(parents=True, exist_ok=True)
        level_width = math.ceil(previous_width / 2)
        level_height = math.ceil(previous_height / 2)
        x_tiles = math.ceil(level_width / TILE_SIZE)
        y_tiles = math.ceil(level_height / TILE_SIZE)
        for y_tile in range(y_tiles):
            for x_tile in range(x_tiles):
                children = [
                    previous_dir / f"{x_tile * 2 + dx}_{y_tile * 2 + dy}.png"
                    for dy in range(2)
                    for dx in range(2)
                ]
                if not any(path.exists() for path in children):
                    continue
                canvas = Image.new("RGBA", (TILE_SIZE * 2, TILE_SIZE * 2), (0, 0, 0, 0))
                for dy in range(2):
                    for dx in range(2):
                        child = previous_dir / f"{x_tile * 2 + dx}_{y_tile * 2 + dy}.png"
                        if child.exists():
                            with Image.open(child) as image:
                                canvas.alpha_composite(image.convert("RGBA"), (dx * TILE_SIZE, dy * TILE_SIZE))
                valid_width = min(TILE_SIZE * 2, previous_width - x_tile * TILE_SIZE * 2)
                valid_height = min(TILE_SIZE * 2, previous_height - y_tile * TILE_SIZE * 2)
                parent = canvas.crop((0, 0, valid_width, valid_height)).resize(
                    (math.ceil(valid_width / 2), math.ceil(valid_height / 2)), Image.Resampling.LANCZOS
                )
                if parent.getbbox() is not None:
                    save_png(parent, level_dir / f"{x_tile}_{y_tile}.png")
        previous_width, previous_height = level_width, level_height


def build_layer(grid_dir: Path, output: Path, layer_id: str) -> dict:
    meta_path = grid_dir / f"{layer_id}.meta.json"
    sum_path = grid_dir / f"{layer_id}.sum.f32"
    count_path = grid_dir / f"{layer_id}.count.u16"
    meta = json.loads(meta_path.read_text())
    width, height = int(meta["width"]), int(meta["height"])
    sums = np.memmap(sum_path, dtype=np.float32, mode="r", shape=(height, width))
    counts = np.memmap(count_path, dtype=np.uint16, mode="r", shape=(height, width))
    samples = sample_values(sums, counts)
    if layer_id == "mbes":
        limits = tuple(float(x) for x in np.percentile(samples, [2, 98]))
        label = "MBES bathymetry"
        units = "m"
        palette = "bathymetry"
    else:
        median = float(np.median(samples))
        spread = float(np.percentile(np.abs(samples - median), 98))
        limits = (median - spread, median + spread)
        label = "MAG total field"
        units = "nT"
        palette = "diverging"
    target = output / layer_id
    complete = target / "complete.json"
    expected = {
        "sum": signature(sum_path),
        "count": signature(count_path),
        "tile_size": TILE_SIZE,
        "limits": list(limits),
        "palette_version": "bathymetry-contrast-v2" if layer_id == "mbes" else "mag-diverging-v1",
    }
    if complete.exists():
        state = json.loads(complete.read_text())
        if state.get("source") == expected:
            log(f"reuse completed {layer_id} tiles")
            return state["layer"]
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    max_level = math.ceil(math.log2(max(width, height)))
    log(f"{layer_id}: values {limits[0]:.3f}..{limits[1]:.3f}, base level {max_level}")
    build_base(sums, counts, layer_id, target, max_level, limits)
    build_lower(target, width, height, max_level)
    layer = {
        "id": layer_id,
        "label": label,
        "units": units,
        "palette": palette,
        "width": width,
        "height": height,
        "maxLevel": max_level,
        "tileSize": TILE_SIZE,
        "left": float(meta["xmin"]),
        "right": float(meta["xmax"]),
        "bottom": float(meta["ymin"]),
        "top": float(meta["ymax"]),
        "pixelX": float(meta["resolution"]),
        "pixelY": -float(meta["resolution"]),
        "min": limits[0],
        "max": limits[1],
        "sampleStride": int(meta["stride"]),
        "acceptedSamples": int(meta["accepted_samples"]),
        "tileRoot": f"geophysics/{layer_id}",
        "extension": "png",
    }
    complete.write_text(json.dumps({"source": expected, "layer": layer}, indent=2) + "\n")
    return layer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    layers = [build_layer(args.grid_dir, args.output, layer_id) for layer_id in ("mbes", "mag")]
    (args.output / "layers.json").write_text(json.dumps(layers, indent=2) + "\n")
    log(f"complete: {args.output / 'layers.json'}")


if __name__ == "__main__":
    main()
