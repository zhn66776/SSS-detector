#!/usr/bin/env python3
"""Build native-resolution MBES image tiles and full MAG point chunks."""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def log(message: str) -> None:
    print(message, flush=True)


def interpolate_stops(normalized: np.ndarray, stops: np.ndarray) -> np.ndarray:
    position = np.clip(normalized, 0.0, 1.0) * (len(stops) - 1)
    lower = np.floor(position).astype(np.int16)
    upper = np.minimum(lower + 1, len(stops) - 1)
    fraction = (position - lower)[..., None]
    return np.rint(stops[lower] * (1 - fraction) + stops[upper] * fraction).astype(np.uint8)


def build_mbes(native_root: Path, output_root: Path, limits: tuple[float, float]) -> dict:
    meta = json.loads((native_root / "mbes-native-meta.json").read_text())
    bins = native_root / "mbes_bins"
    width, height = int(meta["width"]), int(meta["height"])
    max_level = math.ceil(math.log2(max(width, height)))
    destination = output_root / "mbes" / str(max_level)
    destination.mkdir(parents=True, exist_ok=True)
    files = sorted(bins.glob("*/*.f32"), key=lambda p: (int(p.parent.name), int(p.stem)))
    stops = np.array(
        [[48, 18, 59], [50, 64, 159], [32, 144, 140], [94, 201, 98], [225, 220, 55], [244, 110, 36], [180, 4, 38]],
        dtype=np.float32,
    )
    low, high = limits
    palette_version = "bathymetry-contrast-v2"
    palette_marker = output_root / "mbes-palette-version.txt"
    force_recolor = not palette_marker.exists() or palette_marker.read_text().strip() != palette_version
    destination.mkdir(parents=True, exist_ok=True)
    completed = 0
    for index, path in enumerate(files, start=1):
        tx, ty = int(path.parent.name), int(path.stem)
        target = destination / f"{tx}_{ty}.png"
        if not force_recolor and target.exists() and target.stat().st_mtime_ns >= path.stat().st_mtime_ns:
            completed += 1
            continue
        values = np.fromfile(path, dtype=np.float32).reshape(256, 256)
        valid = np.isfinite(values)
        if not np.any(valid):
            continue
        normalized = (np.nan_to_num(values, nan=low) - low) / max(high - low, 1e-9)
        rgb = interpolate_stops(normalized, stops)
        alpha = np.where(valid, 255, 0).astype(np.uint8)
        rgba = np.dstack([rgb, alpha])
        temp = target.with_suffix(".tmp")
        Image.fromarray(rgba).save(temp, format="PNG", compress_level=3, optimize=False)
        os.replace(temp, target)
        completed += 1
        if index % 2000 == 0 or index == len(files):
            log(f"MBES native PNG: {index}/{len(files)}")
    palette_marker.write_text(palette_version + "\n")
    return {
        "id": "mbes_native",
        "label": "MBES native 0.2 m",
        "width": width,
        "height": height,
        "maxLevel": max_level,
        "tileSize": 256,
        "left": float(meta["xmin"]),
        "right": float(meta["xmax"]),
        "bottom": float(meta["ymin"]),
        "top": float(meta["ymax"]),
        "pixelX": float(meta["resolution"]),
        "pixelY": -float(meta["resolution"]),
        "tileRoot": "native/mbes",
        "extension": "png",
        "nativeOnly": True,
        "sourcePoints": int(meta["accepted_points"]),
        "outsidePoints": int(meta["outside_points"]),
        "tileCount": completed,
        "min": low,
        "max": high,
        "paletteVersion": palette_version,
    }


def build_mag(native_root: Path, output_root: Path) -> dict:
    meta = json.loads((native_root / "mag-raw-meta.json").read_text())
    bins = native_root / "mag_raw_bins"
    destination = output_root / "mag_raw"
    destination.mkdir(parents=True, exist_ok=True)
    files = sorted(bins.glob("*/*.bin"), key=lambda p: (int(p.parent.name), int(p.stem)))
    keys: list[str] = []
    for index, path in enumerate(files, start=1):
        tx, ty = int(path.parent.name), int(path.stem)
        key = f"{tx}_{ty}"
        keys.append(key)
        target_dir = destination / str(tx)
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{ty}.js"
        if target.exists() and target.stat().st_mtime_ns >= path.stat().st_mtime_ns:
            continue
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        temp = target.with_suffix(".tmp")
        temp.write_text(f'window.MAG_RAW_B64=window.MAG_RAW_B64||{{}};window.MAG_RAW_B64["{key}"]="{encoded}";\n')
        os.replace(temp, target)
        if index % 1000 == 0 or index == len(files):
            log(f"MAG raw scripts: {index}/{len(files)}")
    manifest = {
        "id": "mag_raw",
        "label": "MAG full raw points",
        "xmin": float(meta["xmin"]),
        "xmax": float(meta["xmax"]),
        "ymin": float(meta["ymin"]),
        "ymax": float(meta["ymax"]),
        "tileM": float(meta["tile_m"]),
        "widthTiles": int(meta["width_tiles"]),
        "heightTiles": int(meta["height_tiles"]),
        "pointCount": int(meta["accepted_points"]),
        "outsidePoints": int(meta["outside_points"]),
        "keys": keys,
        "scriptRoot": "native/mag_raw",
        "recordBytes": 12,
    }
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mbes-min", type=float, required=True)
    parser.add_argument("--mbes-max", type=float, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    mbes = build_mbes(args.native_root, args.output, (args.mbes_min, args.mbes_max))
    mag = build_mag(args.native_root, args.output)
    assets = {"mbes": mbes, "mag": mag}
    (args.output / "assets.json").write_text(json.dumps(assets, indent=2) + "\n")
    log(f"complete: {args.output / 'assets.json'}")


if __name__ == "__main__":
    main()
