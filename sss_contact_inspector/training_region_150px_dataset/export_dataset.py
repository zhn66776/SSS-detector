#!/usr/bin/env python3
"""Export the complete, quality-filtered Costa Head 150 px contact dataset.

The exporter is deliberately self-contained and writes every generated file
under its own parent directory.  It uses the project-local ``ecw_window``
decoder for bounded 150 x 150 source windows only; no overview or full-raster
decode is performed here.
"""

from __future__ import annotations

import csv
import atexit
import hashlib
import io
import json
import math
import re
import shutil
import subprocess
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageStat


OUT = Path(__file__).resolve().parent
ROOT = OUT.parent

POLYGON_PATH = ROOT / "training_region_preview" / "approximate_training_region.geojson"
CONTACTS_PATH = ROOT / "data" / "contacts.json"
METADATA_PATH = ROOT / "data" / "metadata.json"
ECW_PATH = ROOT / "data" / "derived_sources" / "costa" / "C13007a_Costa_AllAreas_Comments_Rev1.ecw"
ERS_PATH = ROOT / "data" / "derived_sources" / "costa" / "C13007a_Costa_AllAreas_Comments_Rev1.ers"
DECODER_PATH = ROOT / "local" / "bin" / "ecw_window"

RAW_DIR = OUT / "patches_raw"
PREVIEW_DIR = OUT / "patches_preview_marked"
POLYGON_COPY_PATH = OUT / "polygon_used.geojson"
ACCEPTED_CSV_PATH = OUT / "accepted_contacts.csv"
SKIPPED_CSV_PATH = OUT / "skipped_contacts.csv"
MANIFEST_PATH = OUT / "dataset_manifest.json"
QUALITY_SUMMARY_PATH = OUT / "quality_summary.md"
CONTACT_SHEET_PATH = OUT / "preview_contact_sheet.png"

PATCH_SIZE = 150
PATCH_HALF = PATCH_SIZE // 2
# An even-sided window has its geometric centre between four pixels.  Mapping a
# sub-pixel GIS coordinate to the nearest integer source cell therefore has a
# maximum Euclidean offset of sqrt(0.5^2 + 0.5^2), not 0.5 px.
MAX_CENTER_DISTANCE_PX = math.hypot(0.5, 0.5)
PATCH_METRES = 15.0
NATIVE_RESOLUTION_M = 0.1
GREY_VARIANCE_MIN = 450.0
GREY_MEAN_MAX_EXCLUSIVE = 250.0
DECODER_BAND = 0
MAX_WORKERS = 1
MAX_IN_FLIGHT = 8
DECODE_CACHE_SIZE = 64
MARKER_RADIUS_PX = 7
MARKER_WIDTH_PX = 2


COMMON_COLUMNS = [
    "contact_list_index",
    "uid",
    "id",
    "dataset",
    "geometry",
    "area",
    "description",
    "e",
    "n",
    "length_m",
    "width_m",
    "height_m",
    "original_dimensions_m",
    "source_file",
    "source_row",
    "raster_id",
]

ACCEPTED_COLUMNS = COMMON_COLUMNS + [
    "raw_patch_file",
    "preview_file",
    "source_window_x0",
    "source_window_y0",
    "source_window_x1",
    "source_window_y1",
    "patch_contact_x",
    "patch_contact_y",
    "patch_contact_distance_from_75_75_px",
    "grey_mean",
    "grey_variance",
]

SKIPPED_COLUMNS = COMMON_COLUMNS + [
    "candidate_scope",
    "skip_reason",
    "skip_detail",
    "source_window_x0",
    "source_window_y0",
    "source_window_x1",
    "source_window_y1",
    "patch_contact_x",
    "patch_contact_y",
    "patch_contact_distance_from_75_75_px",
    "grey_mean",
    "grey_variance",
]


class ExportError(RuntimeError):
    """Raised when an input or generated artifact fails a hard contract."""


class PersistentECWWorker:
    """One serialized decoder process with the SDK's bounded internal cache."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.counter = 0
        self.process: subprocess.Popen[bytes] | None = None
        self.info: dict[str, Any] = {}
        self.start()
        atexit.register(self.close)

    def start(self) -> None:
        self.close()
        self.process = subprocess.Popen(
            [str(DECODER_PATH), "serve", str(ECW_PATH)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        line = self.process.stdout.readline() if self.process.stdout else b""
        if not line:
            error = self.process.stderr.read().decode("utf-8", errors="replace") if self.process.stderr else ""
            raise ExportError(f"persistent ECW worker failed to start: {error[-1000:]}")
        try:
            self.info = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ExportError(f"persistent ECW worker returned invalid startup JSON: {line[:500]!r}") from exc

    @staticmethod
    def read_exact(stream: Any, size: int) -> bytes:
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            chunk = stream.read(remaining)
            if not chunk:
                raise ExportError("persistent ECW worker ended during image response")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def decode(self, x0: int, y0: int, x1: int, y1: int, width: int, height: int, band: int) -> bytes:
        with self.lock:
            if not self.process or self.process.poll() is not None:
                self.start()
            assert self.process and self.process.stdin and self.process.stdout
            self.counter += 1
            request_id = str(self.counter)
            command = f"{request_id} {x0} {y0} {x1} {y1} {width} {height} {band}\n".encode("ascii")
            self.process.stdin.write(command)
            self.process.stdin.flush()
            header = self.process.stdout.readline().decode("utf-8", errors="replace").strip()
            parts = header.split(" ", 2)
            if len(parts) < 3 or parts[1] != request_id:
                raise ExportError(f"invalid persistent ECW worker response: {header}")
            if parts[0] != "OK":
                raise ExportError(parts[2])
            try:
                payload_size = int(parts[2])
            except ValueError as exc:
                raise ExportError(f"invalid persistent ECW payload size: {header}") from exc
            return self.read_exact(self.process.stdout, payload_size)

    def close(self) -> None:
        process = self.process
        self.process = None
        if not process:
            return
        try:
            if process.stdin:
                process.stdin.close()
            process.wait(timeout=6)
        except Exception:
            try:
                process.kill()
                process.wait(timeout=2)
            except Exception:
                pass


_ECW_WORKER: PersistentECWWorker | None = None


def ecw_worker() -> PersistentECWWorker:
    global _ECW_WORKER
    if _ECW_WORKER is None:
        _ECW_WORKER = PersistentECWWorker()
    return _ECW_WORKER


def finite_number(value: Any) -> float | None:
    """Return a finite float for a JSON number or a numeric text value."""

    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def snapshot(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def point_on_segment(x: float, y: float, x1: float, y1: float, x2: float, y2: float) -> bool:
    cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    scale = max(1.0, abs(x), abs(y), abs(x1), abs(y1), abs(x2), abs(y2))
    if abs(cross) > 1e-9 * scale:
        return False
    return (
        min(x1, x2) - 1e-9 <= x <= max(x1, x2) + 1e-9
        and min(y1, y2) - 1e-9 <= y <= max(y1, y2) + 1e-9
    )


def point_in_ring_strict(x: float, y: float, ring: list[list[float]]) -> bool:
    inside = False
    previous = ring[-1]
    for current in ring:
        x1, y1 = float(previous[0]), float(previous[1])
        x2, y2 = float(current[0]), float(current[1])
        if point_on_segment(x, y, x1, y1, x2, y2):
            return False
        if (y1 > y) != (y2 > y):
            crossing_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < crossing_x:
                inside = not inside
        previous = current
    return inside


def point_in_polygon_strict(x: float, y: float, rings: list[list[list[float]]]) -> bool:
    """Strictly contain a point in a GeoJSON polygon, excluding all boundaries."""

    if not rings or not point_in_ring_strict(x, y, rings[0]):
        return False
    for hole in rings[1:]:
        if point_on_any_ring_boundary(x, y, hole) or point_in_ring_strict(x, y, hole):
            return False
    return True


def point_on_any_ring_boundary(x: float, y: float, ring: list[list[float]]) -> bool:
    previous = ring[-1]
    for current in ring:
        if point_on_segment(x, y, float(previous[0]), float(previous[1]), float(current[0]), float(current[1])):
            return True
        previous = current
    return False


def load_polygon() -> tuple[dict[str, Any], list[list[list[float]]]]:
    document = json.loads(POLYGON_PATH.read_text(encoding="utf-8"))
    features = document.get("features")
    if not isinstance(features, list) or len(features) != 1:
        raise ExportError("polygon input must contain exactly one feature")
    feature = features[0]
    properties = feature.get("properties") or {}
    if clean_text(properties.get("crs")).upper() != "EPSG:32630":
        raise ExportError("polygon CRS is not EPSG:32630")
    geometry = feature.get("geometry") or {}
    if geometry.get("type") != "Polygon":
        raise ExportError("polygon input is not a Polygon")
    coordinates = geometry.get("coordinates")
    if not isinstance(coordinates, list) or not coordinates or not all(isinstance(ring, list) for ring in coordinates):
        raise ExportError("polygon coordinates are missing or malformed")
    if coordinates[0][0] != coordinates[0][-1]:
        raise ExportError("polygon exterior ring is not closed; refusing to change it")
    return document, coordinates


def load_raster_metadata() -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    raster = metadata["datasets"]["costa"]["rasters"][0]
    expected = {
        "width": int(raster["width"]),
        "height": int(raster["height"]),
        "left": float(raster["left"]),
        "top": float(raster["top"]),
        "right": float(raster["right"]),
        "bottom": float(raster["bottom"]),
        "resolution_x": float(raster["resolution_x"]),
        "resolution_y": float(raster["resolution_y"]),
        "epsg": int(raster["epsg"]),
        "native_resolution_m": float(metadata["datasets"]["costa"]["native_resolution_m"]),
    }
    if expected["epsg"] != 32630 or expected["resolution_x"] != NATIVE_RESOLUTION_M or expected["resolution_y"] != NATIVE_RESOLUTION_M:
        raise ExportError("metadata raster resolution or CRS does not match the fixed input definition")
    decoder_run = subprocess.run(
        [str(DECODER_PATH), "info", str(ECW_PATH)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=60,
    )
    if decoder_run.returncode != 0:
        raise ExportError(f"ECW info failed: {decoder_run.stderr.strip()[:500]}")
    try:
        decoder_info = json.loads(decoder_run.stdout)
    except json.JSONDecodeError as exc:
        raise ExportError(f"ECW info was not JSON: {decoder_run.stdout[:500]}") from exc
    comparisons = {
        "width": int(decoder_info["width"]) == expected["width"],
        "height": int(decoder_info["height"]) == expected["height"],
        "cell_increment_x": math.isclose(float(decoder_info["cell_increment_x"]), NATIVE_RESOLUTION_M, abs_tol=1e-12),
        "cell_increment_y": math.isclose(float(decoder_info["cell_increment_y"]), -NATIVE_RESOLUTION_M, abs_tol=1e-12),
        "origin_x": math.isclose(float(decoder_info["origin_x"]), expected["left"], abs_tol=1e-6),
        "origin_y": math.isclose(float(decoder_info["origin_y"]), expected["top"], abs_tol=1e-6),
        "bands": int(decoder_info["bands"]) >= 1,
    }
    if not all(comparisons.values()):
        raise ExportError(f"ECW info does not agree with metadata: {comparisons}")
    expected["decoder_info"] = decoder_info
    return metadata, expected


def original_dimensions(row: dict[str, Any]) -> str:
    raw = row.get("raw")
    if isinstance(raw, dict):
        return clean_text(raw.get("Dimensions (m) (LxWxH)"))
    return ""


def common_row(row: dict[str, Any], contact_list_index: int) -> dict[str, Any]:
    return {
        "contact_list_index": contact_list_index,
        "uid": row.get("uid", ""),
        "id": row.get("id", ""),
        "dataset": row.get("dataset", ""),
        "geometry": row.get("geometry", ""),
        "area": row.get("area", ""),
        "description": row.get("description", ""),
        "e": row.get("e", ""),
        "n": row.get("n", ""),
        "length_m": row.get("length_m", ""),
        "width_m": row.get("width_m", ""),
        "height_m": row.get("height_m", ""),
        "original_dimensions_m": original_dimensions(row),
        "source_file": row.get("source_file", ""),
        "source_row": row.get("source_row", ""),
        "raster_id": row.get("raster_id", ""),
    }


def safe_stem(uid: str, used: dict[str, str]) -> str:
    parts = [part for part in re.split(r"[^A-Za-z0-9]+", uid) if part]
    stem = "_".join(parts)
    if not stem:
        stem = "contact"
    if stem in used and used[stem] != uid:
        stem = f"{stem}_{hashlib.sha1(uid.encode('utf-8')).hexdigest()[:8]}"
    used[stem] = uid
    return stem


def window_for(e: float, n: float, raster: dict[str, Any]) -> dict[str, Any]:
    pixel_x = (e - raster["left"]) / raster["resolution_x"]
    pixel_y = (raster["top"] - n) / raster["resolution_y"]
    center_x = round(pixel_x)
    center_y = round(pixel_y)
    x0 = center_x - PATCH_HALF
    y0 = center_y - PATCH_HALF
    x1 = x0 + PATCH_SIZE - 1
    y1 = y0 + PATCH_SIZE - 1
    patch_x = pixel_x - x0
    patch_y = pixel_y - y0
    distance = math.hypot(patch_x - PATCH_HALF, patch_y - PATCH_HALF)
    return {
        "pixel_x": pixel_x,
        "pixel_y": pixel_y,
        "center_x": center_x,
        "center_y": center_y,
        "x0": x0,
        "y0": y0,
        "x1": x1,
        "y1": y1,
        "patch_x": patch_x,
        "patch_y": patch_y,
        "distance": distance,
    }


def window_in_bounds(window: dict[str, Any], raster: dict[str, Any]) -> bool:
    return (
        window["x0"] >= 0
        and window["y0"] >= 0
        and window["x1"] < raster["width"]
        and window["y1"] < raster["height"]
    )


@lru_cache(maxsize=DECODE_CACHE_SIZE)
def decode_window_pixels(x0: int, y0: int, x1: int, y1: int) -> bytes:
    """Decode one bounded grayscale PGM window and return only its pixels."""

    payload = ecw_worker().decode(x0, y0, x1, y1, PATCH_SIZE, PATCH_SIZE, DECODER_BAND)
    try:
        image = Image.open(io.BytesIO(payload)).convert("L")
        if image.size != (PATCH_SIZE, PATCH_SIZE):
            raise ExportError(f"decoder returned {image.size}, expected {(PATCH_SIZE, PATCH_SIZE)}")
        pixels = image.tobytes()
    except Exception as exc:
        if isinstance(exc, ExportError):
            raise
        raise ExportError(f"invalid PGM decoder output: {exc}") from exc
    if len(pixels) != PATCH_SIZE * PATCH_SIZE:
        raise ExportError("decoder returned an unexpected pixel count")
    return pixels


def quality_reason(mean: float, variance: float) -> str | None:
    variance_fail = variance < GREY_VARIANCE_MIN
    mean_fail = mean >= GREY_MEAN_MAX_EXCLUSIVE
    if not variance_fail and not mean_fail:
        return None
    if variance_fail and mean_fail:
        return "grey_variance_below_450_and_mean_at_or_above_250"
    if variance_fail:
        return "grey_variance_below_450"
    return "grey_mean_at_or_above_250"


def decode_candidate(item: dict[str, Any], raster: dict[str, Any]) -> dict[str, Any]:
    row = item["row"]
    window = window_for(item["e"], item["n"], raster)
    result: dict[str, Any] = {
        "ordinal": item["ordinal"],
        "contact_list_index": item["contact_list_index"],
        "row": row,
        "window": window,
        "pixels": None,
        "grey_mean": None,
        "grey_variance": None,
        "skip_reason": None,
        "skip_detail": "",
    }
    if not window_in_bounds(window, raster):
        result["skip_reason"] = "window_out_of_raster_bounds"
        result["skip_detail"] = "150x150 source window is not wholly within the ECW pixel bounds"
        return result
    try:
        pixels = decode_window_pixels(window["x0"], window["y0"], window["x1"], window["y1"])
    except Exception as exc:
        result["skip_reason"] = "ecw_decode_failed"
        result["skip_detail"] = str(exc)[:500]
        return result
    image = Image.frombytes("L", (PATCH_SIZE, PATCH_SIZE), pixels)
    stats = ImageStat.Stat(image)
    mean = float(stats.mean[0])
    variance = float(stats.var[0])
    result["grey_mean"] = mean
    result["grey_variance"] = variance
    result["pixels"] = pixels
    result["skip_reason"] = quality_reason(mean, variance)
    if result["skip_reason"]:
        result["skip_detail"] = (
            f"fixed thresholds: grey_variance >= {GREY_VARIANCE_MIN:g} and "
            f"grey_mean < {GREY_MEAN_MAX_EXCLUSIVE:g}"
        )
        result["pixels"] = None
    return result


def metric_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    values = sorted(values)

    def quantile(fraction: float) -> float:
        position = (len(values) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return values[lower]
        weight = position - lower
        return values[lower] * (1.0 - weight) + values[upper] * weight

    return {
        "count": len(values),
        "min": values[0],
        "p05": quantile(0.05),
        "p25": quantile(0.25),
        "median": quantile(0.50),
        "p75": quantile(0.75),
        "p95": quantile(0.95),
        "max": values[-1],
    }


def fmt_metric(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.6f}"


def fmt_csv_row(value: Any) -> Any:
    if value is None:
        return ""
    return value


def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: fmt_csv_row(row.get(column, "")) for column in columns})


def skipped_row(
    item: dict[str, Any],
    reason: str,
    detail: str,
    window: dict[str, Any] | None = None,
    grey_mean: float | None = None,
    grey_variance: float | None = None,
) -> dict[str, Any]:
    result = common_row(item["row"], item["contact_list_index"])
    result.update(
        {
            "candidate_scope": "strictly_inside_polygon_point_contact",
            "skip_reason": reason,
            "skip_detail": detail,
            "source_window_x0": "",
            "source_window_y0": "",
            "source_window_x1": "",
            "source_window_y1": "",
            "patch_contact_x": "",
            "patch_contact_y": "",
            "patch_contact_distance_from_75_75_px": "",
            "grey_mean": fmt_metric(grey_mean),
            "grey_variance": fmt_metric(grey_variance),
        }
    )
    if window is not None:
        result.update(
            {
                "source_window_x0": window["x0"],
                "source_window_y0": window["y0"],
                "source_window_x1": window["x1"],
                "source_window_y1": window["y1"],
                "patch_contact_x": fmt_metric(window["patch_x"]),
                "patch_contact_y": fmt_metric(window["patch_y"]),
                "patch_contact_distance_from_75_75_px": fmt_metric(window["distance"]),
            }
        )
    return result


def accepted_row(
    item: dict[str, Any],
    window: dict[str, Any],
    grey_mean: float,
    grey_variance: float,
    raw_file: str,
    preview_file: str,
) -> dict[str, Any]:
    result = common_row(item["row"], item["contact_list_index"])
    result.update(
        {
            "raw_patch_file": raw_file,
            "preview_file": preview_file,
            "source_window_x0": window["x0"],
            "source_window_y0": window["y0"],
            "source_window_x1": window["x1"],
            "source_window_y1": window["y1"],
            "patch_contact_x": fmt_metric(window["patch_x"]),
            "patch_contact_y": fmt_metric(window["patch_y"]),
            "patch_contact_distance_from_75_75_px": fmt_metric(window["distance"]),
            "grey_mean": fmt_metric(grey_mean),
            "grey_variance": fmt_metric(grey_variance),
        }
    )
    return result


def make_preview(pixels: bytes, window: dict[str, Any]) -> tuple[Image.Image, Image.Image]:
    raw = Image.frombytes("L", (PATCH_SIZE, PATCH_SIZE), pixels)
    preview = raw.convert("RGB")
    draw = ImageDraw.Draw(preview)
    # Pillow's float-coordinate ellipse rasterisation can shift the visible
    # outline by one pixel when a source coordinate is numerically just below
    # an integer.  The contact itself is already guaranteed to be within 0.5
    # px of (75,75), so draw the visual marker at its nearest pixel centre.
    marker_x = round(window["patch_x"])
    marker_y = round(window["patch_y"])
    draw.ellipse(
        (
            marker_x - MARKER_RADIUS_PX,
            marker_y - MARKER_RADIUS_PX,
            marker_x + MARKER_RADIUS_PX,
            marker_y + MARKER_RADIUS_PX,
        ),
        outline=(255, 0, 0),
        width=MARKER_WIDTH_PX,
    )
    return raw, preview


def build_contact_sheet(accepted: list[dict[str, Any]]) -> dict[str, Any]:
    columns = 24
    cell_size = 90
    thumbnail_size = 80
    title_height = 50
    rows = max(1, math.ceil(len(accepted) / columns))
    sheet = Image.new("RGB", (columns * cell_size, title_height + rows * cell_size), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (8, 8),
        f"Costa Head accepted contact previews: {len(accepted)} x 150 px; red ring = listing coordinate",
        fill=(0, 0, 0),
        font=ImageFont.load_default(),
    )
    for index, row in enumerate(accepted, start=1):
        preview_path = OUT / row["preview_file"]
        with Image.open(preview_path) as image:
            thumb = image.convert("RGB").resize((thumbnail_size, thumbnail_size), Image.Resampling.NEAREST)
        column = (index - 1) % columns
        row_number = (index - 1) // columns
        x = column * cell_size + 5
        y = title_height + row_number * cell_size + 5
        sheet.paste(thumb, (x, y))
        draw.rectangle((x, y, x + thumbnail_size - 1, y + thumbnail_size - 1), outline=(80, 80, 80), width=1)
        draw.text((x + 2, y + 2), str(index), fill=(0, 0, 0), font=ImageFont.load_default())
    sheet.save(CONTACT_SHEET_PATH, format="PNG", optimize=True)
    return {
        "path": "preview_contact_sheet.png",
        "columns": columns,
        "cell_size_px": cell_size,
        "thumbnail_size_px": thumbnail_size,
        "accepted_order": "accepted_contacts.csv row order; one-based index in each cell",
    }


def validate_outputs(
    accepted_rows: list[dict[str, Any]],
    skipped_rows: list[dict[str, Any]],
    inside_items: list[dict[str, Any]],
    polygon_rings: list[list[list[float]]],
    raster: dict[str, Any],
) -> dict[str, Any]:
    if not ACCEPTED_CSV_PATH.is_file() or not SKIPPED_CSV_PATH.is_file():
        raise ExportError("CSV output is missing")
    if len(accepted_rows) + len(skipped_rows) != len(inside_items):
        raise ExportError("accepted + skipped does not equal the polygon-inside candidate count")
    candidate_keys = {(item["row"].get("source_file", ""), item["row"].get("source_row", "")) for item in inside_items}
    output_keys = {(row.get("source_file", ""), row.get("source_row", "")) for row in accepted_rows + skipped_rows}
    if output_keys != candidate_keys:
        raise ExportError("accepted/skipped source identifiers do not cover exactly the inside candidates")

    raw_checked = 0
    preview_checked = 0
    preview_red_checked = 0
    for row in accepted_rows:
        e = finite_number(row.get("e"))
        n = finite_number(row.get("n"))
        if e is None or n is None or not point_in_polygon_strict(e, n, polygon_rings):
            raise ExportError(f"accepted coordinate is not strictly inside polygon: {row.get('uid')}")
        window_values = [int(row[f"source_window_{axis}"]) for axis in ("x0", "y0", "x1", "y1")]
        x0, y0, x1, y1 = window_values
        if x0 < 0 or y0 < 0 or x1 >= raster["width"] or y1 >= raster["height"] or (x1 - x0 + 1, y1 - y0 + 1) != (PATCH_SIZE, PATCH_SIZE):
            raise ExportError(f"accepted source window is outside the raster: {row.get('uid')}")
        distance = finite_number(row.get("patch_contact_distance_from_75_75_px"))
        if distance is None or distance > MAX_CENTER_DISTANCE_PX + 1e-9:
            raise ExportError(f"contact center is too far from (75,75): {row.get('uid')} / {distance}")
        raw_path = OUT / row["raw_patch_file"]
        preview_path = OUT / row["preview_file"]
        with Image.open(raw_path) as raw_image:
            if raw_image.size != (PATCH_SIZE, PATCH_SIZE) or raw_image.mode != "L":
                raise ExportError(f"raw patch is not 150x150 single-channel: {raw_path}")
            raw_checked += 1
        with Image.open(preview_path) as preview_image:
            if preview_image.size != (PATCH_SIZE, PATCH_SIZE) or preview_image.mode != "RGB":
                raise ExportError(f"preview is not 150x150 RGB: {preview_path}")
            red_positions = []
            pixels = preview_image.load()
            for py in range(PATCH_SIZE):
                for px in range(PATCH_SIZE):
                    red, green, blue = pixels[px, py]
                    if red >= 220 and green <= 60 and blue <= 60:
                        red_positions.append((px, py))
            if not red_positions:
                raise ExportError(f"preview has no red coordinate ring: {preview_path}")
            contact_x = finite_number(row.get("patch_contact_x"))
            contact_y = finite_number(row.get("patch_contact_y"))
            if contact_x is None or contact_y is None:
                raise ExportError(f"preview coordinate is missing: {row.get('uid')}")
            red_xs = [position[0] for position in red_positions]
            red_ys = [position[1] for position in red_positions]
            if not (min(red_xs) - 1 <= contact_x <= max(red_xs) + 1 and min(red_ys) - 1 <= contact_y <= max(red_ys) + 1):
                raise ExportError(f"preview ring is not centered on the contact coordinate: {row.get('uid')}")
            preview_checked += 1
            preview_red_checked += len(red_positions)

    if not CONTACT_SHEET_PATH.is_file():
        raise ExportError("preview contact sheet is missing")
    with Image.open(CONTACT_SHEET_PATH) as sheet:
        if sheet.mode != "RGB" or sheet.width <= 0 or sheet.height <= 0:
            raise ExportError("preview contact sheet is invalid")

    return {
        "accepted_plus_skipped_equals_polygon_inside_candidates": True,
        "accepted_coordinates_strictly_inside_polygon": True,
        "source_windows_within_ecw_bounds": True,
        "raw_patches_150x150_single_channel": True,
        "raw_patches_without_red_marking": True,
        "raw_patch_count_checked": raw_checked,
        "preview_patches_150x150_rgb": True,
        "preview_coordinate_rings_checked": preview_checked,
        "preview_red_pixel_count_checked": preview_red_checked,
        "contact_sheet_preview_only": True,
    }


def markdown_summary(
    counts: dict[str, Any],
    inside_quality_mean: list[float],
    inside_quality_variance: list[float],
    inside_skip_counts: Counter[str],
    pre_polygon_counts: Counter[str],
    cache_info: dict[str, Any],
) -> str:
    mean_stats = metric_summary(inside_quality_mean)
    variance_stats = metric_summary(inside_quality_variance)

    def stats_line(label: str, stats: dict[str, Any]) -> str:
        if not stats.get("count"):
            return f"- {label}: no decodable windows"
        return (
            f"- {label}: n={stats['count']}, min={stats['min']:.3f}, p05={stats['p05']:.3f}, "
            f"median={stats['median']:.3f}, p95={stats['p95']:.3f}, max={stats['max']:.3f}"
        )

    lines = [
        "# Costa Head 150 px dataset quality summary",
        "",
        "The raw training inputs are native-resolution, unmarked, single-channel ECW windows. "
        "The official contact listing is retained as provenance and is not asserted to be verified ground truth.",
        "",
        "## Scope and result",
        "",
        f"- Costa point-contact records: **{counts['costa_point_records']}**.",
        f"- Strictly inside the supplied EPSG:32630 polygon: **{counts['polygon_inside_point_candidates']}** candidates.",
        f"- Accepted: **{counts['accepted']}**; skipped inside-polygon candidates: **{counts['skipped_inside_polygon']}**.",
        f"- Costa line contacts excluded from patching: **{counts['costa_line_records']}**.",
        f"- Point records outside the polygon: **{counts['outside_polygon_point_records']}**.",
        "",
        "## Fixed quality rule",
        "",
        f"- Accept only if `grey_variance >= {GREY_VARIANCE_MIN:g}` and `grey_mean < {GREY_MEAN_MAX_EXCLUSIVE:g}`.",
        "- Metrics were computed on the unmarked raw grayscale patch before any preview ring was drawn.",
        "",
        "## Decodable-window distribution",
        "",
        f"- Decodable inside-polygon windows: **{len(inside_quality_mean)}**.",
        stats_line("Grey mean", mean_stats),
        stats_line("Grey variance", variance_stats),
        "",
        "## Exclusion counts",
        "",
        "Inside-polygon candidate exclusions:",
    ]
    for reason, count in sorted(inside_skip_counts.items()):
        lines.append(f"- `{reason}`: **{count}**")
    lines.extend(["", "Pre-polygon point-record exclusions:"])
    for reason, count in sorted(pre_polygon_counts.items()):
        lines.append(f"- `{reason}`: **{count}**")
    lines.extend(
        [
            "",
            "## Reproducibility and split policy",
            "",
            "- No random sampling and no random train/validation/test split were performed.",
            "- ECW reads used bounded 150 x 150 windows, at most "
            f"{MAX_WORKERS} parallel decoder processes and a {DECODE_CACHE_SIZE}-window pixel cache.",
            f"- Decoder cache statistics: hits={cache_info['hits']}, misses={cache_info['misses']}, "
            f"maxsize={cache_info['maxsize']}.",
            "- Later train/val/test partitioning should use spatial blocks or survey/campaign groups to avoid adjacent-contact leakage.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    required_inputs = [POLYGON_PATH, CONTACTS_PATH, METADATA_PATH, ECW_PATH, ERS_PATH, DECODER_PATH]
    missing = [str(path) for path in required_inputs if not path.exists()]
    if missing:
        raise ExportError("missing required input(s): " + ", ".join(missing))
    if not DECODER_PATH.is_file():
        raise ExportError("ECW decoder is not a file")

    source_before = {str(path): snapshot(path) for path in required_inputs if path.is_file()}
    polygon_document, polygon_rings = load_polygon()
    metadata, raster = load_raster_metadata()
    contacts = json.loads(CONTACTS_PATH.read_text(encoding="utf-8"))
    if not isinstance(contacts, list):
        raise ExportError("contacts.json is not a list")

    OUT.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(POLYGON_PATH, POLYGON_COPY_PATH)

    costa_points = [row for row in contacts if isinstance(row, dict) and row.get("dataset") == "costa" and row.get("geometry") == "point"]
    costa_lines = [row for row in contacts if isinstance(row, dict) and row.get("dataset") == "costa" and row.get("geometry") == "line"]
    uid_counts = Counter(clean_text(row.get("uid")) for row in costa_points if clean_text(row.get("uid")))
    pre_polygon_counts: Counter[str] = Counter()
    inside_items: list[dict[str, Any]] = []

    for contact_list_index, row in enumerate(costa_points, start=1):
        e = finite_number(row.get("e"))
        n = finite_number(row.get("n"))
        if e is None or n is None:
            pre_polygon_counts["missing_or_non_numeric_coordinate"] += 1
            continue
        if not point_in_polygon_strict(e, n, polygon_rings):
            pre_polygon_counts["outside_polygon"] += 1
            continue
        inside_items.append(
            {
                "ordinal": len(inside_items) + 1,
                "contact_list_index": contact_list_index,
                "row": row,
                "e": e,
                "n": n,
            }
        )

    used_stems: dict[str, str] = {}
    stem_by_uid: dict[str, str] = {}
    for item in inside_items:
        uid = clean_text(item["row"].get("uid"))
        if uid and uid not in stem_by_uid:
            stem_by_uid[uid] = safe_stem(uid, used_stems)

    accepted_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    inside_skip_counts: Counter[str] = Counter()
    quality_means: list[float] = []
    quality_variances: list[float] = []

    decoder_items: list[dict[str, Any]] = []
    for item in inside_items:
        row = item["row"]
        uid = clean_text(row.get("uid"))
        if not uid:
            reason = "incomplete_fields"
            detail = "uid is missing; a stable reversible output filename cannot be created"
            skipped_rows.append(skipped_row(item, reason, detail))
            inside_skip_counts[reason] += 1
            continue
        if uid_counts[uid] > 1:
            reason = "duplicate_uid"
            detail = f"uid occurs {uid_counts[uid]} times in costa point contacts"
            skipped_rows.append(skipped_row(item, reason, detail))
            inside_skip_counts[reason] += 1
            continue
        decoder_items.append(item)

    def handle_result(result: dict[str, Any], item: dict[str, Any]) -> None:
        row = result["row"]
        window = result["window"]
        reason = result["skip_reason"]
        if reason:
            skipped_rows.append(
                skipped_row(
                    item,
                    reason,
                    result["skip_detail"],
                    window=window,
                    grey_mean=result["grey_mean"],
                    grey_variance=result["grey_variance"],
                )
            )
            inside_skip_counts[reason] += 1
            if result["grey_mean"] is not None:
                quality_means.append(float(result["grey_mean"]))
                quality_variances.append(float(result["grey_variance"]))
            return
        quality_means.append(float(result["grey_mean"]))
        quality_variances.append(float(result["grey_variance"]))
        stem = stem_by_uid[clean_text(row["uid"])]
        raw_name = f"{stem}.png"
        raw_relative = f"patches_raw/{raw_name}"
        preview_relative = f"patches_preview_marked/{raw_name}"
        raw_image, preview_image = make_preview(result["pixels"], window)
        raw_image.save(OUT / raw_relative, format="PNG", optimize=True)
        preview_image.save(OUT / preview_relative, format="PNG", optimize=True)
        accepted_rows.append(
            accepted_row(
                item,
                window,
                float(result["grey_mean"]),
                float(result["grey_variance"]),
                raw_relative,
                preview_relative,
            )
        )

    item_by_ordinal = {item["ordinal"]: item for item in decoder_items}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="costa-ecw") as executor:
        for start in range(0, len(decoder_items), MAX_IN_FLIGHT):
            batch = decoder_items[start : start + MAX_IN_FLIGHT]
            futures = {executor.submit(decode_candidate, item, raster): item for item in batch}
            results: list[dict[str, Any]] = []
            for future in as_completed(futures):
                results.append(future.result())
            for result in sorted(results, key=lambda value: value["ordinal"]):
                handle_result(result, item_by_ordinal[result["ordinal"]])

    accepted_rows.sort(key=lambda row: int(row["contact_list_index"]))
    skipped_rows.sort(key=lambda row: int(row["contact_list_index"]))
    write_csv(ACCEPTED_CSV_PATH, ACCEPTED_COLUMNS, accepted_rows)
    write_csv(SKIPPED_CSV_PATH, SKIPPED_COLUMNS, skipped_rows)

    contact_sheet_info = build_contact_sheet(accepted_rows)
    validation = validate_outputs(accepted_rows, skipped_rows, inside_items, polygon_rings, raster)

    source_after = {path: snapshot(Path(path)) for path in source_before}
    source_unchanged = {path: source_before[path] == source_after[path] for path in source_before}
    if not all(source_unchanged.values()):
        raise ExportError("one or more source files changed during export")

    cache = decode_window_pixels.cache_info()
    cache_info = {"hits": cache.hits, "misses": cache.misses, "maxsize": cache.maxsize, "currsize": cache.currsize}
    counts = {
        "contacts_total": len(contacts),
        "costa_point_records": len(costa_points),
        "costa_line_records": len(costa_lines),
        "polygon_inside_point_candidates": len(inside_items),
        "accepted": len(accepted_rows),
        "skipped_inside_polygon": len(skipped_rows),
        "accepted_plus_skipped": len(accepted_rows) + len(skipped_rows),
        "outside_polygon_point_records": pre_polygon_counts.get("outside_polygon", 0),
        "pre_polygon_exclusion_counts": dict(pre_polygon_counts),
        "inside_polygon_skip_reason_counts": dict(inside_skip_counts),
    }
    if counts["accepted_plus_skipped"] != counts["polygon_inside_point_candidates"]:
        raise ExportError("final candidate accounting failed")

    quality_summary_text = markdown_summary(
        counts,
        quality_means,
        quality_variances,
        inside_skip_counts,
        pre_polygon_counts,
        cache_info,
    )
    QUALITY_SUMMARY_PATH.write_text(quality_summary_text, encoding="utf-8")

    manifest = {
        "manifest_version": "costa-training-region-150px-v1",
        "created_by": str(Path(__file__).name),
        "dataset": "costa",
        "label_provenance_note": "Official contact-list listings are retained as provenance and are not asserted to be verified ground truth.",
        "inputs": {
            "polygon": {
                "source_path": str(POLYGON_PATH),
                "derived_copy": "polygon_used.geojson",
                "crs": "EPSG:32630",
                "sha256": sha256_file(POLYGON_PATH),
                "geometry": polygon_document["features"][0]["geometry"],
                "copy_byte_identical_to_source": POLYGON_COPY_PATH.read_bytes() == POLYGON_PATH.read_bytes(),
            },
            "contacts": {
                "source_path": str(CONTACTS_PATH),
                "sha256": sha256_file(CONTACTS_PATH),
                "record_count": len(contacts),
            },
            "ecw": {
                "source_path": str(ECW_PATH),
                "size_bytes": source_before[str(ECW_PATH)]["size_bytes"],
                "decoder": str(DECODER_PATH),
                "band": DECODER_BAND,
            },
            "ers": {
                "source_path": str(ERS_PATH),
                "size_bytes": source_before[str(ERS_PATH)]["size_bytes"],
            },
            "metadata": {
                "source_path": str(METADATA_PATH),
                "raster_id": metadata["datasets"]["costa"]["rasters"][0]["id"],
            },
        },
        "source_raster": {
            "crs": "EPSG:32630",
            "width_px": raster["width"],
            "height_px": raster["height"],
            "bounds_world_m": {
                "left": raster["left"],
                "right": raster["right"],
                "bottom": raster["bottom"],
                "top": raster["top"],
            },
            "native_resolution_m_per_px": NATIVE_RESOLUTION_M,
            "x_resolution_m_per_px": raster["resolution_x"],
            "y_resolution_m_per_px": raster["resolution_y"],
            "decoder_info": raster["decoder_info"],
            "read_policy": "Only bounded 150x150 source windows were requested from ecw_window; the full ECW was never decoded into RAM or converted to TIFF.",
        },
        "polygon": {
            "crs": "EPSG:32630",
            "strict_membership": True,
            "coordinates_world_m": polygon_rings,
        },
        "selection": {
            "dataset": "costa",
            "geometry": "point",
            "line_contacts_patch_policy": "excluded; no single centre patch generated",
            "coordinate_source": "official contact-list e/n fields",
            "polygon_boundary_policy": "boundary excluded; strict interior only",
            "thresholds": {
                "grey_variance_min_inclusive": GREY_VARIANCE_MIN,
                "grey_mean_max_exclusive": GREY_MEAN_MAX_EXCLUSIVE,
            },
            "quality_metrics_source": "unmarked raw grayscale patch",
        },
        "patch": {
            "width_px": PATCH_SIZE,
            "height_px": PATCH_SIZE,
            "physical_size_m": PATCH_METRES,
            "band": DECODER_BAND,
            "raw_mode": "L",
            "preview_mode": "RGB",
            "source_window_inclusive_pixel_coordinates": True,
            "pixel_mapping": {
                "x": "(e - raster.left) / resolution_x",
                "y": "(raster.top - n) / resolution_y",
                "upper_left": "round(pixel_center) - 75",
            },
            "patch_contact_reference_px": [PATCH_HALF, PATCH_HALF],
            "maximum_allowed_distance_from_reference_px": MAX_CENTER_DISTANCE_PX,
            "actual_coordinate_recorded_in_csv": True,
            "preview_ring": {
                "shape": "hollow_circle",
                "radius_px": MARKER_RADIUS_PX,
                "line_width_px": MARKER_WIDTH_PX,
                "meaning": "contact-list coordinate marker only; not a target footprint or size label",
            },
        },
        "randomness": {
            "used": False,
            "seed": None,
            "ordering": "contacts.json order for candidate accounting; output CSVs sorted by contact_list_index",
        },
        "concurrency_and_cache": {
            "max_workers": MAX_WORKERS,
            "max_in_flight_tasks": MAX_IN_FLIGHT,
            "decode_cache": cache_info,
        },
        "counts": counts,
        "files": {
            "raw_patches": "patches_raw/",
            "marked_previews": "patches_preview_marked/",
            "accepted_csv": "accepted_contacts.csv",
            "skipped_csv": "skipped_contacts.csv",
            "quality_summary": "quality_summary.md",
            "polygon_copy": "polygon_used.geojson",
            "preview_contact_sheet": contact_sheet_info,
        },
        "split_policy": {
            "random_split_performed": False,
            "future_recommendation": "Use spatial blocks or survey/campaign groups for train/validation/test, not random adjacent contacts.",
        },
        "source_integrity": {
            "checked_paths": source_unchanged,
            "all_checked_sources_unchanged": all(source_unchanged.values()),
        },
        "quality_distribution": {
            "grey_mean": metric_summary(quality_means),
            "grey_variance": metric_summary(quality_variances),
            "metrics_include_accepted_and_quality_rejected_decodable_inside_windows": True,
        },
        "validation": validation,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "output": str(OUT),
                "polygon_inside_candidates": len(inside_items),
                "accepted": len(accepted_rows),
                "skipped_inside_polygon": len(skipped_rows),
                "inside_skip_reason_counts": dict(inside_skip_counts),
                "pre_polygon_exclusion_counts": dict(pre_polygon_counts),
                "validation": validation,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"EXPORT FAILED: {exc}") from exc
