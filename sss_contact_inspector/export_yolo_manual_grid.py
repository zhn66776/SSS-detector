#!/usr/bin/env python3
"""Export the complete Costa Head SSS grid for manual YOLO annotation.

This exporter deliberately does not import or inspect any contact list.  It
enumerates one global raster-aligned grid, tests only the two supplied
patch-centre polygons, and reads one bounded 150 x 150 window at a time from
the project-local ECW decoder.

The output is a review/annotation candidate set, not a labelled YOLO dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import time
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image
from shapely.geometry import Point, shape
from shapely.prepared import prep


ROOT = Path(__file__).resolve().parent
ECW_PATH = ROOT / "data" / "derived_sources" / "costa" / "C13007a_Costa_AllAreas_Comments_Rev1.ecw"
ERS_PATH = ROOT / "data" / "derived_sources" / "costa" / "C13007a_Costa_AllAreas_Comments_Rev1.ers"
TRAIN_CENTRES_PATH = ROOT / "training_split_design" / "train_patch_centres_150px.geojson"
VAL_CENTRES_PATH = ROOT / "training_split_design" / "validation_patch_centres_150px.geojson"
MAIN_POLYGON_PATH = ROOT / "training_region_preview" / "approximate_training_region.geojson"
SPLIT_MANIFEST_PATH = ROOT / "training_split_design" / "split_manifest.json"
DECODER_PATH = ROOT / "local" / "bin" / "ecw_window"
OUTPUT_PATH = ROOT / "yolo_manual_annotation_grid_150px_stride100_v1"

CRS = "EPSG:32630"
EPSG = 32630
PIXEL_SIZE_M = 0.1
LEFT_E = 477323.46875
TOP_N = 6568269.5
PATCH_SIZE = 150
STRIDE = 100
OVERLAP = PATCH_SIZE - STRIDE
PATCH_SIDE_M = PATCH_SIZE * PIXEL_SIZE_M
PATCH_RADIUS_M = math.hypot(PATCH_SIDE_M / 2.0, PATCH_SIDE_M / 2.0)
REQUIRED_GAP_M = PATCH_SIDE_M
DECODER_BAND = 0

EXPECTED_WIDTH = 129132
EXPECTED_HEIGHT = 201580

PATCH_FIELDS = [
    "patch_id",
    "split",
    "grid_k",
    "grid_j",
    "x0",
    "y0",
    "x1",
    "y1",
    "centre_pixel_x",
    "centre_pixel_y",
    "centre_easting",
    "centre_northing",
    "corner_ul_easting",
    "corner_ul_northing",
    "corner_ur_easting",
    "corner_ur_northing",
    "corner_lr_easting",
    "corner_lr_northing",
    "corner_ll_easting",
    "corner_ll_northing",
    "footprint_left_easting",
    "footprint_right_easting",
    "footprint_top_northing",
    "footprint_bottom_northing",
    "file_relative_path",
    "decode_status",
    "image_width_px",
    "image_height_px",
    "image_mode",
    "file_bytes",
]

SKIPPED_FIELDS = [
    "patch_id",
    "split",
    "grid_k",
    "grid_j",
    "x0",
    "y0",
    "x1",
    "y1",
    "centre_pixel_x",
    "centre_pixel_y",
    "centre_easting",
    "centre_northing",
    "skip_reason",
    "skip_detail",
]


class ExportError(RuntimeError):
    """A hard contract or input/output validation failure."""


@dataclass(frozen=True)
class Candidate:
    split: str
    x0: int
    y0: int

    @property
    def x1(self) -> int:
        return self.x0 + PATCH_SIZE - 1

    @property
    def y1(self) -> int:
        return self.y0 + PATCH_SIZE - 1

    @property
    def grid_k(self) -> int:
        return self.x0 // STRIDE

    @property
    def grid_j(self) -> int:
        return self.y0 // STRIDE

    @property
    def patch_id(self) -> str:
        return f"costa_{self.split}_x{self.x0:06d}_y{self.y0:06d}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def require_files(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ExportError("required input missing: " + ", ".join(missing))


def load_geojson_polygon(path: Path, role: str):
    document = json.loads(path.read_text(encoding="utf-8"))
    features = document.get("features")
    if not isinstance(features, list) or len(features) != 1:
        raise ExportError(f"{role} must contain exactly one feature")
    feature = features[0]
    properties = feature.get("properties") or {}
    if str(properties.get("crs", "")).upper() != CRS:
        raise ExportError(f"{role} CRS is not {CRS}")
    geometry = feature.get("geometry")
    if not isinstance(geometry, dict) or geometry.get("type") not in {"Polygon", "MultiPolygon"}:
        raise ExportError(f"{role} geometry is not Polygon/MultiPolygon")
    geom = shape(geometry)
    if geom.is_empty or not geom.is_valid:
        raise ExportError(f"{role} geometry is empty or invalid")
    return document, geom


def read_decoder_info() -> dict[str, Any]:
    proc = subprocess.run(
        [str(DECODER_PATH), "info", str(ECW_PATH)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        raise ExportError(f"ECW metadata probe failed: {proc.stderr[-1000:]}")
    try:
        info = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise ExportError(f"ECW metadata probe returned invalid JSON: {proc.stdout[:500]!r}") from exc
    required = {
        "width": EXPECTED_WIDTH,
        "height": EXPECTED_HEIGHT,
        "bands": 1,
    }
    for key, value in required.items():
        if int(info.get(key, -1)) != value:
            raise ExportError(f"ECW {key}={info.get(key)!r}, expected {value}")
    if abs(float(info.get("cell_increment_x", 0.0)) - PIXEL_SIZE_M) > 1e-12:
        raise ExportError("ECW x cell increment does not match 0.1 m")
    if abs(float(info.get("cell_increment_y", 0.0)) + PIXEL_SIZE_M) > 1e-12:
        raise ExportError("ECW y cell increment does not match -0.1 m")
    if abs(float(info.get("origin_x", 0.0)) - LEFT_E) > 1e-9 or abs(float(info.get("origin_y", 0.0)) - TOP_N) > 1e-9:
        raise ExportError("ECW registration origin does not match the fixed input evidence")
    return info


def format_float(value: float) -> str:
    return f"{value:.9f}"


def world_from_pixel(px: int | float, py: int | float) -> tuple[float, float]:
    return LEFT_E + float(px) * PIXEL_SIZE_M, TOP_N - float(py) * PIXEL_SIZE_M


def candidate_window_metadata(candidate: Candidate) -> dict[str, Any]:
    cx = candidate.x0 + PATCH_SIZE // 2
    cy = candidate.y0 + PATCH_SIZE // 2
    centre_e, centre_n = world_from_pixel(cx, cy)
    ul_e, ul_n = world_from_pixel(candidate.x0, candidate.y0)
    ur_e, ur_n = world_from_pixel(candidate.x1, candidate.y0)
    lr_e, lr_n = world_from_pixel(candidate.x1, candidate.y1)
    ll_e, ll_n = world_from_pixel(candidate.x0, candidate.y1)
    # Pixel-edge footprint bounds are also recorded so the interval used by
    # the split validator is explicit.  Corner coordinates use the supplied
    # pixel registration formula at the four outermost sample pixels.
    left_e, _ = world_from_pixel(candidate.x0, 0)
    right_e, _ = world_from_pixel(candidate.x1 + 1, 0)
    _, top_n = world_from_pixel(0, candidate.y0)
    _, bottom_n = world_from_pixel(0, candidate.y1 + 1)
    return {
        "patch_id": candidate.patch_id,
        "split": candidate.split,
        "grid_k": candidate.grid_k,
        "grid_j": candidate.grid_j,
        "x0": candidate.x0,
        "y0": candidate.y0,
        "x1": candidate.x1,
        "y1": candidate.y1,
        "centre_pixel_x": cx,
        "centre_pixel_y": cy,
        "centre_easting": format_float(centre_e),
        "centre_northing": format_float(centre_n),
        "corner_ul_easting": format_float(ul_e),
        "corner_ul_northing": format_float(ul_n),
        "corner_ur_easting": format_float(ur_e),
        "corner_ur_northing": format_float(ur_n),
        "corner_lr_easting": format_float(lr_e),
        "corner_lr_northing": format_float(lr_n),
        "corner_ll_easting": format_float(ll_e),
        "corner_ll_northing": format_float(ll_n),
        "footprint_left_easting": format_float(left_e),
        "footprint_right_easting": format_float(right_e),
        "footprint_top_northing": format_float(top_n),
        "footprint_bottom_northing": format_float(bottom_n),
    }


def grid_candidates(geom, split: str, width: int, height: int) -> list[Candidate]:
    """Enumerate points of one global 100 px grid inside a centre geometry."""

    prepared = prep(geom)
    min_e, min_n, max_e, max_n = geom.bounds
    min_px = (min_e - LEFT_E) / PIXEL_SIZE_M
    max_px = (max_e - LEFT_E) / PIXEL_SIZE_M
    # N decreases with raster y.  A broad integer range plus a strict point
    # test is less error-prone than boundary-specific ceil/floor arithmetic.
    min_py = (TOP_N - max_n) / PIXEL_SIZE_M
    max_py = (TOP_N - min_n) / PIXEL_SIZE_M
    max_x0 = ((width - PATCH_SIZE) // STRIDE) * STRIDE
    max_y0 = ((height - PATCH_SIZE) // STRIDE) * STRIDE
    first_k = max(0, math.floor((min_px - PATCH_SIZE // 2) / STRIDE) - 1)
    last_k = min(max_x0 // STRIDE, math.ceil((max_px - PATCH_SIZE // 2) / STRIDE) + 1)
    first_j = max(0, math.floor((min_py - PATCH_SIZE // 2) / STRIDE) - 1)
    last_j = min(max_y0 // STRIDE, math.ceil((max_py - PATCH_SIZE // 2) / STRIDE) + 1)
    candidates: list[Candidate] = []
    for grid_j in range(first_j, last_j + 1):
        y0 = grid_j * STRIDE
        cy = y0 + PATCH_SIZE // 2
        for grid_k in range(first_k, last_k + 1):
            x0 = grid_k * STRIDE
            cx = x0 + PATCH_SIZE // 2
            e, n = world_from_pixel(cx, cy)
            # Shapely contains is strict: a centre exactly on a polygon edge
            # is not silently assigned to a split.
            if prepared.contains(Point(e, n)):
                candidates.append(Candidate(split, x0, y0))
    return candidates


def by_position(candidates: Iterable[Candidate]) -> dict[tuple[int, int], Candidate]:
    return {(candidate.x0, candidate.y0): candidate for candidate in candidates}


def rectangle_gap_px(a: Candidate, b: Candidate) -> tuple[int, int, float]:
    dx = max(0, abs(a.x0 - b.x0) - PATCH_SIZE)
    dy = max(0, abs(a.y0 - b.y0) - PATCH_SIZE)
    return dx, dy, math.hypot(dx, dy)


def nearest_train_for_validation(
    validation: Candidate,
    train_by_x: dict[int, list[int]],
    sorted_train_x: list[int],
) -> tuple[float, Candidate | None]:
    """Exact nearest rectangle gap for one validation patch.

    For a fixed train x0 the best y0 is one of the two values bracketing the
    validation y0.  Train x0 values are visited in increasing absolute
    distance, and the x-only lower bound permits an exact early stop.
    """

    if not sorted_train_x:
        return math.inf, None
    insert = bisect_left(sorted_train_x, validation.x0)
    left = insert - 1
    right = insert
    best = math.inf
    best_candidate: Candidate | None = None
    visited_x: set[int] = set()
    while left >= 0 or right < len(sorted_train_x):
        left_distance = abs(sorted_train_x[left] - validation.x0) if left >= 0 else math.inf
        right_distance = abs(sorted_train_x[right] - validation.x0) if right < len(sorted_train_x) else math.inf
        distance = min(left_distance, right_distance)
        lower_bound = max(0, distance - PATCH_SIZE)
        if lower_bound > best:
            break
        choices: list[int] = []
        if left >= 0 and left_distance == distance:
            choices.append(sorted_train_x[left])
            left -= 1
        if right < len(sorted_train_x) and right_distance == distance:
            choices.append(sorted_train_x[right])
            right += 1
        for train_x in choices:
            if train_x in visited_x:
                continue
            visited_x.add(train_x)
            train_ys = train_by_x[train_x]
            y_insert = bisect_left(train_ys, validation.y0)
            y_indices = {index for index in (y_insert - 1, y_insert) if 0 <= index < len(train_ys)}
            for y_index in y_indices:
                train_y = train_ys[y_index]
                train_candidate = Candidate("train", train_x, train_y)
                _, _, gap = rectangle_gap_px(validation, train_candidate)
                if gap < best:
                    best = gap
                    best_candidate = train_candidate
    return best, best_candidate


def split_footprint_validation(train: list[Candidate], validation: list[Candidate]) -> dict[str, Any]:
    """Validate every split pair's exact axis-aligned pixel footprint distance."""

    if not train or not validation:
        raise ExportError("train and validation candidate sets must both be non-empty")
    train_by_x: dict[int, list[int]] = {}
    train_by_position: dict[tuple[int, int], Candidate] = {}
    for candidate in train:
        train_by_x.setdefault(candidate.x0, []).append(candidate.y0)
        train_by_position[(candidate.x0, candidate.y0)] = candidate
    for y_values in train_by_x.values():
        y_values.sort()
    sorted_train_x = sorted(train_by_x)

    min_gap = math.inf
    min_pair: tuple[Candidate, Candidate] | None = None
    nearest_queries = 0
    for validation_candidate in validation:
        gap, train_candidate = nearest_train_for_validation(validation_candidate, train_by_x, sorted_train_x)
        nearest_queries += 1
        if train_candidate is not None and gap < min_gap:
            min_gap = gap
            min_pair = (train_candidate, validation_candidate)

    # Separately count actual shared native pixels.  The exact integer test is
    # stricter than the continuous edge-distance calculation at a touching
    # boundary: two windows share pixels iff both top-left differences are
    # below 150 px.
    overlap_count = 0
    first_overlap: tuple[Candidate, Candidate] | None = None
    for validation_candidate in validation:
        x_start = bisect_left(sorted_train_x, validation_candidate.x0 - (PATCH_SIZE - 1))
        x_stop = bisect_left(sorted_train_x, validation_candidate.x0 + PATCH_SIZE)
        for train_x in sorted_train_x[x_start:x_stop]:
            y_values = train_by_x[train_x]
            y_start = bisect_left(y_values, validation_candidate.y0 - (PATCH_SIZE - 1))
            y_stop = bisect_left(y_values, validation_candidate.y0 + PATCH_SIZE)
            for train_y in y_values[y_start:y_stop]:
                overlap_count += 1
                if first_overlap is None:
                    first_overlap = (train_by_position[(train_x, train_y)], validation_candidate)

    min_gap_m = min_gap * PIXEL_SIZE_M
    result: dict[str, Any] = {
        "validation_passed": overlap_count == 0 and min_gap_m + 1e-9 >= REQUIRED_GAP_M,
        "train_patch_count": len(train),
        "validation_patch_count": len(validation),
        "cross_split_pair_count": len(train) * len(validation),
        "exact_nearest_neighbor_queries": nearest_queries,
        "pixel_footprint_definition": "half-open [x0,x0+150) x [y0,y0+150); CSV x1/y1 are inclusive sample indices",
        "minimum_gap_px": min_gap,
        "minimum_gap_m": min_gap_m,
        "required_minimum_gap_px": PATCH_SIZE,
        "required_minimum_gap_m": REQUIRED_GAP_M,
        "overlapping_cross_split_pair_count": overlap_count,
        "overlap_check_passed": overlap_count == 0,
        "minimum_gap_check_passed": min_gap_m + 1e-9 >= REQUIRED_GAP_M,
    }
    if min_pair is not None:
        result["minimum_gap_pair"] = {
            "train_patch_id": min_pair[0].patch_id,
            "validation_patch_id": min_pair[1].patch_id,
            "train_x0": min_pair[0].x0,
            "train_y0": min_pair[0].y0,
            "validation_x0": min_pair[1].x0,
            "validation_y0": min_pair[1].y0,
        }
    if first_overlap is not None:
        result["first_overlap_pair"] = {
            "train_patch_id": first_overlap[0].patch_id,
            "validation_patch_id": first_overlap[1].patch_id,
            "train_x0": first_overlap[0].x0,
            "train_y0": first_overlap[0].y0,
            "validation_x0": first_overlap[1].x0,
            "validation_y0": first_overlap[1].y0,
        }
    return result


class PersistentECWWorker:
    """One serialized ECW worker using only bounded decoder windows."""

    def __init__(self) -> None:
        self.process: subprocess.Popen[bytes] | None = None
        self.request_id = 0
        self.start()

    def start(self) -> None:
        self.close()
        self.process = subprocess.Popen(
            [str(DECODER_PATH), "serve", str(ECW_PATH)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert self.process.stdout is not None
        line = self.process.stdout.readline()
        if not line:
            error = self.process.stderr.read().decode("utf-8", errors="replace") if self.process.stderr else ""
            raise ExportError(f"persistent ECW worker failed to start: {error[-1000:]}")
        try:
            info = json.loads(line.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ExportError(f"persistent ECW worker returned invalid startup metadata: {line[:500]!r}") from exc
        if int(info.get("width", -1)) != EXPECTED_WIDTH or int(info.get("height", -1)) != EXPECTED_HEIGHT:
            raise ExportError(f"persistent ECW worker metadata mismatch: {info}")

    @staticmethod
    def read_exact(stream: Any, size: int) -> bytes:
        parts: list[bytes] = []
        remaining = size
        while remaining:
            chunk = stream.read(remaining)
            if not chunk:
                raise ExportError("persistent ECW worker ended during image response")
            parts.append(chunk)
            remaining -= len(chunk)
        return b"".join(parts)

    def decode(self, candidate: Candidate) -> bytes:
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                if self.process is None or self.process.poll() is not None:
                    self.start()
                assert self.process and self.process.stdin and self.process.stdout
                self.request_id += 1
                request_id = str(self.request_id)
                command = f"{request_id} {candidate.x0} {candidate.y0} {candidate.x1} {candidate.y1} {PATCH_SIZE} {PATCH_SIZE} {DECODER_BAND}\n".encode("ascii")
                self.process.stdin.write(command)
                self.process.stdin.flush()
                header = self.process.stdout.readline().decode("utf-8", errors="replace").strip()
                parts = header.split(" ", 2)
                if len(parts) < 3 or parts[1] != request_id:
                    raise ExportError(f"invalid persistent ECW response: {header}")
                if parts[0] != "OK":
                    raise ExportError(parts[2])
                return self.read_exact(self.process.stdout, int(parts[2]))
            except Exception as exc:  # retry once after a worker restart
                last_error = exc
                self.close()
                if attempt == 0:
                    continue
        raise ExportError(str(last_error) if last_error else "unknown ECW decode failure")

    def close(self) -> None:
        process = self.process
        self.process = None
        if process is None:
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


def image_is_valid(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            if image.format != "PNG" or image.size != (PATCH_SIZE, PATCH_SIZE) or image.mode != "L":
                return False
            image.load()
        return True
    except Exception:
        return False


def write_csv_atomic(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_progress_row(handle, fields: list[str], row: dict[str, Any]) -> None:
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
    writer.writerow(row)


def output_file_for(candidate: Candidate) -> Path:
    return OUTPUT_PATH / "images" / ("train" if candidate.split == "train" else "val") / f"{candidate.patch_id}.png"


def make_patch_row(candidate: Candidate, status: str, path: Path) -> dict[str, Any]:
    row = candidate_window_metadata(candidate)
    row.update(
        {
            "file_relative_path": path.relative_to(OUTPUT_PATH).as_posix(),
            "decode_status": status,
            "image_width_px": PATCH_SIZE,
            "image_height_px": PATCH_SIZE,
            "image_mode": "L",
            "file_bytes": path.stat().st_size,
        }
    )
    return row


def make_skipped_row(candidate: Candidate, reason: str, detail: str) -> dict[str, Any]:
    row = candidate_window_metadata(candidate)
    row.update({"skip_reason": reason, "skip_detail": detail})
    return row


def estimate_space(candidate_count: int) -> dict[str, Any]:
    # A lossless 8-bit 150x150 payload is 22,500 bytes.  1,024 bytes per PNG
    # is a deliberately conservative small-file container allowance; actual
    # sizes are reported after export.
    payload = candidate_count * PATCH_SIZE * PATCH_SIZE
    conservative = candidate_count * (PATCH_SIZE * PATCH_SIZE + 1024)
    return {
        "raw_pixel_payload_bytes": payload,
        "conservative_png_estimate_bytes": conservative,
        "conservative_png_estimate_gib": conservative / (1024**3),
    }


def prepare_candidates() -> dict[str, Any]:
    require_files(
        [
            ECW_PATH,
            ERS_PATH,
            TRAIN_CENTRES_PATH,
            VAL_CENTRES_PATH,
            MAIN_POLYGON_PATH,
            SPLIT_MANIFEST_PATH,
            DECODER_PATH,
        ]
    )
    decoder_info = read_decoder_info()
    train_document, train_geom = load_geojson_polygon(TRAIN_CENTRES_PATH, "train patch-centre polygon")
    val_document, val_geom = load_geojson_polygon(VAL_CENTRES_PATH, "validation patch-centre polygon")
    main_document, main_geom = load_geojson_polygon(MAIN_POLYGON_PATH, "Costa Head main polygon")
    split_manifest = json.loads(SPLIT_MANIFEST_PATH.read_text(encoding="utf-8"))
    if split_manifest.get("crs") != CRS:
        raise ExportError("split manifest CRS is not EPSG:32630")
    if split_manifest.get("patch", {}).get("pixels") != PATCH_SIZE or split_manifest.get("patch", {}).get("stride_px") != STRIDE:
        raise ExportError("split manifest patch/stride does not match the requested grid")
    if abs(float(split_manifest.get("split_rule", {}).get("minimum_train_validation_image_gap_m", -1)) - REQUIRED_GAP_M) > 1e-9:
        raise ExportError("split manifest minimum gap does not match 15 m")

    train = grid_candidates(train_geom, "train", EXPECTED_WIDTH, EXPECTED_HEIGHT)
    validation = grid_candidates(val_geom, "val", EXPECTED_WIDTH, EXPECTED_HEIGHT)
    main_patch_centres = main_geom.buffer(-PATCH_RADIUS_M, join_style=2)
    main_candidates = grid_candidates(main_patch_centres, "buffer", EXPECTED_WIDTH, EXPECTED_HEIGHT)
    train_positions = {(candidate.x0, candidate.y0) for candidate in train}
    validation_positions = {(candidate.x0, candidate.y0) for candidate in validation}
    main_positions = {(candidate.x0, candidate.y0) for candidate in main_candidates}
    overlap_positions = train_positions & validation_positions
    if overlap_positions:
        raise ExportError(f"train/validation centre polygons select {len(overlap_positions)} identical grid windows")
    buffer_positions = main_positions - train_positions - validation_positions
    outside_train = sum(
        1
        for candidate in train
        if candidate.x0 < 0 or candidate.y0 < 0 or candidate.x1 >= EXPECTED_WIDTH or candidate.y1 >= EXPECTED_HEIGHT
    )
    outside_validation = sum(
        1
        for candidate in validation
        if candidate.x0 < 0 or candidate.y0 < 0 or candidate.x1 >= EXPECTED_WIDTH or candidate.y1 >= EXPECTED_HEIGHT
    )
    split_validation = split_footprint_validation(train, validation)
    return {
        "decoder_info": decoder_info,
        "train_document": train_document,
        "validation_document": val_document,
        "main_document": main_document,
        "split_manifest": split_manifest,
        "train": train,
        "validation": validation,
        "main_candidates": main_candidates,
        "buffer_count": len(buffer_positions),
        "outside_train_count": outside_train,
        "outside_validation_count": outside_validation,
        "split_validation": split_validation,
    }


def print_dry_run(prepared: dict[str, Any]) -> None:
    train = prepared["train"]
    validation = prepared["validation"]
    main_candidates = prepared["main_candidates"]
    exportable = len(train) + len(validation) - prepared["outside_train_count"] - prepared["outside_validation_count"]
    estimate = estimate_space(exportable)
    split_validation = prepared["split_validation"]
    payload = {
        "mode": "dry-run",
        "output_path": str(OUTPUT_PATH),
        "input_ecw": str(ECW_PATH),
        "raster": {"width_px": EXPECTED_WIDTH, "height_px": EXPECTED_HEIGHT, "crs": CRS},
        "grid": {
            "patch_px": PATCH_SIZE,
            "stride_px": STRIDE,
            "overlap_px": OVERLAP,
            "grid_origin_px": [0, 0],
            "centre_offset_px": [PATCH_SIZE // 2, PATCH_SIZE // 2],
        },
        "counts": {
            "train_candidate_windows": len(train),
            "validation_candidate_windows": len(validation),
            "main_black_box_full_window_candidates": len(main_candidates),
            "buffer_windows_not_in_any_split": prepared["buffer_count"],
            "train_out_of_raster_windows": prepared["outside_train_count"],
            "validation_out_of_raster_windows": prepared["outside_validation_count"],
            "expected_exportable_windows": exportable,
        },
        "space_estimate": estimate,
        "split_spacing": {
            "manifest_theoretical_minimum_gap_m": REQUIRED_GAP_M,
            "discrete_grid_minimum_footprint_gap_m": split_validation["minimum_gap_m"],
            "discrete_grid_minimum_footprint_gap_px": split_validation["minimum_gap_px"],
            "overlapping_pair_count": split_validation["overlapping_cross_split_pair_count"],
            "passes_15m_requirement": split_validation["validation_passed"],
        },
        "read_policy": "No image is written; only metadata, polygon geometry and the exact global-grid candidate counts are evaluated.",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def load_existing_complete_check() -> None:
    complete = OUTPUT_PATH / "COMPLETE.json"
    if complete.exists():
        raise ExportError(f"output already has COMPLETE.json; use a new output directory: {OUTPUT_PATH}")


def assert_output_mode(resume: bool) -> None:
    if OUTPUT_PATH.exists():
        if not resume and any(OUTPUT_PATH.iterdir()):
            raise ExportError(f"output directory is non-empty; use --resume only for an interrupted export: {OUTPUT_PATH}")
        if resume:
            load_existing_complete_check()
    else:
        OUTPUT_PATH.mkdir(parents=False)
    (OUTPUT_PATH / "images" / "train").mkdir(parents=True, exist_ok=True)
    (OUTPUT_PATH / "images" / "val").mkdir(parents=True, exist_ok=True)


def write_progress_headers() -> tuple[Any, Any]:
    patch_path = OUTPUT_PATH / "patch_index.csv"
    skipped_path = OUTPUT_PATH / "skipped_windows.csv"
    patch_exists = patch_path.exists() and patch_path.stat().st_size > 0
    skipped_exists = skipped_path.exists() and skipped_path.stat().st_size > 0
    patch_handle = patch_path.open("a", encoding="utf-8", newline="")
    skipped_handle = skipped_path.open("a", encoding="utf-8", newline="")
    if not patch_exists:
        patch_handle.write(",".join(PATCH_FIELDS) + "\n")
    if not skipped_exists:
        skipped_handle.write(",".join(SKIPPED_FIELDS) + "\n")
    return patch_handle, skipped_handle


def export_images(prepared: dict[str, Any], resume: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    candidates: list[Candidate] = list(prepared["train"]) + list(prepared["validation"])
    candidate_positions = {(candidate.split, candidate.x0, candidate.y0) for candidate in candidates}
    progress_patch_path = OUTPUT_PATH / "patch_index.csv"
    progress_skip_path = OUTPUT_PATH / "skipped_windows.csv"
    patch_handle, skipped_handle = write_progress_headers()
    patch_handle.flush()
    skipped_handle.flush()
    worker = PersistentECWWorker()
    rows_by_id: dict[str, dict[str, Any]] = {}
    skipped_by_id: dict[str, dict[str, Any]] = {}
    reused = 0
    decoded = 0
    skipped_counts: dict[str, int] = {}
    started = time.monotonic()
    last_report = started
    completed = 0
    try:
        for candidate in candidates:
            completed += 1
            output_file = output_file_for(candidate)
            if resume and output_file.is_file() and image_is_valid(output_file):
                row = make_patch_row(candidate, "resumed_existing_valid_png", output_file)
                rows_by_id[candidate.patch_id] = row
                reused += 1
            elif candidate.x0 < 0 or candidate.y0 < 0 or candidate.x1 >= EXPECTED_WIDTH or candidate.y1 >= EXPECTED_HEIGHT:
                row = make_skipped_row(candidate, "window_out_of_raster_bounds", "150x150 window is not fully inside the ECW raster")
                skipped_by_id[candidate.patch_id] = row
                skipped_counts["window_out_of_raster_bounds"] = skipped_counts.get("window_out_of_raster_bounds", 0) + 1
                append_progress_row(skipped_handle, SKIPPED_FIELDS, row)
            else:
                try:
                    payload = worker.decode(candidate)
                    with Image.open(io.BytesIO(payload)) as decoded_image:
                        decoded_image.load()
                        if decoded_image.size != (PATCH_SIZE, PATCH_SIZE) or decoded_image.mode != "L":
                            raise ExportError(f"decoder returned {decoded_image.mode} {decoded_image.size}, expected L (150,150)")
                        image = decoded_image.copy()
                    temporary = output_file.with_name(f".{output_file.name}.{os.getpid()}.tmp")
                    image.save(temporary, format="PNG", optimize=True)
                    image.close()
                    os.replace(temporary, output_file)
                    if not image_is_valid(output_file):
                        raise ExportError("written PNG failed 150x150 L integrity check")
                    row = make_patch_row(candidate, "decoded", output_file)
                    rows_by_id[candidate.patch_id] = row
                    decoded += 1
                    append_progress_row(patch_handle, PATCH_FIELDS, row)
                except Exception as exc:
                    temporary = output_file.with_name(f".{output_file.name}.{os.getpid()}.tmp")
                    if temporary.exists():
                        temporary.unlink()
                    row = make_skipped_row(candidate, "ecw_decode_failed", str(exc)[:1000])
                    skipped_by_id[candidate.patch_id] = row
                    skipped_counts["ecw_decode_failed"] = skipped_counts.get("ecw_decode_failed", 0) + 1
                    append_progress_row(skipped_handle, SKIPPED_FIELDS, row)

            if completed % 256 == 0:
                patch_handle.flush()
                skipped_handle.flush()
            now = time.monotonic()
            if completed == len(candidates) or completed % 1000 == 0 or now - last_report >= 10.0:
                elapsed = max(now - started, 1e-9)
                rate = completed / elapsed
                remaining = (len(candidates) - completed) / rate if rate else math.inf
                print(
                    f"progress completed={completed}/{len(candidates)} "
                    f"decoded_new={decoded} resumed={reused} skipped={sum(skipped_counts.values())} "
                    f"rate={rate:.2f}/s eta_s={remaining:.0f}",
                    flush=True,
                )
                last_report = now
    finally:
        patch_handle.flush()
        skipped_handle.flush()
        patch_handle.close()
        skipped_handle.close()
        worker.close()

    if set((row["split"], int(row["x0"]), int(row["y0"])) for row in rows_by_id.values()) | set(
        (row["split"], int(row["x0"]), int(row["y0"])) for row in skipped_by_id.values()
    ) != candidate_positions:
        raise ExportError("progress accounting does not cover every train/validation candidate")

    rows = sorted(rows_by_id.values(), key=lambda row: (row["split"], int(row["y0"]), int(row["x0"])))
    skipped_rows = sorted(skipped_by_id.values(), key=lambda row: (row["split"], int(row["y0"]), int(row["x0"])))
    write_csv_atomic(OUTPUT_PATH / "patch_index.csv", PATCH_FIELDS, rows)
    write_csv_atomic(OUTPUT_PATH / "skipped_windows.csv", SKIPPED_FIELDS, skipped_rows)
    return rows, skipped_rows, {
        "newly_decoded": decoded,
        "resumed_existing_valid_png": reused,
        "skip_reason_counts": skipped_counts,
    }


def validate_image_files(rows: list[dict[str, Any]], candidates: list[Candidate]) -> dict[str, Any]:
    expected_paths: set[Path] = set()
    total_bytes = 0
    for row in rows:
        path = OUTPUT_PATH / row["file_relative_path"]
        expected_paths.add(path)
        if not image_is_valid(path):
            raise ExportError(f"invalid exported image: {path}")
        actual_size = path.stat().st_size
        total_bytes += actual_size
        if int(row["file_bytes"]) != actual_size:
            raise ExportError(f"patch_index file size mismatch: {path}")
    actual_pngs = {
        path
        for split in ("train", "val")
        for path in (OUTPUT_PATH / "images" / split).glob("*.png")
    }
    unexpected = actual_pngs - expected_paths
    missing = expected_paths - actual_pngs
    if unexpected or missing:
        raise ExportError(f"image/index mismatch: unexpected={len(unexpected)} missing={len(missing)}")
    if len(rows) != len(candidates):
        raise ExportError(f"exported row count {len(rows)} != candidate count {len(candidates)}")
    return {"patch_count": len(rows), "total_png_bytes": total_bytes, "all_images_valid": True}


def write_readme() -> None:
    readme = """# Costa Head SSS manual annotation grid

This directory holds unlabelled, single-channel grayscale patches for manual object-detection annotation. Draw and review YOLO bounding boxes and class labels before using the patches for training.

The images are cut from the Costa Head ECW at native 0.1 m/px resolution using
a global raster-aligned grid. Each image is 150 x 150 pixels (15 x 15 m), the
grid stride is 100 pixels (10 m), and the overlap is 50 pixels (33.333%). The
global grid origin is raster pixel (0, 0); a patch at `(100*k, 100*j)` has its
recorded centre at `(100*k+75, 100*j+75)`.

Exported patch centres lie strictly inside the supplied train or validation patch-centre polygon. Windows that are outside the ECW or fail
the project-local ECW window decoder are recorded in `skipped_windows.csv`.
Patch selection uses the grid and polygon geometry. The PNG files contain the unannotated source pixels.

`patch_index.csv` records the raster window, centre and corner coordinates,
split, relative image path, and decode status. `split_validation.json` records
the exhaustive cross-split footprint-spacing check. `dataset_manifest.json`
records the input hashes, raster registration, grid rules, counts and byte
totals.
"""
    (OUTPUT_PATH / "README.md").write_text(readme, encoding="utf-8")


def build_manifest(
    prepared: dict[str, Any],
    rows: list[dict[str, Any]],
    skipped_rows: list[dict[str, Any]],
    export_stats: dict[str, Any],
    image_stats: dict[str, Any],
    source_hashes_before: dict[str, str],
    source_snapshots_before: dict[str, dict[str, int]],
    source_snapshots_after: dict[str, dict[str, int]],
    split_validation: dict[str, Any],
) -> dict[str, Any]:
    train_count = sum(1 for row in rows if row["split"] == "train")
    val_count = sum(1 for row in rows if row["split"] == "val")
    skipped_counts: dict[str, int] = {}
    for row in skipped_rows:
        skipped_counts[row["skip_reason"]] = skipped_counts.get(row["skip_reason"], 0) + 1
    candidate_count = len(prepared["train"]) + len(prepared["validation"])
    source_unchanged = source_snapshots_before == source_snapshots_after
    manifest: dict[str, Any] = {
        "manifest_version": "costa-yolo-manual-annotation-grid-150px-stride100-v1",
        "purpose": "Unlabelled raw SSS grid candidate set for later human YOLO detection annotation.",
        "labels_present": False,
        "yolo_bbox_or_class_labels_present": False,
        "all_annotations_still_pending_manual_review": True,
        "inputs": {
            "ecw": {
                "path": str(ECW_PATH),
                "sha256": source_hashes_before[str(ECW_PATH)],
                "size_bytes": source_snapshots_before[str(ECW_PATH)]["size_bytes"],
                "read_only": True,
            },
            "ers": {
                "path": str(ERS_PATH),
                "sha256": source_hashes_before[str(ERS_PATH)],
                "size_bytes": source_snapshots_before[str(ERS_PATH)]["size_bytes"],
                "read_only": True,
            },
            "train_patch_centres": {"path": str(TRAIN_CENTRES_PATH), "sha256": source_hashes_before[str(TRAIN_CENTRES_PATH)]},
            "validation_patch_centres": {"path": str(VAL_CENTRES_PATH), "sha256": source_hashes_before[str(VAL_CENTRES_PATH)]},
            "main_black_box_polygon_for_buffer_count": {"path": str(MAIN_POLYGON_PATH), "sha256": source_hashes_before[str(MAIN_POLYGON_PATH)]},
            "split_manifest": {"path": str(SPLIT_MANIFEST_PATH), "sha256": source_hashes_before[str(SPLIT_MANIFEST_PATH)]},
        },
        "source_raster": {
            "crs": CRS,
            "epsg": EPSG,
            "width_px": EXPECTED_WIDTH,
            "height_px": EXPECTED_HEIGHT,
            "bands_read": 1,
            "native_resolution_m_per_px": PIXEL_SIZE_M,
            "registration": {
                "left": LEFT_E,
                "top": TOP_N,
                "pixel_x": "(E - left) / 0.1",
                "pixel_y": "(top - N) / 0.1",
            },
            "decoder_info": prepared["decoder_info"],
            "read_policy": "Only bounded 150x150 windows were read through the project-local persistent ECW decoder; the full ECW was never loaded into memory.",
        },
        "grid": {
            "patch_width_px": PATCH_SIZE,
            "patch_height_px": PATCH_SIZE,
            "patch_side_m": PATCH_SIDE_M,
            "stride_px": STRIDE,
            "stride_m": STRIDE * PIXEL_SIZE_M,
            "overlap_px": OVERLAP,
            "overlap_fraction": OVERLAP / PATCH_SIZE,
            "grid_origin_pixel": [0, 0],
            "top_left_rule": "x0=100*k, y0=100*j",
            "centre_rule": "centre=(100*k+75,100*j+75)",
            "window_inclusive_indices": "x0..x1 and y0..y1, with x1=x0+149 and y1=y0+149",
        },
        "split_rule": {
            "train_centres": str(TRAIN_CENTRES_PATH),
            "validation_centres": str(VAL_CENTRES_PATH),
            "strict_polygon_membership": True,
            "buffer_definition": "full-window grid centres inside the Costa Head main polygon inset by the split-design footprint radius, excluding train and validation centre candidates",
            "required_train_validation_footprint_gap_m": REQUIRED_GAP_M,
            "buffer_patch_not_in_any_split": True,
        },
        "counts": {
            "train_candidate_windows": len(prepared["train"]),
            "validation_candidate_windows": len(prepared["validation"]),
            "expected_train_validation_windows": candidate_count,
            "exported_train": train_count,
            "exported_validation": val_count,
            "exported_total": len(rows),
            "buffer_windows_discarded_not_exported": prepared["buffer_count"],
            "skipped_total": len(skipped_rows),
            "skipped_by_reason": skipped_counts,
            "decode_failures": skipped_counts.get("ecw_decode_failed", 0),
            "out_of_bounds_windows": skipped_counts.get("window_out_of_raster_bounds", 0),
            "newly_decoded": export_stats["newly_decoded"],
            "resumed_existing_valid_png": export_stats["resumed_existing_valid_png"],
        },
        "storage": image_stats,
        "split_validation": split_validation,
        "source_integrity": {
            "source_snapshots_before": source_snapshots_before,
            "source_snapshots_after": source_snapshots_after,
            "all_fixed_inputs_unchanged_by_size_and_mtime": source_unchanged,
        },
        "artifacts": {
            "images_train": "images/train/",
            "images_validation": "images/val/",
            "patch_index": "patch_index.csv",
            "skipped_windows": "skipped_windows.csv",
            "split_validation": "split_validation.json",
            "readme": "README.md",
        },
    }
    return manifest


def write_complete(manifest: dict[str, Any], split_validation: dict[str, Any], image_stats: dict[str, Any], skipped_rows: list[dict[str, Any]]) -> None:
    required_files = [
        OUTPUT_PATH / "patch_index.csv",
        OUTPUT_PATH / "skipped_windows.csv",
        OUTPUT_PATH / "dataset_manifest.json",
        OUTPUT_PATH / "split_validation.json",
        OUTPUT_PATH / "README.md",
    ]
    for path in required_files:
        if not path.is_file():
            raise ExportError(f"required output file missing before COMPLETE: {path}")
    if skipped_rows:
        raise ExportError("COMPLETE gate refused: one or more expected train/validation windows were skipped")
    if not split_validation["validation_passed"]:
        raise ExportError("COMPLETE gate refused: train/validation footprint validation failed")
    if not image_stats["all_images_valid"]:
        raise ExportError("COMPLETE gate refused: image integrity validation failed")
    complete = {
        "status": "COMPLETE",
        "dataset_manifest": "dataset_manifest.json",
        "patch_index": "patch_index.csv",
        "skipped_windows": "skipped_windows.csv",
        "split_validation": "split_validation.json",
        "expected_train_validation_windows_all_exported": True,
        "decode_failures": 0,
        "labels_present": False,
        "all_yolo_annotations_still_pending_manual_completion": True,
        "total_png_bytes": image_stats["total_png_bytes"],
    }
    temporary = OUTPUT_PATH / ".COMPLETE.json.tmp"
    temporary.write_text(json.dumps(complete, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, OUTPUT_PATH / "COMPLETE.json")


def formal_run(prepared: dict[str, Any], resume: bool) -> None:
    assert_output_mode(resume)
    source_paths = [ECW_PATH, ERS_PATH, TRAIN_CENTRES_PATH, VAL_CENTRES_PATH, MAIN_POLYGON_PATH, SPLIT_MANIFEST_PATH, ERS_PATH]
    source_paths = list(dict.fromkeys(source_paths))
    source_snapshots_before = {str(path): snapshot(path) for path in source_paths}
    source_hashes_before = {str(path): sha256_file(path) for path in source_paths}
    rows, skipped_rows, export_stats = export_images(prepared, resume)
    image_stats = validate_image_files(rows, prepared["train"] + prepared["validation"])
    split_validation = prepared["split_validation"]
    # Recheck all fixed source stat fields after export.  The final hashes are
    # also compared so a source mutation during a long run cannot be hidden.
    source_snapshots_after = {str(path): snapshot(path) for path in source_paths}
    source_hashes_after = {str(path): sha256_file(path) for path in source_paths}
    if source_hashes_after != source_hashes_before:
        raise ExportError("one or more fixed inputs changed while exporting")
    if source_snapshots_after != source_snapshots_before:
        raise ExportError("one or more fixed inputs changed size or mtime while exporting")
    write_readme()
    manifest = build_manifest(
        prepared,
        rows,
        skipped_rows,
        export_stats,
        image_stats,
        source_hashes_before,
        source_snapshots_before,
        source_snapshots_after,
        split_validation,
    )
    write_csv_atomic(OUTPUT_PATH / "patch_index.csv", PATCH_FIELDS, rows)
    write_csv_atomic(OUTPUT_PATH / "skipped_windows.csv", SKIPPED_FIELDS, skipped_rows)
    (OUTPUT_PATH / "split_validation.json").write_text(json.dumps(split_validation, indent=2) + "\n", encoding="utf-8")
    (OUTPUT_PATH / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if skipped_rows or not split_validation["validation_passed"]:
        print(
            json.dumps(
                {
                    "status": "INCOMPLETE",
                    "output_path": str(OUTPUT_PATH),
                    "skipped_windows": len(skipped_rows),
                    "split_validation_passed": split_validation["validation_passed"],
                    "COMPLETE_json_created": False,
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return
    write_complete(manifest, split_validation, image_stats, skipped_rows)
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "output_path": str(OUTPUT_PATH),
                "train": sum(1 for row in rows if row["split"] == "train"),
                "validation": sum(1 for row in rows if row["split"] == "val"),
                "buffer_discarded": prepared["buffer_count"],
                "decode_failures": 0,
                "total_png_bytes": image_stats["total_png_bytes"],
                "minimum_gap_m": split_validation["minimum_gap_m"],
                "COMPLETE_json_created": True,
            },
            indent=2,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="count exact global-grid candidates without writing images")
    parser.add_argument("--resume", action="store_true", help="reuse valid 150x150 L PNGs in an interrupted output")
    args = parser.parse_args()
    try:
        prepared = prepare_candidates()
        if args.dry_run:
            print_dry_run(prepared)
            return 0
        formal_run(prepared, args.resume)
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
