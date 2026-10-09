#!/usr/bin/env python3
"""Project the user-drawn validation inset into Costa's EPSG:32630 frame.

This creates split geometries only; it does not create patches or labels.  The
inset is deliberately recorded as screenshot-derived and approximate so it can
be approved before a large patch export.
"""
from __future__ import annotations

import io
import json
import math
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Polygon, mapping, shape


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "training_split_design"
MAIN_GEOJSON = ROOT / "training_region_preview" / "approximate_training_region.geojson"
METADATA = ROOT / "data" / "metadata.json"
ECW = ROOT / "data" / "derived_sources" / "costa" / "C13007a_Costa_AllAreas_Comments_Rev1.ecw"
DECODER = ROOT / "local" / "bin" / "ecw_window"

PATCH_PX = 150
STRIDE_PX = 100
RESOLUTION_M = 0.1
PATCH_SIDE_M = PATCH_PX * RESOLUTION_M
PATCH_FOOTPRINT_RADIUS_M = math.hypot(PATCH_SIDE_M / 2, PATCH_SIDE_M / 2)
REQUIRED_GAP_M = PATCH_SIDE_M
TRAIN_CENTRE_EXCLUSION_M = REQUIRED_GAP_M + PATCH_FOOTPRINT_RADIUS_M

# Outer-frame and inset vertices read from the user's 2070 x 1678 screenshot.
# Each outer point corresponds to the existing main GeoJSON ring in the same
# order: lower-left, upper-left, upper-right, lower-right.  Small-frame order
# is upper-left, upper-right, lower-right, lower-left.
OUTER_SCREEN_PX = np.array([[678.0, 1103.0], [1226.0, 154.0], [2047.0, 628.0], [1500.0, 1578.0]])
VALIDATION_SCREEN_PX = np.array([[862.0, 843.0], [984.0, 914.0], [812.0, 1211.0], [690.0, 1140.0]])
OVERVIEW_SIZE = (980, 1496)


def write_geojson(path: Path, geometry, properties: dict) -> None:
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [{"type": "Feature", "properties": properties, "geometry": mapping(geometry)}]}, indent=2) + "\n")


def font(size: int):
    for candidate in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def decode_overview(width: int, height: int) -> Image.Image:
    raster = json.loads(METADATA.read_text())["datasets"]["costa"]["rasters"][0]
    proc = subprocess.run([str(DECODER), "window", str(ECW), "0", "0", str(raster["width"] - 1), str(raster["height"] - 1), str(width), str(height), "0"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    image = Image.open(io.BytesIO(proc.stdout)).convert("RGBA")
    if image.size != (width, height):
        raise RuntimeError(f"unexpected ECW overview size: {image.size}")
    return image


def rings_for_draw(geometry, world_to_screen):
    if geometry.geom_type == "Polygon":
        return [[world_to_screen(*point) for point in geometry.exterior.coords]]
    if geometry.geom_type == "MultiPolygon":
        return [[world_to_screen(*point) for point in part.exterior.coords] for part in geometry.geoms]
    raise TypeError(geometry.geom_type)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    document = json.loads(MAIN_GEOJSON.read_text())
    main_polygon = shape(document["features"][0]["geometry"])
    main_vertices = np.array(list(main_polygon.exterior.coords)[:-1])

    # Least-squares affine map: screenshot (x, y, 1) -> projected (E, N).
    design = np.c_[OUTER_SCREEN_PX, np.ones(4)]
    east_coeff = np.linalg.lstsq(design, main_vertices[:, 0], rcond=None)[0]
    north_coeff = np.linalg.lstsq(design, main_vertices[:, 1], rcond=None)[0]
    fitted = np.c_[design @ east_coeff, design @ north_coeff]
    residual_m = np.hypot(*(fitted - main_vertices).T)
    validation_world = np.c_[np.c_[VALIDATION_SCREEN_PX, np.ones(4)] @ east_coeff, np.c_[VALIDATION_SCREEN_PX, np.ones(4)] @ north_coeff]
    validation_screenshot_polygon = Polygon(validation_world)

    # The user clarified that validation must occupy the true lower-left corner:
    # its left and bottom sides must coincide with the main polygon.  Preserve
    # the screenshot inset's two side lengths, but rebuild it from the main
    # lower-left vertex and the main polygon's own edge vectors.
    lower_left, upper_left, upper_right, lower_right = main_vertices
    main_left_edge = upper_left - lower_left
    main_bottom_edge = lower_right - lower_left
    screen_ul, screen_ur, screen_lr, screen_ll = validation_world
    inset_left_edge = ((screen_ul - screen_ll) + (screen_ur - screen_lr)) / 2
    inset_bottom_edge = ((screen_lr - screen_ll) + (screen_ur - screen_ul)) / 2
    left_fraction = float(np.dot(inset_left_edge, main_left_edge) / np.dot(main_left_edge, main_left_edge))
    bottom_fraction = float(np.dot(inset_bottom_edge, main_bottom_edge) / np.dot(main_bottom_edge, main_bottom_edge))
    validation_vertices = np.array([
        lower_left,
        lower_left + left_fraction * main_left_edge,
        lower_left + left_fraction * main_left_edge + bottom_fraction * main_bottom_edge,
        lower_left + bottom_fraction * main_bottom_edge,
    ])
    validation_polygon = Polygon(validation_vertices)
    if validation_polygon.difference(main_polygon).area > 1e-6:
        raise RuntimeError("anchored validation polygon unexpectedly falls outside main polygon")
    main_patch_centres = main_polygon.buffer(-PATCH_FOOTPRINT_RADIUS_M, join_style=2)
    validation_patch_centres = validation_polygon.buffer(-PATCH_FOOTPRINT_RADIUS_M, join_style=2)
    train_patch_centres = main_patch_centres.difference(validation_polygon.buffer(TRAIN_CENTRE_EXCLUSION_M, join_style=2))
    train_region_with_15m_gap = main_polygon.difference(validation_polygon.buffer(REQUIRED_GAP_M, join_style=2))

    common = {"crs": "EPSG:32630", "source": "user screenshot projected through the existing main polygon"}
    write_geojson(OUT / "validation_polygon_approx.geojson", validation_polygon, {**common, "role": "validation area", "note": "screenshot-derived side lengths; anchored to the main polygon lower-left corner and its left/bottom edges"})
    write_geojson(OUT / "train_region_with_15m_gap.geojson", train_region_with_15m_gap, {**common, "role": "train region", "gap_m": REQUIRED_GAP_M})
    write_geojson(OUT / "validation_patch_centres_150px.geojson", validation_patch_centres, {**common, "role": "permitted validation patch centres", "patch_side_m": PATCH_SIDE_M, "centre_inset_m": PATCH_FOOTPRINT_RADIUS_M})
    write_geojson(OUT / "train_patch_centres_150px.geojson", train_patch_centres, {**common, "role": "permitted train patch centres", "patch_side_m": PATCH_SIDE_M, "minimum_image_gap_m": REQUIRED_GAP_M, "validation_exclusion_from_centre_m": TRAIN_CENTRE_EXCLUSION_M})

    raster = json.loads(METADATA.read_text())["datasets"]["costa"]["rasters"][0]
    left, right, top, bottom = raster["left"], raster["right"], raster["top"], raster["bottom"]
    def screen(e, n): return ((e - left) / (right - left) * OVERVIEW_SIZE[0], (top - n) / (top - bottom) * OVERVIEW_SIZE[1])
    image = decode_overview(*OVERVIEW_SIZE)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for ring in rings_for_draw(main_polygon, screen): draw.line(ring, fill=(0, 0, 0, 255), width=7, joint="curve")
    for ring in rings_for_draw(train_region_with_15m_gap, screen): draw.polygon(ring, fill=(30, 190, 95, 55)); draw.line(ring, fill=(35, 140, 75, 230), width=3, joint="curve")
    for ring in rings_for_draw(validation_polygon, screen): draw.polygon(ring, fill=(35, 155, 235, 125)); draw.line(ring, fill=(0, 85, 175, 255), width=4, joint="curve")
    label = "Validation (screenshot-projected)"
    vx, vy = screen(*validation_polygon.representative_point().coords[0]); draw.text((vx + 8, vy - 12), label, fill=(0, 45, 115, 255), stroke_width=2, stroke_fill=(255, 255, 255, 255), font=font(16))
    image = Image.alpha_composite(image, overlay).convert("RGB")
    image.save(OUT / "train_validation_split_overview.png")

    manifest = {
        "crs": "EPSG:32630", "patch": {"pixels": PATCH_PX, "resolution_m": RESOLUTION_M, "side_m": PATCH_SIDE_M, "stride_px": STRIDE_PX, "overlap_px": PATCH_PX - STRIDE_PX},
        "screenshot_projection": {"outer_control_points_px": OUTER_SCREEN_PX.tolist(), "validation_vertices_px": VALIDATION_SCREEN_PX.tolist(), "maximum_outer_control_fit_residual_m": float(max(residual_m)), "unanchored_validation_vertices_world_m": validation_world.tolist(), "validation_side_fractions_of_main_polygon": {"left_edge": left_fraction, "bottom_edge": bottom_fraction}, "anchored_validation_vertices_world_m": validation_vertices.tolist(), "unanchored_validation_outside_main_polygon_m2": float(validation_screenshot_polygon.difference(main_polygon).area)},
        "split_rule": {"validation_patch_centres": "inside validation_polygon inset by the 150px patch half-diagonal", "train_patch_centres": "inside main polygon inset by the patch half-diagonal and outside validation polygon buffer", "minimum_train_validation_image_gap_m": REQUIRED_GAP_M, "patch_footprint_radius_m": PATCH_FOOTPRINT_RADIUS_M, "train_centre_exclusion_m": TRAIN_CENTRE_EXCLUSION_M},
        "areas_m2": {"main": main_polygon.area, "validation_effective": validation_polygon.area, "validation_patch_centres": validation_patch_centres.area, "train_patch_centres": train_patch_centres.area},
    }
    (OUT / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
