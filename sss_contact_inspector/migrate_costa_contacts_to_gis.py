#!/usr/bin/env python3
"""Correct only divergent Costa point contacts from the official GIS SSS Listing.

The nested shapefile is read directly from the original ZIPs.  Existing records
whose GIS values already agree are left byte-for-field untouched.  Costa line
contacts are outside this migration because the GIS SSS Listing contains points
only.  Run without --apply for a read-only audit.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import shutil
import tempfile
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import shapefile


ROOT = Path(__file__).resolve().parent
CONTACTS = ROOT / "data" / "contacts.json"
GIS = ROOT / "source_packages" / "13_86_GIS_Shapefiles.zip"
NESTED = "SSS Listing.zip"
BASE = "SSS Listing/C13007a_CostaHead_SSSListing"
SOURCE_FILE = f"{GIS.name}!/{NESTED}!/{BASE}.dbf"
SOURCE_GEOMETRY_FILE = f"{GIS.name}!/{NESTED}!/{BASE}.shp"


def normal_area(value: object) -> str:
    area = str(value).strip().lower()
    return "afl" if area == "loa" else area


def same_number(a: object, b: object) -> bool:
    return isinstance(a, (int, float)) and isinstance(b, (int, float)) and math.isclose(float(a), float(b), abs_tol=1e-12)


def source_points() -> dict[tuple[str, str], tuple[int, dict]]:
    """Return GIS records keyed by (normalised area, target ID), validating geometry."""
    with zipfile.ZipFile(GIS) as outer:
        nested = outer.read(NESTED)
    with zipfile.ZipFile(io.BytesIO(nested)) as inner:
        reader = shapefile.Reader(
            shp=io.BytesIO(inner.read(f"{BASE}.shp")),
            shx=io.BytesIO(inner.read(f"{BASE}.shx")),
            dbf=io.BytesIO(inner.read(f"{BASE}.dbf")),
        )
    fields = [field[0] for field in reader.fields[1:]]
    values: dict[tuple[str, str], tuple[int, dict]] = {}
    for record_no, shape_record in enumerate(reader.iterShapeRecords(), start=1):
        item = dict(zip(fields, shape_record.record))
        if len(shape_record.shape.points) != 1:
            raise ValueError(f"GIS record {record_no} is not a point")
        e, n = map(float, shape_record.shape.points[0])
        if not (math.isclose(e, float(item["Easting"]), abs_tol=0.01) and math.isclose(n, float(item["Northing"]), abs_tol=0.01)):
            raise ValueError(f"GIS geometry/DBF coordinate mismatch at record {record_no}")
        key = (normal_area(item["Area"]), str(item["Target_ID"]).strip())
        if key in values:
            raise ValueError(f"duplicate GIS target key: {key}")
        values[key] = (record_no, item)
    return values


def gis_fields(record_no: int, item: dict) -> dict:
    return {
        "e": float(item["Easting"]),
        "n": float(item["Northing"]),
        "description": str(item["Tag_Name"]).strip() or "Unspecified",
        "length_m": float(item["Length__m_"]) if item["Length__m_"] is not None else None,
        "width_m": float(item["Width__m_"]) if item["Width__m_"] is not None else None,
        "height_m": float(item["Height__m_"]) if item["Height__m_"] is not None else None,
        "source_file": SOURCE_FILE,
        "source_row": record_no,
        "source_geometry_file": SOURCE_GEOMETRY_FILE,
        "source_provenance": "official GIS SSS Listing shapefile/DBF",
        "raw": {
            "Target_ID": str(item["Target_ID"]).strip(), "Area": str(item["Area"]).strip(),
            "Tag_Name": str(item["Tag_Name"]).strip(), "Easting": float(item["Easting"]),
            "Northing": float(item["Northing"]), "Length__m_": item["Length__m_"],
            "Width__m_": item["Width__m_"], "Height__m_": item["Height__m_"],
        },
    }


def coordinate_diverges(current: dict, expected: dict) -> bool:
    """Treat <= 8 cm as the same point: CSV stores coordinates to 0.1 m."""
    return math.hypot(float(current["e"]) - expected["e"], float(current["n"]) - expected["n"]) > 0.08


def migrate(apply: bool) -> dict:
    gis = source_points()
    contacts = json.loads(CONTACTS.read_text())
    costa_points = [item for item in contacts if item.get("dataset") == "costa" and item.get("geometry") == "point"]
    existing = {(item["area"], item["id"]): item for item in costa_points}
    if len(existing) != len(costa_points):
        raise ValueError("Costa point (area, ID) keys are not unique")
    unknown = sorted(set(existing) - set(gis))
    if unknown:
        raise ValueError(f"existing Costa points missing from GIS source: {unknown[:10]}")

    changed: list[str] = []
    by_field: Counter[str] = Counter()
    for key, item in existing.items():
        record_no, source = gis[key]
        expected = gis_fields(record_no, source)
        # Only a spatially divergent point is an error being corrected here.
        # For matching points, small CSV-versus-DBF rounding and any secondary
        # attribute disagreement are deliberately left untouched.
        if not coordinate_diverges(item, expected):
            continue
        for field in ("e", "n", "description", "length_m", "width_m", "height_m"):
            prior, replacement = item.get(field), expected[field]
            same = same_number(prior, replacement) if isinstance(replacement, (int, float)) or isinstance(prior, (int, float)) else prior == replacement
            if not same:
                by_field[field] += 1
        item.update(expected)
        changed.append(item["uid"])

    additions = []
    for key in sorted(set(gis) - set(existing)):
        record_no, source = gis[key]
        area, target_id = key
        additions.append({
            "uid": f"costa:{area}:{target_id}:gis-{record_no}", "id": target_id,
            "dataset": "costa", "area": area, "geometry": "point", "raster_id": "costa-all-areas",
            **gis_fields(record_no, source),
        })
    contacts.extend(additions)
    summary = {
        "source": SOURCE_FILE, "gis_point_records": len(gis), "existing_costa_point_records": len(costa_points),
        "changed_existing_records": len(changed), "changed_fields": dict(by_field),
        "added_gis_only_records": len(additions), "added_ids": [item["id"] for item in additions],
        "retained_costa_line_records": sum(item.get("dataset") == "costa" and item.get("geometry") == "line" for item in contacts),
        "apply": apply,
    }
    if apply:
        backup = CONTACTS.with_name(f"contacts.pre_gis_shapefile_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json")
        shutil.copy2(CONTACTS, backup)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=CONTACTS.parent, prefix=".contacts.", delete=False) as handle:
            json.dump(contacts, handle, separators=(",", ":"))
            handle.write("\n")
            temporary = Path(handle.name)
        os.replace(temporary, CONTACTS)
        summary["backup"] = str(backup)
        summary["output"] = str(CONTACTS)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="write corrected contacts.json after creating a backup")
    args = parser.parse_args()
    print(json.dumps(migrate(args.apply), indent=2))
