#!/usr/bin/env python3
"""Prepare read-only SSS source archives for the local inspector.

The original ZIP files are never modified. Moray GeoTIFFs are streamed to a
local derived cache because ZIP-deflate does not support efficient random
window access. Costa ECW is only copied when an ECW-enabled GDAL is present.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
import resource
import shutil
import subprocess
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pyproj import CRS, Transformer
from tifffile import TiffFile


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_ROOT = Path(
    "/Volumes/HP P900/MDE_SSS_Contacts_Costa_Head_Moray_Firth_2026-09-20"
)
COSTA_MOSAIC = Path("Costa_Head_CES-86/CES-86_Mosaic.zip")
COSTA_CONTACTS = Path("Costa_Head_CES-86/CES-86_SSS_target_listings.zip")
MORAY_MOSAICS = Path("Moray_Firth_CES-371/CES-371_Sonar_Mosaics.zip")
MORAY_CONTACTS = Path("Moray_Firth_CES-371/CES-371_Contact_Lists.zip")
VERSION = "sss-contact-inspector-prep-v1"


def log(message: str) -> None:
    print(message, flush=True)


def clean(value: Any) -> str:
    return " ".join(str(value or "").replace("\ufeff", "").strip().split())


def number(value: Any) -> float | None:
    text = clean(value).replace(" ", "").replace(",", "")
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def read_zip_text(archive: Path, member: str) -> str:
    with zipfile.ZipFile(archive) as zf:
        return zf.read(member).decode("utf-8-sig", errors="replace")


def member_record(archive: Path, member: str) -> dict[str, Any]:
    with zipfile.ZipFile(archive) as zf:
        info = zf.getinfo(member)
    return {
        "archive": str(archive),
        "member": member,
        "size": info.file_size,
        "compressed_size": info.compress_size,
        "crc32": f"{info.CRC:08x}",
    }


def extract_member(archive: Path, member: str, target: Path) -> None:
    with zipfile.ZipFile(archive) as zf:
        info = zf.getinfo(member)
        if target.exists() and target.stat().st_size == info.file_size:
            log(f"reuse {target.name} ({info.file_size:,} bytes)")
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(target.suffix + ".partial")
        with zf.open(member) as source, temp.open("wb") as destination:
            shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
        if temp.stat().st_size != info.file_size:
            raise IOError(f"short extraction for {member}")
        os.replace(temp, target)
        log(f"extracted {target.name} ({info.file_size:,} bytes)")


def detect_ecw_decoder() -> dict[str, Any]:
    local_decoder = ROOT / "local" / "bin" / "ecw_window"
    if local_decoder.is_file() and os.access(local_decoder, os.X_OK):
        return {
            "available": True,
            "backend": "ecw_window",
            "decoder": str(local_decoder),
            "reason": "project-local arm64 ECW window decoder detected",
        }
    info = shutil.which("gdalinfo")
    translate = shutil.which("gdal_translate")
    result: dict[str, Any] = {
        "available": False,
        "gdalinfo": info,
        "gdal_translate": translate,
        "backend": "gdal",
        "reason": "neither the project-local ECW decoder nor gdalinfo/gdal_translate was found",
    }
    if not info or not translate:
        return result
    try:
        proc = subprocess.run(
            [info, "--formats"], capture_output=True, text=True, timeout=30, check=True
        )
    except Exception as exc:
        result["reason"] = f"GDAL format probe failed: {exc}"
        return result
    has_ecw = bool(re.search(r"^\s*ECW\s", proc.stdout, re.MULTILINE))
    result["available"] = has_ecw
    result["reason"] = (
        "ECW-enabled GDAL detected" if has_ecw else "installed GDAL has no ECW driver"
    )
    return result


def parse_ers(text: str) -> dict[str, Any]:
    def grab(name: str) -> str:
        match = re.search(rf"^\s*{re.escape(name)}\s*=\s*\"?([^\"\r\n]+)", text, re.MULTILINE)
        if not match:
            raise ValueError(f"ERS field missing: {name}")
        return clean(match.group(1))

    width = int(grab("NrOfCellsPerLine"))
    height = int(grab("NrOfLines"))
    resolution_x = float(grab("Xdimension"))
    resolution_y = float(grab("Ydimension"))
    left = float(grab("Eastings"))
    top = float(grab("Northings"))
    projection = grab("Projection")
    datum = grab("Datum")
    if projection != "NUTM30" or datum != "WGS84":
        raise ValueError(f"Unresolved Costa ERS CRS: {datum} / {projection}")
    crs = CRS.from_epsg(32630)
    return {
        "width": width,
        "height": height,
        "resolution_x": resolution_x,
        "resolution_y": resolution_y,
        "left": left,
        "top": top,
        "right": left + width * resolution_x,
        "bottom": top - height * resolution_y,
        "epsg": crs.to_epsg(),
        "crs": crs.name,
        "crs_evidence": "ERS Datum=WGS84 and Projection=NUTM30",
        "registration_note": (
            "ERS RegistrationCoord was treated as the upper-left raster registration; "
            "the ERS does not explicitly state pixel-centre versus pixel-edge semantics."
        ),
    }


def parse_dimensions(text: str) -> tuple[float | None, float | None, float | None]:
    values = re.findall(r"[-+]?\d+(?:\.\d+)?", clean(text))
    parsed = [float(value) for value in values[:3]]
    return tuple((parsed + [None, None, None])[:3])  # type: ignore[return-value]


def iter_csv(archive: Path, member: str):
    with zipfile.ZipFile(archive) as zf:
        with zf.open(member) as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8-sig", errors="replace", newline="")
            yield from csv.DictReader(text)


def parse_costa_contacts(archive: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    contacts: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive) as zf:
        members = sorted(name for name in zf.namelist() if name.lower().endswith(".csv"))
    for member in members:
        area_match = re.search(r"Costa_(Afl|CR[123])_", member, re.IGNORECASE)
        if not area_match:
            continue
        area = area_match.group(1).lower()
        linear = "Linear" in member
        for source_row, row in enumerate(iter_csv(archive, member), start=2):
            if linear:
                e1 = number(row.get("Start Easting (m)"))
                n1 = number(row.get("Start Northing (m)"))
                e2 = number(row.get("End Easting (m)"))
                n2 = number(row.get("End Northing (m)"))
                if None in (e1, n1, e2, n2):
                    rejected.append({"member": member, "row": source_row, "reason": "invalid coordinates"})
                    continue
                contact = {
                    "uid": f"costa:{area}:{clean(row.get('ID'))}:{source_row}",
                    "id": clean(row.get("ID")) or f"row-{source_row}",
                    "dataset": "costa",
                    "area": area,
                    "geometry": "line",
                    "e": (e1 + e2) / 2,
                    "n": (n1 + n2) / 2,
                    "description": "Linear contact",
                    "length_m": number(row.get("Length (m)")),
                    "start_e": e1,
                    "start_n": n1,
                    "end_e": e2,
                    "end_n": n2,
                    "source_file": member,
                    "source_row": source_row,
                    "raw": {clean(k): clean(v) for k, v in row.items()},
                }
            else:
                e = number(row.get("Easting (m)"))
                n = number(row.get("Northing (m)"))
                target_id = clean(row.get("Target ID"))
                if e is None or n is None:
                    rejected.append({"member": member, "row": source_row, "reason": "invalid coordinates"})
                    continue
                length, width, height = parse_dimensions(clean(row.get("Dimensions (m) (LxWxH)")))
                contact = {
                    "uid": f"costa:{area}:{target_id}:{source_row}",
                    "id": target_id or f"row-{source_row}",
                    "dataset": "costa",
                    "area": area,
                    "geometry": "point",
                    "e": e,
                    "n": n,
                    "description": clean(row.get("Description")) or "Unspecified",
                    "length_m": length,
                    "width_m": width,
                    "height_m": height,
                    "source_file": member,
                    "source_row": source_row,
                    "raw": {clean(k): clean(v) for k, v in row.items()},
                }
            contacts.append(contact)
    return contacts, {"rejected": rejected, "source_members": members}


def parse_moray_contacts(archive: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    contacts: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive) as zf:
        members = sorted(name for name in zf.namelist() if name.lower().endswith(".csv"))
    for member in members:
        area = "owf" if "OWF Site" in member else "ecr"
        prefix = area.upper()
        accepted_index = 0
        for source_row, row in enumerate(iter_csv(archive, member), start=2):
            e = number(row.get("Easting"))
            n = number(row.get("Northing"))
            if e is None or n is None:
                rejected.append({"member": member, "row": source_row, "reason": "invalid coordinates"})
                continue
            accepted_index += 1
            description = clean(row.get("Description") or row.get("Description ")) or "Unspecified"
            contact = {
                "uid": f"moray:{area}:{accepted_index:05d}",
                "id": f"{prefix}-{accepted_index:05d}",
                "dataset": "moray",
                "area": area,
                "geometry": "point",
                "e": e,
                "n": n,
                "description": description,
                "length_m": number(row.get("Length [m]")),
                "width_m": number(row.get("Width [m]")),
                "height_m": number(row.get("Height [m]")),
                "magnitude_nt_m": number(row.get("Magnitude [nT/m]")),
                "source_file": member,
                "source_row": source_row,
                "raw": {clean(k): clean(v) for k, v in row.items()},
            }
            contacts.append(contact)
    return contacts, {"rejected": rejected, "source_members": members}


def tiff_metadata(path: Path, tfw_path: Path) -> dict[str, Any]:
    values = [float(line.strip()) for line in tfw_path.read_text().splitlines() if line.strip()]
    if len(values) != 6:
        raise ValueError(f"world file must contain six values: {tfw_path}")
    a, d, b, e, centre_x, centre_y = values
    if abs(b) > 1e-12 or abs(d) > 1e-12 or a <= 0 or e >= 0:
        raise ValueError(f"rotated or invalid world file: {tfw_path}")
    with TiffFile(path) as tif:
        page = tif.pages[0]
        width = int(page.imagewidth)
        height = int(page.imagelength)
        scale = tuple(float(v) for v in page.tags["ModelPixelScaleTag"].value)
        tie = tuple(float(v) for v in page.tags["ModelTiepointTag"].value)
        geokeys = tuple(int(v) for v in page.tags["GeoKeyDirectoryTag"].value)
        projected_epsg = None
        for offset in range(4, len(geokeys), 4):
            key, location, count, value = geokeys[offset : offset + 4]
            if key == 3072 and location == 0 and count == 1:
                projected_epsg = value
        if projected_epsg != 32630:
            raise ValueError(f"unexpected GeoTIFF projected CRS in {path}: {projected_epsg}")
        if not (math.isclose(scale[0], a) and math.isclose(scale[1], abs(e))):
            raise ValueError(f"TFW/GeoTIFF resolution mismatch: {path}")
        left = tie[3]
        top = tie[4]
        if not (math.isclose(centre_x, left + a / 2) and math.isclose(centre_y, top + e / 2)):
            raise ValueError(f"TFW/GeoTIFF origin mismatch: {path}")
        return {
            "id": path.stem,
            "path": str(path.relative_to(ROOT)),
            "tfw": str(tfw_path.relative_to(ROOT)),
            "width": width,
            "height": height,
            "resolution_x": a,
            "resolution_y": abs(e),
            "left": left,
            "top": top,
            "right": left + width * a,
            "bottom": top - height * abs(e),
            "epsg": projected_epsg,
            "compression": page.compression.name,
            "photometric": page.photometric.name,
            "rows_per_strip": int(page.rowsperstrip),
            "dtype": str(page.dtype),
        }


def contains(bounds: dict[str, Any], contact: dict[str, Any]) -> bool:
    return (
        bounds["left"] <= contact["e"] <= bounds["right"]
        and bounds["bottom"] <= contact["n"] <= bounds["top"]
    )


def dataset_summary(contacts: list[dict[str, Any]], areas: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for area in areas:
        subset = [c for c in contacts if c["area"] == area]
        result[area] = {
            "total": len(subset),
            "descriptions": dict(sorted(Counter(c["description"] for c in subset).items())),
            "geometry": dict(sorted(Counter(c["geometry"] for c in subset).items())),
        }
    return result


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--copy-costa-ecw",
        action="store_true",
        help="copy the 1 GB Costa ECW only when an ECW-enabled GDAL is available",
    )
    args = parser.parse_args()
    started = time.perf_counter()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_root = args.source_root.resolve()
    archives = {
        "costa_mosaic": source_root / COSTA_MOSAIC,
        "costa_contacts": source_root / COSTA_CONTACTS,
        "moray_mosaics": source_root / MORAY_MOSAICS,
        "moray_contacts": source_root / MORAY_CONTACTS,
    }
    missing = [str(path) for path in archives.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing source archives:\n" + "\n".join(missing))

    derived = output / "derived_sources"
    moray_dir = derived / "moray"
    costa_dir = derived / "costa"
    moray_dir.mkdir(parents=True, exist_ok=True)
    costa_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archives["moray_mosaics"]) as zf:
        moray_members = sorted(
            name for name in zf.namelist() if name.lower().endswith((".tif", ".tfw"))
        )
    for member in moray_members:
        extract_member(archives["moray_mosaics"], member, moray_dir / Path(member).name)

    ers_member = "C13007a_Costa_AllAreas_Comments_Rev1.ers"
    ers_text = read_zip_text(archives["costa_mosaic"], ers_member)
    (costa_dir / ers_member).write_text(ers_text)
    costa_raster = parse_ers(ers_text)
    costa_raster.update(
        {
            "id": "costa-all-areas",
            "source_member": "C13007a_Costa_AllAreas_Comments_Rev1.ecw",
            "format": "ECW",
            "bands": 2,
            "cell_type": "Unsigned8BitInteger",
        }
    )

    ecw = detect_ecw_decoder()
    if args.copy_costa_ecw:
        if not ecw["available"]:
            raise RuntimeError(f"Cannot copy/use Costa ECW: {ecw['reason']}")
        extract_member(
            archives["costa_mosaic"],
            costa_raster["source_member"],
            costa_dir / costa_raster["source_member"],
        )
        costa_raster["path"] = str((costa_dir / costa_raster["source_member"]).relative_to(ROOT))
        if ecw.get("backend") == "ecw_window":
            probe = subprocess.run(
                [ecw["decoder"], "info", str(costa_dir / costa_raster["source_member"])],
                capture_output=True,
                text=True,
                timeout=60,
                check=True,
            )
            decoder_info = json.loads(probe.stdout)
            if decoder_info["width"] != costa_raster["width"] or decoder_info["height"] != costa_raster["height"]:
                raise RuntimeError(f"ECW decoder/ERS dimension mismatch: {decoder_info}")
            costa_raster["decoder_info"] = decoder_info
    costa_raster["runtime_available"] = bool(ecw["available"] and costa_raster.get("path"))
    costa_raster["runtime_status"] = ecw

    moray_rasters: list[dict[str, Any]] = []
    for tif_path in sorted(moray_dir.glob("*.tif")):
        moray_rasters.append(tiff_metadata(tif_path, tif_path.with_suffix(".tfw")))
    if len(moray_rasters) != 6:
        raise ValueError(f"expected six Moray GeoTIFFs, found {len(moray_rasters)}")

    costa_contacts, costa_audit = parse_costa_contacts(archives["costa_contacts"])
    moray_contacts, moray_audit = parse_moray_contacts(archives["moray_contacts"])
    all_contacts = costa_contacts + moray_contacts

    moray_bounds = {
        "left": min(r["left"] for r in moray_rasters),
        "right": max(r["right"] for r in moray_rasters),
        "bottom": min(r["bottom"] for r in moray_rasters),
        "top": max(r["top"] for r in moray_rasters),
    }
    coverage = {
        "costa": {
            area: {
                "inside": sum(contains(costa_raster, c) for c in costa_contacts if c["area"] == area),
                "outside": sum(not contains(costa_raster, c) for c in costa_contacts if c["area"] == area),
            }
            for area in ("afl", "cr1", "cr2", "cr3")
        },
        "moray": {
            area: {
                "inside": sum(contains(moray_bounds, c) for c in moray_contacts if c["area"] == area),
                "outside": sum(not contains(moray_bounds, c) for c in moray_contacts if c["area"] == area),
            }
            for area in ("owf", "ecr")
        },
    }
    for contact in moray_contacts:
        contact["raster_id"] = next(
            (raster["id"] for raster in moray_rasters if contains(raster, contact)), None
        )
    for contact in costa_contacts:
        contact["raster_id"] = "costa-all-areas" if contains(costa_raster, contact) else None

    contacts_path = output / "contacts.json"
    contacts_path.write_text(json.dumps(all_contacts, separators=(",", ":")) + "\n")
    transform = Transformer.from_crs(32630, 4326, always_xy=True)
    costa_ll = {
        "southwest": transform.transform(costa_raster["left"], costa_raster["bottom"]),
        "northeast": transform.transform(costa_raster["right"], costa_raster["top"]),
    }
    metadata = {
        "version": VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_root": str(source_root),
        "crs": {"epsg": 32630, "name": "WGS 84 / UTM zone 30N", "units": "metres"},
        "datasets": {
            "costa": {
                "label": "Costa Head CES-86",
                "default_area": "afl",
                "areas": [
                    {"id": "afl", "label": "AFL main area"},
                    {"id": "cr1", "label": "CR1"},
                    {"id": "cr2", "label": "CR2"},
                    {"id": "cr3", "label": "CR3"},
                ],
                "native_resolution_m": 0.1,
                "bounds": {k: costa_raster[k] for k in ("left", "right", "bottom", "top")},
                "rasters": [costa_raster],
                "summary": dataset_summary(costa_contacts, ["afl", "cr1", "cr2", "cr3"]),
                "geographic_extent_from_raster": costa_ll,
                "status": "ready" if costa_raster["runtime_available"] else "ecw_decoder_missing",
                "status_message": (
                    "Costa ECW is ready"
                    if costa_raster["runtime_available"]
                    else "Costa source metadata and contacts are available, but native imagery requires an ECW-enabled GDAL/ERDAS ECW SDK."
                ),
            },
            "moray": {
                "label": "Moray Firth CES-371",
                "default_area": "owf",
                "areas": [
                    {"id": "owf", "label": "OWF Site"},
                    {"id": "ecr", "label": "ECR"},
                ],
                "native_resolution_m": 1.0,
                "bounds": moray_bounds,
                "rasters": moray_rasters,
                "summary": dataset_summary(moray_contacts, ["owf", "ecr"]),
                "status": "ready",
                "status_message": "Six 1 m GeoTIFF strips are available for native window reads.",
            },
        },
        "coverage": coverage,
        "contact_audit": {"costa": costa_audit, "moray": moray_audit},
        "sources": {
            key: {
                "path": str(path),
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for key, path in archives.items()
        },
        "source_members": {
            "costa_ecw": member_record(archives["costa_mosaic"], costa_raster["source_member"]),
            "costa_ers": member_record(archives["costa_mosaic"], ers_member),
            "moray_tiffs": [member_record(archives["moray_mosaics"], m) for m in moray_members if m.endswith(".tif")],
        },
        "rules": {
            "default_excluded_descriptions": ["Magnetic Anomaly"],
            "moray_generated_ids": "OWF/ECR CSVs contain no ID column; stable area-row IDs are generated after valid-coordinate parsing.",
            "costa_invalid_rows": "Repeated CSV header rows are rejected as invalid coordinates and listed in contact_audit.",
        },
    }
    elapsed = time.perf_counter() - started
    metadata["preparation"] = {
        "elapsed_seconds": round(elapsed, 3),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024), 2),
        "output_bytes": directory_size(output),
        "worker_limit": 1,
        "method": "8 MiB streaming extraction; TIFF metadata plus ECW header probe; no full-raster arrays",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    log(f"prepared {len(all_contacts):,} contacts in {elapsed:.2f}s")
    log(f"metadata: {output / 'metadata.json'}")


if __name__ == "__main__":
    main()
