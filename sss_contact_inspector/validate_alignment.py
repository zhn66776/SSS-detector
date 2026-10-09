#!/usr/bin/env python3
"""Create inspectable native-pixel crops and a coordinate-registration report."""

from __future__ import annotations

import json
import io
import statistics
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from serve import InspectorData


ROOT = Path(__file__).resolve().parent
SAMPLES = [
    "moray:owf:03795",
    "moray:owf:03054",
    "moray:owf:00943",
    "moray:ecr:00004",
    "moray:ecr:00830",
    "moray:ecr:02418",
]
VISUAL_NOTES = {
    "OWF-03795": "Marker is on a locally contrasting textured response; plausible but not unequivocal at 1 m.",
    "OWF-03054": "Marker is adjacent to a compact bright/dark contrast response.",
    "OWF-00943": "Marker intersects a compact high-contrast response within strong along-track texture.",
    "ECR-00004": "Small low-contrast dark response lies within a few native pixels of the marker.",
    "ECR-00830": "Clear short bright/dark horizontal response crosses the marker.",
    "ECR-02418": "Broad low-contrast dark response surrounds the marker; acoustic expression is weak.",
}
COSTA_COORDINATE_SAMPLES = [
    "costa:afl:S961:2",
    "costa:afl:S1764:807",
    "costa:afl:L39:2",
]
COSTA_IMAGE_SAMPLES = [
    "costa:afl:S961:2",
    "costa:afl:S1571:612",
    "costa:afl:S2383:1425",
    "costa:afl:S2962:2004",
]
COSTA_VISUAL_NOTES = {
    "S961": "Clear compact dark target and adjacent bright response at the marker.",
    "S1571": "Strong elongated dark response and broad bright return intersect the debris marker.",
    "S2383": "High-contrast local response is present at the marker in a dense target field.",
    "S2962": "The decoded grey band is near-uniform white here; this contact is inside the rectangular ERS extent but outside usable grey-image coverage in this decoder/source combination.",
}


def contains(raster: dict, contact: dict) -> bool:
    return (
        raster["left"] <= contact["e"] <= raster["right"]
        and raster["bottom"] <= contact["n"] <= raster["top"]
    )


def main() -> None:
    app = InspectorData(ROOT)
    output = ROOT / "validation"
    output.mkdir(exist_ok=True)
    records = []
    latencies = []
    for uid in SAMPLES:
        contact = app.by_uid[uid]
        started = time.perf_counter()
        image = app.render(
            "moray",
            (contact["e"] - 25, contact["n"] - 25, contact["e"] + 25, contact["n"] + 25),
            50,
            50,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        latencies.append(latency_ms)
        native_path = output / f"{contact['id']}_native_50m.png"
        image.save(native_path)
        enlarged = image.resize((500, 500), Image.Resampling.NEAREST)
        draw = ImageDraw.Draw(enlarged)
        draw.line((250, 214, 250, 238), fill=(255, 28, 28), width=3)
        draw.line((250, 262, 250, 286), fill=(255, 28, 28), width=3)
        draw.line((214, 250, 238, 250), fill=(255, 28, 28), width=3)
        draw.line((262, 250, 286, 250), fill=(255, 28, 28), width=3)
        annotated_path = output / f"{contact['id']}_annotated.png"
        enlarged.save(annotated_path)
        gray = np.asarray(image.convert("L"), dtype=np.float32)
        yy, xx = np.ogrid[:50, :50]
        radius = np.sqrt((xx - 24.5) ** 2 + (yy - 24.5) ** 2)
        centre = gray[radius <= 3]
        annulus = gray[(radius >= 7) & (radius <= 18)]
        raster = next(r for r in app.rasters["moray"] if r["id"] == contact["raster_id"])
        records.append(
            {
                "uid": uid,
                "id": contact["id"],
                "area": contact["area"],
                "description": contact["description"],
                "e": contact["e"],
                "n": contact["n"],
                "raster_id": contact["raster_id"],
                "inside_raster": contains(raster, contact),
                "native_crop": str(native_path.relative_to(ROOT)),
                "annotated_crop": str(annotated_path.relative_to(ROOT)),
                "render_ms": round(latency_ms, 2),
                "centre_mean": round(float(centre.mean()), 2),
                "local_annulus_mean": round(float(annulus.mean()), 2),
                "centre_minus_local": round(float(centre.mean() - annulus.mean()), 2),
                "local_stddev": round(float(annulus.std()), 2),
                "visual_review": VISUAL_NOTES[contact["id"]],
            }
        )

    costa_raster = app.rasters["costa"][0]
    costa_image_samples = []
    if app.metadata["datasets"]["costa"]["status"] == "ready":
        for uid in COSTA_IMAGE_SAMPLES:
            contact = app.by_uid[uid]
            started = time.perf_counter()
            body, headers = app.crop("costa", contact["e"], contact["n"], 50.0)
            latency_ms = (time.perf_counter() - started) * 1000
            image = Image.open(io.BytesIO(body)).convert("RGB")
            native_path = output / f"{contact['id']}_native_50m.png"
            image.save(native_path)
            enlarged = image.resize((600, 600), Image.Resampling.NEAREST)
            draw = ImageDraw.Draw(enlarged)
            for line in ((300, 258, 300, 285), (300, 315, 300, 342), (258, 300, 285, 300), (315, 300, 342, 300)):
                draw.line(line, fill=(255, 28, 28), width=3)
            annotated_path = output / f"{contact['id']}_annotated.png"
            enlarged.save(annotated_path)
            gray = np.asarray(image.convert("L"), dtype=np.float32)
            costa_image_samples.append(
                {
                    "uid": uid,
                    "id": contact["id"],
                    "description": contact["description"],
                    "e": contact["e"],
                    "n": contact["n"],
                    "inside_ers_extent": contains(costa_raster, contact),
                    "native_resolution_m": float(headers["X-Native-Resolution-M"]),
                    "source_pixels": list(image.size),
                    "native_crop": str(native_path.relative_to(ROOT)),
                    "annotated_crop": str(annotated_path.relative_to(ROOT)),
                    "render_ms": round(latency_ms, 2),
                    "pixel_min": int(gray.min()),
                    "pixel_max": int(gray.max()),
                    "pixel_stddev": round(float(gray.std()), 2),
                    "visual_review": COSTA_VISUAL_NOTES[contact["id"]],
                }
            )
    costa = []
    for uid in COSTA_COORDINATE_SAMPLES:
        contact = app.by_uid.get(uid)
        if contact is None:
            candidates = [c for c in app.by_dataset["costa"] if c["area"] == "afl"]
            contact = candidates[len(costa) * max(1, len(candidates) // 3)]
        costa.append(
            {
                "uid": contact["uid"],
                "id": contact["id"],
                "e": contact["e"],
                "n": contact["n"],
                "inside_ers_extent": contains(costa_raster, contact),
                "image_review": "blocked_missing_ecw_decoder",
            }
        )
    report = {
        "crs": app.metadata["crs"],
        "method": (
            "Moray crops are direct 50x50 native 1 m source windows. The red crosshair is added only to a separate "
            "10x nearest-neighbour review copy. Native files are unannotated. Costa samples are coordinate-only because "
            "no ECW decoder is installed."
        ),
        "moray_samples": records,
        "costa_image_samples": costa_image_samples,
        "costa_coordinate_samples": costa,
        "moray_render_ms": {
            "median": round(statistics.median(latencies), 2),
            "maximum": round(max(latencies), 2),
        },
    }
    (output / "alignment_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
