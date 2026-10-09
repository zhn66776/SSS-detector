"""Build a self-contained, read-only-derived Colab bundle for SSS contacts."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BATCH_ROOT = PROJECT_ROOT / "sss_contact_inspector" / "yolo_manual_annotation_grid_150px_stride100_v1_cleaned" / "validation_annotation_batches_1000"
SOURCE_ROOT = Path(__file__).resolve().parent
BUILD_ROOT = PROJECT_ROOT / "sss_contact_l4_colab_bundle"
ZIP_PATH = PROJECT_ROOT / "sss_contact_l4_colab_bundle.zip"
GROUPS = ["group_01", "group_02", "group_03", "group_04"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert_anylabeling_json(json_path: Path) -> list[str]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    rows: list[str] = []
    for shape in payload.get("shapes", []):
        if shape.get("label") != "contact":
            raise ValueError(f"Unsupported label in {json_path}: {shape.get('label')!r}")
        if shape.get("shape_type") != "rectangle" or len(shape.get("points", [])) != 2:
            raise ValueError(f"Unsupported shape in {json_path}: {shape.get('shape_type')!r}")
        points = shape["points"]
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        if not (0 <= x_min <= x_max <= 150 and 0 <= y_min <= y_max <= 150):
            raise ValueError(f"Out-of-bounds box in {json_path}: {points!r}")
        values = (
            0,
            (x_min + x_max) / (2 * 150),
            (y_min + y_max) / (2 * 150),
            (x_max - x_min) / 150,
            (y_max - y_min) / 150,
        )
        rows.append(" ".join(str(value) if index == 0 else f"{value:.8f}" for index, value in enumerate(values)))
    return rows


def copy_source_tree() -> dict[str, dict[str, int]]:
    stats: dict[str, dict[str, int]] = {}
    for group in GROUPS:
        source_images = BATCH_ROOT / group / "images"
        output_images = BUILD_ROOT / "data" / "images" / group
        output_labels = BUILD_ROOT / "data" / "labels" / group
        output_json = BUILD_ROOT / "metadata" / "anylabeling_json" / group
        output_images.mkdir(parents=True, exist_ok=True)
        output_labels.mkdir(parents=True, exist_ok=True)
        output_json.mkdir(parents=True, exist_ok=True)

        image_count = 0
        json_count = 0
        bbox_count = 0
        empty_json_count = 0
        positive_image_count = 0
        negative_image_count = 0
        status_rows: list[dict[str, str | int]] = []

        for image_path in sorted(source_images.glob("*.png")):
            image_count += 1
            shutil.copy2(image_path, output_images / image_path.name)
            json_path = source_images / f"{image_path.stem}.json"
            label_path = output_labels / f"{image_path.stem}.txt"
            if json_path.exists():
                json_count += 1
                shutil.copy2(json_path, output_json / json_path.name)
                label_rows = convert_anylabeling_json(json_path)
                label_path.write_text("\n".join(label_rows) + ("\n" if label_rows else ""), encoding="utf-8")
                bbox_count += len(label_rows)
                if label_rows:
                    positive_image_count += 1
                    status = "positive_anylabeling_json"
                    source = "AnyLabeling JSON"
                else:
                    empty_json_count += 1
                    negative_image_count += 1
                    status = "negative_empty_json"
                    source = "empty AnyLabeling JSON"
            else:
                # The user supplied the current review contract that these
                # remaining group-01..04 images are confirmed negatives. Keep
                # this provenance explicit rather than silently calling them
                # unlabeled data.
                label_path.write_text("", encoding="utf-8")
                negative_image_count += 1
                status = "negative_user_confirmed"
                source = "user-provided negative summary; no per-image sidecar"
            status_rows.append(
                {
                    "group": group,
                    "patch_id": image_path.stem,
                    "status": status,
                    "annotation_source": source,
                    "bbox_count": len(label_rows) if json_path.exists() else 0,
                    "image": f"data/images/{group}/{image_path.name}",
                    "label": f"data/labels/{group}/{label_path.name}",
                }
            )

        with (BUILD_ROOT / "metadata" / f"{group}_annotation_status.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(status_rows[0].keys()))
            writer.writeheader()
            writer.writerows(status_rows)
        stats[group] = {
            "images": image_count,
            "json_files": json_count,
            "positive_images": positive_image_count,
            "negative_images": negative_image_count,
            "bbox_count": bbox_count,
            "empty_json_files": empty_json_count,
            "yolo_label_files": image_count,
        }
    return stats


def main() -> None:
    if BUILD_ROOT.exists() or ZIP_PATH.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {BUILD_ROOT} or {ZIP_PATH}")
    BUILD_ROOT.mkdir(parents=True)
    (BUILD_ROOT / "metadata").mkdir()
    shutil.copy2(SOURCE_ROOT / "contact_eval_helper.py", BUILD_ROOT / "contact_eval_helper.py")
    shutil.copy2(SOURCE_ROOT / "contact_l4_smoke_test.ipynb", BUILD_ROOT / "contact_l4_smoke_test.ipynb")
    group_stats = copy_source_tree()

    shutil.copy2(BATCH_ROOT / "validation_batches.csv", BUILD_ROOT / "metadata" / "validation_batches.csv")
    shutil.copy2(BATCH_ROOT / "batch_manifest.json", BUILD_ROOT / "metadata" / "batch_manifest_source.json")
    shutil.copy2(BATCH_ROOT / "README.md", BUILD_ROOT / "metadata" / "annotation_batches_README.md")
    for group in GROUPS:
        shutil.copy2(BATCH_ROOT / group / "patch_index.csv", BUILD_ROOT / "metadata" / f"{group}_patch_index.csv")

    dataset_yaml = {
        "path": ".",
        "train": "data/images/group_01",
        "val": "data/images/group_02",
        "test": "data/images/group_03",
        "nc": 1,
        "names": ["contact"],
        "channels": 1,
        "note": "This YAML is included for reference. The notebook uses the bundled ONNX checkpoint and evaluates group_02/group_03 directly.",
    }
    (BUILD_ROOT / "contact_one_class.yaml").write_text(json.dumps(dataset_yaml, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "bundle_version": "costa-head-sss-contact-l4-smoke-test-v1",
        "source_batch_root": str(BATCH_ROOT),
        "groups_included": GROUPS,
        "default_evaluation_groups": ["group_02", "group_03"],
        "default_reason": "highest bbox counts: group_02=106, group_03=96",
        "source_patch_contract": {
            "image_size_px": [150, 150],
            "native_channels": 1,
            "native_resolution_m_per_px": 0.1,
            "patch_side_m": 15.0,
            "class": "contact",
        },
        "label_conversion": {
            "source": "AnyLabeling rectangle JSON",
            "target": "YOLO normalized class x_center y_center width height",
            "class_id": 0,
            "missing_json_policy": "empty YOLO file, based on the user's confirmed-negative summary; no per-image negative sidecar was present in the source directory",
        },
        "checkpoint": {
            "repo": "https://huggingface.co/Dinoman1221/sonarvision-yolov8-esi-v6",
            "file": "yolo_esi_v6_fp32.onnx",
            "license": "MIT",
            "input": "3-channel 256x256 ONNX; notebook repeats the native grayscale band to RGB in memory",
            "class_filter": {"source_id": 0, "source_name": "unknown_debris", "output_name": "contact"},
            "warning": "Do not use the model card's leaked core_debris reference artifact; the notebook uses the v6 multi-class model and filters class 0 only.",
        },
        "group_stats": group_stats,
        "source_files": {
            "batch_manifest_sha256": sha256(BATCH_ROOT / "batch_manifest.json"),
            "validation_batches_sha256": sha256(BATCH_ROOT / "validation_batches.csv"),
        },
    }
    (BUILD_ROOT / "metadata" / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (BUILD_ROOT / "README.md").write_text(
        "# Costa Head SSS contact Colab bundle\n\n"
        "This derived bundle includes group_01..group_04 native 150x150 grayscale PNGs, converted one-class YOLO labels, the original AnyLabeling JSON where present, a Colab L4 notebook, and the ONNX inference/evaluation helper.\n\n"
        "The original project images and labels were not modified. The notebook defaults to group_02 and group_03 because they contain the most contact boxes.\n\n"
        "The selected checkpoint is SonarVision YOLOv8-ESI v6. Only its `unknown_debris` class is retained and emitted as `contact`; other model classes are discarded. This is a zero-shot SSS baseline, not a fine-tuned Costa Head model.\n\n"
        "The 3,722 images without AnyLabeling JSON are packaged with empty YOLO files according to the user's confirmed-negative summary. The manifest records that this negative status is not represented by a per-image source sidecar.\n",
        encoding="utf-8",
    )

    with ZipFile(ZIP_PATH, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(BUILD_ROOT.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(BUILD_ROOT.parent))
    print(json.dumps({"bundle": str(BUILD_ROOT), "zip": str(ZIP_PATH), "group_stats": group_stats}, indent=2))


if __name__ == "__main__":
    main()
