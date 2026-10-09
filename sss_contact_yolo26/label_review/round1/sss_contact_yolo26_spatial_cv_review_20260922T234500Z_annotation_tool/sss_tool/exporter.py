"""Versioned export of working copy A. Never overwrites an existing version,
the source ZIP or backup B.

exports/A_final_labels_vNNN/validation/ground_truth_labels/*.txt
exports/A_final_labels_vNNN.zip
exports/A_final_labels_vNNN_manifest.json
exports/A_final_labels_vNNN_changes.csv
exports/A_final_labels_vNNN_review_status.csv
"""
import collections
import csv
import io
import json
import os
import re
import zipfile

from . import TOOL_VERSION
from .common import (A_LABEL_DIR, CLASS_NAMES, EXPORT_DIR, IMAGE_H, IMAGE_W,
                     LABEL_SUBDIR, TOOL_DIR, file_fingerprint, sha256_bytes,
                     sha256_file, utc_now)
from .labels import parse_label_text, validate_boxes
from .store import PATCH_STATUSES, PRED_STATUSES

NAME_RE = re.compile(r"^A_final_labels_v(\d{3,})")


def next_version():
    os.makedirs(EXPORT_DIR, exist_ok=True)
    used = [int(m.group(1)) for n in os.listdir(EXPORT_DIR) for m in [NAME_RE.match(n)] if m]
    return max(used, default=0) + 1


def _box_lines(data):
    return [l for l in data.decode("ascii").splitlines() if l.strip()]


def export_a(store, index, zip_path, b_dir, b_manifest_path, note="", purpose="final_labels"):
    with store.lock:  # consistent snapshot of A + review state
        a_snapshot = dict(store.a_bytes)
        review = json.loads(json.dumps(store.review))
        edit_counts = store.edit_counts()
        history = store.history_summary(limit=0)
        n_ops = store.history["seq"]
    # verify the in-memory snapshot equals what is on disk in A
    disk_mismatch = []
    for pid, data in a_snapshot.items():
        with open(os.path.join(A_LABEL_DIR, pid + ".txt"), "rb") as fh:
            if fh.read() != data:
                disk_mismatch.append(pid)
    if disk_mismatch:
        raise RuntimeError("A on disk differs from the server state for %s" % disk_mismatch[:5])

    ver = next_version()
    name = "A_final_labels_v%03d" % ver
    out_dir = os.path.join(EXPORT_DIR, name)
    zip_out = os.path.join(EXPORT_DIR, name + ".zip")
    man_out = os.path.join(EXPORT_DIR, name + "_manifest.json")
    chg_out = os.path.join(EXPORT_DIR, name + "_changes.csv")
    rev_out = os.path.join(EXPORT_DIR, name + "_review_status.csv")
    for p in (out_dir, zip_out, man_out, chg_out, rev_out):
        if os.path.exists(p):
            raise FileExistsError(p)
    os.makedirs(os.path.join(out_dir, LABEL_SUBDIR))  # fails if it already exists

    patches = index["patches"]
    files = []
    for pid in sorted(a_snapshot):
        data = a_snapshot[pid]
        rel = LABEL_SUBDIR + "/" + pid + ".txt"
        with open(os.path.join(out_dir, rel), "xb") as fh:
            fh.write(data)
        files.append({"path": rel, "patch_id": pid, "bytes": len(data), "sha256": sha256_bytes(data),
                      "boxes": len(_box_lines(data)), "modified_vs_B": data != store.b_bytes[pid]})

    # changes.csv (one row per patch that differs from B)
    changed = [f for f in files if f["modified_vs_B"]]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["patch_id", "spatial_chunk", "original_group", "final_split", "oof_scored",
                "B_box_count", "A_box_count", "boxes_removed_vs_B", "boxes_added_vs_B",
                "B_sha256", "A_sha256", "label_ops_in_history", "last_edit_utc", "edit_actions",
                "patch_review_status", "patch_note"])
    for f in changed:
        pid = f["patch_id"]
        e = patches[pid]
        bl, al = _box_lines(store.b_bytes[pid]), _box_lines(a_snapshot[pid])
        removed = [l for l in bl if l not in al]
        added = [l for l in al if l not in bl]
        ec = edit_counts.get(pid, {})
        pr = review["patches"].get(pid, {})
        w.writerow([pid, e["spatial_chunk"], e["original_group"], e["final_split"], e["scored"],
                    len(bl), len(al), " | ".join(removed), " | ".join(added),
                    sha256_bytes(store.b_bytes[pid]), f["sha256"], ec.get("label_ops", 0),
                    ec.get("last_edit_utc", ""), ";".join(ec.get("actions", [])),
                    pr.get("status", "unreviewed"), pr.get("note", "")])
    changes_csv = buf.getvalue().encode("utf-8")

    # review_status.csv (all non-default prediction / patch reviews)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["target", "patch_id", "prediction_k", "status", "note", "updated_at_utc",
                "prediction_confidence", "prediction_cx", "prediction_cy", "prediction_w", "prediction_h",
                "prediction_fold_ids"])
    for key, v in sorted(review["predictions"].items()):
        pid, k = key.rsplit("#", 1)
        p = v.get("prediction", {})
        w.writerow(["prediction", pid, k, v.get("status"), v.get("note", ""), v.get("updated_at_utc", ""),
                    p.get("confidence"), p.get("cx"), p.get("cy"), p.get("w"), p.get("h"),
                    ",".join(p.get("fold_ids", []))])
    for pid, v in sorted(review["patches"].items()):
        w.writerow(["patch", pid, "", v.get("status"), v.get("note", ""), v.get("updated_at_utc", "")] + [""] * 6)
    review_csv = buf.getvalue().encode("utf-8")

    with open(chg_out, "xb") as fh:
        fh.write(changes_csv)
    with open(rev_out, "xb") as fh:
        fh.write(review_csv)

    by_chunk = collections.defaultdict(lambda: collections.Counter())
    for f in files:
        c = by_chunk[str(patches[f["patch_id"]]["spatial_chunk"])]
        c["patches"] += 1
        c["boxes"] += f["boxes"]
        c["nonempty"] += f["boxes"] > 0
        c["modified_vs_B"] += f["modified_vs_B"]
    pred_status = collections.Counter(v.get("status") for v in review["predictions"].values())
    patch_status = collections.Counter(v.get("status") for v in review["patches"].values())
    b_man_sha = sha256_file(b_manifest_path)

    manifest = {
        "export_name": name,
        "export_version": ver,
        "purpose": purpose,
        "note": note,
        "generated_at_utc": utc_now(),
        "tool": {"name": "SSS contact annotation & OOF review platform", "version": TOOL_VERSION,
                 "platform_dir": TOOL_DIR},
        "source_zip": file_fingerprint(zip_path),
        "backup_B": {"path": b_dir, "label_dir": os.path.join(b_dir, LABEL_SUBDIR),
                     "backup_manifest": b_manifest_path, "backup_manifest_sha256": b_man_sha,
                     "label_files": len(store.b_bytes)},
        "working_A": {"label_dir": A_LABEL_DIR},
        "label_format": {
            "type": "YOLO txt",
            "line": "class_id cx cy width height",
            "coordinates": "normalised to 0-1 relative to the %dx%d image; 8 decimals" % (IMAGE_W, IMAGE_H),
            "classes": {str(k): v for k, v in CLASS_NAMES.items()},
            "empty_file_meaning": "patch has no contact label (file kept, 0 bytes)",
        },
        "counts": {
            "patch_count": len(patches),
            "label_file_count": len(files),
            "empty_label_files": sum(1 for f in files if f["boxes"] == 0),
            "nonempty_label_files": sum(1 for f in files if f["boxes"] > 0),
            "box_count": sum(f["boxes"] for f in files),
            "B_box_count": sum(len(_box_lines(b)) for b in store.b_bytes.values()),
            "spatial_chunks": sorted(int(c) for c in by_chunk),
            "by_spatial_chunk": {c: dict(v) for c, v in sorted(by_chunk.items(), key=lambda x: int(x[0]))},
        },
        "edit_summary": {
            "patches_modified_vs_B": [f["patch_id"] for f in changed],
            "patches_modified_vs_B_count": len(changed),
            "history_operations_total": n_ops,
            "undo_depth": history["undo_depth"],
            "redo_depth": history["redo_depth"],
            "label_actions_in_undo_stack": dict(collections.Counter(
                a for e in edit_counts.values() for a in e["actions"])),
            "operation_log": os.path.join(TOOL_DIR, "undo_log.jsonl"),
        },
        "review_status_summary": {
            "prediction_status_counts": {s: pred_status.get(s, 0) for s in PRED_STATUSES if s != "unreviewed"},
            "patch_status_counts": {s: patch_status.get(s, 0) for s in PATCH_STATUSES if s != "unreviewed"},
            "note": "Review statuses are human judgements only; they never add or delete boxes.",
        },
        "outputs": {
            "label_dir": os.path.join(out_dir, LABEL_SUBDIR),
            "zip": zip_out,
            "changes_csv": {"path": chg_out, "sha256": sha256_bytes(changes_csv), "rows": len(changed)},
            "review_status_csv": {"path": rev_out, "sha256": sha256_bytes(review_csv)},
        },
        "files": [{k: f[k] for k in ("path", "bytes", "sha256", "boxes", "modified_vs_B")} for f in files],
    }

    # ZIP: labels + changes + review status + manifest copy (the ZIP's own hash is only in the external manifest)
    inner_manifest = json.dumps(manifest, indent=1).encode("utf-8")
    with zipfile.ZipFile(zip_out, "x", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(name + "/" + name + "_manifest.json", inner_manifest)
        zf.writestr(name + "/" + name + "_changes.csv", changes_csv)
        zf.writestr(name + "/" + name + "_review_status.csv", review_csv)
        for f in files:
            zi = zipfile.ZipInfo(name + "/" + f["path"], date_time=(2026, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(zi, a_snapshot[f["patch_id"]])

    verification = verify_export(out_dir, zip_out, manifest, name)
    manifest["outputs"]["zip_sha256"] = sha256_file(zip_out)
    manifest["self_verification"] = verification
    with open(man_out, "x", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    return {"name": name, "dir": out_dir, "zip": zip_out, "manifest": man_out, "changes_csv": chg_out,
            "review_status_csv": rev_out, "verification": verification, "counts": manifest["counts"],
            "patches_modified_vs_B": len(changed)}


def verify_export(out_dir, zip_out, manifest, name):
    """Re-read the exported files and ZIP and compare with the manifest."""
    problems = []
    listed = {f["path"]: f for f in manifest["files"]}
    on_disk = sorted(os.listdir(os.path.join(out_dir, LABEL_SUBDIR)))
    if len(on_disk) != len(listed):
        problems.append("file count %d != manifest %d" % (len(on_disk), len(listed)))
    parsed_boxes = 0
    for fn in on_disk:
        rel = LABEL_SUBDIR + "/" + fn
        with open(os.path.join(out_dir, rel), "rb") as fh:
            data = fh.read()
        f = listed.get(rel)
        if f is None or sha256_bytes(data) != f["sha256"]:
            problems.append("hash mismatch %s" % rel)
        try:
            parsed_boxes += len(validate_boxes(parse_label_text(data.decode("ascii"))))
        except Exception as e:  # noqa: BLE001
            problems.append("unparseable %s: %s" % (rel, e))
    zip_files = 0
    with zipfile.ZipFile(zip_out) as zf:
        if zf.testzip() is not None:
            problems.append("zip CRC error")
        for zi in zf.infolist():
            rel = zi.filename[len(name) + 1:]
            if rel in listed:
                zip_files += 1
                if sha256_bytes(zf.read(zi)) != listed[rel]["sha256"]:
                    problems.append("zip hash mismatch %s" % rel)
    if zip_files != len(listed):
        problems.append("zip label count %d != %d" % (zip_files, len(listed)))
    return {"ok": not problems, "problems": problems[:50], "label_files_on_disk": len(on_disk),
            "label_files_in_zip": zip_files, "empty_files": sum(1 for f in listed.values() if f["bytes"] == 0),
            "parsed_boxes": parsed_boxes, "verified_at_utc": utc_now()}
