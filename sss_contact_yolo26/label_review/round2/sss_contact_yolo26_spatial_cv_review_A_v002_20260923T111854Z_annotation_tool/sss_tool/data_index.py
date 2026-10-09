"""Build a read-only per-patch index from the source ZIP (no extraction).

The index reuses the merged OOF predictions stored in html_review/review_data.json
and cross-checks them against predictions_by_fold.csv, predicted_labels/*.txt,
evaluation_by_fold.csv, fold_agreement.csv, fold_plan.csv, unscored.csv and
development_split_manifest.csv. Any failed check is recorded; critical ones
abort the build (we never "fill in" missing OOF data).
"""
import collections
import csv
import io
import json
import os
import struct
import zipfile

from .common import (INDEX_DIR, LABEL_SUBDIR, ZIP_ROOT, atomic_write_json,
                     read_json, utc_now)
from .labels import parse_label_text
from .matching import box_iou

INDEX_PATH = os.path.join(INDEX_DIR, "patch_index.json")
CHECK_REPORT_PATH = os.path.join(INDEX_DIR, "data_check_report.json")
INDEX_FORMAT = 3

EXPECTED_FILES = [
    "development_split_manifest.csv", "fold_plan.csv", "fold_metrics.csv",
    "predictions_by_fold.csv", "evaluation_by_fold.csv", "fold_agreement.csv",
    "unscored.csv", "html_review/review_data.json", "html_review/review_manifest.json",
    "html_review/fp_candidates.csv", "html_review/fn.csv", "html_review/tp.csv",
    "html_review/tn.csv", "html_review/unscored.csv",
]


class DataError(RuntimeError):
    pass


def _csv(z, name):
    return list(csv.DictReader(io.StringIO(z.read(ZIP_ROOT + name).decode("utf-8"))))


def _merge_like_bundle(raw):
    """Re-derive the bundle's merged OOF union: greedy by confidence, IoU >= 0.5
    against each cluster's representative (highest-confidence) box."""
    raw = sorted(raw, key=lambda p: -p["confidence"])
    clusters = []
    for p in raw:
        best, bi = None, 0.0
        for c in clusters:
            v = box_iou(p, c["rep"])
            if v >= 0.5 and v > bi:
                best, bi = c, v
        if best is not None:
            best["members"].append(p)
        else:
            clusters.append({"rep": p, "members": [p]})
    return clusters


def build_index(zip_path, zip_sha256):
    checks = []

    def check(name, ok, detail="", critical=True):
        checks.append({"check": name, "ok": bool(ok), "critical": critical, "detail": detail})
        if critical and not ok:
            raise DataError("data check failed: %s (%s)" % (name, detail))

    z = zipfile.ZipFile(zip_path)
    names = set(z.namelist())
    missing = [f for f in EXPECTED_FILES if ZIP_ROOT + f not in names]
    check("expected CSV/JSON files present in ZIP", not missing, "missing: %s" % missing)

    dev = _csv(z, "development_split_manifest.csv")
    dev_by = {r["patch_id"]: r for r in dev}
    check("development_split_manifest has 9000 unique patches", len(dev) == 9000 == len(dev_by), "%d rows" % len(dev))
    chunks = collections.Counter(int(r["spatial_chunk"]) for r in dev)
    check("spatial chunks are exactly 1..9", sorted(chunks) == list(range(1, 10)), dict(sorted(chunks.items())))
    chunk_group = sorted({(int(r["spatial_chunk"]), r["original_group"]) for r in dev})
    check("each spatial_chunk maps to one original_group", len(chunk_group) == 9, chunk_group, critical=False)

    img_prefix = ZIP_ROOT + "validation/images/"
    gt_prefix = ZIP_ROOT + LABEL_SUBDIR + "/"
    pl_prefix = ZIP_ROOT + "validation/predicted_labels/"
    img_ids = {n[len(img_prefix):-4] for n in names if n.startswith(img_prefix) and n.endswith(".png")}
    gt_ids = {n[len(gt_prefix):-4] for n in names if n.startswith(gt_prefix) and n.endswith(".txt")}
    pl_ids = {n[len(pl_prefix):-4] for n in names if n.startswith(pl_prefix) and n.endswith(".txt")}
    check("9000 PNG images match manifest patch ids", img_ids == set(dev_by), "%d images" % len(img_ids))
    check("9000 ground_truth_labels match manifest patch ids", gt_ids == set(dev_by), "%d labels" % len(gt_ids))
    check("9000 predicted_labels match manifest patch ids", pl_ids == set(dev_by), "%d files" % len(pl_ids), critical=False)

    ihdr = collections.Counter()
    for pid in dev_by:
        with z.open(img_prefix + pid + ".png") as f:
            head = f.read(33)
        if head[:8] != b"\x89PNG\r\n\x1a\n" or head[12:16] != b"IHDR":
            ihdr["not_png"] += 1
            continue
        w, h, bd, ct = struct.unpack(">IIBB", head[16:26])
        ihdr["%dx%d bitdepth=%d colortype=%d" % (w, h, bd, ct)] += 1
    check("all images are 150x150 8-bit grayscale PNG (colour type 0)",
          ihdr == {"150x150 bitdepth=8 colortype=0": 9000}, dict(ihdr))

    rd = json.loads(z.read(ZIP_ROOT + "html_review/review_data.json"))
    zmanifest = json.loads(z.read(ZIP_ROOT + "html_review/review_manifest.json"))
    recs = {r["patch_id"]: r for r in rd["records"]}
    skipped = {s["patch_id"]: s["reason"] for s in rd["skipped"]}
    uns = {r["patch_id"]: r["reason"] for r in _csv(z, "unscored.csv")}
    uns2 = {r["patch_id"] for r in _csv(z, "html_review/unscored.csv")}
    check("review_data records + skipped cover 9000 patches disjointly",
          not (set(recs) & set(skipped)) and set(recs) | set(skipped) == set(dev_by),
          "%d records, %d skipped" % (len(recs), len(skipped)))
    check("224 unscored: review_data.skipped == unscored.csv == html_review/unscored.csv",
          set(skipped) == set(uns) == uns2 and len(uns) == 224, "%d" % len(uns))
    check("8776 OOF-scored patches", len(recs) == 8776, "%d" % len(recs))
    check("review_data prediction floor is 0.05", rd.get("prediction_confidence_floor") == 0.05,
          rd.get("prediction_confidence_floor"))

    fold_plan = _csv(z, "fold_plan.csv")
    fold_val = {f["fold_id"]: [int(c) for c in f["validation_chunks"].split(",")] for f in fold_plan}
    fold_train = {f["fold_id"]: [int(c) for c in f["train_chunks"].split(",")] for f in fold_plan}
    expected_windows = [[i, i + 1] for i in range(1, 9)]
    check("8 adjacent two-chunk validation windows 1-2 .. 8-9",
          sorted(fold_val.values()) == expected_windows, fold_val)
    check("each fold trains on the other 7 chunks",
          all(sorted(fold_train[f] + fold_val[f]) == list(range(1, 10)) for f in fold_val), "")

    ev = _csv(z, "evaluation_by_fold.csv")
    ev_by = collections.defaultdict(list)
    for e in ev:
        ev_by[e["patch_id"]].append(e)
    check("evaluation_by_fold patch set == scored patches", set(ev_by) == set(recs), "%d rows" % len(ev))
    check("fold ids in review_data == evaluation_by_fold",
          all(sorted(r["cv_fold_ids"]) == sorted(e["fold_id"] for e in ev_by[p]) for p, r in recs.items()), "")
    check("each evaluating fold holds out the patch's chunk",
          all(r["spatial_chunk"] in fold_val[f] for r in recs.values() for f in r["cv_fold_ids"]), "")
    per_fold_n = collections.Counter(e["fold_id"] for e in ev)
    check("fold_plan validation_patches_after_buffer == evaluation_by_fold rows",
          all(int(f["validation_patches_after_buffer"]) == per_fold_n[f["fold_id"]] for f in fold_plan),
          dict(per_fold_n))

    fa = {r["patch_id"]: r for r in _csv(z, "fold_agreement.csv")}
    check("fold_agreement covers scored patches", set(fa) == set(recs), "%d" % len(fa))
    check("fold_agreement folds_evaluated == cv_fold_count",
          all(int(fa[p]["folds_evaluated"]) == r["cv_fold_count"] for p, r in recs.items()), "")

    check("spatial_chunk/annotation_group in review_data == development manifest",
          all(r["spatial_chunk"] == int(dev_by[p]["spatial_chunk"]) and r["annotation_group"] == dev_by[p]["original_group"]
              for p, r in recs.items()), "")

    pbf = _csv(z, "predictions_by_fold.csv")
    raw_by = collections.defaultdict(list)
    for p in pbf:
        raw_by[p["patch_id"]].append({
            "fold_id": p["fold_id"], "prediction_index": int(p["prediction_index"]),
            "class_id": int(p["class_id"]), "cx": float(p["cx"]), "cy": float(p["cy"]),
            "w": float(p["w"]), "h": float(p["h"]), "confidence": float(p["confidence"]),
        })
    check("raw fold predictions only on scored patches", set(raw_by) <= set(recs), "")
    check("raw fold predictions come from folds that evaluated the patch",
          all(p["fold_id"] in recs[pid]["cv_fold_ids"] for pid, ps in raw_by.items() for p in ps), "")
    check("prediction class ids are all 0", {p["class_id"] for ps in raw_by.values() for p in ps} <= {0}, "")
    min_conf = min(p["confidence"] for ps in raw_by.values() for p in ps)
    check("saved prediction confidence floor >= 0.05", min_conf >= 0.05, "min=%.6f" % min_conf)

    merged_total = sum(len(r["predictions"]) for r in recs.values())
    src_total = sum(p["source_prediction_count"] for r in recs.values() for p in r["predictions"])
    check("merged predictions account for every raw fold prediction", src_total == len(pbf),
          "%d raw rows, %d merged boxes" % (len(pbf), merged_total))

    merge_mismatch, predlabel_mismatch, gt_mismatch = [], [], []
    patches = {}
    for pid in sorted(dev_by, key=lambda x: (int(dev_by[x]["spatial_chunk"]), x)):
        d = dev_by[pid]
        gt_text = z.read(gt_prefix + pid + ".txt").decode("ascii")
        gt_boxes = parse_label_text(gt_text)
        entry = {
            "patch_id": pid,
            "spatial_chunk": int(d["spatial_chunk"]),
            "original_group": d["original_group"],
            "final_split": d["final_split"],
            "bounds_m": {k: float(d[k]) for k in ("x0", "x1", "y0", "y1")},
            "zip_image_member": img_prefix + pid + ".png",
            "zip_gt_member": gt_prefix + pid + ".txt",
            "zip_pred_label_member": pl_prefix + pid + ".txt",
            "scored": pid in recs,
        }
        if pid in recs:
            r = recs[pid]
            g2 = [(b["class_id"], b["cx"], b["cy"], b["w"], b["h"]) for b in r["ground_truth"]]
            g1 = [(b["class_id"], b["cx"], b["cy"], b["w"], b["h"]) for b in gt_boxes]
            if len(g1) != len(g2) or any(abs(a - b) > 1e-7 for x, y in zip(g1, g2) for a, b in zip(x, y)):
                gt_mismatch.append(pid)
            clusters = _merge_like_bundle(raw_by.get(pid, []))
            preds = []
            ok = len(clusters) == len(r["predictions"])
            for k, (c, m) in enumerate(zip(clusters, r["predictions"])):
                folds = sorted({x["fold_id"] for x in c["members"]})
                if (abs(c["rep"]["confidence"] - m["confidence"]) > 1e-6 or folds != sorted(m["fold_ids"])
                        or len(c["members"]) != m["source_prediction_count"]
                        or any(abs(c["rep"][q] - m[q]) > 1e-6 for q in ("cx", "cy", "w", "h"))):
                    ok = False
                preds.append({
                    "k": k,
                    "class_id": m["class_id"], "cx": m["cx"], "cy": m["cy"], "w": m["w"], "h": m["h"],
                    "confidence": m["confidence"],
                    "fold_ids": m["fold_ids"], "fold_support": m["fold_support"],
                    "representative_fold_id": m["fold_id"],
                    "representative_fold_prediction_index": m["prediction_index"],
                    "source_prediction_count": m["source_prediction_count"],
                    "members": [{"fold_id": x["fold_id"], "prediction_index": x["prediction_index"],
                                 "confidence": x["confidence"], "cx": x["cx"], "cy": x["cy"],
                                 "w": x["w"], "h": x["h"]} for x in c["members"]],
                })
            if not ok:
                merge_mismatch.append(pid)
            pl = parse_pred_label(z.read(pl_prefix + pid + ".txt").decode("ascii"))
            if len(pl) != len(preds) or any(abs(a - b) > 1e-6 for x, p in zip(pl, preds)
                                             for a, b in zip(x, (p["cx"], p["cy"], p["w"], p["h"], p["confidence"]))):
                predlabel_mismatch.append(pid)
            entry.update({
                "cv_fold_ids": r["cv_fold_ids"],
                "cv_fold_count": r["cv_fold_count"],
                "predictions": preds,
                "fold_evaluation": [{k2: (e[k2] if k2 == "fold_id" else int(e[k2]))
                                     for k2 in ("fold_id", "gt_count", "prediction_count", "tp_count", "fp_count", "fn_count")}
                                    for e in ev_by[pid]],
                "fold_agreement": {k2: fa[pid][k2] for k2 in ("folds_evaluated", "folds_with_fp", "folds_with_fn",
                                                             "folds_with_tp", "max_unmatched_confidence", "candidate_type")},
                "zip_baseline": r["baseline"],
            })
        else:
            if z.getinfo(pl_prefix + pid + ".txt").file_size != 0:
                predlabel_mismatch.append(pid)
            entry.update({"cv_fold_ids": [], "cv_fold_count": 0, "predictions": [],
                          "unscored_reason": uns[pid]})
        patches[pid] = entry

    check("review_data GT == ZIP ground_truth_labels", not gt_mismatch, gt_mismatch[:10])
    check("review_data merged predictions reproduce from predictions_by_fold.csv (greedy, IoU>=0.5)",
          not merge_mismatch, "%d mismatching patches %s" % (len(merge_mismatch), merge_mismatch[:10]))
    check("predicted_labels/*.txt == merged predictions (empty for unscored)", not predlabel_mismatch,
          predlabel_mismatch[:10], critical=False)

    folds = [{"fold_id": f["fold_id"], "train_chunks": fold_train[f["fold_id"]],
              "validation_chunks": fold_val[f["fold_id"]],
              "validation_patches_after_buffer": int(f["validation_patches_after_buffer"]),
              "validation_buffer_removed": int(f["validation_buffer_removed"])} for f in fold_plan]
    index = {
        "index_format": INDEX_FORMAT,
        "built_at_utc": utc_now(),
        "source_zip": zip_path,
        "source_zip_sha256": zip_sha256,
        "zip_root": ZIP_ROOT,
        "folds": folds,
        "zip_review_manifest": zmanifest,
        "prediction_confidence_floor": rd.get("prediction_confidence_floor"),
        "chunk_to_group": {str(c): g for c, g in chunk_group},
        "final_split_values": sorted({r["final_split"] for r in dev}),
        "patches": patches,
    }
    report = {"generated_at_utc": utc_now(), "source_zip": zip_path, "source_zip_sha256": zip_sha256,
              "all_critical_passed": all(c["ok"] for c in checks if c["critical"]),
              "all_passed": all(c["ok"] for c in checks), "checks": checks,
              "counts": {"patches": len(patches), "scored": len(recs), "unscored": len(uns),
                         "raw_fold_predictions": len(pbf), "merged_predictions": merged_total,
                         "fold_evaluation_rows": len(ev), "gt_boxes_in_zip": sum(
                             len(parse_label_text(z.read(gt_prefix + p + ".txt").decode())) for p in dev_by)},
              "notes": [
                  "Merged prediction 'prediction_index' in review_data is the representative fold's own index, "
                  "not unique within a patch; this tool identifies merged predictions by position k "
                  "(descending confidence), which equals the ZIP baseline's match indices.",
                  "fold_support counts folds contributing any box >= 0.05 to the merged cluster; the UI also "
                  "shows support restricted to members >= the current confidence threshold.",
              ]}
    z.close()
    return index, report


def parse_pred_label(text):
    out = []
    for line in text.splitlines():
        if line.strip():
            t = line.split()
            out.append(tuple(float(x) for x in t[1:6]))
    return out


def load_or_build_index(zip_path, zip_sha256, log=print):
    idx = read_json(INDEX_PATH)
    if idx and idx.get("index_format") == INDEX_FORMAT and idx.get("source_zip_sha256") == zip_sha256:
        return idx, read_json(CHECK_REPORT_PATH)
    log("building patch index from ZIP (no extraction) ...")
    idx, report = build_index(zip_path, zip_sha256)
    atomic_write_json(INDEX_PATH, idx, compact=True)
    atomic_write_json(CHECK_REPORT_PATH, report)
    return idx, report
