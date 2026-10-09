#!/usr/bin/env python3
"""Build the Drive-upload ZIP for YOLO26 retraining with the round-2 reviewed labels.

Split (unchanged, from split_manifest.csv final_split):
  train = spatial chunks 1-7, val = chunks 8-9, test = chunks 10-11, buffer = 15 m guard (unused)
Labels:
  spatial chunks 1-9  (train, val, chunk 7-9 buffer) -> round-2 export A_round2_labels_v001
  spatial chunks 10-11 (test, chunk 10 buffer)        -> original bundle labels (not re-reviewed yet)
Images: original 150x150 grayscale PNGs from the spatial split bundle (byte copies).
Sources are only read; the ZIP is written as a new file and never overwrites.
"""
import csv
import hashlib
import io
import json
import os
import zipfile
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(HERE))          # .../sss_contact_yolo26
REPO_ROOT = os.path.dirname(PROJECT)                      # .../IRP-external
R1 = os.path.join(PROJECT, "label_review", "round1")
R2 = os.path.join(PROJECT, "label_review", "round2")
BUNDLE = os.path.join(REPO_ROOT, "sss_contact_yolov8_spatial_split_bundle")
EXPORT = os.path.join(R2, "sss_contact_yolo26_spatial_cv_review_A_v002_20260923T111854Z_annotation_tool", "exports", "A_round2_labels_v001")
PREV_V002 = os.path.join(R1, "sss_contact_yolo26_spatial_cv_review_20260922T234500Z_annotation_tool", "exports",
                         "A_final_labels_v002", "validation", "ground_truth_labels")
OUT_DIR = HERE
NAME = "sss_contact_yolo26_spatial_split_A_round2_v001"
OUT_ZIP = os.path.join(OUT_DIR, NAME + ".zip")
SPLITS = ("train", "val", "test", "buffer")


def sha(b):
    return hashlib.sha256(b).hexdigest()


def n_boxes(b):
    return sum(1 for l in b.decode("ascii").splitlines() if l.strip())


def main():
    man = json.load(open(EXPORT + "_manifest.json"))
    new = {}
    for f in man["files"]:
        data = open(os.path.join(EXPORT, f["path"]), "rb").read()
        assert sha(data) == f["sha256"], f["path"]
        new[os.path.basename(f["path"])[:-4]] = data
    assert len(new) == 9000

    rows = list(csv.DictReader(open(os.path.join(BUNDLE, "metadata/split_manifest.csv"))))
    entries, out_rows, stats = [], [], Counter()
    for r in rows:
        pid, split, chunk = r["patch_id"], r["final_split"], int(r["spatial_chunk"])
        img = open(os.path.join(BUNDLE, "data/images", split, pid + ".png"), "rb").read()
        orig = open(os.path.join(BUNDLE, "data/labels", split, pid + ".txt"), "rb").read()
        if chunk <= 9:
            lab, src = new[pid], "A_round2_labels_v001"
            v002 = open(os.path.join(PREV_V002, pid + ".txt"), "rb").read()
        else:
            lab, src, v002 = orig, "original_bundle", orig
        entries.append((split, pid, img, lab))
        k = n_boxes(lab)
        stats[(split, "patches")] += 1
        stats[(split, "positive")] += k > 0
        stats[(split, "boxes")] += k
        stats[(split, "changed_vs_original")] += lab != orig
        stats[(split, "changed_vs_round1_v002")] += lab != v002
        out_rows.append({**{c: r[c] for c in ("patch_id", "original_group", "spatial_chunk", "final_split",
                                               "x0", "x1", "y0", "y1")},
                         "positive": int(k > 0), "box_count": k, "label_source": src,
                         "label_changed_vs_original_bundle": int(lab != orig),
                         "label_changed_vs_round1_v002": int(lab != v002), "label_sha256": sha(lab)})

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(out_rows[0]))
    w.writeheader()
    w.writerows(out_rows)
    keys = ("patches", "positive", "boxes", "changed_vs_original", "changed_vs_round1_v002")
    summary = {s: {k: stats[(s, k)] for k in keys} for s in SPLITS}
    dataset = {"train": "data/images/train", "val": "data/images/val", "test": "data/images/test",
               "nc": 1, "names": ["contact"], "channels": 1}
    readme = (
        "# SSS contact YOLO26 retraining bundle (round-2 labels)\n\n"
        "- train: spatial chunks 1-7 | val: chunks 8-9 | test: chunks 10-11 | buffer: 15 m guard, not used\n"
        "- chunks 1-9 labels: human-reviewed round-2 export A_round2_labels_v001\n"
        "- chunks 10-11 labels: original bundle labels (test not yet re-reviewed)\n"
        "- images: original 150x150 single-channel PNG patches (unchanged)\n"
        "- per-patch label source and SHA-256: metadata/split_manifest_A_round2_v001.csv\n")
    provenance = {"bundle": NAME, "image_source": BUNDLE, "label_source_chunks_1_9": EXPORT,
                  "label_source_chunks_10_11": BUNDLE + "/data/labels",
                  "round2_manifest_sha256": sha(open(EXPORT + "_manifest.json", "rb").read()),
                  "round2_zip_sha256": man["outputs"]["zip_sha256"], "split_summary": summary}

    if os.path.exists(OUT_ZIP):
        raise FileExistsError(OUT_ZIP)
    with zipfile.ZipFile(OUT_ZIP, "x") as z:
        def put(rel, data, deflate=True):
            zi = zipfile.ZipInfo(NAME + "/" + rel, date_time=(2026, 9, 23, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED if deflate else zipfile.ZIP_STORED
            zi.external_attr = 0o644 << 16
            z.writestr(zi, data)
        put("dataset.yaml", json.dumps(dataset, indent=2) + "\n")
        put("README.md", readme)
        put("metadata/split_manifest_A_round2_v001.csv", buf.getvalue())
        put("metadata/provenance.json", json.dumps(provenance, indent=2) + "\n")
        for split, pid, img, lab in sorted(entries):
            put("data/images/%s/%s.png" % (split, pid), img, deflate=False)
            put("data/labels/%s/%s.txt" % (split, pid), lab)

    by_path = {}
    for split, pid, img, lab in entries:
        by_path["data/images/%s/%s.png" % (split, pid)] = img
        by_path["data/labels/%s/%s.txt" % (split, pid)] = lab
    with zipfile.ZipFile(OUT_ZIP) as z:
        assert z.testzip() is None
        names = [n[len(NAME) + 1:] for n in z.namelist()]
        assert set(by_path) <= set(names) and len(names) == len(by_path) + 4
        for rel, data in by_path.items():
            assert z.read(NAME + "/" + rel) == data, rel
    print(json.dumps(summary, indent=1))
    print("zip:", OUT_ZIP, os.path.getsize(OUT_ZIP), "bytes, sha256", sha(open(OUT_ZIP, "rb").read()))


if __name__ == "__main__":
    main()
