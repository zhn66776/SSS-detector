#!/usr/bin/env python3
"""Per-box origin of every label in A_final_labels_v002 -> label_sources_A_v002.csv

source = original_B          box identical to the frozen backup B
         adopted_prediction  box added by 'Adopt prediction as label' (with the OOF confidence)
         drawn_or_edited     box drawn by hand, or an adopted/original box that was later moved/resized/edited
"""
import csv
import io
import json
import os
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(HERE))          # .../sss_contact_yolo26
REPO_ROOT = os.path.dirname(PROJECT)                      # .../IRP-external
R1 = os.path.join(PROJECT, "label_review", "round1")
R2 = os.path.join(PROJECT, "label_review", "round2")
TOOL = os.path.join(R1, "sss_contact_yolo26_spatial_cv_review_20260922T234500Z_annotation_tool")
EXPORT = TOOL + "/exports/A_final_labels_v002/validation/ground_truth_labels"
B = os.path.join(R1, "sss_contact_yolo26_spatial_cv_review_20260922T234500Z_label_backup_B", "validation", "ground_truth_labels")
CV_ZIP = os.path.join(R1, "sss_contact_yolo26_spatial_cv_review_20260922T234500Z.zip")
OUT = os.path.join(HERE, "label_sources_A_v002.csv")


def box_lines(text):
    return [l.strip() for l in text.splitlines() if l.strip()]


dev = {r["patch_id"]: r for r in csv.DictReader(io.StringIO(zipfile.ZipFile(CV_ZIP).read(
    "sss_contact_yolo26_spatial_cv_review/development_split_manifest.csv").decode()))}

# the exact label line each adopt operation appended (ops still in effect = undo stack)
adopted = {}
for op in json.load(open(TOOL + "/history_state.json"))["undo"]:
    if op["kind"] == "labels" and op["action"] == "adopt_prediction":
        added = [l for l in box_lines(op["after"]["text"]) if l not in box_lines(op["before"]["text"])]
        for line in added:
            adopted[(op["patch_id"], line)] = op["detail"]

rows = []
for fn in sorted(os.listdir(EXPORT)):
    pid = fn[:-4]
    b_lines = set(box_lines(open(os.path.join(B, fn)).read()))
    for i, line in enumerate(box_lines(open(os.path.join(EXPORT, fn)).read())):
        _, cx, cy, w, h = line.split()
        d = adopted.get((pid, line))
        src = "original_B" if line in b_lines else "adopted_prediction" if d else "drawn_or_edited"
        rows.append({"patch_id": pid, "split": dev[pid]["final_split"], "spatial_chunk": dev[pid]["spatial_chunk"],
                     "box_index": i, "cx": cx, "cy": cy, "w": w, "h": h, "source": src,
                     "adopted_conf": round(d["confidence"], 4) if d else "",
                     "adopted_fold_ids": ",".join(d["fold_ids"]) if d else ""})

with open(OUT, "x", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(OUT, len(rows), "boxes")
