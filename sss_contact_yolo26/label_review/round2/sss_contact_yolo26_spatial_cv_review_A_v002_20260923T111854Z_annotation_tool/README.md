# SSS contact review platform — round 2

Second review round of spatial chunks 1–9. It uses the same platform code as round 1. Only the
data sources, the default port (8768) and the export name differ.

## Start / stop

```bash
cd /Users/haonanzhang/IRP-external/sss_contact_yolo26/label_review/round2/sss_contact_yolo26_spatial_cv_review_A_v002_20260923T111854Z_annotation_tool
python3 serve.py --host 127.0.0.1 --port 8768
```

Open **http://127.0.0.1:8768/**. Stop with Ctrl+C in that terminal, or run `python3 serve.py --stop`.

Round 1 lives in `…20260922T234500Z_annotation_tool/` on port 8767. Both platforms can run at the
same time without affecting each other.

## Data sources

| Role | Path | Content |
|---|---|---|
| **ZIP** (read-only) | `../sss_contact_yolo26_spatial_cv_review_A_v002_20260923T111854Z.zip` (next to this folder, in `label_review/round2/`) | round-2 8-fold spatial CV trained on the A_v002 labels: OOF predictions, fold CSVs, original 150×150 images |
| **B** (frozen backup) | `../sss_contact_yolo26_spatial_cv_review_A_v002_20260923T111854Z_label_backup_B/validation/ground_truth_labels/` | byte copy of round-1 final labels `A_final_labels_v002` (9,000 files, 2,004 boxes) |
| **A** (working labels) | `working_A/validation/ground_truth_labels/` | starts identical to B; every edit in the web UI is saved here |

## Round-2 baseline (conf 0.20 / IoU 0.50, labels = A_v002)

The platform recomputes the same numbers as the ZIP, with no differences:

| | patches | boxes |
|---|---|---|
| TP | 1,204 | 1,466 |
| FP candidates (possible missed contacts) | 1,626 | 2,004 |
| · unlabelled_model_candidate | 1,226 | 1,483 |
| · extra_prediction_on_labelled_patch | 400 | 521 |
| FN | 386 | 518 |
| TN | 6,114 | – |
| unscored | 224 | – |

FP candidates by confidence:

| confidence | boxes |
|---|---|
| 0.2–0.3 | 743 |
| 0.3–0.5 | 822 |
| 0.5–0.7 | 385 |
| ≥ 0.7 | 54 |

845 candidates are supported by both folds that held the patch out. The ZIP's `fold_metrics.csv`
has no per-fold mAP columns, because the newer Ultralytics trainer returned no metrics dict. This
does not affect the review.

## Export

Use the **Export A…** button, or run `python3 export_labels.py --note "..."`. Each export creates a
new version, `exports/A_round2_labels_vNNN`, and never overwrites an existing one.

Keyboard shortcuts (ignored while a text/number field has focus): **A** adopt the selected prediction as a label, **D** or Delete delete the selected GT box, ← / → previous/next patch, B draw mode, V select mode, Esc deselect, Cmd/Ctrl+Z undo, Cmd/Ctrl+Shift+Z redo.

Editing, undo/redo, review status, recovery and integrity checks work exactly as in round 1. See the
round-1 README for the full description.
