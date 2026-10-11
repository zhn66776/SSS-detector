# SSS contact review platform — round 1

A local browser application for reviewing out-of-fold contact predictions against human annotations. It covers 9,000 patches from spatial chunks 1–9. Source imagery, prediction ZIPs and annotation state are kept outside Git.

## Start and stop

Run from this directory after restoring the review ZIP and backup-B directory beside it:

```sh
python3 serve.py --host 127.0.0.1 --port 8767
```

Open `http://127.0.0.1:8767/`. Stop with Ctrl+C or `python3 serve.py --stop`. Use `--zip` and `--backup-b` to override the defaults.

## Data roles

- **Source ZIP:** read-only images, original labels, spatial metadata and out-of-fold predictions.
- **Backup B:** read-only original annotations.
- **Working A:** editable annotations in `working_A/validation/ground_truth_labels/`.

Images are read from the ZIP on demand. Edits are saved atomically to A. Undo/redo history survives a server restart. Restore the existing history and working files together when resuming a review on another machine.

## Review and editing

Tabs show all patches, unmatched-prediction candidates, false negatives, true positives, true negatives and unscored patches. Confidence and IoU thresholds are adjustable; the defaults are 0.20 and 0.50.

Matching is one-to-one: maximize the number of matched pairs, then maximize total IoU. Reviewers inspect unmatched predictions for missing annotations or model errors.

The interface supports adopting predictions, drawing, moving and resizing boxes, editing coordinates, deleting labels, restoring patches from B, and undo/redo. Left/right arrows navigate patches; B selects draw mode; V selects selection mode; Delete removes a selected label; Esc clears selection; Cmd/Ctrl+Z and Cmd/Ctrl+Shift+Z undo and redo. The additional A/D shortcuts are available in the round-2 platform.

## Export and verification

```sh
python3 export_labels.py --note "Review completed"
python3 verify_integrity.py
python3 -m unittest discover -s tests -v
```

Exports receive a new versioned name under `exports/A_final_labels_vNNN`. The integrity checker verifies the original ZIP and backup B; use `--export NAME` to inspect an export.

The first-round final export is `A_final_labels_v002`: 2,004 boxes in 1,454 nonempty patches, compared with 907 boxes in 748 patches before review. `v001` contains the earlier application acceptance-test annotations.

The 13 matching and label-format tests passed on 9 October 2026.
