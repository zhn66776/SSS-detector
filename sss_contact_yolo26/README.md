# SSS contact detection with YOLO26n

This directory contains two rounds of spatial-CV-assisted label review, retraining helpers, exploratory regional inference, and a local inference map. See the [repository overview](../README.md) and [results record](../docs/RESULTS.md) for the current conclusions.

## Layout

| Directory | Content |
|---|---|
| `label_review/round1/` | Review platform for the original labels; final export `A_final_labels_v002`. |
| `label_review/round2/` | Review platform for the first-round labels; final export `A_round2_labels_v001`. |
| `retrain/sss_contact_yolo26_retrain_A_v002/` | First-round retraining and second-round spatial-CV preparation scripts. |
| `retrain/sss_contact_yolo26_retrain_A_round2_v001/` | Retraining with the second-round labels and validation error inspection helpers. |
| `models/yolo26n_A_round2_v001/` | Local checkpoint location; weights are excluded from Git. |
| `apps/sss_contact_yolo26_test_inference_map/` | Inference server, browser interface and map rendering. |
| `experiments/` | Regional crop and resampling scripts; imagery and output data remain local. |
| `notebooks/` | Historical local notebooks, excluded from Git. English output-free templates are in the repository-level `notebooks/`. |

## Current model and labels

The local `best.pt` checkpoint uses YOLO26n, one grayscale input channel, one `contact` class and `imgsz=256`. Stored validation metrics are precision 0.63557, recall 0.68011, mAP50 0.66530 and mAP50–95 0.38652. It was trained with Ultralytics 8.4.160 on 23 September 2026.

Two human-review rounds increased the chunks 1–9 annotations from 907 to 2,004 and then 2,992 boxes. Test chunks retain their original annotations. The [experiment record](../docs/RESULTS.md) lists each model’s validation labels and scores.

## Run

Follow the [setup guide](../docs/RUNNING.md), then run from this directory:

```sh
cd apps/sss_contact_yolo26_test_inference_map
.venv/bin/python serve_map.py
```

Open `http://127.0.0.1:8770/`. The app supports `--bundle`, `--weights` and `--outputs`. Review-platform ports are 8767 and 8768; their individual READMEs explain setup.

## Paths and local state

The inference app and review platforms derive default paths from their own locations. Keep this directory beside `sss_contact_yolov8_spatial_split_bundle/`, or supply path overrides. Set preparation-script path constants to your local data locations.

Review ZIPs and backup-B labels are read-only. Working labels, history and exports remain local. Existing inference results are saved under `outputs/latest/` or `outputs/uploaded/<dataset>/latest/`; a completed run replaces that dataset's previous output. Recreate virtual environments when moving to a different machine.
