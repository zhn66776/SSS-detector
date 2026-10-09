# Results and interpretation

Recorded on 9 October 2026 from the existing local files. This document separates saved experiment evidence from evaluation work that remains open. It does not report a new training run.

## Data provenance and quality

The active detector uses Costa Head CES-86 patches. The presence of `download_dudgeon_mde.sh` in the workspace does not establish Dudgeon as the source of this training dataset.

The Costa mosaic is an ECW image at 0.1 m/pixel in WGS 84 / UTM zone 30N (EPSG:32630). The ERS registration convention leaves a 0.05 m pixel-centre versus pixel-edge ambiguity. Patch filenames encode source mosaic pixel positions.

The project identified two material source-data problems:

- A section of the Costa CSV target listing contains an arithmetic sequence of coordinates. The official GIS shapefile supplies the point positions used for the correction workflow; line contacts remain a separate source.
- The nominal 0.5 m MBES file contains an integer-valued third column, rather than the genuine elevation values used here. The elevation viewer uses the 2 m grid. Roughness is a geometric diagnostic, not sediment ground truth or backscatter.

The original source files remain local. This repository contains the relevant readers and correction scripts, not the survey records themselves.

## Spatial split

| Split | Spatial chunks | Patches |
|---|---|---:|
| Train | 1–7 | 6,847 |
| Validation | 8–9 | 1,696 |
| Test | 10–11 | 1,605 |
| Cross-split buffer | Boundary patches excluded from train/validation/test | 608 |

Each patch is 15 m wide and the regular stride is 10 m. The split was constructed from patch footprints with a 15 m cross-split separation. Geographic reasoning must use `spatial_chunk`; annotation group IDs reflect annotation order.

## Label review

Spatial cross-validation produced out-of-fold predictions for human review. A prediction unmatched to an annotation is treated as a review candidate; it can indicate a model error or a missing annotation.

| Label version, chunks 1–9 | Boxes | Nonempty patches |
|---|---:|---:|
| Original review backup | 907 | 748 |
| Round 1: `A_final_labels_v002` | 2,004 | 1,454 |
| Round 2: `A_round2_labels_v001` | 2,992 | 2,082 |

The current retraining package contains 2,528 train boxes and 372 validation boxes. The test labels were retained unchanged, with 167 boxes; they have not received the same review. Counts across overlapping patches are not counts of distinct seabed objects.

The review platforms use one-to-one matching that first maximizes the number of matches and then total IoU. Source ZIPs and backup labels are read-only. Edits go to a working copy with persistent undo/redo, and exports use new version names.

Model-assisted review introduces selection and confirmation bias. The historical review record also reports two possible duplicate box pairs and remaining validation omissions. No later independently reviewed test set was found in the inspected local project.

## Model evidence

Current checkpoint: `sss_contact_yolo26/models/yolo26n_A_round2_v001/best.pt` (local only).

| Field | Stored value |
|---|---|
| Checkpoint date | 2026-09-23T13:09:31.819092+00:00 |
| Architecture | YOLO26n, one input channel, one class (`contact`) |
| Ultralytics | 8.4.160 |
| Image size / batch | 256 / 256 |
| Maximum epochs / seed | 80 / 42 |
| Initial learning rate | 0.0001 |
| Precision / recall | 0.63557 / 0.68011 |
| mAP50 / mAP50–95 | 0.66530 / 0.38652 |

Training uses the round-2 package with 2× positive-patch oversampling. The checkpoint metrics were read directly during the October inspection. Earlier local validation notes report approximately 0.665 mAP50 and 0.386 mAP50–95 on CPU, consistent with the stored checkpoint metrics.

The round-1 result of 0.597 mAP50 and the round-2 result of 0.665 used different validation labels. A controlled model comparison requires both models to be evaluated against the same labels. No such comparison is claimed here.

## Regional inference

| Run ID | Input | Patches | Confidence | Merged predictions | Elapsed |
|---|---|---:|---:|---:|---:|
| `20260923T201409Z` | Built-in test region | 1,605 | 0.5 | 236 | 19 s |
| `20260923T201053Z` | Separate 0.5 m image, resampled to 0.1 m | 10,000 | 0.5 | 211 | 109 s |
| `20260923T215642Z` | 2 km × 2 km Costa region | 40,000 | 0.5 | 5,912 | 460 s |

Saved `results.json` entry counts and `contacts.csv` row counts agree with these manifests. The 2 km crop manifest records a distance of at least 100 m from all bundle patches. It comes from the same Costa survey and does not establish performance on a different survey.

A separate frontend performance experiment reused tiled 1 km predictions as synthetic input. Its display timings and counts are distinct from the saved 40,000-patch run above.

Prediction counts alone do not measure detection quality. External reference target lists were collected for the exploratory regions, but no complete, reviewed matching-based evaluation is claimed. Resampling a coarser input does not add native sensor detail.

## Evidence locations

The aggregate, data-free summary is stored in [`evidence/status-2026-10-09.json`](evidence/status-2026-10-09.json). Detailed local evidence remains in:

- `sss_contact_yolov8_spatial_split_bundle/metadata/split_manifest.json`
- Each review platform's `exports/*_manifest.json`
- `sss_contact_yolo26/retrain/sss_contact_yolo26_retrain_A_round2_v001/sss_contact_yolo26_spatial_split_A_round2_v001.zip`, member `metadata/provenance.json` under its bundle root
- The current checkpoint's `train_metrics`
- The inference application's `outputs/latest/` and `outputs/uploaded/*/latest/`
- `sss_contact_yolo26/experiments/*/manifest.json`

These paths describe assets to restore locally; their underlying data and checkpoints are excluded from Git.
