# Experiments and results

YOLO training and inference runs from 23 September 2026, with saved records checked on 9 October 2026. A separate [Qwen3-VL QLoRA experiment from 16 October 2026](QWEN3VL_QLORA.md) includes a completed training notebook and validation results; those results use a different evaluation protocol and are not a direct YOLO benchmark.

## Data sources

The detector uses Costa Head CES-86 patches. `download_dudgeon_mde.sh` supports a separate Dudgeon data-acquisition workflow.

The Costa mosaic is an ECW image at 0.1 m/pixel in WGS 84 / UTM zone 30N (EPSG:32630). Patch filenames encode source mosaic pixel positions. The code uses the ERS registration coordinate as the upper-left grid origin; its pixel-centre versus pixel-edge convention has a 0.05 m ambiguity.

Data inspection identified two source issues:

- A section of the Costa CSV target listing contains an arithmetic sequence of coordinates. The correction workflow uses official GIS point positions and retains CSV line contacts separately.
- The nominal 0.5 m MBES file contains an integer-valued third column. Elevation rendering uses the 2 m LAT grid, and roughness is calculated from that grid's local elevation residuals.

## Spatial split

| Split | Spatial chunks | Patches |
|---|---|---:|
| Train | 1–7 | 6,847 |
| Validation | 8–9 | 1,696 |
| Test | 10–11 | 1,605 |
| Buffer | Cross-split boundary patches | 608 |

Each patch spans 15 m, with a 10 m stride and 5 m overlap. The split uses patch footprints and a 15 m cross-split buffer. `spatial_chunk` records geographic order; group IDs record annotation order.

## Annotation versions

Spatial cross-validation produces out-of-fold predictions for human review. Reviewers inspect unmatched predictions for missing labels or model errors, then accept or edit boxes in the annotation platform.

| Labels for chunks 1–9 | Boxes | Nonempty patches |
|---|---:|---:|
| Original annotations | 907 | 748 |
| Round 1: `A_final_labels_v002` | 2,004 | 1,454 |
| Round 2: `A_round2_labels_v001` | 2,992 | 2,082 |

The round-2 training package contains 2,528 training boxes and 372 validation boxes. Test chunks retain the original 167 boxes and await manual review. Box counts are per-patch annotation instances, including repeated targets in overlapping patches.

The review platforms use one-to-one matching, maximizing the number of matches and then total IoU. Source ZIPs and backup labels are read-only. Edits are saved to a working copy with persistent undo/redo, and exports receive versioned names.

Review candidates were selected from model predictions. Evaluation work includes an independently selected test sample, remaining validation omissions and two possible duplicate box pairs recorded during review.

## Training and validation

Checkpoint: `sss_contact_yolo26/models/yolo26n_A_round2_v001/best.pt`.

| Setting | Value |
|---|---|
| Checkpoint date | 2026-09-23T13:09:31.819092+00:00 |
| Architecture | YOLO26n, one grayscale input channel, one `contact` class |
| Ultralytics | 8.4.160 |
| Image size / batch | 256 / 256 |
| Maximum epochs / seed | 80 / 42 |
| Initial learning rate | 0.0001 |
| Positive-patch oversampling | 2× |
| Validation split | Spatial chunks 8–9, round-2 annotations |
| Precision / recall | 0.63557 / 0.68011 |
| mAP50 / mAP50–95 | 0.66530 / 0.38652 |

The metrics above are stored in the checkpoint's `train_metrics`. A local CPU validation run reproduced approximately 0.665 mAP50 and 0.386 mAP50–95.

| Training run | Validation labels | mAP50 |
|---|---|---:|
| Round 1 | `A_final_labels_v002` | 0.597 |
| Round 2 | `A_round2_labels_v001` | 0.665 |

The two runs used different validation annotations. Re-evaluation on shared annotations is the next model-comparison step.

## Regional inference

| Run ID | Input | Patches | Confidence | Merged predictions | Elapsed |
|---|---|---:|---:|---:|---:|
| `20260923T201409Z` | Test chunks 10–11 | 1,605 | 0.5 | 236 | 19 s |
| `20260923T201053Z` | Separate survey image, 0.5 m native resolution resampled to 0.1 m | 10,000 | 0.5 | 211 | 109 s |
| `20260923T215642Z` | 2 km × 2 km Costa region | 40,000 | 0.5 | 5,912 | 460 s |

The saved prediction and contact CSV counts agree with the run manifests. The 2 km crop comes from the Costa survey, at least 100 m from every dataset-bundle patch. Reference target lists are available for regional detection review.

A separate browser performance test tiled the 1 km prediction results four times to measure display and export performance. The 40,000-patch row above records the later inference run on the 2 km image.

## Saved records

The [aggregate summary](evidence/status-2026-10-09.json) records dataset counts, label versions, checkpoint metrics, inference runs and test results. Detailed assets are stored locally at:

- `sss_contact_yolov8_spatial_split_bundle/metadata/split_manifest.json`
- Each review platform's `exports/*_manifest.json`
- `sss_contact_yolo26/retrain/sss_contact_yolo26_retrain_A_round2_v001/sss_contact_yolo26_spatial_split_A_round2_v001.zip`, containing `metadata/provenance.json` under its bundle root
- The current checkpoint's `train_metrics`
- The inference application's `outputs/latest/` and `outputs/uploaded/*/latest/`
- `sss_contact_yolo26/experiments/*/manifest.json`
