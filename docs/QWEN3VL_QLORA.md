# Qwen3-VL QLoRA sonar-contact experiment

The [completed notebook](../notebooks/qwen3vl_sss_qlora_training_batch_done.ipynb) records a 16 October 2026 Colab run. It is published unchanged, with saved training logs, loss curves and validation detection examples. The [machine-readable evidence](evidence/qwen3vl-qlora-2026-10-16.json) was extracted from those outputs; publication did not rerun training.

## Run in Colab

1. [Open the notebook in Colab](https://colab.research.google.com/github/zhn66776/SSS-detector/blob/main/notebooks/qwen3vl_sss_qlora_training_batch_done.ipynb) and save a copy to Drive.
2. Select a CUDA GPU runtime. The recorded run used an NVIDIA L4 with about 23 GB reported memory. Check current Colab availability and usage costs in the Resources panel.
3. Place `sss_contact_yolo26_spatial_split_A_round2_v001.zip` in My Drive, or set `DATA_ZIP` to its location. The ZIP must contain `split_manifest_A_round2_v001.csv` and the matching image/label structure. Source imagery, dataset ZIPs and model weights are not included in this repository; obtain imagery through the Marine Data Exchange and prepare the round-2 bundle using the project's annotation workflow.
4. Review `DATA_ZIP`, `PILOT_ZIP` and `OUTPUT_ROOT` in Configuration. `PILOT_ZIP` is an optional fallback pointing to the author's earlier 64-image pilot; it is not supplied here. If no full bundle or pilot is found, the notebook offers a ZIP upload. Confirm `DATA MODE: full_round2_bundle` before comparing results with this page.
5. Run the cells in order. The notebook mounts Drive, installs its pinned Python dependencies, downloads the pinned base-model revision and writes each run to a fresh timestamped directory. The GPU runtime supplies PyTorch; package versions and `pip freeze` are saved with each run.

The recorded configuration uses `RUN_BASELINE=True`, `RUN_FINAL_TEST=False` and `AUTO_DISCONNECT=False`. The three-hour limit covers the training loop, not setup and all evaluation time. Rerunning consumes GPU time and creates new output files.

## Model and training

| Setting | Recorded value |
|---|---|
| Model | `Qwen/Qwen3-VL-4B-Instruct` |
| Model revision | `ebb281ec70b05090aa6165b016eac8ec08e71b17` |
| Quantization | 4-bit NF4, double quantization |
| Adapted modules | Language-model attention `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Vision encoder | Frozen |
| LoRA rank / alpha / dropout | 32 / 64 / 0.05 |
| Trainable parameters | 23,592,960 (0.5288% of the reported total) |
| Optimizer / learning rate | AdamW / 0.00005 |
| Epochs / seed | 5 / 42 |
| Microbatch / gradient accumulation | 2 / 2; effective batch 4 images |
| Processor image pixel budget | `min_pixels=max_pixels=256*256` |
| Output | One `contact` class, JSON boxes with coordinates normalized to 0–1000 |
| Checkpoint selection | Lowest validation assistant-token cross-entropy, epoch 3 |

Dependencies installed by the notebook include Transformers 4.57.6, PEFT 0.18.1, Accelerate 1.15.0, bitsandbytes 0.49.2 and huggingface_hub 0.36.2. Other dependencies are not all pinned; the saved per-run version records are needed to reconstruct the full environment.

## Data used in the saved run

| Split | Included positive patches | Annotation boxes |
|---|---:|---:|
| Training | 1,732 | 2,528 |
| Validation | 276 | 372 |
| Available test subset, not evaluated in this notebook | 125 | 167 |

The underlying bundle uses the [geographic split](RESULTS.md#spatial-split). The notebook checks a minimum 15 m train/validation footprint gap; the included positive subsets had an observed minimum gap of 35 m. It excluded 5,115 training, 1,420 validation and 1,480 test patches with unreviewed empty labels, plus the 608 buffer patches. No reviewed negatives were supplied. An optional `REVIEWED_NEGATIVES_CSV` can include audited negatives using `patch_id` and `split` columns.

## Saved validation results

Both models were evaluated on the same 276 positive validation patches with 372 existing annotation boxes. Matching is one-to-one, maximizing match count and then total IoU. At IoU ≥ 0.5:

| Metric | Quantized base model | Best QLoRA adapter |
|---|---:|---:|
| Matched boxes (TP) | 1 | 205 |
| Unmatched predictions (FP relative to existing labels) | 11 | 146 |
| Missed annotation boxes (FN) | 371 | 167 |
| Precision | 8.33% | 58.40% |
| Recall | 0.27% | 55.11% |
| F1 | 0.52% | 56.71% |
| Invalid JSON images | 0 | 0 |

Validation loss fell from 1.90758 before training to 0.87372 at epoch 3, then rose to 0.88773 by epoch 5. Training-loss curves use a fixed 256-image subset in evaluation mode; online training loss is logged separately. The selected adapter's F1 was 67.50% at IoU ≥ 0.25 and 22.13% at IoU ≥ 0.75, indicating sensitivity to localization strictness.

## Interpretation and limits

- These are validation results used during development and checkpoint selection. `RUN_FINAL_TEST=False` means this notebook contains no final test evaluation. Later FP16 base-versus-adapter testing is a separate experiment and is not reproduced by these saved outputs.
- Unreviewed empty labels are not verified background. With no reviewed negatives, true negatives and background false-positive rates cannot be evaluated. The positive-only training setup also leaves background rejection untested.
- Existing annotations may omit contacts, so an unmatched prediction is not automatically a confirmed false alarm.
- Adjacent patches overlap within each split. Counts are per-patch instances, not unique survey contacts.
- Generated coordinates have no detection confidence scores; no AP/mAP is claimed. The YOLO metrics elsewhere use a different protocol and cannot establish which model is better in a controlled comparison.
- This is a single recorded seed/run. The notebook does not evaluate natural-language descriptions or establish production readiness. The QLoRA model is separate from the existing YOLO map-inference application.

## Outputs

Each run saves configuration, environment versions, dataset summaries and selected manifests; epoch and best adapters with processor files; loss histories and plots; baseline/adapter validation predictions and metrics; TP/FP/FN example galleries; and `complete.json`. A TN placeholder explains when confirmed negatives are unavailable. Adapter weights, source data and raw per-image prediction files remain in the run's Drive directory, rather than in this repository.
