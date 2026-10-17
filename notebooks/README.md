# Notebooks

Colab templates for YOLO26 training and spatial cross-validation, plus a completed Qwen3-VL QLoRA training run.

| Notebook | Source workflow |
|---|---|
| `train_round2.ipynb` | YOLO26n training with `A_round2_labels_v001`, 2× positive-patch oversampling and validation on spatial chunks 8–9. |
| `spatial_cv_round2.ipynb` | Spatial cross-validation using the round-1 `A_v002` labels to generate candidates for the second human review. |

The names refer to different stages: the CV notebook generates the second review round; the training notebook consumes that round's final labels. Both require their corresponding data ZIP on Google Drive. Inspect the path settings in the first setup cells before running.

The two YOLO templates use English documentation and start with empty outputs.

## Completed Qwen3-VL QLoRA run

[`qwen3vl_sss_qlora_training_batch_done.ipynb`](qwen3vl_sss_qlora_training_batch_done.ipynb) preserves the uploaded 16 October 2026 training notebook, including outputs, loss curves and validation detection examples. It fine-tunes Qwen3-VL-4B-Instruct with 4-bit QLoRA; the best of five epochs is selected by validation loss.

[Open in Colab](https://colab.research.google.com/github/zhn66776/SSS-detector/blob/main/notebooks/qwen3vl_sss_qlora_training_batch_done.ipynb). Save a copy before running, select a GPU runtime, provide the dataset ZIP on Google Drive and check the configuration paths. The recorded run has no reviewed negative images and does not evaluate the test split. See [setup and results](../docs/QWEN3VL_QLORA.md) for the requirements, metrics and limitations.
