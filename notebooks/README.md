# Notebook templates

Colab templates for YOLO26 training and spatial cross-validation.

| Notebook | Source workflow |
|---|---|
| `train_round2.ipynb` | YOLO26n training with `A_round2_labels_v001`, 2× positive-patch oversampling and validation on spatial chunks 8–9. |
| `spatial_cv_round2.ipynb` | Spatial cross-validation using the round-1 `A_v002` labels to generate candidates for the second human review. |

The names refer to different stages: the CV notebook generates the second review round; the training notebook consumes that round's final labels. Both require their corresponding data ZIP on Google Drive. Inspect the path settings in the first setup cells before running.

The templates use English documentation and start with empty outputs.
