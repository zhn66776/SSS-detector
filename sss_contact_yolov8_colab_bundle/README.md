# Single-channel SSS YOLOv8 Colab bundle

This ZIP contains the derived 150x150 grayscale SSS patches, one-class YOLO
labels, the four group patch indexes, and the small helper module used by
`sss_contact_yolov8_single_channel.ipynb`.

The historical experiment uses this group split:

- train: `group_02` + `group_03`
- validation: `group_04`
- blind test: `group_01`

The detector uses one class, `contact`. `channels: 1` makes Ultralytics load the PNGs as
grayscale and constructs the detector with one input channel. The notebook
uses a small YOLOv8n detector, weight decay, early stopping, and controlled
augmentation. Select thresholds and hyperparameters on validation data, then evaluate the test split using those settings.

Labels represent manually annotated sonar contacts. Obtain source imagery from the Marine Data Exchange and keep the survey data in your local dataset directory.
