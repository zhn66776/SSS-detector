# Single-channel SSS YOLOv8 Colab bundle

This ZIP contains the derived 150x150 grayscale SSS patches, one-class YOLO
labels, the four group patch indexes, and the small helper module used by
`sss_contact_yolov8_single_channel.ipynb`.

The fixed split is deliberately spatial/group based:

- train: `group_02` + `group_03`
- validation: `group_04`
- blind test: `group_01`

The class is `contact` only. `channels: 1` makes Ultralytics load the PNGs as
grayscale and constructs the detector with one input channel. The notebook
uses a small YOLOv8n detector, weight decay, early stopping, and controlled
augmentation. Test results must not be used to select thresholds or tune
hyperparameters.

The labels are derived manual contact annotations in this project; they are
not a claim that every contact is confirmed UXO ground truth. Check the source
data licence before redistributing the ZIP or a trained model.
