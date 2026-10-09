# Costa Head SSS contact Colab bundle

This derived bundle includes group_01..group_04 native 150x150 grayscale PNGs, converted one-class YOLO labels, the original AnyLabeling JSON where present, a Colab L4 notebook, and the ONNX inference/evaluation helper.

The original project images and labels were not modified. The notebook defaults to group_02 and group_03 because they contain the most contact boxes.

The selected checkpoint is SonarVision YOLOv8-ESI v6. Only its `unknown_debris` class is retained and emitted as `contact`; other model classes are discarded. This is a zero-shot SSS baseline, not a fine-tuned Costa Head model.

The 3,722 images without AnyLabeling JSON are packaged with empty YOLO files according to the user's confirmed-negative summary. The manifest records that this negative status is not represented by a per-image source sidecar.
