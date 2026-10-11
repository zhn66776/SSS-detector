# Costa Head SSS contact Colab bundle

The local data bundle includes group_01..group_04 native 150x150 grayscale PNGs, converted one-class YOLO labels, the original AnyLabeling JSON where present, a Colab L4 notebook, and the ONNX inference/evaluation helper.

The notebook defaults to group_02 and group_03, which contain the most contact boxes.

The zero-shot comparison uses the SonarVision YOLOv8-ESI v6 checkpoint and maps its `unknown_debris` class to `contact`.

The bundle uses empty YOLO files for 3,722 negative images. Their negative status comes from a batch-level human review summary, recorded in the manifest.
