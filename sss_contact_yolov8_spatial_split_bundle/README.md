# SSS YOLOv8 spatial split bundle

This bundle contains 150x150 single-channel PNG patches and one-class YOLO labels.

## Split

- train: spatial chunks 1-7
- validation: spatial chunks 8-9
- test: spatial chunks 10-11
- buffer: patches within 15 m of a cross-split footprint, reserved for spatial separation

Group numbers record randomized annotation order. `spatial_chunk` records geographic order.

Group mapping: group_01 -> chunk 3, group_02 -> chunk 6, group_03 -> chunk 5, group_04 -> chunk 8, group_05 -> chunk 2, group_06 -> chunk 7, group_07 -> chunk 9, group_08 -> chunk 10, group_09 -> chunk 1, group_10 -> chunk 4, group_11 -> chunk 11

`dataset.yaml` is ready for Ultralytics. The original AnyLabeling JSON sidecars are retained under `metadata/original_anylabeling_json/` for provenance.
