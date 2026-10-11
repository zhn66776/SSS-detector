# Running the local tools

Run the commands from the repository root and follow each section’s `cd` command.

## Local assets

Place the data and checkpoints in these directories before starting each application:

| Component | Required local assets |
|---|---|
| Inference map, built-in test region | `sss_contact_yolov8_spatial_split_bundle/data/`, `metadata/split_manifest.csv`, and `sss_contact_yolo26/models/yolo26n_A_round2_v001/best.pt` |
| Review platform | The corresponding review ZIP and backup-B directory beside the round's application; restore `working_A/` and history files to resume an existing review |
| Contact inspector | Prepared `sss_contact_inspector/data/`; Costa ECW decoding also requires the local compiled helper |
| MBES viewer | Prepared `sss_mbes_sediment_viewer/derived/`; SSS layers depend on the contact inspector's ECW service |
| SAS viewer | The separate SAS GeoTIFF configured in `sas_native_viewer/render.py` |

To build the Costa decoder, place the ECW SDK source at `sss_contact_inspector/vendor-src/libecw` and run `sss_contact_inspector/native/rebuild_ecw_stack.sh`. The SDK source and compiled libraries are local dependencies.

Set the path constants in the preparation and exploration scripts to match your data locations. Keep original archives and annotation backups read-only.

## Inference map

The recorded inference environment uses Ultralytics 8.4.160, PyTorch 2.8, NumPy 1.26.4 and OpenCV 4.11.0.86. Keep a separate environment for the inspector, whose requirements use NumPy 2.x.

```sh
cd sss_contact_yolo26/apps/sss_contact_yolo26_test_inference_map
.venv/bin/python serve_map.py --port 8770
```

Open `http://127.0.0.1:8770/`. The default weights and data paths are relative to the project. Use `--weights`, `--bundle` and `--outputs` to override them. Check `serve_map.py --help` for the supported options.

To reuse an existing Python installation containing PyTorch, NumPy, OpenCV, Pillow and tifffile, create the application environment with:

```sh
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install 'ultralytics==8.4.160'
```

`--system-site-packages` reuses packages from that Python installation. Create a new virtual environment when moving the application to another machine.

## Review platforms

```sh
cd sss_contact_yolo26/label_review/round1/sss_contact_yolo26_spatial_cv_review_20260922T234500Z_annotation_tool
python3 serve.py --host 127.0.0.1 --port 8767
```

```sh
cd sss_contact_yolo26/label_review/round2/sss_contact_yolo26_spatial_cv_review_A_v002_20260923T111854Z_annotation_tool
python3 serve.py --host 127.0.0.1 --port 8768
```

Set the review ZIP and backup labels with `--zip` and `--backup-b`. Run the matching and label-format tests from each review platform directory:

```sh
python3 -m unittest discover -s tests -v
```

## Inspection tools

```sh
cd sss_contact_inspector
python3 serve.py --port 8766
```

```sh
cd sss_mbes_sediment_viewer
python3 serve.py --port 8091
```

The MBES viewer starts the SSS service on port 8090 if needed. Its build script requires the original local bathymetry and GIS packages. The SAS viewer currently uses port 8767, which conflicts with the default round-1 review port; run one at a time or change the binding in the SAS script.

## Training notebooks

Open the [notebook templates](../notebooks/README.md) in Colab and place the corresponding data ZIP on Google Drive.

The training template evaluates validation chunks 8–9. Spatial CV audits chunks 1–9, using the geometry of test chunks 10–11 to calculate buffer distances. Use validation results to choose model settings, then evaluate the reviewed test set.

## Test results

On 9 October 2026, the existing suites returned 50 passes and 2 failures across 52 tests. The failures are confined to the contact inspector:

- `test_metadata_and_contact_counts`: the test expects 20,983 records; the current contact file has 20,985 after the GIS migration added S3824 and S3825. Preparation metadata also predates those additions.
- `test_original_archives_match_preparation_signature`: original ZIPs were not found at the recorded external-drive paths.

The MBES viewer’s 16 tests and both review platforms’ 13 tests each passed.
