# Running the local tools

Commands below start from the repository root unless a `cd` command says otherwise. They describe the existing workflows; installation in a clean environment was not exercised during the initial repository import.

## Restore assets first

A source-only clone does not contain the survey data, annotation state or checkpoint.

| Component | Required local assets |
|---|---|
| Inference map, built-in test region | `sss_contact_yolov8_spatial_split_bundle/data/`, `metadata/split_manifest.csv`, and `sss_contact_yolo26/models/yolo26n_A_round2_v001/best.pt` |
| Review platform | The corresponding review ZIP and backup-B directory beside the round's application; restore `working_A/` and history files to resume an existing review |
| Contact inspector | Prepared `sss_contact_inspector/data/`; Costa ECW decoding also requires the local compiled helper |
| MBES viewer | Prepared `sss_mbes_sediment_viewer/derived/`; SSS layers depend on the contact inspector's ECW service |
| SAS viewer | The separate SAS GeoTIFF configured in `sas_native_viewer/render.py` |

The ECW SDK source and compiled libraries remain in the existing local workspace. They are excluded from this repository. `native/rebuild_ecw_stack.sh` expects that source at `sss_contact_inspector/vendor-src/libecw`; the script does not download it.

Several earlier preparation and exploration scripts still use machine-specific paths. Read their path constants before running them on a new machine. Keep original archives and annotation backups read-only.

## Inference map

The existing application has its own `.venv`, with Ultralytics 8.4.160. Its saved environment also used PyTorch 2.8, NumPy 1.26.4 and OpenCV 4.11.0.86. The inspector's requirements target a different NumPy range, so keep these environments separate.

```sh
cd sss_contact_yolo26/apps/sss_contact_yolo26_test_inference_map
.venv/bin/python serve_map.py --port 8770
```

Open `http://127.0.0.1:8770/`. The default weights and data paths are relative to the project. Use `--weights`, `--bundle` and `--outputs` to override them. Check `serve_map.py --help` for the supported options.

Existing environment directories should be recreated when moving to another machine. The historical application setup used:

```sh
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install 'ultralytics==8.4.160'
```

This command assumes suitable system packages already exist; it is not a fully specified clean-machine installation. A packaged dependency definition remains future work.

## Review platforms

```sh
cd sss_contact_yolo26/label_review/round1/sss_contact_yolo26_spatial_cv_review_20260922T234500Z_annotation_tool
python3 serve.py --host 127.0.0.1 --port 8767
```

```sh
cd sss_contact_yolo26/label_review/round2/sss_contact_yolo26_spatial_cv_review_A_v002_20260923T111854Z_annotation_tool
python3 serve.py --host 127.0.0.1 --port 8768
```

Both applications provide `--zip` and `--backup-b` overrides. Their source ZIP and backup labels must be available before startup. Matching and label-format unit tests run without those survey assets:

```sh
python3 -m unittest discover -s tests -v
```

Run this command from each review platform directory.

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

Use the English templates under [`../notebooks/`](../notebooks/README.md) in Colab with the corresponding data ZIP restored to Google Drive. Templates have no saved outputs or execution state. Existing local notebooks with historical outputs remain outside Git.

The training template runs validation. Reserve test metrics for a reviewed evaluation set and a fixed evaluation procedure. The spatial-CV template audits chunks 1–9; it does not train on the held-out test chunks.

## Test results at import

On 9 October 2026, the existing suites returned 50 passes and 2 failures across 52 tests. The failures are confined to the contact inspector:

- `test_metadata_and_contact_counts`: the test expects 20,983 records; the current contact file has 20,985 after the GIS migration added S3824 and S3825. Preparation metadata also predates those additions.
- `test_original_archives_match_preparation_signature`: original ZIPs were not found at the recorded external-drive paths.

The MBES viewer's 16 tests and both review platforms' 13 tests each passed. No test expectations or application behavior were changed to make this initial snapshot appear fully passing.
