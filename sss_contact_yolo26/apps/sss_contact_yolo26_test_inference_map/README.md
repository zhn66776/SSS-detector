# Local SSS inference map

A localhost application for single-channel YOLO contact detection on the built-in test region, uploaded sonar images and patch folders. It displays live progress, original patch pixels, a tiled map, merged detection boxes and coordinate exports.

## Start

Prepare the assets and environment in the [setup guide](../../../docs/RUNNING.md), then run from this directory:

```sh
.venv/bin/python serve_map.py --port 8770
```

Open `http://127.0.0.1:8770/`. The server binds to localhost. Default paths resolve relative to this application:

- Dataset: `../../../sss_contact_yolov8_spatial_split_bundle`
- Weights: `../../models/yolo26n_A_round2_v001/best.pt`
- Output: `outputs/`

Override these with `--bundle`, `--weights` and `--outputs`. Other options include `--device` and `--delay`. The default CPU setting was faster than MPS for this small model in the recorded M3 comparison.

## Environment

Use the existing application environment with Ultralytics 8.4.160. Historical validation reproduced approximately 0.665 mAP50 and 0.386 mAP50–95 on CPU. The older system Ultralytics 8.3.189 produced inconsistent decoding for this checkpoint; the application checks for version 8.4 or later.

Inference sets `YOLO_OFFLINE=true`. The application requests an app-local Ultralytics settings location. A moved virtual environment should be recreated. See the repository's [running notes](../../../docs/RUNNING.md) for asset and environment requirements.

## Image inputs

Accepted inputs are 8-bit grayscale PNG or TIFF images at 0.1 m/pixel. Equal RGB channels may be converted to grayscale. An opacity/alpha band or a nodata value can identify invalid pixels. Input preparation preserves the source grayscale values and map pixel spacing.

GeoTIFF metadata supplies pixel spacing and map coordinates. Rotated grids, geographic coordinates, incompatible spacing, color images and unsupported pixel types are rejected. Images without usable georeferencing require the user to confirm the 0.1 m/pixel resolution; their exported detections have pixel coordinates only.

Supported image size: 150 × 150 pixels to one billion pixels. Loading and PNG export use full image arrays, with memory use scaling with image area.

Images are cut into 150 × 150 windows with a 100-pixel stride. Edge windows end at the image boundary. Fully invalid windows are excluded from inference. Uploads stay on the localhost server under `uploads/`; they are reused when identical files are uploaded again.

## Patch folders

A folder or multiple selected files can supply existing 150 × 150, 8-bit grayscale patches. Invalid files are skipped and reported. Each patch is passed directly to inference.

Filenames ending in `_x{X}_y{Y}` provide mosaic pixel positions. Known Costa bundle names also obtain the bundle's UTM placement and original labels. Files without spatial names are arranged in a gallery and require resolution confirmation.

## Weights

The application accepts trusted `.pt` detection checkpoints with a single input channel. It displays class names and training metadata, and uses the checkpoint's image size when recorded, otherwise 256. Load checkpoints from trusted sources: PyTorch loading can execute serialized code.

The application restores saved results for the selected dataset and model.

## Geometry and outputs

Costa bundle pixels use 0.1 m spacing and the source mosaic origin recorded in its manifest. Each patch spans 15 m, with 5 m overlap. The stitched mosaic uses the patch whose centre is nearest to each pixel. Detections from overlapping patches are merged at IoU 0.3 using a spatial index.

| File | Content |
|---|---|
| `test_mosaic_gray.png` | Full-resolution grayscale image or stitched mosaic. |
| `test_mosaic_contacts.png` | Full-resolution image with merged contact boxes. |
| `detections_per_patch.csv` | Per-patch predictions at confidence 0.05 or above. |
| `contacts.csv` | Merged contacts with confidence, box dimensions, supporting patches and map coordinates where available. |
| `results.json` | Per-patch predictions used to restore the map. |
| `run_manifest.json` | Model, input, thresholds, counts and timestamps. |

Built-in results are saved to `outputs/latest/`; uploaded datasets use `outputs/uploaded/<dataset>/latest/`. A run writes to a staging folder and replaces the previous results after completion. The interface's result-removal action is limited to known application output names and is disabled during inference.

Changing the display threshold redraws the map immediately. Saved contact exports use the threshold selected when the run started. The map displays the original test annotations for visual review. Quantitative evaluation is handled in the notebooks.

## Recorded scale

The saved 2 km × 2 km Costa run contains 40,000 inferred patches, 5,912 merged predictions at confidence 0.5 and 460 seconds between its start and finish timestamps. The [experiment record](../../../docs/RESULTS.md#regional-inference) also describes the synthetic input used for browser performance testing.

The browser renders visible map tiles and boxes on canvas. The finest map level preserves source pixels, while lower levels are downsampled for display. Loading and full-resolution output generation use full image arrays.
