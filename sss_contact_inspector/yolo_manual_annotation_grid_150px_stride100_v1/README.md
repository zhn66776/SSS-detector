# Costa Head SSS manual annotation grid

This directory contains an unlabelled, raw single-channel grayscale candidate
set for manual object-detection annotation. It is not a YOLO training dataset:
all YOLO bounding boxes and class labels are still to be drawn and reviewed by
a human annotator. No YOLO `.txt` labels are included here.

The images are cut from the Costa Head ECW at native 0.1 m/px resolution using
a global raster-aligned grid. Each image is 150 x 150 pixels (15 x 15 m), the
grid stride is 100 pixels (10 m), and the overlap is 50 pixels (33.333%). The
global grid origin is raster pixel (0, 0); a patch at `(100*k, 100*j)` has its
recorded centre at `(100*k+75, 100*j+75)`.

Only patch centres strictly inside the supplied train or validation
patch-centre polygon were exported. Windows that are outside the ECW or fail
the project-local ECW window decoder are recorded in `skipped_windows.csv`.
No intensity statistic, near-white rule, contact list, CSV, DBF, shapefile,
or existing label was used to select or label an image. The raw PNGs contain
no circles, text, IDs, watermarks, bounding boxes, or other annotations.

`patch_index.csv` records the raster window, centre and corner coordinates,
split, relative image path, and decode status. `split_validation.json` records
the exhaustive cross-split footprint-spacing check. `dataset_manifest.json`
records the input hashes, raster registration, grid rules, counts and byte
totals.

Do not train a YOLO model directly from this directory until manual labels have
been created and independently checked.
