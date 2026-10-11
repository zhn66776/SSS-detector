# Costa Head SSS manual annotation grid

This directory holds unlabelled, single-channel grayscale patches for manual object-detection annotation. Draw and review YOLO bounding boxes and class labels before using the patches for training.

The images are cut from the Costa Head ECW at native 0.1 m/px resolution using
a global raster-aligned grid. Each image is 150 x 150 pixels (15 x 15 m), the
grid stride is 100 pixels (10 m), and the overlap is 50 pixels (33.333%). The
global grid origin is raster pixel (0, 0); a patch at `(100*k, 100*j)` has its
recorded centre at `(100*k+75, 100*j+75)`.

Exported patch centres lie strictly inside the supplied train or validation patch-centre polygon. Windows that are outside the ECW or fail
the project-local ECW window decoder are recorded in `skipped_windows.csv`.
Patch selection uses the grid and polygon geometry. The PNG files contain the unannotated source pixels.

`patch_index.csv` records the raster window, centre and corner coordinates,
split, relative image path, and decode status. `split_validation.json` records
the exhaustive cross-split footprint-spacing check. `dataset_manifest.json`
records the input hashes, raster registration, grid rules, counts and byte
totals.
