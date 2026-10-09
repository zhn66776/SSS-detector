#!/usr/bin/env python3
"""Live test-area inference map for the SSS contact YOLO26 model (local, offline).

    .venv/bin/python serve_map.py            (defaults: project-relative bundle and weights)
    then open http://127.0.0.1:8770/ and press "Start inference"

Test area = spatial chunks 10-11 (test split; optionally the chunk-10 buffer patches).
Patch IDs are pixel coordinates of the source mosaic (0.1 m/px, north-up), so every
150x150 patch is placed at (x_id, y_id) on the map. Nothing outside this folder is written.

The page can also run on an uploaded 8-bit grayscale sonar image (TIFF/GeoTIFF or PNG, 0.1 m/px),
cut into 150x150 patches every 100 px like the training patches, on an uploaded folder of 150x150 patches,
and with uploaded .pt weights.
Uploads are kept in uploads/ inside this folder.
"""
import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("YOLO_OFFLINE", "true")      # no update checks / telemetry: inference is fully local
# keep Ultralytics' settings file inside this app instead of the user's global
# ~/Library/Application Support/Ultralytics (which the system Ultralytics 8.3 also reads)
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(HERE, ".ultralytics"))
# project-relative defaults: <repo>/sss_contact_yolo26/apps/<this app>/  (override with --bundle / --weights)
PROJECT = os.path.dirname(os.path.dirname(HERE))                  # .../sss_contact_yolo26
REPO_ROOT = os.path.dirname(PROJECT)                              # .../IRP-external
BUNDLE = os.path.join(REPO_ROOT, "sss_contact_yolov8_spatial_split_bundle")
DEFAULT_WEIGHTS = os.path.join(PROJECT, "models", "yolo26n_A_round2_v001", "best.pt")
PATCH = 150
STRIDE = 100               # training patches: 150 px window every 100 px, i.e. 50 px overlap
M_PER_PX = 0.1
PRED_FLOOR = 0.05          # everything >= this is stored; the UI threshold filters live
MERGE_IOU = 0.30           # same contact seen in overlapping patches -> one map contact
MERGE_CELL = 64            # px; spatial index for the merge (only boxes sharing a cell can overlap)
BATCH = 8
IMGSZ = 256                # training imgsz; an uploaded checkpoint's own train imgsz is used when it has one
OUTPUT_FILES = ("test_mosaic_gray.png", "test_mosaic_contacts.png", "detections_per_patch.csv", "contacts.csv",
                "results.json", "run_manifest.json")
# every result file name this app has ever written (current + earlier per-run folders); nothing else is deleted
RESULT_FILE_RE = re.compile(r"^(test_mosaic_gray\.png|test_mosaic_contacts\.png|test_mosaic_contacts_conf[0-9.]+\.jpg|"
                            r"overview\.png|detections_per_patch\.csv|contacts\.csv|contacts_conf[0-9.]+\.csv|"
                            r"results\.json|run_manifest\.json)$")
RUN_DIR_RE = re.compile(r"^(latest|\d{8}T\d{6}Z|\.writing_\d{8}T\d{6}Z)$")

# uploads (images and weights), each in its own folder named after the file's SHA-256
UPLOADS = os.path.join(HERE, "uploads")
IMAGES_DIR = os.path.join(UPLOADS, "images")
MODELS_DIR = os.path.join(UPLOADS, "models")
PATCHSETS_DIR = os.path.join(UPLOADS, "patchsets")
IMAGE_KEY_RE = re.compile(r"^img_[0-9a-f]{12}$")
MODEL_KEY_RE = re.compile(r"^pt_[0-9a-f]{12}$")
PATCHSET_KEY_RE = re.compile(r"^ps_[0-9a-f]{12}$")
IMAGE_EXT = (".tif", ".tiff", ".png")
MAX_PATCH_FILES = 100_000
POS_RE = re.compile(r"_x(\d+)_y(\d+)$")                  # patch file name ends in its mosaic pixel position
BUNDLE_ID_RE = re.compile(r"^costa_val_x\d+_y\d+$")      # CES-86 mosaic pixels: UTM by the bundle formula
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")
GALLERY_GAP = 10           # px between patches laid out by file name when they carry no position
MAX_IMAGE_BYTES = 4 << 30
MAX_WEIGHTS_BYTES = 1 << 30
MAX_PIXELS = 1_000_000_000  # the map shows tiles, so this bounds server memory (checks, contacts PNG), not display
RES_TOL = 0.001            # GeoTIFF pixel size must be 0.1 m +- 1 mm: first version does no resampling
BUSY = ("loading", "running", "mosaic")
TILE_PX = 512              # map background tiles; level z is the image downsampled 2**z times


def reduce2(a, rows=2048):
    """2x2 box average (sizes rounded up, odd edges repeated), a few thousand rows at a time: a memory-mapped
    full-resolution image is never loaded whole."""
    H, W = a.shape
    out = np.empty(((H + 1) // 2, (W + 1) // 2), np.uint8)
    for y0 in range(0, H, rows):                         # rows is even, so every block starts on an even row
        b = np.asarray(a[y0:y0 + rows], dtype=np.uint16)
        if b.shape[0] % 2:
            b = np.vstack([b, b[-1:]])
        if W % 2:
            b = np.hstack([b, b[:, -1:]])
        out[y0 // 2:y0 // 2 + b.shape[0] // 2] = (b[0::2, 0::2] + b[1::2, 0::2] + b[0::2, 1::2] + b[1::2, 1::2] + 2) // 4
    return out


class Pyramid:
    """Factor-2 downsampled copies of a gray image, cut into TILE_PX tiles, so the page only draws the tiles on screen
    at a resolution that fits the zoom instead of one huge image. Level 0 is the image itself (a memory map for
    uploads), so its tiles are exact pixels and it is not copied."""

    def __init__(self, a):
        self.levels = [a]
        while max(self.levels[-1].shape) > TILE_PX:
            self.levels.append(reduce2(self.levels[-1]))

    def tile(self, z, tx, ty):
        from PIL import Image
        a = self.levels[z]
        t = a[ty * TILE_PX:(ty + 1) * TILE_PX, tx * TILE_PX:(tx + 1) * TILE_PX]
        if not t.size:
            raise ValueError("tile outside the image")
        buf = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(t)).save(buf, "PNG", compress_level=1)     # lossless
        return buf.getvalue()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_patches():
    rows = [r for r in csv.DictReader(open(os.path.join(BUNDLE, "metadata/split_manifest.csv")))
            if int(r["spatial_chunk"]) in (10, 11)]
    patches = []
    for r in rows:
        m = re.search(r"_x(\d+)_y(\d+)$", r["patch_id"])
        lab = os.path.join(BUNDLE, "data/labels", r["final_split"], r["patch_id"] + ".txt")
        gt = [list(map(float, l.split()[1:])) for l in open(lab) if l.strip()]
        patches.append({"id": r["patch_id"], "chunk": int(r["spatial_chunk"]), "group": r["original_group"],
                        "split": r["final_split"], "infer": r["final_split"] == "test",
                        "x": int(m.group(1)), "y": int(m.group(2)),
                        "x0": float(r["x0"]), "y1": float(r["y1"]), "gt": gt,
                        "image": os.path.join(BUNDLE, "data/images", r["final_split"], r["patch_id"] + ".png")})
    # UTM of pixel (x_id, y_id): easting = E0 + 0.1*x, northing = N0 - 0.1*y  (checked against every row)
    e0 = {round(p["x0"] - M_PER_PX * p["x"], 3) for p in patches}
    n0 = {round(p["y1"] + M_PER_PX * p["y"], 3) for p in patches}
    assert len(e0) == 1 and len(n0) == 1, (e0, n0)
    patches.sort(key=lambda p: (p["y"], p["x"]))       # north-to-south, west-to-east scan
    return patches, e0.pop(), n0.pop()


_BUNDLE_INDEX = None


def bundle_index():
    """patch_id -> manifest row, over all splits of the bundle (cached)."""
    global _BUNDLE_INDEX
    if _BUNDLE_INDEX is None:
        _BUNDLE_INDEX = {r["patch_id"]: r for r in csv.DictReader(open(os.path.join(BUNDLE, "metadata/split_manifest.csv")))}
    return _BUNDLE_INDEX


def bundle_utm0():
    """UTM of pixel (0, 0) of the CES-86 mosaic; one value for every manifest row (checked for all 10,756)."""
    r = next(iter(bundle_index().values()))
    m = POS_RE.search(r["patch_id"])
    return round(float(r["x0"]) - M_PER_PX * int(m.group(1)), 3), round(float(r["y1"]) + M_PER_PX * int(m.group(2)), 3)


def bundle_gt(r):
    lab = os.path.join(BUNDLE, "data/labels", r["final_split"], r["patch_id"] + ".txt")
    return [list(map(float, l.split()[1:])) for l in open(lab) if l.strip()] if os.path.exists(lab) else []


def stitch(tiles, xmin, ymin, W, H, core):
    """tiles: [(x, y, 150x150 L image)]. With core=True (overlapping patches) every pixel comes from the patch whose
    centre is nearest: pass 1 pastes whole patches (outer rim), pass 2 each patch's central 100x100 px."""
    from PIL import Image
    mosaic = Image.new("L", (W, H), 60)
    for x, y, t in tiles:
        mosaic.paste(t, (x - xmin, y - ymin))
    if core:
        for x, y, t in tiles:
            mosaic.paste(t.crop((25, 25, 125, 125)), (x - xmin + 25, y - ymin + 25))
    return mosaic


def box_iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    i = w * h
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def grid_starts(n):
    """Patch origins along one axis: every STRIDE px, plus one flush with the far edge so the whole image is covered."""
    s = list(range(0, n - PATCH + 1, STRIDE))
    if s[-1] != n - PATCH:
        s.append(n - PATCH)
    return s


def check_ultralytics(allow_old):
    import ultralytics
    ver = tuple(int(x) for x in ultralytics.__version__.split(".")[:2])
    if ver < (8, 4) and not allow_old:
        raise RuntimeError("Ultralytics %s cannot decode YOLO26 correctly (val mAP50-95 0.247 vs 0.387); "
                           "run this server with ultralytics 8.4.x" % ultralytics.__version__)
    return ultralytics.__version__


# ==================================================================== sources
class BundleSource:
    """Test area of the spatial split bundle: the exported chunk 10-11 patches, with labels and the chunk-10 buffer."""
    key = "bundle"
    kind = "bundle"
    name = "测试区 · spatial chunks 10–11"

    def __init__(self, outputs):
        self.patches, self.E0, self.N0 = load_patches()
        self.out_root = outputs
        self.info = {"key": self.key, "kind": self.kind, "name": self.name, "georef": True}

    def describe(self):
        return {"key": self.key, "bundle": BUNDLE}

    def model_input(self, p):
        return p["image"]

    def patch_png(self, p):
        return open(p["image"], "rb").read()

    def mosaic(self, ids, xmin, ymin, W, H):
        # Neighbouring patches overlap by 50 px (5 m) and differ by a few grey levels there, so every mosaic
        # pixel comes from the patch whose centre is nearest. Full resolution (0.1 m/px); no resampling.
        from PIL import Image
        by_id = {p["id"]: p for p in self.patches}
        tiles = []
        for pid in ids:
            with Image.open(by_id[pid]["image"]) as im:
                tiles.append((by_id[pid]["x"], by_id[pid]["y"], im.convert("L")))
        return stitch(tiles, xmin, ymin, W, H, core=True)


class PatchSetSource:
    """A folder (or a multiple selection) of uploaded 150x150 patches.

    File names ending in _x{X}_y{Y} place each patch at those mosaic pixels (overlapping patches are stitched like
    the test area); bundle names (costa_val_x…_y…) also get UTM, and those listed in the bundle manifest get their
    split, chunk and labels. Otherwise the patches are laid out in a grid in file-name order. Every patch is inferred."""
    kind = "patches"

    def __init__(self, d, outputs):
        self.dir = d
        self.meta = meta = json.load(open(os.path.join(d, "meta.json")))
        self.key, self.name = meta["key"], meta["name"]
        self.stack = np.load(os.path.join(d, "patches.npy"), mmap_mode="r")
        geo = meta.get("georef")
        self.E0, self.N0 = (geo["E0"], geo["N0"]) if geo else (None, None)
        self.out_root = os.path.join(outputs, "uploaded", self.key)
        self.cell_inset = 25 if meta["layout"] == "coords" else 0
        idx = bundle_index() if meta.get("matched_bundle") else {}
        self.patches = []
        for i, f in enumerate(meta["files"]):
            r = idx.get(f["stem"])
            self.patches.append({"id": f["id"], "name": f["file"], "i": i, "x": f["x"], "y": f["y"], "infer": True,
                                 "split": r["final_split"] if r else "patch", "chunk": int(r["spatial_chunk"]) if r else "",
                                 "group": r["original_group"] if r else "", "gt": bundle_gt(r) if r else []})
        self.info = dict(patchset_public(meta), kind=self.kind, has_gt=bool(idx))

    def describe(self):
        return {k: self.meta.get(k) for k in ("key", "name", "sha256", "count", "layout", "georef", "res_confirmed",
                                               "matched_bundle", "skipped_count")}

    def model_input(self, p):
        return np.ascontiguousarray(self.stack[p["i"]])[..., None]

    def patch_png(self, p):
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(self.stack[p["i"]])).save(buf, "PNG")
        return buf.getvalue()

    def mosaic(self, ids, xmin, ymin, W, H):
        from PIL import Image
        by_id = {p["id"]: p for p in self.patches}
        tiles = [(by_id[pid]["x"], by_id[pid]["y"], Image.fromarray(np.ascontiguousarray(self.stack[by_id[pid]["i"]])))
                 for pid in ids]
        return stitch(tiles, xmin, ymin, W, H, core=self.cell_inset > 0)


class ImageSource:
    """An uploaded 8-bit grayscale image at 0.1 m/px, cut into 150 px patches every 100 px like the training patches.

    Patch IDs are x{X}_y{Y} pixel origins in the image. Patches with no valid pixel (opacity 0 / nodata,
    or all 0 when the image has neither) are marked "nodata" and not inferred."""
    kind = "image"

    def __init__(self, d, outputs):
        self.dir = d
        self.meta = json.load(open(os.path.join(d, "meta.json")))
        self.key, self.name = self.meta["key"], self.meta["name"]
        self.gray = np.load(os.path.join(d, "gray.npy"), mmap_mode="r")
        mpath = os.path.join(d, "mask.npy")
        valid = np.load(mpath, mmap_mode="r") if os.path.exists(mpath) else self.gray
        geo = self.meta.get("georef")
        self.E0, self.N0 = (geo["E0"], geo["N0"]) if geo else (None, None)
        self.out_root = os.path.join(outputs, "uploaded", self.key)
        H, W = self.gray.shape
        self.patches = []
        for y in grid_starts(H):
            for x in grid_starts(W):
                ok = bool(valid[y:y + PATCH, x:x + PATCH].any())
                self.patches.append({"id": "x%06d_y%06d" % (x, y), "x": x, "y": y, "split": "image" if ok else "nodata",
                                     "infer": ok, "chunk": "", "group": "", "gt": []})
        self.info = dict(image_public(self.meta), kind=self.kind,
                         patches_nodata=sum(p["split"] == "nodata" for p in self.patches))

    def describe(self):
        return {k: self.meta.get(k) for k in ("key", "name", "sha256", "size", "format", "bands", "opacity", "georef",
                                               "res_confirmed")}

    def model_input(self, p):
        # (H, W, 1) uint8 arrays give exactly the predictions of the same patch read from an L-mode PNG
        return np.ascontiguousarray(self.gray[p["y"]:p["y"] + PATCH, p["x"]:p["x"] + PATCH])[..., None]

    def patch_png(self, p):
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(self.gray[p["y"]:p["y"] + PATCH, p["x"]:p["x"] + PATCH])).save(buf, "PNG")
        return buf.getvalue()

    def mosaic(self, ids, xmin, ymin, W, H):
        from PIL import Image
        return Image.fromarray(np.asarray(self.gray))


# ==================================================================== uploads
class UploadError(ValueError):
    """The uploaded file is not usable; the message is shown on the page."""


class NeedsConfirm(Exception):
    """An image without georeferencing: the user has to confirm it is 0.1 m/px before it is used."""


def check_size(w, h):
    if w < PATCH or h < PATCH:
        raise UploadError("图像只有 %d×%d px，小于一个 patch（%d×%d）" % (w, h, PATCH, PATCH))
    if w * h > MAX_PIXELS:
        raise UploadError("图像 %d×%d px（%.0f 百万像素）超过上限 %.0f 百万像素：服务器处理时要在内存里保留几份整图，"
                          "请先裁成几块再分别上传" % (w, h, w * h / 1e6, MAX_PIXELS / 1e6))


PIL_MODE_MSG = {"1": "1 位黑白图像", "P": "调色板（palette）图像", "I;16": "16 位图像", "I;16B": "16 位图像",
                "I": "32 位整数图像", "F": "32 位浮点图像", "CMYK": "CMYK 图像"}


def read_pillow(path):
    """-> HxWxS uint8 bands (L, LA, RGB or RGBA)."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None               # size is checked here, before decoding
    try:
        im = Image.open(path)
    except Exception:  # noqa: BLE001 - any decoder failure means the file cannot be used
        raise UploadError("无法读取这个文件：不是有效的 PNG/TIFF，或者是不支持的压缩方式")
    with im:
        check_size(*im.size)
        if im.mode not in ("L", "LA", "RGB", "RGBA"):
            raise UploadError("这是%s，不是 8 位灰度。第一版不做灰度拉伸，只接受 8 位灰度声呐图"
                              % PIL_MODE_MSG.get(im.mode, "像素格式 %s 的图像" % im.mode))
        a = np.asarray(im)
    return a if a.ndim == 3 else a[..., None]


def tiff_georef(page):
    """Top-left pixel-corner UTM origin and pixel size from GeoTIFF tags, or None if the TIFF has none."""
    def val(code):
        t = page.tags.get(code)
        return t.value if t is not None else None
    keys = {}
    gk = val(34735)                             # GeoKeyDirectoryTag: header + (key, location, count, value)
    if gk:
        for i in range(4, len(gk) - 3, 4):
            if gk[i + 1] == 0:
                keys[gk[i]] = gk[i + 3]
    trans, scale, tie = val(34264), val(33550), val(33922)
    if trans:                                   # ModelTransformationTag
        if abs(trans[1]) > 1e-12 or abs(trans[4]) > 1e-12:
            raise UploadError("GeoTIFF 带旋转（ModelTransformation 不是北向上）；请导出为北向上的图像")
        sx, sy, e0, n0 = trans[0], -trans[5], trans[3], trans[7]
    elif scale and tie:                         # ModelPixelScaleTag + ModelTiepointTag
        sx, sy = scale[0], scale[1]
        i, j, _, x, y = tie[:5]
        e0, n0 = x - i * sx, y + j * sy
    else:
        return None
    if keys.get(1025) == 2:                     # GTRasterTypeGeoKey = PixelIsPoint: tie point is the pixel centre
        e0, n0 = e0 - sx / 2, n0 + sy / 2
    if keys.get(1024) == 2 or (keys.get(2048) and not keys.get(3072)):
        raise UploadError("GeoTIFF 用的是经纬度坐标；需要投影坐标（例如 UTM，单位米），才能按 0.1 m/px 切 patch")
    if abs(sx - M_PER_PX) > RES_TOL or abs(sy - M_PER_PX) > RES_TOL:
        raise UploadError("GeoTIFF 的分辨率是 %.4g × %.4g m/px；模型按 0.1 m/px 训练，第一版不做重采样" % (sx, sy))
    return {"E0": round(e0, 3), "N0": round(n0, 3), "m_per_px": [sx, sy], "epsg": keys.get(3072)}


def read_tiff(path):
    """-> (HxWxS uint8 bands, georef or None, nodata value or None)."""
    import tifffile
    try:
        tf = tifffile.TiffFile(path)
    except Exception:  # noqa: BLE001
        raise UploadError("无法读取这个文件：不是有效的 TIFF")
    with tf:
        page = tf.pages[0]                      # full resolution; later pages are overviews
        if page.dtype != np.uint8:
            raise UploadError("TIFF 像素类型是 %s，不是 8 位无符号整数（uint8）。第一版不做灰度拉伸，"
                              "只接受 8 位灰度声呐图" % page.dtype)
        check_size(page.imagewidth, page.imagelength)
        if int(page.photometric) == 3:
            raise UploadError("这是调色板（palette）TIFF，不是 8 位灰度")
        geo = tiff_georef(page)
        nd = page.tags.get(42113)               # GDAL_NODATA
        nodata = None
        if nd is not None:
            try:
                v = float(str(nd.value).strip("\x00 "))
                nodata = int(v) if v.is_integer() and 0 <= v <= 255 else None
            except ValueError:
                pass
        white0 = int(page.photometric) == 0     # MinIsWhite: 0 is white
        try:
            a, axes = page.asarray(), page.axes
        except Exception:  # noqa: BLE001 - e.g. LZW/JPEG need imagecodecs; Pillow decodes those
            a, axes = None, None
    if a is None:
        try:
            bands = read_pillow(path)
        except UploadError:
            raise UploadError("这个 TIFF 的压缩方式读不了；请导出为不压缩或 Deflate 压缩的 TIFF 再试")
    else:
        bands = np.moveaxis(a, axes.index("S"), -1) if "S" in axes else a[..., None]
    if white0:
        bands = bands.copy()
        bands[..., 0] = 255 - bands[..., 0]
    return bands, geo, nodata


def split_bands(bands):
    """-> (gray, opacity or None, description). A second band (or the 4th of RGBA) is treated as opacity."""
    s = bands.shape[2]
    if s in (1, 2):
        return bands[..., 0], (bands[..., 1] if s == 2 else None), ("灰度 + opacity band" if s == 2 else "灰度")
    if s in (3, 4):
        g = bands[..., 0]
        if not (np.array_equal(g, bands[..., 1]) and np.array_equal(g, bands[..., 2])):
            raise UploadError("这是彩色图像（R、G、B 三个通道不相同）；只接受 8 位灰度声呐图")
        return g, (bands[..., 3] if s == 4 else None), ("RGB（三通道相同，按灰度用）" + (" + alpha" if s == 4 else ""))
    raise UploadError("图像有 %d 个 band；只接受 1 个灰度 band（可再带 1 个 opacity band）" % s)


def image_public(meta):
    return {k: meta.get(k) for k in ("key", "name", "size", "format", "bands", "opacity", "georef", "res_confirmed",
                                      "uploaded_utc")}


def model_public(meta):
    return {k: meta.get(k) for k in ("key", "name", "sha256", "names", "trained_ultralytics", "imgsz", "uploaded_utc")}


def patchset_public(meta):
    return dict({k: meta.get(k) for k in ("key", "name", "count", "layout", "georef", "res_confirmed", "matched_bundle",
                                           "skipped_count", "layout_note", "uploaded_utc")}, skipped=(meta.get("skipped") or [])[:20])


def decode_patch(data):
    """One uploaded patch file -> 150x150 uint8 gray array (same pixel rules as whole images)."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None                        # the size is checked below, before decoding
    try:
        im = Image.open(io.BytesIO(data))
    except Exception:  # noqa: BLE001
        raise UploadError("无法读取（不是有效的 PNG/TIFF）")
    if im.size != (PATCH, PATCH):                        # checked before decoding: a whole mosaic is not loaded here
        raise UploadError("%d×%d px，不是 %d×%d 的 patch%s" % (im.size[0], im.size[1], PATCH, PATCH,
                          "（整幅图请用「选择图像/patch 文件…」只选这一个文件）" if max(im.size) > PATCH else ""))
    try:
        im.load()
    except Exception:  # noqa: BLE001
        raise UploadError("无法读取（压缩方式不支持）")
    if im.mode not in ("L", "LA", "RGB", "RGBA"):
        raise UploadError("%s，不是 8 位灰度" % PIL_MODE_MSG.get(im.mode, "像素格式 " + im.mode))
    a = np.asarray(im)
    if a.ndim == 3:
        if a.shape[2] >= 3 and not (np.array_equal(a[..., 0], a[..., 1]) and np.array_equal(a[..., 0], a[..., 2])):
            raise UploadError("彩色图像（R、G、B 不相同）")
        a = a[..., 0]
    return np.ascontiguousarray(a)


def ingest_patchset(tmp, label, sha, nbytes):
    """Check an uploaded set of patch files and store it as uploads/patchsets/ps_<sha>/ (patches.npy, meta.json).

    Request body: 4-byte big-endian length, JSON {"files": [{"name", "size"}]}, then the files' bytes in that order."""
    key = "ps_" + sha[:12]
    d = os.path.join(PATCHSETS_DIR, key)
    if os.path.exists(os.path.join(d, "meta.json")):
        return json.load(open(os.path.join(d, "meta.json")))
    arrays, entries, skipped, seen = [], [], [], set()
    with open(tmp, "rb") as f:
        hl = int.from_bytes(f.read(4), "big")
        if not 0 < hl <= 64 << 20:
            raise UploadError("上传格式不对")
        files = json.loads(f.read(hl).decode("utf-8"))["files"]
        if not files or len(files) > MAX_PATCH_FILES:
            raise UploadError("一次最多 %d 个 patch 文件（这次 %d 个）" % (MAX_PATCH_FILES, len(files)))
        if 4 + hl + sum(int(x["size"]) for x in files) != nbytes:
            raise UploadError("上传内容不完整")
        for x in files:
            data = f.read(int(x["size"]))
            name = str(x["name"])[-300:]
            base = os.path.basename(name)
            stem, ext = os.path.splitext(base)
            if ext.lower() not in IMAGE_EXT or base.startswith("."):
                skipped.append({"file": name, "reason": "不是 PNG/TIFF"})
                continue
            if stem in seen:
                skipped.append({"file": name, "reason": "文件名和另一个 patch 重复"})
                continue
            try:
                arrays.append(decode_patch(data))
            except UploadError as e:
                skipped.append({"file": name, "reason": str(e)})
                continue
            seen.add(stem)
            entries.append({"file": name, "stem": stem})
    if not arrays:
        why = "；".join("%s：%s" % (s["file"], s["reason"]) for s in skipped[:3])
        raise UploadError("没有可用的 patch（多选/文件夹只用于 150×150 的 8 位灰度 patch；整张大图请一次选一个文件）。" + why)
    # layout: by the pixel position in the file names when every file has one, else a grid in file-name order
    layout, note = "gallery", None
    pos = [POS_RE.search(e["stem"]) for e in entries]
    if all(pos):
        xy = [(int(m.group(1)), int(m.group(2))) for m in pos]
        w = max(x for x, _ in xy) - min(x for x, _ in xy) + PATCH
        h = max(y for _, y in xy) - min(y for _, y in xy) + PATCH
        if len(set(xy)) != len(xy):
            note = "文件名里的坐标有重复，改为按文件名排列"
        elif w * h > MAX_PIXELS:
            note = "文件名坐标覆盖 %.0f 百万像素，超过显示上限，改为按文件名排列" % (w * h / 1e6)
        else:
            layout = "coords"
            for e, (x, y) in zip(entries, xy):
                e["x"], e["y"] = x, y
    order = sorted(range(len(entries)), key=(lambda i: (entries[i]["y"], entries[i]["x"])) if layout == "coords"
                   else (lambda i: entries[i]["stem"]))
    entries, arrays = [entries[i] for i in order], [arrays[i] for i in order]
    if layout == "gallery":
        cols = max(1, int(np.ceil(np.sqrt(len(entries)))))
        for i, e in enumerate(entries):
            e["x"], e["y"] = (i % cols) * (PATCH + GALLERY_GAP), (i // cols) * (PATCH + GALLERY_GAP)
    for i, e in enumerate(entries):
        e["id"] = e["stem"] if SAFE_ID_RE.match(e["stem"]) else "p%05d" % i
    georef = None
    if layout == "coords" and all(BUNDLE_ID_RE.match(e["stem"]) for e in entries):
        e0, n0 = bundle_utm0()
        georef = {"E0": e0, "N0": n0, "m_per_px": [M_PER_PX, M_PER_PX], "from": "bundle patch names (CES-86 mosaic)"}
    matched = sum(e["stem"] in bundle_index() for e in entries)
    stage = os.path.join(PATCHSETS_DIR, ".writing_%s_%d" % (key, time.time_ns()))
    os.makedirs(stage)
    try:
        np.save(os.path.join(stage, "patches.npy"), np.stack(arrays))
        meta = {"key": key, "name": label, "sha256": sha, "bytes": nbytes, "count": len(entries), "layout": layout,
                "layout_note": note, "georef": georef, "res_confirmed": georef is not None, "matched_bundle": matched,
                "skipped_count": len(skipped), "skipped": skipped[:500], "uploaded_utc": utc_now(),
                "files": [{k: e[k] for k in ("id", "file", "stem", "x", "y")} for e in entries]}
        json.dump(meta, open(os.path.join(stage, "meta.json"), "w"), ensure_ascii=False)
        os.replace(stage, d)
    finally:
        if os.path.isdir(stage):
            shutil.rmtree(stage)
    return meta


def list_meta(root, key_re):
    out = []
    for d in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        if key_re.match(d):
            try:
                out.append(json.load(open(os.path.join(root, d, "meta.json"))))
            except (OSError, ValueError):
                pass
    return sorted(out, key=lambda m: m.get("uploaded_utc", ""))


def ingest_image(tmp, name, sha, nbytes):
    """Check an uploaded image and store it as uploads/images/img_<sha>/ (gray.npy, mask.npy, meta.json)."""
    key = "img_" + sha[:12]
    d = os.path.join(IMAGES_DIR, key)
    if os.path.exists(os.path.join(d, "meta.json")):          # same file uploaded before
        return json.load(open(os.path.join(d, "meta.json")))
    if os.path.splitext(name)[1].lower() in (".tif", ".tiff"):
        bands, geo, nodata = read_tiff(tmp)
        fmt = "GeoTIFF" if geo else "TIFF"
    else:
        bands, geo, nodata, fmt = read_pillow(tmp), None, None, "PNG"
    gray, opacity, bands_desc = split_bands(bands)
    gray = np.ascontiguousarray(gray)
    H, W = gray.shape
    valid, notes = None, []
    if opacity is not None and not opacity.all():
        valid = opacity > 0
        notes.append("opacity = 0 视为无数据")
    del bands, opacity                                   # large uploads: free the band stack early
    if nodata is not None and (gray == nodata).any():
        valid = (gray != nodata) if valid is None else valid & (gray != nodata)
        notes.append("nodata = %d 视为无数据" % nodata)
    stage = os.path.join(IMAGES_DIR, ".writing_%s_%d" % (key, time.time_ns()))
    os.makedirs(stage)
    try:
        np.save(os.path.join(stage, "gray.npy"), gray)
        if valid is not None and not valid.all():
            np.save(os.path.join(stage, "mask.npy"), valid.view(np.uint8))
        meta = {"key": key, "name": name, "sha256": sha, "bytes": nbytes, "size": [W, H], "format": fmt,
                "bands": bands_desc, "opacity": "；".join(notes) or None, "georef": geo,
                "res_confirmed": geo is not None, "uploaded_utc": utc_now()}
        json.dump(meta, open(os.path.join(stage, "meta.json"), "w"), indent=1, ensure_ascii=False)
        os.replace(stage, d)
    finally:
        if os.path.isdir(stage):                               # only our own staging folder
            shutil.rmtree(stage)
    return meta


def inspect_weights(path, allow_old):
    """Load a checkpoint the way inference will and check that this page can use it.

    Loading a .pt file unpickles it, i.e. runs code from the file: only upload weights you trust."""
    import torch
    check_ultralytics(allow_old)
    from ultralytics import YOLO
    try:
        m = YOLO(path)
    except Exception as e:  # noqa: BLE001
        raise UploadError("无法作为 Ultralytics 模型加载：%s: %s" % (type(e).__name__, e))
    if m.task != "detect":
        raise UploadError("模型任务是 %s，不是目标检测（detect）" % m.task)
    conv = next((x for x in m.model.modules() if isinstance(x, torch.nn.Conv2d)), None)
    if conv is None or conv.in_channels != 1:
        raise UploadError("模型第一层输入是 %s 通道；这个页面按训练方式给模型单通道灰度 patch，只接受单通道模型"
                          % (conv.in_channels if conv is not None else "?"))
    ckpt = getattr(m, "ckpt", None) or {}
    imgsz = (ckpt.get("train_args") or {}).get("imgsz")
    return {"names": [m.names[i] for i in sorted(m.names)], "trained_ultralytics": ckpt.get("version"),
            "imgsz": imgsz if isinstance(imgsz, int) else None}


def ingest_weights(tmp, name, sha, allow_old):
    key = "pt_" + sha[:12]
    d = os.path.join(MODELS_DIR, key)
    if os.path.exists(os.path.join(d, "meta.json")):
        return json.load(open(os.path.join(d, "meta.json")))
    stage = os.path.join(MODELS_DIR, ".writing_%s_%d" % (key, time.time_ns()))
    os.makedirs(stage)
    try:
        path = os.path.join(stage, "weights.pt")
        shutil.copyfile(tmp, path)
        meta = dict(inspect_weights(path, allow_old), key=key, name=name, sha256=sha, uploaded_utc=utc_now())
        json.dump(meta, open(os.path.join(stage, "meta.json"), "w"), indent=1, ensure_ascii=False)
        os.replace(stage, d)
    finally:
        if os.path.isdir(stage):
            shutil.rmtree(stage)
    return meta


# ==================================================================== runner
class Runner:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self._contacts_cache = None
        self._pyramids, self._pyramid_lock = {}, threading.Lock()
        try:
            name = os.path.relpath(args.weights, REPO_ROOT)
        except ValueError:
            name = args.weights
        self.default_model = {"key": "default", "name": name, "path": args.weights,
                              "sha256": sha256_file(args.weights), "imgsz": IMGSZ}
        self.model = self.default_model
        self._use(BundleSource(args.outputs))

    def _use(self, src):
        """Switch to a data source: geometry, empty results, then the source's last saved run if it matches."""
        self.src = src
        self.patches, self.E0, self.N0 = src.patches, src.E0, src.N0
        self.by_id = {p["id"]: p for p in self.patches}
        self.xmin = min(p["x"] for p in self.patches)
        self.ymin = min(p["y"] for p in self.patches)
        self.W = max(p["x"] for p in self.patches) + PATCH - self.xmin
        self.H = max(p["y"] for p in self.patches) + PATCH - self.ymin
        self.events = []            # [{seq, id, dets}]
        self.results = {}
        self.run = {"status": "idle"}
        self.restore_latest()

    def select(self, source=None, model=None, confirm_res=False):
        """Switch data source and/or weights (only while no inference is running)."""
        with self.lock:
            if self.run.get("status") in BUSY:
                raise RuntimeError("推理正在运行，结束后再切换")
        new_src = new_model = None
        if source and source != self.src.key:
            if source == "bundle":
                new_src = BundleSource(self.args.outputs)
            else:
                root = IMAGES_DIR if IMAGE_KEY_RE.match(source) else PATCHSETS_DIR if PATCHSET_KEY_RE.match(source) else None
                if root is None or not os.path.isdir(os.path.join(root, source)):
                    raise ValueError("unknown source")
                d = os.path.join(root, source)
                meta = json.load(open(os.path.join(d, "meta.json")))
                if not meta.get("res_confirmed"):
                    if not confirm_res:
                        raise NeedsConfirm(meta["name"])
                    meta["res_confirmed"] = True
                    json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1, ensure_ascii=False)
                new_src = (ImageSource if root == IMAGES_DIR else PatchSetSource)(d, self.args.outputs)
        if model and model != self.model["key"]:
            if model == "default":
                new_model = self.default_model
            else:
                if not MODEL_KEY_RE.match(model) or not os.path.isdir(os.path.join(MODELS_DIR, model)):
                    raise ValueError("unknown model")
                meta = json.load(open(os.path.join(MODELS_DIR, model, "meta.json")))
                new_model = dict(meta, path=os.path.join(MODELS_DIR, model, "weights.pt"), imgsz=meta.get("imgsz") or IMGSZ)
        with self.lock:
            if self.run.get("status") in BUSY:
                raise RuntimeError("推理正在运行，结束后再切换")
            if new_model:
                self.model = new_model
            if new_src or new_model:
                self._use(new_src or self.src)

    # ---------------------------------------------------------------- inference
    def start(self, conf, include_buffer):
        with self.lock:
            if self.run.get("status") in BUSY:
                raise RuntimeError("inference already running")
            todo = [p for p in self.patches if p["infer"] or (include_buffer and p["split"] == "buffer")]
            run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            self.events, self.results = [], {}
            self.run = {"status": "loading", "run_id": run_id, "total": len(todo), "done": 0, "conf": conf,
                        "include_buffer": include_buffer, "started_utc": utc_now(), "error": None,
                        "ids": [p["id"] for p in todo]}
            src, model = self.src, self.model
        threading.Thread(target=self._work, args=(todo, run_id, conf, src, model), daemon=True).start()

    def _work(self, todo, run_id, conf, src, model_info):
        try:
            ver = check_ultralytics(self.args.allow_old_ultralytics)
            from ultralytics import YOLO
            model = YOLO(model_info["path"], task="detect")
            model.model.ch = 1
            if hasattr(model.model, "yaml"):
                model.model.yaml["channels"] = 1
            with self.lock:
                self.run.update(status="running", ultralytics=ver)
                if model_info.get("names") is None:          # the default weights are not inspected at start-up
                    model_info["names"] = [model.names[i] for i in sorted(model.names)]
                    model_info["trained_ultralytics"] = (getattr(model, "ckpt", None) or {}).get("version")
            t0 = time.time()
            for i in range(0, len(todo), BATCH):
                chunk = todo[i:i + BATCH]
                res = model.predict([src.model_input(p) for p in chunk], imgsz=model_info["imgsz"], conf=PRED_FLOOR,
                                    iou=0.7, device=self.args.device, verbose=False)
                with self.lock:
                    for p, r in zip(chunk, res):
                        dets = [[round(float(v), 6) for v in b] + [round(float(c), 4)]
                                for b, c in zip(r.boxes.xywhn.cpu().numpy(), r.boxes.conf.cpu().numpy())]
                        self.results[p["id"]] = dets
                        self.events.append({"seq": len(self.events) + 1, "id": p["id"], "dets": dets})
                    self.run["done"] = len(self.results)
                    self.run["elapsed_s"] = round(time.time() - t0, 1)
                if self.args.delay:
                    time.sleep(self.args.delay)
            with self.lock:
                self.run["status"] = "mosaic"
            self._write_outputs(run_id, conf, src, model_info)
            with self.lock:
                self.run.update(status="done", finished_utc=utc_now())
        except Exception as e:  # noqa: BLE001 - reported to the UI
            with self.lock:
                self.run.update(status="error", error="%s: %s" % (type(e).__name__, e))

    # ---------------------------------------------------------------- contacts + outputs
    def merged_contacts(self, conf):
        """Global-pixel boxes >= conf; duplicates from overlapping patches merged (greedy, by confidence).

        Each box joins the earliest merged contact it overlaps with IoU >= MERGE_IOU. Only contacts sharing a
        MERGE_CELL grid cell with the box are tested, which gives the same result as testing all of them (boxes
        that overlap always share a cell) in near-linear instead of quadratic time. Cached per result set."""
        key = (conf, id(self.results), len(self.results), self.run.get("run_id"))
        cached = self._contacts_cache
        if cached and cached[0] == key:
            return cached[1]
        cands = []
        for pid, dets in self.results.items():
            p = self.by_id[pid]
            for cx, cy, w, h, c in dets:
                if c >= conf:
                    x0, y0 = p["x"] + (cx - w / 2) * PATCH, p["y"] + (cy - h / 2) * PATCH
                    cands.append({"box": [x0, y0, x0 + w * PATCH, y0 + h * PATCH], "conf": c, "patches": [pid]})
        cands.sort(key=lambda d: -d["conf"])
        merged, grid = [], {}
        for d in cands:
            b = d["box"]
            cells = [(gx, gy) for gx in range(int(b[0] // MERGE_CELL), int(b[2] // MERGE_CELL) + 1)
                     for gy in range(int(b[1] // MERGE_CELL), int(b[3] // MERGE_CELL) + 1)]
            near = {i for c in cells for i in grid.get(c, ())}
            hit = min((i for i in near if box_iou(b, merged[i]["box"]) >= MERGE_IOU), default=None)
            if hit is None:
                for c in cells:
                    grid.setdefault(c, []).append(len(merged))
                merged.append(d)
            else:
                merged[hit]["patches"].append(d["patches"][0])
        geo = self.E0 is not None
        for i, m in enumerate(merged):
            cxp, cyp = (m["box"][0] + m["box"][2]) / 2, (m["box"][1] + m["box"][3]) / 2
            m.update(contact_id=i + 1, easting=round(self.E0 + M_PER_PX * cxp, 2) if geo else None,
                     northing=round(self.N0 - M_PER_PX * cyp, 2) if geo else None,
                     width_m=round(M_PER_PX * (m["box"][2] - m["box"][0]), 2),
                     height_m=round(M_PER_PX * (m["box"][3] - m["box"][1]), 2))
        self._contacts_cache = (key, merged)
        return merged

    def _write_outputs(self, run_id, conf, src, model_info):
        """Write this run's files, then replace <out_root>/latest/ file by file (previous run is overwritten)."""
        from PIL import ImageDraw
        latest = os.path.join(src.out_root, "latest")
        tmp = os.path.join(src.out_root, ".writing_" + run_id)
        os.makedirs(tmp)
        os.makedirs(latest, exist_ok=True)
        ids = self.run["ids"]
        png = {"compress_level": 1} if self.W * self.H > 50_000_000 else {}   # large uploads: save fast
        mosaic = src.mosaic(ids, self.xmin, self.ymin, self.W, self.H)
        mosaic.save(os.path.join(tmp, "test_mosaic_gray.png"), **png)
        contacts = self.merged_contacts(conf)
        rgb = mosaic.convert("RGB")
        draw = ImageDraw.Draw(rgb)
        for m in contacts:
            b = m["box"]
            draw.rectangle([b[0] - self.xmin - 2, b[1] - self.ymin - 2, b[2] - self.xmin + 2, b[3] - self.ymin + 2],
                           outline=(230, 30, 30), width=3)
        rgb.save(os.path.join(tmp, "test_mosaic_contacts.png"), **png)          # lossless, full resolution
        with open(os.path.join(tmp, "detections_per_patch.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["patch_id", "split", "spatial_chunk", "original_group", "cx", "cy", "w", "h", "confidence"])
            for pid in ids:
                p = self.by_id[pid]
                for d in self.results.get(pid, []):
                    w.writerow([pid, p["split"], p["chunk"], p["group"]] + d)
        with open(os.path.join(tmp, "contacts.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["contact_id", "confidence", "easting", "northing", "width_m", "height_m",
                        "mosaic_x0", "mosaic_y0", "mosaic_x1", "mosaic_y1", "patches"])
            for m in contacts:
                b = [round(v - o, 1) for v, o in zip(m["box"], (self.xmin, self.ymin, self.xmin, self.ymin))]
                w.writerow([m["contact_id"], m["conf"], "" if m["easting"] is None else m["easting"],
                            "" if m["northing"] is None else m["northing"], m["width_m"], m["height_m"]]
                           + b + [";".join(m["patches"])])
        with open(os.path.join(tmp, "results.json"), "w") as f:    # lets the server restore the map after a restart
            json.dump({pid: self.results.get(pid, []) for pid in ids}, f, separators=(",", ":"))
        import ultralytics
        json.dump({"run_id": run_id, "source": src.describe(), "weights": model_info["path"],
                   "weights_name": model_info["name"], "weights_sha256": model_info["sha256"],
                   "ultralytics": ultralytics.__version__, "imgsz": model_info["imgsz"], "device": self.args.device,
                   "stored_prediction_floor": PRED_FLOOR, "map_confidence": conf, "merge_iou": MERGE_IOU,
                   "patch_px": PATCH, "patch_stride_px": STRIDE if src.kind == "image" else None,
                   "patches_inferred": len(ids), "include_buffer": self.run["include_buffer"],
                   "patches_with_detections_at_conf": sum(any(d[4] >= conf for d in self.results[i]) for i in ids),
                   "merged_contacts_at_conf": len(contacts), "mosaic_px": [self.W, self.H],
                   "mosaic_origin_pixel_id": [self.xmin, self.ymin], "m_per_px": M_PER_PX,
                   "utm_of_pixel_id_0_0": [self.E0, self.N0] if self.E0 is not None else None,
                   "started_utc": self.run["started_utc"], "finished_utc": utc_now()},
                  open(os.path.join(tmp, "run_manifest.json"), "w"), indent=1, ensure_ascii=False)
        for fn in OUTPUT_FILES:                       # replace the previous run's files
            os.replace(os.path.join(tmp, fn), os.path.join(latest, fn))
        os.rmdir(tmp)                                 # our own now-empty staging folder
        with self.lock:
            self.run["output_dir"] = latest
            self.run["files"] = list(OUTPUT_FILES)

    def restore_latest(self):
        """Show the source's last completed run (<out_root>/latest/) instead of an empty map."""
        latest = os.path.join(self.src.out_root, "latest")
        try:
            man = json.load(open(os.path.join(latest, "run_manifest.json")))
            res = json.load(open(os.path.join(latest, "results.json")))
        except (OSError, ValueError):
            return
        if (man.get("weights_sha256") != self.model["sha256"] or (man.get("source") or {}).get("key", "bundle") != self.src.key
                or any(pid not in self.by_id for pid in res)):
            return                                    # different model or data: start clean
        self.results = res
        self.events = [{"seq": i + 1, "id": pid, "dets": d} for i, (pid, d) in enumerate(res.items())]
        self.run = {"status": "done", "run_id": man["run_id"], "total": len(res), "done": len(res),
                    "conf": man["map_confidence"], "include_buffer": man["include_buffer"],
                    "started_utc": man["started_utc"], "finished_utc": man["finished_utc"],
                    "ultralytics": man["ultralytics"], "restored_from_disk": True, "ids": list(res),
                    "output_dir": latest, "files": [f for f in OUTPUT_FILES if os.path.exists(os.path.join(latest, f))]}

    def clear_results(self):
        """Delete the current source's saved inference results (known result files only) and reset the map."""
        with self.lock:
            if self.run.get("status") in BUSY:
                raise RuntimeError("inference is running; wait until it has finished")
            root = os.path.realpath(self.src.out_root)
            deleted = []
            for d in sorted(os.listdir(root)) if os.path.isdir(root) else []:
                dpath = os.path.join(root, d)
                if not RUN_DIR_RE.match(d) or os.path.islink(dpath) or not os.path.isdir(dpath):
                    continue
                for fn in sorted(os.listdir(dpath)):
                    fpath = os.path.join(dpath, fn)
                    if RESULT_FILE_RE.match(fn) and os.path.isfile(fpath) and not os.path.islink(fpath):
                        os.remove(fpath)
                        deleted.append(os.path.join(d, fn))
                if not os.listdir(dpath):
                    os.rmdir(dpath)                   # only if nothing else is left in it
            self.results, self.events = {}, []
            self.run = {"status": "idle"}
            return deleted

    # ---------------------------------------------------------------- map background tiles
    def tile(self, layer, z, tx, ty):
        """PNG tile of a background layer: "source" = the uploaded image, "mosaic" = the finished run's mosaic
        (for an uploaded image that is the same picture). Pyramids are built on first use; the last two are kept."""
        from PIL import Image
        src = self.src
        if src.kind == "image":
            key, load = "image:" + src.key, lambda: src.gray
        else:
            d = self.run.get("output_dir")
            path = os.path.join(d, "test_mosaic_gray.png") if d and layer == "mosaic" else None
            if not path or not os.path.exists(path):
                raise ValueError("no such layer")
            Image.MAX_IMAGE_PIXELS = None
            key = "mosaic:%s:%d" % (path, os.stat(path).st_mtime_ns)
            load = lambda: np.asarray(Image.open(path).convert("L"))        # noqa: E731
        with self._pyramid_lock:
            pyr = self._pyramids.get(key)
            if pyr is None:
                pyr = Pyramid(load())
                self._pyramids = dict(list(self._pyramids.items())[-1:] + [(key, pyr)])
        if not 0 <= z < len(pyr.levels):
            raise ValueError("no such level")
        return pyr.tile(z, tx, ty)

    # ---------------------------------------------------------------- API payloads
    def state(self):
        sources = [{"key": BundleSource.key, "kind": BundleSource.kind, "name": BundleSource.name}] + \
                  [dict(image_public(m), kind="image") for m in list_meta(IMAGES_DIR, IMAGE_KEY_RE)] + \
                  [dict(patchset_public(m), kind="patches") for m in list_meta(PATCHSETS_DIR, PATCHSET_KEY_RE)]
        models = [model_public(self.default_model)] + [model_public(m) for m in list_meta(MODELS_DIR, MODEL_KEY_RE)]
        with self.lock:
            return {"patches": [dict({k: p[k] for k in ("id", "chunk", "group", "split", "infer", "x", "y", "gt")},
                                     **({"name": p["name"]} if "name" in p else {})) for p in self.patches],
                    "origin": [self.xmin, self.ymin], "size": [self.W, self.H], "patch_px": PATCH,
                    "cell_inset": getattr(self.src, "cell_inset", 25), "tile_px": TILE_PX,
                    "m_per_px": M_PER_PX, "utm0": [self.E0, self.N0] if self.E0 is not None else None,
                    "source": self.src.info, "sources": sources, "model": model_public(self.model), "models": models,
                    "weights": self.model["path"],
                    "run": {k: v for k, v in self.run.items() if k != "ids"}, "results": self.results}

    def progress(self, since):
        with self.lock:
            return {"run": {k: v for k, v in self.run.items() if k != "ids"},
                    "events": self.events[since:]}


def make_handler(runner, port):
    allowed = {"127.0.0.1:%d" % port, "localhost:%d" % port}
    static = {"/": ("index.html", "text/html; charset=utf-8"),
              "/static/map.js": ("static/map.js", "application/javascript; charset=utf-8"),
              "/static/map.css": ("static/map.css", "text/css; charset=utf-8")}

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *a):
            if len(a) > 1 and str(a[1])[:1] in "45":
                BaseHTTPRequestHandler.log_message(self, fmt, *a)

        def headers_out(self, code, ctype, length, extra):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(length))
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self'; object-src 'none'")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()

        def send(self, code, body, ctype="application/json; charset=utf-8", extra=None):
            if not isinstance(body, bytes):
                body = json.dumps(body).encode()
            self.headers_out(code, ctype, len(body), extra)
            self.wfile.write(body)

        def send_file(self, path, ctype, extra=None):
            with open(path, "rb") as f:               # streamed: uploaded mosaics can be hundreds of MB
                self.headers_out(200, ctype, os.fstat(f.fileno()).st_size, extra)
                shutil.copyfileobj(f, self.wfile, 1 << 20)

        def do_GET(self):
            if self.headers.get("Host") not in allowed:
                self.close_connection = True
                return self.send(403, {"error": "host"})
            u = urlparse(self.path)
            q = parse_qs(u.query)
            try:
                if u.path in static:
                    rel, ct = static[u.path]
                    return self.send(200, open(os.path.join(HERE, rel), "rb").read(), ct, {"Cache-Control": "no-cache"})
                if u.path == "/api/state":
                    return self.send(200, runner.state())
                if u.path == "/api/progress":
                    return self.send(200, runner.progress(int(q.get("since", ["0"])[0])))
                if u.path == "/api/contacts":
                    return self.send(200, runner.merged_contacts(float(q.get("conf", ["0.5"])[0])))
                m = re.match(r"^/api/tile/(source|mosaic)/(\d+)/(\d+)/(\d+)\.png$", u.path)
                if m:                                  # URL carries ?v=<source or run>, so tiles can be cached
                    return self.send(200, runner.tile(m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))),
                                     "image/png", {"Cache-Control": "private, max-age=86400"})
                m = re.match(r"^/api/image/([\w\-]+)\.png$", u.path)
                if m and m.group(1) in runner.by_id:
                    return self.send(200, runner.src.patch_png(runner.by_id[m.group(1)]), "image/png",
                                     {"Cache-Control": "no-cache"})
                m = re.match(r"^/api/output/([\w.\-]+)$", u.path)
                d = runner.run.get("output_dir")
                if m and d and m.group(1) in OUTPUT_FILES and os.path.exists(os.path.join(d, m.group(1))):
                    fn = m.group(1)
                    ct = {"png": "image/png", "jpg": "image/jpeg", "csv": "text/csv", "json": "application/json"}[fn.rsplit(".", 1)[1]]
                    extra = {"Cache-Control": "no-cache"}      # files are overwritten by each new run
                    if "dl" in q:
                        extra["Content-Disposition"] = "attachment; filename=%s" % fn
                    return self.send_file(os.path.join(d, fn), ct, extra)
                if u.path == "/favicon.ico" or u.path.startswith("/apple-touch-icon"):   # browser icon probes
                    return self.send(204, b"", "image/x-icon")
                return self.send(404, {"error": "not found"})
            except (ValueError, OSError) as e:
                return self.send(400, {"error": str(e)})

        def upload(self, q):
            """Stream the request body (a whole file, not multipart) to disk, then check and store it."""
            kind = q.get("kind", [""])[0]
            limit = {"image": MAX_IMAGE_BYTES, "patches": MAX_IMAGE_BYTES, "weights": MAX_WEIGHTS_BYTES}.get(kind)
            name = os.path.basename(unquote(self.headers.get("X-Filename") or "").rstrip("/"))[:200]
            ext = os.path.splitext(name)[1].lower()
            n = int(self.headers.get("Content-Length") or 0)
            err = None
            if limit is None:
                err = (404, "not found")
            elif kind != "patches" and ext not in (IMAGE_EXT if kind == "image" else (".pt",)):
                err = (400, "只接受 %s 文件" % ("/".join(IMAGE_EXT) if kind == "image" else ".pt"))
            elif not 0 < n <= limit:
                err = (413, "文件大小 %.1f MB 超过上限 %.0f MB" % (n / 1e6, limit / 1e6))
            if err:
                self.close_connection = True          # the unread body is dropped with the connection
                return self.send(err[0], {"error": err[1]})
            incoming = os.path.join(UPLOADS, ".incoming")
            os.makedirs(incoming, exist_ok=True)
            tmp = os.path.join(incoming, "%d_%d%s" % (time.time_ns(), threading.get_ident(), ext))
            try:
                h, left = hashlib.sha256(), n
                with open(tmp, "wb") as f:
                    while left:
                        chunk = self.rfile.read(min(1 << 20, left))
                        if not chunk:
                            raise ConnectionError("upload interrupted")
                        f.write(chunk)
                        h.update(chunk)
                        left -= len(chunk)
                if kind == "image":
                    item = dict(image_public(ingest_image(tmp, name, h.hexdigest(), n)), kind="image")
                elif kind == "patches":
                    item = dict(patchset_public(ingest_patchset(tmp, name or "patches", h.hexdigest(), n)), kind="patches")
                elif h.hexdigest() == runner.default_model["sha256"]:     # the default weights, chosen again
                    item = model_public(runner.default_model)
                else:
                    item = model_public(ingest_weights(tmp, name, h.hexdigest(), runner.args.allow_old_ultralytics))
                return self.send(200, {"ok": True, "item": item})
            except ConnectionError:
                self.close_connection = True
                return None
            except (UploadError, RuntimeError) as e:
                return self.send(400, {"error": str(e)})
            except (ValueError, OSError, MemoryError) as e:
                return self.send(400, {"error": "%s: %s" % (type(e).__name__, e)})
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)

        def do_POST(self):
            if self.headers.get("Host") not in allowed or self.headers.get("X-SSS-Tool") != "1":
                self.close_connection = True
                return self.send(403, {"error": "forbidden"})
            u = urlparse(self.path)
            if u.path == "/api/upload":
                return self.upload(parse_qs(u.query))
            n = int(self.headers.get("Content-Length") or 0)
            if n >= 10000:
                self.close_connection = True
                return self.send(413, {"error": "body too large"})
            raw = self.rfile.read(n) if n > 0 else b"{}"
            path = u.path
            if path == "/api/clear_results":
                try:
                    deleted = runner.clear_results()
                    return self.send(200, {"ok": True, "deleted": deleted, "run": {"status": "idle"}})
                except RuntimeError as e:
                    return self.send(409, {"error": str(e)})
            if path == "/api/select":
                try:
                    body = json.loads(raw or b"{}")
                    runner.select(body.get("source"), body.get("model"), bool(body.get("confirm_res")))
                    return self.send(200, {"ok": True})
                except NeedsConfirm as e:
                    return self.send(409, {"error": "needs_res_confirm", "needs_res_confirm": True, "name": str(e)})
                except RuntimeError as e:
                    return self.send(409, {"error": str(e)})
                except (ValueError, OSError) as e:
                    return self.send(400, {"error": str(e)})
            if path != "/api/start":
                return self.send(404, {"error": "not found"})
            try:
                body = json.loads(raw or b"{}")
                conf = float(body.get("conf", 0.5))
                if not PRED_FLOOR <= conf <= 1:
                    raise ValueError("conf must be between %.2f and 1" % PRED_FLOOR)
                runner.start(conf, bool(body.get("include_buffer")))
                return self.send(200, {"ok": True})
            except RuntimeError as e:
                return self.send(409, {"error": str(e)})
            except ValueError as e:
                return self.send(400, {"error": str(e)})

    return H


def main():
    global BUNDLE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--bundle", default=BUNDLE, help="spatial split bundle with metadata/ and data/")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--device", default="cpu", help="cpu (fastest for this nano model on M3) or mps")
    ap.add_argument("--delay", type=float, default=0.0, help="seconds to pause between batches (slow-motion demo)")
    ap.add_argument("--outputs", default=os.path.join(HERE, "outputs"), help="where each run's folder is written")
    ap.add_argument("--allow-old-ultralytics", action="store_true", help="testing only: results will be wrong")
    args = ap.parse_args()
    BUNDLE = os.path.abspath(args.bundle)
    args.weights = os.path.abspath(args.weights)
    import socket
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        probe.bind(("127.0.0.1", args.port))
    except OSError:
        raise SystemExit("端口 %d 已被占用：地图服务可能已经在另一个终端运行。\n"
                         "  - 如果已经在运行，直接打开 http://127.0.0.1:%d/\n"
                         "  - 要重启：在运行它的终端按 Ctrl+C，再重新运行本命令\n"
                         "  - 或换端口：加上 --port %d" % (args.port, args.port, args.port + 1))
    finally:
        probe.close()
    runner = Runner(args)
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(runner, args.port))
    httpd.daemon_threads = True
    print("test patches: %d (+%d buffer) | mosaic %dx%d px | weights %s"
          % (sum(p["split"] == "test" for p in runner.patches), sum(p["split"] == "buffer" for p in runner.patches),
             runner.W, runner.H, args.weights), flush=True)
    print("serving on http://127.0.0.1:%d/  (Ctrl+C to stop)" % args.port, flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    main()
