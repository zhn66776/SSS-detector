"""YOLO label parsing, formatting and validation.

Format per line: ``class_id cx cy w h`` with coordinates normalised to 0-1.
Files in B use 8 decimals and a trailing newline; empty files mean "no contact".
"""
import math

from .common import CONTACT_CLASS_ID, IMAGE_H, IMAGE_W

# Rounding to 8 decimals in the source labels leaves edges up to ~5e-9 outside
# the image (two boxes in B). Tolerance is far below one pixel (1/150 = 0.0067).
BOUNDS_TOL = 1e-6
MAX_BOXES_PER_PATCH = 200


class LabelError(ValueError):
    pass


def parse_label_text(text):
    boxes = []
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        t = line.split()
        if len(t) != 5:
            raise LabelError("line %d: expected 5 fields, got %d" % (n, len(t)))
        try:
            cls = int(t[0])
            cx, cy, w, h = (float(x) for x in t[1:])
        except ValueError:
            raise LabelError("line %d: not numeric: %r" % (n, line))
        boxes.append({"class_id": cls, "cx": cx, "cy": cy, "w": w, "h": h})
    return boxes


def validate_box(b):
    """Return a cleaned box dict or raise LabelError."""
    try:
        cls = int(b.get("class_id", CONTACT_CLASS_ID))
        cx, cy, w, h = (float(b[k]) for k in ("cx", "cy", "w", "h"))
    except (KeyError, TypeError, ValueError):
        raise LabelError("box needs numeric cx, cy, w, h")
    if cls != CONTACT_CLASS_ID:
        raise LabelError("only class %d (contact) is allowed, got %r" % (CONTACT_CLASS_ID, cls))
    for v in (cx, cy, w, h):
        if not math.isfinite(v):
            raise LabelError("coordinates must be finite")
    if w <= 0 or h <= 0:
        raise LabelError("width and height must be positive")
    x0, x1, y0, y1 = cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2
    if x0 < -BOUNDS_TOL or y0 < -BOUNDS_TOL or x1 > 1 + BOUNDS_TOL or y1 > 1 + BOUNDS_TOL:
        raise LabelError(
            "box exceeds image bounds: x %.2f-%.2f px, y %.2f-%.2f px (image %dx%d)"
            % (x0 * IMAGE_W, x1 * IMAGE_W, y0 * IMAGE_H, y1 * IMAGE_H, IMAGE_W, IMAGE_H)
        )
    return {"class_id": cls, "cx": cx, "cy": cy, "w": w, "h": h}


def validate_boxes(boxes):
    if not isinstance(boxes, list):
        raise LabelError("boxes must be a list")
    if len(boxes) > MAX_BOXES_PER_PATCH:
        raise LabelError("too many boxes")
    out = []
    for i, b in enumerate(boxes):
        if not isinstance(b, dict):
            raise LabelError("box %d is not an object" % i)
        try:
            out.append(validate_box(b))
        except LabelError as e:
            raise LabelError("box %d: %s" % (i, e))
    return out


def format_boxes(boxes):
    """Serialise validated boxes; an empty list gives an empty (0-byte) file."""
    lines = []
    for b in boxes:
        lines.append("%d %.8f %.8f %.8f %.8f\n" % (b["class_id"], b["cx"], b["cy"], b["w"], b["h"]))
    data = "".join(lines).encode("ascii")
    # Round-trip check so a saved file always parses back to valid boxes.
    validate_boxes(parse_label_text(data.decode("ascii")))
    return data


def clip_box_to_image(b):
    """Clip a (prediction) box to the image and return YOLO cx, cy, w, h."""
    x0 = max(0.0, b["cx"] - b["w"] / 2)
    x1 = min(1.0, b["cx"] + b["w"] / 2)
    y0 = max(0.0, b["cy"] - b["h"] / 2)
    y1 = min(1.0, b["cy"] + b["h"] / 2)
    if x1 <= x0 or y1 <= y0:
        raise LabelError("box has no area inside the image")
    return {"class_id": CONTACT_CLASS_ID, "cx": (x0 + x1) / 2, "cy": (y0 + y1) / 2, "w": x1 - x0, "h": y1 - y0}


def to_pixels(b):
    return {
        "x0": (b["cx"] - b["w"] / 2) * IMAGE_W,
        "y0": (b["cy"] - b["h"] / 2) * IMAGE_H,
        "x1": (b["cx"] + b["w"] / 2) * IMAGE_W,
        "y1": (b["cy"] + b["h"] / 2) * IMAGE_H,
    }
