"""Paths, hashing and atomic file helpers.

Everything this tool writes goes below TOOL_DIR. The source ZIP and backup B
are only ever opened for reading.
"""
import datetime
import hashlib
import json
import os
import tempfile

TOOL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Source ZIP and backup B live next to this tool folder (label_review/round1/); paths are relative so the
# whole project folder can be moved. Override with serve.py --zip / --backup-b.
ROUND_DIR = os.path.dirname(TOOL_DIR)
DEFAULT_ZIP = os.path.join(ROUND_DIR, "sss_contact_yolo26_spatial_cv_review_20260922T234500Z.zip")
DEFAULT_B_DIR = os.path.join(ROUND_DIR, "sss_contact_yolo26_spatial_cv_review_20260922T234500Z_label_backup_B")

ZIP_ROOT = "sss_contact_yolo26_spatial_cv_review/"
LABEL_SUBDIR = "validation/ground_truth_labels"

A_DIR = os.path.join(TOOL_DIR, "working_A")
A_LABEL_DIR = os.path.join(A_DIR, LABEL_SUBDIR)
INDEX_DIR = os.path.join(TOOL_DIR, "index")
EXPORT_DIR = os.path.join(TOOL_DIR, "exports")
REVIEW_MANIFEST = os.path.join(TOOL_DIR, "review_manifest.json")
REVIEW_STATE = os.path.join(TOOL_DIR, "review_state.json")
HISTORY_STATE = os.path.join(TOOL_DIR, "history_state.json")
UNDO_LOG = os.path.join(TOOL_DIR, "undo_log.jsonl")
PID_FILE = os.path.join(TOOL_DIR, "server.pid")

IMAGE_W = 150
IMAGE_H = 150
CONTACT_CLASS_ID = 0
CLASS_NAMES = {0: "contact"}
PREDICTION_CONFIDENCE_FLOOR = 0.05
DEFAULT_CONF = 0.20
DEFAULT_IOU = 0.50


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds")


def utc_stamp():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def file_fingerprint(path):
    st = os.stat(path)
    return {
        "path": path,
        "sha256": sha256_file(path),
        "size_bytes": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "mtime_utc": datetime.datetime.fromtimestamp(st.st_mtime, datetime.timezone.utc).isoformat(),
    }


def _assert_inside_tool_dir(path):
    real = os.path.realpath(path)
    if not (real == TOOL_DIR or real.startswith(TOOL_DIR + os.sep)):
        raise PermissionError("refusing to write outside the platform directory: %s" % path)


def atomic_write_bytes(path, data):
    """Write via temp file + fsync + os.replace so readers never see half a file."""
    _assert_inside_tool_dir(path)
    d = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=d)
    try:
        os.chmod(tmp, 0o644)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)  # only our own never-published temp file
        raise
    dfd = os.open(d, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def atomic_write_json(path, obj, compact=False):
    if compact:
        text = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    else:
        text = json.dumps(obj, ensure_ascii=False, indent=1)
    atomic_write_bytes(path, (text + "\n").encode("utf-8"))


def append_jsonl(path, obj):
    _assert_inside_tool_dir(path)
    line = json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def read_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
