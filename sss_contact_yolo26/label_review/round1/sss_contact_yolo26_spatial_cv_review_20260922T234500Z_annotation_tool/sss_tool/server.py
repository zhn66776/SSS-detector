"""Local-only HTTP server (stdlib). Binds to 127.0.0.1; serves static files from
the platform directory, original PNGs straight from the source ZIP, and a JSON API.
"""
import json
import os
import random
import re
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .common import (A_LABEL_DIR, DEFAULT_CONF, DEFAULT_IOU, IMAGE_H, IMAGE_W,
                     LABEL_SUBDIR, PREDICTION_CONFIDENCE_FLOOR, TOOL_DIR,
                     file_fingerprint, read_json, sha256_bytes)
from .evaluate import EXTRA, UNLABELLED, summarize
from .exporter import EXPORT_DIR, export_a
from .labels import LabelError, clip_box_to_image, to_pixels
from .store import (PATCH_STATUSES, PRED_STATUS_LABELS, PRED_STATUSES,
                    ConflictError)

STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/static/app.js": ("static/app.js", "application/javascript; charset=utf-8"),
    "/static/app.css": ("static/app.css", "text/css; charset=utf-8"),
}
VIEWS = ("all", "fp", "fn", "tp", "tn", "unscored")
PID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")

FP_NOTE_EN = ("FP = false positive relative to the current annotation set: an OOF prediction that is not "
              "matched to any current A label. It is a review candidate, not a confirmed physical false positive.")
FP_NOTE_ZH = ("FP 仅表示“相对于当前标注集（A）的未匹配预测候选”：模型预测框没有与当前 A 中任何 GT 框一对一匹配。"
              "它只是人工复核线索，并不代表已确认的物理 false positive，也可能是漏标的真实 contact。")


class App:
    def __init__(self, ctx):
        self.ctx = ctx  # dict: index, store, evaluator, zip_path, b_dir, b_manifest_path, ...
        self.index = ctx["index"]
        self.store = ctx["store"]
        self.ev = ctx["evaluator"]
        self.zip = zipfile.ZipFile(ctx["zip_path"])
        self.zip_lock = threading.Lock()
        self.export_lock = threading.Lock()
        self.patches = self.index["patches"]
        self.order = list(self.patches)  # chunk, patch_id order

    # ------------------------------------------------------------------ list
    def _review_summary(self, pid, res):
        out = {}
        for k in res["unmatched_pred"]:
            r = self.store.get_review("prediction", pid, k)
            s = r["status"] if r else "unreviewed"
            out[s] = out.get(s, 0) + 1
        return out

    def _row(self, pid, res):
        e = self.patches[pid]
        preds = e["predictions"]
        matched = {m["k"]: m for m in res["matches"]}
        pr = self.store.get_review("patch", pid)
        return {
            "patch_id": pid, "spatial_chunk": e["spatial_chunk"], "original_group": e["original_group"],
            "final_split": e["final_split"], "scored": e["scored"], "cv_fold_ids": e["cv_fold_ids"],
            "gt_count": res["gt_count"], "pred_count": res["pred_count"], "tp": res["tp"], "fp": res["fp"],
            "fn": res["fn"], "tn": res["tn"], "fp_type": res["fp_type"], "max_fp_conf": res["max_fp_conf"],
            "max_pred_conf": res["max_pred_conf"], "min_tp_iou": res["min_tp_iou"],
            "modified": self.store.is_modified(pid),
            "patch_review": pr["status"] if pr else "unreviewed",
            "fp_review": self._review_summary(pid, res),
            "gt": [dict(b, fn=(i in res["unmatched_gt"])) for i, b in enumerate(self.store.a_boxes[pid])],
            "preds": [{"k": k, "cx": preds[k]["cx"], "cy": preds[k]["cy"], "w": preds[k]["w"], "h": preds[k]["h"],
                       "confidence": preds[k]["confidence"], "matched": k in matched,
                       "iou": matched[k]["iou"] if k in matched else None} for k in res["pred_ks"]],
        }

    def _in_view(self, view, r):
        if view == "all":
            return True
        if view == "unscored":
            return not r["scored"]
        if not r["scored"]:
            return False
        return {"fp": r["fp"] > 0, "fn": r["fn"] > 0, "tp": r["tp"] > 0, "tn": r["tn"]}[view]

    def list(self, q):
        conf, iou = self._thresholds(q)
        view = q.get("view", "all")
        if view not in VIEWS:
            raise LabelError("unknown view")
        results = self.ev.results(conf, iou)
        f_chunk = q.get("chunk", "")
        f_group = q.get("group", "")
        f_split = q.get("split", "")
        f_q = q.get("q", "").strip().lower()
        f_fp_type = q.get("fp_type", "")
        f_fp_review = q.get("fp_review", "")
        f_patch_review = q.get("patch_review", "")
        f_modified = q.get("modified", "")
        f_min_conf = _float(q.get("min_conf"))
        f_max_tp_iou = _float(q.get("max_tp_iou"))
        f_min_fp = _float(q.get("min_fp"))

        def keep(pid):
            e, r = self.patches[pid], results[pid]
            if f_chunk and str(e["spatial_chunk"]) != f_chunk:
                return False
            if f_group and e["original_group"] != f_group:
                return False
            if f_split and e["final_split"] != f_split:
                return False
            if f_q and f_q not in pid.lower():
                return False
            if f_fp_type and r["fp_type"] != f_fp_type:
                return False
            if f_fp_review and f_fp_review not in self._review_summary(pid, r):
                return False
            if f_patch_review:
                pr = self.store.get_review("patch", pid)
                if (pr["status"] if pr else "unreviewed") != f_patch_review:
                    return False
            if f_modified == "yes" and not self.store.is_modified(pid):
                return False
            if f_modified == "no" and self.store.is_modified(pid):
                return False
            if f_min_conf is not None and (r["max_pred_conf"] is None or r["max_pred_conf"] < f_min_conf):
                return False
            if f_max_tp_iou is not None and (r["min_tp_iou"] is None or r["min_tp_iou"] > f_max_tp_iou):
                return False
            if f_min_fp is not None and r["fp"] < f_min_fp:
                return False
            return True

        filtered_all = [p for p in self.order if keep(p)]
        view_counts = {v: sum(1 for p in filtered_all if self._in_view(v, results[p])) for v in VIEWS}
        view_counts_unfiltered = {v: sum(1 for p in self.order if self._in_view(v, results[p])) for v in VIEWS}
        rows = [p for p in filtered_all if self._in_view(view, results[p])]

        sort = q.get("sort", "default")
        desc = q.get("order", "") == "desc"
        if sort == "default":
            sort, desc = {"fp": ("max_fp_conf", True), "tp": ("min_tp_iou", False),
                          "fn": ("fn_count", True)}.get(view, ("chunk", False))
        rows = self._sort(rows, results, sort, desc, q.get("seed", "1"))

        per_page = max(1, min(200, int(_float(q.get("per_page")) or 50)))
        pages = max(1, (len(rows) + per_page - 1) // per_page)
        page = max(1, min(pages, int(_float(q.get("page")) or 1)))
        page_ids = rows[(page - 1) * per_page: page * per_page]
        return {
            "view": view, "thresholds": {"confidence": conf, "iou": iou},
            "sort": sort, "order": "desc" if desc else "asc",
            "page": page, "pages": pages, "per_page": per_page, "count": len(rows),
            "view_counts_filtered": view_counts, "view_counts_all": view_counts_unfiltered,
            "stats_view_filtered": summarize(rows, results, self.index),
            "stats_filters_all_views": summarize(filtered_all, results, self.index),
            "stats_global": summarize(self.order, results, self.index),
            "items": [self._row(p, results[p]) for p in page_ids],
            "store_version": self.store.version,
        }

    def _sort(self, rows, results, sort, desc, seed):
        pos = {p: i for i, p in enumerate(self.order)}
        big = float("inf")

        def num(v, missing):
            return missing if v is None else v

        rank_fp = {UNLABELLED: 0, EXTRA: 1, None: 2}
        pr_rank = {s: i for i, s in enumerate(PRED_STATUSES)}
        keys = {
            "chunk": lambda p: pos[p],
            "patch_id": lambda p: p,
            "max_conf": lambda p: (num(results[p]["max_pred_conf"], -1), -pos[p]),
            "max_fp_conf": lambda p: (num(results[p]["max_fp_conf"], -1), -pos[p]),
            "fp_count": lambda p: (results[p]["fp"], -pos[p]),
            "fn_count": lambda p: (results[p]["fn"], -pos[p]),
            "gt_count": lambda p: (results[p]["gt_count"], -pos[p]),
            "min_tp_iou": lambda p: (num(results[p]["min_tp_iou"], big if not desc else -1), pos[p] * (-1 if desc else 1)),
            "fp_type": lambda p: (rank_fp[results[p]["fp_type"]], -num(results[p]["max_fp_conf"], 0)),
            "fp_review": lambda p: (min((pr_rank[s] for s in self._review_summary(p, results[p])), default=9), pos[p]),
            "patch_review": lambda p: ((self.store.get_review("patch", p) or {"status": "unreviewed"})["status"], pos[p]),
            "modified": lambda p: (self.store.is_modified(p), -pos[p]),
        }
        if sort == "random":
            rows = list(rows)
            random.Random(str(seed)).shuffle(rows)
            return rows
        if sort not in keys:
            raise LabelError("unknown sort")
        return sorted(rows, key=keys[sort], reverse=desc)

    # ------------------------------------------------------------------ detail
    def detail(self, pid, q):
        conf, iou = self._thresholds(q)
        e = self.patches[pid]
        res = self.ev.patch(pid, conf, iou)
        a_boxes = self.store.a_boxes[pid]
        matched_gt = {m["gt"]: m for m in res["matches"]}
        matched_k = {m["k"]: m for m in res["matches"]}
        gt = []
        for i, b in enumerate(a_boxes):
            m = matched_gt.get(i)
            gt.append({"index": i, "yolo": b, "px": to_pixels(b), "matched_k": m["k"] if m else None,
                       "iou": m["iou"] if m else None, "fn": e["scored"] and m is None,
                       "best_iou_any_prediction": res["best_iou_gt"][i] if e["scored"] else None})
        preds = []
        for p in e["predictions"]:
            k = p["k"]
            m = matched_k.get(k)
            above = p["confidence"] >= conf
            rv = self.store.get_review("prediction", pid, k)
            preds.append({
                "k": k, "yolo": {q2: p[q2] for q2 in ("class_id", "cx", "cy", "w", "h")}, "px": to_pixels(p),
                "confidence": p["confidence"], "above_threshold": above,
                "matched": m is not None, "matched_gt": m["gt"] if m else None, "iou": m["iou"] if m else None,
                "fp_candidate": above and m is None, "fp_type": res["fp_type"] if above and m is None else None,
                "fold_ids": p["fold_ids"], "fold_support": p["fold_support"],
                "fold_support_at_threshold": len({x["fold_id"] for x in p["members"] if x["confidence"] >= conf}),
                "source_prediction_count": p["source_prediction_count"],
                "representative_fold_id": p["representative_fold_id"], "members": p["members"],
                "review": rv or {"status": "unreviewed", "note": ""},
            })
        a_data = self.store.a_bytes[pid]
        b_data = self.store.b_bytes[pid]
        zr = self.index["zip_root"]
        pr = self.store.get_review("patch", pid)
        return {
            "patch_id": pid, "thresholds": {"confidence": conf, "iou": iou},
            "meta": {k: e.get(k) for k in ("spatial_chunk", "original_group", "final_split", "bounds_m", "scored",
                                            "unscored_reason", "cv_fold_ids", "cv_fold_count", "fold_evaluation",
                                            "fold_agreement")},
            "zip_baseline": e.get("zip_baseline"),
            "paths": {
                "source_zip": self.ctx["zip_path"],
                "zip_image_member": e["zip_image_member"],
                "zip_gt_member": e["zip_gt_member"],
                "zip_pred_label_member": e["zip_pred_label_member"],
                "prediction_sources": [zr + "html_review/review_data.json (merged OOF union)",
                                       zr + "predictions_by_fold.csv (raw per-fold rows)"],
                "b_label_path": os.path.join(self.ctx["b_dir"], LABEL_SUBDIR, pid + ".txt"),
                "a_label_path": os.path.join(A_LABEL_DIR, pid + ".txt"),
            },
            "counts": {k: res[k] for k in ("gt_count", "pred_count", "tp", "fp", "fn", "tn", "fp_type")},
            "gt": gt,
            "b_gt": [{"index": i, "yolo": b, "px": to_pixels(b)} for i, b in enumerate(self.store.b_boxes[pid])],
            "predictions": preds,
            "a_sha256": sha256_bytes(a_data), "b_sha256": sha256_bytes(b_data),
            "a_text": a_data.decode("ascii"),
            "modified": a_data != b_data,
            "patch_review": pr or {"status": "unreviewed", "note": ""},
            "history": self.store.history_summary(limit=20, pid=pid)["recent"],
            "store_version": self.store.version,
        }

    def _thresholds(self, q):
        conf = _float(q.get("conf"))
        iou = _float(q.get("iou"))
        conf = DEFAULT_CONF if conf is None else conf
        iou = DEFAULT_IOU if iou is None else iou
        if not (PREDICTION_CONFIDENCE_FLOOR - 1e-9 <= conf <= 1.0):
            raise LabelError("confidence threshold must be between the saved floor %.2f and 1"
                             % PREDICTION_CONFIDENCE_FLOOR)
        if not (0.0 < iou <= 1.0):
            raise LabelError("IoU threshold must be in (0, 1]")
        return round(conf, 4), round(iou, 4)

    def image(self, pid):
        member = self.patches[pid]["zip_image_member"]
        with self.zip_lock:
            info = self.zip.getinfo(member)
            data = self.zip.read(member)
        return data, "%08x" % info.CRC

    def meta(self):
        c = self.ctx
        return {
            "tool_dir": TOOL_DIR,
            "roles": {
                "ZIP": {"path": c["zip_path"], "role": "original read-only data + CV results (images read on demand)"},
                "B": {"path": c["b_dir"], "label_dir": os.path.join(c["b_dir"], LABEL_SUBDIR),
                      "role": "frozen read-only backup (never written)"},
                "A": {"path": A_LABEL_DIR, "role": "current editable working labels (all edits saved here)"},
            },
            "image_size": [IMAGE_W, IMAGE_H],
            "defaults": {"confidence": DEFAULT_CONF, "iou": DEFAULT_IOU},
            "prediction_confidence_floor": PREDICTION_CONFIDENCE_FLOOR,
            "chunks": sorted({e["spatial_chunk"] for e in self.patches.values()}),
            "chunk_to_group": self.index["chunk_to_group"],
            "groups": sorted({e["original_group"] for e in self.patches.values()}),
            "final_splits": self.index["final_split_values"],
            "folds": self.index["folds"],
            "pred_statuses": PRED_STATUSES, "pred_status_labels": PRED_STATUS_LABELS,
            "patch_statuses": PATCH_STATUSES,
            "fp_note_en": FP_NOTE_EN, "fp_note_zh": FP_NOTE_ZH,
            "startup": c["startup_report"],
            "baseline_comparison_summary": c["baseline_summary"],
            "zip_review_manifest_counts": self.index["zip_review_manifest"][
                "counts_on_deduplicated_oof_union_at_review_threshold"],
            "history": self.store.history_summary(limit=15),
            "modified_patches": len(self.store.modified_patches()),
            "ui_state": self.store.review.get("ui", {}),
            "store_version": self.store.version,
        }

    def integrity(self):
        c = self.ctx
        init = read_json(os.path.join(TOOL_DIR, "review_manifest.json"), {}).get("source_fingerprints_at_initialisation", {})
        zfp = file_fingerprint(c["zip_path"])
        bad_b = []
        for f in c["b_manifest"]["files"]:
            p = os.path.join(c["b_dir"], f["path"])
            with open(p, "rb") as fh:
                if sha256_bytes(fh.read()) != f["sha256"]:
                    bad_b.append(f["path"])
        zinit = init.get("zip", {})
        return {
            "zip_now": zfp,
            "zip_unchanged_since_init": zinit.get("sha256") == zfp["sha256"] and zinit.get("mtime_ns") == zfp["mtime_ns"],
            "b_files_checked": len(c["b_manifest"]["files"]),
            "b_files_mismatching_backup_manifest": bad_b,
            "b_manifest_sha256_now": file_fingerprint(c["b_manifest_path"])["sha256"],
            "b_manifest_unchanged_since_init": init.get("b_backup_manifest", {}).get("sha256")
            == file_fingerprint(c["b_manifest_path"])["sha256"],
        }

    def exports(self):
        out = []
        if os.path.isdir(EXPORT_DIR):
            for n in sorted(os.listdir(EXPORT_DIR)):
                if n.endswith("_manifest.json"):
                    m = read_json(os.path.join(EXPORT_DIR, n), {})
                    out.append({"name": m.get("export_name"), "generated_at_utc": m.get("generated_at_utc"),
                                "purpose": m.get("purpose"), "note": m.get("note"),
                                "zip": m.get("outputs", {}).get("zip"),
                                "verified": m.get("self_verification", {}).get("ok"),
                                "patches_modified_vs_B": m.get("edit_summary", {}).get("patches_modified_vs_B_count"),
                                "box_count": m.get("counts", {}).get("box_count")})
        return out

    # ------------------------------------------------------------------ mutations
    def post(self, path, body):
        s = self.store
        source = str(body.get("source") or "ui")[:40]
        m = re.match(r"^/api/patch/([^/]+)/(labels|restore_b|adopt)$", path)
        if m:
            pid = self._pid(m.group(1))
            if m.group(2) == "labels":
                r = s.set_labels(pid, body.get("boxes"), str(body.get("action") or "edit")[:40],
                                 body.get("base_sha"), body.get("detail") or {}, source)
            elif m.group(2) == "adopt":
                k = body.get("k")
                preds = self.patches[pid]["predictions"]
                if not isinstance(k, int) or not 0 <= k < len(preds):
                    raise LabelError("unknown prediction")
                p = preds[k]
                nb = clip_box_to_image(p)
                boxes = list(s.a_boxes[pid]) + [nb]
                r = s.set_labels(pid, boxes, "adopt_prediction", body.get("base_sha"),
                                 {"prediction_k": k, "confidence": p["confidence"], "fold_ids": p["fold_ids"],
                                  "prediction_box": {q2: p[q2] for q2 in ("cx", "cy", "w", "h")},
                                  "clipped_to_image": nb != {"class_id": 0, **{q2: p[q2] for q2 in ("cx", "cy", "w", "h")}}},
                                 source)
            else:
                r = s.restore_patch_to_b(pid, body.get("base_sha"), source)
            return r
        if path == "/api/restore_all_b":
            if body.get("confirm") != "RESTORE ALL A TO B":
                raise LabelError("confirmation text missing")
            return s.restore_all_to_b(source)
        if path == "/api/undo":
            return s.undo(source)
        if path == "/api/redo":
            return s.redo(source)
        if path == "/api/clear_redo":
            if body.get("confirm") != "CLEAR REDO":
                raise LabelError("confirmation text missing")
            return s.clear_redo(source)
        if path == "/api/review":
            pid = self._pid(body.get("patch_id", ""))
            k = body.get("k")
            return s.set_review(body.get("target"), pid, k, body.get("status"), body.get("note", ""), source)
        if path == "/api/ui_state":
            ui = body.get("ui")
            if isinstance(ui, dict) and len(json.dumps(ui)) < 5000:
                s.set_ui_state(ui)
            return {"ok": True}
        if path == "/api/export":
            with self.export_lock:
                return export_a(s, self.index, self.ctx["zip_path"], self.ctx["b_dir"], self.ctx["b_manifest_path"],
                                note=str(body.get("note") or "")[:500],
                                purpose=str(body.get("purpose") or "final_labels")[:60])
        raise KeyError(path)

    def _pid(self, pid):
        if not PID_RE.match(pid or "") or pid not in self.patches:
            raise KeyError("unknown patch %r" % pid)
        return pid


def _float(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        raise LabelError("not a number: %r" % v)


def make_handler(app, port):
    allowed_hosts = {"127.0.0.1:%d" % port, "localhost:%d" % port}

    class Handler(BaseHTTPRequestHandler):
        server_version = "SSSReview/1.0"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # quieter: only errors
            if args and str(args[1] if len(args) > 1 else "").startswith(("4", "5")):
                BaseHTTPRequestHandler.log_message(self, fmt, *args)

        def _send(self, code, body, ctype="application/json; charset=utf-8", extra=None):
            if isinstance(body, (dict, list)):
                body = json.dumps(body, ensure_ascii=False).encode("utf-8")
            elif isinstance(body, str):
                body = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy",
                             "default-src 'self'; img-src 'self'; style-src 'self'; script-src 'self'; "
                             "connect-src 'self'; font-src 'self'; object-src 'none'; frame-ancestors 'none'")
            self.send_header("Referrer-Policy", "no-referrer")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _host_ok(self):
            if self.headers.get("Host", "") not in allowed_hosts:
                self.close_connection = True
                self._send(403, {"error": "forbidden host header"})
                return False
            return True

        def do_GET(self):
            if not self._host_ok():
                return
            u = urlparse(self.path)
            q = {k: v[-1] for k, v in parse_qs(u.query).items()}
            try:
                if u.path in STATIC_FILES:
                    rel, ctype = STATIC_FILES[u.path]
                    with open(os.path.join(TOOL_DIR, rel), "rb") as fh:
                        return self._send(200, fh.read(), ctype, {"Cache-Control": "no-cache"})
                if u.path.startswith("/api/image/") and u.path.endswith(".png"):
                    pid = app._pid(u.path[len("/api/image/"):-4])
                    data, etag = app.image(pid)
                    if self.headers.get("If-None-Match") == etag:
                        self.send_response(304)
                        self.send_header("ETag", etag)
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    return self._send(200, data, "image/png",
                                      {"Cache-Control": "private, max-age=86400", "ETag": etag,
                                       "X-Source-Zip-Member": app.patches[pid]["zip_image_member"]})
                if u.path == "/api/meta":
                    return self._send(200, app.meta())
                if u.path == "/api/list":
                    return self._send(200, app.list(q))
                if u.path.startswith("/api/patch/"):
                    return self._send(200, app.detail(app._pid(u.path[len("/api/patch/"):]), q))
                if u.path == "/api/history":
                    return self._send(200, app.store.history_summary(limit=int(_float(q.get("limit")) or 50)))
                if u.path == "/api/exports":
                    return self._send(200, app.exports())
                if u.path == "/api/integrity":
                    return self._send(200, app.integrity())
                if u.path == "/api/baseline_comparison":
                    return self._send(200, read_json(os.path.join(TOOL_DIR, "index", "baseline_comparison.json"), {}))
                if u.path == "/api/data_check_report":
                    return self._send(200, read_json(os.path.join(TOOL_DIR, "index", "data_check_report.json"), {}))
                if u.path == "/favicon.ico":
                    return self._send(204, b"", "image/x-icon")
                return self._send(404, {"error": "not found"})
            except KeyError as e:
                return self._send(404, {"error": "not found: %s" % e})
            except LabelError as e:
                return self._send(400, {"error": str(e)})

        def do_POST(self):
            # always consume the body first so a rejected request cannot corrupt the keep-alive stream
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                n = -1
            if n < 0 or n > 2_000_000:
                self.close_connection = True
                return self._send(413, {"error": "bad or too large request body"})
            raw = self.rfile.read(n) if n else b""
            if not self._host_ok():
                return
            # custom header => cross-site pages cannot send this without a CORS preflight (which we never allow)
            if self.headers.get("X-SSS-Tool") != "1" or not self.headers.get("Content-Type", "").startswith("application/json"):
                return self._send(403, {"error": "missing X-SSS-Tool header or JSON content type"})
            try:
                body = json.loads(raw or b"{}")
                if not isinstance(body, dict):
                    raise LabelError("JSON object expected")
                r = app.post(urlparse(self.path).path, body)
                r = dict(r)
                r["store_version"] = app.store.version
                r["history"] = app.store.history_summary(limit=15)
                r["modified_patches"] = len(app.store.modified_patches())
                return self._send(200, r)
            except KeyError as e:
                return self._send(404, {"error": "not found: %s" % e})
            except (LabelError, ValueError) as e:
                return self._send(400, {"error": str(e)})
            except ConflictError as e:
                return self._send(409, {"error": str(e)})
            except FileExistsError as e:
                return self._send(409, {"error": "refusing to overwrite existing export: %s" % e})

    return Handler


def run_server(app, host, port):
    httpd = ThreadingHTTPServer((host, port), make_handler(app, port))
    httpd.daemon_threads = True
    return httpd
