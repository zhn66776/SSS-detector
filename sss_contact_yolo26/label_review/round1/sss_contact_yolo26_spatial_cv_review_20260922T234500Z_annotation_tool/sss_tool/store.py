"""Working copy A: initialisation from B, atomic label saves, undo/redo history,
operation log and review state.

B is only read. A lives in TOOL_DIR/working_A and is never re-initialised once it
exists. Every label or review change is one history operation:
  history_state.json  - undo/redo stacks (atomically rewritten)
  undo_log.jsonl      - append-only audit log (do / undo / redo / reconcile)
"""
import json
import os
import threading

from .common import (A_DIR, A_LABEL_DIR, HISTORY_STATE, LABEL_SUBDIR,
                     REVIEW_STATE, TOOL_DIR, UNDO_LOG, append_jsonl,
                     atomic_write_bytes, atomic_write_json, read_json,
                     sha256_bytes, utc_now, utc_stamp)
from .labels import LabelError, format_boxes, parse_label_text, validate_boxes

PRED_STATUSES = ["unreviewed", "true_contact_missed_label", "confirmed_negative", "ambiguous"]
PRED_STATUS_LABELS = {
    "unreviewed": "Not reviewed",
    "true_contact_missed_label": "True contact / missing annotation",
    "confirmed_negative": "Confirmed negative",
    "ambiguous": "Ambiguous",
}
PATCH_STATUSES = ["unreviewed", "reviewed", "needs_followup"]
MAX_NOTE_LEN = 2000


class ConflictError(RuntimeError):
    pass


def load_b(b_dir, b_manifest):
    """Read every B label and verify it against backup_manifest.json."""
    out = {}
    bad = []
    for f in b_manifest["files"]:
        rel = f["path"]
        pid = os.path.basename(rel)[:-4]
        with open(os.path.join(b_dir, rel), "rb") as fh:
            data = fh.read()
        if sha256_bytes(data) != f["sha256"] or len(data) != f["bytes"]:
            bad.append(rel)
        out[pid] = data
    return out, bad


def init_or_resume_a(b_bytes, log=print):
    """Create A from B if it does not exist; otherwise resume it untouched."""
    record = {"checked_at_utc": utc_now(), "a_label_dir": A_LABEL_DIR}
    if not os.path.exists(A_DIR):
        staging = os.path.join(TOOL_DIR, "working_A.__initialising_%s" % utc_stamp())
        os.makedirs(os.path.join(staging, LABEL_SUBDIR))
        for pid, data in sorted(b_bytes.items()):
            p = os.path.join(staging, LABEL_SUBDIR, pid + ".txt")
            with open(p, "xb") as fh:
                fh.write(data)
        mismatch = []
        for pid, data in b_bytes.items():
            with open(os.path.join(staging, LABEL_SUBDIR, pid + ".txt"), "rb") as fh:
                if sha256_bytes(fh.read()) != sha256_bytes(data):
                    mismatch.append(pid)
        if mismatch:
            raise RuntimeError("A initialisation copy mismatch (staging kept at %s): %s" % (staging, mismatch[:5]))
        init = {"created_at_utc": utc_now(), "created_from": "B", "label_file_count": len(b_bytes),
                "identical_to_B": True,
                "files": {pid: sha256_bytes(d) for pid, d in sorted(b_bytes.items())}}
        with open(os.path.join(staging, "A_init_record.json"), "x", encoding="utf-8") as fh:
            json.dump(init, fh, indent=1)
        os.rename(staging, A_DIR)
        record.update({"action": "initialised_from_B", "label_file_count": len(b_bytes),
                       "identical_to_B_at_init": True})
        log("initialised working copy A from B (%d files, all SHA-256 identical)" % len(b_bytes))
    else:
        record["action"] = "resumed_existing_A"
        log("resuming existing working copy A (not re-initialised): %s" % A_DIR)
    names = {n[:-4] for n in os.listdir(A_LABEL_DIR) if n.endswith(".txt")}
    missing = sorted(set(b_bytes) - names)
    extra = sorted(names - set(b_bytes))
    if missing:
        raise RuntimeError("working A is missing %d label files (e.g. %s). A is never overwritten "
                           "automatically; see README 'Recovery'." % (len(missing), missing[:5]))
    a_bytes = {}
    for pid in b_bytes:
        with open(os.path.join(A_LABEL_DIR, pid + ".txt"), "rb") as fh:
            a_bytes[pid] = fh.read()
    diff = sorted(p for p in b_bytes if a_bytes[p] != b_bytes[p])
    record.update({"label_files_in_A": len(names), "extra_files_in_A": extra,
                   "patches_differing_from_B": len(diff)})
    init_rec = read_json(os.path.join(A_DIR, "A_init_record.json"))
    if init_rec:
        record["A_init_record"] = {k: init_rec[k] for k in ("created_at_utc", "created_from",
                                                             "label_file_count", "identical_to_B")}
    return a_bytes, record


class Store:
    def __init__(self, index, b_bytes, a_bytes):
        self.lock = threading.RLock()
        self.index = index
        self.b_bytes = b_bytes
        self.a_bytes = a_bytes
        self.a_boxes = {}
        self.b_boxes = {}
        for pid in b_bytes:
            self.b_boxes[pid] = parse_label_text(b_bytes[pid].decode("ascii"))
            self.a_boxes[pid] = parse_label_text(a_bytes[pid].decode("ascii"))
        self.history = read_json(HISTORY_STATE) or {"seq": 0, "undo": [], "redo": []}
        self.review = read_json(REVIEW_STATE) or {"format": 1, "predictions": {}, "patches": {}, "ui": {}}
        self.version = 0
        self.listeners = []  # callables(pid) for cache invalidation
        if not os.path.exists(UNDO_LOG):
            append_jsonl(UNDO_LOG, {"event": "log_created", "time_utc": utc_now(),
                                    "note": "append-only operation log for working copy A"})
        if not os.path.exists(HISTORY_STATE):
            self._persist_history()
        if not os.path.exists(REVIEW_STATE):
            self._persist_review()
        self.reconcile_report = self._reconcile()

    # ---------- helpers ----------
    def label_path(self, pid):
        return os.path.join(A_LABEL_DIR, pid + ".txt")

    def modified_patches(self):
        return sorted(p for p in self.a_bytes if self.a_bytes[p] != self.b_bytes[p])

    def is_modified(self, pid):
        return self.a_bytes[pid] != self.b_bytes[pid]

    def _notify(self, pid):
        self.version += 1
        for fn in self.listeners:
            fn(pid)

    def _write_label(self, pid, data):
        atomic_write_bytes(self.label_path(pid), data)
        self.a_bytes[pid] = data
        self.a_boxes[pid] = parse_label_text(data.decode("ascii"))
        self._notify(pid)

    def _persist_history(self):
        atomic_write_json(HISTORY_STATE, self.history)

    def _persist_review(self):
        atomic_write_json(REVIEW_STATE, self.review)

    def _labels_snapshot(self, data):
        return {"text": data.decode("ascii"), "sha256": sha256_bytes(data),
                "boxes": parse_label_text(data.decode("ascii"))}

    def _new_op(self, **kw):
        self.history["seq"] += 1
        op = {"op_id": self.history["seq"], "time_utc": utc_now()}
        op.update(kw)
        return op

    # ---------- crash reconciliation ----------
    def _reconcile(self):
        """History/log are persisted before the label file is replaced. If the
        process died in between, roll the file forward to the recorded result."""
        report = {"checked_at_utc": utc_now(), "rolled_forward": [], "conflicts": []}
        last = None
        if os.path.exists(UNDO_LOG):
            with open(UNDO_LOG, "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        try:
                            last = json.loads(line)
                        except ValueError:
                            report["conflicts"].append("unparseable trailing log line (ignored)")
        if not last or last.get("kind") not in ("labels", "bulk_labels"):
            return report
        expected = last.get("resulting_labels") or {}
        for pid, exp in expected.items():
            cur = self.a_bytes.get(pid)
            if cur is None:
                continue
            exp_b = exp["text"].encode("ascii")
            if cur == exp_b:
                continue
            alt = (last.get("previous_labels") or {}).get(pid)
            if alt is not None and cur == alt["text"].encode("ascii"):
                self._write_label(pid, exp_b)
                report["rolled_forward"].append(pid)
            else:
                report["conflicts"].append(pid)
        if report["rolled_forward"]:
            append_jsonl(UNDO_LOG, {"event": "startup_roll_forward", "time_utc": utc_now(),
                                    "op_id": last.get("op_id"), "patches": report["rolled_forward"]})
        return report

    # ---------- label edits ----------
    def set_labels(self, pid, boxes, action, base_sha=None, detail=None, source="ui"):
        with self.lock:
            if pid not in self.a_bytes:
                raise KeyError(pid)
            cur = self.a_bytes[pid]
            if base_sha is not None and base_sha != sha256_bytes(cur):
                raise ConflictError("patch %s changed since it was loaded; reload and retry" % pid)
            new_boxes = validate_boxes(boxes)
            data = format_boxes(new_boxes)
            if data == cur:
                return {"changed": False}
            op = self._new_op(kind="labels", action=action, patch_id=pid, source=source,
                              detail=detail or {},
                              before=self._labels_snapshot(cur), after=self._labels_snapshot(data))
            self._commit(op, {pid: (cur, data)}, "do")
            return {"changed": True, "op_id": op["op_id"]}

    def restore_patch_to_b(self, pid, base_sha=None, source="ui"):
        return self.set_labels(pid, self.b_boxes[pid], "restore_patch_to_B", base_sha,
                               {"note": "patch restored to frozen backup B"}, source)

    def restore_all_to_b(self, source="ui"):
        with self.lock:
            changed = self.modified_patches()
            if not changed:
                return {"changed": False, "patches": 0}
            op = self._new_op(kind="bulk_labels", action="restore_all_A_to_B", patch_id=None, source=source,
                              changes=[{"patch_id": p, "before": self._labels_snapshot(self.a_bytes[p]),
                                        "after": self._labels_snapshot(self.b_bytes[p])} for p in changed])
            self._commit(op, {p: (self.a_bytes[p], self.b_bytes[p]) for p in changed}, "do")
            return {"changed": True, "patches": len(changed), "op_id": op["op_id"]}

    def _commit(self, op, label_changes, event, review_change=None):
        """Persist history + log first, then apply the change atomically."""
        if event == "do":
            self.history["undo"].append(op)
            self.history["redo"] = []
        self._persist_history()
        log = {"event": event, "time_utc": utc_now(), "op_id": op["op_id"], "op_time_utc": op["time_utc"],
               "kind": op["kind"], "action": op["action"], "patch_id": op.get("patch_id"),
               "source": op.get("source")}
        if label_changes:
            log["previous_labels"] = {p: self._labels_snapshot(a) for p, (a, b) in label_changes.items()}
            log["resulting_labels"] = {p: self._labels_snapshot(b) for p, (a, b) in label_changes.items()}
            if op.get("detail"):
                log["detail"] = op["detail"]
        if review_change:
            log["review_before"], log["review_after"] = review_change
        append_jsonl(UNDO_LOG, log)
        for p, (a, b) in label_changes.items():
            self._write_label(p, b)
        if review_change:
            self._apply_review(review_change[1])
            self._persist_review()

    # ---------- review status ----------
    def _review_key(self, target, pid, k=None):
        return "%s#%d" % (pid, k) if target == "prediction" else pid

    def get_review(self, target, pid, k=None):
        bucket = self.review["predictions" if target == "prediction" else "patches"]
        return bucket.get(self._review_key(target, pid, k))

    def _apply_review(self, state):
        bucket = self.review["predictions" if state["target"] == "prediction" else "patches"]
        key = state["key"]
        if state.get("status") in (None, "unreviewed") and not state.get("note"):
            bucket.pop(key, None)
        else:
            bucket[key] = {k: state[k] for k in ("status", "note", "updated_at_utc", "prediction") if k in state}

    def set_review(self, target, pid, k, status, note, source="ui"):
        with self.lock:
            if pid not in self.a_bytes:
                raise KeyError(pid)
            allowed = PRED_STATUSES if target == "prediction" else PATCH_STATUSES
            if target not in ("prediction", "patch"):
                raise LabelError("target must be prediction or patch")
            if status not in allowed:
                raise LabelError("status must be one of %s" % allowed)
            note = (note or "").strip()[:MAX_NOTE_LEN]
            key = self._review_key(target, pid, k)
            before = dict(self.get_review(target, pid, k) or {"status": "unreviewed", "note": ""})
            before.update({"target": target, "key": key})
            after = {"target": target, "key": key, "status": status, "note": note, "updated_at_utc": utc_now()}
            if target == "prediction":
                preds = self.index["patches"][pid]["predictions"]
                if not isinstance(k, int) or not 0 <= k < len(preds):
                    raise LabelError("unknown prediction")
                p = preds[k]
                after["prediction"] = {q: p[q] for q in ("cx", "cy", "w", "h", "confidence", "fold_ids")}
            if before.get("status") == status and (before.get("note") or "") == note:
                return {"changed": False}
            op = self._new_op(kind="review", action="set_%s_review" % target, patch_id=pid, source=source,
                              before=before, after=after)
            self._commit(op, {}, "do", (before, after))
            return {"changed": True, "op_id": op["op_id"]}

    # ---------- undo / redo ----------
    def _labels_pairs(self, op, direction):
        if op["kind"] == "labels":
            items = [(op["patch_id"], op["before"], op["after"])]
        elif op["kind"] == "bulk_labels":
            items = [(c["patch_id"], c["before"], c["after"]) for c in op["changes"]]
        else:
            return {}
        out = {}
        for pid, before, after in items:
            src, dst = (after, before) if direction == "undo" else (before, after)
            cur = self.a_bytes[pid]
            if cur != src["text"].encode("ascii"):
                raise ConflictError("cannot %s op %d: patch %s no longer matches the recorded state"
                                    % (direction, op["op_id"], pid))
            out[pid] = (cur, dst["text"].encode("ascii"))
        return out

    def undo(self, source="ui"):
        with self.lock:
            if not self.history["undo"]:
                return {"changed": False}
            op = self.history["undo"][-1]
            changes = self._labels_pairs(op, "undo")
            self.history["undo"].pop()
            self.history["redo"].append(op)
            review = (op["after"], op["before"]) if op["kind"] == "review" else None
            self._commit(op, changes, "undo", review)
            return {"changed": True, "op": summarize_op(op)}

    def redo(self, source="ui"):
        with self.lock:
            if not self.history["redo"]:
                return {"changed": False}
            op = self.history["redo"][-1]
            changes = self._labels_pairs(op, "redo")
            self.history["redo"].pop()
            self.history["undo"].append(op)
            review = (op["before"], op["after"]) if op["kind"] == "review" else None
            self._commit(op, changes, "redo", review)
            return {"changed": True, "op": summarize_op(op)}

    def clear_redo(self, source="ui"):
        """Discard the redo stack (the discarded ops are kept verbatim in the log)."""
        with self.lock:
            if not self.history["redo"]:
                return {"changed": False, "discarded": 0}
            discarded = self.history["redo"]
            self.history["redo"] = []
            self._persist_history()
            append_jsonl(UNDO_LOG, {"event": "clear_redo", "time_utc": utc_now(), "source": source,
                                    "discarded_ops": discarded})
            return {"changed": True, "discarded": len(discarded)}

    def history_summary(self, limit=30, pid=None):
        ops = self.history["undo"]
        if pid:
            ops = [o for o in ops if o.get("patch_id") == pid or
                   any(c["patch_id"] == pid for c in o.get("changes", []))]
        return {
            "undo_depth": len(self.history["undo"]),
            "redo_depth": len(self.history["redo"]),
            "next_undo": summarize_op(self.history["undo"][-1]) if self.history["undo"] else None,
            "next_redo": summarize_op(self.history["redo"][-1]) if self.history["redo"] else None,
            "recent": [summarize_op(o) for o in reversed(ops[-limit:])],
        }

    def edit_counts(self):
        """Per-patch number of label ops and last edit time (current undo stack)."""
        out = {}
        for o in self.history["undo"]:
            pids = [o["patch_id"]] if o["kind"] == "labels" else [c["patch_id"] for c in o.get("changes", [])]
            for p in pids:
                if p:
                    e = out.setdefault(p, {"label_ops": 0, "last_edit_utc": None, "actions": []})
                    e["label_ops"] += 1
                    e["last_edit_utc"] = o["time_utc"]
                    e["actions"].append(o["action"])
        return out

    def set_ui_state(self, ui):
        with self.lock:
            self.review["ui"] = ui
            self._persist_review()


def summarize_op(op):
    s = {"op_id": op["op_id"], "time_utc": op["time_utc"], "kind": op["kind"], "action": op["action"],
         "patch_id": op.get("patch_id"), "source": op.get("source")}
    if op["kind"] == "labels":
        s["boxes_before"] = len(op["before"]["boxes"])
        s["boxes_after"] = len(op["after"]["boxes"])
    elif op["kind"] == "bulk_labels":
        s["patches"] = len(op["changes"])
    else:
        s["status_before"] = op["before"].get("status")
        s["status_after"] = op["after"].get("status")
        s["key"] = op["after"].get("key")
    return s
