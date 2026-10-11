#!/usr/bin/env python3
"""SSS contact annotation & OOF-assisted review platform - local server.

    python3 serve.py --host 127.0.0.1 --port 8767     # start, then open http://127.0.0.1:8767/
    python3 serve.py --check-only                     # run all start-up checks, then exit
    python3 serve.py --stop                           # stop a server started from this directory

Only 127.0.0.1 is accepted as host. Source ZIP and backup B are read only.
"""
import argparse
import json
import os
import signal
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sss_tool import TOOL_VERSION  # noqa: E402
from sss_tool.common import (A_LABEL_DIR, DEFAULT_B_DIR, DEFAULT_ZIP,  # noqa: E402
                             INDEX_DIR, PID_FILE, REVIEW_MANIFEST, TOOL_DIR,
                             atomic_write_json, file_fingerprint, read_json,
                             utc_now)
from sss_tool.data_index import CHECK_REPORT_PATH, INDEX_PATH, load_or_build_index  # noqa: E402
from sss_tool.evaluate import Evaluator, compare_with_zip_baseline  # noqa: E402
from sss_tool.store import Store, init_or_resume_a, load_b  # noqa: E402


def log(msg):
    print("[sss-review] %s" % msg, flush=True)


def startup(zip_path, b_dir):
    if not os.path.isfile(zip_path):
        raise SystemExit("source ZIP not found: %s" % zip_path)
    b_manifest_path = os.path.join(b_dir, "backup_manifest.json")
    if not os.path.isfile(b_manifest_path):
        raise SystemExit("backup B manifest not found: %s" % b_manifest_path)
    for p in (zip_path, b_dir):
        rp = os.path.realpath(p)
        if rp == TOOL_DIR or rp.startswith(TOOL_DIR + os.sep):
            raise SystemExit("source data must not live inside the platform directory")

    log("fingerprinting source ZIP (read only) ...")
    zfp = file_fingerprint(zip_path)
    bmfp = file_fingerprint(b_manifest_path)
    with open(b_manifest_path, "r", encoding="utf-8") as fh:
        b_manifest = json.load(fh)
    log("verifying backup B against backup_manifest.json (read only) ...")
    b_bytes, bad = load_b(b_dir, b_manifest)
    if bad:
        raise SystemExit("backup B does not match its manifest for %d files (e.g. %s); refusing to start"
                         % (len(bad), bad[:3]))
    log("B ok: %d label files, all SHA-256 match backup_manifest.json" % len(b_bytes))

    index, check_report = load_or_build_index(zip_path, zfp["sha256"], log)
    if set(index["patches"]) != set(b_bytes):
        raise SystemExit("patch ids in ZIP index and backup B differ")
    a_bytes, a_record = init_or_resume_a(b_bytes, log)
    store = Store(index, b_bytes, a_bytes)
    if store.reconcile_report["rolled_forward"] or store.reconcile_report["conflicts"]:
        log("reconcile: %s" % store.reconcile_report)
    evaluator = Evaluator(index, store)

    log("recomputing baseline (conf 0.20 / IoU 0.50, B labels) and comparing with ZIP artefacts ...")
    cmp = compare_with_zip_baseline(index, zip_path, store.b_boxes)
    atomic_write_json(os.path.join(INDEX_DIR, "baseline_comparison.json"), cmp)
    baseline_summary = {"all_equal": cmp["all_equal"], "totals": cmp["totals"],
                        "platform_summary": cmp["platform_summary"],
                        "per_patch_differences": len(cmp["review_data_baseline_per_patch_differences"])}
    log("baseline vs ZIP: %s" % ("identical" if cmp["all_equal"] else "DIFFERENCES - see index/baseline_comparison.json"))

    prev = read_json(REVIEW_MANIFEST, {})
    init_fp = prev.get("source_fingerprints_at_initialisation") or {"recorded_at_utc": utc_now(), "zip": zfp,
                                                                    "b_backup_manifest": bmfp}
    zip_same = init_fp["zip"]["sha256"] == zfp["sha256"] and init_fp["zip"]["mtime_ns"] == zfp["mtime_ns"]
    bm_same = init_fp["b_backup_manifest"]["sha256"] == bmfp["sha256"]
    startup_report = {
        "started_at_utc": utc_now(),
        "zip_unchanged_since_initialisation": zip_same,
        "b_manifest_unchanged_since_initialisation": bm_same,
        "b_all_files_match_backup_manifest": True,
        "data_checks_all_passed": check_report["all_passed"] if check_report else None,
        "a": a_record,
        "reconcile": store.reconcile_report,
        "modified_patches_vs_B": len(store.modified_patches()),
        "history_ops": store.history["seq"],
    }
    manifest = {
        "tool": "SSS contact annotation & OOF-assisted review platform",
        "tool_version": TOOL_VERSION,
        "platform_dir": TOOL_DIR,
        "created_at_utc": prev.get("created_at_utc") or utc_now(),
        "roles": {
            "ZIP": {"path": zip_path, "access": "read-only; images/labels/predictions read on demand, never extracted"},
            "B": {"path": b_dir, "label_dir": os.path.join(b_dir, "validation/ground_truth_labels"),
                  "backup_manifest": b_manifest_path, "access": "read-only frozen backup; never written"},
            "A": {"path": A_LABEL_DIR, "access": "editable working copy; the only labels the web UI writes"},
        },
        "source_fingerprints_at_initialisation": init_fp,
        "last_startup": startup_report,
        "index": {"patch_index": INDEX_PATH, "data_check_report": CHECK_REPORT_PATH,
                  "baseline_comparison": os.path.join(INDEX_DIR, "baseline_comparison.json"),
                  "counts": check_report.get("counts") if check_report else None},
        "state_files": {"review_state": os.path.join(TOOL_DIR, "review_state.json"),
                        "history_state": os.path.join(TOOL_DIR, "history_state.json"),
                        "operation_log": os.path.join(TOOL_DIR, "undo_log.jsonl"),
                        "exports": os.path.join(TOOL_DIR, "exports")},
        "defaults": {"confidence_threshold": 0.20, "iou_threshold": 0.50, "prediction_confidence_floor": 0.05,
                     "matching": "one-to-one; maximise #pairs with IoU >= threshold, then total IoU (Hungarian)",
                     "classes": {"0": "contact"}, "image": "150x150 8-bit grayscale PNG from the ZIP"},
        "baseline_comparison_summary": baseline_summary,
        "fp_definition": "OOF prediction unmatched to the current annotation set, selected for human review",
    }
    atomic_write_json(REVIEW_MANIFEST, manifest)
    if not zip_same or not bm_same:
        log("WARNING: source ZIP or B manifest changed since initialisation (see review_manifest.json)")
    return {"index": index, "store": store, "evaluator": evaluator, "zip_path": zip_path, "b_dir": b_dir,
            "b_manifest_path": b_manifest_path, "b_manifest": b_manifest, "startup_report": startup_report,
            "baseline_summary": baseline_summary}


def stop():
    pid = read_json(PID_FILE)
    if not pid or not pid.get("pid"):
        print("no server.pid found - nothing to stop")
        return
    try:
        os.kill(int(pid["pid"]), signal.SIGTERM)
        print("sent SIGTERM to server pid %s (port %s)" % (pid["pid"], pid.get("port")))
    except ProcessLookupError:
        print("server pid %s is not running" % pid["pid"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8767)
    ap.add_argument("--zip", default=DEFAULT_ZIP, help="source CV review ZIP (read only)")
    ap.add_argument("--backup-b", default=DEFAULT_B_DIR, help="frozen backup B directory (read only)")
    ap.add_argument("--check-only", action="store_true", help="run start-up checks and exit")
    ap.add_argument("--stop", action="store_true", help="stop the server recorded in server.pid")
    args = ap.parse_args()
    if args.stop:
        return stop()
    host = "127.0.0.1" if args.host == "localhost" else args.host
    if host != "127.0.0.1":
        raise SystemExit("refusing to bind %r: this tool only listens on 127.0.0.1" % args.host)

    if not args.check_only:
        import socket
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # same option the real server uses, so leftover TIME_WAIT connections are not mistaken for a running server
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, args.port))
        except OSError:
            running = read_json(PID_FILE) or {}
            hint = (" (server.pid says pid %s from this directory is serving it)" % running["pid"]
                    if running.get("pid") and running.get("port") == args.port else "")
            raise SystemExit(
                "port %d on %s is already in use%s.\n"
                "  - if the review server is already running, just open http://%s:%d/\n"
                "  - to stop it: python3 serve.py --stop   (or Ctrl+C in its terminal)\n"
                "  - or start on another port: python3 serve.py --host 127.0.0.1 --port 8768"
                % (args.port, host, hint, host, args.port))
        finally:
            probe.close()

    ctx = startup(os.path.abspath(args.zip), os.path.abspath(args.backup_b))
    if args.check_only:
        log("check-only: done")
        return
    from sss_tool.server import App, run_server
    app = App(ctx)
    httpd = run_server(app, host, args.port)
    atomic_write_json(PID_FILE, {"pid": os.getpid(), "host": host, "port": args.port, "started_at_utc": utc_now()})

    def _term(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _term)
    log("serving on http://%s:%d/  (Ctrl+C to stop)" % (host, args.port))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log("stopping ...")
    finally:
        httpd.server_close()
        atomic_write_json(PID_FILE, {"pid": None, "stopped_at_utc": utc_now()})


if __name__ == "__main__":
    main()
