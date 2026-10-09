#!/usr/bin/env python3
"""Local, persistent manual review server for the unfiltered Costa patches."""

from __future__ import annotations

import argparse
import csv
import json
import threading
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


REVIEW_DIR = Path(__file__).resolve().parent
DATASET_DIR = REVIEW_DIR.parent
CONTACTS_CSV = DATASET_DIR / "accepted_contacts.csv"
RAW_DIR = DATASET_DIR / "patches_raw"
PREVIEW_DIR = DATASET_DIR / "patches_preview_marked"
DECISIONS_JSON = REVIEW_DIR / "review_decisions.json"
DECISIONS_CSV = REVIEW_DIR / "review_decisions.csv"
SELECTED_CSV = REVIEW_DIR / "selected_contacts.csv"
REJECTED_CSV = REVIEW_DIR / "rejected_contacts.csv"
SUMMARY_JSON = REVIEW_DIR / "review_summary.json"
INDEX_HTML = REVIEW_DIR / "index.html"
APP_JS = REVIEW_DIR / "app.js"
STYLES_CSS = REVIEW_DIR / "styles.css"

STATE_LOCK = threading.RLock()
CONTACTS: list[dict[str, str]] = []
CONTACT_BY_UID: dict[str, dict[str, str]] = {}


def load_contacts() -> None:
    global CONTACTS, CONTACT_BY_UID
    with CONTACTS_CSV.open(encoding="utf-8", newline="") as handle:
        CONTACTS = list(csv.DictReader(handle))
    CONTACT_BY_UID = {row["uid"]: row for row in CONTACTS}


def read_decisions() -> dict[str, str]:
    if not DECISIONS_JSON.exists():
        return {}
    try:
        data = json.loads(DECISIONS_JSON.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(uid): status for uid, status in data.items() if uid in CONTACT_BY_UID and status in {"keep", "drop"}}


def counts_for(decisions: dict[str, str]) -> dict[str, int]:
    keep = sum(status == "keep" for status in decisions.values())
    drop = sum(status == "drop" for status in decisions.values())
    return {"total": len(CONTACTS), "keep": keep, "drop": drop, "undecided": len(CONTACTS) - keep - drop}


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_review_exports(decisions: dict[str, str]) -> dict[str, int]:
    now = datetime.now(timezone.utc).isoformat()
    decision_rows = [
        {"uid": row["uid"], "id": row["id"], "status": decisions[row["uid"]], "saved_at_utc": now}
        for row in CONTACTS
        if row["uid"] in decisions
    ]
    with DECISIONS_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["uid", "id", "status", "saved_at_utc"])
        writer.writeheader()
        writer.writerows(decision_rows)

    output_fields = list(CONTACTS[0].keys()) + ["review_status"] if CONTACTS else ["review_status"]
    for path, wanted_status in ((SELECTED_CSV, "keep"), (REJECTED_CSV, "drop")):
        rows = []
        for row in CONTACTS:
            if decisions.get(row["uid"]) == wanted_status:
                output = dict(row)
                output["review_status"] = wanted_status
                rows.append(output)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=output_fields)
            writer.writeheader()
            writer.writerows(rows)

    counts = counts_for(decisions)
    summary = {
        "dataset": "costa",
        "source": str(CONTACTS_CSV),
        "decisions": counts,
        "meaning": {
            "keep": "checkbox checked when leaving the image",
            "drop": "checkbox unchecked when leaving the image",
            "undecided": "not yet visited or not yet saved",
        },
        "updated_at_utc": now,
    }
    atomic_write(SUMMARY_JSON, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return counts


def save_decision(uid: str, status: str) -> dict[str, int]:
    if uid not in CONTACT_BY_UID:
        raise ValueError("unknown uid")
    if status not in {"keep", "drop"}:
        raise ValueError("status must be keep or drop")
    with STATE_LOCK:
        decisions = read_decisions()
        decisions[uid] = status
        atomic_write(DECISIONS_JSON, json.dumps(decisions, ensure_ascii=False, indent=2) + "\n")
        return write_review_exports(decisions)


def export_current() -> dict[str, int]:
    with STATE_LOCK:
        return write_review_exports(read_decisions())


def json_response(handler: BaseHTTPRequestHandler, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def serve_file(handler: BaseHTTPRequestHandler, path: Path, content_type: str) -> None:
    if not path.is_file():
        json_response(handler, {"error": "not found"}, HTTPStatus.NOT_FOUND)
        return
    body = path.read_bytes()
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-cache")
    handler.end_headers()
    handler.wfile.write(body)


class ReviewHandler(BaseHTTPRequestHandler):
    server_version = "CostaPatchReview/1.0"

    def log_message(self, format: str, *args) -> None:
        print(f"[{self.log_date_time_string()}] {format % args}", flush=True)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            serve_file(self, INDEX_HTML, "text/html; charset=utf-8")
            return
        if path == "/app.js":
            serve_file(self, APP_JS, "text/javascript; charset=utf-8")
            return
        if path == "/styles.css":
            serve_file(self, STYLES_CSS, "text/css; charset=utf-8")
            return
        if path == "/api/items":
            items = []
            for index, row in enumerate(CONTACTS):
                filename = Path(row["raw_patch_file"]).name
                items.append(
                    {
                        "index": index,
                        "uid": row["uid"],
                        "id": row["id"],
                        "raw_url": f"/images/raw/{filename}",
                        "preview_url": f"/images/preview/{filename}",
                    }
                )
            json_response(self, {"items": items, "counts": counts_for(read_decisions())})
            return
        if path == "/api/contact-rows":
            json_response(self, {"rows": CONTACTS})
            return
        if path == "/api/state":
            with STATE_LOCK:
                decisions = read_decisions()
                json_response(self, {"decisions": decisions, "counts": counts_for(decisions)})
            return
        if path.startswith("/images/raw/") or path.startswith("/images/preview/"):
            parts = path.split("/")
            if len(parts) != 4 or not parts[3]:
                json_response(self, {"error": "invalid image path"}, HTTPStatus.BAD_REQUEST)
                return
            filename = Path(unquote(parts[3])).name
            directory = RAW_DIR if parts[2] == "raw" else PREVIEW_DIR
            candidate = (directory / filename).resolve()
            if directory.resolve() not in candidate.parents or candidate.suffix.lower() != ".png":
                json_response(self, {"error": "invalid image path"}, HTTPStatus.BAD_REQUEST)
                return
            serve_file(self, candidate, "image/png")
            return
        if path == "/api/export":
            with STATE_LOCK:
                json_response(self, {"counts": export_current()})
            return
        json_response(self, {"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 1_000_000:
                raise ValueError("request too large")
            body = self.rfile.read(length) if length else b"{}"
            payload = json.loads(body.decode("utf-8"))
            if parsed.path == "/api/decision":
                counts = save_decision(str(payload.get("uid", "")), str(payload.get("status", "")))
                json_response(self, {"ok": True, "counts": counts})
                return
            if parsed.path == "/api/export":
                json_response(self, {"ok": True, "counts": export_current()})
                return
            json_response(self, {"error": "not found"}, HTTPStatus.NOT_FOUND)
        except (ValueError, json.JSONDecodeError) as exc:
            json_response(self, {"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # pragma: no cover - local server guard
            json_response(self, {"error": f"{type(exc).__name__}: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the Costa patch manual-review platform")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()
    load_contacts()
    server = ThreadingHTTPServer((args.host, args.port), ReviewHandler)
    print(f"Costa patch review: http://{args.host}:{args.port}", flush=True)
    print(f"items: {len(CONTACTS):,}; review directory: {REVIEW_DIR}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
