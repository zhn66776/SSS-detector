#!/usr/bin/env python3
"""Bounded local HTTP service for native SSS tile and crop inspection."""

from __future__ import annotations

import argparse
import atexit
import io
import json
import math
import mimetypes
import os
import shutil
import subprocess
import threading
import time
from collections import Counter, OrderedDict, defaultdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
from PIL import Image
from tifffile import TiffFile


ROOT = Path(__file__).resolve().parent
TILE_SIZE = 256
MAX_WORKERS = 4
MEMORY_CACHE_ITEMS = 96


class ECWWorker:
    """One persistent, serialized ECW decoder process with a bounded SDK cache."""

    def __init__(self, decoder: str, source: Path):
        self.decoder = decoder
        self.source = source
        self.lock = threading.Lock()
        self.counter = 0
        self.process: subprocess.Popen | None = None
        self.info: dict = {}
        self.start()
        atexit.register(self.close)

    def start(self) -> None:
        self.close()
        self.process = subprocess.Popen(
            [self.decoder, "serve", str(self.source)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        line = self.process.stdout.readline() if self.process.stdout else b""
        if not line:
            error = self.process.stderr.read().decode(errors="replace") if self.process.stderr else ""
            raise RuntimeError(f"ECW worker failed to start: {error[-1000:]}")
        self.info = json.loads(line)

    @staticmethod
    def read_exact(stream, size: int) -> bytes:
        chunks = []
        remaining = size
        while remaining:
            chunk = stream.read(remaining)
            if not chunk:
                raise RuntimeError("ECW worker ended during image response")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def decode(self, x0: int, y0: int, x1: int, y1: int, width: int, height: int, band: int = 0) -> bytes:
        with self.lock:
            if not self.process or self.process.poll() is not None:
                self.start()
            assert self.process and self.process.stdin and self.process.stdout
            self.counter += 1
            request_id = str(self.counter)
            command = f"{request_id} {x0} {y0} {x1} {y1} {width} {height} {band}\n".encode()
            self.process.stdin.write(command)
            self.process.stdin.flush()
            header = self.process.stdout.readline().decode(errors="replace").strip()
            parts = header.split(" ", 2)
            if len(parts) < 3 or parts[1] != request_id:
                raise RuntimeError(f"invalid ECW worker response: {header}")
            if parts[0] != "OK":
                raise RuntimeError(parts[2])
            return self.read_exact(self.process.stdout, int(parts[2]))

    def close(self) -> None:
        process = self.process
        self.process = None
        if not process:
            return
        try:
            if process.stdin:
                process.stdin.close()
            process.wait(timeout=6)
        except Exception:
            process.kill()
            process.wait(timeout=2)


class ByteCache:
    def __init__(self, limit: int = MEMORY_CACHE_ITEMS):
        self.limit = limit
        self.values: OrderedDict[str, tuple[bytes, str]] = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key: str):
        with self.lock:
            value = self.values.pop(key, None)
            if value is not None:
                self.values[key] = value
            return value

    def put(self, key: str, value: tuple[bytes, str]) -> None:
        with self.lock:
            self.values.pop(key, None)
            self.values[key] = value
            while len(self.values) > self.limit:
                self.values.popitem(last=False)


class InspectorData:
    def __init__(self, root: Path):
        self.root = root
        self.data_dir = root / "data"
        self.static_dir = root / "static"
        self.metadata = json.loads((self.data_dir / "metadata.json").read_text())
        self.contacts = json.loads((self.data_dir / "contacts.json").read_text())
        self.by_uid = {contact["uid"]: contact for contact in self.contacts}
        self.by_dataset = defaultdict(list)
        for contact in self.contacts:
            self.by_dataset[contact["dataset"]].append(contact)
        self.rasters: dict[str, list[dict]] = {}
        for dataset_id, dataset in self.metadata["datasets"].items():
            rasters = []
            for item in dataset["rasters"]:
                raster = dict(item)
                if "path" in raster:
                    raster["absolute_path"] = root / raster["path"]
                rasters.append(raster)
            self.rasters[dataset_id] = rasters
        self.ecw_worker: ECWWorker | None = None
        costa = self.rasters.get("costa", [{}])[0]
        status = costa.get("runtime_status", {})
        if costa.get("runtime_available") and status.get("backend") == "ecw_window":
            self.ecw_worker = ECWWorker(status["decoder"], costa["absolute_path"])
        self.cache = ByteCache()
        self.semaphore = threading.BoundedSemaphore(MAX_WORKERS)
        self.stats_lock = threading.Lock()
        self.stats = {
            "started_at": time.time(),
            "tile_requests": 0,
            "crop_requests": 0,
            "cache_hits": 0,
            "tile_render_ms": [],
            "crop_render_ms": [],
        }

    def public_config(self) -> dict:
        datasets = {}
        for dataset_id, source in self.metadata["datasets"].items():
            item = {key: value for key, value in source.items() if key != "rasters"}
            native = float(item["native_resolution_m"])
            bounds = item["bounds"]
            width = bounds["right"] - bounds["left"]
            height = bounds["top"] - bounds["bottom"]
            item["max_zoom"] = max(0, math.ceil(math.log2(max(width, height) / (TILE_SIZE * native))))
            datasets[dataset_id] = item
        return {
            "crs": self.metadata["crs"],
            "datasets": datasets,
            "coverage": self.metadata["coverage"],
            "rules": self.metadata["rules"],
            "preparation": self.metadata["preparation"],
            "limits": {
                "tile_size": TILE_SIZE,
                "max_workers": MAX_WORKERS,
                "memory_cache_items": MEMORY_CACHE_ITEMS,
            },
        }

    @staticmethod
    def inside(raster: dict, e: float, n: float) -> bool:
        return raster["left"] <= e < raster["right"] and raster["bottom"] < n <= raster["top"]

    def decode_rows(self, raster: dict, row_indices: np.ndarray, x_indices: np.ndarray) -> np.ndarray:
        path = raster["absolute_path"]
        unique_rows, inverse = np.unique(row_indices, return_inverse=True)
        output_rows = np.empty((len(unique_rows), len(x_indices), 3), dtype=np.uint8)
        with path.open("rb") as handle, TiffFile(handle) as tif:
            page = tif.pages[0]
            palette = page.colormap
            if palette is None:
                palette8 = np.vstack([np.arange(256, dtype=np.uint8)] * 3)
            else:
                palette8 = (palette // 257).astype(np.uint8)
            for out_index, row in enumerate(unique_rows):
                strip = int(row) // int(page.rowsperstrip)
                handle.seek(page.dataoffsets[strip])
                encoded = handle.read(page.databytecounts[strip])
                decoded, indices, _shape = page.decode(encoded, strip)
                if decoded is None:
                    output_rows[out_index] = 255
                    continue
                local_row = int(row) - int(indices[2])
                values = decoded[0, local_row, :, 0]
                sampled = values[x_indices]
                output_rows[out_index] = palette8[:, sampled].T
        return output_rows[inverse]

    def render_moray(self, bounds: tuple[float, float, float, float], width: int, height: int) -> Image.Image:
        left, bottom, right, top = bounds
        xs = left + (np.arange(width) + 0.5) * (right - left) / width
        ys = top - (np.arange(height) + 0.5) * (top - bottom) / height
        image = np.full((height, width, 3), 248, dtype=np.uint8)
        for raster in self.rasters["moray"]:
            xmask = (xs >= raster["left"]) & (xs < raster["right"])
            ymask = (ys <= raster["top"]) & (ys > raster["bottom"])
            if not np.any(xmask) or not np.any(ymask):
                continue
            source_x = np.floor((xs[xmask] - raster["left"]) / raster["resolution_x"]).astype(int)
            source_y = np.floor((raster["top"] - ys[ymask]) / raster["resolution_y"]).astype(int)
            source_x = np.clip(source_x, 0, raster["width"] - 1)
            source_y = np.clip(source_y, 0, raster["height"] - 1)
            pixels = self.decode_rows(raster, source_y, source_x)
            image[np.ix_(ymask, xmask)] = pixels
        return Image.fromarray(image)

    def render_costa(self, bounds: tuple[float, float, float, float], width: int, height: int) -> Image.Image:
        raster = self.rasters["costa"][0]
        if not raster.get("runtime_available") or "absolute_path" not in raster:
            raise RuntimeError(self.metadata["datasets"]["costa"]["status_message"])
        status = raster["runtime_status"]
        if status.get("backend") == "ecw_window":
            left, bottom, right, top = bounds
            intersection = (
                max(left, raster["left"]),
                max(bottom, raster["bottom"]),
                min(right, raster["right"]),
                min(top, raster["top"]),
            )
            image = Image.new("RGB", (width, height), (248, 248, 248))
            if intersection[0] >= intersection[2] or intersection[1] >= intersection[3]:
                return image
            out_x0 = max(0, round((intersection[0] - left) / (right - left) * width))
            out_x1 = min(width, round((intersection[2] - left) / (right - left) * width))
            out_y0 = max(0, round((top - intersection[3]) / (top - bottom) * height))
            out_y1 = min(height, round((top - intersection[1]) / (top - bottom) * height))
            out_width, out_height = out_x1 - out_x0, out_y1 - out_y0
            if out_width <= 0 or out_height <= 0:
                return image
            source_x0 = max(0, math.floor((intersection[0] - raster["left"]) / raster["resolution_x"]))
            source_x1 = min(raster["width"] - 1, math.ceil((intersection[2] - raster["left"]) / raster["resolution_x"]) - 1)
            source_y0 = max(0, math.floor((raster["top"] - intersection[3]) / raster["resolution_y"]))
            source_y1 = min(raster["height"] - 1, math.ceil((raster["top"] - intersection[1]) / raster["resolution_y"]) - 1)
            if not self.ecw_worker:
                raise RuntimeError("ECW worker is unavailable")
            payload = self.ecw_worker.decode(
                source_x0, source_y0, source_x1, source_y1, out_width, out_height, 0
            )
            decoded = Image.open(io.BytesIO(payload)).convert("RGB")
            image.paste(decoded, (out_x0, out_y0))
            return image
        translate = status["gdal_translate"]
        left, bottom, right, top = bounds
        cmd = [
            translate,
            "-q",
            "-projwin", str(left), str(top), str(right), str(bottom),
            "-outsize", str(width), str(height),
            "-of", "PNG",
            str(raster["absolute_path"]),
            "/vsistdout/",
        ]
        env = dict(os.environ)
        env["ECW_CACHE_MAXMEM"] = str(256 * 1024 * 1024)
        proc = subprocess.run(cmd, capture_output=True, timeout=45, env=env)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.decode(errors="replace")[-1000:])
        return Image.open(io.BytesIO(proc.stdout)).convert("RGB")

    def render(self, dataset: str, bounds: tuple[float, float, float, float], width: int, height: int) -> Image.Image:
        if dataset == "moray":
            return self.render_moray(bounds, width, height)
        if dataset == "costa":
            return self.render_costa(bounds, width, height)
        raise KeyError(dataset)

    def tile(self, dataset_id: str, z: int, x: int, y: int) -> tuple[bytes, str]:
        dataset = self.public_config()["datasets"].get(dataset_id)
        if dataset is None:
            raise KeyError(dataset_id)
        max_zoom = int(dataset["max_zoom"])
        if not (0 <= z <= max_zoom and x >= 0 and y >= 0):
            raise ValueError("tile outside pyramid")
        resolution = dataset["native_resolution_m"] * (2 ** (max_zoom - z))
        span = TILE_SIZE * resolution
        source = dataset["bounds"]
        left = source["left"] + x * span
        top = source["top"] - y * span
        if left >= source["right"] or top <= source["bottom"]:
            raise ValueError("tile outside dataset")
        bounds = (left, top - span, left + span, top)
        key = f"tile:{dataset_id}:{z}:{x}:{y}"
        cached = self.cache.get(key)
        if cached:
            with self.stats_lock:
                self.stats["cache_hits"] += 1
            return cached
        started = time.perf_counter()
        image = self.render(dataset_id, bounds, TILE_SIZE, TILE_SIZE)
        output = io.BytesIO()
        if z == max_zoom:
            image.save(output, format="PNG", compress_level=3)
            content_type = "image/png"
        else:
            image.save(output, format="JPEG", quality=86, optimize=False)
            content_type = "image/jpeg"
        value = (output.getvalue(), content_type)
        self.cache.put(key, value)
        with self.stats_lock:
            self.stats["tile_requests"] += 1
            self.stats["tile_render_ms"].append((time.perf_counter() - started) * 1000)
            self.stats["tile_render_ms"] = self.stats["tile_render_ms"][-500:]
        return value

    def crop(self, dataset_id: str, e: float, n: float, span: float) -> tuple[bytes, dict[str, str]]:
        dataset = self.public_config()["datasets"].get(dataset_id)
        if dataset is None:
            raise KeyError(dataset_id)
        native = float(dataset["native_resolution_m"])
        pixels = max(1, min(768, int(math.ceil(span / native))))
        actual_span = pixels * native
        bounds = (e - actual_span / 2, n - actual_span / 2, e + actual_span / 2, n + actual_span / 2)
        key = f"crop:{dataset_id}:{e:.3f}:{n:.3f}:{actual_span:.3f}"
        cached = self.cache.get(key)
        if cached:
            with self.stats_lock:
                self.stats["cache_hits"] += 1
            body, content_type = cached
        else:
            started = time.perf_counter()
            image = self.render(dataset_id, bounds, pixels, pixels)
            output = io.BytesIO()
            image.save(output, format="PNG", compress_level=3)
            body, content_type = output.getvalue(), "image/png"
            self.cache.put(key, (body, content_type))
            with self.stats_lock:
                self.stats["crop_requests"] += 1
                self.stats["crop_render_ms"].append((time.perf_counter() - started) * 1000)
                self.stats["crop_render_ms"] = self.stats["crop_render_ms"][-500:]
        return body, {
            "Content-Type": content_type,
            "X-Native-Resolution-M": str(native),
            "X-Crop-Span-M": str(actual_span),
            "X-Crop-Pixels": str(pixels),
            "Cache-Control": "private, max-age=3600",
        }

    def filtered_contacts(self, query: dict[str, list[str]]) -> dict:
        dataset = query.get("dataset", ["moray"])[0]
        area = query.get("area", [""])[0]
        include_magnetic = query.get("include_magnetic", ["0"])[0] == "1"
        search = query.get("q", [""])[0].strip().lower()
        categories = {item for item in query.get("category", []) if item}
        categories_active = query.get("categories_active", ["0"])[0] == "1"
        bbox_text = query.get("bbox", [""])[0]
        bbox = None
        if bbox_text:
            values = [float(value) for value in bbox_text.split(",")]
            if len(values) == 4:
                bbox = values
        contacts = []
        for contact in self.by_dataset.get(dataset, []):
            if area and contact["area"] != area:
                continue
            if not include_magnetic and contact["description"].casefold() == "magnetic anomaly":
                continue
            if categories_active and contact["description"] not in categories:
                continue
            # Text search is area-wide so a known ID can locate a contact that is
            # currently outside the viewport. Ordinary browsing remains bbox-lazy.
            if bbox and not search and not (bbox[0] <= contact["e"] <= bbox[2] and bbox[1] <= contact["n"] <= bbox[3]):
                continue
            if search:
                haystack = " ".join((contact["id"], contact["description"], contact["source_file"])).lower()
                if search not in haystack:
                    continue
            contacts.append(contact)
        cluster_m = max(0.0, float(query.get("cluster_m", ["0"])[0]))
        if cluster_m <= 0:
            features = [self.contact_feature(item) for item in contacts[:5000]]
        else:
            groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
            for contact in contacts:
                groups[(math.floor(contact["e"] / cluster_m), math.floor(contact["n"] / cluster_m))].append(contact)
            features = []
            for items in groups.values():
                if len(items) == 1:
                    features.append(self.contact_feature(items[0]))
                else:
                    features.append(
                        {
                            "kind": "cluster",
                            "e": sum(item["e"] for item in items) / len(items),
                            "n": sum(item["n"] for item in items) / len(items),
                            "count": len(items),
                            "descriptions": dict(Counter(item["description"] for item in items)),
                        }
                    )
        return {"features": features, "matched": len(contacts), "returned": len(features)}

    @staticmethod
    def contact_feature(contact: dict) -> dict:
        return {
            "kind": "contact",
            "uid": contact["uid"],
            "id": contact["id"],
            "e": contact["e"],
            "n": contact["n"],
            "description": contact["description"],
            "geometry": contact["geometry"],
        }

    def runtime_stats(self) -> dict:
        with self.stats_lock:
            tile_values = list(self.stats["tile_render_ms"])
            crop_values = list(self.stats["crop_render_ms"])
            result = dict(self.stats)
        result["uptime_seconds"] = round(time.time() - result.pop("started_at"), 1)
        result["tile_render_ms"] = latency_summary(tile_values)
        result["crop_render_ms"] = latency_summary(crop_values)
        result["memory_cache_items"] = len(self.cache.values)
        return result

    def close(self) -> None:
        if self.ecw_worker:
            self.ecw_worker.close()


def latency_summary(values: list[float]) -> dict:
    ordered = sorted(values)
    return {
        "samples": len(values),
        "mean": round(sum(values) / len(values), 2) if values else None,
        "median": round(ordered[len(ordered) // 2], 2) if values else None,
        "max": round(max(values), 2) if values else None,
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "SSSInspector/1.0"

    @property
    def app(self) -> InspectorData:
        return self.server.app  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}", flush=True)

    def send_bytes(self, body: bytes, status=200, headers: dict[str, str] | None = None) -> None:
        self.send_response(status)
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, value, status=200) -> None:
        self.send_bytes(
            json.dumps(value, separators=(",", ":")).encode(),
            status,
            {"Content-Type": "application/json; charset=utf-8", "Cache-Control": "no-store"},
        )

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path == "/api/config":
                self.send_json(self.app.public_config())
                return
            if parsed.path == "/api/contacts":
                self.send_json(self.app.filtered_contacts(query))
                return
            if parsed.path.startswith("/api/contact/"):
                uid = unquote(parsed.path.removeprefix("/api/contact/"))
                contact = self.app.by_uid.get(uid)
                self.send_json(contact if contact else {"error": "not found"}, 200 if contact else 404)
                return
            if parsed.path == "/api/stats":
                self.send_json(self.app.runtime_stats())
                return
            if parsed.path == "/api/tile":
                with self.app.semaphore:
                    body, content_type = self.app.tile(
                        query["dataset"][0], int(query["z"][0]), int(query["x"][0]), int(query["y"][0])
                    )
                self.send_bytes(body, headers={"Content-Type": content_type, "Cache-Control": "private, max-age=3600"})
                return
            if parsed.path == "/api/crop":
                with self.app.semaphore:
                    body, headers = self.app.crop(
                        query["dataset"][0], float(query["e"][0]), float(query["n"][0]), float(query.get("span", ["100"])[0])
                    )
                self.send_bytes(body, headers=headers)
                return
            self.serve_static(parsed.path)
        except (KeyError, ValueError) as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except RuntimeError as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.SERVICE_UNAVAILABLE)
        except BrokenPipeError:
            pass
        except Exception as exc:
            self.send_json({"error": f"{type(exc).__name__}: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def serve_static(self, request_path: str) -> None:
        relative = "index.html" if request_path in ("", "/") else request_path.lstrip("/")
        path = (self.app.static_dir / relative).resolve()
        if self.app.static_dir.resolve() not in path.parents and path != self.app.static_dir.resolve():
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_bytes(body, headers={"Content-Type": content_type, "Cache-Control": "no-cache"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    app = InspectorData(ROOT)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.app = app  # type: ignore[attr-defined]
    server.daemon_threads = True
    print(f"SSS Contact Inspector: http://{args.host}:{args.port}", flush=True)
    print(f"contacts: {len(app.contacts):,}; worker limit: {MAX_WORKERS}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        app.close()


if __name__ == "__main__":
    main()
