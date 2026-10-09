#!/usr/bin/env python3
"""Build a self-contained overview map for the annotated validation patches."""

from __future__ import annotations

import base64
import csv
import io
import json
import math
import subprocess
from pathlib import Path

from PIL import Image, ImageOps


PROJECT = Path("/Users/haonanzhang/IRP-external/sss_contact_inspector")
BATCH_ROOT = PROJECT / "yolo_manual_annotation_grid_150px_stride100_v1_cleaned" / "validation_annotation_batches_1000"
OUT = PROJECT / "validation-patch-label-map.html"
RASTER = PROJECT / "data/derived_sources/costa/C13007a_Costa_AllAreas_Comments_Rev1.ecw"
DECODER = PROJECT / "local/bin/ecw_window"
SOURCE_LEFT = 477323.46875
SOURCE_TOP = 6568269.5
RESOLUTION = 0.1


def read_data() -> dict:
    groups: list[dict] = []
    patches: list[list] = []
    labels: list[list] = []
    easts: list[float] = []
    norths: list[float] = []

    group_dirs = [path for path in sorted(BATCH_ROOT.glob("group_*")) if path.is_dir()]
    for group_index, group_dir in enumerate(group_dirs):
        rows = list(csv.DictReader((group_dir / "patch_index.csv").open(newline="")))
        group = {
            "name": group_dir.name,
            "patches": len(rows),
            "labels": 0,
            "positive": 0,
            "negative": len(rows),
            "empty_json": 0,
            "bounds": None,
        }
        group_bounds = [math.inf, -math.inf, math.inf, -math.inf]

        for row in rows:
            patch_id = row["patch_id"]
            e0 = float(row["footprint_left_easting"])
            e1 = float(row["footprint_right_easting"])
            n0 = float(row["footprint_bottom_northing"])
            n1 = float(row["footprint_top_northing"])
            patch_index = len(patches)
            patch = [group_index, e0, e1, n0, n1, 0, patch_id]
            patches.append(patch)
            easts.extend((e0, e1))
            norths.extend((n0, n1))
            group_bounds[0] = min(group_bounds[0], e0)
            group_bounds[1] = max(group_bounds[1], e1)
            group_bounds[2] = min(group_bounds[2], n0)
            group_bounds[3] = max(group_bounds[3], n1)

            json_path = group_dir / "images" / f"{patch_id}.json"
            if not json_path.exists():
                continue
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            shapes = payload.get("shapes", [])
            if not shapes:
                group["empty_json"] += 1
                continue
            group["positive"] += 1
            group["negative"] -= 1

            for shape_index, shape in enumerate(shapes):
                points = shape.get("points", [])
                if len(points) < 2:
                    continue
                xs = [float(point[0]) for point in points]
                ys = [float(point[1]) for point in points]
                px0, px1 = min(xs), max(xs)
                py0, py1 = min(ys), max(ys)
                label_e0 = e0 + px0 * RESOLUTION
                label_e1 = e0 + px1 * RESOLUTION
                label_n0 = n1 - py1 * RESOLUTION
                label_n1 = n1 - py0 * RESOLUTION
                labels.append([group_index, patch_index, label_e0, label_e1, label_n0, label_n1, shape_index])
                patch[5] += 1
                group["labels"] += 1

        group["bounds"] = group_bounds
        groups.append(group)

    bounds = [min(easts), max(easts), min(norths), max(norths)]
    return {
        "crs": "EPSG:32630",
        "resolution": RESOLUTION,
        "groups": groups,
        "patches": patches,
        "labels": labels,
        "bounds": bounds,
        "totals": {
            "patches": len(patches),
            "labels": len(labels),
            "positive": sum(group["positive"] for group in groups),
            "negative": sum(group["negative"] for group in groups),
            "json_files": sum(group["positive"] + group["empty_json"] for group in groups),
        },
    }


def background_data(bounds: list[float]) -> tuple[str, list[float]]:
    """Return a compact grayscale overview and its projected bounds."""
    if not RASTER.exists() or not DECODER.exists():
        return "", bounds

    pad = 15.0
    left, right = bounds[0] - pad, bounds[1] + pad
    bottom, top = bounds[2] - pad, bounds[3] + pad
    x0 = max(0, round((left - SOURCE_LEFT) / RESOLUTION))
    y0 = max(0, round((SOURCE_TOP - top) / RESOLUTION))
    x1 = round((right - SOURCE_LEFT) / RESOLUTION) - 1
    y1 = round((SOURCE_TOP - bottom) / RESOLUTION) - 1
    width = 900
    height = max(1, round(width * (top - bottom) / (right - left)))
    command = [str(DECODER), "window", str(RASTER), str(x0), str(y0), str(x1), str(y1), str(width), str(height), "0"]
    try:
        result = subprocess.run(command, check=True, capture_output=True, timeout=120)
        image = Image.open(io.BytesIO(result.stdout)).convert("L")
        image = ImageOps.autocontrast(image, cutoff=1).convert("RGB")
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=68, optimize=True)
        encoded = base64.b64encode(output.getvalue()).decode("ascii")
        return encoded, [left, right, bottom, top]
    except (OSError, subprocess.SubprocessError, ValueError):
        return "", bounds


HTML = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SSS validation patch and label map</title>
  <style>
    :root {
      color-scheme: light dark;
      --bg: light-dark(#f6f7f8, #101315);
      --surface: light-dark(#ffffff, #171b1e);
      --surface-2: light-dark(#eef1f3, #20272b);
      --text: light-dark(#182126, #edf2f4);
      --muted: light-dark(#5d6b73, #a9b7bd);
      --border: light-dark(#d4dce0, #344046);
      --accent: light-dark(#0969da, #6cb6ff);
      --label: #ff5d73;
      --positive: #ffb000;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--text); font: 14px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    #app { max-width: 1500px; margin: 0 auto; padding: 20px; }
    header { display: flex; align-items: end; justify-content: space-between; gap: 20px; flex-wrap: wrap; }
    h1 { margin: 0; font-size: 22px; font-weight: 500; letter-spacing: -.01em; }
    .subtitle { color: var(--muted); margin: 4px 0 0; }
    .stats { display: flex; gap: 8px; flex-wrap: wrap; }
    .stat { min-width: 100px; padding: 8px 11px; background: var(--surface); border: 1px solid var(--border); border-radius: 8px; }
    .stat span { display: block; color: var(--muted); font-size: 12px; }
    .stat strong { display: block; font-size: 18px; font-weight: 500; font-variant-numeric: tabular-nums; }
    .layout { display: grid; grid-template-columns: minmax(0, 1fr) 280px; gap: 14px; margin-top: 16px; }
    .map-panel, aside { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; }
    .map-panel { min-width: 0; overflow: hidden; }
    .toolbar { display: flex; align-items: center; gap: 14px; padding: 10px 12px; border-bottom: 1px solid var(--border); flex-wrap: wrap; }
    label { color: var(--muted); }
    select, button { color: var(--text); background: var(--surface-2); border: 1px solid var(--border); border-radius: 6px; padding: 5px 8px; font: inherit; }
    button { cursor: pointer; }
    button:hover, select:focus { border-color: var(--accent); }
    #map-wrap { position: relative; height: min(78vh, 850px); min-height: 560px; background: #d6d9d5; }
    canvas { display: block; width: 100%; height: 100%; touch-action: none; }
    .hint { position: absolute; left: 12px; bottom: 10px; padding: 4px 7px; color: var(--muted); background: color-mix(in srgb, var(--surface) 86%, transparent); border-radius: 4px; font-size: 12px; pointer-events: none; }
    aside { padding: 14px; align-self: start; }
    aside h2 { margin: 0 0 10px; font-size: 16px; font-weight: 500; }
    #details { color: var(--muted); }
    .detail-title { color: var(--text); font-weight: 500; margin-bottom: 10px; overflow-wrap: anywhere; }
    .detail-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 9px 12px; }
    .detail-grid span { display: block; color: var(--muted); font-size: 12px; }
    .detail-grid b { display: block; color: var(--text); font-weight: 500; font-variant-numeric: tabular-nums; }
    .legend { display: flex; gap: 14px; flex-wrap: wrap; padding: 8px 12px; color: var(--muted); font-size: 12px; border-top: 1px solid var(--border); }
    .legend i { display: inline-block; width: 11px; height: 11px; margin-right: 4px; vertical-align: -1px; border-radius: 2px; }
    @media (max-width: 900px) {
      #app { padding: 12px; }
      .layout { grid-template-columns: 1fr; }
      #map-wrap { min-height: 500px; height: 65vh; }
      aside { order: 2; }
    }
    @media (max-width: 560px) {
      .stat { min-width: 84px; }
      .stat strong { font-size: 16px; }
      .toolbar { gap: 8px; }
    }
  </style>
</head>
<body>
<div id="app">
  <header>
    <div>
      <h1>SSS validation patches and contact labels</h1>
      <p class="subtitle">Costa Head validation area · projected coordinates in metres · EPSG:32630</p>
    </div>
    <div class="stats" aria-label="Dataset totals">
      <div class="stat"><span>Patches</span><strong id="stat-patches">–</strong></div>
      <div class="stat"><span>Labels</span><strong id="stat-labels">–</strong></div>
      <div class="stat"><span>Positive</span><strong id="stat-positive">–</strong></div>
      <div class="stat"><span>Negative</span><strong id="stat-negative">–</strong></div>
    </div>
  </header>
  <div class="layout">
    <section class="map-panel" aria-label="Patch and label map">
      <div class="toolbar">
        <label>Group <select id="group-select" aria-label="Filter by annotation group"></select></label>
        <label><input id="patch-toggle" type="checkbox" checked> patch footprints</label>
        <label><input id="label-toggle" type="checkbox" checked> label boxes</label>
        <label><input id="background-toggle" type="checkbox" checked> SSS background</label>
        <button id="fit-button" type="button">Fit view</button>
      </div>
      <div id="map-wrap">
        <canvas id="map" aria-label="Interactive map of validation patch footprints and contact label boxes"></canvas>
        <div class="hint">Drag to pan · scroll to zoom · click a patch or label for details</div>
      </div>
      <div class="legend">
        <span><i style="background:var(--label)"></i>contact label</span>
        <span><i style="background:var(--positive)"></i>patch containing a label</span>
        <span><i style="background:#77838a"></i>all patch footprints</span>
      </div>
    </section>
    <aside>
      <h2>Selected item</h2>
      <div id="details" aria-live="polite">Click a patch or label on the map.</div>
    </aside>
  </div>
</div>
<script>
const DATA = __DATA__;
const BACKGROUND = __BACKGROUND__;
const BG_BOUNDS = __BG_BOUNDS__;
const COLORS = ['#277da1', '#6a4c93', '#43aa8b', '#f8961e', '#f94144', '#577590'];
const mapWrap = document.getElementById('map-wrap');
const canvas = document.getElementById('map');
const ctx = canvas.getContext('2d');
const details = document.getElementById('details');
const state = { group: -1, centerE: 0, centerN: 0, mpp: 1, dpr: 1, cssW: 0, cssH: 0, drag: null, selected: null };

const fmt = (value, digits = 1) => Number(value).toLocaleString(undefined, {maximumFractionDigits: digits});
const fmtInt = value => Number(value).toLocaleString();
const groupName = index => DATA.groups[index]?.name || 'all groups';
const active = item => state.group < 0 || item[0] === state.group;

function resize() {
  const rect = mapWrap.getBoundingClientRect();
  state.dpr = Math.min(window.devicePixelRatio || 1, 2);
  state.cssW = Math.max(1, rect.width);
  state.cssH = Math.max(1, rect.height);
  canvas.width = Math.round(state.cssW * state.dpr);
  canvas.height = Math.round(state.cssH * state.dpr);
  draw();
}

function fitView() {
  let b = DATA.bounds;
  if (state.group >= 0) b = DATA.groups[state.group].bounds;
  const pad = 36;
  const width = b[1] - b[0], height = b[3] - b[2];
  state.centerE = (b[0] + b[1]) / 2;
  state.centerN = (b[2] + b[3]) / 2;
  state.mpp = Math.max(width / Math.max(1, state.cssW - pad * 2), height / Math.max(1, state.cssH - pad * 2));
  state.selected = null;
  draw();
  showEmptyDetails();
}

function worldToScreen(e, n) {
  return [state.cssW / 2 + (e - state.centerE) / state.mpp, state.cssH / 2 + (state.centerN - n) / state.mpp];
}
function screenToWorld(x, y) {
  return [state.centerE + (x - state.cssW / 2) * state.mpp, state.centerN - (y - state.cssH / 2) * state.mpp];
}
function niceStep(raw) {
  const exponent = Math.floor(Math.log10(raw));
  const fraction = raw / Math.pow(10, exponent);
  const nice = fraction >= 5 ? 5 : fraction >= 2 ? 2 : 1;
  return nice * Math.pow(10, exponent);
}
function drawGrid() {
  const left = state.centerE - state.cssW * state.mpp / 2;
  const right = state.centerE + state.cssW * state.mpp / 2;
  const bottom = state.centerN - state.cssH * state.mpp / 2;
  const top = state.centerN + state.cssH * state.mpp / 2;
  const step = niceStep(Math.max(right - left, top - bottom) / 6);
  ctx.strokeStyle = 'rgba(80, 92, 98, .26)';
  ctx.fillStyle = 'rgba(32, 42, 46, .78)';
  ctx.lineWidth = 1;
  ctx.font = '11px -apple-system, sans-serif';
  for (let e = Math.ceil(left / step) * step; e <= right; e += step) {
    const [x] = worldToScreen(e, state.centerN);
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, state.cssH); ctx.stroke();
    ctx.fillText(`${fmt(e, 0)} E`, x + 3, state.cssH - 7);
  }
  for (let n = Math.ceil(bottom / step) * step; n <= top; n += step) {
    const [, y] = worldToScreen(state.centerE, n);
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(state.cssW, y); ctx.stroke();
    ctx.fillText(`${fmt(n, 0)} N`, 5, y - 4);
  }
}

function drawBackground() {
  if (!BACKGROUND || !document.getElementById('background-toggle').checked) return;
  if (!state._backgroundImage) {
    state._backgroundImage = new Image();
    state._backgroundImage.src = `data:image/jpeg;base64,${BACKGROUND}`;
    state._backgroundImage.onload = () => draw();
  }
  const image = state._backgroundImage;
  if (!image.complete) return;
  const b = BG_BOUNDS;
  const [x0, y0] = worldToScreen(b[0], b[3]);
  const [x1, y1] = worldToScreen(b[1], b[2]);
  ctx.globalAlpha = .72;
  ctx.drawImage(image, x0, y0, x1 - x0, y1 - y0);
  ctx.globalAlpha = 1;
}

function draw() {
  if (!state.cssW || !state.cssH) return;
  ctx.setTransform(state.dpr, 0, 0, state.dpr, 0, 0);
  ctx.clearRect(0, 0, state.cssW, state.cssH);
  ctx.fillStyle = '#d6d9d5'; ctx.fillRect(0, 0, state.cssW, state.cssH);
  drawBackground();
  drawGrid();

  if (document.getElementById('patch-toggle').checked) {
    for (let i = 0; i < DATA.patches.length; i += 1) {
      const p = DATA.patches[i];
      if (!active(p)) continue;
      const [x0, y0] = worldToScreen(p[1], p[4]);
      const [x1, y1] = worldToScreen(p[2], p[3]);
      const color = COLORS[p[0] % COLORS.length];
      ctx.fillStyle = p[5] ? `${color}35` : `${color}12`;
      ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
      if (state.mpp < 8 || p[5]) {
        ctx.strokeStyle = p[5] ? `${color}9a` : `${color}42`;
        ctx.lineWidth = p[5] ? 1.2 : .6;
        ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
      }
    }
  }

  if (document.getElementById('label-toggle').checked) {
    for (let i = 0; i < DATA.labels.length; i += 1) {
      const l = DATA.labels[i];
      if (!active(l)) continue;
      const [x0, y0] = worldToScreen(l[2], l[5]);
      const [x1, y1] = worldToScreen(l[3], l[4]);
      const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
      const width = Math.max(2, x1 - x0), height = Math.max(2, y1 - y0);
      ctx.fillStyle = 'rgba(255, 93, 115, .80)';
      ctx.strokeStyle = '#701b2c';
      ctx.lineWidth = state.selected?.type === 'label' && state.selected.index === i ? 2.5 : 1;
      if (state.mpp < 2.5) { ctx.beginPath(); ctx.arc(cx, cy, 3.5, 0, Math.PI * 2); ctx.fill(); ctx.stroke(); }
      else { ctx.fillRect(x0, y0, width, height); ctx.strokeRect(x0, y0, width, height); }
    }
  }

  if (state.selected) {
    const item = state.selected.type === 'label' ? DATA.labels[state.selected.index] : DATA.patches[state.selected.index];
    if (item && active(item)) {
      const b = state.selected.type === 'label' ? item.slice(2, 6) : item.slice(1, 5);
      const [x0, y0] = worldToScreen(b[0], b[3]);
      const [x1, y1] = worldToScreen(b[1], b[2]);
      ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 2.5; ctx.strokeRect(x0 - 2, y0 - 2, x1 - x0 + 4, y1 - y0 + 4);
    }
  }
}

function utmToLatLon(easting, northing) {
  const a = 6378137, ecc = 0.00669438, k0 = 0.9996, zone = 30;
  const ePrime = ecc / (1 - ecc), x = easting - 500000, y = northing;
  const m = y / k0, mu = m / (a * (1 - ecc / 4 - 3 * ecc * ecc / 64 - 5 * ecc * ecc * ecc / 256));
  const e1 = (1 - Math.sqrt(1 - ecc)) / (1 + Math.sqrt(1 - ecc));
  const phi1 = mu + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * Math.sin(2 * mu) + (21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32) * Math.sin(4 * mu) + (151 * e1 ** 3 / 96) * Math.sin(6 * mu) + (1097 * e1 ** 4 / 512) * Math.sin(8 * mu);
  const n1 = a / Math.sqrt(1 - ecc * Math.sin(phi1) ** 2), t1 = Math.tan(phi1) ** 2, c1 = ePrime * Math.cos(phi1) ** 2;
  const r1 = a * (1 - ecc) / Math.pow(1 - ecc * Math.sin(phi1) ** 2, 1.5), d = x / (n1 * k0);
  const lat = phi1 - (n1 * Math.tan(phi1) / r1) * (d * d / 2 - (5 + 3 * t1 + 10 * c1 - 4 * c1 * c1 - 9 * ePrime) * d ** 4 / 24 + (61 + 90 * t1 + 298 * c1 + 45 * t1 * t1 - 252 * ePrime - 3 * c1 * c1) * d ** 6 / 720);
  const lon = (zone * 6 - 183) * Math.PI / 180 + (d - (1 + 2 * t1 + c1) * d ** 3 / 6 + (5 - 2 * c1 + 28 * t1 - 3 * c1 * c1 + 8 * ePrime + 24 * t1 * t1) * d ** 5 / 120) / Math.cos(phi1);
  return [lat * 180 / Math.PI, lon * 180 / Math.PI];
}

function showEmptyDetails() { details.innerHTML = 'Click a patch or label on the map.'; }
function showDetails(type, index) {
  const item = type === 'label' ? DATA.labels[index] : DATA.patches[index];
  if (!item) return;
  const b = type === 'label' ? item.slice(2, 6) : item.slice(1, 5);
  const e = (b[0] + b[1]) / 2, n = (b[2] + b[3]) / 2;
  const [lat, lon] = utmToLatLon(e, n);
  const patch = type === 'label' ? DATA.patches[item[1]] : item;
  const title = type === 'label' ? `Contact label · ${groupName(item[0])}` : `Patch · ${groupName(item[0])}`;
  const name = type === 'label' ? patch[6] : item[6];
  details.innerHTML = `<div class="detail-title">${title}</div><div class="detail-grid"><div><span>Patch ID</span><b>${name}</b></div><div><span>Labels in patch</span><b>${patch[5]}</b></div><div><span>Easting</span><b>${fmt(e)} m</b></div><div><span>Northing</span><b>${fmt(n)} m</b></div><div><span>Latitude</span><b>${fmt(lat, 6)}°</b></div><div><span>Longitude</span><b>${fmt(lon, 6)}°</b></div><div><span>Width</span><b>${fmt(b[1] - b[0], 2)} m</b></div><div><span>Height</span><b>${fmt(b[3] - b[2], 2)} m</b></div></div>`;
}

function hitTest(x, y) {
  let best = null, bestDistance = Infinity;
  if (document.getElementById('label-toggle').checked) {
    for (let i = 0; i < DATA.labels.length; i += 1) {
      const l = DATA.labels[i]; if (!active(l)) continue;
      const [x0, y0] = worldToScreen(l[2], l[5]), [x1, y1] = worldToScreen(l[3], l[4]);
      const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
      const dx = Math.max(x0 - x, 0, x - x1), dy = Math.max(y0 - y, 0, y - y1), distance = Math.hypot(dx, dy);
      if (distance < bestDistance && distance <= 10) { best = {type: 'label', index: i}; bestDistance = distance; }
    }
  }
  if (!best && document.getElementById('patch-toggle').checked) {
    for (let i = 0; i < DATA.patches.length; i += 1) {
      const p = DATA.patches[i]; if (!active(p)) continue;
      const [x0, y0] = worldToScreen(p[1], p[4]), [x1, y1] = worldToScreen(p[2], p[3]);
      if (x >= x0 - 3 && x <= x1 + 3 && y >= y0 - 3 && y <= y1 + 3) {
        const distance = Math.abs(x - (x0 + x1) / 2) + Math.abs(y - (y0 + y1) / 2);
        if (distance < bestDistance) { best = {type: 'patch', index: i}; bestDistance = distance; }
      }
    }
  }
  return best;
}

function pointerPosition(event) { const rect = canvas.getBoundingClientRect(); return [event.clientX - rect.left, event.clientY - rect.top]; }
canvas.addEventListener('pointerdown', event => { const [x, y] = pointerPosition(event); canvas.setPointerCapture(event.pointerId); state.drag = {x, y, centerE: state.centerE, centerN: state.centerN, moved: false}; });
canvas.addEventListener('pointermove', event => { if (!state.drag) return; const [x, y] = pointerPosition(event); const dx = x - state.drag.x, dy = y - state.drag.y; if (Math.hypot(dx, dy) > 3) state.drag.moved = true; state.centerE = state.drag.centerE - dx * state.mpp; state.centerN = state.drag.centerN + dy * state.mpp; draw(); });
canvas.addEventListener('pointerup', event => { const [x, y] = pointerPosition(event); const drag = state.drag; state.drag = null; if (!drag?.moved) { state.selected = hitTest(x, y); if (state.selected) showDetails(state.selected.type, state.selected.index); else showEmptyDetails(); draw(); } });
canvas.addEventListener('wheel', event => { event.preventDefault(); const [x, y] = pointerPosition(event); const before = screenToWorld(x, y); state.mpp *= event.deltaY > 0 ? 1.25 : 0.8; state.mpp = Math.max(.05, Math.min(state.mpp, 10000)); const after = screenToWorld(x, y); state.centerE += before[0] - after[0]; state.centerN += before[1] - after[1]; draw(); }, {passive:false});

function makeControls() {
  const select = document.getElementById('group-select');
  select.innerHTML = `<option value="-1">All groups</option>` + DATA.groups.map((g, i) => `<option value="${i}">${g.name} · ${fmtInt(g.patches)} patches · ${fmtInt(g.labels)} labels</option>`).join('');
  select.addEventListener('change', () => { state.group = Number(select.value); fitView(); });
  ['patch-toggle', 'label-toggle', 'background-toggle'].forEach(id => document.getElementById(id).addEventListener('change', draw));
  document.getElementById('fit-button').addEventListener('click', fitView);
}

document.getElementById('stat-patches').textContent = fmtInt(DATA.totals.patches);
document.getElementById('stat-labels').textContent = fmtInt(DATA.totals.labels);
document.getElementById('stat-positive').textContent = fmtInt(DATA.totals.positive);
document.getElementById('stat-negative').textContent = fmtInt(DATA.totals.negative);
makeControls();
new ResizeObserver(resize).observe(mapWrap);
resize();
fitView();
</script>
</body>
</html>
'''


def main() -> None:
    data = read_data()
    background, bg_bounds = background_data(data["bounds"])
    output = HTML.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    output = output.replace("__BACKGROUND__", json.dumps(background))
    output = output.replace("__BG_BOUNDS__", json.dumps(bg_bounds, separators=(",", ":")))
    OUT.write_text(output, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes)")
    print(json.dumps(data["totals"], indent=2))


if __name__ == "__main__":
    main()
