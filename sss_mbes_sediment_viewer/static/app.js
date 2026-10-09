(() => {
'use strict';
const $ = s => document.querySelector(s);
const C = $('#base'), O = $('#overlay'), bc = C.getContext('2d'), oc = O.getContext('2d');
const map = $('#map'), info = $('#info'), empty = $('#empty');
const COL = {SAND: '#f4d35e', GRAVEL: '#ee7b30', 'Gravelly SAND': '#b77bd1', BEDROCK: '#d73b4b'};
const S = {
  vectors: null, audit: null, sss: null, sssDown: false,
  base: 'sss', layers: new Set(['sediments', 'sss_contacts']), opacity: .3, split: false, splitX: .5,
  scale: 1, ox: 0, oy: 0,                 // px per metre and the screen position of E=0 / N=0
  tiles: new Map(), pending: new Set(),   // decoded tile images keyed by layer/z/x/y
  selected: null, frame: 0,
};

// EPSG:32630 is north-up; canvas Y grows down, so Northing is inverted exactly
// once here.  Every raster tile, vector vertex and cursor readout goes through
// these two functions, which is what keeps the layers aligned.
const world = (x, y) => [(x - S.ox) / S.scale, (S.oy - y) / S.scale];
const screen = (e, n) => [e * S.scale + S.ox, S.oy - n * S.scale];

// ------------------------------------------------------------------ tile layers
// A tile layer is a north-up pyramid: z=maxZoom is native resolution and tile
// (x, y) at level z covers [left + x*span, top - y*span] with span = tile*res.
function layerFor(kind) {
  if (kind === 'sss') return S.sss;
  const m = S.audit && S.audit.mbes;
  if (!m) return null;
  const name = kind === 'bathy' ? 'elevation' : 'roughness';
  if (!S[name]) {
    S[name] = {name, bounds: m.bounds, native: m.resolution_m, maxZoom: m.max_zoom, tile: m.tile_size,
      present: presentSets(kind === 'bathy' ? m.present : m.present_roughness),
      url: (z, x, y) => `data/mbes/${name}/${z}/${x}_${y}.png`};
  }
  return S[name];
}
function presentSets(present) {
  const out = {};
  for (const [z, keys] of Object.entries(present || {})) out[z] = new Set(keys);
  return out;
}
function tileLevel(layer) {
  const mpp = 1 / S.scale;
  return Math.max(0, Math.min(layer.maxZoom, Math.round(layer.maxZoom - Math.log2(mpp / layer.native))));
}
function requestTile(key, url, layer) {
  if (S.pending.has(key)) return;
  S.pending.add(key);
  const img = new Image();
  img.onload = () => {
    S.pending.delete(key);
    S.tiles.set(key, img);
    while (S.tiles.size > 600) S.tiles.delete(S.tiles.keys().next().value);
    schedule();
  };
  img.onerror = () => {
    S.pending.delete(key);
    S.tiles.set(key, 'missing');
    if (layer.name === 'sss') probeSSS();
  };
  img.src = url;
}
// The ECW service on port 8090 may start a few seconds after serve.py, or be
// restarted later; the page keeps probing it and retries failed SSS tiles
// (each at most three times) instead of staying blank until a reload.
let probeTimer = null;
const retries = new Map();
async function loadSSSConfig() {
  try {
    const cfg = await fetch('sss/api/config', {cache: 'no-store'}).then(r => r.ok ? r.json() : null);
    const ds = cfg && cfg.datasets && cfg.datasets.costa;
    if (!ds || ds.status !== 'ready') return false;
    S.sss = S.sss || {name: 'sss', bounds: ds.bounds, native: ds.native_resolution_m, maxZoom: ds.max_zoom, tile: (cfg.limits || {}).tile_size || 256,
      url: (z, x, y) => `sss/api/tile?dataset=costa&z=${z}&x=${x}&y=${y}`};
    return true;
  } catch (e) { return false; }
}
function probeSSS() {
  if (probeTimer) return;
  probeTimer = setTimeout(async () => {
    probeTimer = null;
    const ok = await loadSSSConfig();
    if (ok) {
      for (const [k, v] of S.tiles) {
        if (k.startsWith('sss/') && v === 'missing' && (retries.get(k) || 0) < 3) { retries.set(k, (retries.get(k) || 0) + 1); S.tiles.delete(k); }
      }
    }
    if (S.sssDown !== !ok) { S.sssDown = !ok; }
    schedule();
    if (!ok) probeSSS();
  }, S.sssDown ? 3000 : 500);
}
function drawTiles(layer, clip) {
  const b = layer.bounds, z = tileLevel(layer);
  const res = layer.native * 2 ** (layer.maxZoom - z), span = layer.tile * res;
  const [e0, n1] = world(clip.x, clip.y), [e1, n0] = world(clip.x + clip.w, clip.y + clip.h);
  const x0 = Math.max(0, Math.floor((e0 - b.left) / span)), x1 = Math.floor((Math.min(e1, b.right) - b.left) / span);
  const y0 = Math.max(0, Math.floor((b.top - n1) / span)), y1 = Math.floor((b.top - Math.max(n0, b.bottom)) / span);
  const present = layer.present && layer.present[z];
  bc.save(); bc.beginPath(); bc.rect(clip.x, clip.y, clip.w, clip.h); bc.clip();
  for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
    if (layer.present && !(present && present.has(`${x}_${y}`))) continue;
    const key = `${layer.name}/${z}/${x}/${y}`, img = S.tiles.get(key);
    if (!img) { requestTile(key, layer.url(z, x, y), layer); continue; }
    if (img === 'missing') continue;
    // Neighbouring tiles share rounded edges, so there are no seams and no
    // cumulative stretch: each edge is within half a pixel of its true place.
    const [ax, ay] = screen(b.left + x * span, b.top - y * span);
    const [zx, zy] = screen(b.left + (x + 1) * span, b.top - (y + 1) * span);
    const sx = Math.round(ax), sy = Math.round(ay);
    bc.drawImage(img, sx, sy, Math.round(zx) - sx, Math.round(zy) - sy);
  }
  bc.restore();
}

// ------------------------------------------------------------------ vectors
function bbox(g) {
  const b = [Infinity, Infinity, -Infinity, -Infinity];
  const walk = c => {
    if (typeof c[0] === 'number') { b[0] = Math.min(b[0], c[0]); b[1] = Math.min(b[1], c[1]); b[2] = Math.max(b[2], c[0]); b[3] = Math.max(b[3], c[1]); }
    else for (const k of c) walk(k);
  };
  walk(g.coordinates);
  return b;
}
function visible(f) {
  const [e0, n1] = world(0, 0), [e1, n0] = world(map.clientWidth, map.clientHeight), b = f.bbox;
  return b[2] >= e0 && b[0] <= e1 && b[3] >= n0 && b[1] <= n1;
}
function polygonPath(ctx, g) {
  const rings = g.type === 'Polygon' ? g.coordinates : g.coordinates.flat();
  ctx.beginPath();
  for (const r of rings) {
    r.forEach(([x, y], i) => { const p = screen(x, y); i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]); });
    ctx.closePath();
  }
}
function linePath(ctx, g) {
  const lines = g.type === 'LineString' ? [g.coordinates] : g.coordinates;
  ctx.beginPath();
  for (const l of lines) l.forEach(([x, y], i) => { const p = screen(x, y); i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]); });
}
function label(f) { return f.properties.Layer_normalized || f.properties.Layer || 'other'; }
function alpha(hex, a) { return hex + Math.round(a * 255).toString(16).padStart(2, '0'); }
function drawVectors() {
  const V = S.vectors;
  if (S.layers.has('sediments')) for (const f of V.sediments.features) {
    if (!visible(f)) continue;
    const c = COL[label(f)] || '#999';
    polygonPath(oc, f.geometry);
    oc.fillStyle = alpha(c, S.opacity); oc.fill('evenodd');
    oc.strokeStyle = c; oc.lineWidth = 1.2; oc.stroke();
  }
  if (S.layers.has('sediment_lines')) for (const f of V.sediment_lines.features) {
    if (!visible(f)) continue;
    linePath(oc, f.geometry);
    oc.strokeStyle = '#ff4fd8'; oc.lineWidth = 1.5; oc.stroke();
  }
  for (const k of ['sss_contacts', 'sand_waves']) if (S.layers.has(k)) for (const f of V[k].features) {
    if (!visible(f)) continue;
    const [x, y] = screen(...f.geometry.coordinates);
    oc.beginPath(); oc.arc(x, y, 3, 0, 7);
    oc.fillStyle = k === 'sand_waves' ? '#16c6e8' : '#111'; oc.fill();
    oc.strokeStyle = '#fff'; oc.lineWidth = 1; oc.stroke();
  }
  if (S.selected) {
    const g = S.selected.f.geometry;
    oc.strokeStyle = '#00e5ff'; oc.lineWidth = 2.5;
    if (g.type.endsWith('Polygon')) { polygonPath(oc, g); oc.stroke(); }
    else if (g.type.endsWith('LineString')) { linePath(oc, g); oc.stroke(); }
    else { const [x, y] = screen(...g.coordinates); oc.beginPath(); oc.arc(x, y, 8, 0, 7); oc.stroke(); }
  }
}

// ------------------------------------------------------------------ hit testing
function pointInRing(x, y, ring) {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, yi] = ring[i], [xj, yj] = ring[j];
    if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}
function pointInPolygon(x, y, g) {
  const polys = g.type === 'Polygon' ? [g.coordinates] : g.coordinates;
  return polys.some(rings => pointInRing(x, y, rings[0]) && !rings.slice(1).some(r => pointInRing(x, y, r)));
}
function segmentDistance(x, y, [ax, ay], [bx, by]) {
  const dx = bx - ax, dy = by - ay, l2 = dx * dx + dy * dy;
  const t = l2 ? Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / l2)) : 0;
  return Math.hypot(x - (ax + t * dx), y - (ay + t * dy));
}
function lineDistance(x, y, g) {
  const lines = g.type === 'LineString' ? [g.coordinates] : g.coordinates;
  let best = Infinity;
  for (const l of lines) for (let i = 1; i < l.length; i++) best = Math.min(best, segmentDistance(x, y, l[i - 1], l[i]));
  return best;
}
function hit(x, y) {
  const V = S.vectors, tol = 8 / S.scale;
  let best = null, bestD = tol;
  for (const k of ['sss_contacts', 'sand_waves']) if (S.layers.has(k)) for (const f of V[k].features) {
    const d = Math.hypot(f.geometry.coordinates[0] - x, f.geometry.coordinates[1] - y);
    if (d < bestD) { best = {k, f}; bestD = d; }
  }
  if (best) return best;
  if (S.layers.has('sediment_lines')) for (const f of V.sediment_lines.features) {
    if (!inBox(f.bbox, x, y, tol)) continue;
    const d = lineDistance(x, y, f.geometry);
    if (d < bestD) { best = {k: 'sediment_lines', f}; bestD = d; }
  }
  if (best) return best;
  // nested polygons: the smallest official polygon containing the point wins
  let area = Infinity;
  if (S.layers.has('sediments')) for (const f of V.sediments.features) {
    if (!inBox(f.bbox, x, y, 0) || !pointInPolygon(x, y, f.geometry)) continue;
    const a = +f.properties.Area_m2 || (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]);
    if (a < area) { best = {k: 'sediments', f}; area = a; }
  }
  return best;
}
const inBox = (b, x, y, t) => x >= b[0] - t && x <= b[2] + t && y >= b[1] - t && y <= b[3] + t;
function select(item, x, y) {
  S.selected = item;
  if (!item) {
    info.textContent = `E ${x.toFixed(2)}  N ${y.toFixed(2)}\nNo official feature here.`;
    return;
  }
  const p = item.f.properties, lines = [`${item.k} #${item.f.id}`];
  if (item.k === 'sediments') lines.push(`label: ${label(item.f)}`);
  lines.push(JSON.stringify(p, null, 2));
  info.textContent = lines.join('\n');
}

// ------------------------------------------------------------------ view
function fitBounds(b, pad = .95) {
  const w = map.clientWidth, h = map.clientHeight;
  S.scale = Math.min(w / (b.right - b.left), h / (b.top - b.bottom)) * pad;
  const cx = (b.left + b.right) / 2, cy = (b.bottom + b.top) / 2;
  S.ox = w / 2 - cx * S.scale;
  S.oy = h / 2 + cy * S.scale;
  draw();
}
function centreOn(e, n) {
  S.ox = map.clientWidth / 2 - e * S.scale;
  S.oy = map.clientHeight / 2 + n * S.scale;
  draw();
}
function zoomAt(px, py, factor) {
  const [e, n] = world(px, py);
  S.scale = Math.max(map.clientWidth / 60000, Math.min(80, S.scale * factor));
  S.ox = px - e * S.scale;
  S.oy = py + n * S.scale;
  draw();
}
function sizeCanvases() {
  // Re-checked on every draw: devicePixelRatio changes when the window moves
  // between displays, and a stale backing store would leave old pixels behind.
  const d = devicePixelRatio || 1, w = Math.round(map.clientWidth * d), h = Math.round(map.clientHeight * d);
  for (const c of [C, O]) {
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    c.style.width = map.clientWidth + 'px'; c.style.height = map.clientHeight + 'px';
  }
  return d;
}
function schedule() { if (!S.frame) S.frame = requestAnimationFrame(draw); }
function draw() {
  S.frame = 0;
  const d = sizeCanvases(), w = map.clientWidth, h = map.clientHeight;
  for (const ctx of [bc, oc]) { ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height); ctx.setTransform(d, 0, 0, d, 0, 0); }
  if (!S.audit) return;
  const right = S.base === 'sss' ? 'bathy' : S.base, sx = Math.round(w * S.splitX);
  const panes = S.split ? [['sss', {x: 0, y: 0, w: sx, h}], [right, {x: sx, y: 0, w: w - sx, h}]] : [[S.base, {x: 0, y: 0, w, h}]];
  const notes = [];
  for (const [kind, clip] of panes) {
    if (kind === 'none') continue;
    const layer = layerFor(kind);
    if (!layer || (kind === 'sss' && S.sssDown)) {
      notes.push(kind === 'sss' ? 'SSS tiles unavailable: waiting for the native ECW service on port 8090 (serve.py starts it; retrying automatically)…'
        : 'No MBES tiles: run build.py first');
      continue;
    }
    drawTiles(layer, clip);
  }
  if (S.split) {
    oc.fillStyle = '#fff'; oc.fillRect(sx - 1, 0, 2, h);
    oc.beginPath(); oc.arc(sx, h / 2, 9, 0, 7); oc.fill();  // drag handle
    oc.font = '600 12px system-ui'; oc.fillStyle = '#fff'; oc.strokeStyle = 'rgba(0,0,0,.7)'; oc.lineWidth = 3;
    for (const [t, x] of [['SSS', 8], [{bathy: 'MBES elevation', rough: 'roughness', none: '—'}[right], sx + 8]]) {
      oc.strokeText(t, x, 18); oc.fillText(t, x, 18);
    }
  }
  empty.hidden = !notes.length; empty.textContent = notes.join(' · ');
  drawVectors();
  drawScaleBar(w, h);
  const mpp = 1 / S.scale, native = S.sss ? S.sss.native : 0.1;
  $('#scale').textContent = `${mpp.toFixed(3)} m/px` + (mpp < native ? ` (SSS native ${native} m, magnified ${(native / mpp).toFixed(1)}x)` : '');
}
function drawScaleBar(w, h) {
  // round metre length that fits in ~120 px: 1, 2 or 5 x 10^n
  const target = 120 / S.scale, power = 10 ** Math.floor(Math.log10(target));
  let metres = power;
  for (const f of [1, 2, 5]) if (f * power <= target) metres = f * power;
  const px = metres * S.scale, x = 12, y = h - 14;
  oc.fillStyle = 'rgba(0,0,0,.55)'; oc.fillRect(x - 6, y - 16, px + 12, 24);
  oc.fillStyle = '#fff'; oc.fillRect(x, y, px, 3);
  oc.font = '600 11px system-ui'; oc.textBaseline = 'bottom';
  oc.fillText(metres >= 1000 ? `${metres / 1000} km` : `${metres} m`, x, y - 2);
}

// ------------------------------------------------------------------ search
function search(q) {
  q = q.trim().toLowerCase();
  const status = $('#search-status');
  if (!q) { status.textContent = ''; return; }
  const matches = [];
  for (const k of ['sss_contacts', 'sand_waves', 'sediments', 'sediment_lines']) for (const f of S.vectors[k].features) {
    if (Object.values(f.properties).some(v => String(v).toLowerCase().includes(q))) matches.push({k, f});
  }
  if (!matches.length) { status.textContent = 'no match'; return; }
  const m = matches[0], b = m.f.bbox;
  status.textContent = `${matches.length} match${matches.length > 1 ? 'es' : ''}; showing ${m.k} #${m.f.id}`;
  if (m.f.geometry.type === 'Point') { if (S.scale < 2) S.scale = 4; centreOn(...m.f.geometry.coordinates); }
  else fitBounds({left: b[0], bottom: b[1], right: b[2], top: b[3]}, .8);
  select(m, b[0], b[1]);
  draw();
}

// ------------------------------------------------------------------ self-check
// Compares what this page actually draws (SSS tiles through world/screen) with
// the contact inspector's validated /api/crop path for the same 20 m window.
async function checkAlignment() {
  if (!S.sss || S.sssDown) { info.textContent = 'SSS service unavailable (still probing port 8090); cannot check yet.'; return; }
  const [e, n] = world(map.clientWidth / 2, map.clientHeight / 2);
  const saved = {scale: S.scale, ox: S.ox, oy: S.oy, layers: S.layers, split: S.split, base: S.base};
  Object.assign(S, {scale: 10, base: 'sss', split: false, layers: new Set()});
  centreOn(e, n);
  info.textContent = 'Checking SSS tile alignment…';
  for (let i = 0; i < 200 && S.pending.size; i++) await new Promise(r => setTimeout(r, 50));
  draw();
  const d = devicePixelRatio || 1, W = 200, span = W / 10;
  const cx = Math.round(map.clientWidth / 2), cy = Math.round(map.clientHeight / 2);
  const dev = bc.getImageData(Math.round((cx - W / 2) * d), Math.round((cy - W / 2) * d), Math.round(W * d), Math.round(W * d));
  const dw = Math.round(W * d), a = new Float32Array(W * W);
  for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) {   // box-average device pixels to 0.1 m cells
    let sum = 0, k = 0;
    for (let yy = Math.floor(y * d); yy < Math.floor((y + 1) * d); yy++) for (let xx = Math.floor(x * d); xx < Math.floor((x + 1) * d); xx++) { sum += dev.data[(yy * dw + xx) * 4]; k++; }
    a[y * W + x] = sum / k;
  }
  Object.assign(S, saved); draw();
  const blob = await fetch(`sss/api/crop?dataset=costa&e=${e}&n=${n}&span=${span}`).then(r => r.blob());
  const bmp = await createImageBitmap(blob), off = new OffscreenCanvas(bmp.width, bmp.height), oc2 = off.getContext('2d');
  oc2.drawImage(bmp, 0, 0);
  const b = oc2.getImageData(0, 0, bmp.width, bmp.height).data, bw = bmp.width;
  const corr = (dx, dy) => {
    let k = 0, sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
    for (let y = 8; y < W - 8; y++) for (let x = 8; x < W - 8; x++) {
      const p = a[y * W + x], q = b[((y + dy) * bw + x + dx) * 4];
      k++; sa += p; sb += q; saa += p * p; sbb += q * q; sab += p * q;
    }
    return (sab / k - sa / k * (sb / k)) / Math.sqrt((saa / k - (sa / k) ** 2) * (sbb / k - (sb / k) ** 2));
  };
  let best = {dx: 0, dy: 0, r: -2}, r0 = corr(0, 0);
  for (let dy = -6; dy <= 6; dy++) for (let dx = -6; dx <= 6; dx++) { const r = corr(dx, dy); if (r > best.r) best = {dx, dy, r}; }
  const lines = [`SSS tile rendering vs validated crop API, ${span} m window at E ${e.toFixed(2)} N ${n.toFixed(2)}`,
    `crop ${bmp.width}x${bmp.height} px @ 0.1 m; correlation at zero shift r=${r0.toFixed(3)}`,
    `best shift dx=${(best.dx * 0.1).toFixed(1)} m, dy=${(-best.dy * 0.1).toFixed(1)} m (r=${best.r.toFixed(3)})`,
    Math.abs(best.dx) <= 1 && Math.abs(best.dy) <= 1 && best.r > 0.6
      ? 'OK: both paths agree within one native 0.1 m pixel (the crop path resamples its window by up to one pixel).'
      : 'WARNING: disagreement larger than one native pixel or weak texture here; try a textured area.'];
  info.textContent = lines.join('\n');
}

// ------------------------------------------------------------------ UI wiring
document.querySelectorAll('input[name=base]').forEach(x => x.onchange = e => { S.base = e.target.value; splitLabel(); draw(); });
document.querySelectorAll('.layer').forEach(x => x.onchange = e => { e.target.checked ? S.layers.add(e.target.dataset.k) : S.layers.delete(e.target.dataset.k); draw(); });
$('#opacity').oninput = e => { S.opacity = +e.target.value; draw(); };
$('#fit-mbes').onclick = () => S.audit.mbes && fitBounds(S.audit.mbes.bounds);
$('#fit-sss').onclick = () => S.sss && fitBounds(S.sss.bounds);
$('#check').onclick = checkAlignment;
function splitLabel() {
  $('#split').textContent = S.split ? 'Split: SSS | ' + ({bathy: 'MBES', rough: 'roughness', none: '—'}[S.base === 'sss' ? 'bathy' : S.base]) : 'Split: off';
}
$('#split').onclick = () => { S.split = !S.split; splitLabel(); draw(); };
$('#search').onkeydown = e => { if (e.key === 'Enter') search(e.target.value); };
let down = null, dragSplit = false;
const nearSplit = e => S.split && Math.abs(e.offsetX - map.clientWidth * S.splitX) < 8;
map.onpointerdown = e => { dragSplit = nearSplit(e); down = [e.clientX, e.clientY, S.ox, S.oy]; map.setPointerCapture(e.pointerId); };
map.onpointermove = e => {
  const p = world(e.offsetX, e.offsetY);
  $('#coords').textContent = `E ${p[0].toFixed(1)}  N ${p[1].toFixed(1)} · EPSG:32630`;
  map.style.cursor = dragSplit || (!down && nearSplit(e)) ? 'col-resize' : 'crosshair';
  if (down && dragSplit) { S.splitX = Math.min(.95, Math.max(.05, e.offsetX / map.clientWidth)); draw(); }
  else if (down) { S.ox = down[2] + e.clientX - down[0]; S.oy = down[3] + e.clientY - down[1]; draw(); }
};
map.onpointerup = e => {
  if (down && !dragSplit && Math.hypot(e.clientX - down[0], e.clientY - down[1]) < 3) {
    const [x, y] = world(e.offsetX, e.offsetY);
    select(hit(x, y), x, y);
    draw();
  }
  down = null; dragSplit = false;
};
map.onwheel = e => { e.preventDefault(); zoomAt(e.offsetX, e.offsetY, e.deltaY < 0 ? 1.25 : .8); };

// ------------------------------------------------------------------ startup
function ramp(stops, lo, hi, unit, title) {
  const css = stops.map(([p, c]) => `rgb(${c.join(',')}) ${p * 100}%`).join(',');
  return `<div class="ramp"><b>${title}</b><i style="background:linear-gradient(90deg,${css})"></i><span>${lo.toFixed(1)} ${unit}</span><span class="r">${hi.toFixed(1)} ${unit}</span></div>`;
}
Promise.all([
  fetch('data/vectors.json').then(r => r.json()),
  fetch('data/audit.json').then(r => r.json()),
  loadSSSConfig(),
]).then(([v, a, sssReady]) => {
  S.vectors = v; S.audit = a;
  for (const fc of Object.values(v)) for (const f of fc.features) f.bbox = bbox(f.geometry);
  if (!sssReady) { S.sssDown = true; probeSSS(); }
  $('#legend').innerHTML = Object.entries(COL).map(([k, c]) => `<div><i class="key" style="background:${c}"></i>${k}</div>`).join('')
    + '<div><i class="key dot"></i>SSS contact</div><div><i class="key dot" style="background:#16c6e8"></i>sand wave</div>';
  const m = a.mbes;
  if (m) {
    const E = [[0, [8, 29, 88]], [.25, [33, 102, 172]], [.5, [67, 162, 202]], [.75, [171, 217, 187]], [1, [250, 240, 200]]];
    const R = [[0, [255, 255, 235]], [.5, [245, 160, 60]], [1, [128, 18, 30]]];
    $('#ramps').innerHTML = ramp(E, m.elevation.range[0], m.elevation.range[1], 'm LAT', 'MBES elevation')
      + ramp(R, 0, m.roughness.range[1], 'm', 'roughness');
    $('#method').textContent = `MBES: ${m.member} (${m.resolution_m} m nodes, ${m.source_column}). Roughness: ${m.roughness.method}; ${m.roughness.units}. Valid: ${m.roughness.valid_support}.`;
  } else {
    $('#method').textContent = 'Audit has no MBES build: run build.py.';
  }
  window.costaViewer = {S, draw, world, screen, fitBounds, centreOn};  // console/debug handle
  // The first fit waits for a laid-out map (a hidden tab reports 0 x 0).
  let fitted = false;
  new ResizeObserver(() => {
    if (!map.clientWidth || !map.clientHeight) return;
    draw();
    if (fitted) return;
    fitted = true;
    if (m) fitBounds(m.bounds); else if (S.sss) fitBounds(S.sss.bounds);
  }).observe(map);
}).catch(err => { empty.hidden = false; empty.textContent = `Failed to load data: ${err}`; });
})();
