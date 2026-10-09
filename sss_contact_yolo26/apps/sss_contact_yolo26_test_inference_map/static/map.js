/* Live inference map (local only): test area of the bundle, an uploaded sonar image, or uploaded patches. */
(function () {
  "use strict";
  var NS = "http://www.w3.org/2000/svg", P = 150;
  var S = { st: null, selIdx: -1, results: {}, since: 0, polling: false, sel: null, view: null, mosaicShown: false };
  function $(id) { return document.getElementById(id); }
  function el(tag, attrs, parent) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }
  function conf() { var v = parseFloat($("conf").value); return isFinite(v) ? v : 0.5; }
  function api(p) { return fetch(p, { cache: "no-store" }).then(function (r) { return r.json(); }); }
  function post(p, body) {
    return fetch(p, { method: "POST", headers: { "Content-Type": "application/json", "X-SSS-Tool": "1" }, body: JSON.stringify(body || {}) })
      .then(function (r) { return r.json().then(function (j) { if (!r.ok) { var e = new Error(j.error); e.data = j; throw e; } return j; }); });
  }
  function isUpload() { return S.st && S.st.source.kind === "image"; }
  function isPatchSet() { return S.st && S.st.source.kind === "patches"; }
  function inferByDefault(p) { return p.infer; }
  function baseClass(p, withBuffer) { return p.split === "nodata" ? "nodata" : p.infer || (withBuffer && p.split === "buffer") ? "pending" : "buf"; }

  // ------------------------------------------------------------------ map (canvas)
  // Drawn on one canvas, only what is on screen: the background comes as a tile pyramid from the server
  // (/api/tile, 512 px tiles, level z = 2**z downsampled) and cells, labels and contacts are batched paths.
  var cv, ctx, panZoomOn = false, drawQueued = false, mode = "plain", bg = null;
  var state = {}, flash = {}, tiles = new Map(), cellIdx = null, contactIdx = null, T = 512, BUCKET = 200;
  var COL = {};
  function cssVar(n) { return getComputedStyle(document.documentElement).getPropertyValue(n).trim(); }
  function requestDraw() { if (!drawQueued) { drawQueued = true; requestAnimationFrame(draw); } }
  // buckets of BUCKET map px -> items whose box [x0, y0, x1, y1] touches them (for culling and hit tests);
  // a per-item frame mark avoids duplicates without building a set every frame
  function index(items, boxOf) {
    var idx = { b: {}, n: items.length, marks: new Uint32Array(items.length), stamp: 0, bounds: [Infinity, Infinity, -Infinity, -Infinity] };
    items.forEach(function (it, i) {
      var b = boxOf(it);
      idx.bounds = [Math.min(idx.bounds[0], b[0]), Math.min(idx.bounds[1], b[1]), Math.max(idx.bounds[2], b[2]), Math.max(idx.bounds[3], b[3])];
      for (var gx = Math.floor(b[0] / BUCKET); gx <= Math.floor(b[2] / BUCKET); gx++)
        for (var gy = Math.floor(b[1] / BUCKET); gy <= Math.floor(b[3] / BUCKET); gy++)
          (idx.b[gx + "," + gy] = idx.b[gx + "," + gy] || []).push(i);
    });
    return idx;
  }
  function query(idx, x0, y0, x1, y1) {
    var out = [], bd = idx.bounds, i;
    if (x0 <= bd[0] && y0 <= bd[1] && x1 >= bd[2] && y1 >= bd[3]) {       // everything visible: skip the buckets
      for (i = 0; i < idx.n; i++) out.push(i);
      return out;
    }
    var st = ++idx.stamp;
    for (var gx = Math.floor(x0 / BUCKET); gx <= Math.floor(x1 / BUCKET); gx++)
      for (var gy = Math.floor(y0 / BUCKET); gy <= Math.floor(y1 / BUCKET); gy++) {
        var list = idx.b[gx + "," + gy];
        if (list) for (var j = 0; j < list.length; j++) { i = list[j]; if (idx.marks[i] !== st) { idx.marks[i] = st; out.push(i); } }
      }
    return out;
  }
  var CB = null;                                     // cell boxes, 4 numbers per patch, computed once per map
  function cellBox(p) {
    var ins = S.st.cell_inset, x = p.x - S.st.origin[0] + ins, y = p.y - S.st.origin[1] + ins, sz = P - 2 * ins;
    return [x, y, x + sz, y + sz];
  }
  function contactBox(c) {
    var b = c.box, o = S.st.origin, pad = 3;
    return [b[0] - o[0] - pad, b[1] - o[1] - pad, b[2] - o[0] + pad, b[3] - o[1] + pad];
  }
  function build(st) {
    S.st = st; state = {}; flash = {};
    cv = $("map"); ctx = cv.getContext("2d");
    T = st.tile_px || 512;
    COL = { pending: cssVar("--pending"), done: cssVar("--done"), det: cssVar("--det"), buf: cssVar("--buf"), nodata: "#24272d",
            gt: cssVar("--gt"), red: cssVar("--red"), sel: cssVar("--sel"), bg: "#0f1114" };
    st.patches.forEach(function (p) { state[p.id] = baseClass(p, false); });
    CB = new Float64Array(st.patches.length * 4);
    st.patches.forEach(function (p, i) { var b = cellBox(p); CB[4 * i] = b[0]; CB[4 * i + 1] = b[1]; CB[4 * i + 2] = b[2] - b[0]; CB[4 * i + 3] = b[3] - b[1]; });
    cellIdx = index(st.patches, cellBox); contactIdx = null;
    showBackground();
    fit();
    if (!panZoomOn) { panZoomOn = true; enablePanZoom(); window.addEventListener("resize", applyView); }
  }
  // Before a run: an uploaded image is shown under semi-transparent cells; the bundle test area has no picture yet.
  function showBackground() {
    mode = isUpload() ? "preview" : "plain";
    bg = isUpload() ? { layer: "source", v: "s-" + S.st.source.key } : null;
    requestDraw();
  }
  function levels() { var z = 0, w = Math.max(S.st.size[0], S.st.size[1]); while (w > T) { w = Math.ceil(w / 2); z++; } return z + 1; }
  function tile(z, tx, ty) {
    var url = "/api/tile/" + bg.layer + "/" + z + "/" + tx + "/" + ty + ".png?v=" + encodeURIComponent(bg.v);
    var t = tiles.get(url);
    if (!t) {
      t = new Image(); t.onload = requestDraw; t.src = url; tiles.set(url, t);
      if (tiles.size > 256) tiles.delete(tiles.keys().next().value);      // oldest first (~1 MB each decoded)
    }
    return t.complete && t.naturalWidth ? t : null;
  }
  function drawLevel(z, k, v) {
    var f = Math.pow(2, z), lw = S.st.size[0], lh = S.st.size[1];
    for (var i = 0; i < z; i++) { lw = Math.ceil(lw / 2); lh = Math.ceil(lh / 2); }
    var tx0 = Math.max(0, Math.floor(v[0] / f / T)), tx1 = Math.min(Math.ceil(lw / T) - 1, Math.floor(v[2] / f / T));
    var ty0 = Math.max(0, Math.floor(v[1] / f / T)), ty1 = Math.min(Math.ceil(lh / T) - 1, Math.floor(v[3] / f / T));
    var seam = 0.5 / k;                                                      // hide hairlines between tiles
    for (var ty = ty0; ty <= ty1; ty++)
      for (var tx = tx0; tx <= tx1; tx++) {
        var im = tile(z, tx, ty);
        if (im) ctx.drawImage(im, tx * T * f, ty * T * f, im.naturalWidth * f + seam, im.naturalHeight * f + seam);
      }
  }
  function view() {
    var r = cv.getBoundingClientRect(), s = Math.max(S.view[2] / r.width, S.view[3] / r.height);
    return { r: r, s: s, ox: S.view[0] - (r.width * s - S.view[2]) / 2, oy: S.view[1] - (r.height * s - S.view[3]) / 2 };
  }
  function draw() {
    drawQueued = false;
    if (!S.st) return;
    var dpr = window.devicePixelRatio || 1, w = view(), W = Math.round(w.r.width * dpr), H = Math.round(w.r.height * dpr);
    if (!W || !H) return;
    if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
    var k = dpr / w.s;                                                       // device px per map px
    var v = [w.ox, w.oy, w.ox + w.r.width * w.s, w.oy + w.r.height * w.s];   // visible map rectangle
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = COL.bg; ctx.fillRect(0, 0, W, H);
    ctx.setTransform(k, 0, 0, k, -w.ox * k, -w.oy * k);
    if (bg) {
      var n = levels(), z = Math.max(0, Math.min(n - 1, Math.floor(Math.log2(1 / k))));
      ctx.save(); ctx.beginPath(); ctx.rect(0, 0, S.st.size[0], S.st.size[1]); ctx.clip();
      ctx.imageSmoothingEnabled = true;
      drawLevel(n - 1, k, v);                                                // coarse base while finer tiles load
      ctx.imageSmoothingEnabled = !(z === 0 && k >= 1);                      // pixel-exact when zoomed in
      if (z < n - 1) drawLevel(z, k, v);
      ctx.restore();
    }
    var vis = query(cellIdx, v[0], v[1], v[2], v[3]), pts = S.st.patches;
    var grid = $("show-grid").checked, px1 = 1 / k, groups = {}, i, j, list;
    for (j = 0; j < vis.length; j++) { i = vis[j]; (groups[state[pts[i].id]] = groups[state[pts[i].id]] || []).push(i); }
    function rects(ixs, inset) {
      ctx.beginPath();
      for (var n = 0; n < ixs.length; n++) { var q = 4 * ixs[n]; ctx.rect(CB[q] + inset, CB[q + 1] + inset, CB[q + 2] - 2 * inset, CB[q + 3] - 2 * inset); }
    }
    Object.keys(groups).forEach(function (cls) {
      list = groups[cls];
      if (mode === "plain") {                                                // opaque cells with a 6 px dark gap
        ctx.fillStyle = COL[cls] || COL.pending; rects(list, 3); ctx.fill();
      } else if (mode === "preview" && cls !== "nodata") {                   // image shows through
        ctx.globalAlpha = 0.45; ctx.fillStyle = COL[cls] || COL.pending; rects(list, 0); ctx.fill(); ctx.globalAlpha = 1;
      }
    });
    if (grid && mode !== "plain") {
      ctx.strokeStyle = mode === "mosaic" ? "rgba(255,255,255,.18)" : "rgba(15,17,20,.6)";
      ctx.lineWidth = Math.max(2, px1); rects(vis, 0); ctx.stroke();
    }
    var now = performance.now(), fl = vis.filter(function (i) { return flash[pts[i].id] > now; });
    if (fl.length) {
      ctx.strokeStyle = "#fff"; ctx.lineWidth = Math.max(6, 2 * px1); rects(fl, 0); ctx.stroke();
      setTimeout(requestDraw, 120);
    }
    if ($("show-gt").checked) {
      ctx.strokeStyle = COL.gt; ctx.lineWidth = Math.max(8, 1.5 * px1);
      rects(vis.filter(function (i) { return pts[i].gt.length; }), 0); ctx.stroke();
    }
    if (S.contacts && contactIdx) {                                          // 2.5 screen px, like the old non-scaling stroke
      ctx.strokeStyle = COL.red; ctx.lineWidth = 2.5 * dpr / k; ctx.beginPath();
      query(contactIdx, v[0], v[1], v[2], v[3]).forEach(function (i) { var b = contactBox(S.contacts[i]); ctx.rect(b[0], b[1], b[2] - b[0], b[3] - b[1]); });
      ctx.stroke();
    }
    if (S.sel) {
      var sp = S.st.patches[S.selIdx];
      if (sp) { var b = cellBox(sp); ctx.strokeStyle = COL.sel; ctx.lineWidth = Math.max(14, 3 * px1); ctx.strokeRect(b[0], b[1], b[2] - b[0], b[3] - b[1]); }
    }
  }
  function toMap(e) { var w = view(); return [w.ox + (e.clientX - w.r.left) * w.s, w.oy + (e.clientY - w.r.top) * w.s]; }
  // topmost thing under a map point: a contact box first, then the patch cell
  function hit(m) {
    var tol = 4 * view().s;
    if (S.contacts && contactIdx) {
      var cs = query(contactIdx, m[0] - tol, m[1] - tol, m[0] + tol, m[1] + tol).filter(function (i) {
        var b = contactBox(S.contacts[i]); return m[0] >= b[0] - tol && m[0] <= b[2] + tol && m[1] >= b[1] - tol && m[1] <= b[3] + tol;
      });
      if (cs.length) return { contact: S.contacts[Math.min.apply(null, cs)] };        // highest confidence
    }
    var ps = query(cellIdx, m[0], m[1], m[0], m[1]).filter(function (i) {
      var b = cellBox(S.st.patches[i]); return m[0] >= b[0] && m[0] <= b[2] && m[1] >= b[1] && m[1] <= b[3];
    });
    return ps.length ? { patch: S.st.patches[Math.max.apply(null, ps)] } : null;      // drawn last = on top
  }
  function fit() {
    var w = S.st.size[0], h = S.st.size[1], m = MARGIN;
    S.view = [-m, -m, w + 2 * m, h + 2 * m];
    applyView();
  }
  // Zoom limits: zoomed out at most so the whole test area fits the map panel; zoomed in at most so
  // one 150 px patch fills the panel's shorter side. The view centre is kept on the map while panning.
  var MARGIN = 200;
  function clampView(v) {
    var r = cv.getBoundingClientRect(), W = S.st.size[0], H = S.st.size[1];
    if (!r.width || !r.height) return v;
    var s = Math.max(v[2] / r.width, v[3] / r.height);
    var sMax = Math.max((W + 2 * MARGIN) / r.width, (H + 2 * MARGIN) / r.height);
    var sMin = P / Math.min(r.width, r.height);
    var k = Math.min(Math.max(s, sMin), sMax) / s;
    var cx = v[0] + v[2] / 2, cy = v[1] + v[3] / 2, w = v[2] * k, h = v[3] * k;
    cx = Math.min(Math.max(cx, 0), W); cy = Math.min(Math.max(cy, 0), H);
    return [cx - w / 2, cy - h / 2, w, h];
  }
  function applyView() { S.view = clampView(S.view); requestDraw(); }
  // Zoom by factor k around a screen point; the factor is clamped to the zoom limits first, so at the
  // limit nothing moves (previously the anchor shift still applied and the map slid sideways).
  function zoomAt(clientX, clientY, k) {
    var r = cv.getBoundingClientRect(), W = S.st.size[0], H = S.st.size[1];
    var s = Math.max(S.view[2] / r.width, S.view[3] / r.height);
    var sMax = Math.max((W + 2 * MARGIN) / r.width, (H + 2 * MARGIN) / r.height), sMin = P / Math.min(r.width, r.height);
    k = Math.min(Math.max(s * k, sMin), sMax) / s;
    if (Math.abs(k - 1) < 1e-6) return;
    var px = S.view[0] + (clientX - r.left) * s - (r.width * s - S.view[2]) / 2;
    var py = S.view[1] + (clientY - r.top) * s - (r.height * s - S.view[3]) / 2;
    S.view = [px - (px - S.view[0]) * k, py - (py - S.view[1]) * k, S.view[2] * k, S.view[3] * k];
    applyView();
  }
  function enablePanZoom() {
    // Two fingers (trackpad pinch = wheel+ctrlKey, two-finger up/down, mouse wheel) only zoom;
    // sideways two-finger swipes are ignored. Panning is press-and-drag only.
    cv.addEventListener("wheel", function (e) {
      e.preventDefault();
      var dy = e.deltaY * (e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 400 : 1);
      if (!e.ctrlKey && Math.abs(e.deltaX) > Math.abs(e.deltaY)) return;
      zoomAt(e.clientX, e.clientY, Math.exp(dy * (e.ctrlKey ? 0.01 : 0.0025)));
    }, { passive: false });
    // Safari reports trackpad pinch as gesture events instead of ctrl+wheel
    var lastScale = 1;
    cv.addEventListener("gesturestart", function (e) { e.preventDefault(); lastScale = 1; });
    cv.addEventListener("gesturechange", function (e) { e.preventDefault(); zoomAt(e.clientX, e.clientY, lastScale / e.scale); lastScale = e.scale; });
    cv.addEventListener("gestureend", function (e) { e.preventDefault(); });
    var start = null;
    cv.addEventListener("pointerdown", function (e) {
      if (e.button !== 0) return;
      start = { x: e.clientX, y: e.clientY, v: S.view.slice(), id: e.pointerId }; S.dragMoved = false;
    });
    window.addEventListener("pointermove", function (e) {
      if (!start || e.pointerId !== start.id) return;
      var dx = e.clientX - start.x, dy = e.clientY - start.y;
      if (Math.abs(dx) + Math.abs(dy) > 4) { S.dragMoved = true; cv.classList.add("dragging"); $("tip").hidden = true; }
      var r = cv.getBoundingClientRect(), s = Math.max(start.v[2] / r.width, start.v[3] / r.height);
      S.view = [start.v[0] - dx * s, start.v[1] - dy * s, start.v[2], start.v[3]];
      applyView();
    });
    window.addEventListener("pointerup", function () { start = null; cv.classList.remove("dragging"); setTimeout(function () { S.dragMoved = false; }, 0); });
    cv.addEventListener("click", function (e) {
      if (S.dragMoved) return;
      var h = hit(toMap(e));
      if (h) select(h.contact ? h.contact.patches[0] : h.patch.id);
    });
    var tipQueued = null;
    cv.addEventListener("pointermove", function (e) {
      if (start) return;
      tipQueued = e;
      requestAnimationFrame(function () {
        if (!tipQueued) return;
        var ev = tipQueued; tipQueued = null;
        var h = hit(toMap(ev)), tip = $("tip"), text = "";
        if (h && h.contact) {
          var c = h.contact, b = c.box;
          text = "contact " + c.contact_id + " · conf " + c.conf.toFixed(2) +
            (c.easting != null ? " · E " + c.easting + " N " + c.northing : " · 像素 " + Math.round((b[0] + b[2]) / 2) + ", " + Math.round((b[1] + b[3]) / 2));
        } else if (h && h.patch.name) text = h.patch.name;
        tip.hidden = !text;
        if (text) {
          var r = cv.getBoundingClientRect();
          tip.textContent = text; tip.style.left = (ev.clientX - r.left + 12) + "px"; tip.style.top = (ev.clientY - r.top + 12) + "px";
        }
      });
    });
    cv.addEventListener("pointerleave", function () { tipQueued = null; $("tip").hidden = true; });
  }

  // ------------------------------------------------------------------ results
  function hasDet(id) { return (S.results[id] || []).some(function (d) { return d[4] >= conf(); }); }
  function paint(id, fresh) {
    if (!(id in state)) return;
    state[id] = hasDet(id) ? "det" : "done";
    if (fresh) flash[id] = performance.now() + 400;
    requestDraw();
  }
  function repaintAll() { Object.keys(S.results).forEach(function (id) { paint(id, false); }); }
  function counts(run) {
    var ids = Object.keys(S.results), det = ids.filter(hasDet).length;
    var boxes = ids.reduce(function (n, id) { return n + S.results[id].filter(function (d) { return d[4] >= conf(); }).length; }, 0);
    var h = "已推理 <b>" + ids.length + "</b> / " + (run.total || S.st.patches.filter(inferByDefault).length) +
      " 个 patch · 有检出（≥" + conf().toFixed(2) + "）<b>" + det + "</b> 个 patch · 检出框 <b>" + boxes + "</b>";
    if (S.contacts) h += " · 合并重叠后 contact <b>" + S.contacts.length + "</b> 个";
    if (isUpload() && S.st.source.patches_nodata) h += " · 无数据跳过 " + S.st.source.patches_nodata + " 个 patch";
    if (run.elapsed_s) h += " · 用时 " + run.elapsed_s + " s";
    if (run.ultralytics) h += " · ultralytics " + run.ultralytics;
    $("counts").innerHTML = h;
    $("bar").style.width = (run.total ? 100 * (run.done || 0) / run.total : 0) + "%";
  }
  function busy(run) { return ["loading", "running", "mosaic"].indexOf(run.status) >= 0; }
  function setStatus(run) {
    var t = { idle: "未开始", loading: "加载模型…", running: "推理中…", mosaic: "推理完成，正在拼接完整地图…", done: "完成 ✓", error: "出错：" + run.error }[run.status] || run.status;
    $("status").textContent = t;
    $("status").className = "status" + (run.status === "error" ? " err" : "");
    PICKERS.forEach(function (id) { $(id).disabled = busy(run); });
    $("btn-start").textContent = run.status === "done" || run.status === "error" ? "▶ 重新推理" : "▶ 开始推理";
  }
  function poll() {
    if (S.polling) return;
    S.polling = true;
    (function tick() {
      api("/api/progress?since=" + S.since).then(function (d) {
        d.events.forEach(function (ev) { S.results[ev.id] = ev.dets; paint(ev.id, true); });
        S.since += d.events.length;
        setStatus(d.run); counts(d.run);
        if (S.sel && d.events.some(function (ev) { return ev.id === S.sel; })) showDetail(S.sel);
        if (d.run.status === "done") { S.polling = false; finish(d.run); return; }
        if (d.run.status === "error") { S.polling = false; return; }
        setTimeout(tick, 300);
      }).catch(function () { setTimeout(tick, 1000); });
    })();
  }
  function finish(run) {
    showMosaic(run);
    loadContacts().then(function () { counts(run); });
    $("outputs").hidden = false;
    $("out-dir").textContent = run.output_dir;
    $("out-list").innerHTML = (run.files || []).map(function (f) { return '<li><a href="/api/output/' + f + '?dl=1">' + f + "</a></li>"; }).join("");
  }
  function showMosaic(run) {
    if (S.mosaicShown) return;
    S.mosaicShown = true;
    // full-resolution mosaic (0.1 m/px) as tiles; an uploaded image is its own mosaic
    mode = "mosaic";
    bg = isUpload() ? { layer: "source", v: "s-" + S.st.source.key } : { layer: "mosaic", v: "m-" + S.st.source.key + "-" + run.run_id };
    requestDraw();
  }
  var contactsSeq = 0;
  function loadContacts() {
    var seq = ++contactsSeq;                               // only the latest request is drawn
    return api("/api/contacts?conf=" + conf()).then(function (list) {
      if (seq !== contactsSeq) return;
      S.contacts = list;
      contactIdx = index(list, contactBox);
      requestDraw();
    });
  }
  function resetMap(withBuffer) {
    S.results = {}; S.since = 0; S.contacts = null; contactIdx = null; S.mosaicShown = false; flash = {};
    S.st.patches.forEach(function (p) { state[p.id] = baseClass(p, withBuffer); });
    showBackground();
    $("outputs").hidden = true;
  }

  // ------------------------------------------------------------------ detail
  function select(id) {
    S.sel = id;
    S.selIdx = S.st.patches.findIndex(function (p) { return p.id === id; });
    requestDraw();
    showDetail(id);
  }
  function showDetail(id) {
    var p = S.st.patches.filter(function (q) { return q.id === id; })[0], dets = S.results[id];
    $("empty").hidden = true; $("detail").hidden = false;
    $("d-title").textContent = id;
    var where = S.st.utm0
      ? "中心 E " + (S.st.utm0[0] + S.st.m_per_px * (p.x + 75)).toFixed(1) + " N " + (S.st.utm0[1] - S.st.m_per_px * (p.y + 75)).toFixed(1)
      : "中心像素 " + (p.x + 75) + ", " + (p.y + 75) + "（无地理坐标）";
    var head = isUpload() ? "上传图像" + (p.split === "nodata" ? " · 无数据，未推理" : "")
      : (isPatchSet() ? p.name + " · " : "") + (p.chunk !== "" ? "spatial chunk " + p.chunk + " · " + p.group + " · " + p.split : "不在 bundle 清单里");
    if (!S.st.utm0 && S.st.cell_inset === 0) where = "文件名里没有坐标，按文件名排列";
    $("d-meta").textContent = head + " · " + where +
      " · " + (dets ? "已推理，" + dets.filter(function (d) { return d[4] >= conf(); }).length + " 个检出 ≥ " + conf().toFixed(2) : "尚未推理");
    var img = $("d-img"), key = S.st.source.key + "/" + id;
    if (img.dataset.id !== key) { img.dataset.id = key; img.src = "/api/image/" + id + ".png?k=" + S.st.source.key; }
    var s = $("d-svg"); while (s.firstChild) s.removeChild(s.firstChild);
    function box(b, color, dash, label) {
      el("rect", { x: (b[0] - b[2] / 2) * P, y: (b[1] - b[3] / 2) * P, width: b[2] * P, height: b[3] * P, fill: "none", stroke: color,
        "stroke-width": 2, "vector-effect": "non-scaling-stroke", "stroke-dasharray": dash || "none" }, s);
      if (label) {
        var t = el("text", { x: (b[0] - b[2] / 2) * P, y: (b[1] - b[3] / 2) * P - 1.5, fill: color, "font-size": 5, "font-family": "Menlo, monospace",
          stroke: "#000", "stroke-width": 0.6, "paint-order": "stroke" }, s);
        t.textContent = label;
      }
    }
    if ($("d-gt").checked) p.gt.forEach(function (g) { box(g, "#ff9f1c", "4 2", "label"); });
    var rows = ["<tr><th>#</th><th>置信度</th><th>cx, cy, w, h (YOLO)</th><th>像素 x0,y0–x1,y1</th></tr>"];
    (dets || []).forEach(function (d, i) {
      var hi = d[4] >= conf();
      if (hi || $("d-low").checked) box(d, hi ? "#ff2d2d" : "#9aa1ad", hi ? null : "3 2", d[4].toFixed(2));
      rows.push("<tr><td>" + i + "</td><td>" + d[4].toFixed(3) + (hi ? "" : " (低于阈值)") + "</td><td>" + d.slice(0, 4).map(function (v) { return v.toFixed(3); }).join(", ") +
        "</td><td>" + [(d[0] - d[2] / 2) * P, (d[1] - d[3] / 2) * P, (d[0] + d[2] / 2) * P, (d[1] + d[3] / 2) * P].map(function (v) { return v.toFixed(0); }).join(", ") + "</td></tr>");
    });
    if (!dets || !dets.length) rows.push('<tr><td colspan="4">' + (dets ? "没有检出（≥0.05）" : "尚未推理") + "</td></tr>");
    $("d-table").innerHTML = rows.join("");
  }

  // ------------------------------------------------------------------ data source + model pickers
  function sourceLabel(s) {
    if (s.kind === "bundle") return s.name;
    if (s.kind === "patches") return "📁 " + s.name + "（" + s.count + " 个 patch）";
    return s.name + (s.size ? "（" + s.size[0] + "×" + s.size[1] + "）" : "");
  }
  function fillPickers(st) {
    $("src").innerHTML = st.sources.map(function (s) {
      return '<option value="' + esc(s.key) + '"' + (s.key === st.source.key ? " selected" : "") + ">" + esc(sourceLabel(s)) + "</option>";
    }).join("");
    $("model").innerHTML = st.models.map(function (m) {
      return '<option value="' + esc(m.key) + '"' + (m.key === st.model.key ? " selected" : "") + ">" +
        esc(m.key === "default" ? "默认 · " + m.name : m.name + " · " + m.sha256.slice(0, 8)) + "</option>";
    }).join("");
    var s = st.source, info;
    if (s.kind === "bundle") info = "导出的 150×150 patch · 0.1 m/px · UTM · 含人工标签";
    else if (s.kind === "patches") info = s.count + " 个 150×150 patch · " +
      (s.layout === "coords" ? "按文件名里的 x/y 放到地图上" : "按文件名排列（文件名里没有可用坐标）") +
      (s.layout_note ? "（" + s.layout_note + "）" : "") +
      (s.georef ? " · bundle 文件名 → UTM" : " · 无地理坐标（已确认 0.1 m/px）") +
      (s.matched_bundle ? " · " + s.matched_bundle + " 个在 bundle 清单里（带人工标签）" : "") +
      (s.skipped_count ? " · 跳过 " + s.skipped_count + " 个文件" : "") + " · 全部推理";
    else info = s.size[0] + "×" + s.size[1] + " px · " + s.format + " · " + s.bands +
      (s.georef ? " · 0.1 m/px · 左上角 E " + s.georef.E0 + " N " + s.georef.N0 + (s.georef.epsg ? " · EPSG:" + s.georef.epsg : "")
                : " · 无地理坐标（已确认 0.1 m/px，只输出像素坐标）") +
      (s.opacity ? " · " + s.opacity : "") + " · 150 px patch，每 100 px 一个";
    $("src-info").textContent = info;
    $("src-sub").textContent = s.kind === "bundle" ? "test 区域 · spatial chunks 10–11" : (s.kind === "patches" ? "patch 文件夹 · " : "上传图像 · ") + s.name;
    $("src-info").title = (s.skipped || []).map(function (k) { return "跳过 " + k.file + "：" + k.reason; }).join("\n");
    var m = st.model;
    $("model-info").textContent = (m.names ? "类别 " + m.names.join(", ") + " · " : "") + "imgsz " + (m.imgsz || 256) +
      (m.trained_ultralytics ? " · 训练用 ultralytics " + m.trained_ultralytics : "") + " · SHA-256 " + m.sha256.slice(0, 12);
    $("model-info").title = st.weights;
    document.body.classList.toggle("has-gt", s.kind === "bundle" || !!s.has_gt);
    document.body.classList.toggle("has-buffer", s.kind === "bundle");
    document.body.classList.toggle("has-nodata", s.kind === "image" && s.patches_nodata > 0);
  }
  function note(msg, err) { $("status").textContent = msg; $("status").className = "status" + (err ? " err" : ""); }
  // Switch source/model on the server, then reload the whole map for it.
  function choose(body) {
    return post("/api/select", body).catch(function (e) {
      if (e.data && e.data.needs_res_confirm) {
        var ok = confirm("「" + e.data.name + "」没有地理坐标，页面无法检查它的分辨率。\n\n" +
          "模型按 0.1 m/px 训练；只有在确定这张图就是 0.1 m/px（1 像素 = 10 cm）时才继续，否则目标尺寸不对，检测结果不可信。\n\n" +
          "结果只会有像素坐标，没有 UTM。确认是 0.1 m/px？");
        if (!ok) throw new Error("未确认分辨率，没有切换");
        return post("/api/select", Object.assign({}, body, { confirm_res: true }));
      }
      throw e;
    }).then(load);
  }
  // body: one file, or for kind "patches" a 4-byte length + JSON file list + the files back to back
  function upload(kind, body, label) {
    return new Promise(function (resolve, reject) {
      var x = new XMLHttpRequest();
      x.open("POST", "/api/upload?kind=" + kind);
      x.setRequestHeader("X-SSS-Tool", "1");
      x.setRequestHeader("X-Filename", encodeURIComponent(label));
      x.setRequestHeader("Content-Type", "application/octet-stream");
      x.upload.onprogress = function (e) {
        if (!e.lengthComputable) return;
        note(e.loaded < e.total ? "上传 " + label + " " + Math.round(100 * e.loaded / e.total) + "%"
          : { image: "检查并转换 ", patches: "检查 patch ", weights: "加载并检查模型 " }[kind] + label + "…");
      };
      x.onload = function () {
        var j; try { j = JSON.parse(x.responseText); } catch (e) { j = { error: "HTTP " + x.status }; }
        if (x.status === 200) resolve(j.item); else reject(new Error(j.error || "HTTP " + x.status));
      };
      x.onerror = function () { reject(new Error("上传中断")); };
      x.send(body);
    });
  }
  function packPatches(files) {
    var head = new TextEncoder().encode(JSON.stringify({ files: files.map(function (f) { return { name: f.webkitRelativePath || f.name, size: f.size }; }) }));
    var len = new Uint8Array(4); new DataView(len.buffer).setUint32(0, head.length);
    return new Blob([len, head].concat(files));
  }
  var PICKERS = ["btn-start", "btn-clear", "src", "model", "btn-src-file", "btn-dir-file", "btn-model-file"];
  function pickerBusy(on) { PICKERS.forEach(function (id) { $(id).disabled = on; }); }
  function send(kind, body, label, key) {
    pickerBusy(true);
    var item;
    upload(kind, body, label)
      .then(function (it) { item = it; var b = {}; b[key] = it.key; return choose(b); })
      .then(function () {
        var skip = item.skipped_count ? "，跳过 " + item.skipped_count + " 个文件（" +
          item.skipped.slice(0, 3).map(function (k) { return k.file.split("/").pop() + "：" + k.reason; }).join("；") + (item.skipped_count > 3 ? "…" : "") + "）" : "";
        note({ image: "已载入图像 ", patches: "已载入 ", weights: "已载入模型 " }[kind] + label + (kind === "patches" ? "（" + item.count + " 个 patch）" : "") + skip, !!skip);
      })
      .catch(function (e) { note("未能使用 " + label + "：" + e.message, true); return load(); });
  }
  function onFile(kind, input, fromDir) {
    var files = Array.prototype.slice.call(input.files); input.value = "";   // the same files can be chosen again later
    if (!files.length) return;
    if (kind === "weights" || (!fromDir && files.length === 1)) {
      var f = files[0], ext = (f.name.match(/\.[^.]+$/) || [""])[0].toLowerCase();
      var okExt = kind === "image" ? [".tif", ".tiff", ".png"] : [".pt"];
      if (okExt.indexOf(ext) < 0) return note("只接受 " + okExt.join("/") + " 文件：" + f.name, true);
      return send(kind, f, f.name, kind === "image" ? "source" : "model");
    }
    // several files or a folder: a set of 150x150 patches (hidden files and other types are left out here)
    var pat = files.filter(function (f) { return /\.(png|tiff?)$/i.test(f.name) && f.name.charAt(0) !== "."; });
    if (!pat.length) return note("所选内容里没有 PNG/TIFF 文件", true);
    // only one image among the chosen files (e.g. a folder with a GeoTIFF and its CSV/JSON): a whole image
    if (pat.length === 1) return send("image", pat[0], pat[0].name, "source");
    var total = pat.reduce(function (n, f) { return n + f.size; }, 0);
    if (total > 4 * 1024 * 1024 * 1024) return note("所选 patch 共 " + (total / 1e9).toFixed(1) + " GB，超过 4 GB 上限", true);
    var label = fromDir && pat[0].webkitRelativePath ? pat[0].webkitRelativePath.split("/")[0] : pat[0].name + " 等 " + pat.length + " 个文件";
    send("patches", packPatches(pat), label, "source");
  }
  $("btn-src-file").addEventListener("click", function () { $("src-file").click(); });
  $("btn-model-file").addEventListener("click", function () {
    if (confirm("模型权重（.pt）加载时会执行文件里的代码。只选择你自己训练或信任来源的权重。继续？")) $("model-file").click();
  });
  $("btn-dir-file").addEventListener("click", function () { $("dir-file").click(); });
  $("src-file").addEventListener("change", function () { onFile("image", this, false); });
  $("dir-file").addEventListener("change", function () { onFile("image", this, true); });
  $("model-file").addEventListener("change", function () { onFile("weights", this, false); });
  $("src").addEventListener("change", function () {
    pickerBusy(true);
    choose({ source: this.value }).catch(function (e) { note("无法切换数据：" + e.message, true); return load(); });
  });
  $("model").addEventListener("change", function () {
    pickerBusy(true);
    choose({ model: this.value }).catch(function (e) { note("无法切换模型：" + e.message, true); return load(); });
  });

  // ------------------------------------------------------------------ events
  $("btn-start").addEventListener("click", function () {
    var withBuffer = S.st.source.kind === "bundle" && $("buffer").checked;
    post("/api/start", { conf: conf(), include_buffer: withBuffer })
      .then(function () { resetMap(withBuffer); poll(); })
      .catch(function (e) { note("无法开始：" + e.message, true); });
  });
  $("btn-clear").addEventListener("click", function () {
    if (!confirm("删除当前数据（" + S.st.source.name + "）在 outputs/ 里保存的全部推理结果？\n（只删除本页面生成的推理结果文件，不会删除模型、图片或其他任何文件；删除后不能恢复）")) return;
    post("/api/clear_results", {})
      .then(function (j) {
        resetMap(false); S.sel = null; requestDraw();
        $("detail").hidden = true; $("empty").hidden = false;
        setStatus(j.run); counts(j.run);
        note("已删除 " + j.deleted.length + " 个推理结果文件");
      })
      .catch(function (e) { note("无法删除：" + e.message, true); });
  });
  var confTimer = null;
  $("conf").addEventListener("change", function () {
    clearTimeout(confTimer);
    confTimer = setTimeout(onConf, 250);                   // arrow-key steps: one update after they stop
  });
  function onConf() {
    repaintAll();
    if (S.mosaicShown) loadContacts().then(function () { api("/api/progress?since=999999").then(function (d) { counts(d.run); }); });
    else api("/api/progress?since=999999").then(function (d) { counts(d.run); });
    if (S.sel) showDetail(S.sel);
  }
  $("show-gt").addEventListener("change", requestDraw);
  $("show-grid").addEventListener("change", requestDraw);
  $("btn-fit").addEventListener("click", fit);
  ["d-low", "d-gt"].forEach(function (id) { $(id).addEventListener("change", function () { if (S.sel) showDetail(S.sel); }); });

  function load() {
    return api("/api/state").then(function (st) {
      S.sel = null; S.contacts = null; S.mosaicShown = false;
      $("detail").hidden = true; $("empty").hidden = false; $("outputs").hidden = true;
      if (st.source.kind !== "bundle" && !st.source.has_gt) { $("show-gt").checked = false; $("d-gt").checked = false; }
      fillPickers(st);
      build(st);
      S.results = st.results || {};
      S.since = Object.keys(S.results).length;
      repaintAll(); setStatus(st.run); counts(st.run);
      if (busy(st.run)) { S.since = 0; S.results = {}; poll(); }
      if (st.run.status === "done") finish(st.run);
    });
  }
  load();
})();
