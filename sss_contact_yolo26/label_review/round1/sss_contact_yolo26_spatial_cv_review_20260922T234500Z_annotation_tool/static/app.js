/* SSS contact review - offline front end (no external resources). */
(function () {
  "use strict";

  var W = 150, H = 150;
  var SVGNS = "http://www.w3.org/2000/svg";
  var VIEWS = [
    ["all", "All patches / 全部"],
    ["fp", "FP candidates / 未匹配预测"],
    ["fn", "FN / 未匹配 GT"],
    ["tp", "TP / 匹配"],
    ["tn", "TN"],
    ["unscored", "Unscored / 未评估"]
  ];
  var VIEW_NOTES = {
    all: "All 9,000 patches from spatial chunks 1–9. Classification uses saved OOF predictions and the current A labels.",
    fp: "", // filled from meta (English + Chinese definition)
    fn: "FN = a current A GT box without a one-to-one match among predictions at the current thresholds. It can be a missed detection or an incorrectly drawn label; deleting it is always a manual action. / FN：当前 A 中未被匹配的 GT 框；是否删除只能由人工决定。",
    tp: "TP = matched GT/prediction pair (IoU ≥ threshold, one-to-one). Default order: lowest TP IoU first.",
    tn: "TN = OOF-scored patch with 0 GT boxes in A and 0 predictions at the current confidence threshold. Use 'New random sample' for spot checks.",
    unscored: "Unscored / 未评估：these 224 patches were not held out by any fold after the fold-specific 15 m spatial guards, so no OOF prediction exists. They are never counted as TN. GT boxes can still be reviewed and edited."
  };

  var S = {
    meta: null, conf: 0.2, iou: 0.5, view: "all", page: 1, perPage: 50, sort: "default", order: "", seed: "1",
    filters: {}, list: null, selectedId: null, detail: null, sel: null, mode: "select", zoom: 4,
    overlays: { gt: true, matched: true, unmatched: true, below: false, b: false, text: true }, snap: true,
    drag: null, listReq: 0, detailReq: 0, gray: null
  };

  function $(id) { return document.getElementById(id); }
  function esc(s) {
    return String(s === null || s === undefined ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  function f2(v) { return v === null || v === undefined ? "–" : Number(v).toFixed(2); }
  function f3(v) { return v === null || v === undefined ? "–" : Number(v).toFixed(3); }
  function f4(v) { return v === null || v === undefined ? "–" : Number(v).toFixed(4); }
  function inField(el) { return el && el.closest && el.closest("input, textarea, select, [contenteditable]"); }

  // ------------------------------------------------------------------ API
  function api(path, params) {
    var q = params ? "?" + new URLSearchParams(params).toString() : "";
    return fetch(path + q, { cache: "no-store" }).then(function (r) {
      return r.json().then(function (j) { if (!r.ok) throw new Error(j.error || r.statusText); return j; });
    });
  }
  function post(path, body) {
    return fetch(path, {
      method: "POST", headers: { "Content-Type": "application/json", "X-SSS-Tool": "1" },
      body: JSON.stringify(body || {})
    }).then(function (r) {
      return r.json().then(function (j) { if (!r.ok) { var e = new Error(j.error || r.statusText); e.status = r.status; throw e; } return j; });
    });
  }

  var toastTimer = null;
  function toast(msg, isErr) {
    var t = $("toast");
    t.textContent = msg;
    t.className = "toast" + (isErr ? " err" : "");
    t.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(function () { t.hidden = true; }, isErr ? 6000 : 2600);
  }
  function saveStatus(msg, cls) {
    var s = $("save-status");
    s.textContent = msg;
    s.className = "save-status " + (cls || "");
  }

  // ------------------------------------------------------------------ init
  function init() {
    api("/api/meta").then(function (m) {
      S.meta = m;
      var ui = m.ui_state || {};
      if (typeof ui.conf === "number") S.conf = ui.conf;
      if (typeof ui.iou === "number") S.iou = ui.iou;
      if (ui.view) S.view = ui.view;
      if (ui.overlays) Object.assign(S.overlays, ui.overlays);
      if (ui.zoom) S.zoom = ui.zoom;
      if (ui.perPage) S.perPage = ui.perPage;
      if (ui.selectedId) S.selectedId = ui.selectedId;
      setupHeader(m);
      setupFilters(m);
      setupTabs();
      bindEvents();
      syncThresholdInputs();
      $("zoom").value = String(S.zoom);
      $("f-perpage").value = String(S.perPage);
      Object.keys(S.overlays).forEach(function (k) { var el = $("ov-" + k); if (el) el.checked = S.overlays[k]; });
      updateHistoryButtons(m.history);
      loadList().then(function () { if (S.selectedId) selectPatch(S.selectedId, true); });
    }).catch(function (e) { toast("Failed to load: " + e.message, true); });
  }

  function setupHeader(m) {
    $("role-zip").title = "Source ZIP (read-only; images read on demand): " + m.roles.ZIP.path;
    $("role-b").title = "Frozen backup B (never written): " + m.roles.B.label_dir;
    $("role-a").title = "Working copy A (all edits saved here): " + m.roles.A.path;
    updateRoleA(m.modified_patches);
    var bc = m.baseline_comparison_summary;
    var badge = $("baseline-badge");
    badge.textContent = bc.all_equal ? "baseline 0.20/0.50 = ZIP manifest ✓" : "baseline differs from ZIP ⚠";
    badge.className = "role role-check " + (bc.all_equal ? "ok" : "bad");
    badge.title = "Platform recomputation with B labels vs ZIP review_manifest.json and classification CSVs. Click 'Data checks…' for details.";
    VIEW_NOTES.fp = "<b>" + esc(m.fp_note_en) + "</b><br>" + esc(m.fp_note_zh) +
      "<br>Types (by current A): <code>unlabelled_model_candidate</code> = patch has 0 GT in A; <code>extra_prediction_on_labelled_patch</code> = patch has GT in A but this prediction matched none.";
    $("floor-hint").textContent = "Saved OOF predictions start at " + m.prediction_confidence_floor.toFixed(2) + "; thresholds below that are not available.";
  }
  function updateRoleA(n) {
    $("role-a").textContent = "A · working labels · " + n + " patch" + (n === 1 ? "" : "es") + " edited vs B";
  }

  function setupFilters(m) {
    var c = $("f-chunk");
    m.chunks.forEach(function (ch) {
      var o = document.createElement("option");
      o.value = String(ch); o.textContent = ch + " (" + m.chunk_to_group[String(ch)] + ")"; c.appendChild(o);
    });
    var g = $("f-group");
    m.groups.forEach(function (gr) {
      var chunk = Object.keys(m.chunk_to_group).filter(function (k) { return m.chunk_to_group[k] === gr; })[0];
      var o = document.createElement("option"); o.value = gr; o.textContent = gr + " (chunk " + chunk + ")"; g.appendChild(o);
    });
    var sp = $("f-split");
    m.final_splits.forEach(function (s) { var o = document.createElement("option"); o.value = s; o.textContent = s; sp.appendChild(o); });
    var fr = $("f-fpreview");
    m.pred_statuses.forEach(function (s) { var o = document.createElement("option"); o.value = s; o.textContent = m.pred_status_labels[s]; fr.appendChild(o); });
    var pr = $("f-patchreview"), ps = $("patch-status");
    m.patch_statuses.forEach(function (s) {
      var o = document.createElement("option"); o.value = s; o.textContent = s; pr.appendChild(o);
      var o2 = document.createElement("option"); o2.value = s; o2.textContent = s; ps.appendChild(o2);
    });
  }

  function setupTabs() {
    var nav = $("tabs");
    VIEWS.forEach(function (v) {
      var b = document.createElement("button");
      b.type = "button"; b.dataset.view = v[0];
      b.innerHTML = esc(v[1]) + '<span class="cnt"></span>';
      b.addEventListener("click", function () { S.view = v[0]; S.page = 1; S.sort = "default"; S.order = ""; $("f-sort").value = "default"; $("f-order").value = ""; loadList(); persistUi(); });
      nav.appendChild(b);
    });
  }

  function syncThresholdInputs() {
    $("conf-range").value = S.conf; $("conf-num").value = S.conf.toFixed(2);
    $("iou-range").value = S.iou; $("iou-num").value = S.iou.toFixed(2);
  }

  var thrTimer = null;
  function setThresholds(conf, iou) {
    var floor = S.meta.prediction_confidence_floor;
    if (!(conf >= floor - 1e-9)) { toast("Confidence cannot go below the saved prediction floor " + floor.toFixed(2), true); conf = floor; }
    conf = Math.min(1, conf); iou = Math.max(0.01, Math.min(1, iou));
    S.conf = Math.round(conf * 100) / 100; S.iou = Math.round(iou * 100) / 100;
    syncThresholdInputs();
    clearTimeout(thrTimer);
    thrTimer = setTimeout(function () { loadList(); if (S.selectedId) loadDetail(); persistUi(); }, 150);
  }

  var uiTimer = null;
  function persistUi() {
    clearTimeout(uiTimer);
    uiTimer = setTimeout(function () {
      post("/api/ui_state", { ui: { conf: S.conf, iou: S.iou, view: S.view, overlays: S.overlays, zoom: S.zoom, perPage: S.perPage, selectedId: S.selectedId } }).catch(function () {});
    }, 800);
  }

  // ------------------------------------------------------------------ list
  function readFilters() {
    S.filters = {
      chunk: $("f-chunk").value, group: $("f-group").value, split: $("f-split").value, q: $("f-q").value.trim(),
      fp_type: $("f-fptype").value, fp_review: $("f-fpreview").value, patch_review: $("f-patchreview").value,
      modified: $("f-modified").value, min_conf: $("f-minconf").value, max_tp_iou: $("f-maxtpiou").value, min_fp: $("f-minfp").value
    };
    S.sort = $("f-sort").value; S.order = $("f-order").value; S.perPage = Number($("f-perpage").value);
  }

  function listParams() {
    var p = { view: S.view, conf: S.conf, iou: S.iou, page: S.page, per_page: S.perPage, sort: S.sort, seed: S.seed };
    if (S.order) p.order = S.order;
    Object.keys(S.filters).forEach(function (k) { if (S.filters[k] !== "" && S.filters[k] !== undefined) p[k] = S.filters[k]; });
    return p;
  }

  function loadList() {
    var req = ++S.listReq;
    return api("/api/list", listParams()).then(function (d) {
      if (req !== S.listReq) return;
      S.list = d; S.page = d.page;
      renderTabs(d); renderStats(d); renderList(d); renderViewNote();
    }).catch(function (e) { toast("List error: " + e.message, true); });
  }

  function renderViewNote() {
    var n = $("view-note");
    n.innerHTML = S.view === "fp" ? VIEW_NOTES.fp : esc(VIEW_NOTES[S.view]);
    n.className = "view-note" + (S.view === "fp" ? " fp" : "");
  }

  function renderTabs(d) {
    Array.prototype.forEach.call($("tabs").children, function (b) {
      var v = b.dataset.view;
      b.classList.toggle("on", v === S.view);
      var f = d.view_counts_filtered[v], a = d.view_counts_all[v];
      b.querySelector(".cnt").textContent = f === a ? "(" + a + ")" : "(" + f + " / " + a + ")";
    });
  }

  function statsHtml(label, s) {
    function st(cls, name, val) { return '<span class="stat ' + cls + '">' + name + " <b>" + (val || 0) + "</b></span>"; }
    return '<span class="lbl">' + esc(label) + "</span>" +
      st("", "patches", s.patches) + st("", "GT boxes", s.gt_boxes) + st("", "pred boxes (≥conf)", s.pred_boxes) +
      st("tp", "TP boxes", s.tp_boxes) + st("tp", "TP patches", s.tp_patches) +
      st("fp", "FP cand. boxes", s.fp_boxes) + st("fp", "FP patches", s.fp_patches) +
      st("fn", "FN boxes", s.fn_boxes) + st("fn", "FN patches", s.fn_patches) +
      st("", "TN patches", s.tn_patches) + st("uns", "unscored patches", s.unscored_patches);
  }

  function renderStats(d) {
    $("stats-global").innerHTML = statsHtml("All 9,000 patches @ conf " + S.conf.toFixed(2) + " / IoU " + S.iou.toFixed(2), d.stats_global);
    var vname = VIEWS.filter(function (v) { return v[0] === S.view; })[0][1];
    $("stats-filtered").innerHTML = statsHtml("Current: " + vname + (hasFilters() ? " + filters" : ""), d.stats_view_filtered);
  }
  function hasFilters() { return Object.keys(S.filters).some(function (k) { return S.filters[k]; }); }

  function renderPager(el, d) {
    el.innerHTML = "";
    function btn(txt, page, dis) {
      var b = document.createElement("button"); b.type = "button"; b.textContent = txt; b.disabled = dis;
      b.addEventListener("click", function () { S.page = page; loadList(); });
      el.appendChild(b);
    }
    btn("⏮", 1, d.page <= 1); btn("‹ prev", d.page - 1, d.page <= 1);
    var span = document.createElement("span");
    span.textContent = "page " + d.page + " / " + d.pages + " · " + d.count + " patches · sort: " + d.sort + " " + d.order;
    el.appendChild(span);
    btn("next ›", d.page + 1, d.page >= d.pages); btn("⏭", d.pages, d.page >= d.pages);
  }

  function boxRectSvg(svg, b, cls, stroke, dash, width) {
    var r = document.createElementNS(SVGNS, "rect");
    r.setAttribute("x", (b.cx - b.w / 2) * W); r.setAttribute("y", (b.cy - b.h / 2) * H);
    r.setAttribute("width", b.w * W); r.setAttribute("height", b.h * H);
    r.setAttribute("fill", "none"); r.setAttribute("stroke", stroke); r.setAttribute("stroke-width", width || 1.5);
    r.setAttribute("vector-effect", "non-scaling-stroke");
    if (dash) r.setAttribute("stroke-dasharray", dash);
    if (cls) r.setAttribute("class", cls);
    svg.appendChild(r);
    return r;
  }

  function renderList(d) {
    renderPager($("pager-top"), d); renderPager($("pager-bottom"), d);
    var grid = $("grid");
    grid.innerHTML = "";
    if (!d.items.length) { grid.innerHTML = '<div class="empty">No patches match.</div>'; return; }
    d.items.forEach(function (it) {
      var card = document.createElement("div");
      card.className = "card" + (it.patch_id === S.selectedId ? " sel" : "");
      card.dataset.pid = it.patch_id;
      var th = document.createElement("div"); th.className = "thumb";
      var img = document.createElement("img");
      img.loading = "lazy"; img.decoding = "async"; img.width = 150; img.height = 150; img.alt = it.patch_id;
      img.src = "/api/image/" + encodeURIComponent(it.patch_id) + ".png";
      th.appendChild(img);
      var svg = document.createElementNS(SVGNS, "svg"); svg.setAttribute("viewBox", "0 0 150 150");
      it.preds.forEach(function (p) { boxRectSvg(svg, p, null, p.matched ? "#22c55e" : "#ef4444", null, 1.2); });
      it.gt.forEach(function (g) {
        boxRectSvg(svg, g, null, "#ffb000", null, 1.2);
        if (g.fn) { var e = { cx: g.cx, cy: g.cy, w: g.w + 4 / W, h: g.h + 4 / H }; boxRectSvg(svg, e, null, "#ef4444", "2 2", 1); }
      });
      th.appendChild(svg);
      card.appendChild(th);
      var badges = [];
      if (!it.scored) badges.push('<span class="b uns">unscored / 未评估</span>');
      if (it.tn) badges.push('<span class="b tn">TN</span>');
      if (it.tp) badges.push('<span class="b tp">TP ' + it.tp + (it.min_tp_iou !== null ? " · IoU " + f2(it.min_tp_iou) : "") + "</span>");
      if (it.fp) badges.push('<span class="b fp">FP cand ' + it.fp + " · " + f2(it.max_fp_conf) + "</span>");
      if (it.fn) badges.push('<span class="b fn">FN ' + it.fn + "</span>");
      if (it.fp_type) badges.push('<span class="b fp">' + (it.fp_type === "unlabelled_model_candidate" ? "unlabelled" : "extra on labelled") + "</span>");
      if (it.modified) badges.push('<span class="b mod">edited</span>');
      Object.keys(it.fp_review).forEach(function (s) { if (s !== "unreviewed") badges.push('<span class="b rev">' + esc(s) + " " + it.fp_review[s] + "</span>"); });
      if (it.patch_review !== "unreviewed") badges.push('<span class="b rev">patch: ' + esc(it.patch_review) + "</span>");
      var info = document.createElement("div");
      info.innerHTML = '<div class="pid">' + esc(it.patch_id) + "</div>" +
        '<div class="meta">chunk ' + it.spatial_chunk + " · " + esc(it.original_group) + " · split " + esc(it.final_split) + "</div>" +
        '<div class="meta">GT ' + it.gt_count + " · pred " + it.pred_count + (it.max_pred_conf !== null ? " (max " + f2(it.max_pred_conf) + ")" : "") + "</div>" +
        '<div class="badges">' + badges.join("") + "</div>";
      card.appendChild(info);
      card.addEventListener("click", function () { selectPatch(it.patch_id); });
      grid.appendChild(card);
    });
  }

  // ------------------------------------------------------------------ navigation
  function pageIds() { return S.list ? S.list.items.map(function (i) { return i.patch_id; }) : []; }
  function step(delta) {
    var ids = pageIds();
    var i = ids.indexOf(S.selectedId);
    if (i < 0) { if (ids.length) selectPatch(ids[0]); return; }
    var j = i + delta;
    if (j >= 0 && j < ids.length) { selectPatch(ids[j]); return; }
    if (j >= ids.length && S.list.page < S.list.pages) {
      S.page += 1; loadList().then(function () { var x = pageIds(); if (x.length) selectPatch(x[0]); });
    } else if (j < 0 && S.list.page > 1) {
      S.page -= 1; loadList().then(function () { var x = pageIds(); if (x.length) selectPatch(x[x.length - 1]); });
    }
  }

  function selectPatch(pid, quiet) {
    S.selectedId = pid; S.sel = null; S.gray = null;
    Array.prototype.forEach.call(document.querySelectorAll(".card"), function (c) { c.classList.toggle("sel", c.dataset.pid === pid); });
    var card = document.querySelector('.card[data-pid="' + pid + '"]');
    if (card && !quiet) card.scrollIntoView({ block: "nearest" });
    persistUi();
    return loadDetail(true);
  }

  // ------------------------------------------------------------------ detail
  function loadDetail(newPatch) {
    var pid = S.selectedId;
    if (!pid) return Promise.resolve();
    var req = ++S.detailReq;
    return api("/api/patch/" + encodeURIComponent(pid), { conf: S.conf, iou: S.iou }).then(function (d) {
      if (req !== S.detailReq) return;
      S.detail = d;
      if (S.sel && S.sel.type === "gt" && S.sel.index >= d.gt.length) S.sel = null;
      renderDetail(newPatch);
    }).catch(function (e) { toast("Detail error: " + e.message, true); });
  }

  function renderDetail(newPatch) {
    var d = S.detail;
    $("detail-empty").hidden = true; $("detail").hidden = false;
    $("d-title").textContent = d.patch_id;
    var c = d.counts, bs = [];
    if (!d.meta.scored) bs.push('<span class="b uns">unscored / 未评估 — no OOF prediction</span>');
    else {
      bs.push('<span class="b">GT ' + c.gt_count + "</span>", '<span class="b">pred ' + c.pred_count + "</span>");
      if (c.tp) bs.push('<span class="b tp">TP ' + c.tp + "</span>");
      if (c.fp) bs.push('<span class="b fp">FP cand ' + c.fp + "</span>");
      if (c.fn) bs.push('<span class="b fn">FN ' + c.fn + "</span>");
      if (c.tn) bs.push('<span class="b tn">TN</span>');
    }
    if (d.modified) bs.push('<span class="b mod">A edited vs B</span>');
    $("d-badges").innerHTML = bs.join(" ");
    if (newPatch) saveStatus("", "");

    var stage = $("stage"), img = $("d-img"), svg = $("d-svg");
    var px = W * S.zoom;
    stage.style.width = px + "px"; stage.style.height = px + "px";
    img.style.width = px + "px"; img.style.height = px + "px";
    svg.setAttribute("width", px); svg.setAttribute("height", px);
    var src = "/api/image/" + encodeURIComponent(d.patch_id) + ".png";
    if (img.dataset.pid !== d.patch_id) {
      img.dataset.pid = d.patch_id;
      img.onload = function () { onImageLoad(img); };
      img.src = src;
    }
    stage.classList.toggle("draw", S.mode === "draw");
    drawOverlay();
    renderSelection();
    renderTables();
    renderMeta();
  }

  function onImageLoad(img) {
    // read pixels (same-origin) for the cursor grey-value readout; the displayed image is the untouched PNG
    try {
      var cv = document.createElement("canvas"); cv.width = img.naturalWidth; cv.height = img.naturalHeight;
      var ctx = cv.getContext("2d"); ctx.drawImage(img, 0, 0);
      S.gray = { w: cv.width, data: ctx.getImageData(0, 0, cv.width, cv.height).data };
    } catch (e) { S.gray = null; }
    $("img-info").textContent = "native " + img.naturalWidth + "×" + img.naturalHeight + " PNG from ZIP · shown " + S.zoom + "× nearest-neighbour";
  }

  function pxRect(b) { return { x0: (b.cx - b.w / 2) * W, y0: (b.cy - b.h / 2) * H, x1: (b.cx + b.w / 2) * W, y1: (b.cy + b.h / 2) * H }; }
  // edges of a box being edited: round to 1e-4 px so integer (snapped) edges stay exact after 8-decimal YOLO rounding
  function editRect(b) {
    var r = pxRect(b), q = function (v) { return Math.round(v * 10000) / 10000; };
    return { x0: q(r.x0), y0: q(r.y0), x1: q(r.x1), y1: q(r.y1) };
  }
  function fromPx(r) { return { class_id: 0, cx: (r.x0 + r.x1) / 2 / W, cy: (r.y0 + r.y1) / 2 / H, w: (r.x1 - r.x0) / W, h: (r.y1 - r.y0) / H }; }

  function svgEl(tag, attrs, parent) {
    var e = document.createElementNS(SVGNS, tag);
    Object.keys(attrs).forEach(function (k) { e.setAttribute(k, attrs[k]); });
    (parent || $("d-svg")).appendChild(e);
    return e;
  }
  function rectPx(r, stroke, opts) {
    opts = opts || {};
    return svgEl("rect", {
      x: r.x0, y: r.y0, width: Math.max(0, r.x1 - r.x0), height: Math.max(0, r.y1 - r.y0), fill: opts.fill || "none",
      stroke: stroke, "stroke-width": opts.width || 2, "vector-effect": "non-scaling-stroke",
      "stroke-dasharray": opts.dash || "none"
    });
  }
  function label(x, y, text, color) {
    if (!S.overlays.text) return;
    var t = svgEl("text", { x: x, y: y, fill: color, "font-size": 11 / S.zoom * 1.0 + "", "paint-order": "stroke", stroke: "#000", "stroke-width": 2 / S.zoom * 1.5 + "", "font-family": "Menlo, monospace" });
    t.textContent = text;
  }

  function currentGtRect(i) {
    var dr = S.drag;
    if (dr && dr.kind !== "draw" && dr.index === i && dr.rect) return dr.rect;
    return pxRect(S.detail.gt[i].yolo);
  }

  function drawOverlay() {
    var d = S.detail, svg = $("d-svg");
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    if (!d) return;
    if (S.overlays.b) d.b_gt.forEach(function (g) { rectPx(pxRect(g.yolo), "#c084fc", { dash: "1 3", width: 1.5 }); });
    d.predictions.forEach(function (p) {
      var sel = S.sel && S.sel.type === "pred" && S.sel.k === p.k;
      var r = pxRect(p.yolo);
      if (!p.above_threshold) {
        if (S.overlays.below || sel) { rectPx(r, "#9ca3af", { dash: "3 3", width: sel ? 3 : 1.5 }); label(r.x0 + 0.5, r.y1 + 3.2, "k" + p.k + " " + f2(p.confidence), "#9ca3af"); }
        return;
      }
      if (p.matched && !S.overlays.matched && !sel) return;
      if (!p.matched && !S.overlays.unmatched && !sel) return;
      var col = p.matched ? "#22c55e" : "#ef4444";
      rectPx(r, col, { width: sel ? 3.5 : 2, dash: p.matched ? "none" : "5 2" });
      label(r.x0 + 0.5, r.y1 + 3.2, "k" + p.k + " " + f2(p.confidence) + (p.matched ? " IoU " + f2(p.iou) : " FP?") + " ×" + p.fold_support_at_threshold, col);
    });
    if (S.overlays.gt || (S.sel && S.sel.type === "gt")) {
      d.gt.forEach(function (g, i) {
        if (!S.overlays.gt && !(S.sel && S.sel.type === "gt" && S.sel.index === i)) return;
        var r = currentGtRect(i);
        var sel = S.sel && S.sel.type === "gt" && S.sel.index === i;
        if (g.fn) {
          rectPx({ x0: r.x0 - 1.5, y0: r.y0 - 1.5, x1: r.x1 + 1.5, y1: r.y1 + 1.5 }, "#ef4444", { dash: "3 2", width: 2 });
          label(r.x0 - 1.5, r.y0 - 2.5, "FN GT" + i + (g.best_iou_any_prediction !== null ? " best IoU " + f2(g.best_iou_any_prediction) : ""), "#ef4444");
        } else {
          label(r.x0, r.y0 - 1.5, "GT" + i + (g.iou !== null ? " ↔k" + g.matched_k : ""), "#ffb000");
        }
        rectPx(r, "#ffb000", { width: sel ? 3.5 : 2 });
        if (sel && S.mode === "select") drawHandles(r);
      });
    }
    if (S.drag && S.drag.kind === "draw" && S.drag.rect) rectPx(S.drag.rect, "#5aa9ff", { dash: "4 2", width: 2 });
  }

  var HANDLES = ["nw", "n", "ne", "e", "se", "s", "sw", "w"];
  function handlePos(r, h) {
    var mx = (r.x0 + r.x1) / 2, my = (r.y0 + r.y1) / 2;
    return {
      nw: [r.x0, r.y0], n: [mx, r.y0], ne: [r.x1, r.y0], e: [r.x1, my],
      se: [r.x1, r.y1], s: [mx, r.y1], sw: [r.x0, r.y1], w: [r.x0, my]
    }[h];
  }
  function drawHandles(r) {
    var s = 6 / S.zoom;
    HANDLES.forEach(function (h) {
      var p = handlePos(r, h);
      svgEl("rect", { x: p[0] - s / 2, y: p[1] - s / 2, width: s, height: s, fill: "#fff", stroke: "#000", "stroke-width": 1, "vector-effect": "non-scaling-stroke" });
    });
  }

  // ------------------------------------------------------------------ pointer interaction
  function svgPoint(ev) {
    var rc = $("d-svg").getBoundingClientRect();
    return { x: (ev.clientX - rc.left) / rc.width * W, y: (ev.clientY - rc.top) / rc.height * H };
  }
  function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
  function snap(v) { return S.snap ? Math.round(v) : Math.round(v * 100) / 100; }
  function inside(r, p, pad) { pad = pad || 0; return p.x >= r.x0 - pad && p.x <= r.x1 + pad && p.y >= r.y0 - pad && p.y <= r.y1 + pad; }
  function area(r) { return (r.x1 - r.x0) * (r.y1 - r.y0); }

  function hitCandidates(p) {
    var d = S.detail, out = [];
    d.gt.forEach(function (g, i) { var r = pxRect(g.yolo); if ((S.overlays.gt) && inside(r, p, 1.5 / S.zoom * 2)) out.push({ type: "gt", index: i, a: area(r) }); });
    out.sort(function (a, b) { return a.a - b.a; });
    var preds = [];
    d.predictions.forEach(function (pr) {
      var vis = pr.above_threshold ? (pr.matched ? S.overlays.matched : S.overlays.unmatched) : S.overlays.below;
      var r = pxRect(pr.yolo);
      if (vis && inside(r, p, 1.5 / S.zoom * 2)) preds.push({ type: "pred", k: pr.k, a: area(r) });
    });
    preds.sort(function (a, b) { return a.a - b.a; });
    return out.concat(preds);
  }
  function sameSel(a, b) { return a && b && a.type === b.type && (a.type === "gt" ? a.index === b.index : a.k === b.k); }

  function onPointerDown(ev) {
    if (!S.detail || ev.button !== 0) return;
    var p = svgPoint(ev);
    ev.preventDefault();
    $("d-svg").setPointerCapture(ev.pointerId);
    if (S.mode === "draw") {
      var x = clamp(snap(p.x), 0, W), y = clamp(snap(p.y), 0, H);
      S.drag = { kind: "draw", start: { x: x, y: y }, rect: null };
      return;
    }
    // resize handle of selected GT?
    if (S.sel && S.sel.type === "gt") {
      var r = editRect(S.detail.gt[S.sel.index].yolo), tol = 7 / S.zoom;
      for (var i = 0; i < HANDLES.length; i++) {
        var hp = handlePos(r, HANDLES[i]);
        if (Math.abs(p.x - hp[0]) <= tol && Math.abs(p.y - hp[1]) <= tol) {
          S.drag = { kind: "resize", handle: HANDLES[i], index: S.sel.index, orig: r, start: p, rect: r, moved: false };
          return;
        }
      }
      if (inside(r, p)) { S.drag = { kind: "move", index: S.sel.index, orig: r, start: p, rect: r, moved: false }; return; }
    }
    var cands = hitCandidates(p);
    if (!cands.length) { S.sel = null; renderSelection(); drawOverlay(); renderTables(); return; }
    var idx = 0;
    for (var j = 0; j < cands.length; j++) if (sameSel(cands[j], S.sel)) { idx = (j + 1) % cands.length; break; }
    S.sel = cands[idx];
    if (S.sel.type === "gt") {
      var rr = editRect(S.detail.gt[S.sel.index].yolo);
      S.drag = { kind: "move", index: S.sel.index, orig: rr, start: p, rect: rr, moved: false };
    }
    renderSelection(); drawOverlay(); renderTables();
  }

  function onPointerMove(ev) {
    var p = svgPoint(ev);
    updateCursor(p);
    var dr = S.drag;
    if (!dr) return;
    if (dr.kind === "draw") {
      var x = clamp(snap(p.x), 0, W), y = clamp(snap(p.y), 0, H);
      dr.rect = { x0: Math.min(dr.start.x, x), y0: Math.min(dr.start.y, y), x1: Math.max(dr.start.x, x), y1: Math.max(dr.start.y, y) };
    } else if (dr.kind === "move") {
      var o = dr.orig, dx = snap(p.x - dr.start.x), dy = snap(p.y - dr.start.y);
      dx = clamp(dx, -o.x0, W - o.x1); dy = clamp(dy, -o.y0, H - o.y1);
      dr.rect = { x0: o.x0 + dx, y0: o.y0 + dy, x1: o.x1 + dx, y1: o.y1 + dy };
      dr.moved = dr.moved || Math.abs(dx) > 1e-9 || Math.abs(dy) > 1e-9;
    } else if (dr.kind === "resize") {
      var r = { x0: dr.orig.x0, y0: dr.orig.y0, x1: dr.orig.x1, y1: dr.orig.y1 }, h = dr.handle;
      var sx = clamp(snap(p.x), 0, W), sy = clamp(snap(p.y), 0, H), MIN = 1;
      if (h.indexOf("w") >= 0) r.x0 = Math.min(sx, r.x1 - MIN);
      if (h.indexOf("e") >= 0) r.x1 = Math.max(sx, r.x0 + MIN);
      if (h.indexOf("n") >= 0) r.y0 = Math.min(sy, r.y1 - MIN);
      if (h.indexOf("s") >= 0) r.y1 = Math.max(sy, r.y0 + MIN);
      dr.rect = r;
      dr.moved = true;
    }
    drawOverlay();
    if (dr.kind !== "draw" && dr.rect) fillCoordEditor(dr.rect);
  }

  function onPointerUp() {
    var dr = S.drag;
    S.drag = null;
    if (!dr) return;
    if (dr.kind === "draw") {
      if (!dr.rect || dr.rect.x1 - dr.rect.x0 < 1 || dr.rect.y1 - dr.rect.y0 < 1) { drawOverlay(); return; }
      var boxes = currentBoxes(); boxes.push(fromPx(dr.rect));
      saveLabels(boxes, "draw_box", { new_box_px: dr.rect }, { type: "gt", index: boxes.length - 1 });
    } else if (dr.moved && dr.rect) {
      var b = currentBoxes(); b[dr.index] = fromPx(dr.rect);
      saveLabels(b, dr.kind === "move" ? "move_box" : "resize_box", { gt_index: dr.index, from_px: dr.orig, to_px: dr.rect }, { type: "gt", index: dr.index });
    } else { drawOverlay(); }
  }

  function updateCursor(p) {
    var x = Math.floor(p.x), y = Math.floor(p.y), txt = "x " + p.x.toFixed(1) + ", y " + p.y.toFixed(1) + " px";
    if (S.gray && x >= 0 && y >= 0 && x < W && y < H) txt += " · grey " + S.gray.data[(y * S.gray.w + x) * 4];
    $("cursor-info").textContent = txt;
  }

  // ------------------------------------------------------------------ edits (all go to A)
  function currentBoxes() {
    return S.detail.gt.map(function (g) { return { class_id: 0, cx: g.yolo.cx, cy: g.yolo.cy, w: g.yolo.w, h: g.yolo.h }; });
  }

  function afterMutation(r, newSel) {
    updateHistoryButtons(r.history);
    updateRoleA(r.modified_patches);
    if (newSel !== undefined) S.sel = newSel;
    return Promise.all([loadDetail(), loadList()]);
  }

  function saveLabels(boxes, action, detail, newSel) {
    var d = S.detail;
    saveStatus("saving to A…", "busy");
    return post("/api/patch/" + encodeURIComponent(d.patch_id) + "/labels", { boxes: boxes, action: action, base_sha: d.a_sha256, detail: detail })
      .then(function (r) {
        saveStatus(r.changed ? "✓ saved to A (op #" + r.op_id + ", " + action + ")" : "no change", "ok");
        if (r.changed) toast("Saved to A: " + action + " on " + d.patch_id);
        return afterMutation(r, newSel);
      })
      .catch(function (e) { saveStatus("✗ not saved: " + e.message, "err"); toast("Not saved: " + e.message, true); loadDetail(); });
  }

  function adoptSelected() {
    var p = selectedPred();
    if (!p) return;
    if (p.matched && !confirm("Prediction k" + p.k + " already matches GT" + p.matched_gt + " (IoU " + f2(p.iou) + "). Adding it would create a duplicate label. Continue?")) return;
    if (!p.above_threshold && !confirm("This prediction is below the current confidence threshold. Adopt anyway?")) return;
    var d = S.detail;
    saveStatus("saving to A…", "busy");
    post("/api/patch/" + encodeURIComponent(d.patch_id) + "/adopt", { k: p.k, base_sha: d.a_sha256 }).then(function (r) {
      saveStatus(r.changed ? "✓ prediction k" + p.k + " adopted into A (op #" + r.op_id + ")" : "no change", "ok");
      toast("Adopted prediction k" + p.k + " as a contact label in A");
      return afterMutation(r, { type: "gt", index: d.gt.length });
    }).catch(function (e) { saveStatus("✗ not saved: " + e.message, "err"); toast(e.message, true); loadDetail(); });
  }

  function deleteSelected() {
    if (!S.sel || S.sel.type !== "gt") return;
    var i = S.sel.index, b = currentBoxes(), removed = b[i];
    b.splice(i, 1);
    saveLabels(b, "delete_gt", { gt_index: i, deleted_box: removed, was_fn: S.detail.gt[i].fn }, null);
  }

  function restorePatch() {
    var d = S.detail;
    if (!d.modified) { toast("This patch already equals B."); return; }
    if (!confirm("Restore " + d.patch_id + " in A to the frozen backup B version? (undoable)")) return;
    post("/api/patch/" + encodeURIComponent(d.patch_id) + "/restore_b", { base_sha: d.a_sha256 }).then(function (r) {
      saveStatus("✓ patch restored to B (op #" + r.op_id + ")", "ok");
      return afterMutation(r, null);
    }).catch(function (e) { toast(e.message, true); });
  }

  function undoRedo(which) {
    post("/api/" + which, {}).then(function (r) {
      if (!r.changed) { toast("Nothing to " + which); return; }
      var op = r.op;
      toast((which === "undo" ? "Undid" : "Redid") + " op #" + op.op_id + " " + op.action + (op.patch_id ? " on " + op.patch_id : ""));
      updateHistoryButtons(r.history); updateRoleA(r.modified_patches);
      if (op.patch_id && op.patch_id !== S.selectedId) { S.selectedId = op.patch_id; S.sel = null; }
      saveStatus("✓ " + which + " saved to " + (op.kind === "review" ? "review_state.json" : "A"), "ok");
      return Promise.all([loadDetail(true), loadList()]);
    }).catch(function (e) { toast(which + " failed: " + e.message, true); });
  }

  function updateHistoryButtons(h) {
    if (!h) return;
    $("btn-undo").disabled = !h.undo_depth; $("btn-redo").disabled = !h.redo_depth;
    $("btn-undo").title = h.next_undo ? "Undo op #" + h.next_undo.op_id + " " + h.next_undo.action + " " + (h.next_undo.patch_id || "") + " (Ctrl/Cmd+Z)" : "Nothing to undo";
    $("btn-redo").title = h.next_redo ? "Redo op #" + h.next_redo.op_id + " " + h.next_redo.action + " " + (h.next_redo.patch_id || "") + " (Ctrl/Cmd+Shift+Z)" : "Nothing to redo";
  }

  function saveReview(target, k, status, note) {
    var d = S.detail;
    return post("/api/review", { target: target, patch_id: d.patch_id, k: k, status: status, note: note }).then(function (r) {
      var msg = r.changed ? "✓ review status saved (labels in A unchanged)" : "review unchanged";
      saveStatus(msg, "ok");
      if (r.changed) toast(msg);
      updateHistoryButtons(r.history);
      return Promise.all([loadDetail(), loadList()]);
    }).catch(function (e) { toast("Review not saved: " + e.message, true); });
  }

  // ------------------------------------------------------------------ selection panel & coordinate editor
  function selectedPred() {
    if (!S.sel || S.sel.type !== "pred") return null;
    return S.detail.predictions.filter(function (p) { return p.k === S.sel.k; })[0];
  }

  function renderSelection() {
    var info = $("sel-info"), ed = $("coord-editor");
    $("btn-adopt").disabled = !(S.sel && S.sel.type === "pred");
    $("btn-delete").disabled = !(S.sel && S.sel.type === "gt");
    $("coord-error").textContent = "";
    if (!S.sel || !S.detail) { info.textContent = "Nothing selected. Click a box on the image or a table row."; ed.hidden = true; return; }
    if (S.sel.type === "gt") {
      var g = S.detail.gt[S.sel.index];
      if (!g) { S.sel = null; return renderSelection(); }
      info.innerHTML = "GT" + g.index + " in A" + (g.fn ? ' — <b class="badc">FN (unmatched)</b>' : g.matched_k !== null ? " — matched k" + g.matched_k + " IoU " + f3(g.iou) : "") +
        "<br><span class='muted'>Drag inside to move, drag a handle to resize, Delete key or button to delete.</span>";
      ed.hidden = false;
      fillCoordEditor(editRect(g.yolo));
    } else {
      var p = selectedPred();
      if (!p) { S.sel = null; return renderSelection(); }
      var r = pxRect(p.yolo);
      info.innerHTML = "Prediction k" + p.k + " · conf " + f3(p.confidence) + " · " + (p.above_threshold ? (p.matched ? "matched GT" + p.matched_gt + " IoU " + f3(p.iou) : "<b class='badc'>FP candidate</b> (" + esc(p.fp_type) + ")") : "below threshold") +
        "<br>folds " + esc(p.fold_ids.join(", ")) + " · support " + p.fold_support + " (≥conf: " + p.fold_support_at_threshold + ")" +
        "<br><span class='mono'>px x0 " + r.x0.toFixed(2) + " y0 " + r.y0.toFixed(2) + " x1 " + r.x1.toFixed(2) + " y1 " + r.y1.toFixed(2) + "</span>" +
        "<br><span class='mono'>YOLO " + f4(p.yolo.cx) + " " + f4(p.yolo.cy) + " " + f4(p.yolo.w) + " " + f4(p.yolo.h) + "</span>" +
        "<br><span class='muted'>'Adopt prediction as label' adds this box (clipped to the image) to A as class 0 contact.</span>";
      ed.hidden = true;
    }
  }

  var coordLock = false, coordSource = "px";
  function fillCoordEditor(r) {
    coordLock = true; coordSource = "px";
    $("c-x0").value = r.x0.toFixed(2); $("c-y0").value = r.y0.toFixed(2); $("c-x1").value = r.x1.toFixed(2); $("c-y1").value = r.y1.toFixed(2);
    var b = fromPx(r);
    $("c-cx").value = b.cx.toFixed(6); $("c-cy").value = b.cy.toFixed(6); $("c-w").value = b.w.toFixed(6); $("c-h").value = b.h.toFixed(6);
    coordLock = false;
  }
  function coordFromPxInputs() { return { x0: +$("c-x0").value, y0: +$("c-y0").value, x1: +$("c-x1").value, y1: +$("c-y1").value }; }
  function coordFromYoloInputs() { return { class_id: 0, cx: +$("c-cx").value, cy: +$("c-cy").value, w: +$("c-w").value, h: +$("c-h").value }; }
  function validateYolo(b) {
    var tol = 1e-6;
    if (![b.cx, b.cy, b.w, b.h].every(isFinite)) return "all values must be numbers";
    if (b.w <= 0 || b.h <= 0) return "width and height must be positive";
    if (b.cx - b.w / 2 < -tol || b.cy - b.h / 2 < -tol || b.cx + b.w / 2 > 1 + tol || b.cy + b.h / 2 > 1 + tol) return "box exceeds the 150×150 image";
    return "";
  }
  function onCoordInput(ev) {
    if (coordLock) return;
    var isPx = ["c-x0", "c-y0", "c-x1", "c-y1"].indexOf(ev.target.id) >= 0;
    coordSource = isPx ? "px" : "yolo";
    coordLock = true;
    if (isPx) {
      var b = fromPx(coordFromPxInputs());
      $("c-cx").value = b.cx.toFixed(6); $("c-cy").value = b.cy.toFixed(6); $("c-w").value = b.w.toFixed(6); $("c-h").value = b.h.toFixed(6);
    } else {
      var r = pxRect(coordFromYoloInputs());
      $("c-x0").value = r.x0.toFixed(2); $("c-y0").value = r.y0.toFixed(2); $("c-x1").value = r.x1.toFixed(2); $("c-y1").value = r.y1.toFixed(2);
    }
    coordLock = false;
    $("coord-error").textContent = validateYolo(coordValue());
  }
  // full-precision value from whichever representation the user typed last
  function coordValue() { return coordSource === "px" ? fromPx(coordFromPxInputs()) : coordFromYoloInputs(); }
  function applyCoords() {
    if (!S.sel || S.sel.type !== "gt") return;
    var b = coordValue(), err = validateYolo(b);
    if (err) { $("coord-error").textContent = err; return; }
    var boxes = currentBoxes(), i = S.sel.index;
    var before = boxes[i];
    boxes[i] = b;
    saveLabels(boxes, "edit_coords", { gt_index: i, before: before, after: b }, { type: "gt", index: i });
  }

  // ------------------------------------------------------------------ tables
  function renderTables() {
    var d = S.detail, m = S.meta;
    var gt = ["<tr><th>#</th><th>status</th><th>YOLO cx, cy, w, h</th><th>pixels x0, y0 – x1, y1</th><th>match</th></tr>"];
    d.gt.forEach(function (g) {
      var sel = S.sel && S.sel.type === "gt" && S.sel.index === g.index;
      var st = !d.meta.scored ? "unscored" : g.fn ? '<b class="badc">FN</b>' : "TP";
      gt.push('<tr class="clickable' + (sel ? " sel" : "") + '" data-gt="' + g.index + '"><td>GT' + g.index + "</td><td>" + st + '</td><td class="mono">' +
        [g.yolo.cx, g.yolo.cy, g.yolo.w, g.yolo.h].map(function (v) { return v.toFixed(6); }).join(", ") + '</td><td class="mono">' +
        g.px.x0.toFixed(2) + ", " + g.px.y0.toFixed(2) + " – " + g.px.x1.toFixed(2) + ", " + g.px.y1.toFixed(2) + "</td><td>" +
        (g.matched_k !== null ? "k" + g.matched_k + " IoU " + f3(g.iou) : g.best_iou_any_prediction !== null ? "best IoU " + f3(g.best_iou_any_prediction) : "–") + "</td></tr>");
    });
    if (!d.gt.length) gt.push('<tr><td colspan="5" class="muted">No GT boxes in A (empty label file).</td></tr>');
    $("t-gt").innerHTML = gt.join("");

    $("fp-note-detail").innerHTML = esc(m.fp_note_en) + "<br>" + esc(m.fp_note_zh);
    var pr = ["<tr><th>k</th><th>conf</th><th>status @ thresholds</th><th>IoU</th><th>YOLO / pixels</th><th>folds (support)</th><th>review status (saved separately)</th><th>note</th></tr>"];
    d.predictions.forEach(function (p) {
      var sel = S.sel && S.sel.type === "pred" && S.sel.k === p.k;
      var st = !p.above_threshold ? '<span class="muted">below conf</span>' : p.matched ? '<span class="okc">TP ↔ GT' + p.matched_gt + "</span>" :
        '<b class="badc">FP candidate</b><br><span class="small">' + esc(p.fp_type) + "</span>";
      var opts = m.pred_statuses.map(function (s) { return '<option value="' + s + '"' + (p.review.status === s ? " selected" : "") + ">" + esc(m.pred_status_labels[s]) + "</option>"; }).join("");
      var hint = p.review.status === "true_contact_missed_label" && !p.matched ? '<div class="hint-adopt">Label not changed. Use “Adopt prediction as label” to add it to A.</div>' : "";
      pr.push('<tr class="clickable' + (sel ? " sel" : "") + '" data-k="' + p.k + '"><td>k' + p.k + '</td><td class="num">' + f3(p.confidence) + "</td><td>" + st + '</td><td class="num">' + f3(p.iou) +
        '</td><td class="mono small">' + [p.yolo.cx, p.yolo.cy, p.yolo.w, p.yolo.h].map(function (v) { return v.toFixed(4); }).join(" ") + "<br>" +
        p.px.x0.toFixed(1) + "," + p.px.y0.toFixed(1) + " – " + p.px.x1.toFixed(1) + "," + p.px.y1.toFixed(1) + "</td><td class='small'>" + esc(p.fold_ids.join(", ")) +
        " (" + p.fold_support + "; ≥conf " + p.fold_support_at_threshold + ")<br><span class='muted'>" + p.source_prediction_count + " raw fold box(es)</span></td>" +
        '<td><select data-review-k="' + p.k + '">' + opts + "</select>" + hint + "</td>" +
        '<td><input type="text" data-note-k="' + p.k + '" value="' + esc(p.review.note || "") + '" maxlength="2000"><button type="button" data-savenote-k="' + p.k + '">save</button></td></tr>');
    });
    if (!d.predictions.length) pr.push('<tr><td colspan="8" class="muted">' + (d.meta.scored ? "No saved OOF predictions (≥ 0.05) for this patch." : "Unscored patch: no OOF prediction exists.") + "</td></tr>");
    $("t-pred").innerHTML = pr.join("");

    $("patch-status").value = d.patch_review.status || "unreviewed";
    $("patch-note").value = d.patch_review.note || "";

    var h = ["<tr><th>op</th><th>time (UTC)</th><th>action</th><th>detail</th></tr>"];
    d.history.forEach(function (o) {
      h.push("<tr><td>#" + o.op_id + "</td><td class='mono small'>" + esc(o.time_utc) + "</td><td>" + esc(o.action) + (o.source && o.source !== "ui" ? " <span class='muted'>[" + esc(o.source) + "]</span>" : "") + "</td><td class='small'>" +
        (o.kind === "labels" ? "boxes " + o.boxes_before + " → " + o.boxes_after : o.kind === "review" ? esc(o.key) + ": " + esc(o.status_before) + " → " + esc(o.status_after) : o.patches + " patches") + "</td></tr>");
    });
    if (!d.history.length) h.push('<tr><td colspan="4" class="muted">No edits in the undo history for this patch.</td></tr>');
    $("t-hist").innerHTML = h.join("");
  }

  function renderMeta() {
    var d = S.detail, m = d.meta, p = d.paths;
    var rows = [
      ["patch ID", d.patch_id], ["spatial_chunk", m.spatial_chunk], ["original_group", m.original_group],
      ["final_split (metadata, not CV fold)", m.final_split],
      ["bounds (m)", m.bounds_m ? "x " + m.bounds_m.x0 + " – " + m.bounds_m.x1 + ", y " + m.bounds_m.y0 + " – " + m.bounds_m.y1 : "–"],
      ["OOF scored", m.scored ? "yes" : "NO — unscored / 未评估: " + (m.unscored_reason || "")],
      ["GT / pred (≥conf) / TP / FP / FN", d.counts.gt_count + " / " + d.counts.pred_count + " / " + d.counts.tp + " / " + d.counts.fp + " / " + d.counts.fn],
      ["image (ZIP member)", p.zip_image_member], ["source ZIP", p.source_zip],
      ["original GT (ZIP member)", p.zip_gt_member], ["original GT (backup B)", p.b_label_path],
      ["current label (working A)", p.a_label_path], ["A sha256", d.a_sha256], ["B sha256", d.b_sha256 + (d.modified ? "  (A differs)" : "  (A identical)")],
      ["predictions source", p.prediction_sources.join(" ; ")], ["merged predicted_labels (ZIP)", p.zip_pred_label_member],
      ["A file content", d.a_text ? d.a_text.replace(/\n/g, " ⏎ ") : "(empty file)"]
    ];
    $("t-meta").innerHTML = rows.map(function (r) { return "<tr><td>" + esc(r[0]) + "</td><td>" + esc(r[1]) + "</td></tr>"; }).join("");
    var f = ["<tr><th>fold</th><th>val chunks</th><th colspan='5'>ZIP per-fold evaluation (original labels, conf 0.20 / IoU 0.50): GT · pred · TP · FP · FN</th></tr>"];
    var folds = {};
    S.meta.folds.forEach(function (x) { folds[x.fold_id] = x; });
    (m.fold_evaluation || []).forEach(function (e) {
      f.push("<tr><td class='mono'>" + esc(e.fold_id) + "</td><td>" + esc((folds[e.fold_id] || {}).validation_chunks) + "</td><td>" + e.gt_count + "</td><td>" + e.prediction_count + "</td><td>" + e.tp_count + "</td><td>" + e.fp_count + "</td><td>" + e.fn_count + "</td></tr>");
    });
    if (m.fold_agreement) {
      var a = m.fold_agreement;
      f.push("<tr><td colspan='7' class='small'>fold_agreement.csv: evaluated by " + a.folds_evaluated + " fold(s); folds with FP " + a.folds_with_fp + ", FN " + a.folds_with_fn + ", TP " + a.folds_with_tp +
        (a.max_unmatched_confidence ? "; max unmatched conf " + f3(+a.max_unmatched_confidence) : "") + (a.candidate_type ? "; " + esc(a.candidate_type) : "") + "</td></tr>");
    }
    if (!m.scored) f.push("<tr><td colspan='7' class='muted'>Not held out by any fold (15 m spatial guard) — no OOF evaluation.</td></tr>");
    $("t-folds").innerHTML = f.join("");
  }

  // ------------------------------------------------------------------ dialogs
  function openExport() {
    $("exp-result").textContent = "";
    refreshExports();
    $("dlg-export").showModal();
  }
  function refreshExports() {
    api("/api/exports").then(function (list) {
      var h = ["<tr><th>name</th><th>created (UTC)</th><th>purpose</th><th>edited patches</th><th>boxes</th><th>verified</th><th>note</th></tr>"];
      list.forEach(function (e) { h.push("<tr><td class='mono'>" + esc(e.name) + "</td><td class='mono small'>" + esc(e.generated_at_utc) + "</td><td>" + esc(e.purpose) + "</td><td>" + esc(e.patches_modified_vs_B) + "</td><td>" + esc(e.box_count) + "</td><td>" + (e.verified ? "✓" : "✗") + "</td><td>" + esc(e.note) + "</td></tr>"); });
      if (!list.length) h.push("<tr><td colspan='7' class='muted'>No exports yet.</td></tr>");
      $("exp-list").innerHTML = h.join("");
    });
  }
  function doExport() {
    $("exp-go").disabled = true;
    $("exp-result").textContent = "exporting…";
    post("/api/export", { note: $("exp-note").value, purpose: $("exp-purpose").value }).then(function (r) {
      $("exp-result").textContent = "Created " + r.name + "\nfolder: " + r.dir + "\nzip: " + r.zip + "\nmanifest: " + r.manifest + "\nchanges: " + r.changes_csv +
        "\nlabel files: " + r.counts.label_file_count + " (empty " + r.counts.empty_label_files + ", non-empty " + r.counts.nonempty_label_files + "), boxes " + r.counts.box_count +
        "\npatches modified vs B: " + r.patches_modified_vs_B + "\nself-verification: " + (r.verification.ok ? "OK" : "FAILED " + JSON.stringify(r.verification.problems));
      refreshExports();
    }).catch(function (e) { $("exp-result").textContent = "Export failed: " + e.message; }).then(function () { $("exp-go").disabled = false; });
  }
  function openChecks() {
    $("integrity-result").textContent = "";
    Promise.all([api("/api/data_check_report"), api("/api/baseline_comparison")]).then(function (res) {
      var rep = res[0], cmp = res[1], h = [];
      h.push("<h4>ZIP data checks (" + (rep.all_passed ? "<span class='okc'>all passed</span>" : "<span class='badc'>some failed</span>") + ")</h4><table class='tbl'>");
      rep.checks.forEach(function (c) { h.push("<tr><td>" + (c.ok ? "<span class='okc'>✓</span>" : "<span class='badc'>✗</span>") + "</td><td>" + esc(c.check) + "</td><td class='mono small'>" + esc(typeof c.detail === "string" ? c.detail : JSON.stringify(c.detail)) + "</td></tr>"); });
      h.push("</table><p>" + esc(rep.notes.join(" ")) + "</p>");
      h.push("<h4>Baseline (conf 0.20 / IoU 0.50, B labels) vs ZIP: " + (cmp.all_equal ? "<span class='okc'>identical</span>" : "<span class='badc'>differences</span>") + "</h4><table class='tbl'><tr><th></th><th>platform patches / boxes</th><th>ZIP manifest patches / boxes</th></tr>");
      Object.keys(cmp.totals).forEach(function (k) { var t = cmp.totals[k]; h.push("<tr><td>" + k + "</td><td>" + t.ours.patch_count + " / " + t.ours.box_count + "</td><td>" + t.zip.patch_count + " / " + t.zip.box_count + "</td></tr>"); });
      h.push("</table><table class='tbl'>");
      ["fp_candidates_csv", "fn_csv", "tp_csv", "tn_csv", "unscored_csv"].forEach(function (k) {
        var v = cmp[k];
        h.push("<tr><td>" + k + "</td><td>rows " + v.rows + "</td><td>only platform: " + v.only_platform.length + "</td><td>only ZIP: " + v.only_zip.length + "</td></tr>");
      });
      h.push("<tr><td>review_data.json per-patch baseline</td><td colspan='3'>differences: " + cmp.review_data_baseline_per_patch_differences.length + "</td></tr></table>");
      $("checks-body").innerHTML = h.join("");
    });
    $("dlg-checks").showModal();
  }
  function runIntegrity() {
    $("integrity-result").textContent = "hashing…";
    api("/api/integrity").then(function (r) {
      $("integrity-result").textContent = "ZIP sha256 " + r.zip_now.sha256 + "\nZIP mtime " + r.zip_now.mtime_utc + "\nZIP unchanged since initialisation: " + r.zip_unchanged_since_init +
        "\nB files checked: " + r.b_files_checked + ", mismatching backup_manifest: " + r.b_files_mismatching_backup_manifest.length +
        "\nB manifest unchanged since initialisation: " + r.b_manifest_unchanged_since_init;
    }).catch(function (e) { $("integrity-result").textContent = e.message; });
  }

  // ------------------------------------------------------------------ events
  function bindEvents() {
    $("conf-range").addEventListener("input", function () { setThresholds(+this.value, S.iou); });
    $("iou-range").addEventListener("input", function () { setThresholds(S.conf, +this.value); });
    $("conf-num").addEventListener("change", function () { setThresholds(+this.value, S.iou); });
    $("iou-num").addEventListener("change", function () { setThresholds(S.conf, +this.value); });
    $("thr-reset").addEventListener("click", function () { setThresholds(0.2, 0.5); });

    var filterIds = ["f-chunk", "f-group", "f-split", "f-fptype", "f-fpreview", "f-patchreview", "f-modified", "f-sort", "f-order", "f-perpage", "f-minconf", "f-maxtpiou", "f-minfp"];
    filterIds.forEach(function (id) { $(id).addEventListener("change", function () { readFilters(); S.page = 1; loadList(); persistUi(); }); });
    var qTimer = null;
    $("f-q").addEventListener("input", function () { clearTimeout(qTimer); qTimer = setTimeout(function () { readFilters(); S.page = 1; loadList(); }, 250); });
    $("f-q").addEventListener("keydown", function (e) {
      if (e.key === "Enter") { e.preventDefault(); readFilters(); S.page = 1; loadList().then(function () { var ids = pageIds(); if (ids.length) selectPatch(ids[0]); }); }
    });
    $("filters").addEventListener("submit", function (e) { e.preventDefault(); });
    $("btn-random").addEventListener("click", function () { $("f-sort").value = "random"; S.seed = String(Date.now()); readFilters(); S.page = 1; loadList(); });
    $("btn-clear").addEventListener("click", function () { $("filters").reset(); $("f-perpage").value = String(S.perPage); readFilters(); S.page = 1; loadList(); });

    Object.keys(S.overlays).forEach(function (k) {
      var el = $("ov-" + k);
      if (el) el.addEventListener("change", function () { S.overlays[k] = this.checked; drawOverlay(); persistUi(); });
    });
    $("snap").addEventListener("change", function () { S.snap = this.checked; });
    $("zoom").addEventListener("change", function () { S.zoom = +this.value; if (S.detail) { renderDetail(); onImageLoad($("d-img")); } persistUi(); });
    $("mode-select").addEventListener("click", function () { setMode("select"); });
    $("mode-draw").addEventListener("click", function () { setMode("draw"); });

    var svg = $("d-svg");
    svg.addEventListener("pointerdown", onPointerDown);
    svg.addEventListener("pointermove", onPointerMove);
    svg.addEventListener("pointerup", onPointerUp);
    svg.addEventListener("pointercancel", function () { S.drag = null; drawOverlay(); });

    ["c-x0", "c-y0", "c-x1", "c-y1", "c-cx", "c-cy", "c-w", "c-h"].forEach(function (id) {
      $(id).addEventListener("input", onCoordInput);
      $(id).addEventListener("keydown", function (e) { if (e.key === "Enter" || e.keyCode === 13) { e.preventDefault(); applyCoords(); } });
    });
    $("btn-apply-coords").addEventListener("click", applyCoords);
    $("btn-adopt").addEventListener("click", adoptSelected);
    $("btn-delete").addEventListener("click", deleteSelected);
    $("btn-restore-patch").addEventListener("click", restorePatch);
    $("btn-prev").addEventListener("click", function () { step(-1); });
    $("btn-next").addEventListener("click", function () { step(1); });
    $("btn-undo").addEventListener("click", function () { undoRedo("undo"); });
    $("btn-redo").addEventListener("click", function () { undoRedo("redo"); });
    $("btn-patch-review").addEventListener("click", function () { saveReview("patch", null, $("patch-status").value, $("patch-note").value); });

    $("t-gt").addEventListener("click", function (e) {
      var tr = e.target.closest("tr[data-gt]"); if (!tr) return;
      S.sel = { type: "gt", index: +tr.dataset.gt }; renderSelection(); drawOverlay(); renderTables();
    });
    $("t-pred").addEventListener("click", function (e) {
      if (inField(e.target) || e.target.tagName === "BUTTON") return;
      var tr = e.target.closest("tr[data-k]"); if (!tr) return;
      S.sel = { type: "pred", k: +tr.dataset.k }; renderSelection(); drawOverlay(); renderTables();
    });
    $("t-pred").addEventListener("change", function (e) {
      var k = e.target.dataset.reviewK;
      if (k === undefined) return;
      var note = document.querySelector('[data-note-k="' + k + '"]').value;
      saveReview("prediction", +k, e.target.value, note);
    });
    $("t-pred").addEventListener("click", function (e) {
      var k = e.target.dataset.savenoteK;
      if (k === undefined) return;
      var status = document.querySelector('[data-review-k="' + k + '"]').value;
      saveReview("prediction", +k, status, document.querySelector('[data-note-k="' + k + '"]').value);
    });

    $("btn-export").addEventListener("click", openExport);
    $("exp-go").addEventListener("click", doExport);
    $("exp-close").addEventListener("click", function () { $("dlg-export").close(); });
    $("btn-checks").addEventListener("click", openChecks);
    $("checks-close").addEventListener("click", function () { $("dlg-checks").close(); });
    $("checks-integrity").addEventListener("click", runIntegrity);
    $("btn-restore-all").addEventListener("click", function () { $("restore-all-text").value = ""; $("dlg-restore-all").showModal(); });
    $("restore-all-close").addEventListener("click", function () { $("dlg-restore-all").close(); });
    $("restore-all-go").addEventListener("click", function () {
      post("/api/restore_all_b", { confirm: $("restore-all-text").value }).then(function (r) {
        $("dlg-restore-all").close();
        toast(r.changed ? "Restored " + r.patches + " patches in A to B (op #" + r.op_id + ", undoable)" : "A already equals B");
        updateHistoryButtons(r.history); updateRoleA(r.modified_patches);
        return Promise.all([loadList(), loadDetail()]);
      }).catch(function (e) { toast(e.message, true); });
    });

    document.addEventListener("keydown", function (e) {
      if (inField(document.activeElement) || inField(e.target)) return; // never steal keys from form controls
      if (document.querySelector("dialog[open]")) return;
      var mod = e.ctrlKey || e.metaKey;
      if (mod && (e.key === "z" || e.key === "Z")) { e.preventDefault(); undoRedo(e.shiftKey ? "redo" : "undo"); return; }
      if (mod && (e.key === "y" || e.key === "Y")) { e.preventDefault(); undoRedo("redo"); return; }
      if (mod || e.altKey) return;
      if (e.key === "ArrowRight") { e.preventDefault(); step(1); }
      else if (e.key === "ArrowLeft") { e.preventDefault(); step(-1); }
      else if (e.key === "Delete" || e.key === "Backspace") { if (S.sel && S.sel.type === "gt") { e.preventDefault(); deleteSelected(); } }
      else if (e.key === "Escape") { S.sel = null; S.drag = null; renderSelection(); drawOverlay(); renderTables(); }
      else if (e.key === "b" || e.key === "B") setMode("draw");
      else if (e.key === "v" || e.key === "V") setMode("select");
    });
  }

  function setMode(m) {
    S.mode = m;
    $("mode-select").classList.toggle("on", m === "select");
    $("mode-draw").classList.toggle("on", m === "draw");
    $("stage").classList.toggle("draw", m === "draw");
    drawOverlay();
  }

  init();
})();
