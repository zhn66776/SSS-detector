const state = {
  items: [],
  decisions: {},
  index: 0,
  mode: "raw",
  busy: false,
};

const $ = (id) => document.getElementById(id);

function currentItem() {
  return state.items[state.index] || null;
}

function setMessage(message, error = false) {
  const element = $("server-message");
  element.textContent = message;
  element.classList.toggle("error", error);
}

function setMode(mode) {
  state.mode = mode;
  $("raw-button").classList.toggle("active", mode === "raw");
  $("preview-button").classList.toggle("active", mode === "preview");
  $("raw-button").setAttribute("aria-pressed", String(mode === "raw"));
  $("preview-button").setAttribute("aria-pressed", String(mode === "preview"));
  renderImage();
}

function renderImage() {
  const item = currentItem();
  if (!item) return;
  const image = $("patch-image");
  const loading = $("image-loading");
  loading.classList.remove("hidden");
  image.onload = () => loading.classList.add("hidden");
  image.onerror = () => {
    loading.textContent = "Image could not be loaded";
    loading.classList.remove("hidden");
  };
  image.src = state.mode === "raw" ? item.raw_url : item.preview_url;
  image.alt = `${item.id} ${state.mode === "raw" ? "raw grayscale patch" : "marked preview"}`;
}

function countDecisions() {
  let keep = 0;
  let drop = 0;
  for (const status of Object.values(state.decisions)) {
    if (status === "keep") keep += 1;
    if (status === "drop") drop += 1;
  }
  return { keep, drop, undecided: state.items.length - keep - drop };
}

function renderProgress() {
  const counts = countDecisions();
  $("position-label").textContent = state.items.length ? `Patch ${state.index + 1} / ${state.items.length}` : "No patches to display";
  $("count-label").textContent = `Kept ${counts.keep} · Dropped ${counts.drop} · Not reviewed ${counts.undecided}`;
  $("progress-bar").style.width = `${state.items.length ? ((counts.keep + counts.drop) / state.items.length) * 100 : 0}%`;
}

function renderDetails() {
  const item = currentItem();
  if (!item) return;
  const row = item.row;
  const status = state.decisions[item.uid] || "undecided";
  $("detail-uid").textContent = row.uid || "—";
  $("detail-id").textContent = row.id || "—";
  $("detail-area").textContent = row.area || "—";
  $("detail-description").textContent = row.description || "—";
  $("detail-coordinate").textContent = `${row.e}, ${row.n}`;
  $("detail-size").textContent = `${row.length_m || ""} × ${row.width_m || ""} × ${row.height_m || ""} m`;
  $("detail-mean").textContent = row.grey_mean || "—";
  $("detail-variance").textContent = row.grey_variance || "—";
  $("detail-source").textContent = `${row.source_file}:${row.source_row}`;

  const checkbox = $("keep-checkbox");
  checkbox.checked = status === "keep";
  const statusElement = $("decision-status");
  statusElement.className = "decision-status";
  statusElement.textContent = status === "keep" ? "Kept" : status === "drop" ? "Dropped" : "Not reviewed";
  if (status !== "undecided") statusElement.classList.add(status);
}

function render() {
  renderProgress();
  renderDetails();
  renderImage();
}

async function request(path, options = {}) {
  const response = await fetch(path, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `Request failed: ${response.status}`);
  return payload;
}

async function commitCurrent() {
  const item = currentItem();
  if (!item) return;
  const status = $("keep-checkbox").checked ? "keep" : "drop";
  state.busy = true;
  $("next-button").disabled = true;
  $("previous-button").disabled = true;
  try {
    const payload = await request("/api/decision", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ uid: item.uid, status }),
    });
    state.decisions[item.uid] = status;
    setMessage(status === "keep" ? "Saved: keep." : "Unchecked; saved: drop.", false);
    return payload;
  } finally {
    state.busy = false;
    $("next-button").disabled = false;
    $("previous-button").disabled = false;
  }
}

function firstUndecided() {
  return state.items.findIndex((item) => !state.decisions[item.uid]);
}

async function nextItem() {
  if (state.busy || !currentItem()) return;
  try {
    await commitCurrent();
    if (state.index < state.items.length - 1) {
      state.index += 1;
      render();
    } else {
      const next = firstUndecided();
      if (next >= 0) {
        state.index = next;
        render();
      } else {
        renderProgress();
        setMessage("All patches have been reviewed. You can export the results.", false);
      }
    }
  } catch (error) {
    setMessage(error.message, true);
  }
}

function previousItem() {
  if (state.busy || !currentItem()) return;
  if (state.index > 0) {
    state.index -= 1;
    render();
    setMessage("Previous does not save an unsaved selection. Use Save and next to record a decision.", false);
  }
}

async function exportCurrent() {
  try {
    const payload = await request("/api/export", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
    const counts = payload.counts;
    setMessage(`Exported: ${counts.keep} kept, ${counts.drop} dropped, ${counts.undecided} not reviewed.`, false);
  } catch (error) {
    setMessage(error.message, true);
  }
}

async function init() {
  try {
    const [itemsPayload, statePayload] = await Promise.all([request("/api/items"), request("/api/state")]);
    state.items = itemsPayload.items.map((item) => ({ ...item, row: {} }));
    const rowsByUid = new Map(state.items.map((item) => [item.uid, item]));
    const contacts = await request("/api/contact-rows");
    for (const row of contacts.rows) {
      const item = rowsByUid.get(row.uid);
      if (item) item.row = row;
    }
    state.decisions = statePayload.decisions || {};
    const first = firstUndecided();
    state.index = first >= 0 ? first : 0;
    render();
    setMessage("Save and next records Drop when the keep option is unchecked.", false);
  } catch (error) {
    setMessage(error.message, true);
    $("image-loading").textContent = "Data could not be loaded";
  }
}

$("raw-button").addEventListener("click", () => setMode("raw"));
$("preview-button").addEventListener("click", () => setMode("preview"));
$("next-button").addEventListener("click", nextItem);
$("previous-button").addEventListener("click", previousItem);
$("export-button").addEventListener("click", exportCurrent);
document.addEventListener("keydown", (event) => {
  if (event.target instanceof HTMLInputElement) {
    if (event.key === " " || event.key === "Spacebar") return;
  }
  if (event.key === "ArrowRight") {
    event.preventDefault();
    nextItem();
  } else if (event.key === "ArrowLeft") {
    event.preventDefault();
    previousItem();
  } else if (event.key === " ") {
    event.preventDefault();
    $("keep-checkbox").click();
  }
});

init();
