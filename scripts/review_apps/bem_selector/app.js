const state = {
  page: 0,
  pageCount: 1,
  pageSize: 1,
  totalCount: 0,
  startIndex: 0,
  endIndex: 0,
  savedCount: 0,
  selected: new Set(),
  samples: [],
  layout: null,
  loading: false,
  showContext: true,
};

const figureCache = new Map();
const detailCache = new Map();

const gridEl = document.querySelector("#grid");
const pageIndicatorEl = document.querySelector("#page-indicator");
const rangeIndicatorEl = document.querySelector("#range-indicator");
const selectedIndicatorEl = document.querySelector("#selected-indicator");
const savedIndicatorEl = document.querySelector("#saved-indicator");
const messageEl = document.querySelector("#message");
const csvPathEl = document.querySelector("#csv-path");
const progressFillEl = document.querySelector("#progress-fill");
const contextToggleEl = document.querySelector("#context-toggle");
const pageJumpFormEl = document.querySelector("#page-jump-form");
const pageJumpInputEl = document.querySelector("#page-jump-input");

let resizeTimer = null;
const TILE_WIDTH = 380;
const TILE_HEIGHT = 516;
const FIGURE_CONFIG = {
  responsive: true,
  scrollZoom: true,
  displaylogo: false,
  displayModeBar: false,
  plotGlPixelRatio: 1,
  modeBarButtonsToRemove: ["select2d", "lasso2d", "toImage"],
};

function getLayoutConstants() {
  return { tileWidth: TILE_WIDTH, tileHeight: TILE_HEIGHT, gap: window.innerWidth <= 720 ? 14 : 18 };
}

function computeLayout() {
  const width = gridEl.clientWidth;
  const height = gridEl.clientHeight;
  if (!width || !height) {
    return null;
  }

  const { tileWidth, tileHeight, gap } = getLayoutConstants();
  const columns = Math.max(1, Math.floor((width + gap) / (tileWidth + gap)));
  const rows = Math.max(1, Math.floor((height + gap) / (tileHeight + gap)));

  return {
    columns,
    rows,
    pageSize: columns * rows,
    tileWidth,
    tileHeight,
    gap,
  };
}

function setMessage(text, tone = "neutral") {
  messageEl.textContent = text;
  messageEl.dataset.tone = tone;
}

function applyLayout(layout) {
  if (!layout) {
    return;
  }

  state.layout = layout;
  gridEl.style.setProperty("--grid-columns", String(layout.columns));
  gridEl.style.setProperty("--grid-gap", `${layout.gap}px`);
  gridEl.style.setProperty("--tile-width", `${layout.tileWidth}px`);
  gridEl.style.setProperty("--tile-height", `${layout.tileHeight}px`);
}

function updateProgressBar() {
  const ratio = state.totalCount ? state.endIndex / state.totalCount : 0;
  progressFillEl.style.width = `${Math.max(0, Math.min(ratio, 1)) * 100}%`;
}

function syncPageJumpInput(force = false) {
  if (!pageJumpInputEl) {
    return;
  }

  const maxPage = Math.max(state.pageCount, 1);
  pageJumpInputEl.min = "1";
  pageJumpInputEl.max = String(maxPage);
  pageJumpInputEl.disabled = state.loading || maxPage <= 1;

  if (force || document.activeElement !== pageJumpInputEl) {
    pageJumpInputEl.value = String(Math.min(state.page + 1, maxPage));
  }
}

function updateToolbar() {
  pageIndicatorEl.textContent = `${state.page + 1} / ${state.pageCount}`;
  rangeIndicatorEl.textContent = state.totalCount
    ? `${state.startIndex + 1}-${state.endIndex} / ${state.totalCount}`
    : "0-0";
  selectedIndicatorEl.textContent = String(state.selected.size);
  savedIndicatorEl.textContent = String(state.savedCount);
  if (contextToggleEl) {
    contextToggleEl.textContent = state.showContext ? "Hide Context" : "Show Context";
    contextToggleEl.classList.toggle("is-active", state.showContext);
  }
  document.body.classList.toggle("is-loading", state.loading);
  updateProgressBar();
  syncPageJumpInput();
}

function formatStat(value, fallback = "n/a") {
  return value === null || value === undefined || value === "" ? fallback : String(value);
}

function escapeHTML(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function buildCard(sample) {
  const card = document.createElement("article");
  card.className = "sample-card";
  card.dataset.sampleId = sample.sample_id;

  card.innerHTML = `
    <div class="sample-card__plot-shell">
      <div class="sample-card__plot" data-plot-for="${escapeHTML(sample.sample_id)}">
        <div class="plot-placeholder">Loading model…</div>
      </div>
    </div>
    <div class="sample-card__meta">
      <div class="sample-card__headline">
        <span class="sample-card__name">${escapeHTML(sample.sample_id)}</span>
      </div>
      <div class="meta-grid">
        <span class="meta-pill">floors ${formatStat(sample.n_floors)}</span>
        <span class="meta-pill">zones ${formatStat(sample.n_zones)}</span>
        <span class="meta-pill">windows ${formatStat(sample.n_windows)}</span>
        <span class="meta-pill">type ${formatStat(sample.building_type_id)}</span>
      </div>
      <div class="sample-card__detail">${escapeHTML(sample.hb_program_identifier || "program unknown")}</div>
      <div class="sample-card__detail sample-card__detail--muted">${escapeHTML(sample.hb_construction_set_id || "construction set unknown")}</div>
    </div>
  `;

  syncCardState(card, sample);
  return card;
}

function syncCardState(card, sample) {
  const isSaved = Boolean(sample.saved);
  const isSelected = state.selected.has(sample.sample_id);
  card.classList.toggle("is-saved", isSaved);
  card.classList.toggle("is-selected", !isSaved && isSelected);
}

function syncAllCardStates() {
  for (const sample of state.samples) {
    const card = gridEl.querySelector(`.sample-card[data-sample-id="${CSS.escape(sample.sample_id)}"]`);
    if (card) {
      syncCardState(card, sample);
    }
  }
  updateToolbar();
}

function renderGrid() {
  const oldPlots = gridEl.querySelectorAll(".sample-card__plot");
  for (const plotEl of oldPlots) {
    if (plotEl.data || plotEl._fullData) {
      Plotly.purge(plotEl);
    }
  }
  gridEl.replaceChildren(...state.samples.map((sample) => buildCard(sample)));
  updateToolbar();
  loadVisibleFigures();
}

async function requestJSON(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "Request failed.");
  }
  return payload;
}

function updateCardDetail(sampleId, detail) {
  const sample = state.samples.find((item) => item.sample_id === sampleId);
  if (!sample || !detail) {
    return;
  }
  sample.n_floors = detail.n_floors ?? sample.n_floors;
  sample.n_zones = detail.n_zones ?? sample.n_zones;
  sample.n_windows = detail.n_windows ?? sample.n_windows;
  sample.hb_program_identifier = detail.hb_program_identifier ?? sample.hb_program_identifier;
  sample.hb_construction_set_id = detail.hb_construction_set_id ?? sample.hb_construction_set_id;
  sample.building_type_id = detail.building_type_id ?? sample.building_type_id;

  const card = gridEl.querySelector(`.sample-card[data-sample-id="${CSS.escape(sampleId)}"]`);
  if (!card) {
    return;
  }

  const pills = card.querySelectorAll(".meta-pill");
  if (pills.length >= 4) {
    pills[0].textContent = `floors ${formatStat(sample.n_floors)}`;
    pills[1].textContent = `zones ${formatStat(sample.n_zones)}`;
    pills[2].textContent = `windows ${formatStat(sample.n_windows)}`;
    pills[3].textContent = `type ${formatStat(sample.building_type_id)}`;
  }

  const details = card.querySelectorAll(".sample-card__detail");
  if (details.length >= 2) {
    details[0].textContent = sample.hb_program_identifier || "program unknown";
    details[1].textContent = sample.hb_construction_set_id || "construction set unknown";
  }
}

function figurePlaceholder(sampleId, text) {
  const plotEl = gridEl.querySelector(`[data-plot-for="${CSS.escape(sampleId)}"]`);
  if (!plotEl) {
    return;
  }
  plotEl.innerHTML = `<div class="plot-placeholder">${escapeHTML(text)}</div>`;
}

function isContextTrace(trace) {
  const name = String(trace?.name || "").toLowerCase();
  return name === "context_neighbor" || name === "context_neighbor_wire";
}

async function applyContextVisibilityToPlot(plotEl) {
  if (!plotEl || !plotEl.data) {
    return;
  }
  const indices = [];
  const values = [];
  plotEl.data.forEach((trace, idx) => {
    if (isContextTrace(trace)) {
      indices.push(idx);
      values.push(state.showContext);
    }
  });
  if (!indices.length) {
    return;
  }
  await Plotly.restyle(plotEl, { visible: values }, indices);
}

async function fetchSampleFigure(sampleId) {
  if (figureCache.has(sampleId)) {
    return figureCache.get(sampleId);
  }

  const detail = await requestJSON(`/api/sample?sample_id=${encodeURIComponent(sampleId)}`);
  detailCache.set(sampleId, detail);
  figureCache.set(sampleId, detail.figure);
  updateCardDetail(sampleId, detail);
  return detail.figure;
}

async function renderSampleFigure(sampleId) {
  const plotEl = gridEl.querySelector(`[data-plot-for="${CSS.escape(sampleId)}"]`);
  if (!plotEl) {
    return;
  }

  figurePlaceholder(sampleId, "Loading model…");

  try {
    const figure = await fetchSampleFigure(sampleId);
    if (!gridEl.contains(plotEl)) {
      return;
    }
    plotEl.innerHTML = "";
    await Plotly.newPlot(plotEl, figure.data, figure.layout, FIGURE_CONFIG);
    await applyContextVisibilityToPlot(plotEl);
    plotEl.classList.add("is-ready");
  } catch (error) {
    figurePlaceholder(sampleId, `Could not render model.\n${error.message}`);
  }
}

async function loadVisibleFigures() {
  for (const sample of state.samples) {
    await renderSampleFigure(sample.sample_id);
  }
}

async function loadPage(page) {
  const layout = state.layout || computeLayout();
  if (!layout) {
    return;
  }

  applyLayout(layout);
  state.loading = true;
  state.selected.clear();
  updateToolbar();

  try {
    const payload = await requestJSON(`/api/page?page=${page}&page_size=${layout.pageSize}`);
    state.page = payload.page;
    state.pageCount = payload.page_count;
    state.pageSize = payload.page_size;
    state.totalCount = payload.total_count;
    state.startIndex = payload.start_index;
    state.endIndex = payload.end_index;
    state.savedCount = payload.saved_count;
    state.samples = payload.samples;
    csvPathEl.textContent = payload.csv_path ? `CSV: ${payload.csv_path}` : "";
    renderGrid();
  } catch (error) {
    setMessage(error.message, "error");
  } finally {
    state.loading = false;
    updateToolbar();
  }
}

function toggleSampleSelection(sampleId) {
  const sample = state.samples.find((item) => item.sample_id === sampleId);
  if (!sample || sample.saved) {
    return;
  }

  if (state.selected.has(sampleId)) {
    state.selected.delete(sampleId);
  } else {
    state.selected.add(sampleId);
  }
  syncAllCardStates();
}

async function persistSelected({ reloadPage = true } = {}) {
  if (state.loading || state.selected.size === 0) {
    return true;
  }

  state.loading = true;
  updateToolbar();
  const selectedIds = Array.from(state.selected);

  try {
    const payload = await requestJSON("/api/save", {
      method: "POST",
      body: JSON.stringify({ sample_ids: selectedIds }),
    });

    state.savedCount = payload.saved_count;
    state.selected.clear();
    const addedCount = payload.added.length;
    setMessage(
      addedCount
        ? `Saved ${addedCount} rejected model${addedCount === 1 ? "" : "s"} to the CSV.`
        : "Those models were already saved earlier.",
      addedCount ? "success" : "neutral",
    );

    if (reloadPage) {
      await loadPage(state.page);
    } else {
      state.loading = false;
      updateToolbar();
    }
    return true;
  } catch (error) {
    state.loading = false;
    updateToolbar();
    setMessage(error.message, "error");
    return false;
  }
}

async function removeSaved(sampleId) {
  if (state.loading) {
    return;
  }

  state.loading = true;
  updateToolbar();

  try {
    const payload = await requestJSON("/api/remove", {
      method: "POST",
      body: JSON.stringify({ sample_id: sampleId }),
    });
    if (payload.removed) {
      setMessage(`Removed ${sampleId} from the rejected-model CSV.`, "success");
    } else {
      setMessage(`${sampleId} was not present in the rejected-model CSV.`, "neutral");
    }
    await loadPage(state.page);
  } catch (error) {
    state.loading = false;
    updateToolbar();
    setMessage(error.message, "error");
  }
}

async function refreshLayout(preserveIndex = state.startIndex) {
  const layout = computeLayout();
  if (!layout) {
    return;
  }

  if (state.layout && state.layout.columns === layout.columns && state.layout.rows === layout.rows) {
    return;
  }

  applyLayout(layout);
  const targetPage = Math.floor((preserveIndex || 0) / layout.pageSize);
  const hadSelection = state.selected.size > 0;
  if (hadSelection) {
    const saved = await persistSelected({ reloadPage: false });
    if (!saved) {
      return;
    }
  } else {
    setMessage("");
  }
  await loadPage(targetPage);
}

async function changePage(targetPage) {
  if (state.loading || targetPage === state.page || targetPage < 0 || targetPage >= state.pageCount) {
    return;
  }

  const hadSelection = state.selected.size > 0;
  if (hadSelection) {
    const saved = await persistSelected({ reloadPage: false });
    if (!saved) {
      return;
    }
  } else {
    setMessage("");
  }

  await loadPage(targetPage);
}

function parsePageJumpValue(rawValue) {
  const requestedPage = Number.parseInt(String(rawValue).trim(), 10);
  if (!Number.isFinite(requestedPage)) {
    return null;
  }
  return Math.min(Math.max(requestedPage - 1, 0), Math.max(state.pageCount - 1, 0));
}

function isEditableTarget(target) {
  return (
    target instanceof HTMLInputElement
    || target instanceof HTMLTextAreaElement
    || target instanceof HTMLSelectElement
    || Boolean(target?.isContentEditable)
  );
}

async function submitPageJump() {
  if (!pageJumpInputEl) {
    return;
  }

  const targetPage = parsePageJumpValue(pageJumpInputEl.value);
  if (targetPage === null) {
    syncPageJumpInput(true);
    return;
  }

  await changePage(targetPage);
  syncPageJumpInput(true);
}

gridEl.addEventListener("click", async (event) => {
  if (event.target.closest(".sample-card__plot-shell")) {
    return;
  }

  const card = event.target.closest(".sample-card");
  if (!card) {
    return;
  }

  const sampleId = card.dataset.sampleId;
  const sample = state.samples.find((item) => item.sample_id === sampleId);
  if (!sample) {
    return;
  }

  if (sample.saved) {
    await removeSaved(sampleId);
    return;
  }

  toggleSampleSelection(sampleId);
});

if (contextToggleEl) {
  contextToggleEl.addEventListener("click", async () => {
    state.showContext = !state.showContext;
    updateToolbar();
    const plotNodes = gridEl.querySelectorAll(".sample-card__plot.is-ready");
    for (const plotEl of plotNodes) {
      await applyContextVisibilityToPlot(plotEl);
    }
  });
}

window.addEventListener("keydown", async (event) => {
  if (event.altKey || event.ctrlKey || event.metaKey) {
    return;
  }

  if (isEditableTarget(event.target)) {
    return;
  }

  if (event.key === "Enter") {
    event.preventDefault();
    await persistSelected();
    return;
  }

  if (event.key === "ArrowRight") {
    event.preventDefault();
    await changePage(state.page + 1);
    return;
  }

  if (event.key.toLowerCase() === "d") {
    event.preventDefault();
    await changePage(state.page + 1);
    return;
  }

  if (event.key === "ArrowLeft") {
    event.preventDefault();
    await changePage(state.page - 1);
    return;
  }

  if (event.key.toLowerCase() === "a") {
    event.preventDefault();
    await changePage(state.page - 1);
  }
});

if (pageJumpFormEl && pageJumpInputEl) {
  pageJumpFormEl.addEventListener("submit", async (event) => {
    event.preventDefault();
    await submitPageJump();
  });

  pageJumpInputEl.addEventListener("focus", () => {
    pageJumpInputEl.select();
  });

  pageJumpInputEl.addEventListener("blur", () => {
    syncPageJumpInput(true);
  });
}

window.addEventListener("resize", () => {
  window.clearTimeout(resizeTimer);
  resizeTimer = window.setTimeout(() => {
    refreshLayout();
  }, 120);
});

window.addEventListener("DOMContentLoaded", async () => {
  const layout = computeLayout();
  if (!layout) {
    setMessage("Waiting for the grid layout to become available.");
    return;
  }
  applyLayout(layout);
  await loadPage(0);
});
