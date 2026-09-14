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
};

const gridEl = document.querySelector("#grid");
const pageIndicatorEl = document.querySelector("#page-indicator");
const rangeIndicatorEl = document.querySelector("#range-indicator");
const selectedIndicatorEl = document.querySelector("#selected-indicator");
const savedIndicatorEl = document.querySelector("#saved-indicator");
const messageEl = document.querySelector("#message");
const csvPathEl = document.querySelector("#csv-path");
const progressFillEl = document.querySelector("#progress-fill");
const pageJumpFormEl = document.querySelector("#page-jump-form");
const pageJumpInputEl = document.querySelector("#page-jump-input");

let resizeTimer = null;
const TILE_SIZE = 200;

function getLayoutConstants() {
  return { tileSize: TILE_SIZE, gap: window.innerWidth <= 720 ? 10 : 12 };
}

function computeLayout() {
  const width = gridEl.clientWidth;
  const height = gridEl.clientHeight;
  if (!width || !height) {
    return null;
  }

  const { tileSize, gap } = getLayoutConstants();
  const columns = Math.max(1, Math.floor((width + gap) / (tileSize + gap)));
  const rows = Math.max(1, Math.floor((height + gap) / (tileSize + gap)));

  return {
    columns,
    rows,
    pageSize: columns * rows,
    tileSize,
    gap,
  };
}

function setMessage(text, tone = "neutral") {
  messageEl.textContent = text;
  messageEl.dataset.tone = tone;
}

function escapeHTML(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function applyLayout(layout) {
  if (!layout) {
    return;
  }

  state.layout = layout;
  gridEl.style.setProperty("--grid-columns", String(layout.columns));
  gridEl.style.setProperty("--grid-gap", `${layout.gap}px`);
  gridEl.style.setProperty("--tile-size", `${layout.tileSize}px`);
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
  document.body.classList.toggle("is-loading", state.loading);
  updateProgressBar();
  syncPageJumpInput();
}

function buildCard(sample) {
  const card = document.createElement("button");
  card.type = "button";
  card.className = "sample-card";
  card.dataset.sampleId = sample.sample_id;
  card.dataset.saved = sample.saved ? "true" : "false";

  if (sample.saved) {
    card.classList.add("is-saved");
  }
  if (state.selected.has(sample.sample_id)) {
    card.classList.add("is-selected");
  }

  const neighborPath = sample.neighbor_path
    ? `<path class="footprint footprint--neighbor" d="${sample.neighbor_path}"></path>`
    : "";

  card.innerHTML = `
    <div class="sample-card__frame">
      <svg viewBox="${sample.view_box}" preserveAspectRatio="xMidYMid meet" aria-hidden="true">
        ${neighborPath}
        <path class="footprint footprint--main" d="${sample.main_path}"></path>
      </svg>
    </div>
    <div class="sample-card__meta">
      <span class="sample-card__name">${escapeHTML(sample.sample_id)}</span>
    </div>
  `;
  return card;
}

function renderGrid() {
  gridEl.replaceChildren(...state.samples.map((sample) => buildCard(sample)));
  updateToolbar();
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
  if (state.selected.has(sampleId)) {
    state.selected.delete(sampleId);
  } else {
    state.selected.add(sampleId);
  }
  renderGrid();
}

async function persistSelected({ reloadPage = true } = {}) {
  if (state.loading) {
    return false;
  }

  if (state.selected.size === 0) {
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
        ? `Saved ${addedCount} sample${addedCount === 1 ? "" : "s"} to the CSV.`
        : "Those samples were already saved earlier.",
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
      setMessage(`Removed ${sampleId} from the CSV.`, "success");
    } else {
      setMessage(`${sampleId} was not present in the CSV.`, "neutral");
    }
    await loadPage(state.page);
  } catch (error) {
    state.loading = false;
    updateToolbar();
    setMessage(error.message, "error");
  }
}

function currentPageHasSelection(sampleId) {
  return state.samples.some((sample) => sample.sample_id === sampleId && !sample.saved);
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

gridEl.addEventListener("click", async (event) => {
  const card = event.target.closest(".sample-card");
  if (!card) {
    return;
  }

  const sampleId = card.dataset.sampleId;
  const isSaved = card.dataset.saved === "true";
  if (isSaved) {
    await removeSaved(sampleId);
    return;
  }

  if (currentPageHasSelection(sampleId)) {
    toggleSampleSelection(sampleId);
  }
});

window.addEventListener("keydown", async (event) => {
  if (event.altKey || event.ctrlKey || event.metaKey) {
    return;
  }

  if (isEditableTarget(event.target)) {
    return;
  }

  const key = event.key.toLowerCase();

  if (event.key === "Enter") {
    event.preventDefault();
    await persistSelected();
    return;
  }

  if ((event.key === "ArrowRight" || key === "d") && state.page < state.pageCount - 1) {
    event.preventDefault();
    await changePage(state.page + 1);
    return;
  }

  if ((event.key === "ArrowLeft" || key === "a") && state.page > 0) {
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
  if (resizeTimer) {
    window.clearTimeout(resizeTimer);
  }
  resizeTimer = window.setTimeout(() => {
    refreshLayout();
  }, 120);
});

window.addEventListener("load", () => {
  applyLayout(computeLayout());
  loadPage(0);
});
