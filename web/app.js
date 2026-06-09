"use strict";

const state = {
  view: "search",
  events: [],
  selectedEvent: null,
  selectedZone: null,
  prediction: null,
};

const views = {
  search: document.getElementById("view-search"),
  detail: document.getElementById("view-detail"),
  result: document.getElementById("view-result"),
};

function render() {
  for (const [name, el] of Object.entries(views)) {
    el.hidden = name !== state.view;
  }
}

function fmtMoney(v) {
  return `$${Number(v).toFixed(2)}`;
}

function fmtDate(iso) {
  const d = new Date(iso);
  return d.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
}

async function searchEvents(ev) {
  ev.preventDefault();
  const q = document.getElementById("q").value.trim();
  const city = document.getElementById("city").value.trim();
  const from = document.getElementById("from").value;
  const to = document.getElementById("to").value;
  const params = new URLSearchParams();
  if (q) params.set("q", q);
  if (city) params.set("city", city);
  if (from) params.set("from", from);
  if (to) params.set("to", to);

  const status = document.getElementById("search-status");
  status.textContent = "Searching...";
  try {
    const res = await fetch(`/api/events/search?${params.toString()}`);
    if (!res.ok) throw new Error(`search failed: ${res.status}`);
    state.events = await res.json();
    status.textContent = state.events.length === 0 ? "No events found." : `${state.events.length} event(s)`;
    renderResults();
  } catch (err) {
    status.textContent = `Error: ${err.message}`;
  }
}

function renderResults() {
  const list = document.getElementById("results");
  list.innerHTML = "";
  for (const e of state.events) {
    const li = document.createElement("li");
    li.className = "rounded border border-slate-200 bg-white p-3 cursor-pointer hover:bg-slate-50 min-h-[44px]";
    li.innerHTML = `
      <div class="flex items-start gap-3">
        <div class="shrink-0 rounded bg-slate-100 px-2 py-1 text-xs text-slate-700">${fmtDate(e.event_datetime)}</div>
        <div class="flex-1">
          <div class="font-medium">${e.artist_or_team}</div>
          <div class="text-sm text-slate-600">${e.venue_name} &middot; ${e.city}</div>
        </div>
      </div>`;
    li.addEventListener("click", () => selectEvent(e.event_id));
    list.appendChild(li);
  }
}

async function selectEvent(eventId) {
  try {
    const res = await fetch(`/api/events/${encodeURIComponent(eventId)}`);
    if (!res.ok) throw new Error(`event lookup failed: ${res.status}`);
    state.selectedEvent = await res.json();
    state.selectedZone = null;
    document.getElementById("section").value = "";
    document.getElementById("row").value = "";
    document.getElementById("quantity").value = "2";
    document.getElementById("as-of-datetime").value = "";
    populateTzOptions();
    setAsOfDateBounds();
    document.querySelectorAll(".zone").forEach((el) => el.classList.remove("selected"));
    populateEventHeader();
    updatePredictEnabled();
    state.view = "detail";
    render();
  } catch (err) {
    alert(err.message);
  }
}

function populateEventHeader() {
  const e = state.selectedEvent;
  document.getElementById("event-artist").textContent = e.artist_or_team;
  document.getElementById("event-venue").textContent = `${e.venue_name} · ${e.city}`;
  document.getElementById("event-date").textContent = fmtDate(e.event_datetime);
}

// Timezones offered in the dropdown. Browser's local zone is added (and selected)
// at runtime so the user doesn't have to think about it for the common case.
const TIMEZONE_OPTIONS = [
  "UTC",
  "America/Los_Angeles",
  "America/Denver",
  "America/Chicago",
  "America/New_York",
  "Europe/London",
  "Europe/Paris",
  "Asia/Seoul",
  "Asia/Tokyo",
  "Asia/Shanghai",
  "Australia/Sydney",
];

function populateTzOptions() {
  const sel = document.getElementById("as-of-tz");
  if (sel.options.length > 0) return; // already populated
  const local = Intl.DateTimeFormat().resolvedOptions().timeZone;
  const all = Array.from(new Set([local, ...TIMEZONE_OPTIONS]));
  for (const tz of all) {
    const opt = document.createElement("option");
    opt.value = tz;
    opt.textContent = tz === local ? `${tz} (local)` : tz;
    if (tz === local) opt.selected = true;
    sel.appendChild(opt);
  }
}

// Convert a "wall-clock time in tz" (e.g. "2026-05-03T14:00" in "Asia/Seoul") to UTC.
// We compute it by formatting Date.UTC(...) in tz and measuring the offset.
function wallClockInTzToUtcIso(localStr, tz) {
  // localStr is "YYYY-MM-DDTHH:mm" (no timezone). Treat the wall-clock fields as
  // belonging to `tz`, then return the corresponding UTC instant.
  const [datePart, timePart] = localStr.split("T");
  const [y, m, d] = datePart.split("-").map(Number);
  const [hh, mm] = timePart.split(":").map(Number);
  const utcGuess = Date.UTC(y, m - 1, d, hh, mm);
  // Find what the tz says about that guess; difference = offset.
  const fmt = new Intl.DateTimeFormat("en-US", {
    timeZone: tz, hour12: false,
    year: "numeric", month: "2-digit", day: "2-digit",
    hour: "2-digit", minute: "2-digit", second: "2-digit",
  });
  const parts = Object.fromEntries(
    fmt.formatToParts(new Date(utcGuess)).filter((p) => p.type !== "literal").map((p) => [p.type, p.value])
  );
  const tzWall = Date.UTC(
    Number(parts.year), Number(parts.month) - 1, Number(parts.day),
    Number(parts.hour) % 24, Number(parts.minute), Number(parts.second),
  );
  // utcGuess - tzWall == offset (ms). Real UTC = utcGuess - offset.
  const offset = tzWall - utcGuess;
  return new Date(utcGuess - offset).toISOString();
}

function setAsOfDateBounds() {
  const e = state.selectedEvent;
  if (!e) return;
  const input = document.getElementById("as-of-datetime");
  // datetime-local expects "YYYY-MM-DDTHH:mm" in local time. The bounds use the
  // browser's local zone; we don't try to track the dropdown selection because
  // datetime-local has no API for that.
  function localIso(d) {
    const off = d.getTimezoneOffset();
    return new Date(d.getTime() - off * 60000).toISOString().slice(0, 16);
  }
  input.min = localIso(new Date());
  input.max = localIso(new Date(e.event_datetime));
}

function onZoneClick(ev) {
  const zone = ev.currentTarget.getAttribute("data-zone");
  state.selectedZone = zone;
  document.querySelectorAll(".zone").forEach((el) => el.classList.remove("selected"));
  ev.currentTarget.classList.add("selected");
  updatePredictEnabled();
}

function updatePredictEnabled() {
  const section = document.getElementById("section").value.trim();
  const btn = document.getElementById("predict-btn");
  btn.disabled = !(state.selectedZone || section);
}

async function predict() {
  const e = state.selectedEvent;
  const section = document.getElementById("section").value.trim();
  const row = document.getElementById("row").value.trim();
  const quantity = parseInt(document.getElementById("quantity").value, 10);
  const asOfLocal = document.getElementById("as-of-datetime").value;
  const asOfTz = document.getElementById("as-of-tz").value;
  const body = { event_id: e.event_id, quantity };
  if (state.selectedZone) body.seat_zone = state.selectedZone;
  if (section) body.section = section;
  if (row) body.row = row;
  if (asOfLocal) body.as_of_date = wallClockInTzToUtcIso(asOfLocal, asOfTz);

  const status = document.getElementById("predict-status");
  status.textContent = "Predicting...";
  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(detail.detail || `predict failed: ${res.status}`);
    }
    state.prediction = await res.json();
    status.textContent = "";
    renderPrediction();
    state.view = "result";
    render();
  } catch (err) {
    status.textContent = `Error: ${err.message}`;
  }
}

function renderPrediction() {
  const p = state.prediction;
  document.getElementById("result-price").textContent = fmtMoney(p.predicted_price);
  document.getElementById("result-zone").textContent = `${p.seat_zone} · ${p.target_days_to_event} days to event`;
  document.getElementById("range-lower").textContent = fmtMoney(p.price_lower_bound);
  document.getElementById("range-upper").textContent = fmtMoney(p.price_upper_bound);

  const span = p.price_upper_bound - p.price_lower_bound;
  const pct = span > 0
    ? ((p.predicted_price - p.price_lower_bound) / span) * 100
    : 50;
  const clamped = Math.max(0, Math.min(100, pct));
  document.getElementById("range-marker").style.left = `${clamped}%`;

  const arrows = { UP: "↑", DOWN: "↓", STABLE: "→" };
  document.getElementById("direction-arrow").textContent = arrows[p.predicted_direction] || "→";
  document.getElementById("direction-prob").textContent = `${(p.direction_probability * 100).toFixed(0)}%`;
  document.getElementById("confidence-score").textContent = `${(p.confidence_score * 100).toFixed(0)}%`;
  document.getElementById("model-version").textContent = p.model_version;
}

function init() {
  document.getElementById("search-form").addEventListener("submit", searchEvents);
  document.getElementById("back-to-search").addEventListener("click", () => {
    state.view = "search";
    render();
  });
  document.getElementById("section").addEventListener("input", updatePredictEnabled);
  document.getElementById("predict-btn").addEventListener("click", predict);
  document.getElementById("predict-another").addEventListener("click", () => {
    state.view = "detail";
    render();
  });
  document.querySelectorAll(".zone").forEach((el) => el.addEventListener("click", onZoneClick));
  render();
}

init();
