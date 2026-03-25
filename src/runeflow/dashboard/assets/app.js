/* Runeflow Dashboard — app.js
 * Vanilla JS: zone/provider nav + price table expand + timestamp formatting.
 * No external dependencies. Progressive enhancement only.
 */

"use strict";

// ── Timestamp formatting ──────────────────────────────────────────────────
(function formatTimestamps() {
  const els = document.querySelectorAll(".ts");
  els.forEach(function (el) {
    const raw = el.textContent.trim();
    if (!raw) return;
    try {
      const d = new Date(raw);
      if (isNaN(d.getTime())) return;
      el.textContent = d.toLocaleString("en-GB", {
        year: "numeric", month: "short", day: "numeric",
        hour: "2-digit", minute: "2-digit", hour12: false,
      });
      el.title = raw; // ISO timestamp on hover
    } catch (_) {}
  });
})();

// ── Show all table rows ───────────────────────────────────────────────────
function showAllRows() {
  const table = document.getElementById("priceTable");
  if (!table) return;
  table.querySelectorAll("tr.hidden").forEach(function (tr) {
    tr.classList.remove("hidden");
    tr.classList.add("visible");
  });
  const btn = document.getElementById("showAllBtn");
  if (btn) btn.style.display = "none";
}

// ── Data freshness indicator ──────────────────────────────────────────────
(function freshnessCheck() {
  const metas = document.querySelectorAll("[data-generated-at]");
  metas.forEach(function (el) {
    const ts = el.getAttribute("data-generated-at");
    if (!ts) return;
    const age = (Date.now() - new Date(ts).getTime()) / 1000 / 60; // minutes
    let cls = "fresh-green";
    if (age > 360) cls = "fresh-red";
    else if (age > 60) cls = "fresh-amber";
    el.classList.add(cls);
  });
})();
// ── Dynamic URL filling ───────────────────────────────────────────────────
// Replace {ORIGIN} placeholder in code blocks with actual window.location.origin
(function fillDynamicUrls() {
  var origin = window.location.origin;
  document.querySelectorAll(".js-dynamic-origin").forEach(function (el) {
    el.textContent = el.textContent.replace(/\{ORIGIN\}/g, origin);
  });
})();

// ── Model version formatting ──────────────────────────────────────────────
// Format 12-digit YYYYMMDDHHmm train timestamps into human-readable form
(function formatModelVersions() {
  document.querySelectorAll(".js-model-ver").forEach(function (el) {
    var v = el.textContent.trim();
    if (!/^\d{12}$/.test(v)) return;
    var d = new Date(
      parseInt(v.slice(0, 4), 10),
      parseInt(v.slice(4, 6), 10) - 1,
      parseInt(v.slice(6, 8), 10),
      parseInt(v.slice(8, 10), 10),
      parseInt(v.slice(10, 12), 10)
    );
    if (isNaN(d.getTime())) return;
    el.title = v;
    el.textContent = d.toLocaleString("en-GB", {
      year: "numeric", month: "short", day: "numeric",
      hour: "2-digit", minute: "2-digit", hour12: false,
    });
  });
})();
