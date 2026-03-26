/* Runeflow Dashboard — app.js
 * Vanilla JS: zone/provider nav + price table expand + timestamp formatting.
 * No external dependencies. Progressive enhancement only.
 */

"use strict";

// ── Timestamp formatting ──────────────────────────────────────────────────
(function formatTimestamps() {
  const tz = document.querySelector("[data-timezone]")?.getAttribute("data-timezone") || "UTC";
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
        timeZone: tz,
      });
      el.title = raw; // ISO timestamp on hover
    } catch (_) {}
  });
})();

// ── Show all table rows ───────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", function () {
  var btn = document.getElementById("showAllBtn");
  if (btn) btn.addEventListener("click", showAllRows);
});

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

// ── Scroll price table to current hour ───────────────────────────────────
(function scrollTableToNow() {
  const table = document.getElementById("priceTable");
  if (!table) return;
  const rows = Array.from(table.querySelectorAll("tbody tr[data-start]"));
  if (!rows.length) return;

  const now = Date.now();
  // Find the last row whose slot starts at or before now (i.e. the current slot).
  let startIdx = 0;
  for (let i = 0; i < rows.length; i++) {
    const t = new Date(rows[i].dataset.start).getTime();
    if (isNaN(t)) continue;
    if (t <= now) { startIdx = i; } else { break; }
  }

  // Reveal 24 rows from startIdx.
  rows.forEach(function (tr, i) {
    if (i >= startIdx && i < startIdx + 24) {
      tr.classList.remove("hidden");
      tr.classList.add("visible");
    }
  });
})();

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

// ── Dynamic current price ─────────────────────────────────────────────────
// Fetch tariff.json and resolve the current price based on the zone timezone.
(function updateCurrentPrice() {
  var banner = document.getElementById("currentPriceBanner");
  if (!banner) return;
  var tz = banner.getAttribute("data-timezone");
  var url = banner.getAttribute("data-tariff-url");
  if (!tz || !url) return;

  fetch(url)
    .then(function (r) { return r.json(); })
    .then(function (data) {
      var rates = data.rates || [];
      if (!rates.length) return;

      // Current instant in the zone's wall-clock
      var now = new Date();
      var slot = null;
      for (var i = 0; i < rates.length; i++) {
        var start = new Date(rates[i].start);
        var end = new Date(rates[i].end);
        if (start <= now && now < end) {
          slot = rates[i];
          break;
        }
      }
      if (!slot) return;

      var price = slot.value != null ? slot.value : slot.price;
      if (price == null) return;

      // Determine price class relative to the day's range
      var prices = rates.map(function (r) {
        return r.value != null ? r.value : r.price;
      }).filter(function (v) { return v != null; });
      var min = Math.min.apply(null, prices);
      var max = Math.max.apply(null, prices);
      var range = max - min || 1;
      var rel = (price - min) / range;
      var cls = rel < 0.33 ? "price-cheap" : (rel > 0.67 ? "price-expensive" : "price-normal");

      // Update DOM
      var valueEl = document.getElementById("currentPriceValue");
      if (valueEl) {
        valueEl.textContent = "\u20AC" + price.toFixed(2);
      }
      banner.className = "current-price-banner " + cls;
      banner.style.display = "";
    })
    .catch(function () { /* keep server-rendered fallback */ });
})();

// ── Dynamic prices on zone overview cards ─────────────────────────────────
(function updateZonePrices() {
  var cards = document.querySelectorAll(".js-zone-price");
  if (!cards.length) return;
  cards.forEach(function (el) {
    var url = el.getAttribute("data-tariff-url");
    if (!url) return;
    fetch(url)
      .then(function (r) { return r.json(); })
      .then(function (data) {
        var rates = data.rates || [];
        if (!rates.length) return;
        var now = new Date();
        var slot = null;
        for (var i = 0; i < rates.length; i++) {
          var start = new Date(rates[i].start);
          var end = new Date(rates[i].end);
          if (start <= now && now < end) { slot = rates[i]; break; }
        }
        if (!slot) return;
        var price = slot.value != null ? slot.value : slot.price;
        if (price == null) return;
        // Preserve the /kWh unit span
        var unit = el.querySelector(".price-unit");
        el.textContent = "\u20AC" + price.toFixed(2);
        if (unit) el.appendChild(unit);
        el.style.display = "";
      })
      .catch(function () { /* keep server-rendered fallback */ });
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
