/**
 * zone-search.js — Live search / keyboard navigation for the zone index page.
 * Reads data-search attributes already embedded in the HTML; no inline data needed.
 */
(function () {
  "use strict";

  var input   = document.getElementById("zone-search");
  var counter = document.getElementById("zone-search-count");
  if (!input) return;

  var grid  = document.getElementById("zone-grid");
  var cards = Array.from(document.querySelectorAll("#zone-grid .zone-card"));
  var hilite = -1;  // index into cards[] of the currently highlighted card

  function norm(s) {
    return (s || "").toLowerCase().replace(/[-_\/\s]+/g, " ").trim();
  }

  function visibleCards() {
    return cards.filter(function (c) { return !c.classList.contains("zone-hidden"); });
  }

  /** Return the number of columns currently rendered in the grid. */
  function gridCols() {
    if (!grid) return 1;
    var tpl = getComputedStyle(grid).gridTemplateColumns;
    return tpl ? tpl.split(" ").length : 1;
  }

  /** Move highlight to the card at visIdx in the visible list (no wrapping). */
  function setHighlight(visIdx) {
    var vis = visibleCards();
    if (!vis.length) {
      hilite = -1;
      return;
    }
    // Clamp — no wrapping
    visIdx = Math.max(0, Math.min(vis.length - 1, visIdx));
    cards.forEach(function (c) { c.classList.remove("zone-card--active"); });
    vis[visIdx].classList.add("zone-card--active");
    vis[visIdx].scrollIntoView({ block: "nearest", behavior: "smooth" });
    hilite = cards.indexOf(vis[visIdx]);
  }

  /** Current index of the highlighted card within the *visible* list, or -1. */
  function currentVisIdx() {
    if (hilite < 0) return -1;
    return visibleCards().indexOf(cards[hilite]);
  }

  function applyFilter() {
    var q = norm(input.value);
    hilite = -1;
    cards.forEach(function (c) {
      var match = !q || norm(c.dataset.search || "").indexOf(q) !== -1;
      c.classList.toggle("zone-hidden", !match);
      c.classList.remove("zone-card--active");
    });
    var vis = visibleCards();
    if (q) {
      if (counter) counter.textContent = vis.length + " of " + cards.length + " zones";
      if (vis.length) setHighlight(0);
    } else {
      if (counter) counter.textContent = "";
    }
  }

  input.addEventListener("input", applyFilter);

  input.addEventListener("keydown", function (e) {
    var cur = currentVisIdx();
    var cols = gridCols();

    if (e.key === "ArrowRight") {
      e.preventDefault();
      setHighlight(cur < 0 ? 0 : cur + 1);
    } else if (e.key === "ArrowLeft") {
      e.preventDefault();
      setHighlight(cur < 0 ? 0 : cur - 1);
    } else if (e.key === "ArrowDown") {
      e.preventDefault();
      setHighlight(cur < 0 ? 0 : cur + cols);
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setHighlight(cur < 0 ? 0 : cur - cols);
    } else if (e.key === "Enter") {
      var active = document.querySelector("#zone-grid .zone-card--active");
      if (active) { window.location.href = active.href; }
    } else if (e.key === "Escape") {
      input.value = "";
      applyFilter();
      input.blur();
    }
  });

  // "/" focuses the search field from anywhere on the page
  document.addEventListener("keydown", function (e) {
    if (e.key === "/" && document.activeElement !== input) {
      e.preventDefault();
      input.focus();
      input.select();
    }
  });
})();
