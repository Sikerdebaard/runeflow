/* Runeflow Dashboard — advanced-panels.js
 * Advanced forecast analytics panels: uncertainty breakdown,
 * model agreement, and summary statistics.
 * Only activates when #advancedPanels container exists.
 */
"use strict";

(function () {
  var section = document.getElementById("advancedPanels");
  if (!section) return;

  var chartUrl = section.getAttribute("data-chart-url");
  var tz = section.getAttribute("data-timezone") || "UTC";
  if (!chartUrl) return;

  var COLORS = {
    darkViolet: "#a718c7",
    darkVioletLight: "rgba(167,24,199,0.35)",
    evergreen: "#042a2b",
    teal: "#1a6b6d",
    orchid: "#cfa5b4",
    amber: "#d4875c",
    green: "rgba(104,201,125,0.7)",
    greenFill: "rgba(104,201,125,0.15)",
    purpleFill: "rgba(167,24,199,0.15)",
    nowLine: "rgba(207,165,180,0.7)",
  };

  fetch(chartUrl)
    .then(function (r) { return r.json(); })
    .then(function (data) {
      buildSummaryStats(data);
      buildBandWidthChart(data);
      buildAgreementChart(data);
    })
    .catch(function () {
      section.querySelector(".advanced-panels-body").innerHTML =
        '<p class="muted">Advanced data unavailable.</p>';
    });

  // ── Summary statistics cards ──────────────────────────────────────────
  function buildSummaryStats(data) {
    var container = document.getElementById("summaryStats");
    if (!container) return;

    var s60 = data.series_60min;
    var forecastSlots = s60.filter(function (d) { return d.prediction != null && !d.is_actual; });
    if (!forecastSlots.length) {
      container.style.display = "none";
      return;
    }

    var predictions = forecastSlots.map(function (d) { return d.prediction; });
    var agreements = forecastSlots
      .filter(function (d) { return d.model_agreement != null; })
      .map(function (d) { return d.model_agreement; });
    var combinedWidths = forecastSlots
      .filter(function (d) { return d.upper != null && d.lower != null; })
      .map(function (d) { return d.upper - d.lower; });

    var min = Math.min.apply(null, predictions);
    var max = Math.max.apply(null, predictions);
    var avg = predictions.reduce(function (a, b) { return a + b; }, 0) / predictions.length;

    var avgAgreement = agreements.length
      ? agreements.reduce(function (a, b) { return a + b; }, 0) / agreements.length
      : null;
    var avgWidth = combinedWidths.length
      ? combinedWidths.reduce(function (a, b) { return a + b; }, 0) / combinedWidths.length
      : null;

    var cards = [
      { label: "Forecast min", value: "\u20AC" + min.toFixed(4), sub: "per kWh" },
      { label: "Forecast avg", value: "\u20AC" + avg.toFixed(4), sub: "per kWh" },
      { label: "Forecast max", value: "\u20AC" + max.toFixed(4), sub: "per kWh" },
    ];

    if (avgWidth != null) {
      cards.push({
        label: "Avg uncertainty",
        value: "\u20AC" + avgWidth.toFixed(4),
        sub: "band width",
      });
    }
    if (avgAgreement != null) {
      var pct = (avgAgreement * 100).toFixed(0);
      cards.push({
        label: "Model agreement",
        value: pct + "%",
        sub: agreementLabel(avgAgreement),
      });
    }

    cards.push({
      label: "Forecast hours",
      value: String(forecastSlots.length),
      sub: "ahead",
    });

    var html = "";
    for (var i = 0; i < cards.length; i++) {
      html += '<div class="stat-card">' +
        '<div class="stat-label">' + cards[i].label + "</div>" +
        '<div class="stat-value">' + cards[i].value + "</div>" +
        '<div class="stat-sub">' + cards[i].sub + "</div>" +
        "</div>";
    }
    container.innerHTML = html;
  }

  function agreementLabel(v) {
    if (v >= 0.85) return "high consensus";
    if (v >= 0.6) return "moderate";
    return "low consensus";
  }

  // ── Band width comparison chart ───────────────────────────────────────
  function buildBandWidthChart(data) {
    var canvas = document.getElementById("bandWidthChart");
    if (!canvas || typeof Chart === "undefined") return;

    var s60 = data.series_60min;
    var combinedPts = [];
    var staticPts = [];

    for (var i = 0; i < s60.length; i++) {
      var d = s60[i];
      if (d.upper != null && d.lower != null) {
        combinedPts.push({ x: d.start, y: +(d.upper - d.lower).toFixed(6) });
      }
      if (d.upper_static != null && d.lower_static != null) {
        staticPts.push({ x: d.start, y: +(d.upper_static - d.lower_static).toFixed(6) });
      }
    }

    if (!combinedPts.length) {
      canvas.parentElement.style.display = "none";
      return;
    }

    // Check if static data differs
    var hasStatic = staticPts.length > 0 && staticPts.some(function (pt, idx) {
      return combinedPts[idx] && pt.y !== combinedPts[idx].y;
    });

    var datasets = [
      {
        label: "Combined (weather \u222a model)",
        data: combinedPts,
        borderColor: COLORS.green,
        backgroundColor: COLORS.greenFill,
        borderWidth: 1.5,
        pointRadius: 0,
        stepped: "before",
        fill: true,
        order: 2,
      },
    ];

    if (hasStatic) {
      datasets.push({
        label: "Model quantiles only",
        data: staticPts,
        borderColor: COLORS.darkViolet,
        backgroundColor: COLORS.purpleFill,
        borderWidth: 1.5,
        pointRadius: 0,
        stepped: "before",
        fill: true,
        order: 1,
      });
    }

    new Chart(canvas.getContext("2d"), {
      type: "line",
      data: { datasets: datasets },
      options: chartOptions("EUR/kWh", "Uncertainty band width over forecast horizon"),
    });
  }

  // ── Model agreement chart ─────────────────────────────────────────────
  function buildAgreementChart(data) {
    var canvas = document.getElementById("agreementChart");
    if (!canvas || typeof Chart === "undefined") return;

    var s60 = data.series_60min;
    var pts = [];
    for (var i = 0; i < s60.length; i++) {
      var d = s60[i];
      if (d.model_agreement != null) {
        pts.push({ x: d.start, y: d.model_agreement });
      }
    }

    if (!pts.length) {
      canvas.parentElement.style.display = "none";
      return;
    }

    new Chart(canvas.getContext("2d"), {
      type: "line",
      data: {
        datasets: [{
          label: "Model agreement",
          data: pts,
          borderColor: COLORS.teal,
          backgroundColor: "rgba(26,107,109,0.1)",
          borderWidth: 2,
          pointRadius: 0,
          pointHitRadius: 6,
          stepped: "before",
          fill: true,
        }],
      },
      options: agreementChartOptions(),
    });
  }

  // ── Shared chart options ──────────────────────────────────────────────
  function chartOptions(yLabel) {
    return {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { display: true, position: "top", labels: { usePointStyle: true, pointStyle: "line" } },
        tooltip: {
          callbacks: {
            title: function (items) {
              if (!items.length) return "";
              var d = new Date(items[0].parsed.x);
              return d.toLocaleString("en-GB", {
                timeZone: tz,
                year: "numeric", month: "short", day: "numeric",
                hour: "2-digit", minute: "2-digit", hour12: false,
              });
            },
            label: function (ctx) {
              var val = ctx.parsed.y;
              if (val == null) return null;
              return ctx.dataset.label + ": \u20AC" + val.toFixed(4);
            },
          },
        },
      },
      scales: {
        x: {
          type: "time",
          time: { displayFormats: { hour: "HH:mm", day: "d MMM" }, unit: "hour", stepSize: 6 },
          title: { display: true, text: "Time (" + tz + ")" },
          ticks: {
            source: "auto",
            major: { enabled: true },
            font: function (ctx) { return ctx.tick && ctx.tick.major ? { weight: "bold" } : {}; },
          },
          grid: {
            color: function (ctx) {
              return ctx.tick && ctx.tick.major ? "rgba(0,0,0,0.15)" : "rgba(0,0,0,0.05)";
            },
          },
        },
        y: {
          title: { display: true, text: yLabel },
          ticks: { callback: function (v) { return "\u20AC" + v.toFixed(3); } },
          beginAtZero: true,
        },
      },
    };
  }

  function agreementChartOptions() {
    var opts = chartOptions("Agreement");
    opts.scales.y = {
      title: { display: true, text: "Agreement (0\u20131)" },
      min: 0,
      max: 1,
      ticks: {
        callback: function (v) { return (v * 100).toFixed(0) + "%"; },
        stepSize: 0.25,
      },
    };
    opts.plugins.tooltip.callbacks.label = function (ctx) {
      var val = ctx.parsed.y;
      if (val == null) return null;
      return "Agreement: " + (val * 100).toFixed(1) + "%";
    };
    return opts;
  }
})();
