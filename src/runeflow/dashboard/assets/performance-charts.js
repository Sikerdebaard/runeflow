/**
 * performance-charts.js — Chart rendering for the model performance dashboard.
 * Uses the Chart.js library already bundled at /assets/chart.umd.min.js.
 */

(function () {
  "use strict";

  // Colour palette matching the rest of the dashboard
  var COLOURS = {
    primary: "rgba(78, 155, 117, 0.85)",    // --evcc-green equivalent
    primaryBorder: "rgb(78, 155, 117)",
    secondary: "rgba(90, 140, 200, 0.75)",
    secondaryBorder: "rgb(90, 140, 200)",
    warning: "rgba(220, 160, 60, 0.75)",
    warningBorder: "rgb(220, 160, 60)",
    muted: "rgba(150, 150, 160, 0.6)",
    mutedBorder: "rgb(150, 150, 160)",
  };

  var CHART_DEFAULTS = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          label: function (ctx) {
            var v = ctx.parsed.y != null ? ctx.parsed.y : ctx.parsed.x;
            return " " + (v != null ? v.toFixed(2) : "n/a") + " EUR/MWh";
          },
        },
      },
    },
  };

  function initPerformanceCharts() {
    // Data is embedded as <script type="application/json" id="__perf_json"> to
    // satisfy the Content-Security-Policy (script-src 'self' — no inline JS).
    var el = document.getElementById("__perf_json");
    if (!el || typeof Chart === "undefined") return;
    var data;
    try { data = JSON.parse(el.textContent); } catch (e) { return; }

    if (data._rankings) {
      // Global performance page
      renderMaeBarChart(data);
    } else {
      // Zone performance page
      if (data.horizon_metrics && data.horizon_metrics.length) {
        renderHorizonBarChart(data.horizon_metrics);
      }
      if (data.training_history && data.training_history.length) {
        renderTrainingHistoryChart(data.training_history);
      }
      if (data.forecast_accuracies && data.forecast_accuracies.length) {
        renderAccuracyTimelineChart(data.forecast_accuracies);
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Global: MAE bar chart (horizontal, one bar per zone)
  // ---------------------------------------------------------------------------
  function renderMaeBarChart(data) {
    var canvas = document.getElementById("mae-bar-chart");
    if (!canvas) return;

    // Collect zones with MAE data, sorted best→worst
    var zones = Object.keys(data).filter(function (k) {
      return !k.startsWith("_") && data[k].overall_mae != null;
    });
    zones.sort(function (a, b) {
      return (data[a].overall_mae || 0) - (data[b].overall_mae || 0);
    });

    var labels = zones.map(function (z) { return z; });
    var values = zones.map(function (z) { return data[z].overall_mae; });

    new Chart(canvas.getContext("2d"), {
      type: "bar",
      data: {
        labels: labels,
        datasets: [{
          label: "MAE (EUR/MWh)",
          data: values,
          backgroundColor: COLOURS.primary,
          borderColor: COLOURS.primaryBorder,
          borderWidth: 1,
        }],
      },
      options: Object.assign({}, CHART_DEFAULTS, {
        indexAxis: "y",
        plugins: Object.assign({}, CHART_DEFAULTS.plugins, {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: function (ctx) {
                return " MAE: " + ctx.parsed.x.toFixed(2) + " EUR/MWh";
              },
            },
          },
        }),
        scales: {
          x: {
            beginAtZero: true,
            title: { display: true, text: "MAE (EUR/MWh)" },
          },
          y: { ticks: { font: { size: 11 } } },
        },
      }),
    });
  }

  // ---------------------------------------------------------------------------
  // Zone: Horizon MAE bar chart
  // ---------------------------------------------------------------------------
  function renderHorizonBarChart(horizonMetrics) {
    var canvas = document.getElementById("horizon-bar-chart");
    if (!canvas) return;

    var labels = horizonMetrics.map(function (h) { return h.label; });
    var maes   = horizonMetrics.map(function (h) { return h.mae; });
    var rmses  = horizonMetrics.map(function (h) { return h.rmse; });

    new Chart(canvas.getContext("2d"), {
      type: "bar",
      data: {
        labels: labels,
        datasets: [
          {
            label: "MAE",
            data: maes,
            backgroundColor: COLOURS.primary,
            borderColor: COLOURS.primaryBorder,
            borderWidth: 1,
          },
          {
            label: "RMSE",
            data: rmses,
            backgroundColor: COLOURS.secondary,
            borderColor: COLOURS.secondaryBorder,
            borderWidth: 1,
          },
        ],
      },
      options: Object.assign({}, CHART_DEFAULTS, {
        plugins: Object.assign({}, CHART_DEFAULTS.plugins, {
          legend: { display: true },
          tooltip: {
            callbacks: {
              label: function (ctx) {
                return " " + ctx.dataset.label + ": " + ctx.parsed.y.toFixed(2) + " EUR/MWh";
              },
            },
          },
        }),
        scales: {
          x: { title: { display: true, text: "Forecast Horizon" } },
          y: { beginAtZero: true, title: { display: true, text: "Error (EUR/MWh)" } },
        },
      }),
    });
  }

  // ---------------------------------------------------------------------------
  // Zone: Training history line chart (MAE over model versions)
  // ---------------------------------------------------------------------------
  function renderTrainingHistoryChart(history) {
    var canvas = document.getElementById("training-history-chart");
    if (!canvas) return;

    var labels = history.map(function (r) { return r.model_version || r.trained_at || ""; });
    var maes   = history.map(function (r) { return r.mae; });
    var r2s    = history.map(function (r) { return r.r2; });

    new Chart(canvas.getContext("2d"), {
      type: "line",
      data: {
        labels: labels,
        datasets: [
          {
            label: "MAE (EUR/MWh)",
            data: maes,
            borderColor: COLOURS.primaryBorder,
            backgroundColor: COLOURS.primary,
            yAxisID: "y",
            tension: 0.3,
            pointRadius: 4,
          },
          {
            label: "R²",
            data: r2s,
            borderColor: COLOURS.secondaryBorder,
            backgroundColor: COLOURS.secondary,
            yAxisID: "y2",
            tension: 0.3,
            pointRadius: 4,
          },
        ],
      },
      options: Object.assign({}, CHART_DEFAULTS, {
        plugins: Object.assign({}, CHART_DEFAULTS.plugins, {
          legend: { display: true },
          tooltip: {
            callbacks: {
              label: function (ctx) {
                var v = ctx.parsed.y;
                if (ctx.dataset.label.startsWith("R")) {
                  return " R²: " + (v != null ? v.toFixed(4) : "n/a");
                }
                return " MAE: " + (v != null ? v.toFixed(4) : "n/a") + " EUR/MWh";
              },
            },
          },
        }),
        scales: {
          x: { ticks: { maxRotation: 45, font: { size: 10 } } },
          y: {
            beginAtZero: true,
            position: "left",
            title: { display: true, text: "MAE (EUR/MWh)" },
          },
          y2: {
            position: "right",
            min: 0,
            max: 1,
            title: { display: true, text: "R²" },
            grid: { drawOnChartArea: false },
          },
        },
      }),
    });
  }

  // ---------------------------------------------------------------------------
  // Zone: Per-forecast MAE timeline (line chart over time)
  // ---------------------------------------------------------------------------
  function renderAccuracyTimelineChart(accuracies) {
    var canvas = document.getElementById("accuracy-timeline-chart");
    if (!canvas) return;

    var labels = accuracies.map(function (a) { return a.created_at; });
    var maes   = accuracies.map(function (a) { return a.mae; });
    var rmses  = accuracies.map(function (a) { return a.rmse; });

    new Chart(canvas.getContext("2d"), {
      type: "line",
      data: {
        labels: labels,
        datasets: [
          {
            label: "MAE",
            data: maes,
            borderColor: COLOURS.primaryBorder,
            backgroundColor: COLOURS.primary,
            tension: 0.2,
            pointRadius: 3,
          },
          {
            label: "RMSE",
            data: rmses,
            borderColor: COLOURS.secondaryBorder,
            backgroundColor: COLOURS.secondary,
            tension: 0.2,
            pointRadius: 3,
          },
        ],
      },
      options: Object.assign({}, CHART_DEFAULTS, {
        plugins: Object.assign({}, CHART_DEFAULTS.plugins, {
          legend: { display: true },
          tooltip: {
            callbacks: {
              label: function (ctx) {
                return " " + ctx.dataset.label + ": " + ctx.parsed.y.toFixed(2) + " EUR/MWh";
              },
            },
          },
        }),
        scales: {
          x: { ticks: { maxRotation: 45, font: { size: 10 } } },
          y: { beginAtZero: true, title: { display: true, text: "Error (EUR/MWh)" } },
        },
      }),
    });
  }

  // Run after DOM is ready
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initPerformanceCharts);
  } else {
    initPerformanceCharts();
  }
})();
