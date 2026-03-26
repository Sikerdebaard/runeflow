/* Runeflow Dashboard — forecast-chart.js
 * Interactive price forecast chart using Chart.js.
 * Supports 15-min / 60-min resolution toggle for actual prices.
 */
"use strict";

(function () {
  var container = document.getElementById("chartContainer");
  if (!container) return;

  var chartUrl = container.getAttribute("data-chart-url");
  var tz = container.getAttribute("data-timezone") || "UTC";
  if (!chartUrl) return;

  // Palette (matches PlotService / style.css)
  var COLORS = {
    darkViolet: "#a718c7",
    evergreen: "#042a2b",
    sand: "rgba(247, 212, 188, 0.5)",
    blush: "rgba(250, 227, 227, 0.35)",
    orchid: "#cfa5b4",
    nowLine: "rgba(207, 165, 180, 0.7)",
  };

  var chart = null;
  var chartData = null;
  var currentRes = "60";

  fetch(chartUrl)
    .then(function (r) { return r.json(); })
    .then(function (data) {
      chartData = data;
      buildChart();
      wireButtons();
    })
    .catch(function () {
      container.innerHTML = '<p class="muted">Chart data unavailable.</p>';
    });

  function wireButtons() {
    var btn60 = document.getElementById("btnRes60");
    var btn15 = document.getElementById("btnRes15");
    if (!btn60 || !btn15) return;

    // Hide 15min button if no 15min data
    if (!chartData.series_15min || !chartData.series_15min.length) {
      btn15.style.display = "none";
      return;
    }

    btn60.addEventListener("click", function () {
      if (currentRes === "60") return;
      currentRes = "60";
      btn60.classList.add("active");
      btn15.classList.remove("active");
      updateActuals();
    });

    btn15.addEventListener("click", function () {
      if (currentRes === "15") return;
      currentRes = "15";
      btn15.classList.add("active");
      btn60.classList.remove("active");
      updateActuals();
    });
  }

  function updateActuals() {
    if (!chart || !chartData) return;
    var actuals = buildActualSeries();
    // Dataset index: 0=combined upper, 1=combined lower, 2=static upper, 3=static lower, 4=actuals, 5=forecast
    chart.data.datasets[4].data = actuals;
    chart.update("none");
  }

  function buildActualSeries() {
    var src;
    if (currentRes === "15" && chartData.series_15min && chartData.series_15min.length) {
      src = chartData.series_15min;
    } else {
      src = chartData.series_60min.filter(function (d) { return d.is_actual; });
    }
    return src.map(function (d) {
      return { x: d.start, y: d.value };
    });
  }

  function buildChart() {
    if (typeof Chart === "undefined") {
      // Chart.js not loaded yet — retry
      setTimeout(buildChart, 100);
      return;
    }

    var s60 = chartData.series_60min;

    // Forecast line: show for ALL points that have a prediction,
    // including overlap with actuals
    var forecastPts = [];
    var upperPts = [];
    var lowerPts = [];
    var upperStaticPts = [];
    var lowerStaticPts = [];
    for (var i = 0; i < s60.length; i++) {
      var d = s60[i];
      if (d.prediction != null) {
        forecastPts.push({ x: d.start, y: d.prediction });
        upperPts.push({ x: d.start, y: d.upper != null ? d.upper : d.prediction });
        lowerPts.push({ x: d.start, y: d.lower != null ? d.lower : d.prediction });
        upperStaticPts.push({ x: d.start, y: d.upper_static != null ? d.upper_static : d.prediction });
        lowerStaticPts.push({ x: d.start, y: d.lower_static != null ? d.lower_static : d.prediction });
      }
    }

    // Check if static bounds exist and differ from combined bounds
    var hasStaticBands = upperStaticPts.length > 0 && upperStaticPts.some(function (pt, idx) {
      return pt.y !== upperPts[idx].y || lowerStaticPts[idx].y !== lowerPts[idx].y;
    });

    // "now" annotation
    var nowTs = new Date().toISOString();

    var actuals = buildActualSeries();

    var canvas = document.getElementById("forecastChart");
    var ctx = canvas.getContext("2d");

    // Dataset indices:
    //   0: combined upper, 1: combined lower   (outer envelope — weather ∪ model)
    //   2: static upper,   3: static lower     (inner envelope — model quantiles only)
    //   4: actuals
    //   5: forecast line
    var datasets = [
      // 0: Combined envelope (upper) — weather ensemble ∪ model quantiles (lightgreen)
      {
        label: "95% combined upper",
        data: upperPts,
        borderColor: "rgba(144,238,144,0.6)",
        borderWidth: 0.9,
        borderDash: [4, 3],
        backgroundColor: "rgba(144,238,144,0.20)",
        fill: "+1",
        pointRadius: 0,
        tension: 0,
        stepped: "before",
        order: 6,
      },
      // 1: Combined envelope (lower)
      {
        label: "95% combined lower",
        data: lowerPts,
        borderColor: "rgba(144,238,144,0.6)",
        borderWidth: 0.9,
        borderDash: [4, 3],
        backgroundColor: "rgba(144,238,144,0.20)",
        fill: false,
        pointRadius: 0,
        tension: 0,
        stepped: "before",
        order: 6,
      },
      // 2: Static envelope (upper) — model quantiles only (purple)
      {
        label: "95% model upper",
        data: hasStaticBands ? upperStaticPts : [],
        borderColor: "rgba(167,24,199,0.5)",
        borderWidth: 0.9,
        borderDash: [4, 3],
        backgroundColor: "rgba(167,24,199,0.18)",
        fill: "+1",
        pointRadius: 0,
        tension: 0,
        stepped: "before",
        order: 5,
      },
      // 3: Static envelope (lower)
      {
        label: "95% model lower",
        data: hasStaticBands ? lowerStaticPts : [],
        borderColor: "rgba(167,24,199,0.5)",
        borderWidth: 0.9,
        borderDash: [4, 3],
        backgroundColor: "rgba(167,24,199,0.18)",
        fill: false,
        pointRadius: 0,
        tension: 0,
        stepped: "before",
        order: 5,
      },
      // 4: Actuals (orange, like matplotlib)
      {
        label: "Actual",
        data: actuals,
        borderColor: "#e88a2a",
        backgroundColor: "#e88a2a",
        borderWidth: 1.5,
        pointRadius: 0,
        pointHitRadius: 6,
        stepped: "before",
        fill: false,
        order: 1,
      },
      // 5: Forecast line (purple, like matplotlib)
      {
        label: "Forecast",
        data: forecastPts,
        borderColor: COLORS.darkViolet,
        backgroundColor: COLORS.darkViolet,
        borderWidth: 2,
        pointRadius: 0,
        pointHitRadius: 6,
        stepped: "before",
        fill: false,
        order: 2,
      },
    ];

    chart = new Chart(ctx, {
      type: "line",
      data: {
        datasets: datasets,
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          mode: "index",
          intersect: false,
        },
        plugins: {
          legend: {
            display: true,
            position: "top",
            labels: {
              usePointStyle: true,
              generateLabels: function (chart) {
                var labels = [];
                // Actual
                labels.push({
                  text: "Actual",
                  fillStyle: "#e88a2a",
                  strokeStyle: "#e88a2a",
                  lineWidth: 1.5,
                  pointStyle: "line",
                  hidden: !chart.isDatasetVisible(4),
                  datasetIndex: 4,
                });
                // Forecast
                labels.push({
                  text: "Forecast",
                  fillStyle: COLORS.darkViolet,
                  strokeStyle: COLORS.darkViolet,
                  lineWidth: 2,
                  pointStyle: "line",
                  hidden: !chart.isDatasetVisible(5),
                  datasetIndex: 5,
                });
                // Combined 95% envelope (green)
                labels.push({
                  text: "95% weather \u222a model",
                  fillStyle: "rgba(144,238,144,0.20)",
                  strokeStyle: "rgba(144,238,144,0.6)",
                  lineWidth: 1,
                  pointStyle: "rect",
                  hidden: !chart.isDatasetVisible(0),
                  datasetIndex: 0,
                });
                // Static 95% envelope (purple) — only if present
                if (chart.data.datasets[2].data.length > 0) {
                  labels.push({
                    text: "95% model quantiles",
                    fillStyle: "rgba(167,24,199,0.18)",
                    strokeStyle: "rgba(167,24,199,0.5)",
                    lineWidth: 1,
                    pointStyle: "rect",
                    hidden: !chart.isDatasetVisible(2),
                    datasetIndex: 2,
                  });
                }
                return labels;
              },
            },
          },
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
                var lbl = ctx.dataset.label;
                if (lbl !== "Actual" && lbl !== "Forecast") {
                  return null;
                }
                var val = ctx.parsed.y;
                if (val == null) return null;
                return ctx.dataset.label + ": \u20AC" + val.toFixed(4) + "/kWh";
              },
            },
          },
          // "Now" line annotation via custom plugin
          nowLine: { timestamp: nowTs, tz: tz },
        },
        scales: {
          x: {
            type: "time",
            time: {
              tooltipFormat: "PPpp",
              displayFormats: {
                hour: "HH:mm",
                day: "d MMM",
              },
              unit: "hour",
              stepSize: 6,
            },
            title: { display: true, text: "Time (" + tz + ")" },
            ticks: {
              source: "auto",
              major: { enabled: true },
              font: function (ctx) {
                return ctx.tick && ctx.tick.major ? { weight: "bold" } : {};
              },
            },
            grid: {
              color: function (ctx) {
                // Heavier gridline at midnight
                if (ctx.tick && ctx.tick.major) return "rgba(0,0,0,0.15)";
                return "rgba(0,0,0,0.05)";
              },
            },
          },
          y: {
            title: { display: true, text: "EUR/kWh" },
            ticks: {
              callback: function (v) { return "\u20AC" + v.toFixed(3); },
            },
          },
        },
      },
      plugins: [nowLinePlugin],
    });
  }

  // Custom Chart.js plugin: draw a vertical "now" line
  var nowLinePlugin = {
    id: "nowLine",
    afterDraw: function (chart) {
      var opts = chart.options.plugins.nowLine;
      if (!opts || !opts.timestamp) return;
      var xScale = chart.scales.x;
      var yScale = chart.scales.y;
      var nowX = xScale.getPixelForValue(new Date(opts.timestamp).getTime());
      if (nowX < xScale.left || nowX > xScale.right) return;
      var ctx = chart.ctx;
      ctx.save();
      ctx.beginPath();
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = COLORS.nowLine;
      ctx.lineWidth = 1.5;
      ctx.moveTo(nowX, yScale.top);
      ctx.lineTo(nowX, yScale.bottom);
      ctx.stroke();
      // Label
      ctx.fillStyle = COLORS.orchid;
      ctx.font = "11px system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("now", nowX, yScale.top - 4);
      ctx.restore();
    },
  };
})();
