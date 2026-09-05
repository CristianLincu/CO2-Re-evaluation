/* Dashboard rendering. Reads data/latest.json, produced by pipeline/run.py. */

const COLORS = {
  past: "#94c5e4",
  optimised: "#8fd6a8",
  baseline: "#f0c671",
  band: "rgba(143, 214, 168, 0.18)",
  grid: "#444",
  text: "#d4d4d4",
  muted: "#a0a0a0",
};

const LAYOUT = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { color: COLORS.text, family: "Segoe UI, system-ui, sans-serif", size: 12 },
  margin: { l: 55, r: 20, t: 30, b: 50 },
  xaxis: { gridcolor: COLORS.grid, zerolinecolor: COLORS.grid },
  yaxis: { gridcolor: COLORS.grid, zerolinecolor: COLORS.grid },
  legend: { orientation: "h", y: -0.2 },
  hovermode: "x unified",
};

const CONFIG = { responsive: true, displayModeBar: false };

const STALE_AFTER_MINUTES = 90;

function setStatus(kind, message) {
  const el = document.getElementById("status");
  el.className = `status-bar ${kind}`;
  el.textContent = message;
}

function fmt(value, digits = 1) {
  return value === null || value === undefined || Number.isNaN(value)
    ? "—"
    : Number(value).toFixed(digits);
}

/* --- headline metrics ----------------------------------------------------- */

function renderMetrics(payload) {
  const meta = payload.meta || {};
  const diag = payload.diagnostics || {};

  const cards = [
    {
      label: "Measured now",
      value: fmt(meta.currentCo2),
      unit: "g CO₂/kWh",
      note: "latest Energinet reading",
    },
    {
      label: "Counterfactual",
      value: fmt(meta.baselineCo2),
      unit: "g CO₂/kWh",
      note: "next 4 h if dispatch is held fixed",
    },
    {
      label: "Optimised",
      value: fmt(meta.optimisedCo2),
      unit: "g CO₂/kWh",
      note: "next 4 h under the proposed schedule",
      highlight: true,
    },
    {
      label: "Reduction",
      value: meta.reductionPct === null || meta.reductionPct === undefined ? "—" : fmt(meta.reductionPct),
      unit: "%",
      note: "attributable to re-dispatch",
      highlight: true,
    },
  ];

  document.getElementById("metrics").innerHTML = cards
    .map(
      (c) => `
      <div class="metric${c.highlight ? " metric-accent" : ""}">
        <div class="metric-label">${c.label}</div>
        <div class="metric-value">${c.value}<span class="metric-unit">${c.unit}</span></div>
        <div class="metric-note">${c.note}</div>
      </div>`
    )
    .join("");

  const flags = [];
  if (diag.feasible !== undefined) {
    flags.push(
      diag.feasible
        ? '<span class="flag ok">schedule feasible</span>'
        : '<span class="flag warn">constraint violation</span>'
    );
  }
  if (diag.inDistributionPct !== undefined) {
    const ok = diag.inDistributionPct >= 90;
    flags.push(
      `<span class="flag ${ok ? "ok" : "warn"}">${fmt(diag.inDistributionPct, 0)}% of steps within model support</span>`
    );
  }
  if (diag.withinFloorPct !== undefined) {
    const ok = diag.withinFloorPct >= 99;
    flags.push(
      `<span class="flag ${ok ? "ok" : "warn"}">${fmt(diag.withinFloorPct, 0)}% of steps above the historical floor</span>`
    );
  }
  if (diag.maxBalanceErrorMW !== undefined) {
    flags.push(`<span class="flag ok">max balance error ${fmt(diag.maxBalanceErrorMW, 3)} MW</span>`);
  }
  if (diag.runtimeSeconds) {
    flags.push(`<span class="flag">solved in ${fmt(diag.runtimeSeconds, 0)} s</span>`);
  }
  document.getElementById("metrics").insertAdjacentHTML(
    "afterend",
    `<div class="flags">${flags.join("")}</div>`
  );
}

/* --- charts --------------------------------------------------------------- */

function renderCo2Chart(payload) {
  const chart = payload.co2Chart || {};
  const past = chart.past || { timestamps: [], values: [] };
  const fc = chart.forecast || { timestamps: [], optimised: [], baseline: [], uncertainty: [] };

  // Join the history and forecast so the lines meet rather than floating apart.
  const lastTime = past.timestamps[past.timestamps.length - 1];
  const lastValue = past.values[past.values.length - 1];
  const bridge = (series) => (lastTime ? [lastValue, ...series] : series);
  const bridgeT = (series) => (lastTime ? [lastTime, ...series] : series);

  const upper = fc.optimised.map((v, i) => v + (fc.uncertainty[i] || 0));
  const lower = fc.optimised.map((v, i) => Math.max(0, v - (fc.uncertainty[i] || 0)));

  const traces = [
    {
      x: fc.timestamps.concat([...fc.timestamps].reverse()),
      y: upper.concat([...lower].reverse()),
      fill: "toself",
      fillcolor: COLORS.band,
      line: { width: 0 },
      hoverinfo: "skip",
      showlegend: false,
      type: "scatter",
    },
    {
      x: past.timestamps,
      y: past.values,
      name: "Measured",
      mode: "lines",
      line: { color: COLORS.past, width: 2 },
    },
    {
      x: bridgeT(fc.timestamps),
      y: bridge(fc.baseline),
      name: "Hold current (counterfactual)",
      mode: "lines",
      line: { color: COLORS.baseline, width: 2, dash: "dash" },
    },
    {
      x: bridgeT(fc.timestamps),
      y: bridge(fc.optimised),
      name: "Optimised",
      mode: "lines",
      line: { color: COLORS.optimised, width: 2.5 },
    },
  ];

  const layout = {
    ...LAYOUT,
    yaxis: { ...LAYOUT.yaxis, title: "g CO₂/kWh", rangemode: "tozero" },
    xaxis: { ...LAYOUT.xaxis, title: "Denmark local time" },
    shapes: lastTime
      ? [
          {
            type: "line",
            x0: lastTime,
            x1: lastTime,
            yref: "paper",
            y0: 0,
            y1: 1,
            line: { color: COLORS.muted, width: 1, dash: "dot" },
          },
        ]
      : [],
  };

  Plotly.newPlot("co2-chart", traces, layout, CONFIG);
}

const DISPATCH_SERIES = [
  ["ProductionGe100MW", "Plants ≥100 MW", "#f0c671"],
  ["ProductionLt100MW", "Plants <100 MW", "#d69f5a"],
  ["Exchange_DK1_DE", "DK1-DE", "#e07a5f"],
  ["Exchange_DK1_NO", "DK1-NO", "#8fd6a8"],
  ["Exchange_DK1_GB", "DK1-GB", "#94c5e4"],
  ["Exchange_DK2_SE", "DK2-SE", "#b0a1d8"],
];

function renderDispatchChart(payload) {
  const traj = payload.trajectories || {};
  const times = (payload.forecasts || {}).timestamps || [];
  if (!times.length) return;

  const traces = DISPATCH_SERIES.filter(([key]) => traj[key]).map(([key, label, colour]) => ({
    x: times,
    y: traj[key],
    name: label,
    mode: "lines",
    line: { color: colour, width: 2, shape: "hv" },
  }));

  traces.push({
    x: times,
    y: (payload.forecasts || {}).renewables || [],
    name: "Renewables forecast",
    mode: "lines",
    line: { color: COLORS.muted, width: 1.5, dash: "dot" },
    yaxis: "y2",
  });

  const layout = {
    ...LAYOUT,
    yaxis: { ...LAYOUT.yaxis, title: "Scheduled power (MW)" },
    yaxis2: {
      title: "Renewables (MW)",
      overlaying: "y",
      side: "right",
      showgrid: false,
      color: COLORS.muted,
    },
    xaxis: { ...LAYOUT.xaxis, title: "Denmark local time" },
    margin: { ...LAYOUT.margin, r: 60 },
  };

  Plotly.newPlot("dispatch-chart", traces, layout, CONFIG);
}

function renderScatter(payload) {
  const points = (payload.scatter3d || {}).points || [];
  if (!points.length) return;

  const trace = {
    x: points.map((p) => p.x),
    y: points.map((p) => p.y),
    z: points.map((p) => p.z),
    mode: "markers",
    type: "scatter3d",
    marker: {
      size: 4,
      color: points.map((p) => p.co2),
      colorscale: [
        [0, "#8fd6a8"],
        [0.5, "#f0c671"],
        [1, "#e07a5f"],
      ],
      colorbar: { title: { text: "g/kWh", side: "right" }, thickness: 12 },
      opacity: 0.85,
    },
    hovertemplate:
      "Plants ≥100 MW: %{x:.0f} MW<br>Renewables: %{y:.0f} MW<br>Net exchange: %{z:.0f} MW<extra></extra>",
  };

  const layout = {
    ...LAYOUT,
    margin: { l: 0, r: 0, t: 10, b: 0 },
    scene: {
      xaxis: { title: "Plants ≥100 MW", gridcolor: COLORS.grid, color: COLORS.text },
      yaxis: { title: "Renewables", gridcolor: COLORS.grid, color: COLORS.text },
      zaxis: { title: "Net exchange", gridcolor: COLORS.grid, color: COLORS.text },
    },
  };

  Plotly.newPlot("scatter-chart", [trace], layout, CONFIG);
}

/* --- allocation table ----------------------------------------------------- */

function renderAllocation(payload) {
  const table = payload.allocationTable || { columns: [], rows: [] };
  const head = document.getElementById("allocation-head");
  const body = document.getElementById("allocation-body");

  head.innerHTML = table.columns.map((c) => `<th>${c}</th>`).join("");

  const co2Start = table.columns.indexOf("CO2 Baseline");
  const forecastStart = table.columns.indexOf("Demand Forecast");

  body.innerHTML = table.rows
    .map((row, i) => {
      const shade = Math.min(4, Math.floor((i / Math.max(1, table.rows.length - 1)) * 4));
      const cells = row
        .map((value, j) => {
          let cls = "";
          if (co2Start >= 0 && j >= co2Start) cls = "col-co2";
          else if (forecastStart >= 0 && j >= forecastStart) cls = "col-green";
          return `<td class="${cls}">${value}</td>`;
        })
        .join("");
      return `<tr class="row-time-${shade} row-blue-${shade} row-green-${shade} row-co2-${shade}">${cells}</tr>`;
    })
    .join("");

  // Card layout for narrow screens.
  document.getElementById("allocation-cards").innerHTML = table.rows
    .map((row) => {
      const items = table.columns
        .slice(1)
        .map((c, j) => `<dt>${c}</dt><dd>${row[j + 1]}</dd>`)
        .join("");
      return `<div class="card"><h3>${row[0]}</h3><dl>${items}</dl></div>`;
    })
    .join("");
}

/* --- validation panel ----------------------------------------------------- */

function renderValidation(payload) {
  const v = payload.validation || {};
  const co2 = v.co2;
  const fc = v.forecast;
  const blocks = [];

  if (co2) {
    blocks.push(`
      <div class="val-card">
        <h3>Emissions model</h3>
        <dl>
          <dt>Intensity MAE, chronological hold-out</dt><dd>${fmt(co2.intensity_mae)} g/kWh</dd>
          <dt>Total-emissions R²</dt><dd>${fmt(co2.total_r2, 3)}</dd>
          <dt>Monotone in every supply source</dt><dd>${co2.probe_deployed && co2.probe_deployed.monotone_non_decreasing ? "yes" : "no"}</dd>
          <dt>Training rows</dt><dd>${co2.n_train ? co2.n_train.toLocaleString() : "—"}</dd>
        </dl>
      </div>`);
  }

  if (fc && fc.demand && fc.renewables) {
    const last = (m) => m.metrics[m.metrics.length - 1];
    const d = last(fc.demand);
    const r = last(fc.renewables);
    blocks.push(`
      <div class="val-card">
        <h3>Forecasts at 4 hours ahead</h3>
        <dl>
          <dt>Demand RMSE</dt><dd>${fmt(d.model_rmse, 0)} MW <span class="muted">vs ${fmt(d.persistence_rmse, 0)} persistence</span></dd>
          <dt>Renewables RMSE</dt><dd>${fmt(r.model_rmse, 0)} MW <span class="muted">vs ${fmt(r.persistence_rmse, 0)} persistence</span></dd>
          <dt>Raw Energinet forecast</dt><dd>${fmt(r.energinet_rmse, 0)} MW</dd>
        </dl>
      </div>`);
  }

  document.getElementById("validation").innerHTML = blocks.join("");
}

/* --- historical backtest -------------------------------------------------- */

function renderBacktest(bt) {
  if (!bt) return;

  const cards = [
    {
      label: "Origins replayed",
      value: bt.n_periods.toLocaleString(),
      unit: `${bt.folds} expanding folds`,
      note: `${bt.window[0]} → ${bt.window[1]}`,
    },
    {
      label: "Ranking slope",
      value: fmt(bt.ranking_slope, 2),
      unit: `r = ${fmt(bt.ranking_r, 2)}`,
      note: "predicted vs metered difference",
    },
    {
      label: "Pipeline claim",
      value: fmt(bt.pooled_reduction_vs_real_pct, 0),
      unit: "%",
      note: "vs the dispatch that actually ran",
    },
    {
      label: "After corrections",
      value: fmt(bt.capped_discounted_pct, 0),
      unit: "%",
      note: "historical floor + ranking discount",
      highlight: true,
    },
  ];

  document.getElementById("backtest-metrics").innerHTML = cards
    .map(
      (c) => `
      <div class="metric${c.highlight ? " metric-accent" : ""}">
        <div class="metric-label">${c.label}</div>
        <div class="metric-value">${c.value}<span class="metric-unit">${c.unit}</span></div>
        <div class="metric-note">${c.note}</div>
      </div>`
    )
    .join("");

  document.getElementById("backtest-detail").innerHTML = `
    <div class="val-card">
      <h3>What the model can do</h3>
      <dl>
        <dt>R² on realised dispatch</dt><dd>${fmt(bt.model_r2, 3)}</dd>
        <dt>Mean absolute error</dt><dd>${fmt(bt.model_mae_tph, 0)} t/h</dd>
        <dt>Matched pairs in the ranking test</dt><dd>${bt.ranking_n.toLocaleString()}</dd>
        <dt>Sign agreement</dt><dd>${fmt(bt.ranking_sign_agreement, 0)}%</dd>
        <dt>Slope along the optimiser's move</dt><dd>${fmt(bt.ranking_aligned_slope, 2)}</dd>
      </dl>
    </div>
    <div class="val-card">
      <h3>What the optimiser proposed</h3>
      <dl>
        <dt>Negative predicted emissions</dt><dd>${fmt(bt.negative_prediction_pct, 0)}%</dd>
        <dt>Steps inside the historical floor</dt><dd>${fmt(bt.within_floor_pct, 1)}%</dd>
        <dt>Share from less domestic generation</dt><dd>${fmt(bt.gen_share_pct, 0)}%</dd>
        <dt>Share from cross-border reallocation</dt><dd>${fmt(bt.xb_share_pct, 0)}%</dd>
        <dt>Steps still balanced under realised weather</dt><dd>${fmt(bt.steps_within_tolerance_pct, 0)}%</dd>
      </dl>
    </div>`;
}

/* --- boot ----------------------------------------------------------------- */

function renderStatus(payload) {
  if (payload.status === "error") {
    setStatus(
      "error",
      `Last update failed: ${payload.errorMessage || "unknown error"}. Showing the most recent successful result.`
    );
    return;
  }

  const updated = payload.lastUpdated ? new Date(payload.lastUpdated) : null;
  if (!updated) {
    setStatus("error", "No data available.");
    return;
  }

  const ageMinutes = (Date.now() - updated.getTime()) / 60000;
  const local = updated.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });

  if (ageMinutes > STALE_AFTER_MINUTES) {
    setStatus("stale", `Data last updated ${local} — ${Math.round(ageMinutes)} minutes ago, which is stale.`);
  } else {
    setStatus("ok", `Updated ${local} · refreshes every 30 minutes`);
  }
}

async function main() {
  try {
    const response = await fetch(`data/latest.json?t=${Date.now()}`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();

    renderStatus(payload);
    renderMetrics(payload);
    renderCo2Chart(payload);
    renderAllocation(payload);
    renderDispatchChart(payload);
    renderValidation(payload);
    renderScatter(payload);

    try {
      const backtestResponse = await fetch(`data/backtest.json?t=${Date.now()}`);
      if (backtestResponse.ok) renderBacktest(await backtestResponse.json());
    } catch (_) {
      /* the live schedule should still render if the static file is missing */
    }

    const meta = payload.meta || {};
    document.getElementById("footer-meta").textContent = meta.reductionBasis
      ? `Reduction measured against the ${meta.reductionBasis}.`
      : "";
  } catch (error) {
    setStatus("error", `Could not load data: ${error.message}`);
  }
}

main();
