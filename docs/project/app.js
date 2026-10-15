"use strict";
const loadControl = document.querySelector("#load");
const acceptanceControl = document.querySelector("#acceptance");
const chart = document.querySelector("#chart");
const resultRows = document.querySelector("#result-rows");
const sourceURL =
  "https://github.com/asp53826/vidur/blob/spec-decode/docs/assets/spec_decode_sweep/results.csv";
const names = {
  decode_shared_kv: "KV read once",
  decode_per_token: "Per-token cost",
};
const fmt = (n, digits = 2) => Number(n).toFixed(digits);
const escapeXML = (text) =>
  String(text).replace(
    /[&<>"']/g,
    (c) =>
      ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&apos;",
      })[c],
  );

function render(rows) {
  const qps = Number(loadControl.value);
  const alpha = Number(acceptanceControl.value);
  const base = rows.find((r) => r.qps === qps && r.k === 0);
  const selected = rows.filter((r) => r.qps === qps && r.alpha === alpha);
  const speed = (r) => base.tpot_p50_ms / r.tpot_p50_ms;
  const max = Math.max(2, Math.ceil(Math.max(...selected.map(speed))));
  const width = 680,
    height = 260,
    left = 82,
    right = 35,
    top = 28,
    bottom = 212;
  const x = (v) => left + (v / max) * (width - left - right);
  const positions = [65, 120, 175];
  let svg = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="chart-title chart-desc"><title id="chart-title">Predicted median TPOT speedup at ${qps} requests per second and acceptance ${alpha}</title><desc id="chart-desc">Two attention assumptions for each proposal count. Values above 1 are faster than baseline. The numerical results are also in the table below.</desc>`;
  for (let i = 0; i <= max; i++) {
    svg += `<line x1="${x(i)}" y1="${top}" x2="${x(i)}" y2="${bottom}" stroke="${i === 1 ? "#77869e" : "#e4ebf6"}" ${i === 1 ? 'stroke-dasharray="4 4"' : ""}/><text x="${x(i)}" y="235" font-family="Arial,sans-serif" font-size="12" fill="#4b607e" text-anchor="middle">${i}×</text>`;
  }
  [2, 4, 8].forEach((k, index) => {
    const shared = selected.find(
      (r) => r.k === k && r.verify_attention_model === "decode_shared_kv",
    );
    const perToken = selected.find(
      (r) => r.k === k && r.verify_attention_model === "decode_per_token",
    );
    const s = speed(shared),
      p = speed(perToken),
      y = positions[index];
    svg += `<text x="0" y="${y + 4}" font-family="Arial,sans-serif" font-size="13" fill="#172d4f">k = ${k}</text><line x1="${x(p)}" y1="${y}" x2="${x(s)}" y2="${y}" stroke="#c8d6ef" stroke-width="9" stroke-linecap="round"/>`;
    [
      [p, "#6958bc", -14],
      [s, "#2458cb", 25],
    ].forEach(([value, color, offset]) => {
      svg += `<circle cx="${x(value)}" cy="${y}" r="6" fill="${color}"/><text x="${x(value)}" y="${y + offset}" text-anchor="middle" fill="${color}" font-family="Arial,sans-serif" font-size="13" font-weight="600">${fmt(value)}×</text>`;
    });
  });
  chart.innerHTML = svg + "</svg>";
  const exampleShared = selected.find(
    (r) => r.k === 4 && r.verify_attention_model === "decode_shared_kv",
  );
  const examplePerToken = selected.find(
    (r) => r.k === 4 && r.verify_attention_model === "decode_per_token",
  );
  const s = speed(exampleShared),
    p = speed(examplePerToken);
  document.querySelector("#finding").innerHTML =
    `<strong>At k = 4: ${fmt(s)}× vs. ${fmt(p)}×.</strong> ${p < 1 && s > 1 ? "The two attention assumptions disagree on whether speculation helps." : "Both attention assumptions predict faster median TPOT, but disagree on how much."} The no-speculation baseline is ${fmt(base.tpot_p50_ms)} ms per output token.`;
  const note = document.querySelector("#load-note");
  note.classList.toggle("overload", qps === 6);
  note.textContent =
    qps === 6
      ? "Overload caveat: 6 requests/s exceeds the baseline replica’s capacity. These ratios depend on run length and show direction only."
      : "Setup: Splitwise conversation lengths, 300 requests, Sarathi, 2,048-token budget, one replica, tensor parallelism 1. Predictions from bundled A100 profiles.";
  resultRows.innerHTML = [base, ...selected]
    .map(
      (r) =>
        `<tr><th scope="row">${r.k === 0 ? "0 (baseline)" : r.k}</th><td>${escapeXML(names[r.verify_attention_model] || "No speculation")}</td><td>${fmt(r.tpot_p50_ms)} ms</td><td>${fmt(r.ttft_p50_ms)} ms</td><td>${fmt(r.e2e_p50_s)} s</td><td>${fmt(speed(r))}×</td></tr>`,
    )
    .join("");
}

fetch("results.json")
  .then((response) => {
    if (!response.ok) throw new Error("Results could not be loaded.");
    return response.json();
  })
  .then((rows) => {
    render(rows);
    loadControl.addEventListener("change", () => render(rows));
    acceptanceControl.addEventListener("change", () => render(rows));
  })
  .catch(() => {
    chart.innerHTML = `<p>The interactive results could not load. <a href="${sourceURL}">Open the complete results CSV</a> to inspect the recorded predictions.</p>`;
    loadControl.disabled = true;
    acceptanceControl.disabled = true;
  });
