import { drawField, setupDrawableCanvas } from "./canvas.js";
import { PhaseDiagramView } from "./phase-diagram.js";

const GRID = 64;
const BACKEND_URL = window.GRAYDIFF_BACKEND_URL || "http://127.0.0.1:8001";

// ---------- standard seed: one centered blob + a small fixed perturbation,
// matching graydiff.solver.standard_seed's recipe closely enough for a
// live demo (exact bit-for-bit match isn't necessary here). ----------
function standardSeed() {
  const U = new Float32Array(GRID * GRID).fill(1);
  const V = new Float32Array(GRID * GRID).fill(0);
  const r = Math.max(Math.floor(GRID / 10), 2);
  const c = Math.floor(GRID / 2);
  for (let dy = -r; dy < r; dy++) {
    for (let dx = -r; dx < r; dx++) {
      const y = ((c + dy) % GRID + GRID) % GRID;
      const x = ((c + dx) % GRID + GRID) % GRID;
      U[y * GRID + x] = 0.5;
      V[y * GRID + x] = 0.25;
    }
  }
  for (let i = 0; i < U.length; i++) {
    U[i] = Math.min(1, Math.max(0, U[i] + (Math.random() - 0.5) * 0.2));
    V[i] = Math.min(1, Math.max(0, V[i] + (Math.random() - 0.5) * 0.2));
  }
  return { U, V };
}

function to2D(flat) {
  const rows = [];
  for (let y = 0; y < GRID; y++) rows.push(Array.from(flat.slice(y * GRID, (y + 1) * GRID)));
  return rows;
}

// ---------------------------------------------------------------- playground
async function initPlayground() {
  const statusEl = document.getElementById("playground-status");
  const canvas = document.getElementById("playground-canvas");
  const pfSlider = document.getElementById("pf-slider");
  const pkSlider = document.getElementById("pk-slider");
  const pfVal = document.getElementById("pf-val");
  const pkVal = document.getElementById("pk-val");
  const playBtn = document.getElementById("play-btn");
  const pauseBtn = document.getElementById("pause-btn");
  const resetBtn = document.getElementById("reset-btn");

  let session;
  try {
    session = await ort.InferenceSession.create("data/surrogate.onnx");
  } catch (err) {
    statusEl.textContent = "Failed to load model (see console). Serve this directory over HTTP, not file://.";
    console.error(err);
    return;
  }
  statusEl.textContent = "Model loaded. Click the grid to paint a seed, then Play.";

  let { U, V } = standardSeed();
  let playing = false;

  function render() {
    drawField(canvas, to2D(V), { vmin: 0, vmax: 1 });
  }
  render();

  async function step() {
    const F = parseFloat(pfSlider.value);
    const k = parseFloat(pkSlider.value);
    const Fgrid = new Float32Array(GRID * GRID).fill(F);
    const kgrid = new Float32Array(GRID * GRID).fill(k);
    const input = new Float32Array(4 * GRID * GRID);
    input.set(U, 0);
    input.set(V, GRID * GRID);
    input.set(Fgrid, 2 * GRID * GRID);
    input.set(kgrid, 3 * GRID * GRID);

    const tensor = new ort.Tensor("float32", input, [1, 4, GRID, GRID]);
    const out = await session.run({ input: tensor });
    const outData = out.output.data;
    U = outData.slice(0, GRID * GRID);
    V = outData.slice(GRID * GRID, 2 * GRID * GRID);
    render();
  }

  async function loop() {
    if (!playing) return;
    await step();
    requestAnimationFrame(loop);
  }

  playBtn.addEventListener("click", () => {
    playing = true;
    playBtn.disabled = true;
    pauseBtn.disabled = false;
    loop();
  });
  pauseBtn.addEventListener("click", () => {
    playing = false;
    playBtn.disabled = false;
    pauseBtn.disabled = true;
  });
  resetBtn.addEventListener("click", () => {
    ({ U, V } = standardSeed());
    render();
  });

  canvas.addEventListener("mousedown", (e) => {
    const rect = canvas.getBoundingClientRect();
    const gx = Math.floor(((e.clientX - rect.left) / rect.width) * GRID);
    const gy = Math.floor(((e.clientY - rect.top) / rect.height) * GRID);
    const rad = 3;
    for (let dy = -rad; dy <= rad; dy++) {
      for (let dx = -rad; dx <= rad; dx++) {
        const y = ((gy + dy) % GRID + GRID) % GRID;
        const x = ((gx + dx) % GRID + GRID) % GRID;
        U[y * GRID + x] = 0.5;
        V[y * GRID + x] = 0.25;
      }
    }
    render();
  });

  pfSlider.addEventListener("input", () => { pfVal.textContent = pfSlider.value; });
  pkSlider.addEventListener("input", () => { pkVal.textContent = pkSlider.value; });

  // Presets: load precomputed target/(F,k) pairs so a passive visitor sees
  // results without drawing.
  try {
    const presets = await (await fetch("data/presets.json")).json();
    const wrap = document.getElementById("preset-buttons");
    for (const p of presets.presets) {
      const btn = document.createElement("button");
      btn.className = "secondary";
      btn.textContent = p.name;
      btn.addEventListener("click", () => {
        pfSlider.value = p.F; pkSlider.value = p.k;
        pfVal.textContent = p.F.toFixed(3); pkVal.textContent = p.k.toFixed(3);
        ({ U, V } = standardSeed());
        render();
      });
      wrap.appendChild(btn);
    }
  } catch (err) {
    console.warn("Could not load presets.json", err);
  }
}

// ------------------------------------------------------------- loss chart
function drawLossChart(canvas, lossHistory) {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!lossHistory.length) return;
  const lo = Math.min(...lossHistory);
  const hi = Math.max(...lossHistory);
  const span = hi - lo || 1;
  ctx.strokeStyle = "#5ec8f8";
  ctx.lineWidth = 2;
  ctx.beginPath();
  lossHistory.forEach((v, i) => {
    const x = (i / (lossHistory.length - 1 || 1)) * canvas.width;
    const y = canvas.height - ((v - lo) / span) * canvas.height;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.fillStyle = "#9098ab";
  ctx.font = "11px sans-serif";
  ctx.fillText("loss over optimization iterations", 6, 14);
}

// -------------------------------------------------------------- inverse UI
function initInverseDesign() {
  const drawCanvas = document.getElementById("draw-canvas");
  const drawable = setupDrawableCanvas(drawCanvas, { brushRadius: 12 });
  document.getElementById("clear-draw-btn").addEventListener("click", () => drawable.clear());

  const phaseView = new PhaseDiagramView(document.getElementById("phase-diagram-canvas"));
  phaseView.load(BACKEND_URL).catch((err) => console.warn("phase diagram load failed", err));

  const statusEl = document.getElementById("solve-status");
  const solveBtn = document.getElementById("solve-btn");
  const resultPanel = document.getElementById("result-panel");

  solveBtn.addEventListener("click", async () => {
    const mask = drawable.getMaskArray();
    statusEl.textContent = "Solving… (grid search + gradient descent, this can take 1-2 minutes)";
    solveBtn.disabled = true;
    try {
      const res = await fetch(`${BACKEND_URL}/inverse-design`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mask, n_starts: 4, n_iters: 150 }),
      });
      if (!res.ok) throw new Error(`backend returned ${res.status}`);
      const result = await res.json();

      statusEl.textContent = `Solved in ${result.elapsed_seconds.toFixed(1)}s. ` +
        `Recovered F=${result.F.toFixed(4)}, k=${result.k.toFixed(4)} -> real solver regime: ${result.regime}`;

      phaseView.drawTrajectory(result.F_history, result.k_history);
      drawLossChart(document.getElementById("loss-chart-canvas"), result.loss_history);

      drawField(document.getElementById("result-target"), invertMaskToField(mask), { vmin: 0, vmax: 1 });
      drawField(document.getElementById("result-surrogate"), result.surrogate_field, { vmin: 0, vmax: 1 });
      drawField(document.getElementById("result-solver"), result.solver_field, { vmin: 0, vmax: 1 });
      document.getElementById("result-summary").textContent =
        `F=${result.F.toFixed(4)}, k=${result.k.toFixed(4)} — the real NumPy solver, run independently ` +
        `at these recovered parameters, produces a pattern classified as "${result.regime}".`;
      resultPanel.classList.remove("hidden");
    } catch (err) {
      console.error(err);
      statusEl.textContent = `Failed to reach the backend at ${BACKEND_URL} (is it running? see web/backend/README). ${err}`;
    } finally {
      solveBtn.disabled = false;
    }
  });
}

function invertMaskToField(mask) {
  // downsample the drawn mask to a small preview grid for display only
  return mask;
}

initPlayground();
initInverseDesign();
