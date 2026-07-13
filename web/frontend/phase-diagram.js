// Renders the classified (F,k) phase diagram (from notebook 00's cache) as
// a background, with the inverse-design optimizer's search trajectory
// drawn on top -- the abstract gradient-descent search made visible.

const REGIME_COLORS = {
  dead: "#1b2a4a",
  uniform: "#2fb8c9",
  spots: "#b06a4a",
  mazes: "#4a9d5c",
  stripes: "#c9b23a",
};

export class PhaseDiagramView {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.cache = null;
  }

  async load(baseUrl) {
    const res = await fetch(`${baseUrl}/phase-diagram`).catch(() => null);
    if (res && res.ok) {
      this.cache = await res.json();
    } else {
      // fall back to the static copy shipped alongside the frontend
      this.cache = await (await fetch("data/phase_diagram.json")).json();
    }
    this.drawBackground();
  }

  toPixel(F, k) {
    const { F_range, k_range } = this.cache;
    const x = ((F - F_range[0]) / (F_range[1] - F_range[0])) * this.canvas.width;
    const y = (1 - (k - k_range[0]) / (k_range[1] - k_range[0])) * this.canvas.height;
    return [x, y];
  }

  drawBackground() {
    const { ctx, canvas, cache } = this;
    const { F_values, k_values, labels } = cache;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const cellW = canvas.width / F_values.length;
    const cellH = canvas.height / k_values.length;
    for (let i = 0; i < F_values.length; i++) {
      for (let j = 0; j < k_values.length; j++) {
        const label = labels[i][j];
        ctx.fillStyle = REGIME_COLORS[label] || "#333";
        const x = i * cellW;
        const y = (k_values.length - 1 - j) * cellH;
        ctx.fillRect(x, y, cellW + 1, cellH + 1);
      }
    }
    this.drawLegend();
  }

  drawLegend() {
    const { ctx } = this;
    ctx.font = "10px sans-serif";
    let x = 4;
    for (const [label, color] of Object.entries(REGIME_COLORS)) {
      ctx.fillStyle = color;
      ctx.fillRect(x, 4, 10, 10);
      ctx.fillStyle = "white";
      ctx.fillText(label, x + 13, 13);
      x += ctx.measureText(label).width + 26;
    }
  }

  drawTrajectory(F_history, k_history, { finalMarker = true } = {}) {
    this.drawBackground();
    const { ctx } = this;
    ctx.strokeStyle = "#ffffffcc";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    F_history.forEach((F, i) => {
      const [x, y] = this.toPixel(F, k_history[i]);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();

    const [sx, sy] = this.toPixel(F_history[0], k_history[0]);
    ctx.fillStyle = "#ffffff";
    ctx.beginPath(); ctx.arc(sx, sy, 4, 0, Math.PI * 2); ctx.fill();

    if (finalMarker) {
      const [fx, fy] = this.toPixel(F_history.at(-1), k_history.at(-1));
      ctx.fillStyle = "#ff4d4d";
      ctx.beginPath(); ctx.arc(fx, fy, 5, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = "#000";
      ctx.stroke();
    }
  }
}
