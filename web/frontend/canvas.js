// Small drawing/rendering helpers shared by the forward playground and the
// draw-a-target / result canvases. No dependencies -- plain canvas 2D API.

// A coarse approximation of matplotlib's viridis colormap (a handful of
// control points, linearly interpolated) so fields rendered here look like
// the same colormap used throughout the project's notebooks.
const VIRIDIS_STOPS = [
  [0.267, 0.005, 0.329], [0.283, 0.141, 0.458], [0.254, 0.265, 0.530],
  [0.207, 0.372, 0.553], [0.164, 0.471, 0.558], [0.128, 0.567, 0.551],
  [0.135, 0.659, 0.518], [0.267, 0.749, 0.441], [0.478, 0.821, 0.318],
  [0.741, 0.873, 0.150], [0.993, 0.906, 0.144],
];

export function viridis(t) {
  t = Math.max(0, Math.min(1, t));
  const n = VIRIDIS_STOPS.length - 1;
  const idx = Math.min(n - 1, Math.floor(t * n));
  const frac = t * n - idx;
  const a = VIRIDIS_STOPS[idx];
  const b = VIRIDIS_STOPS[idx + 1];
  const lerp = (i) => a[i] + (b[i] - a[i]) * frac;
  return [Math.round(lerp(0) * 255), Math.round(lerp(1) * 255), Math.round(lerp(2) * 255)];
}

// Draw a 2D array of values (any range) onto a canvas, normalized to
// [0,1] and colored with the viridis approximation above.
export function drawField(canvas, field2D, { vmin = null, vmax = null } = {}) {
  const H = field2D.length;
  const W = field2D[0].length;
  let lo = vmin, hi = vmax;
  if (lo === null || hi === null) {
    lo = Infinity; hi = -Infinity;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const v = field2D[y][x];
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  const span = hi - lo || 1;

  const off = document.createElement("canvas");
  off.width = W; off.height = H;
  const octx = off.getContext("2d");
  const img = octx.createImageData(W, H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const v = (field2D[y][x] - lo) / span;
      const [r, g, b] = viridis(v);
      const i = (y * W + x) * 4;
      img.data[i] = r; img.data[i + 1] = g; img.data[i + 2] = b; img.data[i + 3] = 255;
    }
  }
  octx.putImageData(img, 0, 0);

  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(off, 0, 0, canvas.width, canvas.height);
}

// Attach mouse/touch paint handlers to a canvas, painting filled white
// circles on a black background -- a rough stand-in for a hand-drawn
// pattern. Returns helpers to read the mask back out and clear it.
export function setupDrawableCanvas(canvas, { brushRadius = 10 } = {}) {
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  let drawing = false;

  function paintAt(x, y) {
    ctx.fillStyle = "white";
    ctx.beginPath();
    ctx.arc(x, y, brushRadius, 0, Math.PI * 2);
    ctx.fill();
  }

  function eventPos(e) {
    const rect = canvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return [
      (clientX - rect.left) * (canvas.width / rect.width),
      (clientY - rect.top) * (canvas.height / rect.height),
    ];
  }

  const start = (e) => { drawing = true; const [x, y] = eventPos(e); paintAt(x, y); e.preventDefault(); };
  const move = (e) => { if (!drawing) return; const [x, y] = eventPos(e); paintAt(x, y); e.preventDefault(); };
  const end = () => { drawing = false; };

  canvas.addEventListener("mousedown", start);
  canvas.addEventListener("mousemove", move);
  window.addEventListener("mouseup", end);
  canvas.addEventListener("touchstart", start, { passive: false });
  canvas.addEventListener("touchmove", move, { passive: false });
  canvas.addEventListener("touchend", end);

  return {
    clear() {
      ctx.fillStyle = "black";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    },
    getMaskArray() {
      const { width: W, height: H } = canvas;
      const img = ctx.getImageData(0, 0, W, H).data;
      const mask = [];
      for (let y = 0; y < H; y++) {
        const row = [];
        for (let x = 0; x < W; x++) {
          row.push(img[(y * W + x) * 4] / 255); // red channel as grayscale value
        }
        mask.push(row);
      }
      return mask;
    },
  };
}
