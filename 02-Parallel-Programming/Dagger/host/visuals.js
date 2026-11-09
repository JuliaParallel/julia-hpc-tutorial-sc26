// High-resolution canvas plots for the final exercise's Visualization panel.
// The harness sends omega/psi/u; this file turns them into real figures.

const VISUAL_COLORS = {
  coolwarm: [[59, 76, 192], [221, 221, 221], [180, 4, 38]],
  viridis: [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]],
};

function visualColor(name, t) {
  const stops = VISUAL_COLORS[name] || VISUAL_COLORS.viridis;
  t = Math.max(0, Math.min(1, t));
  const pos = t * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(pos)), f = pos - i;
  const a = stops[i], b = stops[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

function visualSample(rows, nx, ny, x, y) {
  const fx = x * (nx - 1), fy = y * (ny - 1);
  const i = Math.max(0, Math.min(nx - 2, Math.floor(fx)));
  const j = Math.max(0, Math.min(ny - 2, Math.floor(fy)));
  const tx = fx - i, ty = fy - j;
  const a = rows[i][j], b = rows[i + 1][j], c = rows[i][j + 1], d = rows[i + 1][j + 1];
  return (1 - tx) * (1 - ty) * a + tx * (1 - ty) * b + (1 - tx) * ty * c + tx * ty * d;
}

function visualResample(rows, N) {
  const nx = rows.length, ny = rows[0].length;
  return Array.from({length: N}, (_, i) =>
    Array.from({length: N}, (_, j) => visualSample(rows, nx, ny, i / (N - 1), j / (N - 1))));
}

function visualRange(rows, symmetric, percentile) {
  const values = [];
  for (const row of rows) for (const v of row) values.push(v);
  if (!symmetric) return [Math.min(...values), Math.max(...values)];
  const mags = values.map(Math.abs).sort((a, b) => a - b);
  const at = Math.min(mags.length - 1, Math.floor(percentile * (mags.length - 1)));
  return [-mags[at], mags[at]];
}

function visualNumber(v) {
  const a = Math.abs(v);
  if (a >= 10) return v.toFixed(0);
  if (a >= 1) return v.toFixed(1);
  if (a >= 0.01) return v.toFixed(2);
  return v.toExponential(1);
}

function visualCanvas(canvas, container, aspect, dpr) {
  const width = Math.min(760, Math.max(320, container.clientWidth - 8));
  const height = Math.round(width * aspect);
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  return {ctx, w: canvas.width, h: canvas.height, dpr};
}

function visualAxes(ctx, layout, dpr) {
  const {L, T, pw, ph} = layout;
  ctx.strokeStyle = "#2d4350";
  ctx.lineWidth = 1 * dpr;
  ctx.strokeRect(L, T, pw, ph);
  ctx.fillStyle = "#9fb4bf";
  ctx.font = `${11 * dpr}px ui-monospace, monospace`;
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (const t of [0, 0.5, 1]) {
    const x = L + t * pw;
    ctx.beginPath();
    ctx.moveTo(x, T + ph);
    ctx.lineTo(x, T + ph + 4 * dpr);
    ctx.stroke();
    ctx.fillText(String(t), x, T + ph + 7 * dpr);
  }
  ctx.fillText("x", L + pw / 2, T + ph + 27 * dpr);
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (const t of [0, 0.5, 1]) {
    const y = T + (1 - t) * ph;
    ctx.beginPath();
    ctx.moveTo(L, y);
    ctx.lineTo(L - 4 * dpr, y);
    ctx.stroke();
    ctx.fillText(String(t), L - 7 * dpr, y);
  }
  ctx.save();
  ctx.translate(10 * dpr, T + ph / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.fillText("y", 0, 0);
  ctx.restore();
}

function visualColorbar(ctx, layout, colormap, min, max, dpr) {
  const {T, ph, w} = layout;
  const x = w - 78 * dpr, bar = 14 * dpr;
  const grad = ctx.createLinearGradient(0, T, 0, T + ph);
  for (let i = 0; i <= 10; i++) {
    const t = i / 10;
    const [r, g, b] = visualColor(colormap, t);
    grad.addColorStop(t, `rgb(${r}, ${g}, ${b})`);
  }
  ctx.fillStyle = grad;
  ctx.fillRect(x, T, bar, ph);
  ctx.strokeStyle = "#2d4350";
  ctx.strokeRect(x, T, bar, ph);
  ctx.fillStyle = "#9fb4bf";
  ctx.font = `${10 * dpr}px ui-monospace, monospace`;
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  ctx.fillText(visualNumber(max), x + bar + 6 * dpr, T);
  ctx.fillText(visualNumber(min), x + bar + 6 * dpr, T + ph);
}

function visualField(canvas, container, rows, options) {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const {ctx, w, h} = visualCanvas(canvas, container, 0.72, dpr);
  const layout = {L: 46 * dpr, T: 12 * dpr, pw: 0, ph: 0, w, h};
  layout.pw = Math.max(1, Math.round(w - layout.L - 108 * dpr));
  layout.ph = Math.max(1, Math.round(h - layout.T - 42 * dpr));
  const nx = rows.length, ny = rows[0].length;
  const [min, max] = visualRange(rows, options.symmetric, options.percentile || 0.9);
  const span = max - min || 1;
  const image = ctx.createImageData(layout.pw, layout.ph);
  for (let py = 0; py < layout.ph; py++) {
    const y = 1 - py / (layout.ph - 1);
    for (let px = 0; px < layout.pw; px++) {
      const value = visualSample(rows, nx, ny, px / (layout.pw - 1), y);
      const [r, g, b] = visualColor(options.colormap, (value - min) / span);
      const k = 4 * (py * layout.pw + px);
      image.data[k] = r;
      image.data[k + 1] = g;
      image.data[k + 2] = b;
      image.data[k + 3] = 255;
    }
  }
  ctx.putImageData(image, layout.L, layout.T);
  if (options.contours) {
    const N = 192, fine = visualResample(rows, N);
    const [lo, hi] = options.contourRange || [min, max];
    const levels = options.levels || 8;
    ctx.strokeStyle = options.contourColor || "rgba(15, 25, 35, 0.55)";
    ctx.lineWidth = 1.2 * dpr;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.beginPath();
    for (let l = 1; l <= levels; l++) {
      const level = lo + (hi - lo) * l / (levels + 1);
      for (let i = 0; i < N - 1; i++) {
        for (let j = 0; j < N - 1; j++) {
          const a = fine[i][j], b = fine[i + 1][j], c = fine[i + 1][j + 1], d = fine[i][j + 1];
          const points = [];
          const edge = (v1, v2, x1, y1, x2, y2) => {
            if ((v1 - level) * (v2 - level) < 0) {
              const t = (level - v1) / (v2 - v1);
              points.push([x1 + (x2 - x1) * t, y1 + (y2 - y1) * t]);
            }
          };
          edge(a, b, i, j, i + 1, j);
          edge(b, c, i + 1, j, i + 1, j + 1);
          edge(d, c, i, j + 1, i + 1, j + 1);
          edge(a, d, i, j, i, j + 1);
          const at = p => [layout.L + p[0] / (N - 1) * layout.pw,
                           layout.T + (1 - p[1] / (N - 1)) * layout.ph];
          if (points.length === 2) {
            const p = at(points[0]), q = at(points[1]);
            ctx.moveTo(p[0], p[1]);
            ctx.lineTo(q[0], q[1]);
          } else if (points.length === 4) {
            const p = at(points[0]), q = at(points[1]), r = at(points[2]), s = at(points[3]);
            ctx.moveTo(p[0], p[1]); ctx.lineTo(q[0], q[1]);
            ctx.moveTo(r[0], r[1]); ctx.lineTo(s[0], s[1]);
          }
        }
      }
    }
    ctx.stroke();
  }
  visualAxes(ctx, layout, dpr);
  visualColorbar(ctx, layout, options.colormap, min, max, dpr);
}

function visualProfile(canvas, container, u) {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const {ctx, w, h} = visualCanvas(canvas, container, 0.58, dpr);
  const L = 46 * dpr, T = 12 * dpr, R = 24 * dpr, B = 42 * dpr;
  const pw = w - L - R, ph = h - T - B;
  const umin = Math.min(...u), umax = Math.max(...u);
  const pad = (umax - umin || 1) * 0.08;
  const x0 = umin - pad, x1 = umax + pad;
  const X = value => L + (value - x0) / (x1 - x0) * pw;
  ctx.strokeStyle = "#2d4350";
  ctx.lineWidth = 1 * dpr;
  ctx.strokeRect(L, T, pw, ph);
  ctx.strokeStyle = "#41525c";
  ctx.setLineDash([3 * dpr, 4 * dpr]);
  ctx.beginPath();
  ctx.moveTo(X(0), T);
  ctx.lineTo(X(0), T + ph);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.strokeStyle = "#73e4c3";
  ctx.lineWidth = 2 * dpr;
  ctx.beginPath();
  u.forEach((value, j) => {
    const x = X(value), y = T + (1 - j / (u.length - 1)) * ph;
    if (j === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.strokeStyle = "#2d4350";
  ctx.lineWidth = 1 * dpr;
  ctx.fillStyle = "#9fb4bf";
  ctx.font = `${11 * dpr}px ui-monospace, monospace`;
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (const value of [x0, 0, x1]) {
    if (value < x0 || value > x1) continue;
    ctx.fillText(visualNumber(value), X(value), T + ph + 7 * dpr);
  }
  ctx.fillText("u", L + pw / 2, T + ph + 27 * dpr);
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (const t of [0, 0.5, 1]) ctx.fillText(String(t), L - 7 * dpr, T + (1 - t) * ph);
  ctx.save();
  ctx.translate(10 * dpr, T + ph / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.fillText("y", 0, 0);
  ctx.restore();
}

function renderVisualization(container, data) {
  container.replaceChildren();
  if (!data) return;
  if (data.error) {
    const error = document.createElement("p");
    error.className = "visual-note";
    error.textContent = `Visualization failed: ${data.error}`;
    container.append(error);
    return;
  }
  const types = data.types || [];
  if (!types.length) {
    const empty = document.createElement("p");
    empty.className = "visual-note";
    empty.textContent = "No visualization selected.";
    container.append(empty);
    return;
  }
  for (const type of types) {
    const figure = document.createElement("figure");
    figure.className = "visual-card";
    const caption = document.createElement("figcaption");
    const canvas = document.createElement("canvas");
    figure.append(caption, canvas);
    container.append(figure);
    if (type === "vorticity") {
      caption.textContent = "Vorticity ω · blue negative, red positive";
      visualField(canvas, container, data.omega, {
        colormap: "coolwarm", symmetric: true, percentile: 0.9,
        contours: true, levels: 1, contourRange: [0, 0], contourColor: "rgba(20, 30, 40, 0.45)",
      });
    } else if (type === "streamfunction") {
      caption.textContent = "Streamfunction ψ · streamlines";
      visualField(canvas, container, data.psi, {
        colormap: "viridis", symmetric: false, contours: true, levels: 9,
      });
    } else if (type === "velocity") {
      caption.textContent = "u(y) at x = 0.5";
      visualProfile(canvas, container, data.u);
    }
  }
  if (data.vortex) {
    const note = document.createElement("p");
    note.className = "visual-note";
    note.textContent = `Primary vortex: ψ = ${data.vortex.psi} at ` +
      `x = ${data.vortex.x}, y = ${data.vortex.y}`;
    container.append(note);
  }
}
