// Decode terminal SGR styles without interpreting HTML or terminal commands.
// Used for both Julia source and session output.
function ansiSegments(text) {
  const palette = ["#9fb4bf", "#ff8f8f", "#72e3c3", "#ffbd75", "#6acde8", "#c8a5ff", "#6acde8", "#e7f0f4",
                   "#8197a3", "#ff8f8f", "#72e3c3", "#ffbd75", "#6acde8", "#c8a5ff", "#6acde8", "#ffffff"];
  function indexed(n) {
    if (n < 16) return palette[n];
    if (n >= 232) { const v = 8 + (n - 232) * 10; return `rgb(${v}, ${v}, ${v})`; }
    const levels = [0, 95, 135, 175, 215, 255], v = n - 16;
    return `rgb(${levels[Math.floor(v / 36)]}, ${levels[Math.floor(v / 6) % 6]}, ${levels[v % 6]})`;
  }
  const runs = [], regex = /\x1b\[([0-9;]*)m/g;
  let style = {}, start = 0;
  function append(value) { if (value) runs.push({text: value, style: {...style}}); }
  for (const match of text.matchAll(regex)) {
    append(text.slice(start, match.index));
    const codes = match[1].split(";").map(Number);
    for (let i = 0; i < codes.length; i++) {
      const code = codes[i];
      if (code === 0) style = {};
      else if (code === 1) style.fontWeight = "bold";
      else if (code === 3) style.fontStyle = "italic";
      else if (code === 4) style.textDecoration = "underline";
      else if (code === 22) delete style.fontWeight;
      else if (code === 23) delete style.fontStyle;
      else if (code === 24) delete style.textDecoration;
      else if (code === 39) delete style.color;
      else if (code === 49) delete style.backgroundColor;
      else if ((code >= 30 && code <= 37) || (code >= 90 && code <= 97)) {
        style.color = palette[code >= 90 ? code - 90 + 8 : code - 30];
      } else if (code === 38 || code === 48) {
        const property = code === 38 ? "color" : "backgroundColor";
        if (codes[i + 1] === 2 && i + 4 < codes.length) {
          const rgb = codes.slice(i + 2, i + 5);
          if (rgb.every(v => v >= 0 && v <= 255)) style[property] = `rgb(${rgb.join(", ")})`;
          i += 4;
        } else if (codes[i + 1] === 5 && i + 2 < codes.length) {
          const n = codes[i + 2];
          if (n >= 0 && n <= 255) style[property] = indexed(n);
          i += 2;
        }
      }
    }
    start = match.index + match[0].length;
  }
  append(text.slice(start));
  return runs;
}

if (typeof module !== "undefined") module.exports = {ansiSegments};
