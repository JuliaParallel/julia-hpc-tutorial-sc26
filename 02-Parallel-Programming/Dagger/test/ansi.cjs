// No Node dependencies: verify terminal colors survive the browser conversion.
const assert = require("node:assert/strict");
const {ansiSegments} = require("../host/ansi.js");

let runs = ansiSegments("plain \x1b[38;2;200;165;255mfunction\x1b[0m tail");
assert.equal(runs.map(r => r.text).join(""), "plain function tail");
assert.equal(runs[1].style.color, "rgb(200, 165, 255)");
assert.deepEqual(runs[2].style, {});
runs = ansiSegments("\x1b[1;3;38;5;196mred\x1b[22;23;39mnormal");
assert.deepEqual(runs[0].style, {fontWeight:"bold", fontStyle:"italic", color:"rgb(255, 0, 0)"});
assert.deepEqual(runs[1].style, {});
assert.equal(ansiSegments("\x1b[36mcyan")[0].style.color, "#6acde8");
assert.equal(ansiSegments("\x1b[38;5;244mgray")[0].style.color, "rgb(128, 128, 128)");
assert.equal(ansiSegments("<script>λ</script>")[0].text, "<script>λ</script>");

// Optional real Highlights output verifies the entire Julia→browser path.
if (process.argv[2]) {
  const {code, ansi} = JSON.parse(require("node:fs").readFileSync(process.argv[2], "utf8"));
  runs = ansiSegments(ansi);
  assert.equal(runs.map(r => r.text).join(""), code);
  assert.ok(new Set(runs.map(r => r.style.color).filter(Boolean)).size >= 6);
}
console.log("ANSI rendering checks passed");
