"use strict";
const $ = id => document.getElementById(id);
let config, state = {}, cursor = 0, selectedDemo = null;
const panes = new Map();          // demo id -> console pane
const sectionPages = new Map();   // section id -> {el} shown in the main area
const paneSections = new Map();   // demo id -> section id
const runPanes = new Map();       // run id -> console pane
const runSources = new Map();     // run id -> executed sources
const COLLAPSE_KEY = "sc26-console-collapsed";
const ACTIVE_KEY = "sc26-console-active";
let collapsedState = {};
try {
  const saved = JSON.parse(localStorage.getItem(COLLAPSE_KEY) || "{}");
  // Older consoles stored a plain list of collapsed ids.
  collapsedState = Array.isArray(saved) ? Object.fromEntries(saved.map(id => [id, true])) : saved;
} catch (e) {}
function saveCollapsed() {
  try { localStorage.setItem(COLLAPSE_KEY, JSON.stringify(collapsedState)); } catch (e) {}
}

// Each pane is its own side-by-side Algorithm/REPL grid.
function syncWorkspaces() {
  for (const workspace of document.querySelectorAll(".workspace")) {
    // "10" = left pane folded, "01" = right pane folded; CSS drops that pane's height.
    workspace.dataset.collapsed = [...workspace.querySelectorAll(":scope > .panel")]
      .map(panel => panel.classList.contains("collapsed") ? "1" : "0").join("");
  }
}
function setCollapsed(section, value, save=true) {
  const button = section.querySelector(":scope > .panel-head .collapse,:scope > .side-head .collapse");
  if (!button) return;
  section.classList.toggle("collapsed", value);
  button.setAttribute("aria-expanded", String(!value));
  button.title = value ? "Expand section" : "Collapse section";
  collapsedState[section.id] = value;
  if (save) saveCollapsed();
  syncWorkspaces();
  // A hidden timeline has no measurable width, so redraw once it is visible.
  if (!value && section.classList.contains("timeline-panel")) {
    const pane = panes.get(section.dataset.pane);
    if (pane && pane.activeTrace) draw(pane);
  }
}
function initCollapsible() {
  for (const section of document.querySelectorAll(".panel,.side-section")) {
    const button = section.querySelector(":scope > .panel-head .collapse,:scope > .side-head .collapse");
    if (!button) continue;
    const toggle = () => setCollapsed(section, !section.classList.contains("collapsed"));
    button.onclick = toggle;
    // Clicking the heading is a bigger target; the chevron keeps focus semantics.
    const heading = section.querySelector(":scope > .panel-head h3,:scope > .side-head .eyebrow");
    if (heading) heading.onclick = event => {
      if (!event.target.closest("button,select,label,input")) toggle();
    };
    const stored = collapsedState[section.id];
    const startCollapsed = stored === undefined ? section.dataset.defaultCollapsed === "1" : stored;
    setCollapsed(section, startCollapsed, false);
  }
  syncWorkspaces();
}
function expandAncestors(element) {
  for (let node = element; node; node = node.parentElement) {
    if (node.classList?.contains("panel") && node.classList.contains("collapsed")) setCollapsed(node, false);
  }
}

function demoById(id) {
  return config.demos.find(demo => demo[0] === id);
}
function layoutLabel(layout) {
  return layout === "mpi" ? "TWO MPI RANKS / ONE ODO NODE" : "ONE JULIA PROCESS";
}

function paneMarkup(demo, heading = demo[1]) {
  const [id, , layout] = demo;
  const pid = `pane-${id}`;
  const gpus = id === "05_multigpu" ? `
<label class="gpu-control">GPUs <select data-role="gpus" aria-label="GPUs">
<option>2</option>
<option>4</option>
</select>
</label>` : "";
  const collapse = (target, label) => `<button class="collapse" type="button" aria-expanded="true" aria-controls="${target}" aria-labelledby="${label}" title="Collapse section">
<span class="chevron" aria-hidden="true">▾</span>
</button>`;
  return `
<section class="panel pane" id="${pid}" data-demo="${id}">
<div class="panel-head">
<h3 id="${pid}-title">${heading}</h3>
<div class="panel-tools">
<span class="tag">${layout === "mpi" ? "Two MPI ranks" : "One Julia process"}</span>${gpus}
<button class="primary" data-role="run" disabled>Run demo <span>↗</span></button>
${collapse(pid + "-body", pid + "-title")}
</div>
</div>
<div class="panel-body" id="${pid}-body">
<div class="workspace">
<section class="panel" id="${pid}-source">
<div class="panel-head">
<h3 id="${pid}-source-title">Algorithm</h3>
<div class="panel-tools">
<span class="tag" data-role="source-label">Selected demo</span>
${collapse(pid + "-source-body", pid + "-source-title")}
</div>
</div>
<div class="panel-body" id="${pid}-source-body">
<div class="source" data-role="source">
<p class="placeholder">Loading algorithm…</p>
</div>
</div>
</section>
<section class="panel" id="${pid}-repl">
<div class="panel-head">
<h3 id="${pid}-repl-title">Julia REPL</h3>
<div class="panel-tools">
<select data-role="mode" aria-label="REPL layout">
<option value="single">One process</option>
<option value="mpi">Two MPI ranks</option>
</select>
${collapse(pid + "-repl-body", pid + "-repl-title")}
</div>
</div>
<div class="panel-body" id="${pid}-repl-body">
<p class="hint">Variables persist in this layout. Switching layouts or cancelling resets Julia. MPI runs your expression on every rank; keep work uniform.</p>
<label class="sr-only" for="${pid}-code">Julia expression</label>
<textarea id="${pid}-code" data-role="code" spellcheck="false">a = Dagger.@spawn sum(1:1_000_000)
fetch(a)</textarea>
<div class="repl-footer">
<span>Ctrl / ⌘ + Enter to evaluate</span>
<button data-role="eval" disabled>Evaluate →</button>
</div>
</div>
</section>
</div>
<section class="panel timeline-panel" id="${pid}-timeline" data-pane="${id}">
<div class="panel-head">
<h3 id="${pid}-timeline-title">Task timeline</h3>
<div class="panel-tools">
<div class="plot-controls">
<label>
<input data-role="moves" type="checkbox" checked> Transfers</label>
<label>Zoom <input data-role="zoom" type="range" min="1" max="12" value="1" step="0.5">
</label>
<button data-role="download" disabled>Save JSON ↓</button>
</div>
${collapse(pid + "-timeline-body", pid + "-timeline-title")}
</div>
</div>
<div class="panel-body" id="${pid}-timeline-body">
<div class="metrics" data-role="metrics">
<span>Waiting for a logged run</span>
</div>
<div class="history" data-role="history" aria-label="Run history">
</div>
<div class="scroll-plot timeline" data-role="timeline">
<p class="placeholder">Compute tasks and data movement will appear on processor lanes.</p>
</div>
<div class="plot-key">
<span class="cyan">━ Compute</span>
<span class="amber">━ Data movement</span>
<span>Transfer arrows: source → destination rank</span>
<span>Hover a task for details · seconds from logging start</span>
</div>
<p class="hint">Dagger host task timings include compilation and transfer overhead. GPU lanes are not a kernel profile. Run again to see a warm session.</p>
<div class="summary" data-role="summary">
</div>
</div>
</section>
<section class="panel" id="${pid}-output">
<div class="panel-head">
<h3 id="${pid}-output-title">Session output</h3>
<div class="panel-tools">
<button data-role="clear">Clear display</button>
${collapse(pid + "-output-body", pid + "-output-title")}
</div>
</div>
<div class="panel-body" id="${pid}-output-body">
<pre class="output" data-role="output" aria-live="polite">Waiting for a run.
</pre>
</div>
</section>
</div>
</section>`;
}

function createPane(demo, heading) {
  const holder = document.createElement("div");
  holder.innerHTML = paneMarkup(demo, heading).trim();
  const el = holder.firstElementChild;
  const role = name => el.querySelector(`[data-role="${name}"]`);
  const pane = {
    id: demo[0], title: demo[1], layout: demo[2], el,
    source: role("source"), sourceLabel: role("source-label"),
    mode: role("mode"), code: role("code"), eval: role("eval"), run: role("run"),
    gpus: role("gpus"), output: role("output"), clear: role("clear"),
    metrics: role("metrics"), history: role("history"), timeline: role("timeline"),
    summary: role("summary"), moves: role("moves"), zoom: role("zoom"),
    download: role("download"), traces: [], activeTrace: null, version: 0,
  };
  pane.mode.value = demo[2];
  pane.run.onclick = () => post("/api/run", {op: "demo", demo: pane.id, gpus: pane.gpus?.value});
  pane.eval.onclick = () => post("/api/run", {op: "eval", code: pane.code.value, mode: pane.mode.value, demo: pane.id});
  pane.code.onkeydown = event => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.preventDefault();
      if (!pane.eval.disabled) pane.eval.click();
    }
  };
  pane.clear.onclick = () => pane.output.replaceChildren();
  pane.moves.onchange = pane.zoom.oninput = () => draw(pane);
  pane.download.onclick = () => {
    if (!pane.activeTrace) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(pane.activeTrace, null, 2)], {type: "application/json"}));
    const link = document.createElement("a");
    link.href = url;
    link.download = `dagger-${pane.activeTrace.run || "timeline"}.json`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  return pane;
}

function buildSidebar() {
  const nav = $("demos");
  for (const section of config.sections) {
    const demos = section.demos.map(demoById).filter(Boolean);
    if (demos.length === 1) {
      nav.append(navButton(section.number, demos[0]));
      continue;
    }
    const group = document.createElement("div");
    group.className = "nav-group";
    const head = document.createElement("button");
    head.type = "button";
    head.className = "nav-group-head";
    head.dataset.section = section.id;
    const number = document.createElement("span");
    number.className = "num";
    number.textContent = section.number;
    head.append(number, document.createTextNode(section.title));
    head.onclick = () => activate(section.demos[0]);
    const children = document.createElement("div");
    children.className = "nav-children";
    for (const demo of demos) children.append(navButton(null, demo));
    group.append(head, children);
    nav.append(group);
  }
}
function navButton(number, demo) {
  const button = document.createElement("button");
  button.type = "button";
  button.dataset.demo = demo[0];
  if (number !== null) {
    const span = document.createElement("span");
    span.className = "num";
    span.textContent = number;
    button.append(span);
  }
  button.append(document.createTextNode(demo[1]));
  button.onclick = () => activate(demo[0]);
  return button;
}

function buildConsoles() {
  for (const section of config.sections) {
    const demos = section.demos.map(demoById).filter(Boolean);
    if (demos.length === 1) {
      // The page heading already names the section; the console block just says Demo.
      const pane = createPane(demos[0], "Demo");
      panes.set(pane.id, pane);
      paneSections.set(pane.id, section.id);
      sectionPages.set(section.id, {el: pane.el});
      $("consoles").append(pane.el);
      continue;
    }
    const group = document.createElement("section");
    const gid = `section-${section.id}`;
    group.className = "panel console-group";
    group.id = gid;
    group.innerHTML = `
<div class="panel-head">
<h3 id="${gid}-title">Demo</h3>
<div class="panel-tools">
<span class="tag">${section.number} · ${demos.length} demos</span>
<button class="collapse" type="button" aria-expanded="true" aria-controls="${gid}-body" aria-labelledby="${gid}-title" title="Collapse section">
<span class="chevron" aria-hidden="true">▾</span>
</button>
</div>
</div>
<div class="panel-body" id="${gid}-body"></div>`;
    const body = group.querySelector(".panel-body");
    for (const demo of demos) {
      const pane = createPane(demo);
      panes.set(pane.id, pane);
      paneSections.set(pane.id, section.id);
      body.append(pane.el);
    }
    sectionPages.set(section.id, {el: group});
    $("consoles").append(group);
  }
}

// One section owns the main area, like a page; the sidebar chooses it.
let activeSection = null;
function showSection(sectionId) {
  for (const [id, page] of sectionPages) page.el.hidden = id !== sectionId;
  activeSection = sectionId;
}

function activate(demoId, {scroll = true, expand = true} = {}) {
  const pane = panes.get(demoId);
  if (!pane) return;
  const sectionId = paneSections.get(demoId);
  const changedPage = activeSection !== sectionId;
  showSection(sectionId);
  selectedDemo = demoId;
  try { localStorage.setItem(ACTIVE_KEY, demoId); } catch (e) {}
  if (expand) expandAncestors(pane.el);
  // A newly visible timeline had no measurable width while hidden.
  for (const other of panes.values()) {
    if (paneSections.get(other.id) === sectionId && other.activeTrace) draw(other);
  }
  if (scroll) {
    if (changedPage) window.scrollTo({top: 0, behavior: "smooth"});
    else pane.el.scrollIntoView({behavior: "smooth", block: "start"});
  }
  $("demo-title").textContent = pane.title;
  $("layout-label").textContent = layoutLabel(pane.layout);
  for (const button of document.querySelectorAll("#demos button")) {
    button.classList.toggle("active", button.dataset.demo === demoId);
  }
}

function alertError(message) {
  $("alert").textContent = message;
  $("alert").hidden = false;
}
async function post(path, body={}) {
  $("alert").hidden = true;
  try {
    const response = await fetch(path, {method: "POST", headers: {"Content-Type": "application/json", "X-Console-Token": config.token}, body: JSON.stringify(body)});
    const data = await response.json();
    if (!response.ok) throw Error(data.error);
  } catch (error) {
    alertError(error.message);
  }
}
function ansi(parent, text) {
  for (const run of ansiSegments(text)) {
    const span = document.createElement("span");
    span.textContent = run.text;
    Object.assign(span.style, run.style);
    parent.append(span);
  }
}
function showSources(pane, sources, label="Selected demo") {
  pane.sourceLabel.textContent = label;
  pane.source.replaceChildren();
  for (const source of sources) {
    const heading = document.createElement("h4"), pre = document.createElement("pre");
    heading.textContent = source.file;
    ansi(pre, source.ansi || source.code);
    pane.source.append(heading, pre);
  }
}
async function loadSource(pane) {
  const version = ++pane.version;
  pane.source.textContent = "Loading algorithm…";
  try {
    const response = await fetch(`/api/source?demo=${encodeURIComponent(pane.id)}`);
    const data = await response.json();
    if (!response.ok) throw Error(data.error);
    if (version === pane.version) showSources(pane, data.sources);
  } catch (error) {
    if (version === pane.version) pane.source.textContent = error.message;
  }
}
function appendOutput(pane, text) {
  const follow = pane.output.scrollTop + pane.output.clientHeight >= pane.output.scrollHeight - 30;
  ansi(pane.output, text);
  if (follow) pane.output.scrollTop = pane.output.scrollHeight;
  while (pane.output.childNodes.length > 6000) pane.output.firstChild.remove();
}
function controls() {
  $("status").textContent = state.status;
  $("connect").disabled = state.connecting || state.connected;
  $("disconnect").disabled = !state.connected && !state.connecting;
  $("cancel").disabled = !state.busy;
  const ready = Boolean(state.job) && !state.busy;
  for (const pane of panes.values()) pane.run.disabled = pane.eval.disabled = !ready;
}
// Route an event to the console that produced it; system chatter follows the selection.
function paneFor(item) {
  if (item.demo && panes.has(item.demo)) return panes.get(item.demo);
  if (item.run && runPanes.has(item.run)) return runPanes.get(item.run);
  return panes.get(selectedDemo) || panes.values().next().value;
}
function event(item) {
  if (item.kind === "output") {
    const pane = (item.run && runPanes.get(item.run)) || panes.get(selectedDemo) || panes.values().next().value;
    if (pane) appendOutput(pane, item.text);
  } else if (item.kind === "run") {
    const pane = paneFor(item);
    if (!pane) return;
    runPanes.set(item.run, pane);
    runSources.set(item.run, []);
    appendOutput(pane, `\n━━ ${item.label} ━━\n`);
  } else if (item.kind === "source") {
    const pane = paneFor(item);
    if (!pane) return;
    const sources = runSources.get(item.run) || [];
    sources.push(item);
    runSources.set(item.run, sources);
    ++pane.version;
    showSources(pane, sources, "Executed source");
  } else if (item.kind === "timeline") {
    const pane = paneFor(item);
    if (!pane) return;
    const trace = normalizeTimeline(item);
    pane.traces.push(trace);
    if (pane.traces.length > 20) pane.traces.shift();
    pane.activeTrace = trace;
    draw(pane);
  } else if (item.kind === "error" || (item.kind === "done" && !item.ok)) {
    alertError(item.error);
  }
}
function svg(tag, attrs={}, text) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attrs)) element.setAttribute(key, value);
  if (text !== undefined) element.textContent = text;
  return element;
}
function draw(pane) {
  $("task-tooltip").hidden = true;
  if (!pane.activeTrace) return;
  const trace = pane.activeTrace, events = trace.events.filter(e => pane.moves.checked || e.category === "compute");
  pane.download.disabled = false;
  pane.history.replaceChildren();
  pane.traces.forEach((t, i) => {
    const button = document.createElement("button");
    button.textContent = `${i + 1} · ${t.title}`;
    button.classList.toggle("active", t === trace);
    button.onclick = () => {
      pane.activeTrace = t;
      if (t.sources?.length) showSources(pane, t.sources, "Executed source");
      draw(pane);
    };
    pane.history.append(button);
  });
  const all = trace.events, lanes = [...new Set(events.map(e => e.lane))].sort();
  const duration = Math.max(0.000001, ...all.map(e => e.start + e.duration));
  const compute = all.filter(e => e.category === "compute");
  pane.metrics.replaceChildren();
  for (const [value, label] of [[duration.toFixed(3), "seconds"], [compute.length, "compute tasks"], [new Set(all.map(e => e.rank)).size, "ranks"], [lanes.length, "lanes"]]) {
    const span = document.createElement("span"), strong = document.createElement("strong");
    strong.textContent = value;
    span.append(strong, document.createTextNode(label));
    pane.metrics.append(span);
  }
  const left = 235, right = 25, row = 33, top = 35;
  const width = Math.max(700, pane.timeline.clientWidth) * Number(pane.zoom.value), height = top + row * lanes.length + 38;
  const plot = svg("svg", {width, height, role: "img", "aria-label": `${trace.title} task timeline`}), x = t => left + t / duration * (width - left - right);
  for (let i = 0; i <= 8; i++) {
    const t = i * duration / 8, at = x(t);
    plot.append(svg("line", {x1: at, x2: at, y1: top - 10, y2: height - 28, stroke: "#2d4350"}));
    plot.append(svg("text", {x: at, y: height - 8, "text-anchor": "middle"}, t.toFixed(duration < 1 ? 3 : 2)));
  }
  lanes.forEach((lane, i) => {
    const y = top + i * row;
    const label = svg("text", {x: 5, y: y + 14}, lane.length > 33 ? lane.slice(0, 32) + "…" : lane);
    label.append(svg("title", {}, lane));
    plot.append(label);
    plot.append(svg("line", {x1: left, x2: width - right, y1: y + 10, y2: y + 10, stroke: "#203541"}));
    events.filter(e => e.lane === lane).sort((a, b) => b.duration - a.duration).forEach(e => {
      const rect = svg("rect", {x: x(e.start), y: y + (e.category === "move" ? 15 : 1), width: Math.max(2, x(e.start + e.duration) - x(e.start)), height: e.category === "move" ? 5 : 12, rx: 2, fill: e.category === "compute" ? "#6acde8" : "#ffbd75"});
      const details = taskDetails(e);
      rect.setAttribute("tabindex", "0");
      rect.setAttribute("role", "img");
      rect.setAttribute("aria-label", `${e.name}\n${details}`);
      const show = point => {
        const tip = $("task-tooltip"), name = document.createElement("strong"), body = document.createElement("div");
        name.textContent = e.name;
        body.textContent = details;
        tip.replaceChildren(name, body);
        tip.hidden = false;
        tip.style.left = `${Math.max(8, Math.min(point.clientX + 14, window.innerWidth - tip.offsetWidth - 8))}px`;
        tip.style.top = `${Math.max(8, Math.min(point.clientY + 14, window.innerHeight - tip.offsetHeight - 8))}px`;
      };
      rect.onpointerenter = rect.onpointermove = show;
      rect.onfocus = () => {
        const box = rect.getBoundingClientRect();
        show({clientX: box.x, clientY: box.bottom});
      };
      rect.onpointerleave = rect.onblur = () => $("task-tooltip").hidden = true;
      plot.append(rect);
    });
  });
  pane.timeline.replaceChildren(plot);
  const totals = new Map();
  compute.forEach(e => totals.set(e.name, (totals.get(e.name) || 0) + e.duration));
  const ordered = [...totals].sort((a, b) => b[1] - a[1]).slice(0, 8), max = Math.max(1e-9, ...ordered.map(entry => entry[1]));
  const heading = document.createElement("h4");
  heading.textContent = "Compute time by function · summed across lanes, may overlap";
  pane.summary.replaceChildren(heading);
  ordered.forEach(([name, value]) => {
    const rowEl = document.createElement("div"), label = document.createElement("span"), track = document.createElement("div"),
          bar = document.createElement("div"), amount = document.createElement("span");
    rowEl.className = "bar-row";
    label.className = "bar-name";
    label.textContent = name;
    label.title = name;
    track.className = "bar-track";
    bar.className = "bar-fill";
    bar.style.width = `${100 * value / max}%`;
    amount.textContent = `${(value * 1000).toFixed(2)} ms`;
    track.append(bar);
    rowEl.append(label, track, amount);
    pane.summary.append(rowEl);
  });
}
async function poll() {
  try {
    const initial = cursor === 0;
    const response = await fetch(`/api/state?after=${cursor}`);
    state = await response.json();
    state.events.forEach(event);
    cursor = state.seq;
    controls();
    if (initial) for (const pane of panes.values()) pane.output.scrollTop = pane.output.scrollHeight;
  } catch (error) {
    $("status").textContent = "Local server unavailable";
  }
  setTimeout(poll, 700);
}
async function init() {
  config = await(await fetch("/api/config")).json();
  $("remote").textContent = `${config.host}:${config.remote_root}`;
  buildSidebar();
  buildConsoles();
  initCollapsible();
  let start = config.sections[0].demos[0];
  try {
    const saved = localStorage.getItem(ACTIVE_KEY);
    if (saved && panes.has(saved)) start = saved;
  } catch (e) {}
  activate(start, {scroll: false, expand: false});
  for (const pane of panes.values()) loadSource(pane);
  $("connect").onclick = () => post("/api/connect");
  $("disconnect").onclick = () => post("/api/disconnect");
  $("cancel").onclick = () => post("/api/cancel");
  window.addEventListener("resize", () => {
    for (const pane of panes.values()) if (pane.activeTrace) draw(pane);
  });
  poll();
}
init().catch(error => alertError(error.message));
