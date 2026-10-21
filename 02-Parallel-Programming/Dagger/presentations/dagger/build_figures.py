"""Rebuild the deck's SVGs with Python's standard library.

Data: passing odo-fixed rehearsal trials and the saved eight-GPU task trace,
recorded October 14, 2026. Data extracts omit usernames and machine names.
"""

import csv
import math
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "assets"
MINT, VIOLET, AMBER = "#73e2bd", "#b8a0f5", "#ffc184"
WHITE, MUTED, LINE, PANEL = "#edf3f5", "#adbdc9", "#415b6b", "#182530"


class SVG:
    def __init__(self, name, width, height, title):
        self.name = name
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'role="img" aria-label="{escape(title)}">',
            f"<title>{escape(title)}</title>",
            '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
            f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M0 0 L10 5 L0 10" fill="{MUTED}"/></marker></defs>',
        ]

    def text(self, x, y, text, size=24, color=WHITE, anchor="start", weight=400):
        self.parts.append(
            f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" '
            f'font-family="DejaVu Sans, sans-serif" font-weight="{weight}" '
            f'text-anchor="{anchor}">{escape(str(text))}</text>'
        )

    def rect(self, x, y, w, h, fill=PANEL, stroke=LINE, radius=14):
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
        )

    def line(self, x1, y1, x2, y2, color=LINE, arrow=False, width=2, both=False):
        marker = ' marker-end="url(#arrow)"' if arrow else ""
        if both:
            marker += ' marker-start="url(#arrow)"'
        self.parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{color}" stroke-width="{width}"{marker}/>'
        )

    def box(self, x, y, w, h, label, color=MINT, sub=None, size=25):
        self.rect(x, y, w, h, stroke=color)
        self.text(x + w / 2, y + h / 2 + (8 if sub is None else -1), label,
                  size, color, "middle", 600)
        if sub:
            self.text(x + w / 2, y + h / 2 + 30, sub, 18, MUTED, "middle")

    def save(self):
        (OUT / self.name).write_text("\n".join(self.parts + ["</svg>"]) + "\n")


def diagrams():
    s = SVG("model.svg", 1120, 390, "Describe tasks and access; Dagger schedules across processors")
    s.text(30, 35, "YOUR PROGRAM", 17, MUTED, weight=600)
    s.box(30, 80, 265, 90, "Function calls", sub="tasks")
    s.box(30, 220, 265, 90, "Data access", VIOLET, "reads + writes")
    s.line(300, 125, 420, 185, arrow=True)
    s.line(300, 265, 420, 205, arrow=True)
    s.box(430, 135, 250, 120, "Dagger", MINT, "schedule + move data", 32)
    s.text(820, 35, "AVAILABLE PROCESSORS", 17, MUTED, weight=600)
    for i, (label, sub) in enumerate([
        ("CPU threads", "within a process"), ("GPU devices", "one or more per node"),
        ("MPI ranks", "within one node"),
    ]):
        y = 65 + 110 * i
        s.line(686, 195, 805, y + 44, arrow=True)
        s.box(815, y, 275, 88, label, [MINT, VIOLET, AMBER][i], sub)
    s.save()

    s = SVG("tasks.svg", 420, 390, "Two independent producers feed multiply, then sum")
    s.box(15, 20, 155, 80, "a", MINT, "rand")
    s.box(245, 20, 155, 80, "b", MINT, "rand")
    s.line(92, 105, 183, 170, arrow=True)
    s.line(322, 105, 232, 170, arrow=True)
    s.box(120, 180, 175, 75, "c = a × b", VIOLET)
    s.line(207, 262, 207, 305, arrow=True)
    s.box(120, 315, 175, 65, "d = sum(c)", AMBER, size=23)
    s.save()

    s = SVG("mutation.svg", 1100, 165, "A write of B precedes the next read of B")
    s.box(45, 25, 310, 95, "B ← B + A", MINT, "reads A, updates B")
    s.line(365, 72, 725, 72, arrow=True)
    s.text(550, 55, "B must be ready", 21, MUTED, "middle")
    s.box(740, 25, 310, 95, "C ← B", VIOLET, "reads B, overwrites C")
    s.save()

    s = SVG("tiles.svg", 430, 390, "4096 square matrix divided into sixteen 1024 square tiles")
    for i in range(4):
        for j in range(4):
            color = [MINT, VIOLET, AMBER, MINT][(i + j) % 4]
            s.rect(64 + j * 77, 35 + i * 77, 67, 67, fill=PANEL, stroke=color, radius=8)
    s.text(210, 380, "16 tiles · independent updates", 20, MUTED, "middle")
    s.text(210, 20, "4096", 19, MUTED, "middle")
    s.save()

    s = SVG("topology.svg", 1120, 350, "One Julia process splits one matrix product across two GPUs on one node")
    s.rect(15, 15, 1090, 295, fill=PANEL)
    s.text(45, 50, "ONE SERVER / ONE JULIA PROCESS / TWO GPU DEVICES", 19, MUTED, weight=600)
    s.box(80, 80, 180, 60, "A₁", MINT)
    s.box(470, 80, 180, 60, "Shared B", AMBER, size=23)
    s.box(860, 80, 180, 60, "A₂", MINT)
    s.line(170, 145, 240, 195, arrow=True)
    s.line(950, 145, 880, 195, arrow=True)
    s.line(530, 145, 360, 195, arrow=True)
    s.line(590, 145, 760, 195, arrow=True)
    s.box(85, 205, 360, 85, "GPU 1", VIOLET, "C₁ = A₁ × B", 28)
    s.box(675, 205, 360, 85, "GPU 2", VIOLET, "C₂ = A₂ × B", 28)
    s.text(560, 340, "One output matrix: C = [ C₁ ; C₂ ] = A × B", 23, WHITE, "middle")
    s.save()

    s = SVG("multinode.svg", 1120, 260, "Two servers execute one Dagger program through MPI ranks on CPU threads")
    for x, host, rank in [(20, "SERVER A", 0), (670, "SERVER B", 1)]:
        s.rect(x, 15, 430, 200, fill=PANEL)
        s.text(x + 25, 55, host, 20, MUTED, weight=600)
        s.box(x + 40, 80, 350, 100, f"MPI rank {rank}", MINT, "CPU threads", 28)
    s.line(460, 130, 660, 130, arrow=True, both=True)
    s.text(560, 105, "network", 21, AMBER, "middle")
    s.text(560, 250, "Every rank builds the same task graph", 23, WHITE, "middle")
    s.save()

    s = SVG("multinode-gpu.svg", 1120, 350, "MPI connects two servers, each running tasks on two GPU devices")
    for x, host, rank in [(20, "SERVER A", 0), (670, "SERVER B", 1)]:
        s.rect(x, 15, 430, 285, fill=PANEL)
        s.text(x + 25, 55, host, 20, MUTED, weight=600)
        s.box(x + 40, 80, 350, 65, f"MPI rank {rank}", MINT, size=26)
        for i in range(2):
            gx = x + 30 + 195 * i
            s.line(x + 215, 150, gx + 87, 195, arrow=True)
            s.box(gx, 205, 175, 70, f"GPU {i + 1}", VIOLET, size=26)
    s.line(460, 112, 660, 112, arrow=True, both=True)
    s.text(560, 88, "network", 21, AMBER, "middle")
    s.text(560, 340, "One distributed algorithm / four GPU devices / two servers", 23, WHITE, "middle")
    s.save()

    s = SVG("cholesky.svg", 480, 385, "First panel of a four by four tiled Cholesky factorization")
    colors = {"P": MINT, "S": VIOLET, "U": AMBER, "G": "#8abdea"}
    for i in range(4):
        for j in range(4):
            x, y = 40 + j * 88, 22 + i * 88
            if j > i:
                s.rect(x, y, 76, 76, fill="#14202a", stroke="#253a48", radius=10)
                continue
            key = "P" if i == j == 0 else "S" if j == 0 else "U" if i == j else "G"
            s.rect(x, y, 76, 76, stroke=colors[key], radius=10)
            s.text(x + 38, y + 50, key, 31, colors[key], "middle", 600)
    s.save()

    s = SVG("panel.svg", 1130, 400, "Selected Cholesky dependencies for panel one and the next diagonal factor")
    # Edges before nodes, so their endpoints never overlap labels.
    for a, b in [((190, 195), (290, 90)), ((190, 195), (290, 290)),
                 ((485, 90), (605, 65)), ((485, 90), (605, 190)),
                 ((485, 290), (605, 190)), ((485, 290), (605, 315)),
                 ((800, 65), (900, 65))]:
        s.line(*a, *b, arrow=True)
    s.box(10, 155, 180, 80, "P(1,1)", MINT)
    s.box(295, 50, 185, 80, "S(2,1)", VIOLET)
    s.box(295, 250, 185, 80, "S(3,1)", VIOLET)
    s.box(610, 25, 185, 80, "U(2,2)", AMBER)
    s.box(610, 150, 185, 80, "G(3,2)", "#8abdea")
    s.box(610, 275, 185, 80, "U(3,3)", AMBER)
    s.box(905, 25, 200, 80, "P(2,2)", MINT)
    s.text(1005, 145, "next panel", 20, MINT, "middle")
    s.text(10, 390, "Selected dependencies · P factor / S solve / U diagonal update / G off-diagonal update", 19, MUTED)
    s.save()

    s = SVG("feedback.svg", 1120, 150, "Edit, run the benchmark, check PASS, and view a plot")
    for i, (label, color) in enumerate([
        ("Edit", MINT), ("Run warm", VIOLET), ("Check PASS", MINT), ("Plot", AMBER),
    ]):
        x = 10 + i * 282
        s.box(x, 20, 240, 90, label, color)
        if i < 3:
            s.line(x + 248, 65, x + 273, 65, arrow=True)
    s.save()


def benchmarks():
    with (ROOT / "data/odo-fixed.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 20 and all(r["pass"] == "true" for r in rows)
    s = SVG("throughput.svg", 1120, 435, "Measured GPU throughput at fixed tile size 1024")
    left, right, top, bottom = 110, 1050, 60, 348
    ymax = 1250
    for tick in [0, 300, 600, 900, 1200]:
        y = bottom - tick / ymax * (bottom - top)
        s.line(left, y, right, y, color="#314653", width=1)
        s.text(left - 18, y + 7, tick, 20, MUTED, "end")
    s.text(left, 24, "GFLOP/s  ↑ higher is better", 21, MUTED)
    for n in [2048, 4096, 8192]:
        x = left + (math.log2(n) - 11) / 2 * (right - left)
        s.text(x, 390, f"{n:,}", 22, MUTED, "middle")
    s.text(580, 428, "Matrix size N · logarithmic spacing", 20, MUTED, "middle")
    for target, label, color, dy in [("gpu", "1 GPU", MINT, -17), ("gpu-8gpus", "8 GPUs", VIOLET, 28)]:
        points = []
        for n in [2048, 4096, 8192]:
            group = [r for r in rows if r["target"] == target and int(r["n"]) == n]
            row = min(group, key=lambda r: float(r["seconds"]))
            x = left + (math.log2(n) - 11) / 2 * (right - left)
            y = bottom - float(row["gflops"]) / ymax * (bottom - top)
            points.append((x, y))
            s.parts.append(f'<circle cx="{x}" cy="{y}" r="6" fill="{color}"/>')
            s.text(x, y + dy, f'{float(row["gflops"]):,.0f}', 20, color, "middle")
        s.parts.append('<polyline points="' + " ".join(f"{x},{y}" for x, y in points)
                       + f'" fill="none" stroke="{color}" stroke-width="3"/>')
        legend_x = 730 if target == "gpu" else 900
        s.line(legend_x, 16, legend_x + 28, 16, color=color, width=4)
        s.text(legend_x + 40, 24, label, 22, color)
    s.save()

    s = SVG("comparison.svg", 1120, 300, "One GPU is faster than eight GPUs in the 16384 rehearsal sample")
    s.text(20, 27, "N = 16,384    /    tile = 2,048    /    elapsed seconds ↓ lower is better", 22, MUTED)
    for i, (target, label, color) in enumerate([("gpu", "1 GPU", MINT), ("gpu-8gpus", "8 GPUs", VIOLET)]):
        row = min((r for r in rows if r["target"] == target and int(r["n"]) == 16384),
                  key=lambda r: float(r["seconds"]))
        seconds = float(row["seconds"])
        y = 80 + i * 106
        s.text(20, y + 38, label, 29, color, weight=600)
        s.rect(210, y, seconds / .4 * 690, 57, fill=color, stroke=color, radius=8)
        s.text(230 + seconds / .4 * 690, y + 39, f"{seconds:.3f} s", 31, color, weight=600)
    s.save()


def timeline():
    with (ROOT / "data/odo-trace.csv").open() as stream:
        events = list(csv.DictReader(stream))
    s = SVG("timeline.svg", 1120, 430, "Eight GPU processor lanes from an instrumented Dagger run")
    left, right, top = 125, 1085, 55
    end = max(float(e["start_ms"]) + float(e["duration_ms"]) for e in events)
    extent = math.ceil(end / 10) * 10
    for gpu in range(1, 9):
        y = top + (gpu - 1) * 38
        s.text(10, y + 20, f"GPU {gpu}", 20, MUTED)
        s.rect(left, y, right - left, 25, fill=PANEL, stroke=PANEL, radius=4)
    for event in events:
        x = left + float(event["start_ms"]) / extent * (right - left)
        w = max(.6, float(event["duration_ms"]) / extent * (right - left))
        y = top + (int(event["device"]) - 1) * 38
        color = MINT if event["category"] == "compute" else VIOLET
        # Half-height rows distinguish event categories even in overlapping spans.
        y += 1 if event["category"] == "compute" else 14
        s.rect(round(x, 3), y, round(w, 3), 10, fill=color, stroke=color, radius=1)
    for i in range(5):
        x = left + i / 4 * (right - left)
        s.text(x, 390, f"{i / 4 * extent:.0f}", 18, MUTED, "middle")
    s.text(605, 425, "Milliseconds since first recorded event", 20, MUTED, "middle")
    s.text(126, 25, "Host task events", 20, MINT)
    s.text(405, 25, "Data-movement events", 20, VIOLET)
    s.save()


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    diagrams()
    benchmarks()
    timeline()
    print(f"Wrote 11 SVG figures to {OUT}")
