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
KERNEL = {"P": MINT, "S": VIOLET, "U": AMBER, "G": "#8abdea"}   # the slides' kernel key


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

    s = SVG("two-gpus.svg", 1040, 454,
            "One matrix product A times B equals C with two GPUs available on one server")
    # A and C: single matrices here, no block rows yet. Shared center axis x = 520.
    s.rect(232, 87, 122, 180, fill=PANEL, stroke=LINE, radius=12)
    s.text(293, 191, "A", 36, WHITE, "middle", 600)
    s.text(293, 64, "A · 1024 × 512", 19, MUTED, "middle")
    s.text(400, 198, "×", 60, WHITE, "middle", 700)
    s.rect(447, 115, 116, 124, fill=PANEL, stroke=AMBER, radius=10)
    s.text(505, 190, "B", 38, AMBER, "middle", 600)
    s.text(505, 64, "B · 512 × 512", 19, MUTED, "middle")
    s.text(624, 198, "=", 60, WHITE, "middle", 700)
    s.rect(686, 87, 122, 180, fill=PANEL, stroke=LINE, radius=12)
    s.text(747, 191, "C", 36, WHITE, "middle", 600)
    s.text(747, 64, "C · 1024 × 512", 19, MUTED, "middle")
    # The two devices that will run the product.
    s.box(90, 320, 340, 85, "GPU 1", MINT, size=28)
    s.box(610, 320, 340, 85, "GPU 2", VIOLET, size=28)
    s.text(520, 444, "One Julia process · two GPUs · one product C = A × B",
           21, MUTED, "middle")
    s.save()

    s = SVG("product-tiles.svg", 1040, 454,
            "A splits into two block rows; each multiplied by replicated B on its own GPU; the block rows form C")
    dashed = ' stroke-dasharray="4 5"'
    # A: one matrix with a horizontal split into two block rows. Center axis x = 520.
    s.parts.append(f'<rect x="232" y="87" width="122" height="180" rx="12" fill="none" stroke="{LINE}" stroke-width="1.5"/>')
    s.rect(238, 93, 110, 80, fill=PANEL, stroke=MINT, radius=8)
    s.rect(238, 181, 110, 80, fill=PANEL, stroke=VIOLET, radius=8)
    s.parts.append(f'<line x1="232" y1="177" x2="354" y2="177" stroke="{MUTED}" stroke-width="1.5"{dashed}/>')
    s.text(293, 143, "A₁", 30, MINT, "middle", 600)
    s.text(293, 231, "A₂", 30, VIOLET, "middle", 600)
    s.text(293, 64, "A · 1024 × 512", 19, MUTED, "middle")
    s.text(220, 147, "GPU 1", 21, MINT, "end", 600)
    s.text(220, 235, "GPU 2", 21, VIOLET, "end", 600)
    # Operators.
    s.text(400, 198, "×", 60, WHITE, "middle", 700)
    s.text(624, 198, "=", 60, WHITE, "middle", 700)
    # B: one matrix, replicated, read by both GPUs (ghost copy behind).
    s.rect(459, 131, 116, 124, fill=PANEL, stroke=AMBER, radius=10)
    s.parts[-1] = s.parts[-1].replace('stroke-width="1.5"', 'stroke-width="1.4" opacity="0.6"')
    s.rect(447, 115, 116, 124, fill=PANEL, stroke=AMBER, radius=10)
    s.text(505, 190, "B", 38, AMBER, "middle", 600)
    s.text(505, 64, "B · 512 × 512", 19, MUTED, "middle")
    s.text(587, 127, "× 2", 21, AMBER, "start", 600)
    s.text(505, 278, "replicated · read by both GPUs", 18, MUTED, "middle")
    # C: one output matrix assembling the two block rows.
    s.parts.append(f'<rect x="686" y="87" width="122" height="180" rx="12" fill="none" stroke="{LINE}" stroke-width="1.5"/>')
    s.rect(692, 93, 110, 80, fill=PANEL, stroke=MINT, radius=8)
    s.rect(692, 181, 110, 80, fill=PANEL, stroke=VIOLET, radius=8)
    s.parts.append(f'<line x1="686" y1="177" x2="808" y2="177" stroke="{MUTED}" stroke-width="1.5"{dashed}/>')
    s.text(747, 143, "C₁", 30, MINT, "middle", 600)
    s.text(747, 231, "C₂", 30, VIOLET, "middle", 600)
    s.text(747, 64, "C · 1024 × 512", 19, MUTED, "middle")
    s.text(824, 147, "GPU 1", 21, MINT, "start", 600)
    s.text(824, 235, "GPU 2", 21, VIOLET, "start", 600)
    # Assignment chips: which device computes which block row.
    s.box(90, 320, 340, 85, "GPU 1", MINT, "C₁ = A₁ × B", 28)
    s.box(610, 320, 340, 85, "GPU 2", VIOLET, "C₂ = A₂ × B", 28)
    s.text(520, 444, "One Julia process · one scope · two GPUs · one product C = A × B",
           21, MUTED, "middle")
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

    # Step k = 2 of a 5 x 5 tile grid: what each line of the loop nest writes
    # and reads, next to the line itself.
    s = SVG("cholesky-step.svg", 1180, 470,
            "Step k equals 2 of a five by five tiled Cholesky: column 1 is finished; POTRF "
            "factors tile 2,2; TRSM solves tiles 3,2 to 5,2; SYRK updates diagonal tiles 3,3 "
            "to 5,5; GEMM updates tiles 4,3, 5,3 and 5,4. GEMM of tile 5,3 reads tiles 5,2 and 3,2.")
    k, size, step, left, top = 2, 62, 72, 46, 50
    centre = lambda m, n: (left + (n - 1) * step + size / 2, top + (m - 1) * step + size / 2)
    for j in range(1, 6):
        s.text(left + (j - 1) * step + size / 2, top - 14, "k = 2" if j == k else j, 17,
               MINT if j == k else MUTED, "middle", 600 if j == k else 400)
        s.text(left - 16, top + (j - 1) * step + size / 2 + 6, j, 17, MUTED, "middle")
    for m in range(1, 6):
        for n in range(1, 6):
            x, y = left + (n - 1) * step, top + (m - 1) * step
            if n > m:
                s.rect(x, y, size, size, fill="#14202a", stroke="#253a48", radius=10)
                continue
            if n < k:
                s.rect(x, y, size, size, fill="#1b2a35", stroke="#2c4250", radius=10)
                s.text(x + size / 2, y + 41, "L", 24, "#6d8291", "middle", 600)
                continue
            key = "P" if m == n == k else "S" if n == k else "U" if m == n else "G"
            s.rect(x, y, size, size, stroke=KERNEL[key], radius=10)
            s.text(x + size / 2, y + 42, key, 28, KERNEL[key], "middle", 600)
    # G(5,3) reads S(5,2) and S(3,2).
    for source in ((5, 2), (3, 2)):
        (x1, y1), (x2, y2) = centre(*source), centre(5, 3)
        length = math.hypot(x2 - x1, y2 - y1)
        ux, uy = (x2 - x1) / length, (y2 - y1) / length
        s.line(round(x1 + ux * 22, 1), round(y1 + uy * 22, 1), round(x2 - ux * 26, 1),
               round(y2 - uy * 26, 1), KERNEL["G"], arrow=True, width=2.5)
    s.text(left, top + 5 * step + 22, "G(5,3) reads S(5,2) and S(3,2)", 17, KERNEL["G"])
    legend = [
        ("P", "T[k,k] = chol(T[k,k])", "factor the diagonal tile"),
        ("S", "T[m,k] = T[m,k] / T[k,k]'", "each tile below it: reads T[k,k]"),
        ("U", "T[n,n] -= T[n,k] * T[n,k]'", "each later diagonal tile: reads T[n,k]"),
        ("G", "T[m,n] -= T[m,k] * T[n,k]'", "each tile below that: reads T[m,k], T[n,k]"),
    ]
    for i, (key, code, what) in enumerate(legend):
        y = 58 + i * 92
        s.rect(470, y, 50, 50, stroke=KERNEL[key], radius=9)
        s.text(495, y + 35, key, 26, KERNEL[key], "middle", 600)
        s.parts.append(
            f'<text x="545" y="{y + 21}" fill="{WHITE}" font-size="22" '
            f'font-family="DejaVu Sans Mono, monospace">{escape(code)}</text>')
        s.text(545, y + 48, what, 18, MUTED)
    s.text(470, 448, "L = finished columns of the factor. Step k + 1 works on column k + 1.", 18, MUTED)
    s.save()
    gray_scott_adi()


def gray_scott_adi():
    """One ADI time step of the final exercise: column solves, then row solves."""
    s = SVG("gray-scott-adi.svg", 1180, 430,
            "One ADI time step on the n by n grid of u. First half step: the explicit part u "
            "times L couples neighbours along y, then every column is solved with the Cholesky "
            "factor, n independent solves. Second half step: the explicit part L times u couples "
            "neighbours along x, then every row is solved, again n independent solves.")
    size, top = 260, 92
    for x, title, explicit, color in ((150, "first half step", "explicit: u · L (along y)", MINT),
                                      (770, "second half step", "explicit: L · u (along x)", AMBER)):
        s.text(x + size / 2, top - 52, title, 24, color, "middle", 600)
        s.text(x + size / 2, top - 20, explicit, 19, MUTED, "middle")
        s.rect(x, top, size, size, stroke=LINE, radius=10)
    for k in range(8):   # column solves: vertical arrows
        cx = 150 + 20 + k * 31
        s.line(cx, top + 16, cx, top + size - 16, MINT, arrow=True, width=2.2)
    for k in range(8):   # row solves: horizontal arrows
        cy = top + 20 + k * 31
        s.line(770 + 16, cy, 770 + size - 16, cy, AMBER, arrow=True, width=2.2)
    s.text(150 + size / 2, top + size + 36, "implicit along x: every column", 20, WHITE, "middle")
    s.text(150 + size / 2, top + size + 64, "u[:, j] = Fu \\ ru[:, j]", 20, MINT, "middle")
    s.text(770 + size / 2, top + size + 36, "implicit along y: every row", 20, WHITE, "middle")
    s.text(770 + size / 2, top + size + 64, "u[i, :] = Fu \\ ru[i, :]", 20, AMBER, "middle")
    s.line(150 + size + 40, top + size / 2, 770 - 40, top + size / 2, arrow=True, width=2.5)
    s.text(590, top + size / 2 - 16, "all columns done", 18, MUTED, "middle")
    s.text(590, top + size / 2 + 34, "then the rows", 18, MUTED, "middle")
    s.save()


MAGMA = [(0, 0, 4), (40, 11, 84), (101, 21, 110), (159, 42, 99), (212, 72, 66),
         (245, 125, 21), (250, 193, 39), (252, 255, 164)]   # as in host/visuals.js


def magma(t):
    t = min(1.0, max(0.0, t)) * (len(MAGMA) - 1)
    i = min(len(MAGMA) - 2, int(t))
    f = t - i
    return bytes(round(a + (b - a) * f) for a, b in zip(MAGMA[i], MAGMA[i + 1]))


def png(width, height, rows):
    """A truecolor PNG from rows of RGB bytes, with the standard library."""
    import struct
    import zlib
    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff))
    raw = b"".join(b"\x00" + row for row in rows)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def gray_scott():
    """Three frames of the activator v from data/gray-scott-v.bin
    (gray_scott_data.jl), drawn as the console draws them."""
    import base64
    n, scale = 256, 2
    data = (ROOT / "data" / "gray-scott-v.bin").read_bytes()
    colours = [magma(i / 255) for i in range(256)]
    captions = [("t = 200", "the seeded squares react"), ("t = 1000", "spots grow and split"),
                ("t = 6000", "a labyrinth, for F = 0.055, k = 0.062")]
    size = n * scale
    s = SVG("gray-scott.svg", 3 * size + 2 * 40, size + 90,
            "The activator v of the final exercise at t = 200, 1000 and 6000: fourteen seeded "
            "squares become spots, the spots split, and they fill the grid with a labyrinth of stripes.")
    for frame, (time_label, what) in enumerate(captions):
        v = data[frame * n * n:(frame + 1) * n * n]
        rows = []
        for py in range(size):
            # x is the first index, y the second one, upwards, as in the console.
            fy = (size - 1 - py) / scale
            j0 = int(fy); ty = fy - j0; j1 = (j0 + 1) % n
            row = bytearray()
            for px in range(size):
                fx = px / scale
                i0 = int(fx); tx = fx - i0; i1 = (i0 + 1) % n
                value = ((v[i0 + j0 * n] * (1 - tx) + v[i1 + j0 * n] * tx) * (1 - ty)
                         + (v[i0 + j1 * n] * (1 - tx) + v[i1 + j1 * n] * tx) * ty)
                row += colours[round(value)]
            rows.append(bytes(row))
        image = base64.b64encode(png(size, size, rows)).decode()
        x = frame * (size + 40)
        s.parts.append(f'<image x="{x}" y="0" width="{size}" height="{size}" '
                       f'href="data:image/png;base64,{image}"/>')
        s.text(x + size / 2, size + 42, time_label, 30, WHITE, "middle", 600)
        s.text(x + size / 2, size + 78, what, 26, MUTED, "middle")
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
    gray_scott()
    benchmarks()
    timeline()
    print(f"Wrote 16 SVG figures to {OUT}")
