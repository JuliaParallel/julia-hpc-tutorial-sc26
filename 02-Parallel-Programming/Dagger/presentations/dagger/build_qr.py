"""QR codes for the slides: the console install page and the reference
solutions. Needs the `qrcode` package (pip install qrcode):
    python3 presentations/dagger/build_qr.py
writes assets/qr-console.svg and assets/qr-solutions.svg."""
from pathlib import Path
import qrcode
import qrcode.image.svg

ROOT = Path(__file__).resolve().parent
LINKS = {
    "qr-console.svg": "https://fda-tome.github.io/sc26/",
    "qr-solutions.svg": "https://github.com/JuliaParallel/julia-hpc-tutorial-sc26/tree/main/"
                        "02-Parallel-Programming/Dagger/solution",
}

for name, url in LINKS.items():
    code = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=10, border=4,
                         image_factory=qrcode.image.svg.SvgPathFillImage)
    code.add_data(url)
    code.make(fit=True)
    code.make_image().save(str(ROOT / "assets" / name))
    print(name, "->", url)
