#!/usr/bin/env python3
"""Export harness results as PNG/SVG on the presenter's laptop.

python3 harness/plot.py results/results.csv --output results/performance
Requires matplotlib; no Python or plotting installation is needed on Odo.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", type=Path,
                        default=Path(__file__).resolve().parents[1] / "results/results.csv")
    parser.add_argument("--output", type=Path, default=Path("results/performance"))
    parser.add_argument("--label", help="Show only this participant's results")
    args = parser.parse_args()
    series = {}
    with args.csv.open(newline="") as source:
        for row in csv.DictReader(source):
            if row["pass"].lower() != "true" or (args.label and row["label"] != args.label):
                continue
            name = (f'{row["impl"]} {row["target"]}, '
                    f'{row["nranks"]} ranks × {row["nthreads"]} threads, tile {row["bs"]}')
            n = int(row["n"])
            best = series.setdefault(name, {})
            seconds = float(row["seconds"])
            if n not in best or seconds < best[n][0]:
                best[n] = (seconds, float(row["gflops"]))
    if not series:
        parser.error("No passing results match the selection")
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
    for name, values in sorted(series.items()):
        ns = sorted(values)
        axes[0].plot(ns, [values[n][1] for n in ns], "o-", label=name)
        axes[1].plot(ns, [values[n][0] for n in ns], "o-", label=name)
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_xlabel("Matrix size N")
        axis.grid(True, alpha=0.2)
    axes[0].set_ylabel("GFLOP/s (higher is faster)")
    axes[1].set_ylabel("Seconds (lower is faster)")
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylim(bottom=0)
    fig.suptitle("Tiled Cholesky — best passing trial, after compilation")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", fontsize=9, ncol=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg"):
        destination = args.output.with_suffix(suffix)
        fig.savefig(destination, dpi=160, bbox_inches="tight", pad_inches=0.15)
        print(destination)
    plt.close(fig)


if __name__ == "__main__":
    main()
