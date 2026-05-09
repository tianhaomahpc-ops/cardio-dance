#!/usr/bin/env python3
"""Plot pseudo-ECG CSV from monodomain output.

Reads output/<run>/pseudo_ecg.csv (columns: t_ms, lead_*, ...) and produces
a multi-panel PNG with one trace per lead.
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True, help="path to pseudo_ecg.csv")
    p.add_argument("--out", required=True, help="output PNG path")
    p.add_argument("--title", default="Pseudo-ECG")
    args = p.parse_args()

    data = {}
    with open(args.csv) as f:
        reader = csv.reader(f)
        header = next(reader)
        for col in header:
            data[col] = []
        for row in reader:
            for col, val in zip(header, row):
                data[col].append(float(val))

    t = np.array(data["t_ms"])
    leads = [k for k in header if k != "t_ms"]
    if not leads:
        sys.exit("no leads in CSV")

    fig, axs = plt.subplots(
        len(leads), 1, figsize=(10, 2.5 * len(leads)), sharex=True
    )
    if len(leads) == 1:
        axs = [axs]
    for ax, lead in zip(axs, leads):
        v = np.array(data[lead])
        ax.plot(t, v, lw=1.4, color="#1f77b4")
        ax.axhline(0, color="0.5", lw=0.5, ls="--")
        ax.set_ylabel(f"{lead}\n(arbitrary units)")
        ax.grid(alpha=0.3)
        # Annotate peak-to-peak.
        ptp = float(np.max(v) - np.min(v))
        ax.set_title(f"{lead}: peak-to-peak {ptp:.3g}", loc="right", fontsize=9)
    axs[-1].set_xlabel("time (ms)")
    fig.suptitle(args.title, fontsize=12)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
