#!/usr/bin/env python3
"""Sample V_m at internal probe points across all snapshot cycles and plot.

For each probe (x,y,z), find the mesh node closest to it (once), then walk
every CycleNNNNNN/data.pvtu and record V_m at that node. Plot all probes
on one figure with t_ms = cycle * dt_ms_per_cycle.
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv


def parse_probes(spec):
    """Parse a list of 'name@x,y,z' specs."""
    probes = []
    for s in spec.split(";"):
        s = s.strip()
        if not s:
            continue
        if "@" not in s:
            raise ValueError(f"bad probe '{s}', expected name@x,y,z")
        name, coords = s.split("@", 1)
        x, y, z = [float(v) for v in coords.split(",")]
        probes.append((name.strip(), np.array([x, y, z])))
    return probes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument(
        "--probes",
        required=True,
        help="semicolon-separated 'name@x,y,z' specs",
    )
    ap.add_argument("--out", required=True, help="output PNG path")
    ap.add_argument("--dt-ms-per-cycle", type=float, default=0.05)
    ap.add_argument("--title", default="V_m at internal probes")
    args = ap.parse_args()

    run = Path(args.run)
    base = run / "heart" / "monodomain"
    if not base.is_dir():
        sys.exit(f"missing {base}")
    pat = re.compile(r"^Cycle(\d+)$")
    cycles = sorted(
        (int(m.group(1)), p / "data.pvtu")
        for p in base.iterdir()
        for m in [pat.match(p.name)]
        if m and (p / "data.pvtu").exists()
    )
    if not cycles:
        sys.exit("no Cycle* found")

    probes = parse_probes(args.probes)
    print(f"Probes: {[(n, c.tolist()) for n, c in probes]}", file=sys.stderr)

    # Load first frame, find nearest node for each probe.
    first = pv.read(str(cycles[0][1]))
    pts = first.points
    node_idx = []
    actual_pos = []
    for name, target in probes:
        d2 = np.sum((pts - target) ** 2, axis=1)
        i = int(np.argmin(d2))
        node_idx.append(i)
        actual_pos.append(pts[i])
        print(
            f"  {name}: target {target.tolist()}, snapped to {pts[i].tolist()} "
            f"(dist {np.sqrt(d2[i]):.2f} mm)",
            file=sys.stderr,
        )

    # Walk all cycles.
    n = len(cycles)
    t = np.zeros(n)
    vm = np.zeros((n, len(probes)))
    for k, (cycle, path) in enumerate(cycles):
        m = pv.read(str(path))
        vm_arr = m.point_data["Vm"]
        for j, ni in enumerate(node_idx):
            vm[k, j] = vm_arr[ni]
        t[k] = cycle * args.dt_ms_per_cycle
        if k % 20 == 0 or k == n - 1:
            print(f"  cycle {k + 1}/{n}", file=sys.stderr)

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(probes)))
    for j, ((name, target), pos, c) in enumerate(zip(probes, actual_pos, colors)):
        label = f"{name}  ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) mm"
        ax.plot(t, vm[:, j], color=c, lw=1.5, label=label)
    ax.set_xlabel("t (ms)")
    ax.set_ylabel("V_m (mV)")
    ax.set_title(args.title)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.axhline(0, color="0.6", lw=0.5, ls="--")
    ax.axhline(-85, color="0.6", lw=0.5, ls=":")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
