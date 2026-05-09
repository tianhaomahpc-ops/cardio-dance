#!/usr/bin/env python3
"""Render a movie of V_m on a heart mesh from MFEM ParaView output.

Reads every CycleXXXXXX/data.pvtu under <run_dir>/heart/monodomain/, renders
the V_m point-data field on the unstructured mesh, writes one PNG per
frame, then encodes an MP4 via ffmpeg.

Usage:
    plot_vm_movie.py --run output/half_ellipsoid_purkinje --out vm_movie.mp4

Notes:
- pyvista renders off-screen. We pin the camera to a 3/4 anatomical view by
  default; override via --camera-azimuth and --camera-elevation if needed.
- Color scale is fixed across frames so depolarization is comparable
  between snapshots.
"""

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyvista as pv


def load_purkinje_network(path: Path):
    """Parse a cardio-dance .network file.

    Returns (pv.PolyData with line segments, terminal point coords array).
    """
    with open(path) as f:
        n_nodes, n_edges = (int(x) for x in f.readline().split())
        nodes = np.zeros((n_nodes, 3), dtype=float)
        is_terminal = np.zeros(n_nodes, dtype=bool)
        for i in range(n_nodes):
            tokens = f.readline().split()
            nodes[i] = [float(t) for t in tokens[:3]]
            is_terminal[i] = int(tokens[3]) != 0
        edges = []
        for _ in range(n_edges):
            tokens = f.readline().split()
            edges.append((int(tokens[0]), int(tokens[1])))

    lines = np.zeros((len(edges), 3), dtype=int)
    for k, (a, b) in enumerate(edges):
        lines[k] = (2, a, b)
    poly = pv.PolyData(nodes, lines=lines.flatten())
    terminals = nodes[is_terminal]
    return poly, terminals


def collect_frames(run_dir: Path):
    base = run_dir / "heart" / "monodomain"
    if not base.is_dir():
        sys.exit(f"missing {base}")
    pat = re.compile(r"^Cycle(\d+)$")
    cycles = []
    for p in base.iterdir():
        m = pat.match(p.name)
        if m and (p / "data.pvtu").exists():
            cycles.append((int(m.group(1)), p / "data.pvtu"))
    cycles.sort()
    return cycles


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="output/<name> directory")
    ap.add_argument("--out", required=True, help="output mp4 path")
    ap.add_argument("--field", default="Vm", help="point-data field to render")
    ap.add_argument("--vmin", type=float, default=-90.0)
    ap.add_argument("--vmax", type=float, default=30.0)
    ap.add_argument("--cmap", default="RdBu_r")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--azimuth", type=float, default=-60.0)
    ap.add_argument("--elevation", type=float, default=20.0)
    ap.add_argument(
        "--clip-y",
        action="store_true",
        help="Clip mesh at y>=0 to expose endocardium / wall interior",
    )
    ap.add_argument(
        "--purkinje-network",
        default=None,
        help="optional .network file to overlay as red tubes",
    )
    ap.add_argument(
        "--show-wireframe",
        action="store_true",
        help="Overlay mesh edges (slows render, only useful for coarse meshes)",
    )
    ap.add_argument(
        "--dt-ms-per-cycle",
        type=float,
        default=0.05,
        help="PDE dt in ms; cycle index * this = sim time",
    )
    args = ap.parse_args()

    run_dir = Path(args.run)
    out_path = Path(args.out)
    cycles = collect_frames(run_dir)
    if not cycles:
        sys.exit("no Cycle*/data.pvtu found")
    print(f"found {len(cycles)} frames", file=sys.stderr)

    pv.global_theme.allow_empty_mesh = True
    pv.set_plot_theme("document")

    purkinje_poly = None
    purkinje_terminals = None
    if args.purkinje_network:
        purkinje_poly, purkinje_terminals = load_purkinje_network(
            Path(args.purkinje_network)
        )
        print(
            f"loaded Purkinje: {purkinje_poly.n_points} nodes, "
            f"{purkinje_poly.n_cells} edges, "
            f"{len(purkinje_terminals)} terminals",
            file=sys.stderr,
        )

    frame_dir = Path(tempfile.mkdtemp(prefix="vm_movie_"))
    print(f"writing PNG frames into {frame_dir}", file=sys.stderr)

    for idx, (cycle, path) in enumerate(cycles):
        mesh = pv.read(str(path))
        if args.field not in mesh.point_data:
            sys.exit(f"field {args.field!r} not in {path}")
        if args.clip_y:
            mesh = mesh.clip(normal="y", origin=(0, 0, 0), invert=False)
        t_ms = cycle * args.dt_ms_per_cycle

        plotter = pv.Plotter(off_screen=True, window_size=(900, 700))
        plotter.add_mesh(
            mesh,
            scalars=args.field,
            cmap=args.cmap,
            clim=(args.vmin, args.vmax),
            show_edges=args.show_wireframe,
            edge_color="gray",
            line_width=0.3,
            scalar_bar_args={
                "title": f"{args.field} (mV)",
                "vertical": True,
                "fmt": "%.0f",
                "title_font_size": 14,
                "label_font_size": 12,
            },
            lighting=True,
            specular=0.4,
            opacity=0.9 if purkinje_poly is not None else 1.0,
        )

        if purkinje_poly is not None:
            tubes = purkinje_poly.tube(radius=0.4)
            plotter.add_mesh(
                tubes, color="#ffd700", lighting=True, specular=1.0
            )
            plotter.add_mesh(
                pv.PolyData(purkinje_terminals),
                color="red",
                point_size=8,
                render_points_as_spheres=True,
            )
        plotter.add_text(
            f"t = {t_ms:7.1f} ms  (frame {idx + 1}/{len(cycles)})",
            position="upper_edge",
            font_size=14,
            color="black",
        )
        plotter.view_isometric()
        plotter.camera.azimuth = args.azimuth
        plotter.camera.elevation = args.elevation
        plotter.camera.zoom(1.2)

        png_path = frame_dir / f"frame_{idx:05d}.png"
        plotter.screenshot(str(png_path), transparent_background=False)
        plotter.close()
        if (idx + 1) % 5 == 0 or idx == len(cycles) - 1:
            print(f"  rendered {idx + 1}/{len(cycles)}", file=sys.stderr)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "warning",
        "-framerate",
        str(args.fps),
        "-i",
        str(frame_dir / "frame_%05d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        str(out_path),
    ]
    print("running:", " ".join(cmd), file=sys.stderr)
    subprocess.check_call(cmd)
    print(f"wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
