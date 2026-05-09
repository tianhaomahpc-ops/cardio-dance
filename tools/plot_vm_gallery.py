#!/usr/bin/env python3
"""Render mesh-wireframe + cross-section gallery from a monodomain run.

Produces a fixed set of PNGs in <run>/gallery/:
  mesh_wireframe_t0.png        - resting heart, wireframe visible
  mesh_wireframe_t20.png       - peak activation, wireframe visible
  mesh_apex_closeup.png        - zoomed-in apex showing tet detail
  cross_y0_t{5,15,25,40,80}.png   - y=0 sagittal-like slices over time
  cross_z0_t20.png             - z=0 cut (transverse-ish through long axis)
  cross_x_apex_t20.png         - x=mid cut showing wall ring + apex view
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pyvista as pv


def load_purkinje(path):
    if not path:
        return None, None
    p = Path(path)
    with open(p) as f:
        n_n, n_e = map(int, f.readline().split())
        nodes = []
        is_term = []
        for _ in range(n_n):
            t = f.readline().split()
            nodes.append([float(x) for x in t[:3]])
            is_term.append(int(t[3]) != 0)
        edges = []
        for _ in range(n_e):
            t = f.readline().split()
            edges.append([int(t[0]), int(t[1])])
    nodes = np.array(nodes)
    lines = np.zeros((len(edges), 3), dtype=int)
    for k, (a, b) in enumerate(edges):
        lines[k] = (2, a, b)
    poly = pv.PolyData(nodes, lines=lines.flatten())
    terms = nodes[np.array(is_term)] if any(is_term) else nodes[:0]
    return poly, terms


def render(mesh, *, title, out_path, purkinje=None, terminals=None,
           show_edges=False, edge_width=0.5, scalars="Vm", clim=(-90, 30),
           cmap="RdBu_r", azimuth=-60.0, elevation=20.0, zoom=1.2,
           opacity=1.0, window=(900, 700)):
    p = pv.Plotter(off_screen=True, window_size=window)
    p.add_mesh(
        mesh,
        scalars=scalars,
        cmap=cmap,
        clim=clim,
        show_edges=show_edges,
        edge_color="black",
        line_width=edge_width,
        opacity=opacity,
        scalar_bar_args={"title": f"{scalars} (mV)", "vertical": True,
                         "fmt": "%.0f"},
        lighting=True,
        specular=0.3,
    )
    if purkinje is not None:
        p.add_mesh(purkinje.tube(radius=0.4), color="#ffd700")
        if terminals is not None and len(terminals) > 0:
            p.add_mesh(pv.PolyData(terminals), color="red", point_size=8,
                       render_points_as_spheres=True)
    p.add_text(title, position="upper_edge", font_size=11, color="black")
    p.view_isometric()
    p.camera.azimuth = azimuth
    p.camera.elevation = elevation
    p.camera.zoom(zoom)
    p.screenshot(str(out_path))
    p.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--purkinje", default=None,
                    help="optional .network file")
    ap.add_argument("--dt-ms-per-cycle", type=float, default=0.05)
    args = ap.parse_args()

    pv.set_plot_theme("document")
    run = Path(args.run)
    gallery = run / "gallery"
    gallery.mkdir(parents=True, exist_ok=True)

    purkinje_poly, terminals = load_purkinje(args.purkinje)

    def cycle_path(t_ms):
        cycle = int(round(t_ms / args.dt_ms_per_cycle))
        return run / "heart" / "monodomain" / f"Cycle{cycle:06d}" / "data.pvtu"

    print("=== Mesh wireframe ===", file=sys.stderr)
    # t=0 resting state
    m0 = pv.read(str(cycle_path(0.05)))
    render(
        m0, title=f"Mesh structure (t=0 ms, resting): {m0.n_cells} tets",
        out_path=gallery / "mesh_wireframe_t0.png",
        purkinje=purkinje_poly, terminals=terminals,
        show_edges=True, edge_width=0.4,
    )

    # t=20 peak activation
    m20 = pv.read(str(cycle_path(20.0)))
    render(
        m20, title="Mesh wireframe at t=20 ms (peak activation)",
        out_path=gallery / "mesh_wireframe_t20.png",
        purkinje=purkinje_poly, terminals=terminals,
        show_edges=True, edge_width=0.4,
    )

    # Apex close-up: clip a small box around the apex (x in [25,35]).
    apex = m20.clip_box(bounds=(25, 35, -8, 8, -8, 8), invert=False)
    render(
        apex, title="Apex close-up (x in [25,35] mm), wireframe visible",
        out_path=gallery / "mesh_apex_closeup.png",
        show_edges=True, edge_width=0.6, zoom=1.5,
        purkinje=purkinje_poly, terminals=terminals,
    )

    print("=== Cross-sections from different angles ===", file=sys.stderr)
    # 1) y=0 sagittal-like
    for t in (5, 15, 25, 40, 80):
        m = pv.read(str(cycle_path(float(t))))
        cut = m.clip(normal="y", origin=(0, 0, 0), invert=False)
        render(
            cut, title=f"Cross-section y>=0  t={t} ms",
            out_path=gallery / f"cross_y0_t{t}.png",
            purkinje=purkinje_poly, terminals=terminals,
            azimuth=-90, elevation=15,
        )

    # 2) z=0 horizontal cut at peak activation
    cut_z = m20.clip(normal="z", origin=(0, 0, 0), invert=False)
    render(
        cut_z, title="Cross-section z>=0 at t=20 ms (top half)",
        out_path=gallery / "cross_z0_t20.png",
        purkinje=purkinje_poly, terminals=terminals,
        azimuth=0, elevation=80,
    )

    # 3) x = mid (x=17.5) transverse slab showing wall ring at t=20
    cut_x = m20.clip(normal="x", origin=(17.5, 0, 0), invert=False)
    render(
        cut_x, title="Transverse cut at x=17.5 mm, t=20 ms (wall ring)",
        out_path=gallery / "cross_x_mid_t20.png",
        purkinje=purkinje_poly, terminals=terminals,
        azimuth=180, elevation=0, zoom=1.4,
    )

    # 4) Combined two-plane cut (y>=0 AND z>=0) — shows both inner and outer
    combo = m20.clip(normal="y", origin=(0, 0, 0), invert=False).clip(
        normal="z", origin=(0, 0, 0), invert=False
    )
    render(
        combo, title="Two-plane cut (y>=0, z>=0) at t=20 ms",
        out_path=gallery / "cross_y0_z0_t20.png",
        purkinje=purkinje_poly, terminals=terminals,
        azimuth=-45, elevation=30,
    )

    print(f"wrote {gallery}", file=sys.stderr)


if __name__ == "__main__":
    main()
