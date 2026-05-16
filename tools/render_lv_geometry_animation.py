#!/usr/bin/env python3
"""Render an MP4 animation of LV mesh + Purkinje cable + helical fiber field.

Three phases (24 fps total, ~8 sec):
  Phase 1 (0.0 - 2.0 s, 48 frames): orbit the camera 360 degrees around the
    intact LV. Shows the epicardial surface fibers and Purkinje cable
    peeking through the cavity.
  Phase 2 (2.0 - 5.0 s, 72 frames): orbit again but with a clip plane whose
    normal tracks the camera, so the half FACING AWAY from camera is cut.
    Reveals the transmural fibers (counter-rotating helices) and the
    interior Purkinje cable.
  Phase 3 (5.0 - 8.0 s, 72 frames): camera stays at a basal-lateral
    position; a transverse clip plane sweeps from z=+4 mm (base) to
    z=-14 mm (near apex), revealing the helical pattern at each level.

Wrap with xvfb-run on headless boxes:
  xvfb-run -a -s "-screen 0 1920x1080x24" \\
      python3 tools/render_lv_geometry_animation.py
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "3.3")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

import numpy as np
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_lv_purkinje_fiber import (  # noqa: E402
    read_mfem_mesh,
    read_fiber_gf,
    read_purkinje_network,
    build_lv_grid,
    build_purkinje_polydata,
    build_fiber_glyphs,
    render_panel,
    write_png,
)


def build_terminals_polydata(p_nodes: np.ndarray, p_term: np.ndarray) -> vtk.vtkPolyData:
    pts = vtk.vtkPoints()
    pts.SetData(numpy_to_vtk(p_nodes[p_term].copy()))
    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    verts = vtk.vtkCellArray()
    for k in range(int(p_term.sum())):
        verts.InsertNextCell(1)
        verts.InsertCellPoint(k)
    pd.SetVerts(verts)
    return pd


def orbit_camera_dir(theta: float, elevation_deg: float) -> tuple[float, float, float]:
    el = math.radians(elevation_deg)
    cz = math.sin(el)
    rho = math.cos(el)
    return (rho * math.cos(theta), rho * math.sin(theta), cz)


def main():
    repo = Path(__file__).resolve().parent.parent
    mesh_path = repo / "benchmarks/lv_ellipsoid/lv_ellipsoid.mesh"
    fiber_path = repo / "benchmarks/lv_ellipsoid/fiber_f.gf"
    purkinje_path = repo / "config/lv_purkinje.network"
    frames_dir = repo / "output/lv_geom_anim_frames"
    out_mp4 = repo / "docs/media/lv_geometry_overview.mp4"

    print(f"reading mesh: {mesh_path}", flush=True)
    verts, hexes = read_mfem_mesh(mesh_path)

    print(f"reading fibers: {fiber_path}", flush=True)
    fiber_f = read_fiber_gf(fiber_path, verts.shape[0])

    print(f"reading purkinje: {purkinje_path}", flush=True)
    p_nodes, p_term, p_edges = read_purkinje_network(purkinje_path)

    lv_ug = build_lv_grid(verts, hexes)
    purkinje_pd = build_purkinje_polydata(p_nodes, p_edges)
    terms_pd = build_terminals_polydata(p_nodes, p_term)
    fiber_pd = build_fiber_glyphs(verts, fiber_f, sample_stride=6, scale_mm=1.8)

    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    W, H = 960, 720
    FPS = 24
    N_INTACT = 48
    N_CUT_ORBIT = 72
    N_TRANSVERSE = 72

    idx = 0

    # Phase 1: orbit intact LV.
    print(f"phase 1: orbit intact ({N_INTACT} frames)", flush=True)
    for k in range(N_INTACT):
        theta = 2.0 * math.pi * k / N_INTACT
        cam_dir = orbit_camera_dir(theta, elevation_deg=18.0)
        t_sec = idx / FPS
        title = f"t={t_sec:4.1f}s  orbit (intact)   theta={math.degrees(theta):5.1f} deg"
        arr = render_panel(
            W, H, lv_ug, purkinje_pd, terms_pd, fiber_pd,
            camera_dir=cam_dir, view_up=(0, 0, 1),
            clip_normal=None, title=title,
        )
        write_png(arr, frames_dir / f"frame_{idx:05d}.png")
        idx += 1

    # Phase 2: orbit with cut plane tracking camera (normal points toward camera).
    print(f"phase 2: orbit cutaway ({N_CUT_ORBIT} frames)", flush=True)
    for k in range(N_CUT_ORBIT):
        theta = 2.0 * math.pi * k / N_CUT_ORBIT
        cam_dir = orbit_camera_dir(theta, elevation_deg=18.0)
        # Cut keeps the half on the camera side; normal points TOWARD camera.
        clip_n = (math.cos(theta), math.sin(theta), 0.0)
        t_sec = idx / FPS
        title = f"t={t_sec:4.1f}s  cutaway orbit   theta={math.degrees(theta):5.1f} deg"
        arr = render_panel(
            W, H, lv_ug, purkinje_pd, terms_pd, fiber_pd,
            camera_dir=cam_dir, view_up=(0, 0, 1),
            clip_normal=clip_n, clip_origin=(0.0, 0.0, -5.0), title=title,
        )
        write_png(arr, frames_dir / f"frame_{idx:05d}.png")
        idx += 1

    # Phase 3: fixed basal-lateral camera, transverse plane sweeps base -> apex.
    print(f"phase 3: transverse sweep ({N_TRANSVERSE} frames)", flush=True)
    z_top = 4.0
    z_bot = -14.0
    cam_dir = orbit_camera_dir(theta=math.radians(-65.0), elevation_deg=35.0)
    for k in range(N_TRANSVERSE):
        s = k / max(N_TRANSVERSE - 1, 1)
        z_cut = z_top + s * (z_bot - z_top)
        t_sec = idx / FPS
        title = f"t={t_sec:4.1f}s  transverse cut  z={z_cut:+5.1f} mm"
        arr = render_panel(
            W, H, lv_ug, purkinje_pd, terms_pd, fiber_pd,
            camera_dir=cam_dir, view_up=(0, 0, 1),
            clip_normal=(0.0, 0.0, -1.0), clip_origin=(0.0, 0.0, z_cut),
            title=title,
        )
        write_png(arr, frames_dir / f"frame_{idx:05d}.png")
        idx += 1

    # Stitch with ffmpeg.
    print(f"stitching {idx} frames -> {out_mp4}", flush=True)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(FPS),
        "-i", str(frames_dir / "frame_%05d.png"),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "20",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True)
    print(f"wrote: {out_mp4}", flush=True)


if __name__ == "__main__":
    main()
