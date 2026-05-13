#!/usr/bin/env python3
"""Render an MP4 of the LV electromechanical simulation from ParaView frames.

Loads `output/<case>/heart/monodomain/Cycle*/data.pvtu`, warps the mesh by the
saved `displacement` field, colors by `Vm` (left half) and by `Ta_kPa`
(right half), composites each frame, and stitches with ffmpeg.

Run after the simulation finishes. Requires python3-vtk9, numpy, ffmpeg.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonColor import vtkNamedColors  # noqa: F401

# Force software OSMesa-style rendering. The VTK 9.1 packaged in
# python3-vtk9 still defers to the X system for OpenGL context creation,
# so this script must be wrapped with xvfb-run (the run_lv_em_video.sh
# wrapper does so).
os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "3.3")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")


def list_frames(case_dir: Path) -> list[Path]:
    pattern = case_dir / "heart/monodomain/Cycle*/data.pvtu"
    files = sorted(glob.glob(str(pattern)))
    if not files:
        raise SystemExit(f"no PVTU frames under {pattern}")
    return [Path(f) for f in files]


def read_frame(pvtu_path: Path):
    reader = vtk.vtkXMLPUnstructuredGridReader()
    reader.SetFileName(str(pvtu_path))
    reader.Update()
    return reader.GetOutput()


def warp_by_displacement(ug: vtk.vtkUnstructuredGrid, factor: float = 1.0):
    arr = ug.GetPointData().GetArray("displacement")
    if arr is None:
        return ug
    ug.GetPointData().SetActiveVectors("displacement")
    warp = vtk.vtkWarpVector()
    warp.SetInputData(ug)
    warp.SetScaleFactor(factor)
    warp.Update()
    return warp.GetOutput()


def make_lut(scalar_range, name: str):
    lut = vtk.vtkLookupTable()
    lut.SetRange(*scalar_range)
    if name == "Vm":
        lut.SetHueRange(0.66, 0.0)  # blue -> red
        lut.SetSaturationRange(0.9, 0.9)
        lut.SetValueRange(0.85, 1.0)
    else:  # Ta_kPa
        lut.SetHueRange(0.55, 0.0)
        lut.SetSaturationRange(0.0, 1.0)
        lut.SetValueRange(0.95, 0.95)
    lut.Build()
    return lut


def make_mapper(ug, field: str, lut):
    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputData(ug)
    mapper.SetScalarModeToUsePointFieldData()
    mapper.SelectColorArray(field)
    mapper.SetLookupTable(lut)
    mapper.SetScalarRange(*lut.GetRange())
    mapper.SetUseLookupTableScalarRange(True)
    return mapper


def make_camera(bounds):
    # bounds = (xmin, xmax, ymin, ymax, zmin, zmax)
    cx = 0.5 * (bounds[0] + bounds[1])
    cy = 0.5 * (bounds[2] + bounds[3])
    cz = 0.5 * (bounds[4] + bounds[5])
    extent = max(bounds[1] - bounds[0],
                 bounds[3] - bounds[2],
                 bounds[5] - bounds[4])
    cam = vtk.vtkCamera()
    cam.SetFocalPoint(cx, cy, cz)
    cam.SetPosition(cx + 1.6 * extent, cy + 1.1 * extent, cz + 0.6 * extent)
    cam.SetViewUp(0.0, 0.0, 1.0)
    cam.ParallelProjectionOff()
    cam.SetClippingRange(0.1 * extent, 6.0 * extent)
    return cam


def make_panel_renderer(field, lut, title, camera, viewport):
    """Build one renderer for either Vm or Ta_kPa. Actor's mapper is rebound
    each frame in render_pair() — we keep one RenderWindow open across the
    whole video to avoid the X-server-context churn that crashes VTK 9.1."""
    ren = vtk.vtkRenderer()
    ren.SetBackground(0.06, 0.06, 0.08)
    ren.SetViewport(*viewport)
    ren.SetActiveCamera(camera)

    actor = vtk.vtkActor()
    actor.GetProperty().SetAmbient(0.25)
    actor.GetProperty().SetDiffuse(0.7)
    actor.GetProperty().EdgeVisibilityOff()
    ren.AddActor(actor)

    sb = vtk.vtkScalarBarActor()
    sb.SetLookupTable(lut)
    sb.SetTitle(title)
    sb.SetNumberOfLabels(5)
    sb.SetWidth(0.10)
    sb.SetHeight(0.50)
    sb.SetPosition(0.86, 0.25)
    sb.GetLabelTextProperty().SetColor(1, 1, 1)
    sb.GetTitleTextProperty().SetColor(1, 1, 1)
    sb.GetTitleTextProperty().SetFontSize(14)
    sb.GetLabelTextProperty().SetFontSize(12)
    ren.AddActor2D(sb)
    return ren, actor


def render_pair(ug, renderers, lut_pair, render_window, w2i, png_path):
    ren_vm, ren_ta = renderers
    actor_vm = ren_vm.GetActors().GetLastActor()
    actor_ta = ren_ta.GetActors().GetLastActor()
    actor_vm.SetMapper(make_mapper(ug, "Vm", lut_pair[0]))
    actor_ta.SetMapper(make_mapper(ug, "Ta_kPa", lut_pair[1]))
    render_window.Render()
    w2i.Modified()
    w2i.Update()
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(str(png_path))
    writer.SetInputData(w2i.GetOutput())
    writer.Write()




def cycle_to_time_ms(pvtu_path: Path, dt_pde_ms: float) -> float:
    m = re.search(r"Cycle(\d+)", str(pvtu_path))
    return int(m.group(1)) * dt_pde_ms if m else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="output/lv_ellipsoid_em_video",
                    help="output/<name> root containing heart/monodomain/Cycle*")
    ap.add_argument("--frames-dir", default="output/lv_ellipsoid_em_video/render_frames",
                    help="where to dump per-frame PNGs")
    ap.add_argument("--out", default="output/lv_ellipsoid_em_video/lv_em.mp4",
                    help="final mp4 path")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--width", type=int, default=720, help="single-panel width in px")
    ap.add_argument("--height", type=int, default=720, help="frame height in px")
    ap.add_argument("--warp", type=float, default=1.0,
                    help="displacement amplification (1.0 = physical)")
    ap.add_argument("--dt-pde-ms", type=float, default=0.05,
                    help="EP timestep (for cycle->ms conversion)")
    args = ap.parse_args()

    case_dir = Path(args.case)
    frames_dir = Path(args.frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    pvtus = list_frames(case_dir)
    print(f"found {len(pvtus)} frames in {case_dir}")

    # Pass 1: compute global ranges for Vm and Ta_kPa across all frames.
    # Clip to physical bounds so any single corrupted frame can't blow up
    # the color scale for the whole video.
    vm_lo_phys, vm_hi_phys = -100.0, 60.0   # mV
    ta_hi_phys = 200.0                       # kPa (Land typical max < 150)
    vm_min, vm_max = +np.inf, -np.inf
    ta_max = -np.inf
    for f in pvtus:
        ug = read_frame(f)
        pd = ug.GetPointData()
        vm_arr = pd.GetArray("Vm")
        if vm_arr is not None:
            r = vm_arr.GetRange()
            if vm_lo_phys < r[0] < vm_hi_phys:
                vm_min = min(vm_min, r[0])
            if vm_lo_phys < r[1] < vm_hi_phys:
                vm_max = max(vm_max, r[1])
        ta_arr = pd.GetArray("Ta_kPa")
        if ta_arr is not None:
            r = ta_arr.GetRange()
            if 0.0 <= r[1] < ta_hi_phys:
                ta_max = max(ta_max, r[1])
    if not np.isfinite(vm_min) or not np.isfinite(vm_max):
        vm_min, vm_max = -90.0, 30.0
    if not np.isfinite(ta_max) or ta_max <= 0.0:
        ta_max = 50.0
    print(f"Vm range: [{vm_min:.2f}, {vm_max:.2f}] mV")
    print(f"Ta range: [0.00, {ta_max:.2f}] kPa")

    lut_vm = make_lut((vm_min, vm_max), "Vm")
    lut_ta = make_lut((0.0, max(ta_max, 1.0)), "Ta_kPa")

    # Compute camera once from the first warped frame. Use the entire mesh
    # bounds (no warp dependency) so the camera frames the LV consistently
    # across all frames.
    first = warp_by_displacement(read_frame(pvtus[0]), factor=args.warp)
    cam = make_camera(first.GetBounds())

    # Single composite render window: left half = Vm, right half = Ta_kPa.
    full_size = (2 * args.width, args.height)
    ren_vm, _ = make_panel_renderer("Vm", lut_vm, "Vm (mV)", cam,
                                     viewport=(0.0, 0.0, 0.5, 1.0))
    ren_ta, _ = make_panel_renderer("Ta_kPa", lut_ta, "Ta (kPa)", cam,
                                     viewport=(0.5, 0.0, 1.0, 1.0))
    rw = vtk.vtkRenderWindow()
    rw.SetOffScreenRendering(1)
    rw.AddRenderer(ren_vm)
    rw.AddRenderer(ren_ta)
    rw.SetSize(*full_size)

    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(rw)
    w2i.SetInputBufferTypeToRGB()
    w2i.ReadFrontBufferOff()

    for i, f in enumerate(pvtus):
        ug = read_frame(f)
        ug_warp = warp_by_displacement(ug, factor=args.warp)
        png_path = frames_dir / f"frame_{i:04d}.png"
        render_pair(ug_warp, (ren_vm, ren_ta), (lut_vm, lut_ta), rw, w2i, png_path)
        if i % 10 == 0:
            print(f"  rendered {i+1}/{len(pvtus)}")

    print(f"frames written to {frames_dir}")

    # ffmpeg assemble.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-framerate", str(args.fps),
        "-i", str(frames_dir / "frame_%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        str(out_path),
    ]
    print("ffmpeg:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"wrote video: {out_path}")


if __name__ == "__main__":
    main()
