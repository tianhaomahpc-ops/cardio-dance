#!/usr/bin/env python3
"""Render an MP4 of LV electromechanical evolution overlaid on the static
Purkinje + fiber geometry.

Three panels per frame, all sharing the same camera:
  Left:   warped (deformed) LV, colored by Vm, with Purkinje cable + PVJ
          terminals + fiber direction tubes overlaid.
  Middle: same field but with a sagittal clip plane y = 0, revealing the
          transmural Vm pattern and the interior Purkinje cable.
  Right:  warped LV, colored by Ta_kPa (active tension), with the same
          Purkinje + fiber overlay.

The fiber tubes are sampled from the simulation's stored fiber_f field at
the warped (current) point positions so they track the deformation; the
Purkinje cable is drawn at its reference position (it is a 1 D solver on
the undeformed cavity).

Wrap with xvfb-run on headless boxes.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "3.3")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

import numpy as np
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_lv_purkinje_fiber import (  # noqa: E402
    read_purkinje_network,
    build_purkinje_polydata,
    write_png,
)


def list_frames(case_dir: Path) -> list[Path]:
    pattern = case_dir / "heart/monodomain/Cycle*/data.pvtu"
    files = sorted(glob.glob(str(pattern)))
    if not files:
        raise SystemExit(f"no PVTU frames under {pattern}")
    return [Path(f) for f in files]


def read_pvtu(path: Path) -> vtk.vtkUnstructuredGrid:
    r = vtk.vtkXMLPUnstructuredGridReader()
    r.SetFileName(str(path))
    r.Update()
    return r.GetOutput()


def warp_by_displacement(ug: vtk.vtkUnstructuredGrid, factor: float) -> vtk.vtkUnstructuredGrid:
    if ug.GetPointData().GetArray("displacement") is None:
        return ug
    ug.GetPointData().SetActiveVectors("displacement")
    warp = vtk.vtkWarpVector()
    warp.SetInputData(ug)
    warp.SetScaleFactor(factor)
    warp.Update()
    return warp.GetOutput()


def clip_ug(ug: vtk.vtkUnstructuredGrid, normal, origin) -> vtk.vtkUnstructuredGrid:
    plane = vtk.vtkPlane()
    plane.SetNormal(*normal)
    plane.SetOrigin(*origin)
    clipper = vtk.vtkClipDataSet()
    clipper.SetInputData(ug)
    clipper.SetClipFunction(plane)
    clipper.InsideOutOff()
    clipper.Update()
    return clipper.GetOutput()


def make_vm_lut(vm_lo: float, vm_hi: float) -> vtk.vtkLookupTable:
    lut = vtk.vtkLookupTable()
    lut.SetRange(vm_lo, vm_hi)
    lut.SetHueRange(0.66, 0.0)
    lut.SetSaturationRange(0.9, 0.9)
    lut.SetValueRange(0.85, 1.0)
    lut.Build()
    return lut


def make_ta_lut(ta_hi: float) -> vtk.vtkLookupTable:
    lut = vtk.vtkLookupTable()
    lut.SetRange(0.0, ta_hi)
    lut.SetHueRange(0.55, 0.0)
    lut.SetSaturationRange(0.0, 1.0)
    lut.SetValueRange(0.95, 0.95)
    lut.Build()
    return lut


def make_field_mapper(ug, field: str, lut) -> vtk.vtkDataSetMapper:
    m = vtk.vtkDataSetMapper()
    m.SetInputData(ug)
    m.SetScalarModeToUsePointFieldData()
    m.SelectColorArray(field)
    m.SetLookupTable(lut)
    m.SetScalarRange(*lut.GetRange())
    m.SetUseLookupTableScalarRange(True)
    return m


def build_fiber_glyph_polydata(ug: vtk.vtkUnstructuredGrid,
                               sample_stride: int,
                               scale_mm: float) -> vtk.vtkPolyData:
    """Subsample fiber_f from a frame's VTU and emit short line segments
    around each chosen point, suitable for tubing."""
    n_pts = ug.GetNumberOfPoints()
    fiber_arr = ug.GetPointData().GetArray("fiber_f")
    if fiber_arr is None or n_pts == 0:
        return vtk.vtkPolyData()
    fiber = vtk_to_numpy(fiber_arr).reshape(-1, 3)
    pts_np = vtk_to_numpy(ug.GetPoints().GetData()).reshape(-1, 3)
    pick = np.arange(0, n_pts, sample_stride)
    p0 = pts_np[pick] - 0.5 * scale_mm * fiber[pick]
    p1 = pts_np[pick] + 0.5 * scale_mm * fiber[pick]
    all_pts = np.vstack([p0, p1])
    pts = vtk.vtkPoints()
    pts.SetData(numpy_to_vtk(all_pts))
    lines = vtk.vtkCellArray()
    n = len(pick)
    for k in range(n):
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, k)
        line.GetPointIds().SetId(1, n + k)
        lines.InsertNextCell(line)
    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    pd.SetLines(lines)
    return pd


def make_purkinje_actors(purkinje_pd: vtk.vtkPolyData,
                          terms_pd: vtk.vtkPolyData,
                          tube_radius: float = 0.10,
                          term_radius: float = 0.20):
    tubes = vtk.vtkTubeFilter()
    tubes.SetInputData(purkinje_pd)
    tubes.SetRadius(tube_radius)
    tubes.SetNumberOfSides(6)
    tubes.Update()
    purk_mapper = vtk.vtkPolyDataMapper()
    purk_mapper.SetInputConnection(tubes.GetOutputPort())
    purk_mapper.ScalarVisibilityOff()
    purk_actor = vtk.vtkActor()
    purk_actor.SetMapper(purk_mapper)
    purk_actor.GetProperty().SetColor(1.0, 0.25, 0.2)
    purk_actor.GetProperty().SetOpacity(0.95)

    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(term_radius)
    sphere.SetThetaResolution(8)
    sphere.SetPhiResolution(8)
    glyph = vtk.vtkGlyph3D()
    glyph.SetSourceConnection(sphere.GetOutputPort())
    glyph.SetInputData(terms_pd)
    glyph.SetScaleModeToDataScalingOff()
    glyph.Update()
    term_mapper = vtk.vtkPolyDataMapper()
    term_mapper.SetInputConnection(glyph.GetOutputPort())
    term_mapper.ScalarVisibilityOff()
    term_actor = vtk.vtkActor()
    term_actor.SetMapper(term_mapper)
    term_actor.GetProperty().SetColor(1.0, 0.85, 0.2)
    return purk_actor, term_actor


def make_fiber_actor(fiber_pd: vtk.vtkPolyData, tube_radius: float = 0.05):
    tubes = vtk.vtkTubeFilter()
    tubes.SetInputData(fiber_pd)
    tubes.SetRadius(tube_radius)
    tubes.SetNumberOfSides(3)
    tubes.Update()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(tubes.GetOutputPort())
    mapper.ScalarVisibilityOff()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.95, 0.95, 0.95)
    actor.GetProperty().SetOpacity(0.85)
    return actor


def add_scalar_bar(ren, lut, title, position=(0.86, 0.20)):
    sb = vtk.vtkScalarBarActor()
    sb.SetLookupTable(lut)
    sb.SetTitle(title)
    sb.SetNumberOfLabels(5)
    sb.SetWidth(0.10)
    sb.SetHeight(0.55)
    sb.SetPosition(*position)
    sb.GetTitleTextProperty().SetColor(1, 1, 1)
    sb.GetTitleTextProperty().SetFontSize(14)
    sb.GetLabelTextProperty().SetColor(1, 1, 1)
    sb.GetLabelTextProperty().SetFontSize(11)
    ren.AddActor2D(sb)


def add_label(ren, text, height_px):
    t = vtk.vtkTextActor()
    t.SetInput(text)
    t.GetTextProperty().SetFontSize(18)
    t.GetTextProperty().SetColor(0.95, 0.95, 0.95)
    t.SetDisplayPosition(12, height_px - 28)
    ren.AddActor2D(t)


def configure_camera(ren, focus, cam_dir, distance, view_up=(0, 0, 1)):
    cam = ren.GetActiveCamera()
    cam.SetFocalPoint(*focus)
    cam.SetPosition(focus[0] + distance * cam_dir[0],
                    focus[1] + distance * cam_dir[1],
                    focus[2] + distance * cam_dir[2])
    cam.SetViewUp(*view_up)
    ren.ResetCameraClippingRange()


def render_frame_panels(
    ug_warp: vtk.vtkUnstructuredGrid,
    purkinje_pd: vtk.vtkPolyData,
    terms_pd: vtk.vtkPolyData,
    lut_vm: vtk.vtkLookupTable,
    lut_ta: vtk.vtkLookupTable,
    width: int,
    height: int,
    time_ms: float,
    cycle_idx: int,
    fiber_stride: int,
    fiber_scale: float,
) -> np.ndarray:
    """Render a 3-panel composite frame as a HxWx3 uint8 numpy array."""
    fiber_pd = build_fiber_glyph_polydata(ug_warp, fiber_stride, fiber_scale)
    ug_cut = clip_ug(ug_warp, normal=(0.0, -1.0, 0.0), origin=(0.0, 0.0, -5.0))
    fiber_pd_cut = build_fiber_glyph_polydata(ug_cut, fiber_stride, fiber_scale)

    panel_w = width // 3

    cam_dir = (0.65, -0.75, 0.30)
    focus = (0.0, 0.0, -5.0)
    cam_dist = 60.0

    panels = []
    for kind in ("vm_full", "vm_cut", "ta_full"):
        ren = vtk.vtkRenderer()
        ren.SetBackground(0.06, 0.06, 0.09)

        if kind == "vm_full":
            field_ug = ug_warp
            lut = lut_vm
            field = "Vm"
            label = f"Vm (mV)  intact   t={time_ms:6.2f} ms"
            scalar_bar_title = "Vm (mV)"
            fpd = fiber_pd
        elif kind == "vm_cut":
            field_ug = ug_cut
            lut = lut_vm
            field = "Vm"
            label = f"Vm (mV)  sagittal cut y=0"
            scalar_bar_title = "Vm (mV)"
            fpd = fiber_pd_cut
        else:  # ta_full
            field_ug = ug_warp
            lut = lut_ta
            field = "Ta_kPa"
            label = f"Ta (kPa)  intact"
            scalar_bar_title = "Ta (kPa)"
            fpd = fiber_pd

        mapper = make_field_mapper(field_ug, field, lut)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetAmbient(0.30)
        actor.GetProperty().SetDiffuse(0.65)
        actor.GetProperty().SetOpacity(0.85)
        ren.AddActor(actor)

        purk_actor, term_actor = make_purkinje_actors(purkinje_pd, terms_pd)
        ren.AddActor(purk_actor)
        ren.AddActor(term_actor)

        fib_actor = make_fiber_actor(fpd)
        ren.AddActor(fib_actor)

        add_scalar_bar(ren, lut, scalar_bar_title)
        add_label(ren, label, height)

        configure_camera(ren, focus, cam_dir, cam_dist)

        rw = vtk.vtkRenderWindow()
        rw.SetOffScreenRendering(1)
        rw.SetSize(panel_w, height)
        rw.AddRenderer(ren)
        rw.Render()

        w2i = vtk.vtkWindowToImageFilter()
        w2i.SetInput(rw)
        w2i.SetInputBufferTypeToRGB()
        w2i.ReadFrontBufferOff()
        w2i.Update()
        img = w2i.GetOutput()
        dims = img.GetDimensions()
        arr = np.frombuffer(img.GetPointData().GetScalars(), dtype=np.uint8)
        arr = arr.reshape(dims[1], dims[0], 3)
        panels.append(np.ascontiguousarray(arr[::-1]))

    return np.hstack(panels)


def scan_ranges(pvtus: list[Path]):
    """Pass 1: bound Vm and Ta_kPa for stable colormaps across the whole video."""
    vm_lo_phys, vm_hi_phys = -100.0, 60.0
    ta_hi_phys = 200.0
    vm_min, vm_max = +np.inf, -np.inf
    ta_max = -np.inf
    for f in pvtus:
        ug = read_pvtu(f)
        pd = ug.GetPointData()
        vm = pd.GetArray("Vm")
        if vm is not None:
            r = vm.GetRange()
            if vm_lo_phys < r[0] < vm_hi_phys:
                vm_min = min(vm_min, r[0])
            if vm_lo_phys < r[1] < vm_hi_phys:
                vm_max = max(vm_max, r[1])
        ta = pd.GetArray("Ta_kPa")
        if ta is not None:
            r = ta.GetRange()
            if 0.0 <= r[1] < ta_hi_phys:
                ta_max = max(ta_max, r[1])
    if not np.isfinite(vm_min) or not np.isfinite(vm_max):
        vm_min, vm_max = -90.0, 30.0
    if not np.isfinite(ta_max) or ta_max <= 0.0:
        ta_max = 50.0
    return vm_min, vm_max, ta_max


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="output/lv_ellipsoid_em_stewart")
    ap.add_argument("--purkinje", default="config/lv_purkinje.network")
    ap.add_argument("--frames-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--width", type=int, default=1800)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--warp", type=float, default=1.0)
    ap.add_argument("--dt-pde-ms", type=float, default=0.05)
    ap.add_argument("--fiber-stride", type=int, default=24,
                    help="subsample stride for fiber glyphs (over output points)")
    ap.add_argument("--fiber-scale", type=float, default=1.4,
                    help="fiber segment length in mm")
    args = ap.parse_args()

    case = Path(args.case)
    frames_dir = Path(args.frames_dir or case / "geom_overlay_frames")
    out_mp4 = Path(args.out or f"docs/media/{case.name}_geom_overlay.mp4")

    pvtus = list_frames(case)
    print(f"found {len(pvtus)} frames in {case}", flush=True)
    vm_lo, vm_hi, ta_hi = scan_ranges(pvtus)
    print(f"Vm range [{vm_lo:.2f}, {vm_hi:.2f}] mV   Ta max {ta_hi:.2f} kPa", flush=True)
    lut_vm = make_vm_lut(vm_lo, vm_hi)
    lut_ta = make_ta_lut(max(ta_hi, 1.0))

    p_nodes, p_term, p_edges = read_purkinje_network(Path(args.purkinje))
    purkinje_pd = build_purkinje_polydata(p_nodes, p_edges)
    terms_pd = vtk.vtkPolyData()
    tpts = vtk.vtkPoints()
    tpts.SetData(numpy_to_vtk(p_nodes[p_term].copy()))
    terms_pd.SetPoints(tpts)
    verts = vtk.vtkCellArray()
    for k in range(int(p_term.sum())):
        verts.InsertNextCell(1)
        verts.InsertCellPoint(k)
    terms_pd.SetVerts(verts)

    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    cycle_re = re.compile(r"Cycle(\d+)")
    for i, f in enumerate(pvtus):
        m = cycle_re.search(str(f))
        cyc = int(m.group(1)) if m else i
        t_ms = cyc * args.dt_pde_ms
        ug = read_pvtu(f)
        ug_warp = warp_by_displacement(ug, args.warp)
        img = render_frame_panels(
            ug_warp, purkinje_pd, terms_pd, lut_vm, lut_ta,
            args.width, args.height, t_ms, cyc,
            args.fiber_stride, args.fiber_scale,
        )
        write_png(img, frames_dir / f"frame_{i:04d}.png")
        if (i % 5 == 0) or (i == len(pvtus) - 1):
            print(f"  frame {i+1}/{len(pvtus)}  t={t_ms:.2f} ms", flush=True)

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(args.fps),
        "-i", str(frames_dir / "frame_%04d.png"),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True)
    print(f"wrote: {out_mp4}", flush=True)


if __name__ == "__main__":
    main()
