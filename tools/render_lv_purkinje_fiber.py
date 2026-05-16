#!/usr/bin/env python3
"""Render LV mesh + Purkinje network + fiber field from multiple angles.

Outputs:
  docs/media/lv_geometry_overview.png   (3x2 grid: 3 camera angles + 3 cuts)

Each panel shows:
  - LV myocardial shell as a semi-transparent surface (epi side)
  - Purkinje network as red line segments; terminal nodes as small spheres
  - Fiber f direction as short blue line segments at a subsampled set of
    vertices, length scaled to match local element size
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "3.3")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

import numpy as np
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk


# ---------- IO helpers ----------

def read_mfem_mesh(path: Path):
    with path.open() as f:
        lines = f.readlines()
    i = 0
    elems = None
    verts = None
    while i < len(lines):
        tag = lines[i].strip()
        if tag == "elements":
            n = int(lines[i + 1])
            elems = []
            for k in range(n):
                parts = lines[i + 2 + k].split()
                # MFEM element line: attr geom_type v0 v1 ...
                # geom 5 = CUBE (hex)
                elems.append([int(x) for x in parts[2:]])
            i += 2 + n
        elif tag == "vertices":
            n = int(lines[i + 1])
            sdim = int(lines[i + 2])
            verts = np.array(
                [[float(x) for x in lines[i + 3 + k].split()] for k in range(n)],
                dtype=np.float64,
            )
            assert verts.shape == (n, sdim)
            i += 3 + n
        else:
            i += 1
    return verts, elems


def read_fiber_gf(path: Path, n_vertices: int) -> np.ndarray:
    with path.open() as f:
        # Skip header until blank line, then read n_vertices vec3.
        header_done = False
        rows = []
        for ln in f:
            if not header_done:
                if ln.strip() == "":
                    header_done = True
                continue
            parts = ln.split()
            if len(parts) == 3:
                rows.append([float(x) for x in parts])
            if len(rows) == n_vertices:
                break
    return np.asarray(rows, dtype=np.float64)


def read_purkinje_network(path: Path):
    with path.open() as f:
        first = f.readline().split()
        n_nodes, n_edges = int(first[0]), int(first[1])
        nodes = np.zeros((n_nodes, 3), dtype=np.float64)
        is_term = np.zeros(n_nodes, dtype=bool)
        for k in range(n_nodes):
            parts = f.readline().split()
            nodes[k] = [float(parts[0]), float(parts[1]), float(parts[2])]
            is_term[k] = int(parts[3]) != 0
        edges = np.zeros((n_edges, 2), dtype=np.int64)
        for k in range(n_edges):
            parts = f.readline().split()
            # File uses 1-based indices per the generator; subtract 1.
            edges[k] = [int(parts[0]) - 1, int(parts[1]) - 1]
    return nodes, is_term, edges


# ---------- VTK builders ----------

def build_lv_grid(verts: np.ndarray, hexes: list[list[int]]) -> vtk.vtkUnstructuredGrid:
    pts = vtk.vtkPoints()
    pts.SetData(numpy_to_vtk(verts.copy()))
    ug = vtk.vtkUnstructuredGrid()
    ug.SetPoints(pts)
    ug.Allocate(len(hexes))
    hex_cell = vtk.vtkHexahedron()
    for conn in hexes:
        ids = hex_cell.GetPointIds()
        for j, v in enumerate(conn):
            ids.SetId(j, v)
        ug.InsertNextCell(hex_cell.GetCellType(), ids)
    return ug


def build_purkinje_polydata(nodes: np.ndarray, edges: np.ndarray) -> vtk.vtkPolyData:
    pts = vtk.vtkPoints()
    pts.SetData(numpy_to_vtk(nodes.copy()))
    lines = vtk.vtkCellArray()
    for a, b in edges:
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, int(a))
        line.GetPointIds().SetId(1, int(b))
        lines.InsertNextCell(line)
    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    pd.SetLines(lines)
    return pd


def build_fiber_glyphs(verts: np.ndarray, fiber_f: np.ndarray,
                       sample_stride: int, scale_mm: float) -> vtk.vtkPolyData:
    pick = np.arange(0, verts.shape[0], sample_stride)
    p0 = verts[pick] - 0.5 * scale_mm * fiber_f[pick]
    p1 = verts[pick] + 0.5 * scale_mm * fiber_f[pick]
    n = len(pick)
    pts = vtk.vtkPoints()
    all_pts = np.vstack([p0, p1])
    pts.SetData(numpy_to_vtk(all_pts))

    lines = vtk.vtkCellArray()
    # Color by helical angle: alpha = atan2(f.e_long, f.e_phi). e_phi at point
    # x=(x,y,z) is (-sin phi, cos phi, 0) = (-y, x, 0)/sqrt(x^2+y^2); e_long is
    # roughly +/-z hat away from base. Use signed projection of f onto z to get
    # alpha sign: alpha = asin(-f.z) for our convention f = cos a e_phi - sin a e_long.
    alpha_rad = np.arcsin(np.clip(-fiber_f[pick, 2], -1.0, 1.0))
    alpha_deg = np.degrees(alpha_rad)

    cell_data = vtk.vtkFloatArray()
    cell_data.SetName("alpha_deg")
    cell_data.SetNumberOfComponents(1)
    cell_data.SetNumberOfTuples(n)

    for k in range(n):
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, k)
        line.GetPointIds().SetId(1, n + k)
        lines.InsertNextCell(line)
        cell_data.SetTuple1(k, float(alpha_deg[k]))

    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    pd.SetLines(lines)
    pd.GetCellData().AddArray(cell_data)
    pd.GetCellData().SetActiveScalars("alpha_deg")
    return pd


# ---------- Rendering ----------

def make_lookup_table_diverging():
    # Cool->white->warm centered at 0 (helical angle).
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(128)
    for i in range(128):
        t = i / 127.0  # 0 -> -60 deg, 1 -> +60 deg
        if t < 0.5:
            s = (0.5 - t) * 2.0  # 1 at endo, 0 at mid
            r = 0.20 + 0.80 * (1 - s)
            g = 0.30 + 0.70 * (1 - s)
            b = 1.00
        else:
            s = (t - 0.5) * 2.0
            r = 1.00
            g = 0.30 + 0.70 * (1 - s)
            b = 0.20 + 0.80 * (1 - s)
        lut.SetTableValue(i, r, g, b, 1.0)
    lut.SetRange(-60.0, 60.0)
    lut.Build()
    return lut


def render_panel(
    width: int,
    height: int,
    lv_ug: vtk.vtkUnstructuredGrid,
    purkinje_pd: vtk.vtkPolyData,
    terms_pd: vtk.vtkPolyData,
    fiber_pd: vtk.vtkPolyData,
    camera_dir: tuple[float, float, float],
    view_up: tuple[float, float, float],
    clip_normal: tuple[float, float, float] | None,
    title: str,
    clip_origin: tuple[float, float, float] = (0.0, 0.0, -5.0),
    cam_focus: tuple[float, float, float] = (0.0, 0.0, -5.0),
    cam_distance: float = 65.0,
    show_colorbar: bool = True,
) -> np.ndarray:
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.07, 0.07, 0.09)

    # LV shell (semi-transparent surface; optionally clipped).
    lv_source = lv_ug
    if clip_normal is not None:
        plane = vtk.vtkPlane()
        plane.SetOrigin(*clip_origin)
        plane.SetNormal(*clip_normal)
        clipper = vtk.vtkClipDataSet()
        clipper.SetInputData(lv_ug)
        clipper.SetClipFunction(plane)
        clipper.InsideOutOff()
        clipper.Update()
        lv_source = clipper.GetOutput()

    lv_geom = vtk.vtkGeometryFilter()
    lv_geom.SetInputData(lv_source)
    lv_geom.Update()
    lv_mapper = vtk.vtkPolyDataMapper()
    lv_mapper.SetInputConnection(lv_geom.GetOutputPort())
    lv_mapper.ScalarVisibilityOff()
    lv_actor = vtk.vtkActor()
    lv_actor.SetMapper(lv_mapper)
    lv_actor.GetProperty().SetColor(0.78, 0.72, 0.65)
    lv_actor.GetProperty().SetOpacity(0.35 if clip_normal is None else 0.55)
    lv_actor.GetProperty().SetEdgeVisibility(False)
    renderer.AddActor(lv_actor)

    # Purkinje lines (tubed for visibility).
    purk_tubes = vtk.vtkTubeFilter()
    purk_tubes.SetInputData(purkinje_pd)
    purk_tubes.SetRadius(0.08)
    purk_tubes.SetNumberOfSides(6)
    purk_tubes.Update()
    purk_mapper = vtk.vtkPolyDataMapper()
    purk_mapper.SetInputConnection(purk_tubes.GetOutputPort())
    purk_mapper.ScalarVisibilityOff()
    purk_actor = vtk.vtkActor()
    purk_actor.SetMapper(purk_mapper)
    purk_actor.GetProperty().SetColor(1.0, 0.25, 0.2)
    purk_actor.GetProperty().SetOpacity(0.9)
    renderer.AddActor(purk_actor)

    # Terminal spheres.
    term_glyph = vtk.vtkGlyph3D()
    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(0.18)
    sphere.SetThetaResolution(8)
    sphere.SetPhiResolution(8)
    term_glyph.SetSourceConnection(sphere.GetOutputPort())
    term_glyph.SetInputData(terms_pd)
    term_glyph.SetScaleModeToDataScalingOff()
    term_glyph.Update()
    term_mapper = vtk.vtkPolyDataMapper()
    term_mapper.SetInputConnection(term_glyph.GetOutputPort())
    term_mapper.ScalarVisibilityOff()
    term_actor = vtk.vtkActor()
    term_actor.SetMapper(term_mapper)
    term_actor.GetProperty().SetColor(1.0, 0.85, 0.2)
    renderer.AddActor(term_actor)

    # Fiber line segments (tubed) colored by axial alignment.
    fib_tubes = vtk.vtkTubeFilter()
    fib_tubes.SetInputData(fiber_pd)
    fib_tubes.SetRadius(0.10)
    fib_tubes.SetNumberOfSides(4)
    fib_tubes.Update()
    fib_mapper = vtk.vtkPolyDataMapper()
    fib_mapper.SetInputConnection(fib_tubes.GetOutputPort())
    lut = make_lookup_table_diverging()
    fib_mapper.SetLookupTable(lut)
    fib_mapper.SetScalarRange(-60.0, 60.0)
    fib_mapper.SetScalarModeToUseCellData()
    fib_mapper.SelectColorArray("alpha_deg")
    fib_mapper.ScalarVisibilityOn()
    fib_actor = vtk.vtkActor()
    fib_actor.SetMapper(fib_mapper)
    renderer.AddActor(fib_actor)

    # Colorbar for fiber helical angle.
    if show_colorbar:
        cbar = vtk.vtkScalarBarActor()
        cbar.SetLookupTable(lut)
        cbar.SetTitle("alpha (deg)")
        cbar.SetNumberOfLabels(5)
        cbar.SetMaximumWidthInPixels(60)
        cbar.SetMaximumHeightInPixels(int(height * 0.55))
        cbar.GetTitleTextProperty().SetColor(0.95, 0.95, 0.95)
        cbar.GetTitleTextProperty().SetFontSize(14)
        cbar.GetLabelTextProperty().SetColor(0.95, 0.95, 0.95)
        cbar.GetLabelTextProperty().SetFontSize(12)
        cbar.SetPosition(0.90, 0.22)
        renderer.AddActor2D(cbar)

    # Title.
    txt = vtk.vtkTextActor()
    txt.SetInput(title)
    txt.GetTextProperty().SetFontSize(20)
    txt.GetTextProperty().SetColor(0.95, 0.95, 0.95)
    txt.SetDisplayPosition(15, height - 30)
    renderer.AddActor2D(txt)

    # Camera.
    cam = renderer.GetActiveCamera()
    cam.SetFocalPoint(*cam_focus)
    cam.SetPosition(cam_focus[0] + cam_distance * camera_dir[0],
                    cam_focus[1] + cam_distance * camera_dir[1],
                    cam_focus[2] + cam_distance * camera_dir[2])
    cam.SetViewUp(*view_up)
    renderer.ResetCameraClippingRange()

    rw = vtk.vtkRenderWindow()
    rw.SetOffScreenRendering(1)
    rw.SetSize(width, height)
    rw.AddRenderer(renderer)
    rw.Render()

    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(rw)
    w2i.SetInputBufferTypeToRGB()
    w2i.ReadFrontBufferOff()
    w2i.Update()
    vtk_img = w2i.GetOutput()
    dims = vtk_img.GetDimensions()
    arr = np.frombuffer(vtk_img.GetPointData().GetScalars(), dtype=np.uint8)
    arr = arr.reshape(dims[1], dims[0], 3)
    # VTK image origin is bottom-left; flip vertically.
    return np.ascontiguousarray(arr[::-1])


def compose_grid(panels: list[np.ndarray], cols: int) -> np.ndarray:
    rows = (len(panels) + cols - 1) // cols
    h, w, _ = panels[0].shape
    canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for k, p in enumerate(panels):
        r = k // cols
        c = k % cols
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = p
    return canvas


def write_png(arr: np.ndarray, path: Path):
    importer = vtk.vtkImageImport()
    importer.CopyImportVoidPointer(arr.tobytes(), arr.nbytes)
    importer.SetDataScalarTypeToUnsignedChar()
    importer.SetNumberOfScalarComponents(3)
    h, w, _ = arr.shape
    importer.SetWholeExtent(0, w - 1, 0, h - 1, 0, 0)
    importer.SetDataExtentToWholeExtent()
    importer.Update()

    # VTK PNG writer wants origin at bottom-left; flip back.
    flipped = np.ascontiguousarray(arr[::-1])
    importer2 = vtk.vtkImageImport()
    importer2.CopyImportVoidPointer(flipped.tobytes(), flipped.nbytes)
    importer2.SetDataScalarTypeToUnsignedChar()
    importer2.SetNumberOfScalarComponents(3)
    importer2.SetWholeExtent(0, w - 1, 0, h - 1, 0, 0)
    importer2.SetDataExtentToWholeExtent()
    importer2.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(str(path))
    writer.SetInputConnection(importer2.GetOutputPort())
    writer.Write()


# ---------- Main ----------

def main():
    repo = Path(__file__).resolve().parent.parent
    mesh_path = repo / "benchmarks/lv_ellipsoid/lv_ellipsoid.mesh"
    fiber_path = repo / "benchmarks/lv_ellipsoid/fiber_f.gf"
    purkinje_path = repo / "config/lv_purkinje.network"
    out_path = repo / "docs/media/lv_geometry_overview.png"

    print(f"reading mesh: {mesh_path}", flush=True)
    verts, hexes = read_mfem_mesh(mesh_path)
    print(f"  vertices={verts.shape[0]}  hexes={len(hexes)}", flush=True)

    print(f"reading fibers: {fiber_path}", flush=True)
    fiber_f = read_fiber_gf(fiber_path, verts.shape[0])

    print(f"reading purkinje: {purkinje_path}", flush=True)
    p_nodes, p_term, p_edges = read_purkinje_network(purkinje_path)
    print(f"  nodes={p_nodes.shape[0]}  edges={p_edges.shape[0]}  "
          f"terminals={int(p_term.sum())}", flush=True)

    lv_ug = build_lv_grid(verts, hexes)
    purkinje_pd = build_purkinje_polydata(p_nodes, p_edges)

    terms_pd = vtk.vtkPolyData()
    term_pts = vtk.vtkPoints()
    term_pts.SetData(numpy_to_vtk(p_nodes[p_term].copy()))
    terms_pd.SetPoints(term_pts)
    verts_cell = vtk.vtkCellArray()
    for k in range(int(p_term.sum())):
        verts_cell.InsertNextCell(1)
        verts_cell.InsertCellPoint(k)
    terms_pd.SetVerts(verts_cell)

    fiber_pd = build_fiber_glyphs(verts, fiber_f, sample_stride=6, scale_mm=1.8)

    W, H = 900, 760
    panels = []
    panels.append(render_panel(
        W, H, lv_ug, purkinje_pd, terms_pd, fiber_pd,
        camera_dir=(0.6, -0.7, 0.4), view_up=(0, 0, 1),
        clip_normal=None,
        title="(a) iso view  red=Purkinje cable, yellow=PVJ terminals",
    ))
    panels.append(render_panel(
        W, H, lv_ug, purkinje_pd, terms_pd, fiber_pd,
        camera_dir=(0.0, -1.0, 0.0), view_up=(0, 0, 1),
        clip_normal=None,
        title="(b) anterior view (-Y)",
    ))
    panels.append(render_panel(
        W, H, lv_ug, purkinje_pd, terms_pd, fiber_pd,
        camera_dir=(0.0, 0.0, 1.0), view_up=(0, 1, 0),
        clip_normal=None,
        title="(c) basal (+Z, looking down apex axis)",
    ))
    # Slice panels: cut so the half AWAY from camera is removed,
    # revealing fibers + Purkinje on the cut face.
    panels.append(render_panel(
        W, H, lv_ug, purkinje_pd, terms_pd, fiber_pd,
        camera_dir=(0.6, -0.7, 0.4), view_up=(0, 0, 1),
        clip_normal=(0.0, -1.0, 0.0), clip_origin=(0.0, 0.0, -5.0),
        title="(d) sagittal cut y=0 (transmural fibers visible)",
    ))
    panels.append(render_panel(
        W, H, lv_ug, purkinje_pd, terms_pd, fiber_pd,
        camera_dir=(0.0, 0.0, -1.0), view_up=(0, 1, 0),
        clip_normal=(0.0, 0.0, 1.0), clip_origin=(0.0, 0.0, -5.0),
        title="(e) transverse cut z=-5 mm (mid-cavity ring)",
    ))
    panels.append(render_panel(
        W, H, lv_ug, purkinje_pd, terms_pd, fiber_pd,
        camera_dir=(1.0, 0.0, 0.0), view_up=(0, 0, 1),
        clip_normal=(1.0, 0.0, 0.0), clip_origin=(0.0, 0.0, -5.0),
        title="(f) coronal cut x=0",
    ))

    grid = compose_grid(panels, cols=3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_png(grid, out_path)
    print(f"wrote: {out_path}", flush=True)


if __name__ == "__main__":
    main()
