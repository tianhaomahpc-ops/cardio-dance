#!/usr/bin/env python3
import argparse
import pathlib
import sys

import gmsh


def parse_args():
  parser = argparse.ArgumentParser(
      description="Generate conforming heart+torso box tetra mesh with Gmsh")
  parser.add_argument("--out-mesh", required=True, help="Output .msh path")
  parser.add_argument("--h-mm", type=float, required=True, help="Target tetra edge length (mm)")
  parser.add_argument("--padding-mm", type=float, default=1.0, help="Torso padding around heart box (mm)")
  parser.add_argument("--heart-lx-mm", type=float, default=20.0)
  parser.add_argument("--heart-ly-mm", type=float, default=7.0)
  parser.add_argument("--heart-lz-mm", type=float, default=3.0)
  parser.add_argument("--heart-x0-mm", type=float, default=0.0)
  parser.add_argument("--heart-y0-mm", type=float, default=0.0)
  parser.add_argument("--heart-z0-mm", type=float, default=0.0)
  parser.add_argument("--heart-attr", type=int, default=1)
  parser.add_argument("--torso-attr", type=int, default=2)
  parser.add_argument("--msh-version", type=float, default=2.2)
  parser.add_argument("--optimize", type=int, default=0, help="1=enable Netgen optimization")
  return parser.parse_args()


def is_inside_heart(x, y, z, x0, y0, z0, lx, ly, lz):
  tol = 1e-9
  return (x0 + tol < x < x0 + lx - tol and
          y0 + tol < y < y0 + ly - tol and
          z0 + tol < z < z0 + lz - tol)


def count_tets_for_volume(vol_tag):
  elem_types, elem_tags, _ = gmsh.model.mesh.getElements(3, vol_tag)
  total = 0
  for et, tags in zip(elem_types, elem_tags):
    if et == 4:
      total += len(tags)
  return total


def main():
  args = parse_args()
  if args.h_mm <= 0.0:
    raise ValueError("--h-mm must be > 0")
  if args.padding_mm <= 0.0:
    raise ValueError("--padding-mm must be > 0")
  if args.heart_attr == args.torso_attr:
    raise ValueError("--heart-attr and --torso-attr must differ")

  out_path = pathlib.Path(args.out_mesh)
  out_path.parent.mkdir(parents=True, exist_ok=True)

  hx0 = args.heart_x0_mm
  hy0 = args.heart_y0_mm
  hz0 = args.heart_z0_mm
  hlx = args.heart_lx_mm
  hly = args.heart_ly_mm
  hlz = args.heart_lz_mm

  tx0 = hx0 - args.padding_mm
  ty0 = hy0 - args.padding_mm
  tz0 = hz0 - args.padding_mm
  tlx = hlx + 2.0 * args.padding_mm
  tly = hly + 2.0 * args.padding_mm
  tlz = hlz + 2.0 * args.padding_mm

  gmsh.initialize()
  try:
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("wholebody_conforming")

    occ = gmsh.model.occ
    torso = occ.addBox(tx0, ty0, tz0, tlx, tly, tlz)
    heart = occ.addBox(hx0, hy0, hz0, hlx, hly, hlz)
    occ.fragment([(3, torso)], [(3, heart)])
    occ.synchronize()

    volumes = [tag for dim, tag in gmsh.model.getEntities(3)]
    if len(volumes) != 2:
      raise RuntimeError(f"expected 2 volumes after fragment, got {len(volumes)}")

    heart_vol = None
    torso_vol = None
    for tag in volumes:
      cx, cy, cz = occ.getCenterOfMass(3, tag)
      if is_inside_heart(cx, cy, cz, hx0, hy0, hz0, hlx, hly, hlz):
        heart_vol = tag
      else:
        torso_vol = tag

    if heart_vol is None or torso_vol is None:
      # Fallback for degenerate COM classification: smaller volume is heart.
      volumes_by_size = sorted(volumes, key=lambda tag: occ.getMass(3, tag))
      heart_vol, torso_vol = volumes_by_size[0], volumes_by_size[1]

    gmsh.model.addPhysicalGroup(3, [heart_vol], args.heart_attr)
    gmsh.model.setPhysicalName(3, args.heart_attr, "heart")
    gmsh.model.addPhysicalGroup(3, [torso_vol], args.torso_attr)
    gmsh.model.setPhysicalName(3, args.torso_attr, "torso")

    gmsh.option.setNumber("Mesh.MshFileVersion", args.msh_version)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", args.h_mm)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", args.h_mm)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)

    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), args.h_mm)
    gmsh.model.mesh.generate(3)
    if args.optimize:
      gmsh.model.mesh.optimize("Netgen")

    gmsh.write(str(out_path))

    heart_ne = count_tets_for_volume(heart_vol)
    torso_ne = count_tets_for_volume(torso_vol)
    print(f"[gmsh] wrote: {out_path}")
    print(f"[gmsh] h_mm={args.h_mm}, padding_mm={args.padding_mm}")
    print(f"[gmsh] attrs: heart={args.heart_attr}, torso={args.torso_attr}")
    print(f"[gmsh] elements: heart={heart_ne}, torso={torso_ne}, total={heart_ne + torso_ne}")
  finally:
    gmsh.finalize()


if __name__ == "__main__":
  try:
    main()
  except Exception as ex:
    print(f"generate_conforming_wholebody_gmsh failed: {ex}", file=sys.stderr)
    sys.exit(1)
