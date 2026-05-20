// Generate a half-ellipsoid hollow shell mesh approximating a left ventricle.
//
// Geometry (default; overridable via flags):
//   Outer ellipsoid semi-axes: a=35, b=22, c=22 mm (long axis along +x; basal
//                              plane at x=0; apex at x=+35).
//   Inner cavity semi-axes:    a_in=28, b_in=15, c_in=15 mm (5-7 mm wall).
//   Half:                      x >= 0 only. Basal plane x=0 is the open mouth.
//
// Approach:
//   1. Build a Cartesian hex mesh covering the outer bounding box at user-
//      supplied step `h_mm`.
//   2. Mark each element by centroid: keep if (centroid is inside outer
//      ellipsoid) AND (outside inner ellipsoid) AND (x >= 0).
//   3. Boundary-face attributes are heuristically classified:
//        attr 1 = epicardium  (centroid at element-face is closer to outer surface)
//        attr 2 = endocardium (closer to inner surface)
//        attr 3 = base        (face normal aligned with +x and centroid at x≈0)
//   4. Element volume attribute is 1 (single ventricular tissue class for now).
//   5. Optional fiber field: aligned with the local "long-axis tangent" -- a
//      simple deterministic rule of (x, y, z) -> unit tangent of the ellipsoid
//      meridian. Saved as three scalar grid functions if --emit-fibers is set.
//
// Outputs:
//   <out_dir>/heart.mesh         (MFEM v1.0 ASCII)
//   <out_dir>/fiber_f.gf         (only if --emit-fibers)
//   <out_dir>/fiber_s.gf
//   <out_dir>/fiber_n.gf

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mfem.hpp"

namespace {

struct Args {
  double a_out = 35.0;
  double b_out = 22.0;
  double c_out = 22.0;
  double a_in  = 28.0;
  double b_in  = 15.0;
  double c_in  = 15.0;
  double h_mm  = 1.5;
  std::string out_dir = "benchmarks/half_ellipsoid";
  bool emit_fibers = false;
};

Args ParseArgs(int argc, char* argv[]) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string s = argv[i];
    auto need = [&](const char* name) {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string("Missing value for ") + name);
      }
      return std::string(argv[++i]);
    };
    if      (s == "--a-out")        a.a_out = std::stod(need("--a-out"));
    else if (s == "--b-out")        a.b_out = std::stod(need("--b-out"));
    else if (s == "--c-out")        a.c_out = std::stod(need("--c-out"));
    else if (s == "--a-in")         a.a_in  = std::stod(need("--a-in"));
    else if (s == "--b-in")         a.b_in  = std::stod(need("--b-in"));
    else if (s == "--c-in")         a.c_in  = std::stod(need("--c-in"));
    else if (s == "--h-mm")         a.h_mm  = std::stod(need("--h-mm"));
    else if (s == "--out-dir")      a.out_dir = need("--out-dir");
    else if (s == "--emit-fibers")  a.emit_fibers = true;
    else if (s == "--help" || s == "-h") {
      std::cout
          << "usage: generate_half_ellipsoid_case [options]\n"
          << "  --a-out / --b-out / --c-out   outer semi-axes mm (default 35,22,22)\n"
          << "  --a-in / --b-in / --c-in      inner semi-axes mm (default 28,15,15)\n"
          << "  --h-mm                        cartesian voxel size mm (default 1.5)\n"
          << "  --out-dir                     output dir (default benchmarks/half_ellipsoid)\n"
          << "  --emit-fibers                 also write fiber_f/s/n.gf\n";
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown flag: " + s);
    }
  }
  return a;
}

bool InsideOuter(double x, double y, double z, const Args& a) {
  return (x*x)/(a.a_out*a.a_out) + (y*y)/(a.b_out*a.b_out) + (z*z)/(a.c_out*a.c_out) <= 1.0;
}

bool InsideInner(double x, double y, double z, const Args& a) {
  return (x*x)/(a.a_in*a.a_in) + (y*y)/(a.b_in*a.b_in) + (z*z)/(a.c_in*a.c_in) <= 1.0;
}

bool InShell(double x, double y, double z, const Args& a) {
  return x >= 0.0 && InsideOuter(x, y, z, a) && !InsideInner(x, y, z, a);
}

}  // namespace

int main(int argc, char* argv[]) {
  try {
    const Args a = ParseArgs(argc, argv);

    // Cartesian box covering [0, a_out] x [-b_out, b_out] x [-c_out, c_out],
    // discretized so that each cell has side ~ h_mm.
    const int nx = std::max(2, static_cast<int>(std::round(a.a_out / a.h_mm)));
    const int ny = std::max(2, static_cast<int>(std::round(2.0 * a.b_out / a.h_mm)));
    const int nz = std::max(2, static_cast<int>(std::round(2.0 * a.c_out / a.h_mm)));

    mfem::Mesh full = mfem::Mesh::MakeCartesian3D(
        nx, ny, nz, mfem::Element::TETRAHEDRON,
        a.a_out, 2.0 * a.b_out, 2.0 * a.c_out);
    // Shift so y, z are centered around 0; x already starts at 0.
    {
      mfem::Vector vshift(3);
      vshift[0] = 0.0;
      vshift[1] = -a.b_out;
      vshift[2] = -a.c_out;
      for (int v = 0; v < full.GetNV(); ++v) {
        double* p = full.GetVertex(v);
        p[0] += vshift[0];
        p[1] += vshift[1];
        p[2] += vshift[2];
      }
    }

    // Mark elements: keep if centroid is in shell.
    std::vector<int> keep(full.GetNE(), 0);
    int n_keep = 0;
    for (int e = 0; e < full.GetNE(); ++e) {
      mfem::Array<int> verts;
      full.GetElementVertices(e, verts);
      double cx = 0, cy = 0, cz = 0;
      for (int j = 0; j < verts.Size(); ++j) {
        const double* v = full.GetVertex(verts[j]);
        cx += v[0]; cy += v[1]; cz += v[2];
      }
      cx /= verts.Size(); cy /= verts.Size(); cz /= verts.Size();
      if (InShell(cx, cy, cz, a)) {
        keep[e] = 1;
        ++n_keep;
      }
    }

    if (n_keep == 0) {
      throw std::runtime_error(
          "No elements survived the shell carve -- check geometry parameters.");
    }

    // Build a new mesh containing only the kept elements. Vertex remap.
    std::vector<int> vert_map(full.GetNV(), -1);
    int n_new_verts = 0;
    for (int e = 0; e < full.GetNE(); ++e) {
      if (!keep[e]) continue;
      mfem::Array<int> verts;
      full.GetElementVertices(e, verts);
      for (int j = 0; j < verts.Size(); ++j) {
        if (vert_map[verts[j]] < 0) {
          vert_map[verts[j]] = n_new_verts++;
        }
      }
    }

    // Need to create the new mesh. Use the constructor that takes raw
    // vertex/element arrays. We'll output as MFEM ASCII directly to keep
    // the boundary face classification simple.
    std::filesystem::create_directories(a.out_dir);

    // Pre-compute boundary faces of the new mesh: each face is "boundary"
    // iff exactly one of its incident elements is in `keep`. We classify
    // each boundary face by centroid distance to outer / inner / basal.
    // For a tetrahedron with 4 faces, MFEM gives faces via GetElementFaces.
    std::vector<std::array<int,3>> bdr_faces;
    std::vector<int> bdr_attrs;

    mfem::Array<int> face_marker(full.GetNumFaces());
    face_marker = 0;
    // For each element-face, count how many kept elements share it.
    // Method: walk all elements, for each face increment the face_marker for
    // each kept-element side. Then face is boundary if marker == 1 (only one
    // kept element on it).
    std::vector<std::vector<int>> face_to_kept_elem(full.GetNumFaces());
    for (int e = 0; e < full.GetNE(); ++e) {
      if (!keep[e]) continue;
      mfem::Array<int> faces, ori;
      full.GetElementFaces(e, faces, ori);
      for (int k = 0; k < faces.Size(); ++k) {
        face_to_kept_elem[faces[k]].push_back(e);
        ++face_marker[faces[k]];
      }
    }

    // Classify each boundary face.
    for (int f = 0; f < full.GetNumFaces(); ++f) {
      if (face_marker[f] != 1) continue;
      mfem::Array<int> fv;
      full.GetFaceVertices(f, fv);
      std::array<int,3> tri = {-1, -1, -1};
      if (fv.Size() != 3) {
        // Skip non-tri faces (cartesian source produced quads at boundary
        // for hex; but we used TETRAHEDRON so all faces are tris).
        continue;
      }
      double cx = 0, cy = 0, cz = 0;
      for (int j = 0; j < 3; ++j) {
        const double* v = full.GetVertex(fv[j]);
        cx += v[0]; cy += v[1]; cz += v[2];
        tri[j] = vert_map[fv[j]];
      }
      cx /= 3.0; cy /= 3.0; cz /= 3.0;

      // Attribute classification heuristic:
      //   1 (epicardium): close to outer surface (outer_dist < inner_dist + 0.5*h)
      //   2 (endocardium): close to inner surface
      //   3 (base):       cx very close to 0
      const double outer_score = std::abs(
          (cx*cx)/(a.a_out*a.a_out) + (cy*cy)/(a.b_out*a.b_out) +
          (cz*cz)/(a.c_out*a.c_out) - 1.0);
      const double inner_score = std::abs(
          (cx*cx)/(a.a_in*a.a_in) + (cy*cy)/(a.b_in*a.b_in) +
          (cz*cz)/(a.c_in*a.c_in) - 1.0);
      int attr = (outer_score < inner_score) ? 1 : 2;
      if (cx < 0.5 * a.h_mm) attr = 3;  // base override
      bdr_faces.push_back(tri);
      bdr_attrs.push_back(attr);
    }

    // Write MFEM ASCII v1.0 format.
    const std::string mesh_path = a.out_dir + "/heart.mesh";
    std::ofstream out(mesh_path);
    out << "MFEM mesh v1.0\n\n";
    out << "dimension\n3\n\n";
    out << "elements\n" << n_keep << "\n";
    for (int e = 0; e < full.GetNE(); ++e) {
      if (!keep[e]) continue;
      mfem::Array<int> verts;
      full.GetElementVertices(e, verts);
      // <attribute> <element_geom_type=4 for tet> v0 v1 v2 v3
      out << "1 4";
      for (int j = 0; j < verts.Size(); ++j) {
        out << " " << vert_map[verts[j]];
      }
      out << "\n";
    }
    out << "\nboundary\n" << bdr_faces.size() << "\n";
    for (size_t i = 0; i < bdr_faces.size(); ++i) {
      // <attribute> <face_geom=2 for triangle> v0 v1 v2
      out << bdr_attrs[i] << " 2 "
          << bdr_faces[i][0] << " " << bdr_faces[i][1] << " " << bdr_faces[i][2]
          << "\n";
    }
    out << "\nvertices\n" << n_new_verts << "\n3\n";
    std::vector<std::array<double,3>> new_verts(n_new_verts);
    for (int v = 0; v < full.GetNV(); ++v) {
      const int nv = vert_map[v];
      if (nv < 0) continue;
      const double* p = full.GetVertex(v);
      new_verts[nv] = {p[0], p[1], p[2]};
    }
    for (const auto& v : new_verts) {
      out << v[0] << " " << v[1] << " " << v[2] << "\n";
    }

    out.close();

    std::cout << "Wrote " << mesh_path << "\n";
    std::cout << "  elements:    " << n_keep << " tets\n";
    std::cout << "  vertices:    " << n_new_verts << "\n";
    std::cout << "  boundary:    " << bdr_faces.size() << " tris (epi/endo/base)\n";
    {
      int n_epi = 0, n_endo = 0, n_base = 0;
      for (int a_attr : bdr_attrs) {
        if (a_attr == 1) ++n_epi;
        else if (a_attr == 2) ++n_endo;
        else if (a_attr == 3) ++n_base;
      }
      std::cout << "    epi=" << n_epi << " endo=" << n_endo
                << " base=" << n_base << "\n";
    }

    if (a.emit_fibers) {
      // Emit constant fibers aligned with x (long axis) for now. Real fiber
      // generation (rule-based, e.g. Bayer 2012) is a TODO.
      mfem::Mesh shell(mesh_path.c_str(), 1, 1);
      mfem::H1_FECollection fec(1, 3);
      mfem::FiniteElementSpace fes(&shell, &fec, 3, mfem::Ordering::byVDIM);
      mfem::GridFunction f_gf(&fes), s_gf(&fes), n_gf(&fes);

      auto set_const = [&](mfem::GridFunction& g, double x, double y, double z) {
        const int n = g.Size() / 3;
        for (int i = 0; i < n; ++i) {
          g(3 * i + 0) = x;
          g(3 * i + 1) = y;
          g(3 * i + 2) = z;
        }
      };
      set_const(f_gf, 1.0, 0.0, 0.0);
      set_const(s_gf, 0.0, 1.0, 0.0);
      set_const(n_gf, 0.0, 0.0, 1.0);

      const std::string ff = a.out_dir + "/fiber_f.gf";
      const std::string fs = a.out_dir + "/fiber_s.gf";
      const std::string fn = a.out_dir + "/fiber_n.gf";
      std::ofstream(ff) << f_gf;
      std::ofstream(fs) << s_gf;
      std::ofstream(fn) << n_gf;
      std::cout << "Wrote " << ff << ", " << fs << ", " << fn << "\n";
    }

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << std::endl;
    return 1;
  }
}
