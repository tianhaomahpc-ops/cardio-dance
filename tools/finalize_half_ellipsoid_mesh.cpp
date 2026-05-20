// Post-processor for the gmsh-generated half-ellipsoid mesh.
//
// Loads benchmarks/half_ellipsoid/heart.msh, reclassifies every boundary
// triangle by its true face centroid (gmsh's surface-bounding-box trick is
// unreliable for the curved cavity surface after BooleanIntersection), and
// writes a clean MFEM v1.0 mesh + constant fiber grid functions.
//
// Boundary attributes:
//   1 = epicardium (centroid closer to outer ellipsoid surface)
//   2 = endocardium (centroid closer to inner ellipsoid surface)
//   3 = base       (centroid x < 0.5 * h_mm)

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "mfem.hpp"

namespace {

struct Args {
  std::string msh = "benchmarks/half_ellipsoid/heart.msh";
  std::string out_dir = "benchmarks/half_ellipsoid";
  double a_out = 35.0, b_out = 22.0, c_out = 22.0;
  double a_in  = 28.0, b_in  = 15.0, c_in  = 15.0;
  double base_eps_mm = 0.5;
  bool emit_fibers = false;
};

Args ParseArgs(int argc, char* argv[]) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    auto need = [&]() {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + s);
      return std::string(argv[++i]);
    };
    if      (s == "--msh")          a.msh = need();
    else if (s == "--out-dir")      a.out_dir = need();
    else if (s == "--a-out")        a.a_out = std::stod(need());
    else if (s == "--b-out")        a.b_out = std::stod(need());
    else if (s == "--c-out")        a.c_out = std::stod(need());
    else if (s == "--a-in")         a.a_in  = std::stod(need());
    else if (s == "--b-in")         a.b_in  = std::stod(need());
    else if (s == "--c-in")         a.c_in  = std::stod(need());
    else if (s == "--base-eps-mm")  a.base_eps_mm = std::stod(need());
    else if (s == "--emit-fibers")  a.emit_fibers = true;
    else throw std::runtime_error("unknown flag " + s);
  }
  return a;
}

}  // namespace

int main(int argc, char* argv[]) {
  try {
    const Args a = ParseArgs(argc, argv);
    mfem::Mesh m(a.msh.c_str(), 1, 1);
    std::cout << "loaded " << a.msh << ": NV=" << m.GetNV()
              << " NE=" << m.GetNE() << " NB=" << m.GetNBE() << "\n";

    int n_epi = 0, n_endo = 0, n_base = 0;
    for (int b = 0; b < m.GetNBE(); ++b) {
      mfem::Array<int> verts;
      m.GetBdrElementVertices(b, verts);
      double cx = 0, cy = 0, cz = 0;
      for (int v : verts) {
        const double* p = m.GetVertex(v);
        cx += p[0]; cy += p[1]; cz += p[2];
      }
      cx /= verts.Size(); cy /= verts.Size(); cz /= verts.Size();

      int attr;
      if (cx < a.base_eps_mm) {
        attr = 3; ++n_base;
      } else {
        const double outer = std::abs(
            (cx*cx)/(a.a_out*a.a_out) +
            (cy*cy)/(a.b_out*a.b_out) +
            (cz*cz)/(a.c_out*a.c_out) - 1.0);
        const double inner = std::abs(
            (cx*cx)/(a.a_in*a.a_in) +
            (cy*cy)/(a.b_in*a.b_in) +
            (cz*cz)/(a.c_in*a.c_in) - 1.0);
        if (outer < inner) { attr = 1; ++n_epi; }
        else               { attr = 2; ++n_endo; }
      }
      m.SetBdrAttribute(b, attr);
    }
    m.SetAttributes();
    std::cout << "  reclassified: epi=" << n_epi
              << " endo=" << n_endo << " base=" << n_base << "\n";

    std::filesystem::create_directories(a.out_dir);
    const std::string out_path = a.out_dir + "/heart.mesh";
    std::ofstream out(out_path);
    m.Print(out);
    out.close();
    std::cout << "wrote " << out_path << "\n";

    if (a.emit_fibers) {
      mfem::H1_FECollection fec(1, 3);
      mfem::FiniteElementSpace fes(&m, &fec, 3, mfem::Ordering::byVDIM);
      mfem::GridFunction f(&fes), s(&fes), n(&fes);
      const int npt = f.Size() / 3;
      for (int i = 0; i < npt; ++i) {
        f(3*i+0) = 1.0; f(3*i+1) = 0.0; f(3*i+2) = 0.0;
        s(3*i+0) = 0.0; s(3*i+1) = 1.0; s(3*i+2) = 0.0;
        n(3*i+0) = 0.0; n(3*i+1) = 0.0; n(3*i+2) = 1.0;
      }
      std::ofstream(a.out_dir + "/fiber_f.gf") << f;
      std::ofstream(a.out_dir + "/fiber_s.gf") << s;
      std::ofstream(a.out_dir + "/fiber_n.gf") << n;
      std::cout << "wrote constant fiber_f/s/n.gf in " << a.out_dir << "\n";
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << std::endl;
    return 1;
  }
}
