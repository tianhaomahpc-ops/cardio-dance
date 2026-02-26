#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include "mfem.hpp"

namespace {

// Save one vector field in MFEM GridFunction format.
void WriteVectorField(const std::string& path, mfem::GridFunction& gf) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Cannot open output file: " + path);
  }
  gf.Save(out);
}

bool IsBoundaryVertex(const double x,
                      const double y,
                      const double z,
                      const double lx,
                      const double ly,
                      const double lz) {
  const double tol = 1e-12;
  return x <= tol || y <= tol || z <= tol || x >= (lx - tol) || y >= (ly - tol) ||
         z >= (lz - tol);
}

void MakeUnstructuredTetByVertexJitter(mfem::Mesh& mesh,
                                       const double lx,
                                       const double ly,
                                       const double lz,
                                       const double hx,
                                       const double hy,
                                       const double hz,
                                       const double jitter_scale) {
  const double ax = jitter_scale * hx;
  const double ay = jitter_scale * hy;
  const double az = jitter_scale * hz;

  for (int i = 0; i < mesh.GetNV(); ++i) {
    double* v = mesh.GetVertex(i);
    if (IsBoundaryVertex(v[0], v[1], v[2], lx, ly, lz)) {
      continue;
    }

    const double s = static_cast<double>(i + 1);
    v[0] += ax * std::sin(0.73 * s + 0.11);
    v[1] += ay * std::sin(1.13 * s + 0.37);
    v[2] += az * std::sin(0.97 * s + 0.53);
  }

  // Ensure all tetrahedra keep positive orientation after perturbation.
  mfem::DenseMatrix jac;
  for (int e = 0; e < mesh.GetNE(); ++e) {
    mesh.GetElementJacobian(e, jac);
    if (jac.Det() <= std::numeric_limits<double>::epsilon()) {
      throw std::runtime_error(
          "Generated tetra mesh has non-positive Jacobian after jitter.");
    }
  }
}

}  // namespace

int main(int argc, char* argv[]) {
  mfem::Mpi::Init(argc, argv);

  if (mfem::Mpi::WorldSize() != 1) {
    if (mfem::Mpi::WorldRank() == 0) {
      std::cerr << "generate_niederer_case must run with -np 1" << std::endl;
    }
    return 1;
  }

  try {
    std::filesystem::path out_dir = "benchmarks/niederer";
    bool out_dir_set = false;

    // Slab-like geometry for Niederer-style benchmark: 20 x 7 x 3 mm.
    int nx = 40;
    int ny = 14;
    int nz = 6;
    double lx_mm = 20.0;
    double ly_mm = 7.0;
    double lz_mm = 3.0;
    double jitter_scale = 0.12;

    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      auto need_value = [&](const char* key) -> std::string {
        if (i + 1 >= argc) {
          throw std::runtime_error(std::string("Missing value for ") + key);
        }
        return std::string(argv[++i]);
      };

      if (arg == "--nx") {
        nx = std::stoi(need_value("--nx"));
      } else if (arg == "--ny") {
        ny = std::stoi(need_value("--ny"));
      } else if (arg == "--nz") {
        nz = std::stoi(need_value("--nz"));
      } else if (arg == "--lx-mm") {
        lx_mm = std::stod(need_value("--lx-mm"));
      } else if (arg == "--ly-mm") {
        ly_mm = std::stod(need_value("--ly-mm"));
      } else if (arg == "--lz-mm") {
        lz_mm = std::stod(need_value("--lz-mm"));
      } else if (arg == "--jitter-scale") {
        jitter_scale = std::stod(need_value("--jitter-scale"));
      } else if (arg.rfind("--", 0) == 0) {
        throw std::runtime_error("Unknown option: " + arg);
      } else if (!out_dir_set) {
        out_dir = std::filesystem::path(arg);
        out_dir_set = true;
      } else {
        throw std::runtime_error("Unexpected positional argument: " + arg);
      }
    }

    if (nx <= 0 || ny <= 0 || nz <= 0) {
      throw std::runtime_error("nx, ny, nz must be > 0");
    }
    if (lx_mm <= 0.0 || ly_mm <= 0.0 || lz_mm <= 0.0) {
      throw std::runtime_error("lx-mm, ly-mm, lz-mm must be > 0");
    }
    if (jitter_scale < 0.0) {
      throw std::runtime_error("jitter-scale must be >= 0");
    }

    std::filesystem::create_directories(out_dir);

    // Start from Cartesian tets, then jitter interior vertices to remove structure.
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D(
        nx, ny, nz, mfem::Element::TETRAHEDRON, lx_mm, ly_mm, lz_mm, false);
    if (jitter_scale > 0.0) {
      MakeUnstructuredTetByVertexJitter(mesh,
                                        lx_mm,
                                        ly_mm,
                                        lz_mm,
                                        lx_mm / static_cast<double>(nx),
                                        ly_mm / static_cast<double>(ny),
                                        lz_mm / static_cast<double>(nz),
                                        jitter_scale);
    }

    {
      std::ofstream mesh_out(out_dir / "niederer_benchmark.mesh");
      if (!mesh_out) {
        throw std::runtime_error("Cannot write mesh file");
      }
      mesh.Print(mesh_out);
    }

    const int dim = mesh.Dimension();
    mfem::H1_FECollection fec(1, dim);
    mfem::FiniteElementSpace vfes(&mesh, &fec, dim, mfem::Ordering::byVDIM);

    // Constant orthonormal fiber-sheet-normal fields.
    mfem::GridFunction fiber_f(&vfes);
    mfem::GridFunction fiber_s(&vfes);
    mfem::GridFunction fiber_n(&vfes);

    mfem::Vector f(dim), s(dim), n(dim);
    f = 0.0;
    s = 0.0;
    n = 0.0;
    f[0] = 1.0;
    s[1] = 1.0;
    n[2] = 1.0;

    mfem::VectorConstantCoefficient f_coeff(f);
    mfem::VectorConstantCoefficient s_coeff(s);
    mfem::VectorConstantCoefficient n_coeff(n);

    fiber_f.ProjectCoefficient(f_coeff);
    fiber_s.ProjectCoefficient(s_coeff);
    fiber_n.ProjectCoefficient(n_coeff);

    WriteVectorField((out_dir / "fiber_f.gf").string(), fiber_f);
    WriteVectorField((out_dir / "fiber_s.gf").string(), fiber_s);
    WriteVectorField((out_dir / "fiber_n.gf").string(), fiber_n);

    std::cout << "Generated Niederer benchmark files in: " << out_dir << std::endl;
  } catch (const std::exception& ex) {
    std::cerr << "generate_niederer_case failed: " << ex.what() << std::endl;
    return 2;
  }

  return 0;
}
