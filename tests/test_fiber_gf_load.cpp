#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "mfem.hpp"

#include "config/SimulationConfig.hpp"
#include "space/Assembler.hpp"

int main(int argc, char* argv[]) {
  mfem::Mpi::Init(argc, argv);

  if (mfem::Mpi::WorldSize() != 1) {
    return 0;
  }

  constexpr int dim = 3;
  std::filesystem::path work = std::filesystem::temp_directory_path() / "mono_fiber_gf_load";
  std::filesystem::create_directories(work);

  const std::string mesh_file = (work / "fiber_test.mesh").string();
  const std::string f_file = (work / "fiber_f.gf").string();
  const std::string s_file = (work / "fiber_s.gf").string();
  const std::string n_file = (work / "fiber_n.gf").string();

  {
    // Build a small unstructured tetra mesh and constant vector fibers.
    mfem::Mesh mesh = mfem::Mesh::MakeCartesian3D(
        4, 3, 2, mfem::Element::TETRAHEDRON, 12.0, 9.0, 6.0, false);
    for (int i = 0; i < mesh.GetNV(); ++i) {
      double* v = mesh.GetVertex(i);
      const bool boundary = (v[0] <= 1e-12 || v[1] <= 1e-12 || v[2] <= 1e-12 ||
                             v[0] >= 12.0 - 1e-12 || v[1] >= 9.0 - 1e-12 ||
                             v[2] >= 6.0 - 1e-12);
      if (boundary) {
        continue;
      }
      const double s = static_cast<double>(i + 1);
      v[0] += 0.15 * std::sin(0.61 * s);
      v[1] += 0.15 * std::sin(1.03 * s + 0.2);
      v[2] += 0.15 * std::sin(0.87 * s + 0.4);
    }
    mfem::H1_FECollection fec(1, dim);
    mfem::FiniteElementSpace vfes(&mesh, &fec, dim, mfem::Ordering::byVDIM);

    mfem::GridFunction gf_f(&vfes), gf_s(&vfes), gf_n(&vfes);

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

    gf_f.ProjectCoefficient(f_coeff);
    gf_s.ProjectCoefficient(s_coeff);
    gf_n.ProjectCoefficient(n_coeff);

    {
      std::ofstream out(mesh_file);
      mesh.Print(out);
    }
    {
      std::ofstream out(f_file);
      gf_f.Save(out);
    }
    {
      std::ofstream out(s_file);
      gf_s.Save(out);
    }
    {
      std::ofstream out(n_file);
      gf_n.Save(out);
    }
  }

  mono::SimulationConfig cfg;
  cfg.mesh_path = mesh_file;
  cfg.use_fiber_gf = true;
  cfg.fiber_f_path = f_file;
  cfg.fiber_s_path = s_file;
  cfg.fiber_n_path = n_file;
  cfg.dt_pde_ms = 0.02;

  mono::Assembler assembler(cfg, MPI_COMM_WORLD);
  if (!assembler.HasFiberFields()) {
    std::cerr << "Fiber fields were not loaded from .gf files" << std::endl;
    return 1;
  }

  auto check_fiber = [&](const mfem::ParGridFunction* gf,
                         const mfem::Vector& expected,
                         const char* name) -> int {
    // Validate all vector components in byVDIM ordering.
    const int ndofs = gf->ParFESpace()->GetNDofs();
    for (int d = 0; d < ndofs; ++d) {
      for (int c = 0; c < dim; ++c) {
        const int idx = mfem::Ordering::Map<mfem::Ordering::byVDIM>(ndofs, dim, d, c);
        const double v = (*gf)(idx);
        if (!std::isfinite(v) || std::abs(v - expected[c]) > 1e-12) {
          const int idx_nodes = mfem::Ordering::Map<mfem::Ordering::byNODES>(ndofs, dim, d, c);
          const double v_nodes = (*gf)(idx_nodes);
          const int ord = static_cast<int>(gf->ParFESpace()->GetOrdering());
          std::cerr << "Invalid " << name << " value at dof=" << d << " comp=" << c
                    << " got_vdim=" << v << " got_nodes=" << v_nodes
                    << " expected=" << expected[c] << " ordering=" << ord
                    << std::endl;
          return 1;
        }
      }
    }
    return 0;
  };

  mfem::Vector ef(dim), es(dim), en(dim);
  ef = 0.0;
  es = 0.0;
  en = 0.0;
  ef[0] = 1.0;
  es[1] = 1.0;
  en[2] = 1.0;
  if (check_fiber(assembler.FiberF(), ef, "fiber_f")) return 3;
  if (check_fiber(assembler.FiberS(), es, "fiber_s")) return 4;
  if (check_fiber(assembler.FiberN(), en, "fiber_n")) return 5;

  if (assembler.K().Height() != assembler.K().Width() || assembler.K().Height() <= 0) {
    std::cerr << "Invalid stiffness matrix assembled with loaded fibers" << std::endl;
    return 2;
  }

  return 0;
}
