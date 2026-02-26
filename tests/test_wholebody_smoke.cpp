#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include "mfem.hpp"

#include "config/SimulationConfig.hpp"
#include "coupling/InterfaceMapper.hpp"
#include "ode/TT06Model.hpp"
#include "solver/ExtracellularRecoverySolver.hpp"
#include "solver/LinearSolverFactory.hpp"
#include "solver/MonodomainStepper.hpp"
#include "solver/TorsoPotentialSolver.hpp"
#include "space/Assembler.hpp"

namespace {

struct Box3 {
  double xmin = 0.0;
  double xmax = 0.0;
  double ymin = 0.0;
  double ymax = 0.0;
  double zmin = 0.0;
  double zmax = 0.0;
};

Box3 ComputeBoundingBox(const mfem::Mesh& mesh) {
  Box3 b;
  b.xmin = b.ymin = b.zmin = std::numeric_limits<double>::infinity();
  b.xmax = b.ymax = b.zmax = -std::numeric_limits<double>::infinity();
  for (int i = 0; i < mesh.GetNV(); ++i) {
    const double* v = mesh.GetVertex(i);
    b.xmin = std::min(b.xmin, v[0]);
    b.xmax = std::max(b.xmax, v[0]);
    b.ymin = std::min(b.ymin, v[1]);
    b.ymax = std::max(b.ymax, v[1]);
    b.zmin = std::min(b.zmin, v[2]);
    b.zmax = std::max(b.zmax, v[2]);
  }
  return b;
}

std::string GenerateConformingWholebodyMesh(const std::string& heart_mesh_path,
                                            const std::filesystem::path& work_dir) {
  mfem::Mesh heart_mesh(heart_mesh_path.c_str(), 1, 1);
  const Box3 hb = ComputeBoundingBox(heart_mesh);
  const double pad = 8.0;
  const double xmin = hb.xmin - pad;
  const double xmax = hb.xmax + pad;
  const double ymin = hb.ymin - pad;
  const double ymax = hb.ymax + pad;
  const double zmin = hb.zmin - pad;
  const double zmax = hb.zmax + pad;
  const double lx = xmax - xmin;
  const double ly = ymax - ymin;
  const double lz = zmax - zmin;

  mfem::Mesh wholebody_mesh =
      mfem::Mesh::MakeCartesian3D(20, 15, 11, mfem::Element::TETRAHEDRON, lx, ly, lz, false);
  for (int i = 0; i < wholebody_mesh.GetNV(); ++i) {
    double* v = wholebody_mesh.GetVertex(i);
    v[0] += xmin;
    v[1] += ymin;
    v[2] += zmin;
  }

  int heart_elem_count = 0;
  for (int e = 0; e < wholebody_mesh.GetNE(); ++e) {
    mfem::Array<int> vtx;
    wholebody_mesh.GetElementVertices(e, vtx);
    double cx = 0.0;
    double cy = 0.0;
    double cz = 0.0;
    for (int j = 0; j < vtx.Size(); ++j) {
      const double* v = wholebody_mesh.GetVertex(vtx[j]);
      cx += v[0];
      cy += v[1];
      cz += v[2];
    }
    cx /= static_cast<double>(vtx.Size());
    cy /= static_cast<double>(vtx.Size());
    cz /= static_cast<double>(vtx.Size());
    const bool in_heart = cx >= hb.xmin && cx <= hb.xmax && cy >= hb.ymin && cy <= hb.ymax &&
                          cz >= hb.zmin && cz <= hb.zmax;
    wholebody_mesh.SetAttribute(e, in_heart ? 1 : 2);
    if (in_heart) {
      ++heart_elem_count;
    }
  }
  wholebody_mesh.SetAttributes();
  if (heart_elem_count == 0) {
    throw std::runtime_error("test_wholebody_smoke generated no heart elements");
  }

  std::filesystem::create_directories(work_dir);
  const std::filesystem::path wholebody_path = work_dir / "wholebody_conforming.mesh";
  std::ofstream out(wholebody_path);
  wholebody_mesh.Print(out);
  return wholebody_path.string();
}

}  // namespace

int main(int argc, char* argv[]) {
  mfem::Mpi::Init(argc, argv);
  const int rank = mfem::Mpi::WorldRank();

  try {
    const std::filesystem::path repo_root = std::filesystem::path(__FILE__).parent_path().parent_path();
    const std::string heart_mesh = (repo_root / "benchmarks/niederer/niederer_benchmark.mesh").string();

    std::filesystem::path work = std::filesystem::temp_directory_path() / "mono_wholebody_smoke";
    std::string wholebody_mesh_path;
    if (rank == 0) {
      std::filesystem::remove_all(work);
      std::filesystem::create_directories(work);
      wholebody_mesh_path = GenerateConformingWholebodyMesh(heart_mesh, work);
    }
    char mesh_path_buf[1024] = {0};
    if (rank == 0) {
      const auto n = std::min<size_t>(wholebody_mesh_path.size(), sizeof(mesh_path_buf) - 1);
      std::copy(wholebody_mesh_path.begin(), wholebody_mesh_path.begin() + n, mesh_path_buf);
    }
    MPI_Bcast(mesh_path_buf, static_cast<int>(sizeof(mesh_path_buf)), MPI_CHAR, 0, MPI_COMM_WORLD);
    wholebody_mesh_path = std::string(mesh_path_buf);
    MPI_Barrier(MPI_COMM_WORLD);

    mono::SimulationConfig cfg;
    cfg.enable_wholebody = true;
    cfg.use_conforming_wholebody = true;
    cfg.wholebody_mesh_path = wholebody_mesh_path;
    cfg.heart_volume_attrs = {1};
    cfg.torso_volume_attrs = {2};
    cfg.use_fiber_gf = false;
    cfg.dt_pde_ms = 0.02;
    cfg.dt_ode_ms = 0.01;
    cfg.sigma_i_f_mS_per_mm = 0.174;
    cfg.sigma_i_s_mS_per_mm = 0.019;
    cfg.sigma_i_n_mS_per_mm = 0.019;
    cfg.sigma_e_f_mS_per_mm = 0.625;
    cfg.sigma_e_s_mS_per_mm = 0.236;
    cfg.sigma_e_n_mS_per_mm = 0.236;
    cfg.sigma_torso_mS_per_mm = 0.2;
    cfg.interface_map_max_dist_mm = 0.2;
    cfg.torso_dirichlet_penalty = 1e6;
    cfg.ksp_max_it = 300;
    cfg.ksp_rtol = 1e-8;

    auto wholebody_serial = std::make_unique<mfem::Mesh>(cfg.wholebody_mesh_path.c_str(), 1, 1);
    auto wholebody_parent = std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, *wholebody_serial);
    mfem::Array<int> heart_attrs(1);
    heart_attrs[0] = 1;
    mfem::Array<int> torso_attrs(1);
    torso_attrs[0] = 2;
    auto heart_sub = mfem::ParSubMesh::CreateFromDomain(*wholebody_parent, heart_attrs);
    auto torso_sub = mfem::ParSubMesh::CreateFromDomain(*wholebody_parent, torso_attrs);

    auto heart_pmesh = std::make_unique<mfem::ParSubMesh>(std::move(heart_sub));
    auto torso_pmesh = std::make_unique<mfem::ParSubMesh>(std::move(torso_sub));

    mono::Assembler assembler(cfg, MPI_COMM_WORLD, std::move(heart_pmesh));
    mono::TT06Model tt06(assembler.TrueVSize());
    tt06.InitializeRestState(-85.23);
    mono::LinearSystemSolver linear_solver(cfg, MPI_COMM_WORLD);
    mono::MonodomainStepper stepper(cfg, assembler, tt06, linear_solver);
    stepper.InitializeVm(-85.23);
    stepper.Bootstrap();

    mono::ExtracellularRecoverySolver ue_solver(cfg, assembler, MPI_COMM_WORLD);
    ue_solver.Solve(stepper.VmTrue());
    mono::TorsoPotentialSolver torso_solver(cfg, MPI_COMM_WORLD, std::move(torso_pmesh));
    mono::InterfaceMapper mapper(cfg, assembler, torso_solver, MPI_COMM_WORLD);
    torso_solver.SetConstrainedDofs(mapper.TorsoConstrainedDofs());

    mfem::Vector torso_bc;
    mapper.MapHeartToTorso(ue_solver.UeTrue(), torso_bc);
    torso_solver.Solve(torso_bc);

    if (mapper.GlobalNumConstrainedDofs() <= 0) {
      if (rank == 0) {
        std::cerr << "No torso constrained dofs were mapped from heart" << std::endl;
      }
      return 2;
    }
    if (mapper.GlobalMaxConstrainedDistMm() > 1e-6) {
      if (rank == 0) {
        std::cerr << "Conforming mapper expected near-zero mapping distance, got "
                  << mapper.GlobalMaxConstrainedDistMm() << std::endl;
      }
      return 4;
    }

    auto all_finite = [](const mfem::Vector& v) {
      for (int i = 0; i < v.Size(); ++i) {
        if (!std::isfinite(v[i])) {
          return false;
        }
      }
      return true;
    };
    int local_ok = all_finite(ue_solver.UeTrue()) && all_finite(torso_solver.UTTrue()) ? 1 : 0;
    int global_ok = 0;
    MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (global_ok == 0) {
      if (rank == 0) {
        std::cerr << "wholebody smoke produced non-finite ue/uT values" << std::endl;
      }
      return 3;
    }
  } catch (const std::exception& ex) {
    if (rank == 0) {
      std::cerr << "wholebody smoke failed: " << ex.what() << std::endl;
    }
    return 1;
  }

  return 0;
}
