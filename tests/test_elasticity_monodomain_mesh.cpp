#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>

#include "mfem.hpp"

#include "mechanics/ElasticitySolver.hpp"

int main(int argc, char* argv[]) {
  mfem::Mpi::Init(argc, argv);
  mfem::Hypre::Init();

  if (mfem::Mpi::WorldSize() != 1) {
    return 0;
  }

  const int rank = mfem::Mpi::WorldRank();
  try {
    const std::filesystem::path repo_root = std::filesystem::path(__FILE__).parent_path().parent_path();
    const std::filesystem::path mesh_path = repo_root / "benchmarks/niederer/niederer_benchmark.mesh";

    if (!std::filesystem::exists(mesh_path)) {
      throw std::runtime_error("monodomain mesh not found: " + mesh_path.string());
    }

    mono::ElasticityOptions opts;
    opts.order = 1;
    opts.young_modulus = 12.0;
    opts.poisson_ratio = 0.30;
    opts.traction_x = 1.0;
    opts.traction_y = 0.0;
    opts.traction_z = 0.0;
    opts.boundary_tolerance = 1e-6;
    opts.max_iter = 400;
    opts.rel_tol = 1e-10;
    opts.abs_tol = 1e-14;
    opts.print_level = 0;

    mono::ElasticitySolver solver(MPI_COMM_WORLD);
    const mono::ElasticityResult result = solver.Solve(mesh_path.string(), opts);

    if (!std::isfinite(result.displacement_l2_norm) || !std::isfinite(result.displacement_max_abs)) {
      throw std::runtime_error("elasticity displacement norm is not finite");
    }
    if (result.displacement_l2_norm <= 1e-12 || result.displacement_max_abs <= 1e-12) {
      throw std::runtime_error("elasticity displacement is unexpectedly zero");
    }
    if (result.num_cg_iterations <= 0) {
      throw std::runtime_error("elasticity CG did not perform iterations");
    }
  } catch (const std::exception& ex) {
    if (rank == 0) {
      std::cerr << ex.what() << std::endl;
    }
    return 1;
  }

  return 0;
}
