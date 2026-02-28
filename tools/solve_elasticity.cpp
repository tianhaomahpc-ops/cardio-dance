#include <exception>
#include <iostream>
#include <string>

#include "mfem.hpp"

#include "mechanics/ElasticitySolver.hpp"

int main(int argc, char* argv[]) {
  mfem::Mpi::Init(argc, argv);
  mfem::Hypre::Init();

  const int rank = mfem::Mpi::WorldRank();
  const int world = mfem::Mpi::WorldSize();

  const char* mesh_file = "benchmarks/niederer/niederer_benchmark.mesh";
  const char* output_prefix = "";
  int order = 1;
  int max_iter = 500;
  int print_level = 0;
  double young_modulus = 10.0;
  double poisson_ratio = 0.30;
  double traction_x = 1.0;
  double traction_y = 0.0;
  double traction_z = 0.0;
  double boundary_tolerance = 1e-6;
  double rel_tol = 1e-10;
  double abs_tol = 1e-14;

  mfem::OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh", "Path to MFEM mesh used by the monodomain solver.");
  args.AddOption(&output_prefix, "-out", "--output-prefix",
                 "Output file prefix; writes <prefix>.mesh.XXXXXX and <prefix>.disp.XXXXXX if not empty.");
  args.AddOption(&order, "-o", "--order", "Finite element order for displacement space.");
  args.AddOption(&young_modulus, "-E", "--young-modulus", "Young's modulus.");
  args.AddOption(&poisson_ratio, "-nu", "--poisson-ratio", "Poisson ratio.");
  args.AddOption(&traction_x, "-tx", "--traction-x", "Traction component in x direction.");
  args.AddOption(&traction_y, "-ty", "--traction-y", "Traction component in y direction.");
  args.AddOption(&traction_z, "-tz", "--traction-z", "Traction component in z direction.");
  args.AddOption(&boundary_tolerance, "-btol", "--boundary-tolerance",
                 "Relative tolerance used for x-min/x-max boundary tagging.");
  args.AddOption(&max_iter, "-it", "--max-iter", "Maximum CG iterations.");
  args.AddOption(&rel_tol, "-rtol", "--relative-tol", "Relative tolerance for CG.");
  args.AddOption(&abs_tol, "-atol", "--absolute-tol", "Absolute tolerance for CG.");
  args.AddOption(&print_level, "-pl", "--print-level", "CG solver print level.");

  args.Parse();
  if (!args.Good()) {
    if (rank == 0) {
      args.PrintUsage(std::cout);
    }
    return 1;
  }

  if (rank == 0) {
    args.PrintOptions(std::cout);
  }

  try {
    mono::ElasticityOptions opts;
    opts.order = order;
    opts.young_modulus = young_modulus;
    opts.poisson_ratio = poisson_ratio;
    opts.traction_x = traction_x;
    opts.traction_y = traction_y;
    opts.traction_z = traction_z;
    opts.boundary_tolerance = boundary_tolerance;
    opts.max_iter = max_iter;
    opts.rel_tol = rel_tol;
    opts.abs_tol = abs_tol;
    opts.print_level = print_level;

    std::string mesh_out_prefix;
    std::string disp_out_prefix;
    if (output_prefix != nullptr && std::string(output_prefix).size() > 0) {
      mesh_out_prefix = std::string(output_prefix) + ".mesh";
      disp_out_prefix = std::string(output_prefix) + ".disp";
    }

    mono::ElasticitySolver solver(MPI_COMM_WORLD);
    const mono::ElasticityResult result =
        solver.Solve(mesh_file, opts, mesh_out_prefix, disp_out_prefix);

    if (rank == 0) {
      std::cout << "[elasticity] MPI size=" << world << "\n"
                << "[elasticity] local NE=" << result.num_elements
                << ", local NV=" << result.num_vertices << "\n"
                << "[elasticity] CG iters=" << result.num_cg_iterations
                << ", final residual=" << result.final_residual_norm << "\n"
                << "[elasticity] ||u||_2=" << result.displacement_l2_norm
                << ", ||u||_inf=" << result.displacement_max_abs << std::endl;
    }
  } catch (const std::exception& ex) {
    std::cerr << "[rank " << rank << "/" << world << "] elasticity failed: " << ex.what() << std::endl;
    return 2;
  }

  return 0;
}
