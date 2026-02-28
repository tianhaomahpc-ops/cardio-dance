#pragma once

#include <string>

#include "mfem.hpp"

namespace mono {

struct ElasticityOptions {
  int order = 1;
  double young_modulus = 10.0;
  double poisson_ratio = 0.30;

  // Boundary traction vector applied on the x-max boundary.
  double traction_x = 1.0;
  double traction_y = 0.0;
  double traction_z = 0.0;

  // Relative tolerance for tagging x-min/x-max boundary faces.
  double boundary_tolerance = 1e-6;

  int max_iter = 500;
  double rel_tol = 1e-10;
  double abs_tol = 1e-14;
  int print_level = 0;
};

struct ElasticityResult {
  int dim = 0;
  int num_elements = 0;
  int num_vertices = 0;
  int fixed_bdr_attr = 1;
  int traction_bdr_attr = 2;
  int num_cg_iterations = 0;
  double final_residual_norm = 0.0;
  double displacement_l2_norm = 0.0;
  double displacement_max_abs = 0.0;
};

class ElasticitySolver {
 public:
  explicit ElasticitySolver(MPI_Comm comm);

  ElasticityResult Solve(const std::string& mesh_path,
                         const ElasticityOptions& options,
                         const std::string& output_mesh_prefix = "",
                         const std::string& output_disp_prefix = "") const;

 private:
  static void RelabelBoundaryByXAxis(mfem::Mesh& mesh,
                                     double rel_tol,
                                     int fixed_attr,
                                     int traction_attr);

  MPI_Comm comm_;
};

}  // namespace mono
