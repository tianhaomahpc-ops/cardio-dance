#pragma once

#include <memory>

#include "mfem.hpp"

#include "mechanics/ElasticitySolver.hpp"

namespace mono {

struct CalciumDrivenElasticityOptions {
  ElasticityOptions elasticity;
  // Hill activation based on cytosolic Ca2+ (mM):
  // a = Ca^n / (Ca^n + Ca50^n)
  double ca_half_mM = 2e-4;
  double ca_hill = 2.0;
};

struct CalciumDrivenElasticityStats {
  int num_cg_iterations = 0;
  double final_residual_norm = 0.0;
  double displacement_l2_norm = 0.0;
  double displacement_max_abs = 0.0;
  double calcium_mean_mM = 0.0;
  double activation = 0.0;
  double traction_x = 0.0;
  double traction_y = 0.0;
  double traction_z = 0.0;
};

// Quasi-static elasticity solve on a fixed heart mesh, driven by Ca2+-dependent
// boundary traction scaling at each output frame.
class CalciumDrivenElasticitySolver {
 public:
  CalciumDrivenElasticitySolver(MPI_Comm comm,
                                const mfem::ParMesh& reference_mesh,
                                int expected_cai_true_size,
                                const CalciumDrivenElasticityOptions& options);

  CalciumDrivenElasticityStats SolveFromCytosolicCalcium(const mfem::Vector& cai_true);

  mfem::ParMesh& Mesh() { return *pmesh_; }
  const mfem::ParMesh& Mesh() const { return *pmesh_; }
  mfem::ParGridFunction& Displacement() { return *displacement_; }
  const mfem::ParGridFunction& Displacement() const { return *displacement_; }

 private:
  static void RelabelBoundaryByXAxis(mfem::ParMesh& mesh,
                                     double rel_tol,
                                     int fixed_attr,
                                     int traction_attr);

  MPI_Comm comm_;
  CalciumDrivenElasticityOptions options_;
  int expected_cai_true_size_ = 0;

  std::unique_ptr<mfem::ParMesh> pmesh_;
  std::unique_ptr<mfem::H1_FECollection> fec_;
  std::unique_ptr<mfem::ParFiniteElementSpace> fes_;
  std::unique_ptr<mfem::ParBilinearForm> a_form_;
  std::unique_ptr<mfem::ParGridFunction> displacement_;
  mfem::Array<int> ess_tdofs_;
  mfem::Array<int> traction_bdr_;
};

}  // namespace mono

