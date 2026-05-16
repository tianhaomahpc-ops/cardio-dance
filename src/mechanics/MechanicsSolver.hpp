#pragma once

#include <memory>

#include "mfem.hpp"

#include "config/SimulationConfig.hpp"
#include "mechanics/HolzapfelOgdenMaterial.hpp"

namespace mono {

// Quasi-static finite-strain mechanics solver tailored for cardiac LV
// (Jiang-Chen-Cai 2020 style):
//   * Holzapfel-Ogden anisotropic hyperelasticity (passive)
//   * Active stress along fiber direction supplied as a ParGridFunction
//   * Newton-Krylov outer loop (PETSc SNES when available, MFEM Newton fallback)
//   * Overlapping additive Schwarz preconditioner via PETSc PCASM
//   * Boundary conditions:
//       - essential displacement clamp on bdr_base_attr
//       - Robin pericardial spring on bdr_epi_attr (k_peri u.n)
//       - Neumann endocardial pressure on bdr_endo_attr (-p J F^-T n_ref)
//
// This solver shares the EP ParMesh; it builds its own vector-valued
// ParFiniteElementSpace (dim copies of EP scalar space).
class MechanicsSolver {
 public:
  MechanicsSolver(const SimulationConfig& cfg,
                  mfem::ParFiniteElementSpace& ep_pfes,
                  mfem::VectorCoefficient& f0_coeff,
                  mfem::VectorCoefficient& s0_coeff,
                  MPI_Comm comm);

  ~MechanicsSolver();

  // Update active tension field for the next solve (kPa, scalar grid function
  // on the EP space). Stored by reference; caller must keep it alive.
  void SetActiveTension(const mfem::ParGridFunction& ta_kPa);

  // Update endocardial pressure (Pa). Quasi-static load.
  void SetEndocardialPressurePa(double p_pa);

  // Solve quasi-static problem for displacement; returns SNES iteration count.
  int Solve();

  // Accessors.
  mfem::ParFiniteElementSpace& VectorFES() { return *vec_fes_; }
  mfem::ParGridFunction& Displacement() { return *u_; }
  const mfem::ParGridFunction& Displacement() const { return *u_; }

  // Returns per-node sarcomere stretch lambda along the fiber direction
  // computed from current displacement: lambda = ||F f0||.
  void ComputeFiberStretch(mfem::ParGridFunction& lambda_scalar_gf) const;

  // Returns J = det(F) sampled at nodes (P1 only; integration-point exact
  // requires QuadratureSpace which is left for a follow-up).
  void ComputeJacobianDet(mfem::ParGridFunction& J_scalar_gf) const;

 private:
  const SimulationConfig& cfg_;
  MPI_Comm comm_;
  mfem::ParFiniteElementSpace& ep_pfes_;
  mfem::VectorCoefficient& f0_coeff_;
  mfem::VectorCoefficient& s0_coeff_;

  std::unique_ptr<mfem::H1_FECollection> vec_fec_;
  std::unique_ptr<mfem::ParFiniteElementSpace> vec_fes_;
  std::unique_ptr<mfem::ParGridFunction> u_;

  HolzapfelOgdenMaterial::Params ho_params_;
  std::unique_ptr<class HolzapfelOgdenModel> ho_model_;
  // Active-tension model: kept here because mfem::ParNonlinearForm takes
  // ownership of the integrator but not of the underlying HyperelasticModel.
  std::unique_ptr<mfem::HyperelasticModel> active_model_;
  bool active_integrator_added_ = false;

  // Nonlinear residual form (passive + active + Robin + Neumann).
  std::unique_ptr<mfem::ParNonlinearForm> nlform_;

  // Active tension grid function (non-owning).
  const mfem::ParGridFunction* ta_gf_ = nullptr;
  double endo_pressure_pa_ = 0.0;
  bool endo_pressure_integrator_added_ = false;
  // Backing storage for the endo-pressure integrator's pressure value
  // (kPa, dead-load). The integrator holds a const reference into this so
  // that SetEndocardialPressurePa can re-tune the load without rebuilding
  // the nonlinear form.
  double endo_pressure_kpa_ref_ = 0.0;
  mfem::Array<int> endo_marker_;

  // Essential dofs (base clamp).
  mfem::Array<int> ess_tdofs_;
  // Boundary-attribute marker for the pericardial Robin spring (epi face);
  // owned here because mfem::ParNonlinearForm stores it by reference.
  mfem::Array<int> epi_marker_;

  // PETSc-based Newton-Krylov; allocated lazily on first Solve().
  std::unique_ptr<mfem::Solver> newton_;
  std::unique_ptr<mfem::Solver> ksp_;
  std::unique_ptr<mfem::Solver> prec_;

  void BuildSpaceAndForm();
  void IdentifyEssentialDofs();
  void EnsureSolver();
};

}  // namespace mono
