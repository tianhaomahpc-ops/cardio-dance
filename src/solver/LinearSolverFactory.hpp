#pragma once

#include <limits>
#include <memory>
#include <vector>

#include "mfem.hpp"

#ifdef MFEM_USE_PETSC
#include <petscksp.h>
#endif

#include "config/SimulationConfig.hpp"

namespace mono {

// Unified interface over PETSc(KSP) and MFEM CG backends.
class LinearSystemSolver {
 public:
  LinearSystemSolver(const SimulationConfig& cfg,
                     MPI_Comm comm,
                     const mfem::ParFiniteElementSpace* pfes = nullptr);
  ~LinearSystemSolver();

  void SetOperator(const mfem::HypreParMatrix& A);
  // EM coupling: force re-creation of the cached PETSc AIJ matrix on the
  // next SetOperator() call (needed when A has been rebuilt due to
  // deformation-modified conductivity). No-op when MFEM_USE_PETSC is off.
  void InvalidatePetscOperator();
  void Solve(const mfem::Vector& rhs, mfem::Vector& x);
  int LastNumIterations() const { return last_num_iterations_; }
  double LastFinalNorm() const { return last_final_norm_; }

 private:
  bool use_petsc_;
  MPI_Comm comm_;
  const mfem::ParFiniteElementSpace* pfes_ = nullptr;
  int max_it_;
  double rtol_;
  bool use_hypre_boomeramg_ = true;
  bool petsc_use_geometric_asm_ = true;
  int petsc_asm_nx_ = 1;
  int petsc_asm_ny_ = 1;
  int petsc_asm_nz_ = 1;
  int last_num_iterations_ = -1;
  double last_final_norm_ = std::numeric_limits<double>::quiet_NaN();

  std::unique_ptr<mfem::CGSolver> cg_;
  std::unique_ptr<mfem::HypreBoomerAMG> amg_;

#ifdef MFEM_USE_PETSC
  std::unique_ptr<mfem::PetscLinearSolver> petsc_ksp_;
  std::unique_ptr<mfem::PetscParMatrix> petsc_A_;
  std::vector<IS> petsc_asm_subdomains_;
#endif
};

}  // namespace mono
