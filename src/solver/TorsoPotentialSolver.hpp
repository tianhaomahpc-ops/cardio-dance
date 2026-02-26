#pragma once

#include <limits>
#include <memory>

#include "mfem.hpp"

#include "config/SimulationConfig.hpp"

namespace mono {

// Solve torso potential using penalty-enforced interface constraints:
//   -div(sigma_T grad(uT)) = 0
// with approximate Dirichlet data on mapped interface nodes.
class TorsoPotentialSolver {
 public:
  TorsoPotentialSolver(const SimulationConfig& cfg, MPI_Comm comm);
  TorsoPotentialSolver(const SimulationConfig& cfg,
                       MPI_Comm comm,
                       std::unique_ptr<mfem::ParMesh> pmesh_override);

  void SetConstrainedDofs(const mfem::Array<int>& constrained_tdofs);
  void Solve(const mfem::Vector& interface_bc_true);

  mfem::ParFiniteElementSpace& PFES() { return *pfes_; }
  const mfem::ParFiniteElementSpace& PFES() const { return *pfes_; }
  mfem::ParGridFunction& UT() { return *ut_gf_; }
  const mfem::ParGridFunction& UT() const { return *ut_gf_; }
  const mfem::Vector& UTTrue() const { return ut_true_; }
  int NumConstrainedDofs() const { return num_constrained_dofs_; }
  int LastNumIterations() const { return last_num_iterations_; }
  double LastFinalNorm() const { return last_final_norm_; }

 private:
  class PenaltyOperator : public mfem::Operator {
   public:
    PenaltyOperator(const mfem::HypreParMatrix& K, const mfem::Vector& mask, double penalty);
    void Mult(const mfem::Vector& x, mfem::Vector& y) const override;

   private:
    const mfem::HypreParMatrix& K_;
    const mfem::Vector& mask_;
    double penalty_ = 0.0;
  };

  const SimulationConfig& cfg_;
  MPI_Comm comm_;

  std::unique_ptr<mfem::Mesh> serial_mesh_;
  std::unique_ptr<mfem::ParMesh> pmesh_;
  std::unique_ptr<mfem::H1_FECollection> fec_;
  std::unique_ptr<mfem::ParFiniteElementSpace> pfes_;
  std::unique_ptr<mfem::ParGridFunction> ut_gf_;

  std::unique_ptr<mfem::ParBilinearForm> k_form_;
  std::unique_ptr<mfem::HypreParMatrix> K_;

  mfem::Vector mask_true_;
  mfem::Vector rhs_;
  mfem::Vector ut_true_;
  int num_constrained_dofs_ = 0;

  std::unique_ptr<PenaltyOperator> penalty_operator_;
  std::unique_ptr<mfem::CGSolver> cg_;
  int last_num_iterations_ = -1;
  double last_final_norm_ = std::numeric_limits<double>::quiet_NaN();
};

}  // namespace mono
