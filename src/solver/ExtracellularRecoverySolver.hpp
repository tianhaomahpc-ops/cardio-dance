#pragma once

#include <limits>
#include <memory>

#include "mfem.hpp"

#include "config/SimulationConfig.hpp"
#include "space/Assembler.hpp"
#include "space/FiberTensorCoefficient.hpp"

namespace mono {

// Recover extracellular potential ue on the heart domain:
//   -div((sigma_i + sigma_e) grad(ue)) = div(sigma_i grad(vm))
// implemented in discrete form as K1 * ue = -K2 * vm with one pinned dof.
class ExtracellularRecoverySolver {
 public:
  ExtracellularRecoverySolver(const SimulationConfig& cfg, Assembler& assembler, MPI_Comm comm);

  void Solve(const mfem::Vector& vm_true);

  const mfem::Vector& UeTrue() const { return ue_true_; }
  mfem::ParGridFunction& Ue() { return *ue_gf_; }
  const mfem::ParGridFunction& Ue() const { return *ue_gf_; }
  int LastNumIterations() const { return last_num_iterations_; }
  double LastFinalNorm() const { return last_final_norm_; }

 private:
  class PinnedOperator : public mfem::Operator {
   public:
    PinnedOperator(const mfem::HypreParMatrix& K, int local_pin_dof, double penalty);
    void Mult(const mfem::Vector& x, mfem::Vector& y) const override;

   private:
    const mfem::HypreParMatrix& K_;
    int local_pin_dof_ = -1;
    double penalty_ = 0.0;
  };

  const SimulationConfig& cfg_;
  Assembler& assembler_;
  MPI_Comm comm_;
  int rank_ = 0;

  std::unique_ptr<mfem::VectorGridFunctionCoefficient> fiber_f_coeff_;
  std::unique_ptr<mfem::VectorGridFunctionCoefficient> fiber_s_coeff_;
  std::unique_ptr<mfem::VectorGridFunctionCoefficient> fiber_n_coeff_;
  std::unique_ptr<mfem::VectorConstantCoefficient> const_f_coeff_;
  std::unique_ptr<mfem::VectorConstantCoefficient> const_s_coeff_;
  std::unique_ptr<mfem::VectorConstantCoefficient> const_n_coeff_;
  mfem::VectorCoefficient* f_coeff_ = nullptr;
  mfem::VectorCoefficient* s_coeff_ = nullptr;
  mfem::VectorCoefficient* n_coeff_ = nullptr;

  std::unique_ptr<FiberTensorCoefficient> k1_coeff_;
  std::unique_ptr<FiberTensorCoefficient> k2_coeff_;
  std::unique_ptr<mfem::ParBilinearForm> k1_form_;
  std::unique_ptr<mfem::ParBilinearForm> k2_form_;
  std::unique_ptr<mfem::HypreParMatrix> K1_;
  std::unique_ptr<mfem::HypreParMatrix> K2_;

  int local_pin_dof_ = -1;
  double pin_penalty_ = 1e8;

  mfem::Vector ue_true_;
  mfem::Vector rhs_;
  std::unique_ptr<mfem::ParGridFunction> ue_gf_;

  std::unique_ptr<PinnedOperator> pinned_operator_;
  std::unique_ptr<mfem::CGSolver> cg_;
  int last_num_iterations_ = -1;
  double last_final_norm_ = std::numeric_limits<double>::quiet_NaN();
};

}  // namespace mono
