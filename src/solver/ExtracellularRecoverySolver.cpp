#include "solver/ExtracellularRecoverySolver.hpp"

#include <algorithm>
#include <stdexcept>

namespace mono {

ExtracellularRecoverySolver::PinnedOperator::PinnedOperator(const mfem::HypreParMatrix& K,
                                                            int local_pin_dof,
                                                            double penalty)
    : mfem::Operator(K.Height()),
      K_(K),
      local_pin_dof_(local_pin_dof),
      penalty_(penalty) {}

void ExtracellularRecoverySolver::PinnedOperator::Mult(const mfem::Vector& x, mfem::Vector& y) const {
  K_.Mult(x, y);
  if (local_pin_dof_ >= 0) {
    y[local_pin_dof_] += penalty_ * x[local_pin_dof_];
  }
}

ExtracellularRecoverySolver::ExtracellularRecoverySolver(const SimulationConfig& cfg,
                                                         Assembler& assembler,
                                                         MPI_Comm comm)
    : cfg_(cfg), assembler_(assembler), comm_(comm) {
  MPI_Comm_rank(comm_, &rank_);

  const int n_true = assembler_.TrueVSize();
  if (n_true <= 0) {
    throw std::runtime_error("ExtracellularRecoverySolver requires non-empty heart FE space");
  }

  const int dim = assembler_.PFES().GetParMesh()->Dimension();
  mfem::Vector f(dim), s(dim), n(dim);
  f = 0.0;
  s = 0.0;
  n = 0.0;
  if (dim > 0) f[0] = 1.0;
  if (dim > 1) s[1] = 1.0;
  if (dim > 2) n[2] = 1.0;

  if (assembler_.HasFiberFields()) {
    fiber_f_coeff_ = std::make_unique<mfem::VectorGridFunctionCoefficient>(assembler_.FiberF());
    fiber_s_coeff_ = std::make_unique<mfem::VectorGridFunctionCoefficient>(assembler_.FiberS());
    fiber_n_coeff_ = std::make_unique<mfem::VectorGridFunctionCoefficient>(assembler_.FiberN());
    f_coeff_ = fiber_f_coeff_.get();
    s_coeff_ = fiber_s_coeff_.get();
    n_coeff_ = fiber_n_coeff_.get();
  } else {
    const_f_coeff_ = std::make_unique<mfem::VectorConstantCoefficient>(f);
    const_s_coeff_ = std::make_unique<mfem::VectorConstantCoefficient>(s);
    const_n_coeff_ = std::make_unique<mfem::VectorConstantCoefficient>(n);
    f_coeff_ = const_f_coeff_.get();
    s_coeff_ = const_s_coeff_.get();
    n_coeff_ = const_n_coeff_.get();
  }

  k1_coeff_ = std::make_unique<FiberTensorCoefficient>(dim,
                                                        cfg_.sigma_i_f_mS_per_mm + cfg_.sigma_e_f_mS_per_mm,
                                                        cfg_.sigma_i_s_mS_per_mm + cfg_.sigma_e_s_mS_per_mm,
                                                        cfg_.sigma_i_n_mS_per_mm + cfg_.sigma_e_n_mS_per_mm,
                                                        *f_coeff_,
                                                        *s_coeff_,
                                                        *n_coeff_,
                                                        cfg_.enable_regional_heart_models
                                                            ? cfg_.fibrosis_volume_attrs
                                                            : std::vector<int>{},
                                                        cfg_.enable_regional_heart_models
                                                            ? cfg_.fibrosis_sigma_scale
                                                            : 1.0);
  k2_coeff_ = std::make_unique<FiberTensorCoefficient>(dim,
                                                        cfg_.sigma_i_f_mS_per_mm,
                                                        cfg_.sigma_i_s_mS_per_mm,
                                                        cfg_.sigma_i_n_mS_per_mm,
                                                        *f_coeff_,
                                                        *s_coeff_,
                                                        *n_coeff_,
                                                        cfg_.enable_regional_heart_models
                                                            ? cfg_.fibrosis_volume_attrs
                                                            : std::vector<int>{},
                                                        cfg_.enable_regional_heart_models
                                                            ? cfg_.fibrosis_sigma_scale
                                                            : 1.0);

  k1_form_ = std::make_unique<mfem::ParBilinearForm>(&assembler_.PFES());
  k1_form_->AddDomainIntegrator(new mfem::DiffusionIntegrator(*k1_coeff_));
  k1_form_->Assemble();
  k1_form_->Finalize();
  K1_.reset(k1_form_->ParallelAssemble());

  k2_form_ = std::make_unique<mfem::ParBilinearForm>(&assembler_.PFES());
  k2_form_->AddDomainIntegrator(new mfem::DiffusionIntegrator(*k2_coeff_));
  k2_form_->Assemble();
  k2_form_->Finalize();
  K2_.reset(k2_form_->ParallelAssemble());

  const HYPRE_BigInt* offsets = assembler_.PFES().GetTrueDofOffsets();
  const HYPRE_BigInt local_begin = offsets[rank_];
  const HYPRE_BigInt local_end = offsets[rank_ + 1];
  if (0 >= local_begin && 0 < local_end) {
    local_pin_dof_ = 0;
  }
  pin_penalty_ = std::max(1e8, cfg_.torso_dirichlet_penalty);

  ue_true_.SetSize(n_true);
  ue_true_ = 0.0;
  rhs_.SetSize(n_true);
  rhs_ = 0.0;
  ue_gf_ = std::make_unique<mfem::ParGridFunction>(&assembler_.PFES());
  *ue_gf_ = 0.0;

  pinned_operator_ = std::make_unique<PinnedOperator>(*K1_, local_pin_dof_, pin_penalty_);
  cg_ = std::make_unique<mfem::CGSolver>(comm_);
  cg_->iterative_mode = true;
  cg_->SetRelTol(cfg_.ksp_rtol);
  cg_->SetAbsTol(0.0);
  cg_->SetMaxIter(cfg_.ksp_max_it);
  cg_->SetPrintLevel(0);
  cg_->SetOperator(*pinned_operator_);
}

void ExtracellularRecoverySolver::Solve(const mfem::Vector& vm_true) {
  if (vm_true.Size() != ue_true_.Size()) {
    throw std::runtime_error("ExtracellularRecoverySolver::Solve got mismatched vm size");
  }

  K2_->Mult(vm_true, rhs_);
  rhs_ *= -1.0;
  cg_->Mult(rhs_, ue_true_);
  last_num_iterations_ = cg_->GetNumIterations();
  last_final_norm_ = cg_->GetFinalNorm();
  if (local_pin_dof_ >= 0) {
    ue_true_[local_pin_dof_] = 0.0;
  }
  ue_gf_->SetFromTrueDofs(ue_true_);
}

}  // namespace mono
