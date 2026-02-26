#include "solver/TorsoPotentialSolver.hpp"

#include <stdexcept>

namespace mono {

TorsoPotentialSolver::PenaltyOperator::PenaltyOperator(const mfem::HypreParMatrix& K,
                                                       const mfem::Vector& mask,
                                                       double penalty)
    : mfem::Operator(K.Height()), K_(K), mask_(mask), penalty_(penalty) {}

void TorsoPotentialSolver::PenaltyOperator::Mult(const mfem::Vector& x, mfem::Vector& y) const {
  K_.Mult(x, y);
  for (int i = 0; i < y.Size(); ++i) {
    y[i] += penalty_ * mask_[i] * x[i];
  }
}

TorsoPotentialSolver::TorsoPotentialSolver(const SimulationConfig& cfg, MPI_Comm comm)
    : cfg_(cfg), comm_(comm) {
  if (cfg_.torso_mesh_path.empty()) {
    throw std::runtime_error("TorsoPotentialSolver requires torso_mesh_path");
  }

  serial_mesh_ = std::make_unique<mfem::Mesh>(cfg_.torso_mesh_path.c_str(), 1, 1);
  pmesh_ = std::make_unique<mfem::ParMesh>(comm_, *serial_mesh_);
  const int dim = pmesh_->Dimension();

  fec_ = std::make_unique<mfem::H1_FECollection>(1, dim);
  pfes_ = std::make_unique<mfem::ParFiniteElementSpace>(pmesh_.get(), fec_.get());
  ut_gf_ = std::make_unique<mfem::ParGridFunction>(pfes_.get());
  *ut_gf_ = 0.0;

  k_form_ = std::make_unique<mfem::ParBilinearForm>(pfes_.get());
  mfem::ConstantCoefficient sigma_torso(cfg_.sigma_torso_mS_per_mm);
  k_form_->AddDomainIntegrator(new mfem::DiffusionIntegrator(sigma_torso));
  k_form_->Assemble();
  k_form_->Finalize();
  K_.reset(k_form_->ParallelAssemble());

  const int n_true = pfes_->GetTrueVSize();
  mask_true_.SetSize(n_true);
  mask_true_ = 0.0;
  rhs_.SetSize(n_true);
  rhs_ = 0.0;
  ut_true_.SetSize(n_true);
  ut_true_ = 0.0;

  penalty_operator_ =
      std::make_unique<PenaltyOperator>(*K_, mask_true_, cfg_.torso_dirichlet_penalty);
  cg_ = std::make_unique<mfem::CGSolver>(comm_);
  cg_->iterative_mode = true;
  cg_->SetRelTol(cfg_.ksp_rtol);
  cg_->SetAbsTol(0.0);
  cg_->SetMaxIter(cfg_.ksp_max_it);
  cg_->SetPrintLevel(0);
  cg_->SetOperator(*penalty_operator_);
}

TorsoPotentialSolver::TorsoPotentialSolver(const SimulationConfig& cfg,
                                           MPI_Comm comm,
                                           std::unique_ptr<mfem::ParMesh> pmesh_override)
    : cfg_(cfg), comm_(comm) {
  if (!pmesh_override) {
    throw std::runtime_error("TorsoPotentialSolver requires non-null pmesh_override");
  }
  pmesh_ = std::move(pmesh_override);
  const int dim = pmesh_->Dimension();

  fec_ = std::make_unique<mfem::H1_FECollection>(1, dim);
  pfes_ = std::make_unique<mfem::ParFiniteElementSpace>(pmesh_.get(), fec_.get());
  ut_gf_ = std::make_unique<mfem::ParGridFunction>(pfes_.get());
  *ut_gf_ = 0.0;

  k_form_ = std::make_unique<mfem::ParBilinearForm>(pfes_.get());
  mfem::ConstantCoefficient sigma_torso(cfg_.sigma_torso_mS_per_mm);
  k_form_->AddDomainIntegrator(new mfem::DiffusionIntegrator(sigma_torso));
  k_form_->Assemble();
  k_form_->Finalize();
  K_.reset(k_form_->ParallelAssemble());

  const int n_true = pfes_->GetTrueVSize();
  mask_true_.SetSize(n_true);
  mask_true_ = 0.0;
  rhs_.SetSize(n_true);
  rhs_ = 0.0;
  ut_true_.SetSize(n_true);
  ut_true_ = 0.0;

  penalty_operator_ =
      std::make_unique<PenaltyOperator>(*K_, mask_true_, cfg_.torso_dirichlet_penalty);
  cg_ = std::make_unique<mfem::CGSolver>(comm_);
  cg_->iterative_mode = true;
  cg_->SetRelTol(cfg_.ksp_rtol);
  cg_->SetAbsTol(0.0);
  cg_->SetMaxIter(cfg_.ksp_max_it);
  cg_->SetPrintLevel(0);
  cg_->SetOperator(*penalty_operator_);
}

void TorsoPotentialSolver::SetConstrainedDofs(const mfem::Array<int>& constrained_tdofs) {
  mask_true_ = 0.0;
  num_constrained_dofs_ = 0;
  for (int i = 0; i < constrained_tdofs.Size(); ++i) {
    const int tdof = constrained_tdofs[i];
    if (tdof < 0 || tdof >= mask_true_.Size()) {
      throw std::runtime_error("TorsoPotentialSolver constrained dof out of range");
    }
    if (mask_true_[tdof] < 0.5) {
      mask_true_[tdof] = 1.0;
      ++num_constrained_dofs_;
    }
  }
}

void TorsoPotentialSolver::Solve(const mfem::Vector& interface_bc_true) {
  if (interface_bc_true.Size() != ut_true_.Size()) {
    throw std::runtime_error("TorsoPotentialSolver::Solve got mismatched interface_bc size");
  }

  rhs_ = 0.0;
  for (int i = 0; i < rhs_.Size(); ++i) {
    if (mask_true_[i] > 0.5) {
      rhs_[i] = cfg_.torso_dirichlet_penalty * interface_bc_true[i];
      // Keep constrained nodes close to target value in warm start.
      ut_true_[i] = interface_bc_true[i];
    }
  }

  cg_->Mult(rhs_, ut_true_);
  last_num_iterations_ = cg_->GetNumIterations();
  last_final_norm_ = cg_->GetFinalNorm();
  for (int i = 0; i < ut_true_.Size(); ++i) {
    if (mask_true_[i] > 0.5) {
      ut_true_[i] = interface_bc_true[i];
    }
  }
  ut_gf_->SetFromTrueDofs(ut_true_);
}

}  // namespace mono
