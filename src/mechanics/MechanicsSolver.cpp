#include "mechanics/MechanicsSolver.hpp"

#include <stdexcept>

#include "mechanics/ActiveStressCoefficient.hpp"
#include "mechanics/HolzapfelOgdenModel.hpp"

namespace mono {

namespace {

// Adds the active stress contribution to the residual: integral T_a (Ff0)(X)f0 : grad N.
// We implement this via a HyperelasticIntegrator wrapping an "active model"
// W_active = T_a * (F f0 . f0). This is not strictly a strain energy but
// dW/dF = T_a * (Ff0)(X)f0 reproduces the desired Piola. For tangent we
// neglect T_a * I_4f cross terms (loose coupling: T_a fixed during Newton).
class ActiveTensionHyperelasticModel : public mfem::HyperelasticModel {
 public:
  ActiveTensionHyperelasticModel(const mfem::ParGridFunction& ta_gf,
                                  mfem::VectorCoefficient& f0_coeff)
      : ta_gf_(ta_gf), f0_coeff_(f0_coeff), ta_coeff_(&ta_gf_) {}

  // Helper: fetch f0 and T_a at the current IP via base-class Ttr.
  bool EvalAtIP(double& Ta, mfem::Vector& f0, int dim) const {
    if (!this->Ttr) return false;
    const mfem::IntegrationPoint& ip = this->Ttr->GetIntPoint();
    Ta = ta_coeff_.Eval(*this->Ttr, ip);
    if (Ta <= 0.0) return false;
    f0.SetSize(dim);
    f0_coeff_.Eval(f0, *this->Ttr, ip);
    const double nf = f0.Norml2();
    if (nf < 1e-12) return false;
    f0 /= nf;
    return true;
  }

  double EvalW(const mfem::DenseMatrix& F) const override {
    double Ta;
    mfem::Vector f0;
    if (!EvalAtIP(Ta, f0, F.Size())) return 0.0;
    mfem::Vector Ff0(F.Size());
    F.Mult(f0, Ff0);
    double dot = 0.0;
    for (int i = 0; i < f0.Size(); ++i) dot += Ff0(i) * f0(i);
    return Ta * (dot - 1.0);
  }

  void EvalP(const mfem::DenseMatrix& F, mfem::DenseMatrix& P) const override {
    const int dim = F.Size();
    P.SetSize(dim);
    P = 0.0;
    double Ta;
    mfem::Vector f0;
    if (!EvalAtIP(Ta, f0, dim)) return;
    mfem::Vector Ff0(dim);
    F.Mult(f0, Ff0);
    for (int i = 0; i < dim; ++i)
      for (int j = 0; j < dim; ++j)
        P(i, j) = Ta * Ff0(i) * f0(j);
  }

  void AssembleH(const mfem::DenseMatrix& F,
                 const mfem::DenseMatrix& DS,
                 const double weight,
                 mfem::DenseMatrix& A) const override {
    const int dim = F.Size();
    const int dof = DS.Height();
    double Ta;
    mfem::Vector f0;
    if (!EvalAtIP(Ta, f0, dim)) return;

    // A((k_a)*dim+i, (k_b)*dim+j) += w * DS(k_a,m) * (Ta delta_{i,j_?} ...) * DS(k_b,n).
    // dP_{ai}/dF_{bj} = T_a * delta_{a,b} f0_j f0_i
    // sum over m: DS(k_a, m) * dP_{i,m}/dF_{j,n} * DS(k_b, n)
    //          = T_a * sum_m DS(k_a,m) * delta_{i,j} * f0_m * f0_? ...
    // Cleaner: pre-compute c_{i,j,?,?} = T_a delta_{i,j} f0_?(j) f0_?(i)? Skip;
    // use explicit 4-index loop.
    for (int ka = 0; ka < dof; ++ka) {
      for (int kb = 0; kb < dof; ++kb) {
        const double w_kakb_mn_sum = [&]() {
          double acc = 0.0;
          for (int m = 0; m < dim; ++m) {
            for (int n = 0; n < dim; ++n) {
              acc += DS(ka, m) * f0(m) * DS(kb, n) * f0(n);
            }
          }
          return acc;
        }();
        for (int i = 0; i < dim; ++i) {
          // dP_{ai}/dF_{aj} = T_a f0_j f0_i  (only diagonal in 'a').
          // Contracted: sum_{m,n} DS(ka,m) DS(kb,n) f0(m) f0(n) * Ta (only when i==j).
          A(ka * dim + i, kb * dim + i) += weight * Ta * w_kakb_mn_sum;
        }
      }
    }
  }

 private:
  const mfem::ParGridFunction& ta_gf_;
  mfem::VectorCoefficient& f0_coeff_;
  mutable mfem::GridFunctionCoefficient ta_coeff_;
};

}  // namespace

MechanicsSolver::MechanicsSolver(const SimulationConfig& cfg,
                                  mfem::ParFiniteElementSpace& ep_pfes,
                                  mfem::VectorCoefficient& f0_coeff,
                                  mfem::VectorCoefficient& s0_coeff,
                                  MPI_Comm comm)
    : cfg_(cfg),
      comm_(comm),
      ep_pfes_(ep_pfes),
      f0_coeff_(f0_coeff),
      s0_coeff_(s0_coeff) {
  ho_params_.a   = cfg.mech_ho_a;
  ho_params_.b   = cfg.mech_ho_b;
  ho_params_.af  = cfg.mech_ho_af;
  ho_params_.bf  = cfg.mech_ho_bf;
  ho_params_.as  = cfg.mech_ho_as;
  ho_params_.bs  = cfg.mech_ho_bs;
  ho_params_.afs = cfg.mech_ho_afs;
  ho_params_.bfs = cfg.mech_ho_bfs;
  ho_params_.kappa = cfg.mech_ho_kappa;

  BuildSpaceAndForm();
  IdentifyEssentialDofs();
}

MechanicsSolver::~MechanicsSolver() = default;

void MechanicsSolver::BuildSpaceAndForm() {
  mfem::ParMesh* pmesh = ep_pfes_.GetParMesh();
  const int dim = pmesh->Dimension();
  if (dim != 3) {
    throw std::runtime_error("MechanicsSolver: 3D mesh required");
  }
  const int order = ep_pfes_.GetFE(0)->GetOrder();
  vec_fec_ = std::make_unique<mfem::H1_FECollection>(order, dim);
  vec_fes_ = std::make_unique<mfem::ParFiniteElementSpace>(pmesh, vec_fec_.get(), dim);
  u_ = std::make_unique<mfem::ParGridFunction>(vec_fes_.get());
  *u_ = 0.0;

  ho_model_ = std::make_unique<HolzapfelOgdenModel>(ho_params_, f0_coeff_, s0_coeff_);

  nlform_ = std::make_unique<mfem::ParNonlinearForm>(vec_fes_.get());
  // Passive HO hyperelastic integrator.
  nlform_->AddDomainIntegrator(new mfem::HyperelasticNLFIntegrator(ho_model_.get()));
}

void MechanicsSolver::IdentifyEssentialDofs() {
  ess_tdofs_.SetSize(0);
  mfem::ParMesh* pmesh = ep_pfes_.GetParMesh();
  if (pmesh->bdr_attributes.Size() == 0) return;
  const int max_attr = pmesh->bdr_attributes.Max();
  mfem::Array<int> ess_marker(max_attr);
  ess_marker = 0;
  if (cfg_.mech_bdr_base_attr >= 1 && cfg_.mech_bdr_base_attr <= max_attr) {
    ess_marker[cfg_.mech_bdr_base_attr - 1] = 1;
  }
  vec_fes_->GetEssentialTrueDofs(ess_marker, ess_tdofs_);
  nlform_->SetEssentialTrueDofs(ess_tdofs_);
}

void MechanicsSolver::SetActiveTension(const mfem::ParGridFunction& ta_kPa) {
  if (ta_gf_ == &ta_kPa) return;
  ta_gf_ = &ta_kPa;
  // Add the active-tension integrator on first bind; subsequent T_a updates
  // flow through the bound ParGridFunction reference inside active_model_.
  if (!active_integrator_added_) {
    active_model_ = std::make_unique<ActiveTensionHyperelasticModel>(*ta_gf_, f0_coeff_);
    nlform_->AddDomainIntegrator(new mfem::HyperelasticNLFIntegrator(active_model_.get()));
    active_integrator_added_ = true;
  }
}

void MechanicsSolver::SetEndocardialPressurePa(double p_pa) {
  endo_pressure_pa_ = p_pa;
  // Note: pressure-follower implementation requires a custom boundary integrator
  // to assemble -p * (J F^{-T} n_ref) on bdr_endo_attr. Left as a follow-up
  // (Phase 1d) — currently the load is silently ignored if not connected.
  // For initial wiring, document and proceed.
}

void MechanicsSolver::EnsureSolver() {
  if (newton_) return;
#ifdef MFEM_USE_PETSC
  auto* sn = new mfem::PetscNonlinearSolver(comm_, *nlform_, "-mech_");
  sn->SetMaxIter(cfg_.mech_snes_max_it);
  sn->SetRelTol(cfg_.mech_snes_rtol);
  sn->SetAbsTol(cfg_.mech_snes_atol);
  sn->SetPrintLevel(cfg_.mech_snes_print_level);
  newton_.reset(sn);
#else
  auto* gmres = new mfem::HypreGMRES(comm_);
  gmres->SetMaxIter(cfg_.mech_ksp_max_it);
  gmres->SetTol(cfg_.mech_ksp_rtol);
  gmres->SetKDim(60);
  ksp_.reset(gmres);
  auto* nw = new mfem::NewtonSolver(comm_);
  nw->SetOperator(*nlform_);
  nw->SetSolver(*ksp_);
  nw->SetMaxIter(cfg_.mech_snes_max_it);
  nw->SetRelTol(cfg_.mech_snes_rtol);
  nw->SetAbsTol(cfg_.mech_snes_atol);
  nw->SetPrintLevel(cfg_.mech_snes_print_level);
  newton_.reset(nw);
#endif
}

int MechanicsSolver::Solve() {
  EnsureSolver();

  mfem::Vector x_true(vec_fes_->GetTrueVSize());
  u_->GetTrueDofs(x_true);

  // Apply zero on essential dofs as starting condition (clamped base).
  for (int i = 0; i < ess_tdofs_.Size(); ++i) x_true[ess_tdofs_[i]] = 0.0;

  mfem::Vector zero;  // empty rhs => Newton solves R(u) = 0
  newton_->Mult(zero, x_true);

  u_->SetFromTrueDofs(x_true);
  return cfg_.mech_snes_max_it;  // SNES iterations are accessible via PETSc API; placeholder
}

void MechanicsSolver::ComputeFiberStretch(mfem::ParGridFunction& lambda_gf) const {
  lambda_gf.SetSpace(&ep_pfes_);
  lambda_gf = 1.0;
  // Per-true-DOF nodal stretch: lambda = || (I + grad u) f0 ||.
  // Approximate by averaging element gradients at nodes (P1).
  mfem::ParGridFunction grad_norm(&ep_pfes_);
  grad_norm = 0.0;
  // For initial wiring we leave nodal averaging to a follow-up; return 1.0.
}

void MechanicsSolver::ComputeJacobianDet(mfem::ParGridFunction& J_gf) const {
  J_gf.SetSpace(&ep_pfes_);
  J_gf = 1.0;
  // Same note as ComputeFiberStretch.
}

}  // namespace mono
