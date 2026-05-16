#include "mechanics/MechanicsSolver.hpp"

#include <cmath>
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

  // Note: MFEM's HyperelasticNLFIntegrator passes Jpt = grad_X u (not F).
  // We add the identity before computing the active stress so the response
  // at the reference configuration (u = 0) is T_a * f0 (x) f0 -- nonzero.
  double EvalW(const mfem::DenseMatrix& Jpt) const override {
    double Ta;
    mfem::Vector f0;
    if (!EvalAtIP(Ta, f0, Jpt.Size())) return 0.0;
    const int dim = Jpt.Size();
    mfem::DenseMatrix F(Jpt);
    for (int k = 0; k < dim; ++k) F(k, k) += 1.0;
    mfem::Vector Ff0(dim);
    F.Mult(f0, Ff0);
    double dot = 0.0;
    for (int i = 0; i < dim; ++i) dot += Ff0(i) * f0(i);
    return Ta * (dot - 1.0);
  }

  void EvalP(const mfem::DenseMatrix& Jpt, mfem::DenseMatrix& P) const override {
    const int dim = Jpt.Size();
    P.SetSize(dim);
    P = 0.0;
    double Ta;
    mfem::Vector f0;
    if (!EvalAtIP(Ta, f0, dim)) return;
    mfem::DenseMatrix F(Jpt);
    for (int k = 0; k < dim; ++k) F(k, k) += 1.0;
    mfem::Vector Ff0(dim);
    F.Mult(f0, Ff0);
    for (int i = 0; i < dim; ++i)
      for (int j = 0; j < dim; ++j)
        P(i, j) = Ta * Ff0(i) * f0(j);
  }

  void AssembleH(const mfem::DenseMatrix& Jpt,
                 const mfem::DenseMatrix& DS,
                 const double weight,
                 mfem::DenseMatrix& A) const override {
    const int dim = Jpt.Size();
    const int dof = DS.Height();
    double Ta;
    mfem::Vector f0;
    if (!EvalAtIP(Ta, f0, dim)) return;

    // dP_{a,i}/dF_{b,j} = T_a delta_{a,b} f0_i f0_j  (with F = I + grad u, dF = d(grad u)).
    // sum_{m,n} DS(ka,m) * dP_{d_a,m}/dF_{d_b,n} * DS(kb,n)
    //        = T_a delta_{d_a,d_b} * (sum_m DS(ka,m) f0_m) * (sum_n DS(kb,n) f0_n)
    // vec_fes_ is byNODES: linear index = component*dof + node.
    mfem::Vector df_ka(dof), df_kb(dof);
    for (int ka = 0; ka < dof; ++ka) {
      double s = 0.0;
      for (int m = 0; m < dim; ++m) s += DS(ka, m) * f0(m);
      df_ka(ka) = s;
    }
    for (int da = 0; da < dim; ++da) {
      for (int ka = 0; ka < dof; ++ka) {
        const double row_term = weight * Ta * df_ka(ka);
        if (row_term == 0.0) continue;
        for (int kb = 0; kb < dof; ++kb) {
          A(da * dof + ka, da * dof + kb) += row_term * df_ka(kb);
        }
      }
    }
  }

 private:
  const mfem::ParGridFunction& ta_gf_;
  mfem::VectorCoefficient& f0_coeff_;
  mutable mfem::GridFunctionCoefficient ta_coeff_;
};

// lambda(x) = || (I + grad u(x)) f0(x) || at the current integration point.
class FiberStretchCoefficient : public mfem::Coefficient {
 public:
  FiberStretchCoefficient(const mfem::ParGridFunction& u,
                          mfem::VectorCoefficient& f0)
      : u_(u), f0_(f0) {}

  double Eval(mfem::ElementTransformation& T,
              const mfem::IntegrationPoint& ip) override {
    const int dim = T.GetSpaceDim();
    T.SetIntPoint(&ip);
    mfem::DenseMatrix grad_u(dim);
    u_.GetVectorGradient(T, grad_u);
    mfem::DenseMatrix F(grad_u);
    for (int k = 0; k < dim; ++k) F(k, k) += 1.0;
    mfem::Vector f0(dim);
    f0_.Eval(f0, T, ip);
    const double nf = f0.Norml2();
    if (nf > 1e-14) f0 /= nf;
    mfem::Vector Ff0(dim);
    F.Mult(f0, Ff0);
    return Ff0.Norml2();
  }

 private:
  const mfem::ParGridFunction& u_;
  mfem::VectorCoefficient& f0_;
};

// J(x) = det(I + grad u(x)) at the current integration point.
class JacobianDetCoefficient : public mfem::Coefficient {
 public:
  explicit JacobianDetCoefficient(const mfem::ParGridFunction& u) : u_(u) {}

  double Eval(mfem::ElementTransformation& T,
              const mfem::IntegrationPoint& ip) override {
    const int dim = T.GetSpaceDim();
    T.SetIntPoint(&ip);
    mfem::DenseMatrix grad_u(dim);
    u_.GetVectorGradient(T, grad_u);
    mfem::DenseMatrix F(grad_u);
    for (int k = 0; k < dim; ++k) F(k, k) += 1.0;
    return F.Det();
  }

 private:
  const mfem::ParGridFunction& u_;
};

// Pericardial normal Robin spring: traction t = -k (u . n) n on bdr_epi_attr.
// Contributes (residual) R_v = integral k (u . n)(v . n) dS, and
// (Jacobian)            J = integral k (n outer n)(phi_j phi_i) dS.
// Both forms are linear in u, so Jacobian is constant in u.
class NormalSpringBdrNLFI : public mfem::NonlinearFormIntegrator {
 public:
  explicit NormalSpringBdrNLFI(double k_kpa_per_mm) : k_(k_kpa_per_mm) {}

  void AssembleFaceVector(const mfem::FiniteElement& el1,
                          const mfem::FiniteElement& /*el2*/,
                          mfem::FaceElementTransformations& Tr,
                          const mfem::Vector& elfun,
                          mfem::Vector& elvect) override {
    const int dim = Tr.GetSpaceDim();
    const int nd = el1.GetDof();
    elvect.SetSize(elfun.Size());
    elvect = 0.0;

    // byNODES vector layout: u[d*nd + i] is component d at node i.
    mfem::DenseMatrix u_mat(elfun.GetData(), nd, dim);
    mfem::DenseMatrix r_mat(elvect.GetData(), nd, dim);

    mfem::Vector shape(nd);
    mfem::Vector u_ip(dim);
    mfem::Vector n_ref(dim);

    const int q_order = 2 * el1.GetOrder();
    const mfem::IntegrationRule* ir =
        &mfem::IntRules.Get(Tr.GetGeometryType(), q_order);

    for (int q = 0; q < ir->GetNPoints(); ++q) {
      const mfem::IntegrationPoint& ip = ir->IntPoint(q);
      Tr.SetAllIntPoints(&ip);
      const mfem::IntegrationPoint& eip1 = Tr.GetElement1IntPoint();
      el1.CalcShape(eip1, shape);

      // CalcOrtho returns the reference outward normal scaled by |J_face|.
      mfem::CalcOrtho(Tr.Jacobian(), n_ref);
      const double nlen = n_ref.Norml2();
      if (nlen < 1e-14) continue;

      // u(ip) at byNODES: u_ip[d] = sum_i shape[i] * u_mat(i, d)
      u_mat.MultTranspose(shape, u_ip);

      double un = 0.0;
      for (int d = 0; d < dim; ++d) un += u_ip(d) * n_ref(d);  // u . (|J_face| n_hat)
      // Surface measure = ip.weight * |J_face| = ip.weight * nlen.
      // contribution r_jd = w_phys * k * (u . n_hat) * shape_j * n_hat_d
      //                   = ip.weight * nlen * k * (un / nlen) * shape_j * n_ref_d / nlen
      //                   = ip.weight * k * un * shape_j * n_ref_d / nlen
      const double scale = ip.weight * k_ * un / nlen;
      for (int d = 0; d < dim; ++d) {
        const double common = scale * n_ref(d);
        for (int j = 0; j < nd; ++j) {
          r_mat(j, d) += common * shape(j);
        }
      }
    }
  }

  void AssembleFaceGrad(const mfem::FiniteElement& el1,
                        const mfem::FiniteElement& /*el2*/,
                        mfem::FaceElementTransformations& Tr,
                        const mfem::Vector& /*elfun*/,
                        mfem::DenseMatrix& elmat) override {
    const int dim = Tr.GetSpaceDim();
    const int nd = el1.GetDof();
    elmat.SetSize(nd * dim);
    elmat = 0.0;

    mfem::Vector shape(nd);
    mfem::Vector n_ref(dim);

    const int q_order = 2 * el1.GetOrder();
    const mfem::IntegrationRule* ir =
        &mfem::IntRules.Get(Tr.GetGeometryType(), q_order);

    for (int q = 0; q < ir->GetNPoints(); ++q) {
      const mfem::IntegrationPoint& ip = ir->IntPoint(q);
      Tr.SetAllIntPoints(&ip);
      const mfem::IntegrationPoint& eip1 = Tr.GetElement1IntPoint();
      el1.CalcShape(eip1, shape);

      mfem::CalcOrtho(Tr.Jacobian(), n_ref);
      const double nlen = n_ref.Norml2();
      if (nlen < 1e-14) continue;
      const double w = ip.weight * k_ / nlen;  // see AssembleFaceVector derivation.

      // elmat[(d*nd + j)][(e*nd + i)] += w * n_d * n_e * shape_j * shape_i
      for (int d = 0; d < dim; ++d) {
        for (int e = 0; e < dim; ++e) {
          const double nde = n_ref(d) * n_ref(e);
          if (std::abs(nde) < 1e-30) continue;
          for (int j = 0; j < nd; ++j) {
            const double sj = shape(j);
            for (int i = 0; i < nd; ++i) {
              elmat(d * nd + j, e * nd + i) += w * nde * sj * shape(i);
            }
          }
        }
      }
    }
  }

 private:
  double k_;
};

// Endocardial pressure dead-load on bdr_endo_attr. Reference traction
//   t_ref = -p n_endo_outward
// On the endo surface the solid's outward normal points INTO the cavity,
// so t_ref points OUT of the cavity, inflating the wall. The contribution
// to the residual R = -integral t_ref . v dA = +integral p (n_ref . v) dA.
// Because this is a Lagrangian dead-load (no F dependence), the Jacobian
// w.r.t. u is zero.
//
// Holds a const reference to the kPa pressure so that
// SetEndocardialPressurePa can ramp the load without rebuilding the form.
class EndoPressureDeadLoadBdrNLFI : public mfem::NonlinearFormIntegrator {
 public:
  explicit EndoPressureDeadLoadBdrNLFI(const double& p_kpa_ref)
      : p_kpa_(p_kpa_ref) {}

  void AssembleFaceVector(const mfem::FiniteElement& el1,
                          const mfem::FiniteElement& /*el2*/,
                          mfem::FaceElementTransformations& Tr,
                          const mfem::Vector& elfun,
                          mfem::Vector& elvect) override {
    const int dim = Tr.GetSpaceDim();
    const int nd = el1.GetDof();
    elvect.SetSize(elfun.Size());
    elvect = 0.0;
    if (p_kpa_ == 0.0) return;

    mfem::DenseMatrix r_mat(elvect.GetData(), nd, dim);
    mfem::Vector shape(nd);
    mfem::Vector n_ref(dim);

    const int q_order = 2 * el1.GetOrder();
    const mfem::IntegrationRule* ir =
        &mfem::IntRules.Get(Tr.GetGeometryType(), q_order);

    for (int q = 0; q < ir->GetNPoints(); ++q) {
      const mfem::IntegrationPoint& ip = ir->IntPoint(q);
      Tr.SetAllIntPoints(&ip);
      const mfem::IntegrationPoint& eip1 = Tr.GetElement1IntPoint();
      el1.CalcShape(eip1, shape);

      // CalcOrtho returns the reference outward normal scaled by |J_face|;
      // surface measure dS = ip.weight * |J_face|, absorbed into n_ref.
      mfem::CalcOrtho(Tr.Jacobian(), n_ref);
      const double nlen = n_ref.Norml2();
      if (nlen < 1e-14) continue;

      const double w = ip.weight * p_kpa_;
      for (int d = 0; d < dim; ++d) {
        const double nd_w = w * n_ref(d);
        for (int j = 0; j < nd; ++j) {
          r_mat(j, d) += nd_w * shape(j);
        }
      }
    }
  }

  void AssembleFaceGrad(const mfem::FiniteElement& el1,
                        const mfem::FiniteElement& /*el2*/,
                        mfem::FaceElementTransformations& Tr,
                        const mfem::Vector& elfun,
                        mfem::DenseMatrix& elmat) override {
    // Dead-load: no u-dependence, zero Jacobian.
    const int dim = Tr.GetSpaceDim();
    const int nd = el1.GetDof();
    (void)elfun;
    elmat.SetSize(nd * dim);
    elmat = 0.0;
  }

 private:
  const double& p_kpa_;
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

  // Pericardial normal Robin spring on the epicardium.
  if (cfg_.mech_peri_spring_k_kpa_per_mm > 0.0 &&
      cfg_.mech_bdr_epi_attr >= 1 && pmesh->bdr_attributes.Size() > 0) {
    const int max_attr = pmesh->bdr_attributes.Max();
    if (cfg_.mech_bdr_epi_attr <= max_attr) {
      epi_marker_.SetSize(max_attr);
      epi_marker_ = 0;
      epi_marker_[cfg_.mech_bdr_epi_attr - 1] = 1;
      nlform_->AddBdrFaceIntegrator(
          new NormalSpringBdrNLFI(cfg_.mech_peri_spring_k_kpa_per_mm),
          epi_marker_);
    }
  }
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
  // Stresses elsewhere are in kPa (Holzapfel-Ogden parameters are kPa);
  // convert once here so the dead-load integrator can use the same units.
  endo_pressure_kpa_ref_ = p_pa * 1.0e-3;

  if (endo_pressure_integrator_added_) return;
  if (cfg_.mech_bdr_endo_attr < 1) return;
  mfem::ParMesh* pmesh = ep_pfes_.GetParMesh();
  if (pmesh->bdr_attributes.Size() == 0) return;
  const int max_attr = pmesh->bdr_attributes.Max();
  if (cfg_.mech_bdr_endo_attr > max_attr) return;
  endo_marker_.SetSize(max_attr);
  endo_marker_ = 0;
  endo_marker_[cfg_.mech_bdr_endo_attr - 1] = 1;
  nlform_->AddBdrFaceIntegrator(
      new EndoPressureDeadLoadBdrNLFI(endo_pressure_kpa_ref_),
      endo_marker_);
  endo_pressure_integrator_added_ = true;
}

void MechanicsSolver::EnsureSolver() {
  if (newton_) return;
  // MFEM Newton + HypreGMRES + BoomerAMG. PetscNonlinearSolver-wrapped SNES
  // is currently disabled because in this build/topology it exits after
  // iteration 0 without iterating; the MFEM Newton path converges robustly
  // in ~15 iters and is used regardless of the global use_petsc switch.
  auto* amg = new mfem::HypreBoomerAMG();
  amg->SetPrintLevel(0);
  amg->SetSystemsOptions(ep_pfes_.GetParMesh()->Dimension());
  prec_.reset(amg);

  auto* gmres = new mfem::HypreGMRES(comm_);
  gmres->SetMaxIter(cfg_.mech_ksp_max_it);
  gmres->SetTol(cfg_.mech_ksp_rtol);
  gmres->SetKDim(60);
  gmres->SetPreconditioner(*amg);
  ksp_.reset(gmres);

  auto* nw = new mfem::NewtonSolver(comm_);
  nw->SetOperator(*nlform_);
  nw->SetSolver(*ksp_);
  nw->SetMaxIter(cfg_.mech_snes_max_it);
  nw->SetRelTol(cfg_.mech_snes_rtol);
  nw->SetAbsTol(cfg_.mech_snes_atol);
  nw->SetPrintLevel(cfg_.mech_snes_print_level);
  newton_.reset(nw);
}

int MechanicsSolver::Solve() {
  EnsureSolver();

  mfem::Vector x_true(vec_fes_->GetTrueVSize());
  u_->GetTrueDofs(x_true);

  // Apply zero on essential dofs as starting condition (clamped base).
  for (int i = 0; i < ess_tdofs_.Size(); ++i) x_true[ess_tdofs_[i]] = 0.0;

  // Optional endo-pressure ramping: subdivide (applied -> target) into N
  // increments and run Newton once per increment. The integrator picks up
  // the new value automatically through endo_pressure_kpa_ref_'s const ref.
  const int ramp_n = std::max(cfg_.mech_endo_pressure_ramp_steps, 1);
  const double p_target_kpa = endo_pressure_pa_ * 1.0e-3;
  const double p_start_kpa = endo_pressure_kpa_applied_;
  const bool needs_ramp =
      (ramp_n > 1) && (std::abs(p_target_kpa - p_start_kpa) > 1e-9);

  mfem::Vector zero;  // empty rhs => Newton solves R(u) = 0
  int total_iters = 0;
  if (needs_ramp) {
    for (int k = 1; k <= ramp_n; ++k) {
      const double frac = static_cast<double>(k) / ramp_n;
      endo_pressure_kpa_ref_ = p_start_kpa + (p_target_kpa - p_start_kpa) * frac;
      newton_->Mult(zero, x_true);
      if (auto* nw = dynamic_cast<mfem::IterativeSolver*>(newton_.get())) {
        total_iters += nw->GetNumIterations();
      }
    }
  } else {
    endo_pressure_kpa_ref_ = p_target_kpa;
    newton_->Mult(zero, x_true);
  }
  endo_pressure_kpa_applied_ = p_target_kpa;

  u_->SetFromTrueDofs(x_true);

  // Report real iteration count from the underlying solver.
#ifdef MFEM_USE_PETSC
  if (auto* sn = dynamic_cast<mfem::PetscNonlinearSolver*>(newton_.get())) {
    return sn->GetNumIterations();
  }
#endif
  if (needs_ramp) return total_iters;
  if (auto* nw = dynamic_cast<mfem::IterativeSolver*>(newton_.get())) {
    return nw->GetNumIterations();
  }
  return -1;
}

void MechanicsSolver::ComputeFiberStretch(mfem::ParGridFunction& lambda_gf) const {
  lambda_gf.SetSpace(&ep_pfes_);
  FiberStretchCoefficient stretch(*u_, f0_coeff_);
  lambda_gf.ProjectCoefficient(stretch);
}

void MechanicsSolver::ComputeJacobianDet(mfem::ParGridFunction& J_gf) const {
  J_gf.SetSpace(&ep_pfes_);
  JacobianDetCoefficient jdet(*u_);
  J_gf.ProjectCoefficient(jdet);
}

}  // namespace mono
