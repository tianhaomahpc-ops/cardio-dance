#include "mechanics/HolzapfelOgdenModel.hpp"

namespace mono {

HolzapfelOgdenModel::HolzapfelOgdenModel(const HolzapfelOgdenMaterial::Params& params,
                                          mfem::VectorCoefficient& f0_coeff,
                                          mfem::VectorCoefficient& s0_coeff)
    : params_(params),
      f0_coeff_(f0_coeff),
      s0_coeff_(s0_coeff) {}

bool HolzapfelOgdenModel::SampleFibers(mfem::Vector& f0, mfem::Vector& s0) const {
  if (!this->Ttr) {
    // Defensive fallback: aligned axes.
    f0.SetSize(3); f0 = 0.0; f0(0) = 1.0;
    s0.SetSize(3); s0 = 0.0; s0(1) = 1.0;
    return false;
  }
  const mfem::IntegrationPoint& ip = this->Ttr->GetIntPoint();
  f0.SetSize(3);
  s0.SetSize(3);
  f0_coeff_.Eval(f0, *this->Ttr, ip);
  s0_coeff_.Eval(s0, *this->Ttr, ip);
  const double nf = f0.Norml2();
  const double ns = s0.Norml2();
  if (nf > 1e-12) f0 /= nf; else { f0 = 0.0; f0(0) = 1.0; }
  if (ns > 1e-12) s0 /= ns; else { s0 = 0.0; s0(1) = 1.0; }
  return true;
}

namespace {
// MFEM's HyperelasticNLFIntegrator passes Jpt = grad_X u (the physical-space
// displacement gradient), NOT the deformation gradient F. Add the identity
// before delegating to HolzapfelOgdenMaterial which expects F.
inline mfem::DenseMatrix MakeF(const mfem::DenseMatrix& Jpt) {
  mfem::DenseMatrix F(Jpt);
  const int dim = F.Size();
  for (int k = 0; k < dim; ++k) F(k, k) += 1.0;
  return F;
}
}  // namespace

double HolzapfelOgdenModel::EvalW(const mfem::DenseMatrix& Jpt) const {
  mfem::Vector f0, s0;
  SampleFibers(f0, s0);
  const mfem::DenseMatrix F = MakeF(Jpt);
  return HolzapfelOgdenMaterial::StrainEnergy(F, f0, s0, params_);
}

void HolzapfelOgdenModel::EvalP(const mfem::DenseMatrix& Jpt,
                                 mfem::DenseMatrix& P) const {
  mfem::Vector f0, s0;
  SampleFibers(f0, s0);
  const mfem::DenseMatrix F = MakeF(Jpt);
  HolzapfelOgdenMaterial::PiolaPassive(F, f0, s0, params_, P);
}

void HolzapfelOgdenModel::AssembleH(const mfem::DenseMatrix& Jpt,
                                     const mfem::DenseMatrix& DS,
                                     const double weight,
                                     mfem::DenseMatrix& A) const {
  mfem::Vector f0, s0;
  SampleFibers(f0, s0);
  const int dim = Jpt.Size();
  const int dof = DS.Height();
  const mfem::DenseMatrix F = MakeF(Jpt);

  mfem::DenseMatrix C(9);
  HolzapfelOgdenMaterial::Tangent(F, f0, s0, params_, C);

  // vec_fes_ uses byNODES, so the element vector is laid out as
  // [u_x_0..u_x_{dof-1}, u_y_0..u_y_{dof-1}, u_z_*]. The matching elmat row
  // for (component i, node ka) is row = i*dof + ka.
  // contrib(i*dof+ka, j*dof+kb) += weight * sum_{m,n} DS(ka,m) * dP_{i,m}/dF_{j,n} * DS(kb,n).
  for (int ka = 0; ka < dof; ++ka) {
    for (int kb = 0; kb < dof; ++kb) {
      for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
          double acc = 0.0;
          for (int m = 0; m < dim; ++m) {
            for (int n = 0; n < dim; ++n) {
              acc += DS(ka, m) * C(i * 3 + m, j * 3 + n) * DS(kb, n);
            }
          }
          A(i * dof + ka, j * dof + kb) += weight * acc;
        }
      }
    }
  }
}

}  // namespace mono
