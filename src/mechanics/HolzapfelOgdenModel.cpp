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

double HolzapfelOgdenModel::EvalW(const mfem::DenseMatrix& F) const {
  mfem::Vector f0, s0;
  SampleFibers(f0, s0);
  return HolzapfelOgdenMaterial::StrainEnergy(F, f0, s0, params_);
}

void HolzapfelOgdenModel::EvalP(const mfem::DenseMatrix& F,
                                 mfem::DenseMatrix& P) const {
  mfem::Vector f0, s0;
  SampleFibers(f0, s0);
  HolzapfelOgdenMaterial::PiolaPassive(F, f0, s0, params_, P);
}

void HolzapfelOgdenModel::AssembleH(const mfem::DenseMatrix& F,
                                     const mfem::DenseMatrix& DS,
                                     const double weight,
                                     mfem::DenseMatrix& A) const {
  mfem::Vector f0, s0;
  SampleFibers(f0, s0);
  const int dim = F.Size();
  const int dof = DS.Height();

  mfem::DenseMatrix C(9);
  HolzapfelOgdenMaterial::Tangent(F, f0, s0, params_, C);

  // A((k_a)*dim + i, (k_b)*dim + j) += weight * sum_{m,n} DS(k_a,m) * dP_{i,m}/dF_{j,n} * DS(k_b,n).
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
          A(ka * dim + i, kb * dim + j) += weight * acc;
        }
      }
    }
  }
}

}  // namespace mono
