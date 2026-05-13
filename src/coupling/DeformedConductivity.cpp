#include "coupling/DeformedConductivity.hpp"

#include <cmath>

namespace mono {

namespace {

inline double Det3(const mfem::DenseMatrix& F) {
  return F(0,0)*(F(1,1)*F(2,2)-F(1,2)*F(2,1))
       - F(0,1)*(F(1,0)*F(2,2)-F(1,2)*F(2,0))
       + F(0,2)*(F(1,0)*F(2,1)-F(1,1)*F(2,0));
}

}  // namespace

DeformedConductivity::DeformedConductivity(int dim,
                                            FiberTensorCoefficient& base,
                                            const mfem::ParGridFunction* displacement)
    : mfem::MatrixCoefficient(dim), base_(base), u_gf_(displacement) {}

void DeformedConductivity::SetDisplacement(const mfem::ParGridFunction* u_gf) {
  u_gf_ = u_gf;
}

void DeformedConductivity::Eval(mfem::DenseMatrix& K,
                                 mfem::ElementTransformation& T,
                                 const mfem::IntegrationPoint& ip) {
  const int dim = GetWidth();
  mfem::DenseMatrix D_spatial(dim);
  base_.Eval(D_spatial, T, ip);

  if (u_gf_ == nullptr) {
    K = D_spatial;
    return;
  }

  // F = I + grad u  (in reference configuration).
  T.SetIntPoint(&ip);
  mfem::DenseMatrix grad_u(dim);
  u_gf_->GetVectorGradient(T, grad_u);
  mfem::DenseMatrix F(dim);
  F = grad_u;
  for (int i = 0; i < dim; ++i) F(i, i) += 1.0;

  // F^{-1}.
  mfem::DenseMatrix Finv(dim);
  Finv = F;
  Finv.Invert();
  const double J = (dim == 3) ? Det3(F) : F.Det();
  if (!(J > 0.0) || !std::isfinite(J)) {
    K = D_spatial;  // fallback if F is degenerate
    return;
  }

  // K_ref = J * F^{-1} * D_spatial * F^{-T}.
  mfem::DenseMatrix tmp(dim);
  mfem::Mult(Finv, D_spatial, tmp);
  mfem::DenseMatrix Finv_T(dim);
  Finv_T.Transpose(Finv);
  mfem::Mult(tmp, Finv_T, K);
  K *= J;
}

}  // namespace mono
