#include "mechanics/ActiveStressCoefficient.hpp"

namespace mono {

ActiveStressCoefficient::ActiveStressCoefficient(
    int dim,
    const mfem::ParGridFunction& ta_kPa,
    mfem::VectorCoefficient& f0_coeff,
    const mfem::ParGridFunction& displacement)
    : mfem::MatrixCoefficient(dim),
      ta_gf_(ta_kPa),
      f0_coeff_(f0_coeff),
      u_gf_(displacement),
      ta_coeff_(&ta_kPa) {}

void ActiveStressCoefficient::Eval(mfem::DenseMatrix& K,
                                    mfem::ElementTransformation& T,
                                    const mfem::IntegrationPoint& ip) {
  const int dim = GetWidth();
  K.SetSize(dim);
  K = 0.0;

  // Evaluate active tension scalar at integration point.
  T.SetIntPoint(&ip);
  const double Ta = ta_coeff_.Eval(T, ip);
  if (Ta <= 0.0) return;

  // Fiber direction f0.
  mfem::Vector f0(dim);
  f0_coeff_.Eval(f0, T, ip);
  const double n2 = f0.Norml2();
  if (n2 < 1e-12) return;
  f0 /= n2;

  // Displacement gradient at the IP, then F = I + grad u.
  mfem::DenseMatrix grad_u(dim);
  u_gf_.GetVectorGradient(T, grad_u);
  mfem::DenseMatrix F(dim);
  F = grad_u;
  for (int i = 0; i < dim; ++i) F(i, i) += 1.0;

  // P_active = Ta * (F f0) (X) f0.
  mfem::Vector Ff0(dim);
  F.Mult(f0, Ff0);
  for (int i = 0; i < dim; ++i)
    for (int j = 0; j < dim; ++j)
      K(i, j) = Ta * Ff0(i) * f0(j);
}

}  // namespace mono
