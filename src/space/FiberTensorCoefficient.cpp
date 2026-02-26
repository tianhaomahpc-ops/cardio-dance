#include "space/FiberTensorCoefficient.hpp"

#include <cmath>

namespace mono {

FiberTensorCoefficient::FiberTensorCoefficient(int dim,
                                               double sigma_f,
                                               double sigma_s,
                                               double sigma_n,
                                               mfem::VectorCoefficient& f_coeff,
                                               mfem::VectorCoefficient& s_coeff,
                                               mfem::VectorCoefficient& n_coeff)
    : mfem::MatrixCoefficient(dim),
      sigma_f_(sigma_f),
      sigma_s_(sigma_s),
      sigma_n_(sigma_n),
      f_coeff_(f_coeff),
      s_coeff_(s_coeff),
      n_coeff_(n_coeff) {}

void FiberTensorCoefficient::Normalize(mfem::Vector& v, const mfem::Vector& fallback) {
  const double n = v.Norml2();
  if (n > 1e-14) {
    v /= n;
  } else {
    v = fallback;
  }
}

void FiberTensorCoefficient::Eval(mfem::DenseMatrix& K,
                                  mfem::ElementTransformation& T,
                                  const mfem::IntegrationPoint& ip) {
  K.SetSize(height, width);
  K = 0.0;

  mfem::Vector f(height), s(height), n(height);
  mfem::Vector f_fb(height), s_fb(height), n_fb(height);
  f_fb = 0.0;
  s_fb = 0.0;
  n_fb = 0.0;
  if (height > 0) f_fb[0] = 1.0;
  if (height > 1) s_fb[1] = 1.0;
  if (height > 2) n_fb[2] = 1.0;

  f_coeff_.Eval(f, T, ip);
  s_coeff_.Eval(s, T, ip);
  n_coeff_.Eval(n, T, ip);

  // Normalize at quadrature point to protect against non-unit input fields.
  Normalize(f, f_fb);
  Normalize(s, s_fb);
  Normalize(n, n_fb);

  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      K(i, j) = sigma_f_ * f[i] * f[j] + sigma_s_ * s[i] * s[j] + sigma_n_ * n[i] * n[j];
    }
  }
}

}  // namespace mono
