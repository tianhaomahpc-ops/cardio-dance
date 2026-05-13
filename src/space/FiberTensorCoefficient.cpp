#include "space/FiberTensorCoefficient.hpp"

#include <cmath>

namespace mono {

FiberTensorCoefficient::FiberTensorCoefficient(int dim,
                                               double sigma_f,
                                               double sigma_s,
                                               double sigma_n,
                                               mfem::VectorCoefficient& f_coeff,
                                               mfem::VectorCoefficient& s_coeff,
                                               mfem::VectorCoefficient& n_coeff,
                                               const std::vector<int>& fibrosis_attrs,
                                               double fibrosis_scale,
                                               const std::vector<int>& av_delay_attrs,
                                               double av_delay_scale)
    : mfem::MatrixCoefficient(dim),
      sigma_f_(sigma_f),
      sigma_s_(sigma_s),
      sigma_n_(sigma_n),
      f_coeff_(f_coeff),
      s_coeff_(s_coeff),
      n_coeff_(n_coeff),
      fibrosis_attrs_(fibrosis_attrs.begin(), fibrosis_attrs.end()),
      fibrosis_scale_(fibrosis_scale),
      av_delay_attrs_(av_delay_attrs.begin(), av_delay_attrs.end()),
      av_delay_scale_(av_delay_scale) {}

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

  double scale = 1.0;
  if (!fibrosis_attrs_.empty() && T.Attribute > 0 &&
      fibrosis_attrs_.find(T.Attribute) != fibrosis_attrs_.end()) {
    scale = fibrosis_scale_;
  } else if (!av_delay_attrs_.empty() && T.Attribute > 0 &&
             av_delay_attrs_.find(T.Attribute) != av_delay_attrs_.end()) {
    scale = av_delay_scale_;
  }
  const double sigma_f = sigma_f_ * scale;
  const double sigma_s = sigma_s_ * scale;
  const double sigma_n = sigma_n_ * scale;

  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      K(i, j) = sigma_f * f[i] * f[j] + sigma_s * s[i] * s[j] + sigma_n * n[i] * n[j];
    }
  }
}

}  // namespace mono
