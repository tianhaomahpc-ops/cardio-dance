#pragma once

#include <unordered_set>
#include <vector>

#include "mfem.hpp"

namespace mono {

// Conductivity tensor coefficient:
// D = sigma_f f f^T + sigma_s s s^T + sigma_n n n^T
// where f,s,n are local fiber-sheet-normal directions.
class FiberTensorCoefficient : public mfem::MatrixCoefficient {
 public:
  FiberTensorCoefficient(int dim,
                         double sigma_f,
                         double sigma_s,
                         double sigma_n,
                         mfem::VectorCoefficient& f_coeff,
                         mfem::VectorCoefficient& s_coeff,
                         mfem::VectorCoefficient& n_coeff,
                         const std::vector<int>& fibrosis_attrs = {},
                         double fibrosis_scale = 1.0,
                         const std::vector<int>& av_delay_attrs = {},
                         double av_delay_scale = 1.0);

  void Eval(mfem::DenseMatrix& K,
            mfem::ElementTransformation& T,
            const mfem::IntegrationPoint& ip) override;

 private:
  double sigma_f_;
  double sigma_s_;
  double sigma_n_;
  mfem::VectorCoefficient& f_coeff_;
  mfem::VectorCoefficient& s_coeff_;
  mfem::VectorCoefficient& n_coeff_;
  std::unordered_set<int> fibrosis_attrs_;
  double fibrosis_scale_ = 1.0;
  std::unordered_set<int> av_delay_attrs_;
  double av_delay_scale_ = 1.0;

  static void Normalize(mfem::Vector& v, const mfem::Vector& fallback);
};

}  // namespace mono
