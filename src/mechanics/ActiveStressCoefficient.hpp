#pragma once

#include "mfem.hpp"

namespace mono {

// Active first-Piola contribution along the reference fiber direction:
//   P_active = T_a(x) * (F f0) (X) f0
//
// T_a is a scalar ParGridFunction (kPa) supplied externally (Land 2017 output).
// f0 is provided as a VectorCoefficient (reference fiber direction).
class ActiveStressCoefficient : public mfem::MatrixCoefficient {
 public:
  ActiveStressCoefficient(int dim,
                          const mfem::ParGridFunction& ta_kPa,
                          mfem::VectorCoefficient& f0_coeff,
                          const mfem::ParGridFunction& displacement);

  // Eval returns P_active at integration point in the matrix K (3x3).
  // The deformation gradient F is computed from displacement gradient.
  void Eval(mfem::DenseMatrix& K,
            mfem::ElementTransformation& T,
            const mfem::IntegrationPoint& ip) override;

 private:
  const mfem::ParGridFunction& ta_gf_;
  mfem::VectorCoefficient& f0_coeff_;
  const mfem::ParGridFunction& u_gf_;
  mfem::GridFunctionCoefficient ta_coeff_;
};

}  // namespace mono
