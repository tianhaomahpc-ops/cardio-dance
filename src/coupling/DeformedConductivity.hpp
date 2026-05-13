#pragma once

#include <memory>

#include "mfem.hpp"

#include "space/FiberTensorCoefficient.hpp"

namespace mono {

// Pull-back of the spatial fiber-aligned conductivity tensor to the
// reference configuration through the deformation gradient F = I + grad u.
//
//   D_ref = J * F^{-1} * D_spatial * F^{-T}
//
// where D_spatial is provided by a base FiberTensorCoefficient evaluated at
// the same integration point. When 'enabled' is false (no deformation yet),
// returns D_spatial unchanged.
class DeformedConductivity : public mfem::MatrixCoefficient {
 public:
  DeformedConductivity(int dim,
                       FiberTensorCoefficient& base,
                       const mfem::ParGridFunction* displacement = nullptr);

  // Re-bind to displacement field. nullptr disables pull-back (identity F).
  void SetDisplacement(const mfem::ParGridFunction* u_gf);

  void Eval(mfem::DenseMatrix& K,
            mfem::ElementTransformation& T,
            const mfem::IntegrationPoint& ip) override;

 private:
  FiberTensorCoefficient& base_;
  const mfem::ParGridFunction* u_gf_;
};

}  // namespace mono
