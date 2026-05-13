#pragma once

#include <memory>

#include "mfem.hpp"

#include "mechanics/HolzapfelOgdenMaterial.hpp"

namespace mono {

// MFEM HyperelasticModel adaptor that delegates to HolzapfelOgdenMaterial.
// Fiber/sheet directions are supplied as VectorCoefficients on the
// reference configuration; sampled at each integration point.
class HolzapfelOgdenModel : public mfem::HyperelasticModel {
 public:
  HolzapfelOgdenModel(const HolzapfelOgdenMaterial::Params& params,
                      mfem::VectorCoefficient& f0_coeff,
                      mfem::VectorCoefficient& s0_coeff);

  // MFEM hyperelastic interface: F = J (deformation gradient).
  // Each call samples f0/s0 at this->Ttr->GetIntPoint(), which the
  // HyperelasticNLFIntegrator sets before invoking these methods.
  double EvalW(const mfem::DenseMatrix& F) const override;
  void EvalP(const mfem::DenseMatrix& F, mfem::DenseMatrix& P) const override;
  void AssembleH(const mfem::DenseMatrix& F,
                 const mfem::DenseMatrix& DS,
                 const double weight,
                 mfem::DenseMatrix& A) const override;

 private:
  // Sample f0 and s0 from coefficients at the current Ttr-bound IP.
  // Returns false if Ttr is unset (defensive).
  bool SampleFibers(mfem::Vector& f0, mfem::Vector& s0) const;

  HolzapfelOgdenMaterial::Params params_;
  mfem::VectorCoefficient& f0_coeff_;
  mfem::VectorCoefficient& s0_coeff_;
};

}  // namespace mono
