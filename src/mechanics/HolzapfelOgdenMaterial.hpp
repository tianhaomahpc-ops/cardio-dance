#pragma once

#include "mfem.hpp"

namespace mono {

// Anisotropic Holzapfel-Ogden hyperelastic material (Holzapfel & Ogden 2009).
//
// W = (a/2b)[exp(b(I1-3)) - 1]
//   + sum_{i in {f,s}} (a_i / 2 b_i) [exp(b_i <I4i-1>^2) - 1]
//   + (a_fs / 2 b_fs) [exp(b_fs I8fs^2) - 1]
//   + (kappa/2) (J - 1)^2                                 (volumetric)
//
// with I1 = tr(C), I4f = f0.C.f0, I4s = s0.C.s0, I8fs = f0.C.s0,
//      C = F^T F, J = det F.
// <x> = max(x, 0) prevents fiber compression contribution.
//
// All routines work in 3D (dim=3). For 2D (planar), I4s/I8fs are skipped.
class HolzapfelOgdenMaterial {
 public:
  struct Params {
    // Units: a* in kPa, b* dimensionless. Defaults from Holzapfel-Ogden 2009 Table 1
    // (transversely isotropic LV approximation).
    double a   = 0.059;   // kPa
    double b   = 8.023;
    double af  = 18.472;  // kPa
    double bf  = 16.026;
    double as  = 2.481;   // kPa
    double bs  = 11.120;
    double afs = 0.216;   // kPa
    double bfs = 11.436;
    double kappa = 1000.0;   // kPa, volumetric penalty (near-incompressible)
  };

  // Compute first Piola-Kirchhoff stress P_passive(F).
  //   F  : 3x3 deformation gradient
  //   f0 : reference fiber direction (unit)
  //   s0 : reference sheet direction (unit, orthogonal to f0)
  //   P  : (out) 3x3 first Piola stress, kPa
  static void PiolaPassive(const mfem::DenseMatrix& F,
                           const mfem::Vector& f0,
                           const mfem::Vector& s0,
                           const Params& p,
                           mfem::DenseMatrix& P);

  // Compute material tangent A = d P_passive / d F as a 9x9 matrix
  // (row index = (a-1)*3 + i, column index = (b-1)*3 + j; P_ai vs F_bj).
  // Uses central finite differences fallback if EnableAnalyticTangent() == false.
  static void Tangent(const mfem::DenseMatrix& F,
                      const mfem::Vector& f0,
                      const mfem::Vector& s0,
                      const Params& p,
                      mfem::DenseMatrix& A_99);

  // For debug / verification.
  static double StrainEnergy(const mfem::DenseMatrix& F,
                             const mfem::Vector& f0,
                             const mfem::Vector& s0,
                             const Params& p);

  static bool& EnableAnalyticTangent();  // default: true
};

}  // namespace mono
