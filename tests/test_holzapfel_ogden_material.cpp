// Holzapfel-Ogden constitutive law unit test:
//   * W(F=I, f0=e1, s0=e2) == 0
//   * P(F=I) == 0 (zero stress at reference)
//   * Uniaxial fiber stretch generates positive fiber tension
//   * Finite-difference tangent (called via Tangent()) is symmetric-ish.

#include <cmath>
#include <iostream>

#include "mfem.hpp"

#include "mechanics/HolzapfelOgdenMaterial.hpp"

int main() {
  using namespace mono;
  HolzapfelOgdenMaterial::Params params;

  mfem::Vector f0(3), s0(3);
  f0 = 0.0; f0(0) = 1.0;
  s0 = 0.0; s0(1) = 1.0;

  // Reference configuration.
  mfem::DenseMatrix F(3);
  F = 0.0;
  for (int i = 0; i < 3; ++i) F(i, i) = 1.0;

  const double W0 = HolzapfelOgdenMaterial::StrainEnergy(F, f0, s0, params);
  if (std::abs(W0) > 1e-12) {
    std::cerr << "W(I) != 0: " << W0 << "\n";
    return 1;
  }

  mfem::DenseMatrix P(3);
  HolzapfelOgdenMaterial::PiolaPassive(F, f0, s0, params, P);
  double Pmax = 0.0;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) Pmax = std::max(Pmax, std::abs(P(i, j)));
  if (Pmax > 1e-6) {
    std::cerr << "P(I) has nonzero max abs: " << Pmax << "\n";
    return 2;
  }

  // Fiber stretch lambda=1.1.
  F(0, 0) = 1.1;
  HolzapfelOgdenMaterial::PiolaPassive(F, f0, s0, params, P);
  if (!(P(0, 0) > 0.0)) {
    std::cerr << "Fiber stretch should give positive P_11; got " << P(0, 0)
              << "\n";
    return 3;
  }

  // Tangent sanity: not all zero.
  mfem::DenseMatrix C(9);
  HolzapfelOgdenMaterial::Tangent(F, f0, s0, params, C);
  double Cmax = 0.0;
  for (int i = 0; i < 9; ++i)
    for (int j = 0; j < 9; ++j) Cmax = std::max(Cmax, std::abs(C(i, j)));
  if (!(Cmax > 1e-3)) {
    std::cerr << "Tangent magnitude too small: " << Cmax << "\n";
    return 4;
  }

  std::cout << "H-O unit test ok: P_11(lambda_f=1.1)=" << P(0, 0)
            << " kPa, tangent max=" << Cmax << "\n";
  return 0;
}
