#include "mechanics/HolzapfelOgdenMaterial.hpp"

#include <algorithm>
#include <cmath>

namespace mono {

namespace {

inline double Macaulay(double x) { return x > 0.0 ? x : 0.0; }

inline void Outer(const mfem::Vector& a, const mfem::Vector& b,
                  mfem::DenseMatrix& M) {
  const int n = a.Size();
  M.SetSize(n);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) M(i, j) = a(i) * b(j);
}

inline double Det3(const mfem::DenseMatrix& F) {
  return F(0,0)*(F(1,1)*F(2,2)-F(1,2)*F(2,1))
       - F(0,1)*(F(1,0)*F(2,2)-F(1,2)*F(2,0))
       + F(0,2)*(F(1,0)*F(2,1)-F(1,1)*F(2,0));
}

// 3x3 cofactor (transpose of inverse times det). cof = J F^{-T}.
inline void Cofactor3(const mfem::DenseMatrix& F, mfem::DenseMatrix& cof) {
  cof.SetSize(3);
  cof(0,0) =  F(1,1)*F(2,2) - F(1,2)*F(2,1);
  cof(0,1) = -(F(1,0)*F(2,2) - F(1,2)*F(2,0));
  cof(0,2) =  F(1,0)*F(2,1) - F(1,1)*F(2,0);
  cof(1,0) = -(F(0,1)*F(2,2) - F(0,2)*F(2,1));
  cof(1,1) =  F(0,0)*F(2,2) - F(0,2)*F(2,0);
  cof(1,2) = -(F(0,0)*F(2,1) - F(0,1)*F(2,0));
  cof(2,0) =  F(0,1)*F(1,2) - F(0,2)*F(1,1);
  cof(2,1) = -(F(0,0)*F(1,2) - F(0,2)*F(1,0));
  cof(2,2) =  F(0,0)*F(1,1) - F(0,1)*F(1,0);
}

inline double Inner(const mfem::Vector& a, const mfem::DenseMatrix& C,
                    const mfem::Vector& b) {
  const int n = a.Size();
  double s = 0.0;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) s += a(i) * C(i, j) * b(j);
  return s;
}

}  // namespace

bool& HolzapfelOgdenMaterial::EnableAnalyticTangent() {
  static bool flag = true;
  return flag;
}

double HolzapfelOgdenMaterial::StrainEnergy(const mfem::DenseMatrix& F,
                                            const mfem::Vector& f0,
                                            const mfem::Vector& s0,
                                            const Params& p) {
  // C = F^T F
  mfem::DenseMatrix C(3);
  mfem::MultAtB(F, F, C);
  const double I1 = C(0,0) + C(1,1) + C(2,2);
  const double J  = Det3(F);
  const double I4f = Inner(f0, C, f0);
  const double I4s = Inner(s0, C, s0);
  const double I8fs = Inner(f0, C, s0);
  // Isochoric split for the ground matrix: bar(I1) = J^{-2/3} I1.
  // At F=I, bar(I1)=3 and dW/dF = 0, so the reference is stress-free.
  const double Jp = std::pow(std::max(J, 1e-12), -2.0 / 3.0);
  const double barI1 = Jp * I1;

  const double Wiso = (p.a / (2.0 * p.b)) * (std::exp(p.b * (barI1 - 3.0)) - 1.0);
  const double df = Macaulay(I4f - 1.0);
  const double ds = Macaulay(I4s - 1.0);
  const double Wf = (p.af / (2.0 * p.bf)) * (std::exp(p.bf * df * df) - 1.0);
  const double Ws = (p.as / (2.0 * p.bs)) * (std::exp(p.bs * ds * ds) - 1.0);
  const double Wfs = (p.afs / (2.0 * p.bfs)) * (std::exp(p.bfs * I8fs * I8fs) - 1.0);
  const double Wvol = 0.5 * p.kappa * (J - 1.0) * (J - 1.0);
  return Wiso + Wf + Ws + Wfs + Wvol;
}

void HolzapfelOgdenMaterial::PiolaPassive(const mfem::DenseMatrix& F,
                                          const mfem::Vector& f0,
                                          const mfem::Vector& s0,
                                          const Params& p,
                                          mfem::DenseMatrix& P) {
  P.SetSize(3);
  P = 0.0;

  // Invariants.
  mfem::DenseMatrix C(3);
  mfem::MultAtB(F, F, C);
  const double I1 = C(0,0) + C(1,1) + C(2,2);
  const double J  = Det3(F);
  const double I4f = Inner(f0, C, f0);
  const double I4s = Inner(s0, C, s0);
  const double I8fs = Inner(f0, C, s0);
  // Isochoric I1 (Flory split): bar(I1) = J^{-2/3} I1.
  const double Jp = std::pow(std::max(J, 1e-12), -2.0 / 3.0);
  const double barI1 = Jp * I1;

  // Derivatives of W wrt invariants. The ground matrix term uses bar(I1) so
  // that P_iso vanishes at the reference configuration F = I.
  const double dW_dBarI1 = 0.5 * p.a * std::exp(p.b * (barI1 - 3.0));
  const double df = Macaulay(I4f - 1.0);
  const double ds = Macaulay(I4s - 1.0);
  const double dW_dI4f = p.af * df * std::exp(p.bf * df * df);
  const double dW_dI4s = p.as * ds * std::exp(p.bs * ds * ds);
  const double dW_dI8fs = p.afs * I8fs * std::exp(p.bfs * I8fs * I8fs);
  const double dU_dJ = p.kappa * (J - 1.0);

  // P = 2 dW/dBarI1 * J^{-2/3} (F - (I1/3) F^{-T})       [isochoric ground]
  //   + 2 dW/dI4f (F f0) o f0 + 2 dW/dI4s (F s0) o s0    [anisotropic fiber/sheet]
  //   + dW/dI8fs ((F f0) o s0 + (F s0) o f0)             [fiber-sheet coupling]
  //   + dU/dJ * cof(F)                                    [volumetric]

  // F * f0, F * s0.
  mfem::Vector Ff0(3), Fs0(3);
  F.Mult(f0, Ff0);
  F.Mult(s0, Fs0);

  // cof(F) = J * F^{-T}; reused for both volumetric and isochoric ground.
  mfem::DenseMatrix cof(3);
  Cofactor3(F, cof);
  const double inv_J = (std::abs(J) > 1e-12) ? 1.0 / J : 0.0;

  // Isochoric ground: 2 dW/dBarI1 * J^{-2/3} (F - (I1/3) F^{-T}).
  // F^{-T} = cof / J.
  const double iso_coef = 2.0 * dW_dBarI1 * Jp;
  const double iso_devc = I1 / 3.0;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      P(i, j) += iso_coef * (F(i, j) - iso_devc * cof(i, j) * inv_J);

  // Fiber: 2 dW/dI4f (Ff0) o f0.
  if (df > 0.0) {
    mfem::DenseMatrix T(3);
    Outer(Ff0, f0, T);
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) P(i, j) += 2.0 * dW_dI4f * T(i, j);
  }

  // Sheet: 2 dW/dI4s (Fs0) o s0.
  if (ds > 0.0) {
    mfem::DenseMatrix T(3);
    Outer(Fs0, s0, T);
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) P(i, j) += 2.0 * dW_dI4s * T(i, j);
  }

  // FS coupling: dW/dI8fs * ((Ff0) o s0 + (Fs0) o f0).
  {
    mfem::DenseMatrix T1(3), T2(3);
    Outer(Ff0, s0, T1);
    Outer(Fs0, f0, T2);
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) P(i, j) += dW_dI8fs * (T1(i, j) + T2(i, j));
  }

  // Volumetric: dU/dJ * cof(F).
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) P(i, j) += dU_dJ * cof(i, j);
}

void HolzapfelOgdenMaterial::Tangent(const mfem::DenseMatrix& F,
                                      const mfem::Vector& f0,
                                      const mfem::Vector& s0,
                                      const Params& p,
                                      mfem::DenseMatrix& A_99) {
  // Finite-difference tangent for now; an analytic version can be added later.
  // dP_{ai} / dF_{bj}, indices flattened as row=(a)*3+i, col=(b)*3+j.
  A_99.SetSize(9);
  A_99 = 0.0;
  const double eps = 1e-7;
  mfem::DenseMatrix Fp(F), Fm(F), Pp(3), Pm(3);
  for (int b = 0; b < 3; ++b) {
    for (int j = 0; j < 3; ++j) {
      Fp = F;
      Fm = F;
      Fp(b, j) += eps;
      Fm(b, j) -= eps;
      PiolaPassive(Fp, f0, s0, p, Pp);
      PiolaPassive(Fm, f0, s0, p, Pm);
      const int col = b * 3 + j;
      for (int a = 0; a < 3; ++a) {
        for (int i = 0; i < 3; ++i) {
          const int row = a * 3 + i;
          A_99(row, col) = (Pp(a, i) - Pm(a, i)) / (2.0 * eps);
        }
      }
    }
  }
}

}  // namespace mono
