#include "ode/StewartPurkinjeModel.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <istream>
#include <ostream>
#include <stdexcept>

namespace mono {

// Stewart-Aslanidi-Boyett-Zhang 2009 Purkinje cell. Constants taken from the
// CellML repository (units: mV, ms, mM, uA/uF, mS/uF, mm^-3 etc.). Parameters
// follow the published model so values flow naturally through the equations.
struct StewartPurkinjeModel::Constants {
  // Cellular geometry / capacitance scaling.
  double R = 8314.472;
  double T = 310.0;
  double F = 96485.3415;
  double Cm_pF = 0.185;       // pF (used only for unit reconciliation)
  double V_c = 0.016404;      // uL
  double V_sr = 0.001094;
  double V_ss = 0.00005468;

  // External concentrations (mM).
  double Ko = 5.4;
  double Nao = 140.0;
  double Cao = 2.0;

  // Maximal conductances (mS/uF).
  double g_Na = 130.5744;
  double g_CaL = 3.98e-5 * 1.0;  // L/(F*ms) -> rescaled inside
  double g_to = 0.08184;
  double g_Kr = 0.0918;
  double g_Ks = 0.2352;
  double g_K1 = 0.065;
  double g_f_Na = 0.0145654;
  double g_f_K = 0.0234346;
  double g_pCa = 0.1238;
  double g_pK = 0.0146;
  double g_bNa = 0.00029;
  double g_bCa = 0.000592;
  double g_sus = 0.0227;       // sustained outward, Stewart-specific

  // NCX, NaK, pumps.
  double K_NaCa = 1000.0;
  double K_sat = 0.1;
  double alpha = 2.5;
  double gamma = 0.35;
  double Km_Nai = 87.5;
  double Km_Ca = 1.38;
  double P_NaK = 2.724;
  double Km_K = 1.0;
  double Km_Na = 40.0;
  double K_pCa = 0.0005;

  // SR uptake / release.
  double V_max_up = 0.006375;
  double K_up = 0.00025;
  double V_rel = 0.102;
  double k1_prime = 0.15;
  double k2_prime = 0.045;
  double k3 = 0.06;
  double k4 = 0.005;
  double EC50_SR = 1.5;
  double max_sr = 2.5;
  double min_sr = 1.0;
  double V_leak = 0.00036;
  double V_xfer = 0.0038;

  // Buffers.
  double Buf_c = 0.2;
  double K_buf_c = 0.001;
  double Buf_sr = 10.0;
  double K_buf_sr = 0.3;
  double Buf_ss = 0.4;
  double K_buf_ss = 0.00025;
};

const StewartPurkinjeModel::Constants& StewartPurkinjeModel::Params() {
  static const Constants c;
  return c;
}

namespace {

inline double SafeExp(double x) {
  if (x > 50.0) return std::exp(50.0);
  if (x < -50.0) return std::exp(-50.0);
  return std::exp(x);
}

inline double RushLarsen(double x, double x_inf, double tau, double dt) {
  if (tau <= 1e-12 || !std::isfinite(tau)) return x_inf;
  return x_inf - (x_inf - x) * std::exp(-dt / tau);
}

}  // namespace

StewartPurkinjeModel::StewartPurkinjeModel(int n_local_true_dofs)
    : n_nodes_(n_local_true_dofs),
      states_(static_cast<size_t>(n_local_true_dofs) * kNumStates, 0.0) {
  if (n_nodes_ <= 0) {
    throw std::runtime_error("StewartPurkinjeModel requires n_local_true_dofs > 0");
  }
}

void StewartPurkinjeModel::InitializeRestState(double v_rest_mv) {
  // Stewart 2009 published rest values; if v_rest_mv is finite, override Vm.
  const double V0 = std::isfinite(v_rest_mv) ? v_rest_mv : -85.23;
  for (int n = 0; n < n_nodes_; ++n) {
    double* s = &states_[static_cast<size_t>(n) * kNumStates];
    s[0]  = V0;       // V
    s[1]  = 0.00132;  // m
    s[2]  = 0.7573;   // h
    s[3]  = 0.7225;   // j
    s[4]  = 0.00621;  // xr1
    s[5]  = 0.4712;   // xr2
    s[6]  = 0.0095;   // xs
    s[7]  = 2.42e-8;  // r
    s[8]  = 0.999998; // s
    s[9]  = 3.373e-5; // d
    s[10] = 0.7888;   // f
    s[11] = 0.9755;   // f2
    s[12] = 0.9953;   // fCass
    s[13] = 0.0457;   // y (HCN/I_f gate)
    s[14] = 0.000126; // Cai
    s[15] = 3.64;     // CaSR
    s[16] = 0.00036;  // CaSS
    s[17] = 8.604;    // Nai
    s[18] = 136.89;   // Ki
    s[19] = 0.9073;   // Rprime (RyR gating)
  }
}

namespace {

// Compute and return per-node I_ion plus optionally write d_state into dstate
// if non-null. Currents use the Stewart 2009 conventions: positive = outward.
double NodeCurrents(double V, const double* s, double* dstate) {
  const auto& c = StewartPurkinjeModel::Params();
  const double m   = s[1];
  const double h   = s[2];
  const double j   = s[3];
  const double xr1 = s[4];
  const double xr2 = s[5];
  const double xs  = s[6];
  const double r   = s[7];
  const double sg  = s[8];
  const double d   = s[9];
  const double f   = s[10];
  const double f2  = s[11];
  const double fCass = s[12];
  const double y   = s[13];
  const double Cai = s[14];
  const double CaSR = s[15];
  const double CaSS = s[16];
  const double Nai = s[17];
  const double Ki  = s[18];
  const double Rp  = s[19];

  const double RT_F = c.R * c.T / c.F;
  const double E_Na = RT_F * std::log(c.Nao / std::max(Nai, 1e-6));
  const double E_K  = RT_F * std::log(c.Ko  / std::max(Ki,  1e-6));
  const double E_Ks = RT_F * std::log((c.Ko + 0.03 * c.Nao) /
                                       std::max(Ki + 0.03 * Nai, 1e-6));
  const double E_Ca = 0.5 * RT_F * std::log(c.Cao / std::max(Cai, 1e-9));

  // I_Na (fast sodium)
  const double I_Na = c.g_Na * m * m * m * h * j * (V - E_Na);

  // I_to (transient outward) -- Purkinje variant
  const double I_to = c.g_to * r * sg * (V - E_K);

  // I_Kr (rapid delayed rectifier)
  const double I_Kr = c.g_Kr * std::sqrt(c.Ko / 5.4) * xr1 * xr2 * (V - E_K);

  // I_Ks (slow delayed rectifier)
  const double I_Ks = c.g_Ks * xs * xs * (V - E_Ks);

  // I_K1 (inward rectifier)
  const double alpha_K1 = 0.1 / (1.0 + SafeExp(0.06 * (V - E_K - 200.0)));
  const double beta_K1 = (3.0 * SafeExp(0.0002 * (V - E_K + 100.0)) +
                          SafeExp(0.1 * (V - E_K - 10.0))) /
                          (1.0 + SafeExp(-0.5 * (V - E_K)));
  const double xK1_inf = alpha_K1 / (alpha_K1 + beta_K1);
  const double I_K1 = c.g_K1 * std::sqrt(c.Ko / 5.4) * xK1_inf * (V - E_K);

  // I_CaL (L-type calcium): TT06/Stewart Goldman-Hodgkin-Katz form.
  //   I_CaL = g_CaL d f f2 fCass * 4 (V-15) F^2/RT *
  //           (0.25 CaSS e^{2(V-15)F/RT} - Cao) / (e^{2(V-15)F/RT} - 1)
  // Note: BOTH the (V-15) prefactor AND the exponent use the offset voltage;
  // a previous version of this file used exp(2VF/RT) in the denominator while
  // multiplying by (V-15), producing huge non-physiological CaL drive.
  const double VFRT = V * c.F / (c.R * c.T);
  const double Vshift = V - 15.0;
  const double VshiftFRT = Vshift * c.F / (c.R * c.T);
  const double exp2_Vshift_FRT = SafeExp(2.0 * VshiftFRT);
  double I_CaL = c.g_CaL * d * f * f2 * fCass * 4.0 * Vshift *
                 (c.F * c.F / (c.R * c.T));
  const double denom = exp2_Vshift_FRT - 1.0;
  if (std::fabs(denom) > 1e-9) {
    I_CaL *= (0.25 * CaSS * exp2_Vshift_FRT - c.Cao) / denom;
  } else {
    // Limit V -> 15 mV: GHK reduces to driving force * concentrations.
    I_CaL *= 0.25 * CaSS - c.Cao;
  }
  (void)VFRT;  // VFRT (without 15 mV offset) still used by I_NaCa/I_NaK below

  // I_NaCa (Na/Ca exchanger)
  const double e_g_VFRT = SafeExp(c.gamma * VFRT);
  const double e_g_m1_VFRT = SafeExp((c.gamma - 1.0) * VFRT);
  const double NaCa_num = c.K_NaCa *
      (e_g_VFRT * Nai * Nai * Nai * c.Cao -
       e_g_m1_VFRT * c.Nao * c.Nao * c.Nao * Cai * c.alpha);
  const double NaCa_den =
      (c.Km_Nai * c.Km_Nai * c.Km_Nai + c.Nao * c.Nao * c.Nao) *
      (c.Km_Ca + c.Cao) *
      (1.0 + c.K_sat * e_g_m1_VFRT);
  const double I_NaCa = NaCa_num / NaCa_den;

  // I_NaK (Na/K pump)
  const double I_NaK = c.P_NaK * (c.Ko / (c.Ko + c.Km_K)) *
                       (Nai / (Nai + c.Km_Na)) /
                       (1.0 + 0.1245 * SafeExp(-0.1 * VFRT) +
                        0.0353 * SafeExp(-VFRT));

  // I_pCa, I_pK
  const double I_pCa = c.g_pCa * Cai / (c.K_pCa + Cai);
  const double I_pK = c.g_pK * (V - E_K) /
                      (1.0 + SafeExp((25.0 - V) / 5.98));

  // I_bNa, I_bCa
  const double I_bNa = c.g_bNa * (V - E_Na);
  const double I_bCa = c.g_bCa * (V - E_Ca);

  // I_f (funny / HCN) -- Stewart-distinctive, splits Na/K components.
  const double I_f_Na = c.g_f_Na * y * (V - E_Na);
  const double I_f_K  = c.g_f_K  * y * (V - E_K);

  // I_sus (sustained outward)
  const double I_sus = c.g_sus * (V - E_K);

  const double I_ion = I_Na + I_to + I_Kr + I_Ks + I_K1 + I_CaL +
                       I_NaCa + I_NaK + I_pCa + I_pK + I_bNa + I_bCa +
                       I_f_Na + I_f_K + I_sus;

  if (dstate == nullptr) {
    return I_ion;
  }

  // ----- Gating steady-state and time-constants -----
  // m
  const double m_inf = 1.0 / std::pow(1.0 + SafeExp((-56.86 - V) / 9.03), 2.0);
  const double a_m = 1.0 / (1.0 + SafeExp((-60.0 - V) / 5.0));
  const double b_m = 0.1 / (1.0 + SafeExp((V + 35.0) / 5.0)) +
                     0.1 / (1.0 + SafeExp((V - 50.0) / 200.0));
  const double tau_m = a_m * b_m;

  // h
  const double h_inf = 1.0 / std::pow(1.0 + SafeExp((V + 71.55) / 7.43), 2.0);
  double a_h, b_h;
  if (V >= -40.0) {
    a_h = 0.0;
    b_h = 0.77 / (0.13 * (1.0 + SafeExp(-(V + 10.66) / 11.1)));
  } else {
    a_h = 0.057 * SafeExp(-(V + 80.0) / 6.8);
    b_h = 2.7 * SafeExp(0.079 * V) + 3.1e5 * SafeExp(0.3485 * V);
  }
  const double tau_h = 1.0 / (a_h + b_h);

  // j
  const double j_inf = h_inf;
  double a_j, b_j;
  if (V >= -40.0) {
    a_j = 0.0;
    b_j = 0.6 * SafeExp(0.057 * V) /
          (1.0 + SafeExp(-0.1 * (V + 32.0)));
  } else {
    a_j = (-2.5428e4 * SafeExp(0.2444 * V) -
           6.948e-6 * SafeExp(-0.04391 * V)) *
          (V + 37.78) /
          (1.0 + SafeExp(0.311 * (V + 79.23)));
    b_j = 0.02424 * SafeExp(-0.01052 * V) /
          (1.0 + SafeExp(-0.1378 * (V + 40.14)));
  }
  const double tau_j = 1.0 / (a_j + b_j);

  // xr1, xr2
  const double xr1_inf = 1.0 / (1.0 + SafeExp((-26.0 - V) / 7.0));
  const double a_xr1 = 450.0 / (1.0 + SafeExp((-45.0 - V) / 10.0));
  const double b_xr1 = 6.0 / (1.0 + SafeExp((V + 30.0) / 11.5));
  const double tau_xr1 = a_xr1 * b_xr1;

  const double xr2_inf = 1.0 / (1.0 + SafeExp((V + 88.0) / 24.0));
  const double a_xr2 = 3.0 / (1.0 + SafeExp((-60.0 - V) / 20.0));
  const double b_xr2 = 1.12 / (1.0 + SafeExp((V - 60.0) / 20.0));
  const double tau_xr2 = a_xr2 * b_xr2;

  // xs
  const double xs_inf = 1.0 / (1.0 + SafeExp((-5.0 - V) / 14.0));
  const double a_xs = 1400.0 / std::sqrt(1.0 + SafeExp((5.0 - V) / 6.0));
  const double b_xs = 1.0 / (1.0 + SafeExp((V - 35.0) / 15.0));
  const double tau_xs = a_xs * b_xs + 80.0;

  // r, s -- Stewart Purkinje variant
  const double r_inf = 1.0 / (1.0 + SafeExp((20.0 - V) / 13.0));
  const double tau_r = 10.45 * SafeExp(-std::pow((V + 40.0) / 25.0, 2.0)) + 7.3;
  const double s_inf = 1.0 / (1.0 + SafeExp((V + 27.0) / 13.0));
  const double tau_s = 85.0 * SafeExp(-std::pow((V + 25.0) / 32.0, 2.0)) +
                       5.0 / (1.0 + SafeExp((V - 40.0) / 5.0)) + 42.0;

  // d, f, f2, fCass
  const double d_inf = 1.0 / (1.0 + SafeExp((-8.0 - V) / 7.5));
  const double a_d = 1.4 / (1.0 + SafeExp((-35.0 - V) / 13.0)) + 0.25;
  const double b_d = 1.4 / (1.0 + SafeExp((V + 5.0) / 5.0));
  const double g_d = 1.0 / (1.0 + SafeExp((50.0 - V) / 20.0));
  const double tau_d = a_d * b_d + g_d;

  const double f_inf = 1.0 / (1.0 + SafeExp((V + 20.0) / 7.0));
  const double tau_f = 1102.5 * SafeExp(-std::pow((V + 27.0) / 15.0, 2.0)) +
                       200.0 / (1.0 + SafeExp((13.0 - V) / 10.0)) +
                       180.0 / (1.0 + SafeExp((V + 30.0) / 10.0)) + 20.0;

  const double f2_inf = 0.67 / (1.0 + SafeExp((V + 35.0) / 7.0)) + 0.33;
  const double tau_f2 = 562.0 * SafeExp(-std::pow((V + 27.0) / 240.0, 2.0)) +
                        31.0 / (1.0 + SafeExp((25.0 - V) / 10.0)) +
                        80.0 / (1.0 + SafeExp((V + 30.0) / 10.0));

  const double fCass_inf = 0.6 / (1.0 + std::pow(CaSS / 0.05, 2.0)) + 0.4;
  const double tau_fCass = 80.0 / (1.0 + std::pow(CaSS / 0.05, 2.0)) + 2.0;

  // y (I_f gate) -- Stewart 2009 formulation
  const double y_inf = 1.0 / (1.0 + SafeExp((V + 80.6) / 6.8));
  const double tau_y = 4000.0 / (SafeExp(-2.9 - 0.04 * V) +
                                 SafeExp(3.6 + 0.11 * V));

  // Concentration / RyR derivatives (Forward Euler)
  // Calcium release flux from SR (RyR-gated)
  const double kCaSR = c.max_sr - (c.max_sr - c.min_sr) /
                                    (1.0 + std::pow(c.EC50_SR / std::max(CaSR, 1e-12), 2.0));
  const double k1 = c.k1_prime / std::max(kCaSR, 1e-9);
  const double k2 = c.k2_prime * kCaSR;
  const double dRprime = -k2 * CaSS * Rp + c.k4 * (1.0 - Rp);
  const double O_ryr = k1 * CaSS * CaSS * Rp /
                        (c.k3 + k1 * CaSS * CaSS);

  const double I_rel = c.V_rel * O_ryr * (CaSR - CaSS);
  const double I_up  = c.V_max_up / (1.0 + std::pow(c.K_up / std::max(Cai, 1e-12), 2.0));
  const double I_leak = c.V_leak * (CaSR - Cai);
  const double I_xfer = c.V_xfer * (CaSS - Cai);

  // Buffer factors
  const double Bi = 1.0 / (1.0 + c.Buf_c * c.K_buf_c /
                                   std::pow(Cai + c.K_buf_c, 2.0));
  const double Bsr = 1.0 / (1.0 + c.Buf_sr * c.K_buf_sr /
                                   std::pow(CaSR + c.K_buf_sr, 2.0));
  const double Bss = 1.0 / (1.0 + c.Buf_ss * c.K_buf_ss /
                                   std::pow(CaSS + c.K_buf_ss, 2.0));

  // dCai/dt
  const double dCai = Bi * ((I_leak - I_up) * c.V_sr / c.V_c + I_xfer -
                            (I_bCa + I_pCa - 2.0 * I_NaCa) /
                              (2.0 * c.V_c * c.F) * c.Cm_pF * 1e-6);
  const double dCaSR = Bsr * (I_up - I_leak - I_rel);
  const double dCaSS = Bss * ((-I_CaL * c.Cm_pF * 1e-6) /
                                (2.0 * c.V_ss * c.F) +
                              I_rel * c.V_sr / c.V_ss -
                              I_xfer * c.V_c / c.V_ss);

  const double dNai = -(I_Na + I_bNa + 3.0 * I_NaK + 3.0 * I_NaCa + I_f_Na) *
                      c.Cm_pF * 1e-6 / (c.V_c * c.F);
  const double dKi  = -(I_to + I_Kr + I_Ks + I_K1 + I_pK + I_sus + I_f_K -
                        2.0 * I_NaK) *
                      c.Cm_pF * 1e-6 / (c.V_c * c.F);

  // Pack derivatives:
  // For gates we store (x_inf, tau) so caller can apply Rush-Larsen.
  dstate[1]  = m_inf;   dstate[2]  = h_inf;   dstate[3]  = j_inf;
  dstate[4]  = xr1_inf; dstate[5]  = xr2_inf; dstate[6]  = xs_inf;
  dstate[7]  = r_inf;   dstate[8]  = s_inf;   dstate[9]  = d_inf;
  dstate[10] = f_inf;   dstate[11] = f2_inf;  dstate[12] = fCass_inf;
  dstate[13] = y_inf;
  // Tau returned through a parallel array via negative sign trick is not nice;
  // instead caller passes a second buffer, but to keep API tight we encode
  // tau in dstate index +20 (kNumStates = 20). Stash tau values right after
  // states block (caller allocates 2*kNumStates).
  dstate[20 + 1]  = tau_m;
  dstate[20 + 2]  = tau_h;
  dstate[20 + 3]  = tau_j;
  dstate[20 + 4]  = tau_xr1;
  dstate[20 + 5]  = tau_xr2;
  dstate[20 + 6]  = tau_xs;
  dstate[20 + 7]  = tau_r;
  dstate[20 + 8]  = tau_s;
  dstate[20 + 9]  = tau_d;
  dstate[20 + 10] = tau_f;
  dstate[20 + 11] = tau_f2;
  dstate[20 + 12] = tau_fCass;
  dstate[20 + 13] = tau_y;

  // Forward-Euler concentrations / RyR
  dstate[14] = dCai;
  dstate[15] = dCaSR;
  dstate[16] = dCaSS;
  dstate[17] = dNai;
  dstate[18] = dKi;
  dstate[19] = dRprime;

  return I_ion;
}

}  // namespace

double StewartPurkinjeModel::ComputeNodeIion(double V, const double* s) {
  return NodeCurrents(V, s, nullptr);
}

void StewartPurkinjeModel::ComputeIion(const mfem::Vector& vm_true,
                                       mfem::Vector& iion_true) const {
  if (vm_true.Size() != n_nodes_) {
    throw std::runtime_error("StewartPurkinjeModel::ComputeIion size mismatch");
  }
  if (iion_true.Size() != n_nodes_) {
    iion_true.SetSize(n_nodes_);
  }
  for (int i = 0; i < n_nodes_; ++i) {
    const double* s = &states_[static_cast<size_t>(i) * kNumStates];
    iion_true[i] = ComputeNodeIion(vm_true[i], s);
  }
}

void StewartPurkinjeModel::AdvanceStates(double dt_pde_ms, double dt_ode_ms,
                                         const mfem::Vector& vm_next_true) {
  if (vm_next_true.Size() != n_nodes_) {
    throw std::runtime_error("StewartPurkinjeModel::AdvanceStates size mismatch");
  }
  if (dt_pde_ms <= 0.0 || dt_ode_ms <= 0.0) {
    throw std::runtime_error("StewartPurkinjeModel: dt must be > 0");
  }
  const int n_sub = static_cast<int>(std::ceil(dt_pde_ms / dt_ode_ms));
  const double dt = dt_pde_ms / static_cast<double>(n_sub);

  // Per-thread scratch (single-thread here -- MFEM Vector is not thread-safe).
  double dstate[2 * kNumStates];

  for (int n = 0; n < n_nodes_; ++n) {
    double* s = &states_[static_cast<size_t>(n) * kNumStates];
    const double V = vm_next_true[n];
    for (int sub = 0; sub < n_sub; ++sub) {
      s[0] = V;
      NodeCurrents(V, s, dstate);
      // Rush-Larsen for gates indices 1..13
      for (int gi = 1; gi <= 13; ++gi) {
        s[gi] = RushLarsen(s[gi], dstate[gi], dstate[20 + gi], dt);
      }
      // Forward Euler for concentrations / RyR (14..19)
      for (int ci = 14; ci <= 19; ++ci) {
        s[ci] += dt * dstate[ci];
      }
      // Clamp positivity
      s[14] = std::max(s[14], 1e-9);
      s[15] = std::max(s[15], 1e-9);
      s[16] = std::max(s[16], 1e-9);
      s[17] = std::max(s[17], 1e-3);
      s[18] = std::max(s[18], 1e-3);
      s[19] = std::clamp(s[19], 0.0, 1.0);
    }
    s[0] = V;
  }
}

void StewartPurkinjeModel::SaveState(std::ostream& os) const {
  const int32_t n = n_nodes_;
  os.write(reinterpret_cast<const char*>(&n), sizeof(n));
  os.write(reinterpret_cast<const char*>(states_.data()),
           static_cast<std::streamsize>(states_.size() * sizeof(double)));
}

void StewartPurkinjeModel::LoadState(std::istream& is) {
  int32_t n = 0;
  is.read(reinterpret_cast<char*>(&n), sizeof(n));
  if (n != n_nodes_) {
    throw std::runtime_error("StewartPurkinjeModel::LoadState size mismatch");
  }
  is.read(reinterpret_cast<char*>(states_.data()),
          static_cast<std::streamsize>(states_.size() * sizeof(double)));
  if (!is) {
    throw std::runtime_error("StewartPurkinjeModel::LoadState read failed");
  }
}

}  // namespace mono
