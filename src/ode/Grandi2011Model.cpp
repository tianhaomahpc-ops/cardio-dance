#include "ode/Grandi2011Model.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <istream>
#include <ostream>
#include <stdexcept>

namespace mono {
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

constexpr double R_const = 8314.472;
constexpr double T_const = 310.0;
constexpr double F_const = 96485.3415;
// Atrial conductances and constants (mS/uF unless noted).
constexpr double Nao = 140.0;
constexpr double Ko  = 5.4;
constexpr double Cao = 1.8;
constexpr double g_Na  = 23.0;
constexpr double g_CaL = 0.27;
constexpr double g_Ks  = 0.0035;
constexpr double g_Kr  = 0.035;
constexpr double g_K1  = 0.0525;
constexpr double g_tof = 0.165;     // I_to,fast (atrial-specific)
constexpr double g_kur = 0.045;     // I_Kur, atrial ultra-rapid K+
constexpr double g_bNa = 0.0006;
constexpr double g_bCa = 0.000725;
constexpr double K_pCa = 0.0005;
constexpr double g_pCa = 0.05;
constexpr double P_NaK = 1.4;
constexpr double Km_K = 1.5;
constexpr double Km_Na = 11.0;
constexpr double K_NaCa = 1100.0;
constexpr double Km_Nai = 87.5;
constexpr double Km_Ca = 1.38;
constexpr double K_sat = 0.27;
constexpr double alpha_NCX = 0.35;
constexpr double gamma_NCX = 0.35;
constexpr double Cm_pF = 0.110;
constexpr double V_c   = 25.84e-3;  // uL

double NodeIion(double V, const double* s, double* dstate) {
  const double m   = s[1];
  const double h   = s[2];
  const double jg  = s[3];
  const double d   = s[4];
  const double f   = s[5];
  const double xs  = s[6];
  const double xr  = s[7];
  const double xtof = s[8];
  const double ytof = s[9];
  const double xkur = s[10];
  const double ykur = s[11];
  const double Cai = s[12];
  const double Nai = s[13];
  const double Ki  = s[14];

  const double RT_F = R_const * T_const / F_const;
  const double E_Na = RT_F * std::log(Nao / std::max(Nai, 1e-6));
  const double E_K  = RT_F * std::log(Ko / std::max(Ki, 1e-6));
  const double E_Ks = RT_F * std::log((Ko + 0.018 * Nao) /
                                       std::max(Ki + 0.018 * Nai, 1e-6));
  const double E_Ca = 0.5 * RT_F * std::log(Cao / std::max(Cai, 1e-9));

  const double I_Na  = g_Na * m * m * m * h * jg * (V - E_Na);
  const double I_CaL = g_CaL * d * f * (V - 60.0);  // simplified driving force
  const double I_Kr  = g_Kr * std::sqrt(Ko / 5.4) * xr * (V - E_K);
  const double I_Ks  = g_Ks * xs * xs * (V - E_Ks);
  const double I_tof = g_tof * xtof * ytof * (V - E_K);
  const double I_Kur = g_kur * xkur * ykur * (V - E_K);
  const double xK1_inf = 1.0 / (1.0 + SafeExp(0.1 * (V + 75.0)));
  const double I_K1  = g_K1 * xK1_inf * (V - E_K);
  const double I_bNa = g_bNa * (V - E_Na);
  const double I_bCa = g_bCa * (V - E_Ca);

  const double VFRT = V * F_const / (R_const * T_const);
  const double NaK_denom = (1.0 + 0.1245 * SafeExp(-0.1 * VFRT) +
                            0.0353 * SafeExp(-VFRT));
  const double I_NaK = P_NaK * (Ko / (Ko + Km_K)) *
                       (Nai / (Nai + Km_Na)) / NaK_denom;

  const double e_g_VFRT = SafeExp(gamma_NCX * VFRT);
  const double e_g_m1_VFRT = SafeExp((gamma_NCX - 1.0) * VFRT);
  const double NaCa_num = K_NaCa *
      (e_g_VFRT * Nai * Nai * Nai * Cao -
       e_g_m1_VFRT * Nao * Nao * Nao * Cai * alpha_NCX);
  const double NaCa_den =
      (Km_Nai * Km_Nai * Km_Nai + Nao * Nao * Nao) *
      (Km_Ca + Cao) *
      (1.0 + K_sat * e_g_m1_VFRT);
  const double I_NaCa = NaCa_num / NaCa_den;

  const double I_pCa = g_pCa * Cai / (K_pCa + Cai);

  const double I_ion = I_Na + I_CaL + I_Kr + I_Ks + I_tof + I_Kur + I_K1 +
                       I_bNa + I_bCa + I_NaK + I_NaCa + I_pCa;

  if (dstate == nullptr) {
    return I_ion;
  }

  // Gates -- atrial-tuned where applicable.
  const double m_inf = 1.0 / std::pow(1.0 + SafeExp((-56.86 - V) / 9.03), 2.0);
  const double a_m = 1.0 / (1.0 + SafeExp((-60.0 - V) / 5.0));
  const double b_m = 0.1 / (1.0 + SafeExp((V + 35.0) / 5.0)) +
                     0.1 / (1.0 + SafeExp((V - 50.0) / 200.0));
  const double tau_m = a_m * b_m;

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

  // I_CaL gates -- d, f
  const double d_inf = 1.0 / (1.0 + SafeExp((-8.0 - V) / 7.5));
  const double tau_d = (1.4 / (1.0 + SafeExp((-35.0 - V) / 13.0)) + 0.25) *
                       (1.4 / (1.0 + SafeExp((V + 5.0) / 5.0))) +
                       1.0 / (1.0 + SafeExp((50.0 - V) / 20.0));
  const double f_inf = 1.0 / (1.0 + SafeExp((V + 20.0) / 7.0));
  const double tau_f = 1102.5 * SafeExp(-std::pow((V + 27.0) / 15.0, 2.0)) +
                       200.0 / (1.0 + SafeExp((13.0 - V) / 10.0)) +
                       180.0 / (1.0 + SafeExp((V + 30.0) / 10.0)) + 20.0;

  // I_Ks
  const double xs_inf = 1.0 / (1.0 + SafeExp((-5.0 - V) / 14.0));
  const double tau_xs = 1400.0 / std::sqrt(1.0 + SafeExp((5.0 - V) / 6.0)) *
                         (1.0 / (1.0 + SafeExp((V - 35.0) / 15.0))) + 80.0;

  // I_Kr
  const double xr_inf = 1.0 / (1.0 + SafeExp((-26.0 - V) / 7.0));
  const double tau_xr = 450.0 / (1.0 + SafeExp((-45.0 - V) / 10.0)) *
                         (6.0 / (1.0 + SafeExp((V + 30.0) / 11.5)));

  // I_to atrial
  const double xtof_inf = 1.0 / (1.0 + SafeExp((20.0 - V) / 13.0));
  const double tau_xtof = 10.45 * SafeExp(-std::pow((V + 40.0) / 25.0, 2.0)) + 7.3;
  const double ytof_inf = 1.0 / (1.0 + SafeExp((V + 27.0) / 13.0));
  const double tau_ytof = 85.0 * SafeExp(-std::pow((V + 25.0) / 32.0, 2.0)) +
                          5.0 / (1.0 + SafeExp((V - 40.0) / 5.0)) + 42.0;

  // I_Kur (atrial-specific, fast activation, slow inactivation)
  const double xkur_inf = 1.0 / (1.0 + SafeExp(-(V + 6.0) / 8.6));
  const double tau_xkur = 9.0 / (1.0 + SafeExp((V + 5.0) / 12.0)) + 0.5;
  const double ykur_inf = 1.0 / (1.0 + SafeExp((V + 7.5) / 10.0));
  const double tau_ykur = 590.0 / (1.0 + SafeExp((V + 60.0) / 10.0)) + 3050.0;

  // Concentration derivatives (Forward Euler)
  const double scale = Cm_pF * 1e-6 / (V_c * F_const);
  const double dCai = -(I_bCa + I_pCa - 2.0 * I_NaCa) * 0.5 * scale;
  const double dNai = -(I_Na + I_bNa + 3.0 * I_NaK + 3.0 * I_NaCa) * scale;
  const double dKi  = -(I_tof + I_Kur + I_Kr + I_Ks + I_K1 -
                         2.0 * I_NaK) * scale;

  // Pack outputs: gates 1..11 in dstate[1..11] = inf, dstate[20+i] = tau.
  dstate[1]  = m_inf;    dstate[20 + 1]  = tau_m;
  dstate[2]  = h_inf;    dstate[20 + 2]  = tau_h;
  dstate[3]  = j_inf;    dstate[20 + 3]  = tau_j;
  dstate[4]  = d_inf;    dstate[20 + 4]  = tau_d;
  dstate[5]  = f_inf;    dstate[20 + 5]  = tau_f;
  dstate[6]  = xs_inf;   dstate[20 + 6]  = tau_xs;
  dstate[7]  = xr_inf;   dstate[20 + 7]  = tau_xr;
  dstate[8]  = xtof_inf; dstate[20 + 8]  = tau_xtof;
  dstate[9]  = ytof_inf; dstate[20 + 9]  = tau_ytof;
  dstate[10] = xkur_inf; dstate[20 + 10] = tau_xkur;
  dstate[11] = ykur_inf; dstate[20 + 11] = tau_ykur;
  dstate[12] = dCai;
  dstate[13] = dNai;
  dstate[14] = dKi;

  return I_ion;
}

}  // namespace

Grandi2011Model::Grandi2011Model(int n_local_true_dofs)
    : n_nodes_(n_local_true_dofs),
      states_(static_cast<size_t>(n_local_true_dofs) * kNumStates, 0.0) {
  if (n_nodes_ <= 0) {
    throw std::runtime_error("Grandi2011Model requires n_local_true_dofs > 0");
  }
}

void Grandi2011Model::InitializeRestState(double v_rest_mv) {
  const double V0 = std::isfinite(v_rest_mv) ? v_rest_mv : -75.0;
  for (int n = 0; n < n_nodes_; ++n) {
    double* s = &states_[static_cast<size_t>(n) * kNumStates];
    s[0]  = V0;       // V
    s[1]  = 0.00132;  // m
    s[2]  = 0.7573;   // h
    s[3]  = 0.7225;   // j
    s[4]  = 3.37e-5;  // d
    s[5]  = 0.7888;   // f
    s[6]  = 0.0095;   // xs
    s[7]  = 0.00621;  // xr
    s[8]  = 2.42e-8;  // xtof
    s[9]  = 0.999998; // ytof
    s[10] = 0.000;    // xkur
    s[11] = 1.0;      // ykur
    s[12] = 1.5e-4;   // Cai
    s[13] = 9.0;      // Nai
    s[14] = 136.5;    // Ki
  }
}

void Grandi2011Model::ComputeIion(const mfem::Vector& vm_true,
                                  mfem::Vector& iion_true) const {
  if (vm_true.Size() != n_nodes_) {
    throw std::runtime_error("Grandi2011Model::ComputeIion size mismatch");
  }
  if (iion_true.Size() != n_nodes_) {
    iion_true.SetSize(n_nodes_);
  }
  for (int i = 0; i < n_nodes_; ++i) {
    const double* s = &states_[static_cast<size_t>(i) * kNumStates];
    iion_true[i] = NodeIion(vm_true[i], s, nullptr);
  }
}

void Grandi2011Model::AdvanceStates(double dt_pde_ms, double dt_ode_ms,
                                    const mfem::Vector& vm_next_true) {
  if (vm_next_true.Size() != n_nodes_) {
    throw std::runtime_error("Grandi2011Model::AdvanceStates size mismatch");
  }
  if (dt_pde_ms <= 0.0 || dt_ode_ms <= 0.0) {
    throw std::runtime_error("Grandi2011Model: dt must be > 0");
  }
  const int n_sub = static_cast<int>(std::ceil(dt_pde_ms / dt_ode_ms));
  const double dt = dt_pde_ms / static_cast<double>(n_sub);

  double dstate[2 * kNumStates + 8];

  for (int n = 0; n < n_nodes_; ++n) {
    double* s = &states_[static_cast<size_t>(n) * kNumStates];
    const double V = vm_next_true[n];
    for (int sub = 0; sub < n_sub; ++sub) {
      s[0] = V;
      NodeIion(V, s, dstate);
      // Gates 1..11
      for (int gi = 1; gi <= 11; ++gi) {
        s[gi] = RushLarsen(s[gi], dstate[gi], dstate[20 + gi], dt);
      }
      // Concentrations 12..14
      for (int ci = 12; ci <= 14; ++ci) {
        s[ci] += dt * dstate[ci];
      }
      s[12] = std::max(s[12], 1e-9);
      s[13] = std::max(s[13], 1e-3);
      s[14] = std::max(s[14], 1e-3);
    }
    s[0] = V;
  }
}

void Grandi2011Model::SaveState(std::ostream& os) const {
  const int32_t n = n_nodes_;
  os.write(reinterpret_cast<const char*>(&n), sizeof(n));
  os.write(reinterpret_cast<const char*>(states_.data()),
           static_cast<std::streamsize>(states_.size() * sizeof(double)));
}

void Grandi2011Model::LoadState(std::istream& is) {
  int32_t n = 0;
  is.read(reinterpret_cast<char*>(&n), sizeof(n));
  if (n != n_nodes_) {
    throw std::runtime_error("Grandi2011Model::LoadState size mismatch");
  }
  is.read(reinterpret_cast<char*>(states_.data()),
          static_cast<std::streamsize>(states_.size() * sizeof(double)));
  if (!is) {
    throw std::runtime_error("Grandi2011Model::LoadState read failed");
  }
}

}  // namespace mono
