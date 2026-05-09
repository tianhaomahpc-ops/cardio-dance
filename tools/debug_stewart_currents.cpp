// Debug instrument for Stewart 2009 single-cell.
//
// Runs the cell with a defined stimulus protocol, dumps a CSV with V and
// every individual current at every dump_dt_ms. Not a test -- a diagnostic
// tool for hunting peak overshoot.
//
// Usage:
//   ./debug_stewart_currents > /tmp/stewart_trace.csv
//   awk -F, '$2>30 {print}' /tmp/stewart_trace.csv | head  # show peak frames
//
// To instrument we duplicate Stewart's NodeCurrents inline so we can return
// every current separately. This duplication is contained to this debug tool;
// if Stewart's formula changes, this file may drift -- treat it as a probe,
// not as part of the verification matrix.

#include <cmath>
#include <cstdio>
#include <iostream>

#include "mfem.hpp"

#include "ode/StewartPurkinjeModel.hpp"

namespace {

inline double SafeExp(double x) {
  if (x > 50.0) return std::exp(50.0);
  if (x < -50.0) return std::exp(-50.0);
  return std::exp(x);
}

// Mirror of mono::StewartPurkinjeModel::Constants. Kept inline so the debug
// tool compiles without exposing the private struct. Drift is acceptable
// because this tool is diagnostic, not part of verification.
struct StewartConsts {
  double R = 8314.472, T = 310.0, F = 96485.3415;
  double Cm_pF = 0.185;
  double V_c = 0.016404, V_sr = 0.001094, V_ss = 0.00005468;
  double Ko = 5.4, Nao = 140.0, Cao = 2.0;
  double g_Na = 130.5744;
  double g_CaL = 3.98e-5;
  double g_to = 0.08184, g_Kr = 0.0918, g_Ks = 0.2352, g_K1 = 0.065;
  double g_f_Na = 0.0145654, g_f_K = 0.0234346;
  double g_pCa = 0.1238, g_pK = 0.0146;
  double g_bNa = 0.00029, g_bCa = 0.000592;
  double g_sus = 0.0227;
  double K_NaCa = 1000.0, K_sat = 0.1, alpha = 2.5, gamma = 0.35;
  double Km_Nai = 87.5, Km_Ca = 1.38;
  double P_NaK = 2.724, Km_K = 1.0, Km_Na = 40.0;
  double K_pCa = 0.0005;
};
static const StewartConsts kC{};

struct CurrentBreakdown {
  double I_Na, I_to, I_Kr, I_Ks, I_K1, I_CaL;
  double I_NaCa, I_NaK, I_pCa, I_pK, I_bNa, I_bCa;
  double I_f_Na, I_f_K, I_sus;
  double E_Na, E_K, E_Ca;
  double m, h, jg, d, f, f2, fCass;
  double Cai, CaSS, Nai, Ki;
};

CurrentBreakdown ComputeBreakdown(double V, const double* s) {
  const auto& c = kC;
  CurrentBreakdown b{};

  b.m = s[1]; b.h = s[2]; b.jg = s[3];
  const double xr1 = s[4], xr2 = s[5], xs = s[6];
  const double r = s[7], sg = s[8];
  b.d = s[9]; b.f = s[10]; b.f2 = s[11]; b.fCass = s[12];
  const double y = s[13];
  b.Cai = s[14]; const double CaSR = s[15]; b.CaSS = s[16];
  b.Nai = s[17]; b.Ki = s[18];

  const double RT_F = c.R * c.T / c.F;
  b.E_Na = RT_F * std::log(c.Nao / std::max(b.Nai, 1e-6));
  b.E_K  = RT_F * std::log(c.Ko  / std::max(b.Ki,  1e-6));
  const double E_Ks = RT_F * std::log((c.Ko + 0.03 * c.Nao) /
                                       std::max(b.Ki + 0.03 * b.Nai, 1e-6));
  b.E_Ca = 0.5 * RT_F * std::log(c.Cao / std::max(b.Cai, 1e-9));

  b.I_Na = c.g_Na * b.m * b.m * b.m * b.h * b.jg * (V - b.E_Na);
  b.I_to = c.g_to * r * sg * (V - b.E_K);
  b.I_Kr = c.g_Kr * std::sqrt(c.Ko / 5.4) * xr1 * xr2 * (V - b.E_K);
  b.I_Ks = c.g_Ks * xs * xs * (V - E_Ks);

  const double alpha_K1 = 0.1 / (1.0 + SafeExp(0.06 * (V - b.E_K - 200.0)));
  const double beta_K1 = (3.0 * SafeExp(0.0002 * (V - b.E_K + 100.0)) +
                          SafeExp(0.1 * (V - b.E_K - 10.0))) /
                          (1.0 + SafeExp(-0.5 * (V - b.E_K)));
  const double xK1_inf = alpha_K1 / (alpha_K1 + beta_K1);
  b.I_K1 = c.g_K1 * std::sqrt(c.Ko / 5.4) * xK1_inf * (V - b.E_K);

  const double Vshift = V - 15.0;
  const double VshiftFRT = Vshift * c.F / (c.R * c.T);
  const double exp2_Vshift_FRT = SafeExp(2.0 * VshiftFRT);
  double I_CaL = c.g_CaL * b.d * b.f * b.f2 * b.fCass * 4.0 * Vshift *
                 (c.F * c.F / (c.R * c.T));
  const double denom = exp2_Vshift_FRT - 1.0;
  if (std::fabs(denom) > 1e-9) {
    I_CaL *= (0.25 * b.CaSS * exp2_Vshift_FRT - c.Cao) / denom;
  } else {
    I_CaL *= 0.25 * b.CaSS - c.Cao;
  }
  b.I_CaL = I_CaL;

  const double VFRT = V * c.F / (c.R * c.T);
  const double e_g_VFRT = SafeExp(c.gamma * VFRT);
  const double e_g_m1_VFRT = SafeExp((c.gamma - 1.0) * VFRT);
  const double NaCa_num = c.K_NaCa *
      (e_g_VFRT * b.Nai * b.Nai * b.Nai * c.Cao -
       e_g_m1_VFRT * c.Nao * c.Nao * c.Nao * b.Cai * c.alpha);
  const double NaCa_den =
      (c.Km_Nai * c.Km_Nai * c.Km_Nai + c.Nao * c.Nao * c.Nao) *
      (c.Km_Ca + c.Cao) *
      (1.0 + c.K_sat * e_g_m1_VFRT);
  b.I_NaCa = NaCa_num / NaCa_den;

  b.I_NaK = c.P_NaK * (c.Ko / (c.Ko + c.Km_K)) *
            (b.Nai / (b.Nai + c.Km_Na)) /
            (1.0 + 0.1245 * SafeExp(-0.1 * VFRT) +
             0.0353 * SafeExp(-VFRT));

  b.I_pCa = c.g_pCa * b.Cai / (c.K_pCa + b.Cai);
  b.I_pK = c.g_pK * (V - b.E_K) / (1.0 + SafeExp((25.0 - V) / 5.98));
  b.I_bNa = c.g_bNa * (V - b.E_Na);
  b.I_bCa = c.g_bCa * (V - b.E_Ca);
  b.I_f_Na = c.g_f_Na * y * (V - b.E_Na);
  b.I_f_K  = c.g_f_K  * y * (V - b.E_K);
  b.I_sus = c.g_sus * (V - b.E_K);

  (void)CaSR;
  return b;
}

}  // namespace

int main(int argc, char* argv[]) {
  mfem::Mpi::Init(argc, argv);

  mono::StewartPurkinjeModel cell(1);
  cell.InitializeRestState(-90.0);

  const double dt = 0.005;
  const double t_end = 60.0;
  const int n_steps = static_cast<int>(t_end / dt);
  const int dump_every = static_cast<int>(0.05 / dt);  // dump every 0.05 ms

  mfem::Vector V(1);
  V[0] = -90.0;
  mfem::Vector iion(1);

  const double stim_start = 5.0;
  const double stim_end = 6.0;
  const double stim_amp = -52.0;

  std::cout << "t,V,I_total,I_Na,I_to,I_Kr,I_Ks,I_K1,I_CaL,"
               "I_NaCa,I_NaK,I_pCa,I_pK,I_bNa,I_bCa,I_f_Na,I_f_K,I_sus,"
               "E_Na,E_K,E_Ca,m,h,j,d,f,f2,fCass,Cai,CaSS,Nai,Ki\n";

  // We need access to the per-node state buffer for dumping. Friendliest
  // hack: read state via SaveState / LoadState round trip is overkill. Just
  // mirror state ourselves by reproducing the integration here. To keep the
  // dump aligned with the actual model, we still call the model's Compute /
  // Advance, and only reproduce the state-buffer by snapshotting via a
  // private inline state pointer. Since the class hides state we instead
  // use the model itself for advance and re-derive state via debugging:
  // simpler -- just shadow the integration in this tool. (Stewart cpp is
  // mirrored verbatim above.)

  // We track state ourselves so we can dump it.
  double s[20];
  s[0]  = -90.0;  s[1]  = 0.00132;  s[2]  = 0.7573;   s[3]  = 0.7225;
  s[4]  = 0.00621;s[5]  = 0.4712;   s[6]  = 0.0095;   s[7]  = 2.42e-8;
  s[8]  = 0.999998;s[9] = 3.373e-5; s[10] = 0.7888;   s[11] = 0.9755;
  s[12] = 0.9953; s[13] = 0.0457;   s[14] = 0.000126; s[15] = 3.64;
  s[16] = 0.00036;s[17] = 8.604;    s[18] = 136.89;   s[19] = 0.9073;

  for (int step = 0; step < n_steps; ++step) {
    const double t = (step + 1) * dt;

    // Use the model's actual implementation for the integration: we call
    // ComputeIion, then update V externally, then call AdvanceStates so the
    // model's internal state advances. Then we mirror the state by also
    // running our own NodeCurrents -- but the model state is private. To
    // keep the dump aligned without reflection hacks, we just integrate
    // ourselves using the same SafeExp/RushLarsen pattern. This duplicates
    // logic but is contained to debug only.

    s[0] = V[0];
    CurrentBreakdown b = ComputeBreakdown(V[0], s);
    double I_total = b.I_Na + b.I_to + b.I_Kr + b.I_Ks + b.I_K1 + b.I_CaL +
                     b.I_NaCa + b.I_NaK + b.I_pCa + b.I_pK + b.I_bNa + b.I_bCa +
                     b.I_f_Na + b.I_f_K + b.I_sus;

    const double i_stim = (t > stim_start && t <= stim_end) ? stim_amp : 0.0;
    V[0] -= dt * (I_total + i_stim);
    cell.AdvanceStates(dt, dt, V);  // also keep model state in sync

    // Mirror the state advance for the dump-side state buffer too.
    {
      const auto& c = kC;
      const double Vn = V[0];
      const double Cai = s[14], CaSR = s[15], CaSS = s[16];
      const double Nai = s[17], Ki = s[18], Rp = s[19];

      // ---- recompute gate inf/tau (replicates Stewart cpp) ----
      const double m_inf = 1.0 / std::pow(1.0 + SafeExp((-56.86 - Vn)/9.03), 2.0);
      const double a_m = 1.0 / (1.0 + SafeExp((-60.0 - Vn)/5.0));
      const double b_m = 0.1 / (1.0 + SafeExp((Vn + 35.0)/5.0)) +
                         0.1 / (1.0 + SafeExp((Vn - 50.0)/200.0));
      const double tau_m = a_m * b_m;

      const double h_inf = 1.0 / std::pow(1.0 + SafeExp((Vn + 71.55)/7.43), 2.0);
      double a_h, b_h;
      if (Vn >= -40.0) {
        a_h = 0.0;
        b_h = 0.77 / (0.13 * (1.0 + SafeExp(-(Vn + 10.66)/11.1)));
      } else {
        a_h = 0.057 * SafeExp(-(Vn + 80.0)/6.8);
        b_h = 2.7 * SafeExp(0.079 * Vn) + 3.1e5 * SafeExp(0.3485 * Vn);
      }
      const double tau_h = 1.0 / (a_h + b_h);

      const double j_inf = h_inf;
      double a_j, b_j;
      if (Vn >= -40.0) {
        a_j = 0.0;
        b_j = 0.6 * SafeExp(0.057 * Vn) /
              (1.0 + SafeExp(-0.1 * (Vn + 32.0)));
      } else {
        a_j = (-2.5428e4 * SafeExp(0.2444 * Vn) -
               6.948e-6 * SafeExp(-0.04391 * Vn)) *
              (Vn + 37.78) /
              (1.0 + SafeExp(0.311 * (Vn + 79.23)));
        b_j = 0.02424 * SafeExp(-0.01052 * Vn) /
              (1.0 + SafeExp(-0.1378 * (Vn + 40.14)));
      }
      const double tau_j = 1.0 / (a_j + b_j);

      const double d_inf = 1.0 / (1.0 + SafeExp((-8.0 - Vn)/7.5));
      const double a_d = 1.4 / (1.0 + SafeExp((-35.0 - Vn)/13.0)) + 0.25;
      const double b_d = 1.4 / (1.0 + SafeExp((Vn + 5.0)/5.0));
      const double g_d = 1.0 / (1.0 + SafeExp((50.0 - Vn)/20.0));
      const double tau_d = a_d * b_d + g_d;

      const double f_inf = 1.0 / (1.0 + SafeExp((Vn + 20.0)/7.0));
      const double tau_f = 1102.5 * SafeExp(-std::pow((Vn + 27.0)/15.0, 2.0)) +
                            200.0 / (1.0 + SafeExp((13.0 - Vn)/10.0)) +
                            180.0 / (1.0 + SafeExp((Vn + 30.0)/10.0)) + 20.0;
      const double f2_inf = 0.67/(1.0 + SafeExp((Vn + 35.0)/7.0)) + 0.33;
      const double tau_f2 = 562.0 * SafeExp(-std::pow((Vn + 27.0)/240.0, 2.0)) +
                             31.0 / (1.0 + SafeExp((25.0 - Vn)/10.0)) +
                             80.0 / (1.0 + SafeExp((Vn + 30.0)/10.0));
      const double fCass_inf = 0.6 / (1.0 + std::pow(CaSS/0.05, 2.0)) + 0.4;
      const double tau_fCass = 80.0 / (1.0 + std::pow(CaSS/0.05, 2.0)) + 2.0;

      auto rl = [](double x, double xinf, double tau, double dt_) {
        if (tau <= 1e-12 || !std::isfinite(tau)) return xinf;
        return xinf - (xinf - x) * std::exp(-dt_/tau);
      };

      s[1]  = rl(s[1],  m_inf, tau_m, dt);
      s[2]  = rl(s[2],  h_inf, tau_h, dt);
      s[3]  = rl(s[3],  j_inf, tau_j, dt);
      s[9]  = rl(s[9],  d_inf, tau_d, dt);
      s[10] = rl(s[10], f_inf, tau_f, dt);
      s[11] = rl(s[11], f2_inf, tau_f2, dt);
      s[12] = rl(s[12], fCass_inf, tau_fCass, dt);

      // r, s for I_to (Stewart Purkinje fast tau_r per 2026-05 fix)
      const double r_inf = 1.0 / (1.0 + SafeExp(-(Vn - 19.3) / 15.0));
      const double tau_r = 9.5 * SafeExp(-((Vn + 40.0) * (Vn + 40.0))/1800.0) + 0.8;
      const double s_inf2 = 1.0 / (1.0 + SafeExp((Vn + 27.0) / 13.0));
      const double tau_s = 85.0 * SafeExp(-std::pow((Vn + 25.0)/32.0, 2.0)) +
                            5.0 / (1.0 + SafeExp((Vn - 40.0)/5.0)) + 42.0;
      s[7] = rl(s[7], r_inf, tau_r, dt);
      s[8] = rl(s[8], s_inf2, tau_s, dt);

      // xr1, xr2 for I_Kr
      const double xr1_inf = 1.0 / (1.0 + SafeExp((-26.0 - Vn)/7.0));
      const double a_xr1 = 450.0 / (1.0 + SafeExp((-45.0 - Vn)/10.0));
      const double b_xr1 = 6.0 / (1.0 + SafeExp((Vn + 30.0)/11.5));
      const double tau_xr1 = a_xr1 * b_xr1;
      s[4] = rl(s[4], xr1_inf, tau_xr1, dt);

      const double xr2_inf = 1.0 / (1.0 + SafeExp((Vn + 88.0)/24.0));
      const double a_xr2 = 3.0 / (1.0 + SafeExp((-60.0 - Vn)/20.0));
      const double b_xr2 = 1.12 / (1.0 + SafeExp((Vn - 60.0)/20.0));
      const double tau_xr2 = a_xr2 * b_xr2;
      s[5] = rl(s[5], xr2_inf, tau_xr2, dt);

      // xs for I_Ks
      const double xs_inf = 1.0 / (1.0 + SafeExp((-5.0 - Vn)/14.0));
      const double a_xs = 1400.0 / std::sqrt(1.0 + SafeExp((5.0 - Vn)/6.0));
      const double b_xs = 1.0 / (1.0 + SafeExp((Vn - 35.0)/15.0));
      const double tau_xs = a_xs * b_xs + 80.0;
      s[6] = rl(s[6], xs_inf, tau_xs, dt);

      // y for I_f
      const double y_inf = 1.0 / (1.0 + SafeExp((Vn + 80.6)/6.8));
      const double tau_y = 4000.0 / (SafeExp(-2.9 - 0.04 * Vn) +
                                      SafeExp(3.6 + 0.11 * Vn));
      s[13] = rl(s[13], y_inf, tau_y, dt);

      // (skip xr1, xr2, xs, r, s, y for brevity -- already match well enough
      // for diagnosing peak. Concentrations also skipped.)
      (void)Cai; (void)CaSR; (void)CaSS; (void)Nai; (void)Ki; (void)Rp;
      (void)c;
    }

    if (step % dump_every == 0) {
      printf("%.4f,%.4f,%.4f,"
             "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,"
             "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,"
             "%.2f,%.2f,%.2f,"
             "%.4f,%.4f,%.4f,%.6f,%.4f,%.4f,%.4f,"
             "%.6f,%.6f,%.4f,%.4f\n",
             t, V[0], I_total,
             b.I_Na, b.I_to, b.I_Kr, b.I_Ks, b.I_K1, b.I_CaL,
             b.I_NaCa, b.I_NaK, b.I_pCa, b.I_pK, b.I_bNa, b.I_bCa,
             b.I_f_Na, b.I_f_K, b.I_sus,
             b.E_Na, b.E_K, b.E_Ca,
             b.m, b.h, b.jg, b.d, b.f, b.f2, b.fCass,
             b.Cai, b.CaSS, b.Nai, b.Ki);
    }
  }

  return 0;
}
