// Stewart 2009 single-cell physiological-range trace test (1000 ms).
//
// KNOWN BROKEN (2026-05): the in-tree compact CellML port produces non-
// physiological action-potential peaks (~+120 mV vs paper ~+30 mV). Until the
// model is replaced with a CellML-generated source-of-truth (see TODO in
// src/ode/StewartPurkinjeModel.hpp), this test is run with strict thresholds
// and is EXPECTED TO FAIL. The CMake target is built but the ctest entry is
// disabled via WILL_FAIL so CI signals the regression without aborting the
// suite.
//
// Reference values (Stewart, Aslanidi, Boyett, Zhang 2009, fig 2):
//   V_rest    in [-92, -80] mV
//   V_peak    in [+15, +40] mV   (dome-shaped Purkinje AP)
//   APD90     in [280, 400] ms
//   V_at_min  < -75 mV (cell repolarizes)
//
// Integration: forward Euler on V driven by -I_ion -- I_stim. I_stim is
// applied as a *current* (uA/uF) added to the dV/dt balance, not as a direct
// voltage perturbation, so the upstroke shape reflects ionic dynamics rather
// than the integration scheme.

#include <cmath>
#include <cstdio>
#include <iostream>

#include "mfem.hpp"

#include "ode/StewartPurkinjeModel.hpp"

int main(int argc, char* argv[]) {
  mfem::Mpi::Init(argc, argv);

  const int n_nodes = 1;
  mono::StewartPurkinjeModel cell(n_nodes);
  cell.InitializeRestState(-90.0);

  const double dt = 0.005;     // ms
  const double t_end = 1000.0; // ms
  const int n_steps = static_cast<int>(t_end / dt);

  mfem::Vector V(1);
  V[0] = -90.0;
  mfem::Vector iion(1);
  iion = 0.0;

  // Stimulus as a current (negative = depolarizing, TT06 convention).
  const double stim_start = 50.0;
  const double stim_end = 51.0;
  const double stim_amp = -52.0;  // uA/uF, applied for 1 ms

  double v_peak = -120.0;
  double v_rest_pre = V[0];
  double t_apd_start = -1.0;
  double t_repol_90 = -1.0;
  double v_at_min = 1e30;
  const double upstroke_threshold = -55.0;

  for (int s = 0; s < n_steps; ++s) {
    const double t = (s + 1) * dt;

    cell.ComputeIion(V, iion);
    const double i_stim = (t > stim_start && t <= stim_end) ? stim_amp : 0.0;
    // dV/dt = -(I_ion + I_stim) under TT06 convention.
    V[0] -= dt * (iion[0] + i_stim);
    cell.AdvanceStates(dt, dt, V);

    if (t < stim_start) {
      v_rest_pre = V[0];
    }
    if (V[0] > v_peak) v_peak = V[0];
    if (t_apd_start < 0.0 && V[0] >= upstroke_threshold && t >= stim_start) {
      t_apd_start = t;
    }
    if (t > stim_end + 5.0 && t_apd_start > 0.0 && t_repol_90 < 0.0) {
      const double v90 = v_peak - 0.9 * (v_peak - v_rest_pre);
      if (V[0] <= v90) {
        t_repol_90 = t;
      }
    }
    if (t > stim_end + 5.0) {
      v_at_min = std::min(v_at_min, V[0]);
    }
  }

  const double apd90 = (t_repol_90 > 0.0 && t_apd_start > 0.0)
                            ? (t_repol_90 - t_apd_start)
                            : -1.0;

  std::cout << "[stewart] v_rest_pre=" << v_rest_pre
            << " v_peak=" << v_peak
            << " apd90=" << apd90 << " ms"
            << " v_at_min=" << v_at_min << std::endl;

  // Strict physiological assertions (Stewart et al. 2009 Fig. 2).
  // CMake marks this test WILL_FAIL while the compact CellML port is broken.
  bool ok = true;
  if (v_rest_pre < -92.0 || v_rest_pre > -80.0) ok = false;
  if (v_peak < 15.0 || v_peak > 40.0) ok = false;
  if (apd90 < 280.0 || apd90 > 400.0) ok = false;
  if (v_at_min > -75.0) ok = false;

  return ok ? 0 : 1;
}
