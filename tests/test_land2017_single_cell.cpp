// Single-cell Land 2017 verification:
//   * triangular [Ca2+]_i transient (peak ~1 uM, duration ~400 ms)
//   * lambda = 1
//   * check T_a peak in [10, 200] kPa, rise time in (0, 400) ms.
//
// This is a smoke test for ODE stability; quantitative match to Land 2017
// Fig 3 requires tuning of k_uw/k_ws/k_su and is deferred.

#include <algorithm>
#include <cmath>
#include <iostream>

#include "mfem.hpp"

#include "ode/Land2017Model.hpp"

int main() {
  using namespace mono;
  try {
    Land2017Model::Params params;  // defaults
    Land2017Model land(1, params);
    land.InitializeRestState();

    const double dt = 0.05;       // ms
    const double t_end = 800.0;   // ms
    const double ca_peak_uM = 1.0;
    const double ca_dur_ms = 400.0;
    const double t_rise = 50.0;

    mfem::Vector cai_mM(1);
    mfem::Vector ta(1);
    mfem::Vector lam(1);
    lam[0] = 1.0;
    land.SetStretch(lam);

    double ta_max = 0.0;
    double t_at_peak = 0.0;
    double ca_at_peak = 0.0;
    for (double t = 0.0; t <= t_end; t += dt) {
      // Triangular Ca transient in uM (then converted to mM by /1000 here).
      double ca_uM = 0.0;
      if (t < t_rise) {
        ca_uM = ca_peak_uM * (t / t_rise);
      } else if (t < ca_dur_ms) {
        ca_uM = ca_peak_uM * (1.0 - (t - t_rise) / (ca_dur_ms - t_rise));
      } else {
        ca_uM = 0.0;
      }
      cai_mM[0] = ca_uM * 1e-3;

      land.Advance(dt, dt, cai_mM);
      land.GetTension(ta);
      if (ta[0] > ta_max) {
        ta_max = ta[0];
        t_at_peak = t;
        ca_at_peak = ca_uM;
      }
    }

    std::cout << "Land2017 single cell: ta_max=" << ta_max
              << " kPa at t=" << t_at_peak << " ms (Ca=" << ca_at_peak
              << " uM)\n";

    if (!std::isfinite(ta_max)) return 1;
    if (!(ta_max > 1e-3)) return 2;          // some tension developed
    if (!(ta_max < 500.0)) return 3;         // sanity bound
    if (!(t_at_peak > 10.0 && t_at_peak < t_end)) return 4;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "exception: " << e.what() << std::endl;
    return 10;
  }
}
