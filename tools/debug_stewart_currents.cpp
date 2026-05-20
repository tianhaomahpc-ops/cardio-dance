// Stewart 2009 single-cell V(t) trace dumper.
//
// Drives the production StewartPurkinjeModel directly (no shadow integration)
// and writes a CSV with t, V, I_total. Used for visual inspection of AP
// shape against published Stewart 2009 figures.
//
// Usage:
//   ./debug_stewart_currents > /tmp/stewart_trace.csv
//   awk -F, '$2>0 {print}' /tmp/stewart_trace.csv | head    # peak frames

#include <cmath>
#include <cstdio>
#include <iostream>

#include "mfem.hpp"

#include "ode/StewartPurkinjeModel.hpp"

int main(int argc, char* argv[]) {
  mfem::Mpi::Init(argc, argv);

  mono::StewartPurkinjeModel cell(1);
  cell.InitializeRestState(-90.0);

  const double dt = 0.005;
  const double t_end = 1000.0;
  const int n_steps = static_cast<int>(t_end / dt);
  const int dump_every = static_cast<int>(0.05 / dt);

  mfem::Vector V(1);
  V[0] = -90.0;
  mfem::Vector iion(1);

  const double stim_start = 50.0;
  const double stim_end = 51.0;
  const double stim_amp = -52.0;

  std::cout << "t,V,I_total\n";

  for (int step = 0; step < n_steps; ++step) {
    const double t = (step + 1) * dt;

    cell.ComputeIion(V, iion);
    const double i_stim = (t > stim_start && t <= stim_end) ? stim_amp : 0.0;
    V[0] -= dt * (iion[0] + i_stim);
    cell.AdvanceStates(dt, dt, V);

    if (step % dump_every == 0) {
      printf("%.4f,%.4f,%.4f\n", t, V[0], iion[0]);
    }
  }

  return 0;
}
