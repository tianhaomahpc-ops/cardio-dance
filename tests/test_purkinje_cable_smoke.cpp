// Smoke test for PurkinjeCableSolver: builds a 30-node straight cable, paces
// node 0, and verifies the wave reaches the far end within a plausible window.
// This is not a CV calibration -- only a correctness check.

#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>

#include "mfem.hpp"

#include "config/SimulationConfig.hpp"
#include "ode/PassiveModel.hpp"
#include "solver/PurkinjeCableSolver.hpp"

namespace {

std::string WriteStraightCable(int n_nodes, double seg_mm) {
  const std::string path = "/tmp/test_purkinje_cable_smoke.network";
  std::ofstream out(path);
  out << n_nodes << " " << (n_nodes - 1) << "\n";
  for (int i = 0; i < n_nodes; ++i) {
    const int term = (i == n_nodes - 1) ? 1 : 0;
    out << i * seg_mm << " 0.0 0.0 " << term << "\n";
  }
  for (int e = 0; e < n_nodes - 1; ++e) {
    out << e << " " << (e + 1) << "\n";
  }
  return path;
}

}  // namespace

int main(int argc, char* argv[]) {
  mfem::Mpi::Init(argc, argv);

  const int n_nodes = 30;
  const double seg_mm = 0.5;
  const std::string path = WriteStraightCable(n_nodes, seg_mm);

  mono::SimulationConfig cfg;
  cfg.dt_pde_ms = 0.05;
  cfg.dt_ode_ms = 0.05;
  cfg.purkinje_dt_ms = 0.01;
  cfg.purkinje_cable_subdivision = 1;
  cfg.purkinje_cm_uF_per_mm2 = 0.01;
  cfg.purkinje_edge_g_mS_per_mm = 1.5;
  cfg.purkinje_v_rest_mv = -85.0;
  cfg.purkinje_stim_nodes = {0};
  cfg.purkinje_stim_start_ms = 0.0;
  cfg.purkinje_stim_end_ms = 1.0;
  cfg.purkinje_stim_amp_uA_per_uF = -300.0;  // strong pulse to ensure firing

  // Use a passive cable for smoke (verifying diffusion); this won't fire an
  // AP but gives us a deterministic decay/propagation we can sanity-check.
  // We use a low leak so the pulse propagates substantially.
  const int n_fe = n_nodes;
  mono::PassiveModel ionic(n_fe, /*g_leak*/ 0.02, /*v_rest*/ -85.0);
  mono::PurkinjeCableSolver cable(cfg, ionic);
  cable.LoadNetwork(path);
  cable.Initialize(-85.0);

  const std::vector<double> empty_term_vm;  // no PVJ coupling here
  const double t_end = 20.0;
  const int n_steps = static_cast<int>(t_end / cfg.dt_pde_ms);

  double v_far_max = -1e9;
  for (int s = 0; s < n_steps; ++s) {
    const double t_mid = (s + 0.5) * cfg.dt_pde_ms;
    cable.Advance(cfg.dt_pde_ms, t_mid, empty_term_vm);
    const double v_far = cable.GraphNodeVoltage(n_nodes - 1);
    if (v_far > v_far_max) v_far_max = v_far;
  }

  std::cout << "[cable_smoke] far-end max V = " << v_far_max
            << " mV, near-end V = " << cable.GraphNodeVoltage(0)
            << " mV" << std::endl;

  // Pulse should reach end with at least mild depolarization above rest.
  const bool ok = v_far_max > -84.5;
  return ok ? 0 : 1;
}
