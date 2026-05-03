// Purkinje cable conduction-velocity test.
//
// Builds a 50 mm straight Stewart cable, paces one end, measures the time
// difference between activation (V crosses -40 mV) at two probes 30 mm apart.
// Asserts CV in [1.5, 4.0] m/s -- physiological human Purkinje range
// (Janse 1969 / Boyden 2010).
//
// KNOWN LIMITATION (2026-05): the in-tree compact Stewart port is broken at
// the AP-peak level (see test_stewart_single_cell.cpp). CV depends primarily
// on dV/dt|max which can still be in range even when the peak overshoots.
// Test is marked WILL_FAIL until Stewart matches CellML reference, but kept
// so CI signals when CV starts behaving correctly.

#include <cmath>
#include <fstream>
#include <iostream>
#include <string>

#include "mfem.hpp"

#include "config/SimulationConfig.hpp"
#include "ode/StewartPurkinjeModel.hpp"
#include "solver/PurkinjeCableSolver.hpp"

namespace {

std::string WriteStraightCable(int n_nodes, double seg_mm) {
  const std::string path = "/tmp/test_purkinje_cable_cv.network";
  std::ofstream out(path);
  out << n_nodes << " " << (n_nodes - 1) << "\n";
  for (int i = 0; i < n_nodes; ++i) {
    const int term = (i == n_nodes - 1) ? 1 : 0;
    out << (i * seg_mm) << " 0.0 0.0 " << term << "\n";
  }
  for (int e = 0; e < n_nodes - 1; ++e) {
    out << e << " " << (e + 1) << "\n";
  }
  return path;
}

}  // namespace

int main(int argc, char* argv[]) {
  mfem::Mpi::Init(argc, argv);

  const int n_nodes = 51;        // 50 segments
  const double seg_mm = 1.0;     // 1 mm per segment -> 50 mm total
  const std::string path = WriteStraightCable(n_nodes, seg_mm);

  mono::SimulationConfig cfg;
  cfg.dt_pde_ms = 0.01;
  cfg.dt_ode_ms = 0.01;
  cfg.purkinje_dt_ms = 0.005;
  cfg.purkinje_cable_subdivision = 1;     // FE node per graph node
  cfg.purkinje_cm_uF_per_mm2 = 0.01;
  cfg.purkinje_edge_g_mS_per_mm = 1.5;     // axial conductance per mm
  cfg.purkinje_v_rest_mv = -90.0;
  cfg.purkinje_stim_nodes = {0};
  cfg.purkinje_stim_start_ms = 0.0;
  cfg.purkinje_stim_end_ms = 1.0;
  cfg.purkinje_stim_amp_uA_per_uF = -52.0;

  const int n_fe = n_nodes;
  mono::StewartPurkinjeModel ionic(n_fe);
  mono::PurkinjeCableSolver cable(cfg, ionic);
  cable.LoadNetwork(path);
  cable.Initialize(cfg.purkinje_v_rest_mv);

  const std::vector<double> empty_term_vm;
  const double t_end = 60.0;
  const int n_steps = static_cast<int>(t_end / cfg.dt_pde_ms);

  // Probes at 10 mm and 40 mm -> 30 mm separation.
  const int probe_a_node = 10;
  const int probe_b_node = 40;
  const double probe_separation_mm = (probe_b_node - probe_a_node) * seg_mm;
  const double activation_threshold = -40.0;
  double t_act_a = -1.0;
  double t_act_b = -1.0;

  for (int s = 0; s < n_steps; ++s) {
    const double t_mid = (s + 0.5) * cfg.dt_pde_ms;
    cable.Advance(cfg.dt_pde_ms, t_mid, empty_term_vm);

    const double t = (s + 1) * cfg.dt_pde_ms;
    if (t_act_a < 0.0 && cable.GraphNodeVoltage(probe_a_node) >= activation_threshold) {
      t_act_a = t;
    }
    if (t_act_b < 0.0 && cable.GraphNodeVoltage(probe_b_node) >= activation_threshold) {
      t_act_b = t;
      break;
    }
  }

  if (t_act_a < 0.0 || t_act_b < 0.0) {
    std::cout << "[cable_cv] failed to record activation at both probes (t_a="
              << t_act_a << ", t_b=" << t_act_b << ")" << std::endl;
    return 1;
  }

  const double dt_act_ms = t_act_b - t_act_a;
  const double cv_mm_per_ms = probe_separation_mm / dt_act_ms;
  const double cv_m_per_s = cv_mm_per_ms;  // mm/ms == m/s

  std::cout << "[cable_cv] t_act_a=" << t_act_a
            << " t_act_b=" << t_act_b
            << " dt_act=" << dt_act_ms << " ms"
            << " CV=" << cv_m_per_s << " m/s" << std::endl;

  const bool ok = (cv_m_per_s >= 1.5 && cv_m_per_s <= 4.0);
  return ok ? 0 : 1;
}
