// PVJ active-coupling sign / direction test.
//
// Verifies that PvjCoupler::BuildHeartCouplingCurrent injects a *depolarizing*
// (negative I_ion-style) current into the myocardial RHS when the Purkinje
// terminal is more depolarized than the matched myocardial DOF, and a
// *repolarizing* current when the polarity is reversed.
//
// This is an algebraic test: it does not run the full PDE -- it sets known
// V_p and V_m values, calls BuildHeartCouplingCurrent, and asserts the sign
// of the produced current at the mapped DOF.
//
// Mesh fixture: 4-element 1-D segment (5 nodes, attribute = 1) created
// in-process. A 2-node Purkinje "network" is dropped on top with one terminal
// near node 0.

#include <cmath>
#include <fstream>
#include <iostream>
#include <string>

#include "mfem.hpp"

#include "config/SimulationConfig.hpp"
#include "ode/PassiveModel.hpp"
#include "solver/PurkinjeCableSolver.hpp"
#include "solver/PvjCoupler.hpp"

namespace {

std::string WriteTinyNetwork(double x_root, double x_term) {
  const std::string path = "/tmp/test_pvj_active_sign.network";
  std::ofstream out(path);
  // 2 nodes, 1 edge. Node 0 at x_root (interior), Node 1 at x_term (terminal).
  out << "2 1\n";
  out << x_root << " 0 0 0\n";
  out << x_term << " 0 0 1\n";
  out << "0 1\n";
  return path;
}

}  // namespace

int main(int argc, char* argv[]) {
  mfem::Mpi::Init(argc, argv);
  const int rank = mfem::Mpi::WorldRank();

  // Build a 1-D segment mesh of length 4 mm with 4 elements.
  mfem::Mesh serial_mesh = mfem::Mesh::MakeCartesian1D(4, 4.0);
  mfem::ParMesh pmesh(MPI_COMM_WORLD, serial_mesh);
  mfem::H1_FECollection fec(1, 1);
  mfem::ParFiniteElementSpace pfes(&pmesh, &fec);
  mfem::ParGridFunction vm_gf(&pfes);

  // Place Purkinje terminal at x = 1.0 mm, which (in serial) is mesh node 1.
  const std::string net_path = WriteTinyNetwork(/*root*/ 0.0, /*term*/ 1.0);

  mono::SimulationConfig cfg;
  cfg.purkinje_cable_subdivision = 1;
  cfg.purkinje_cm_uF_per_mm2 = 0.01;
  cfg.purkinje_edge_g_mS_per_mm = 1.5;
  cfg.purkinje_v_rest_mv = -85.0;
  cfg.pvj_g_mS = 0.5;
  cfg.pvj_max_dist_mm = 0.5;
  cfg.pvj_current_scale = 1.0;
  cfg.purkinje_dt_ms = 0.01;

  // Use a passive cable so we can directly set V_p via Initialize.
  mono::PassiveModel ionic(2, /*g_leak*/ 0.0, /*v_rest*/ 0.0);
  mono::PurkinjeCableSolver cable(cfg, ionic);
  cable.LoadNetwork(net_path);
  cable.Initialize(0.0);

  mono::PvjCoupler pvj(cfg, pfes, cable);

  if (rank == 0) {
    std::cout << "[pvj_sign] mapped " << pvj.GlobalNumMappedPvj()
              << "/" << cable.NumTerminals()
              << " (max_dist=" << pvj.GlobalMaxMappedDistMm() << " mm)"
              << std::endl;
  }
  if (pvj.GlobalNumMappedPvj() != 1) {
    if (rank == 0) std::cout << "[pvj_sign] expected 1 mapped terminal" << std::endl;
    return 1;
  }

  // --- Case A: V_p = +30, V_m = -85 -> Purkinje more depolarized.
  //   Expected: heart current at mapped DOF = pvj_g * (V_m - V_p) < 0
  //   (negative I_ion semantics = depolarizing).
  // --- Case B: swapped polarities -> current > 0 (repolarizing myocardium).

  auto run_case = [&](double Vp, double Vm) {
    // Inject Vp by re-initializing cable to a uniform Vp (both nodes share Vp
    // so terminal voltage = Vp).
    cable.Initialize(Vp);
    mfem::Vector vm_true(pfes.GetTrueVSize());
    vm_true = Vm;
    mfem::Vector heart_current_true(pfes.GetTrueVSize());
    pvj.BuildHeartCouplingCurrent(vm_true, heart_current_true);
    double max_abs = 0.0;
    int max_idx = -1;
    for (int i = 0; i < heart_current_true.Size(); ++i) {
      if (std::fabs(heart_current_true[i]) > std::fabs(max_abs)) {
        max_abs = heart_current_true[i];
        max_idx = i;
      }
    }
    return std::pair<int, double>{max_idx, max_abs};
  };

  bool ok = true;

  {
    auto [idx, val] = run_case(/*Vp*/ 30.0, /*Vm*/ -85.0);
    if (rank == 0) {
      std::cout << "[pvj_sign] case A (V_p=+30, V_m=-85): tdof=" << idx
                << " I_pvj=" << val << " (expect negative = depolarizing)"
                << std::endl;
    }
    if (val >= 0.0) ok = false;
    // Magnitude: g * (V_m - V_p) * scale = 0.5 * (-85 - 30) * 1.0 = -57.5
    if (std::fabs(val + 57.5) > 1e-6) ok = false;
  }

  {
    auto [idx, val] = run_case(/*Vp*/ -85.0, /*Vm*/ 30.0);
    if (rank == 0) {
      std::cout << "[pvj_sign] case B (V_p=-85, V_m=+30): tdof=" << idx
                << " I_pvj=" << val << " (expect positive = repolarizing)"
                << std::endl;
    }
    if (val <= 0.0) ok = false;
    if (std::fabs(val - 57.5) > 1e-6) ok = false;
  }

  return ok ? 0 : 1;
}
