#pragma once

#include <iosfwd>
#include <vector>

#include "mfem.hpp"

namespace mono {

// Node-local active-tension model after Land et al. (2017):
//   "A model of cardiac contraction based on novel measurements of tension
//    development in human cardiomyocytes," J Mol Cell Cardiol 106:68-83.
//
// 8 ODE states per node:
//   [0] CaTRPN  : Ca-bound troponin fraction          [-]
//   [1] B       : blocked tropomyosin fraction        [-]
//   [2] S       : post-power-stroke crossbridges      [-]
//   [3] W       : pre-power-stroke crossbridges       [-]
//   [4] zeta_s  : distortion of S state               [-]
//   [5] zeta_w  : distortion of W state               [-]
//   [6] Cd      : slow cooperativity factor           [-]
//   [7] lam_int : low-pass filtered sarcomere length  [-]
//
// Inputs:
//   [Ca2+]_i in mM   (Land uses uM internally; we accept mM and convert)
//   lambda = sarcomere stretch along fiber (1.0 at rest)
//
// Output:
//   T_a in kPa, computed by GetTension().
class Land2017Model {
 public:
  struct Params {
    double Tref      = 120.0;     // kPa, reference active tension
    double Ca50_uM   = 0.805;     // uM
    double n_trpn    = 2.0;       // Hill coeff for TRPN
    double k_trpn    = 0.1;       // ms^-1
    double n_tm      = 5.0;       // Hill coeff for thin filament
    double TRPN50    = 0.35;
    double k_tm_unb  = 0.04;      // ms^-1 (TmB -> TmU baseline)
    double phi       = 2.23;      // unblocking cooperativity
    double k_uw      = 0.026;     // ms^-1
    double k_ws      = 0.004;     // ms^-1
    double k_su      = 0.00015;   // ms^-1 (S -> U detachment)
    double gamma_s   = 0.0085;
    double gamma_w   = 0.615;
    double beta_0    = 2.3;       // length-dep force coefficient
    double beta_1    = -2.4;
    double lambda_min = 0.87;
    double lambda_max = 1.2;
    double r_s       = 0.25;      // duty ratio
    double r_w       = 0.5;
    double A_eff     = 25.0;      // distortion gain
    double cd_tau    = 200.0;     // ms, Cd time constant
    double lam_tau   = 100.0;     // ms, sarcomere length filter constant
  };

  explicit Land2017Model(int n_local_true_dofs);
  Land2017Model(int n_local_true_dofs, const Params& params);

  // Reset all nodes to algebraic rest state at lambda=1, Ca=0.
  void InitializeRestState();

  // Set per-node sarcomere stretch (default 1.0 if never set).
  void SetStretch(const mfem::Vector& lambda_true);

  // Advance ODE states one PDE timestep with substepping of dt_ode_ms.
  // cai_true_mM: per-node cytosolic [Ca2+] in mM (will be converted to uM).
  void Advance(double dt_pde_ms, double dt_ode_ms, const mfem::Vector& cai_true_mM);

  // Compute active tension per node (kPa).
  void GetTension(mfem::Vector& ta_kPa_true) const;

  // Checkpoint I/O (binary). Header stores nodes and param signature.
  void SaveState(std::ostream& os) const;
  void LoadState(std::istream& is);

  int NumNodes() const { return n_nodes_; }
  const Params& GetParams() const { return params_; }

 private:
  static constexpr int kNumStates = 8;

  int n_nodes_;
  Params params_;
  std::vector<double> states_;     // n_nodes x 8
  std::vector<double> lambda_;     // n_nodes (current sarcomere stretch)

  inline double* StatePtr(int node) {
    return states_.data() + static_cast<size_t>(node) * kNumStates;
  }
  inline const double* StatePtr(int node) const {
    return states_.data() + static_cast<size_t>(node) * kNumStates;
  }

  // Evaluate algebraic length-dependent activation factor h(lambda).
  double LengthFactor(double lam) const;

  // Forward-Euler step for one node. Returns recommended substep count (>=1).
  void StepNode(int node, double dt_ms, double cai_uM);
};

}  // namespace mono
