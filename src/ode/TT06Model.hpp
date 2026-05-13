#pragma once

#include <iosfwd>
#include <vector>

#include "mfem.hpp"
#include "ode/IonicModel.hpp"

namespace mono {

// Node-local TT06 ionic model wrapper.
// Stores full CellML state per true DOF and provides:
//   1) I_ion evaluation at a supplied Vm field
//   2) ODE state advancement with Rush-Larsen + Forward Euler split.
class TT06Model : public IonicModel {
 public:
  explicit TT06Model(int n_local_true_dofs);

  // Initialize each node with TT06 default constants/resting state.
  void InitializeRestState(double v_rest_mv) override;
  // Evaluate ionic current using current states and externally supplied Vm.
  void ComputeIion(const mfem::Vector& vm_true, mfem::Vector& iion_true) const override;
  // Advance ionic states over one PDE step using substepping dt_ode_ms.
  void AdvanceStates(double dt_pde_ms, double dt_ode_ms, const mfem::Vector& vm_next_true) override;

  // Binary checkpoint I/O for restart.
  void SaveState(std::ostream& os) const override;
  void LoadState(std::istream& is) override;

  int NumNodes() const override { return n_nodes_; }
  const char* ModelTag() const override { return "TT06"; }

  // TT06 state index 3 is cytosolic Ca_i (mM). Initial rest value 1.26e-4.
  bool GetCytosolicCalcium(mfem::Vector& cai_true_mM) const override;
  static constexpr int CaiStateIndex() { return 3; }

 private:
  static constexpr int kNumStates = 19;
  static constexpr int kNumRates = 19;
  static constexpr int kNumConsts = 54;
  static constexpr int kNumAlg = 71;

  int n_nodes_;
  std::vector<double> states_;    // n_nodes x 19
  std::vector<double> rates_;     // n_nodes x 19
  std::vector<double> constants_; // n_nodes x 54

  double* StatePtr(int node) { return states_.data() + node * kNumStates; }
  const double* StatePtr(int node) const { return states_.data() + node * kNumStates; }
  double* RatePtr(int node) { return rates_.data() + node * kNumRates; }
  double* ConstPtr(int node) { return constants_.data() + node * kNumConsts; }
  const double* ConstPtr(int node) const { return constants_.data() + node * kNumConsts; }

  static double RushLarsenUpdate(double x, double x_inf, double tau, double dt);
};

}  // namespace mono
