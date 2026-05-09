#pragma once

#include <iosfwd>
#include <string>
#include <vector>

#include "mfem.hpp"

#include "ode/IIonicModel.hpp"

namespace mono {

// Stewart-Aslanidi-Noble-Noble-Boyett-Zhang 2009 Purkinje cell model.
//
// Implementation delegates to the CellML codegen output in
// ode/stewart_generated.{h,c}; this class only manages per-DOF state
// storage and applies the Rush-Larsen + Forward Euler operator-split
// integration that mirrors TT06Model. The codegen is the source-of-truth
// for the equations; do not hand-edit those files.
//
// State layout (matches codegen):
//   STATES[0]  = V_m
//   STATES[1]  = K_i        STATES[11] = Ca_ss
//   STATES[2]  = Na_i       STATES[12] = d
//   STATES[3]  = Ca_i       STATES[13] = f
//   STATES[4]  = y          STATES[14] = f2
//   STATES[5]  = Xr1        STATES[15] = fCass
//   STATES[6]  = Xr2        STATES[16] = s
//   STATES[7]  = Xs         STATES[17] = r
//   STATES[8]  = m          STATES[18] = Ca_SR
//   STATES[9]  = h          STATES[19] = R_prime
//   STATES[10] = j
class StewartPurkinjeModel : public IIonicModel {
 public:
  explicit StewartPurkinjeModel(int n_local_true_dofs);

  void InitializeRestState(double v_rest_mv) override;
  void ComputeIion(const mfem::Vector& vm_true,
                   mfem::Vector& iion_true) const override;
  void AdvanceStates(double dt_pde_ms, double dt_ode_ms,
                     const mfem::Vector& vm_next_true) override;

  void SaveState(std::ostream& os) const override;
  void LoadState(std::istream& is) override;

  int NumNodes() const override { return n_nodes_; }
  std::string ModelId() const override { return "Stewart2009"; }

 private:
  static constexpr int kNumStates = 20;
  static constexpr int kNumRates = 20;
  static constexpr int kNumConsts = 52;
  static constexpr int kNumAlg = 76;

  int n_nodes_;
  std::vector<double> states_;
  std::vector<double> rates_;
  std::vector<double> constants_;

  double* StatePtr(int node) { return states_.data() + node * kNumStates; }
  const double* StatePtr(int node) const {
    return states_.data() + node * kNumStates;
  }
  double* RatePtr(int node) { return rates_.data() + node * kNumRates; }
  double* ConstPtr(int node) { return constants_.data() + node * kNumConsts; }
  const double* ConstPtr(int node) const {
    return constants_.data() + node * kNumConsts;
  }

  static double RushLarsenUpdate(double x, double x_inf, double tau, double dt);
};

}  // namespace mono
