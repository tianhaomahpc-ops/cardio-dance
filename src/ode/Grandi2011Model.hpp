#pragma once

#include <iosfwd>
#include <vector>

#include "mfem.hpp"
#include "ode/IonicModel.hpp"

namespace mono {

// Node-local Grandi2011 ionic model wrapper.
class Grandi2011Model : public IonicModel {
 public:
  explicit Grandi2011Model(int n_local_true_dofs);

  void InitializeRestState(double v_rest_mv) override;
  void ComputeIion(const mfem::Vector& vm_true, mfem::Vector& iion_true) const override;
  void ComputeCytosolicCalcium(mfem::Vector& cai_true) const override;
  void AdvanceStates(double dt_pde_ms, double dt_ode_ms, const mfem::Vector& vm_next_true) override;

  void SaveState(std::ostream& os) const override;
  void LoadState(std::istream& is) override;

  int NumNodes() const override { return n_nodes_; }
  const char* ModelTag() const override { return "GRANDI2011"; }

 private:
  static constexpr int kNumStates = 41;
  static constexpr int kNumRates = 41;
  static constexpr int kNumConsts = 137;
  static constexpr int kNumAlg = 120;
  static constexpr int kVmStateIdx = 20;

  int n_nodes_ = 0;
  std::vector<double> states_;    // n_nodes x 41
  std::vector<double> rates_;     // n_nodes x 41
  std::vector<double> constants_; // n_nodes x 137

  double* StatePtr(int node) { return states_.data() + static_cast<size_t>(node) * kNumStates; }
  const double* StatePtr(int node) const {
    return states_.data() + static_cast<size_t>(node) * kNumStates;
  }
  double* RatePtr(int node) { return rates_.data() + static_cast<size_t>(node) * kNumRates; }
  const double* RatePtr(int node) const {
    return rates_.data() + static_cast<size_t>(node) * kNumRates;
  }
  double* ConstPtr(int node) { return constants_.data() + static_cast<size_t>(node) * kNumConsts; }
  const double* ConstPtr(int node) const {
    return constants_.data() + static_cast<size_t>(node) * kNumConsts;
  }
};

}  // namespace mono
