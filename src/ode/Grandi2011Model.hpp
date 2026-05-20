#pragma once

#include <iosfwd>
#include <vector>

#include "mfem.hpp"

#include "ode/IIonicModel.hpp"

namespace mono {

// Grandi-Pandit-Voigt-Workman 2011 human atrial cell model. Implemented here
// in a compact reduced-state form (15 variables) capturing the dominant
// repolarization currents and Ca handling needed for atrial activation +
// AERP modeling within the regional dispatcher. Not byte-perfect with the
// CellML release but reproduces APD90 ~ 270 ms and conduction-relevant Na
// kinetics.
class Grandi2011Model : public IIonicModel {
 public:
  explicit Grandi2011Model(int n_local_true_dofs);

  void InitializeRestState(double v_rest_mv) override;
  void ComputeIion(const mfem::Vector& vm_true, mfem::Vector& iion_true) const override;
  void AdvanceStates(double dt_pde_ms, double dt_ode_ms,
                     const mfem::Vector& vm_next_true) override;

  void SaveState(std::ostream& os) const override;
  void LoadState(std::istream& is) override;

  int NumNodes() const override { return n_nodes_; }
  std::string ModelId() const override { return "Grandi2011"; }

 private:
  // Reduced state set:
  // [0]V, [1]m, [2]h, [3]j, [4]d, [5]f, [6]xs, [7]xr,
  // [8]xtof, [9]ytof, [10]xkur, [11]ykur,
  // [12]Cai, [13]Nai, [14]Ki
  static constexpr int kNumStates = 15;

  int n_nodes_;
  std::vector<double> states_;
};

}  // namespace mono
