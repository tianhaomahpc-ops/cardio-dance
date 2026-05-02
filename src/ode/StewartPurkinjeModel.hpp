#pragma once

#include <iosfwd>
#include <vector>

#include "mfem.hpp"

#include "ode/IIonicModel.hpp"

namespace mono {

// Stewart-Aslanidi-Boyett-Zhang 2009 Purkinje cell model.
// 20 state variables, Rush-Larsen for gating and Forward Euler for
// concentrations -- mirrors the TT06Model integration scheme. Currents include
// I_Na, I_CaL, I_to, I_Ks, I_Kr, I_K1, I_NaCa, I_NaK, I_pCa, I_pK, I_bNa,
// I_bCa, I_f (HCN funny current responsible for automaticity), I_sus.
class StewartPurkinjeModel : public IIonicModel {
 public:
  explicit StewartPurkinjeModel(int n_local_true_dofs);

  void InitializeRestState(double v_rest_mv) override;
  void ComputeIion(const mfem::Vector& vm_true, mfem::Vector& iion_true) const override;
  void AdvanceStates(double dt_pde_ms, double dt_ode_ms,
                     const mfem::Vector& vm_next_true) override;

  void SaveState(std::ostream& os) const override;
  void LoadState(std::istream& is) override;

  int NumNodes() const override { return n_nodes_; }

 private:
  // 20 states match the Stewart 2009 CellML ordering used here:
  //   [0]V, [1]m, [2]h, [3]j, [4]xr1, [5]xr2, [6]xs, [7]r, [8]s, [9]d,
  //   [10]f, [11]f2, [12]fCass, [13]y, [14]Cai, [15]CaSR, [16]CaSS,
  //   [17]Nai, [18]Ki, [19]Rprime
  static constexpr int kNumStates = 20;

  // Per-node state stored contiguously: states_[node*kNumStates + i].
  int n_nodes_;
  std::vector<double> states_;

 public:
  // Single shared parameter set; Stewart cells in this codebase are uniform.
  // Constants follow Stewart 2009 (units: mV, ms, mM, uA/uF, mS/uF).
  struct Constants;
  static const Constants& Params();

 private:

  // Per-node ionic-current evaluation. Returns I_ion (uA/uF) at the given
  // (V, state). When advance_state is non-null, also populates the differential
  // updates so callers can integrate without recomputing currents.
  static double ComputeNodeIion(double V, const double* s);
  static void StepNode(double V, double dt, double* s);
};

}  // namespace mono
