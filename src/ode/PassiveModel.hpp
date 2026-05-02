#pragma once

#include <iosfwd>
#include <vector>

#include "mfem.hpp"

#include "ode/IIonicModel.hpp"

namespace mono {

// Leak-only ionic model used for AV-delay bridge and fibrotic regions where we
// want to suppress active depolarization but still allow electrotonic spread.
//   I_ion = g_leak * (V - V_rest)
// No internal gating state; checkpoint I/O is a no-op size header for symmetry
// with other models.
class PassiveModel : public IIonicModel {
 public:
  PassiveModel(int n_local_true_dofs, double g_leak_mS_per_uF, double v_rest_mv);

  void InitializeRestState(double v_rest_mv) override;
  void ComputeIion(const mfem::Vector& vm_true, mfem::Vector& iion_true) const override;
  void AdvanceStates(double dt_pde_ms, double dt_ode_ms,
                     const mfem::Vector& vm_next_true) override;

  void SaveState(std::ostream& os) const override;
  void LoadState(std::istream& is) override;

  int NumNodes() const override { return n_nodes_; }

  void SetLeakConductance(double g_mS_per_uF) { g_leak_ = g_mS_per_uF; }
  double LeakConductance() const { return g_leak_; }
  double RestPotential() const { return v_rest_; }

 private:
  int n_nodes_;
  double g_leak_;
  double v_rest_;
};

}  // namespace mono
