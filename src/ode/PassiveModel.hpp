#pragma once

#include "ode/IonicModel.hpp"

namespace mono {

// Passive linear membrane model:
//   Iion = G * (V - E_R)
class PassiveModel : public IonicModel {
 public:
  PassiveModel(int n_local_true_dofs, double e_rest_mv, double g_mS_per_uF);

  void InitializeRestState(double v_rest_mv) override;
  void ComputeIion(const mfem::Vector& vm_true, mfem::Vector& iion_true) const override;
  void AdvanceStates(double dt_pde_ms, double dt_ode_ms, const mfem::Vector& vm_next_true) override;

  void SaveState(std::ostream& os) const override;
  void LoadState(std::istream& is) override;

  int NumNodes() const override { return n_nodes_; }
  const char* ModelTag() const override { return "PASSIVE"; }

 private:
  int n_nodes_ = 0;
  double e_rest_mv_ = -85.0;
  double g_mS_per_uF_ = 6.0643e-4;
};

}  // namespace mono
