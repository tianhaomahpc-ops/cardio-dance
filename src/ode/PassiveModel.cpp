#include "ode/PassiveModel.hpp"

#include <cstdint>
#include <istream>
#include <ostream>
#include <stdexcept>

namespace mono {

PassiveModel::PassiveModel(int n_local_true_dofs, double g_leak_mS_per_uF, double v_rest_mv)
    : n_nodes_(n_local_true_dofs),
      g_leak_(g_leak_mS_per_uF),
      v_rest_(v_rest_mv) {
  if (n_nodes_ <= 0) {
    throw std::runtime_error("PassiveModel requires n_local_true_dofs > 0");
  }
}

void PassiveModel::InitializeRestState(double v_rest_mv) {
  v_rest_ = v_rest_mv;
}

void PassiveModel::ComputeIion(const mfem::Vector& vm_true, mfem::Vector& iion_true) const {
  if (vm_true.Size() != n_nodes_) {
    throw std::runtime_error("PassiveModel::ComputeIion: vm_true size mismatch");
  }
  if (iion_true.Size() != n_nodes_) {
    iion_true.SetSize(n_nodes_);
  }
  for (int i = 0; i < n_nodes_; ++i) {
    // Leak current: positive when V > V_rest (repolarizing). The TT06 sign
    // convention used downstream treats positive I_ion as outward, matching.
    iion_true[i] = g_leak_ * (vm_true[i] - v_rest_);
  }
}

void PassiveModel::AdvanceStates(double /*dt_pde_ms*/, double /*dt_ode_ms*/,
                                 const mfem::Vector& vm_next_true) {
  if (vm_next_true.Size() != n_nodes_) {
    throw std::runtime_error("PassiveModel::AdvanceStates: vm_next_true size mismatch");
  }
  // Stateless model.
}

void PassiveModel::SaveState(std::ostream& os) const {
  const int32_t n = n_nodes_;
  os.write(reinterpret_cast<const char*>(&n), sizeof(n));
  os.write(reinterpret_cast<const char*>(&g_leak_), sizeof(g_leak_));
  os.write(reinterpret_cast<const char*>(&v_rest_), sizeof(v_rest_));
}

void PassiveModel::LoadState(std::istream& is) {
  int32_t n = 0;
  is.read(reinterpret_cast<char*>(&n), sizeof(n));
  if (n != n_nodes_) {
    throw std::runtime_error("PassiveModel::LoadState node count mismatch");
  }
  is.read(reinterpret_cast<char*>(&g_leak_), sizeof(g_leak_));
  is.read(reinterpret_cast<char*>(&v_rest_), sizeof(v_rest_));
  if (!is) {
    throw std::runtime_error("PassiveModel::LoadState failed to read");
  }
}

}  // namespace mono
