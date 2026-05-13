#include "ode/PassiveModel.hpp"

#include <cmath>
#include <cstdint>
#include <istream>
#include <ostream>
#include <stdexcept>

namespace mono {

PassiveModel::PassiveModel(int n_local_true_dofs, double e_rest_mv, double g_mS_per_uF)
    : n_nodes_(n_local_true_dofs), e_rest_mv_(e_rest_mv), g_mS_per_uF_(g_mS_per_uF) {
  if (n_nodes_ < 0) {
    throw std::runtime_error("PassiveModel requires n_local_true_dofs >= 0");
  }
  if (g_mS_per_uF_ < 0.0) {
    throw std::runtime_error("PassiveModel requires non-negative conductance");
  }
}

void PassiveModel::InitializeRestState(double v_rest_mv) {
  if (std::isfinite(v_rest_mv)) {
    e_rest_mv_ = v_rest_mv;
  }
}

void PassiveModel::ComputeIion(const mfem::Vector& vm_true, mfem::Vector& iion_true) const {
  if (vm_true.Size() != n_nodes_) {
    throw std::runtime_error("PassiveModel::ComputeIion vm_true size mismatch");
  }
  if (iion_true.Size() != n_nodes_) {
    iion_true.SetSize(n_nodes_);
  }
  for (int i = 0; i < n_nodes_; ++i) {
    iion_true[i] = g_mS_per_uF_ * (vm_true[i] - e_rest_mv_);
  }
}

void PassiveModel::AdvanceStates(double,
                                 double,
                                 const mfem::Vector& vm_next_true) {
  if (vm_next_true.Size() != n_nodes_) {
    throw std::runtime_error("PassiveModel::AdvanceStates vm_next_true size mismatch");
  }
}

void PassiveModel::SaveState(std::ostream& os) const {
  const int32_t n = n_nodes_;
  os.write(reinterpret_cast<const char*>(&n), sizeof(n));
  os.write(reinterpret_cast<const char*>(&e_rest_mv_), sizeof(e_rest_mv_));
  os.write(reinterpret_cast<const char*>(&g_mS_per_uF_), sizeof(g_mS_per_uF_));
}

void PassiveModel::LoadState(std::istream& is) {
  int32_t n = 0;
  is.read(reinterpret_cast<char*>(&n), sizeof(n));
  if (n != n_nodes_) {
    throw std::runtime_error("PassiveModel::LoadState node count mismatch");
  }
  is.read(reinterpret_cast<char*>(&e_rest_mv_), sizeof(e_rest_mv_));
  is.read(reinterpret_cast<char*>(&g_mS_per_uF_), sizeof(g_mS_per_uF_));
  if (!is) {
    throw std::runtime_error("PassiveModel::LoadState failed to read data");
  }
}

}  // namespace mono
