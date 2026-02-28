#include "ode/Grandi2011Model.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <istream>
#include <ostream>
#include <stdexcept>

#include "ode/grandi2011_generated.h"

namespace mono {

Grandi2011Model::Grandi2011Model(int n_local_true_dofs)
    : n_nodes_(n_local_true_dofs),
      states_(static_cast<size_t>(std::max(n_local_true_dofs, 0)) * kNumStates, 0.0),
      rates_(static_cast<size_t>(std::max(n_local_true_dofs, 0)) * kNumRates, 0.0),
      constants_(static_cast<size_t>(std::max(n_local_true_dofs, 0)) * kNumConsts, 0.0) {
  if (n_nodes_ < 0) {
    throw std::runtime_error("Grandi2011Model requires n_local_true_dofs >= 0");
  }
}

void Grandi2011Model::InitializeRestState(double v_rest_mv) {
  for (int i = 0; i < n_nodes_; ++i) {
    double* c = ConstPtr(i);
    double* r = RatePtr(i);
    double* s = StatePtr(i);
    grandi2011_initConsts(c, r, s);
    if (std::isfinite(v_rest_mv)) {
      s[kVmStateIdx] = v_rest_mv;
    }
  }
}

void Grandi2011Model::ComputeIion(const mfem::Vector& vm_true, mfem::Vector& iion_true) const {
  if (vm_true.Size() != n_nodes_) {
    throw std::runtime_error("Grandi2011Model::ComputeIion vm_true size mismatch");
  }
  if (iion_true.Size() != n_nodes_) {
    iion_true.SetSize(n_nodes_);
  }

  double local_states[kNumStates];
  double local_rates[kNumRates];
  double local_alg[kNumAlg];

  for (int i = 0; i < n_nodes_; ++i) {
    const double* s_stored = StatePtr(i);
    const double* r_stored = RatePtr(i);
    const double* c = ConstPtr(i);

    std::memcpy(local_states, s_stored, sizeof(local_states));
    std::memcpy(local_rates, r_stored, sizeof(local_rates));
    local_states[kVmStateIdx] = vm_true[i];

    grandi2011_computeRates(0.0, const_cast<double*>(c), local_rates, local_states, local_alg);
    iion_true[i] = local_alg[117];
  }
}

void Grandi2011Model::ComputeCytosolicCalcium(mfem::Vector& cai_true) const {
  if (cai_true.Size() != n_nodes_) {
    cai_true.SetSize(n_nodes_);
  }
  for (int i = 0; i < n_nodes_; ++i) {
    cai_true[i] = std::max(StatePtr(i)[11], 0.0); // Grandi2011 Ca_i
  }
}

void Grandi2011Model::AdvanceStates(double dt_pde_ms,
                                    double dt_ode_ms,
                                    const mfem::Vector& vm_next_true) {
  if (vm_next_true.Size() != n_nodes_) {
    throw std::runtime_error("Grandi2011Model::AdvanceStates vm_next_true size mismatch");
  }
  if (dt_pde_ms <= 0.0 || dt_ode_ms <= 0.0) {
    throw std::runtime_error("Grandi2011Model::AdvanceStates dt must be > 0");
  }

  const int n_sub = static_cast<int>(std::ceil(dt_pde_ms / dt_ode_ms));
  const double dt = dt_pde_ms / static_cast<double>(n_sub);

  double local_alg[kNumAlg];

  for (int node = 0; node < n_nodes_; ++node) {
    double* s = StatePtr(node);
    double* r = RatePtr(node);
    double* c = ConstPtr(node);

    for (int sub = 0; sub < n_sub; ++sub) {
      s[kVmStateIdx] = vm_next_true[node];
      grandi2011_computeRates((sub + 1) * dt, c, r, s, local_alg);

      for (int k = 0; k < kNumStates; ++k) {
        if (k == kVmStateIdx) {
          continue;
        }
        s[k] += dt * r[k];
      }

      // Keep concentration-like states non-negative for numerical safety.
      s[11] = std::max(s[11], 1e-12); // Ca_i
      s[12] = std::max(s[12], 1e-12); // Ca_jn
      s[13] = std::max(s[13], 1e-12); // Ca_sl
      s[18] = std::max(s[18], 1e-12); // Ca_sr
      s[16] = std::max(s[16], 1e-9);  // Na_jn
      s[17] = std::max(s[17], 1e-9);  // Na_sl
      s[28] = std::max(s[28], 1e-9);  // Na_i
      s[25] = std::max(s[25], 1e-9);  // K_i
    }

    s[kVmStateIdx] = vm_next_true[node];
  }
}

void Grandi2011Model::SaveState(std::ostream& os) const {
  const int32_t n = n_nodes_;
  os.write(reinterpret_cast<const char*>(&n), sizeof(n));
  os.write(reinterpret_cast<const char*>(states_.data()),
           static_cast<std::streamsize>(states_.size() * sizeof(double)));
  os.write(reinterpret_cast<const char*>(rates_.data()),
           static_cast<std::streamsize>(rates_.size() * sizeof(double)));
  os.write(reinterpret_cast<const char*>(constants_.data()),
           static_cast<std::streamsize>(constants_.size() * sizeof(double)));
}

void Grandi2011Model::LoadState(std::istream& is) {
  int32_t n = 0;
  is.read(reinterpret_cast<char*>(&n), sizeof(n));
  if (n != n_nodes_) {
    throw std::runtime_error("Grandi2011Model::LoadState node count mismatch");
  }
  is.read(reinterpret_cast<char*>(states_.data()),
          static_cast<std::streamsize>(states_.size() * sizeof(double)));
  is.read(reinterpret_cast<char*>(rates_.data()),
          static_cast<std::streamsize>(rates_.size() * sizeof(double)));
  is.read(reinterpret_cast<char*>(constants_.data()),
          static_cast<std::streamsize>(constants_.size() * sizeof(double)));
  if (!is) {
    throw std::runtime_error("Grandi2011Model::LoadState failed to read data");
  }
}

}  // namespace mono
