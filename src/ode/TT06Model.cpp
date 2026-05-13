#include "ode/TT06Model.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <istream>
#include <limits>
#include <ostream>
#include <stdexcept>

#include "ode/tt06_generated.h"

namespace mono {
namespace {

// Mapping for Rush-Larsen-updated gating variables:
// x' = (x_inf(V)-x) / tau(V)
struct GateMap {
  int state_idx;
  int inf_idx;
  int tau_idx;
};

constexpr GateMap kGateMaps[] = {
    {4, 0, 34},   // Xr1
    {5, 1, 35},   // Xr2
    {6, 2, 36},   // Xs
    {7, 3, 37},   // m
    {8, 4, 38},   // h
    {9, 5, 39},   // j
    {11, 6, 42},  // d
    {12, 7, 20},  // f
    {13, 8, 21},  // f2
    {14, 9, 22},  // fCass
    {15, 10, 23}, // s
    {16, 11, 24}, // r
};

constexpr int kFeStateIdx[] = {1, 2, 3, 10, 17, 18};

}  // namespace

TT06Model::TT06Model(int n_local_true_dofs)
    : n_nodes_(n_local_true_dofs),
      states_(static_cast<size_t>(n_local_true_dofs) * kNumStates, 0.0),
      rates_(static_cast<size_t>(n_local_true_dofs) * kNumRates, 0.0),
      constants_(static_cast<size_t>(n_local_true_dofs) * kNumConsts, 0.0) {
  if (n_nodes_ < 0) {
    throw std::runtime_error("TT06Model requires n_local_true_dofs >= 0");
  }
}

void TT06Model::InitializeRestState(double v_rest_mv) {
  for (int i = 0; i < n_nodes_; ++i) {
    double* c = ConstPtr(i);
    double* r = RatePtr(i);
    double* s = StatePtr(i);
    initConsts(c, r, s);

    // Disable embedded CellML stimulus; monodomain-level I_stim is used instead.
    c[51] = 0.0;

    if (std::isfinite(v_rest_mv)) {
      s[0] = v_rest_mv;
    }
  }
}

void TT06Model::ComputeIion(const mfem::Vector& vm_true, mfem::Vector& iion_true) const {
  if (vm_true.Size() != n_nodes_) {
    throw std::runtime_error("ComputeIion: vm_true size mismatch");
  }
  if (iion_true.Size() != n_nodes_) {
    iion_true.SetSize(n_nodes_);
  }

  // Stack temporaries avoid mutating persisted node states for pure I_ion eval.
  double local_states[kNumStates];
  double local_rates[kNumRates];
  double local_alg[kNumAlg];

  for (int i = 0; i < n_nodes_; ++i) {
    const double* s_stored = StatePtr(i);
    const double* c = ConstPtr(i);

    std::memcpy(local_states, s_stored, sizeof(local_states));
    std::memcpy(local_rates, rates_.data() + static_cast<size_t>(i) * kNumRates, sizeof(local_rates));

    local_states[0] = vm_true[i];
    computeVariables(0.0, const_cast<double*>(c), local_rates, local_states, local_alg);

    // Ionic current excludes external stimulus term ALGEBRAIC[61].
    const double iion = local_alg[46] + local_alg[53] + local_alg[47] + local_alg[48] +
                        local_alg[51] + local_alg[54] + local_alg[49] + local_alg[50] +
                        local_alg[55] + local_alg[52] + local_alg[57] + local_alg[56];
    iion_true[i] = iion;
  }
}

double TT06Model::RushLarsenUpdate(double x, double x_inf, double tau, double dt) {
  if (tau <= 1e-12 || !std::isfinite(tau)) {
    return x_inf;
  }
  const double expo = std::exp(-dt / tau);
  return x_inf - (x_inf - x) * expo;
}

void TT06Model::AdvanceStates(double dt_pde_ms, double dt_ode_ms, const mfem::Vector& vm_next_true) {
  if (vm_next_true.Size() != n_nodes_) {
    throw std::runtime_error("AdvanceStates: vm_next_true size mismatch");
  }
  if (dt_pde_ms <= 0.0 || dt_ode_ms <= 0.0) {
    throw std::runtime_error("AdvanceStates: dt must be > 0");
  }

  // Use integer substeps so ODE and PDE stay synchronized at step boundaries.
  const int n_sub = static_cast<int>(std::ceil(dt_pde_ms / dt_ode_ms));
  const double dt = dt_pde_ms / static_cast<double>(n_sub);

  double local_alg[kNumAlg];

  for (int node = 0; node < n_nodes_; ++node) {
    double* s = StatePtr(node);
    double* r = RatePtr(node);
    double* c = ConstPtr(node);

    for (int sub = 0; sub < n_sub; ++sub) {
      s[0] = vm_next_true[node];
      computeRates((sub + 1) * dt, c, r, s, local_alg);

      // Rush-Larsen for gating variables.
      for (const auto& gm : kGateMaps) {
        s[gm.state_idx] = RushLarsenUpdate(s[gm.state_idx], local_alg[gm.inf_idx], local_alg[gm.tau_idx], dt);
      }

      // Forward Euler for concentration-like states and slow variables.
      for (const int idx : kFeStateIdx) {
        s[idx] += dt * r[idx];
      }

      // Keep critical concentrations positive to avoid breakdown.
      s[3] = std::max(s[3], 1e-12);   // Ca_i
      s[10] = std::max(s[10], 1e-12); // Ca_ss
      s[17] = std::max(s[17], 1e-12); // Ca_SR
    }

    s[0] = vm_next_true[node];
  }
}

bool TT06Model::GetCytosolicCalcium(mfem::Vector& cai_true_mM) const {
  cai_true_mM.SetSize(n_nodes_);
  constexpr int idx = CaiStateIndex();
  for (int node = 0; node < n_nodes_; ++node) {
    cai_true_mM[node] = states_[static_cast<size_t>(node) * kNumStates + idx];
  }
  return true;
}

void TT06Model::SaveState(std::ostream& os) const {
  const int32_t n = n_nodes_;
  os.write(reinterpret_cast<const char*>(&n), sizeof(n));
  os.write(reinterpret_cast<const char*>(states_.data()), static_cast<std::streamsize>(states_.size() * sizeof(double)));
  os.write(reinterpret_cast<const char*>(rates_.data()), static_cast<std::streamsize>(rates_.size() * sizeof(double)));
  os.write(reinterpret_cast<const char*>(constants_.data()), static_cast<std::streamsize>(constants_.size() * sizeof(double)));
}

void TT06Model::LoadState(std::istream& is) {
  int32_t n = 0;
  is.read(reinterpret_cast<char*>(&n), sizeof(n));
  if (n != n_nodes_) {
    throw std::runtime_error("TT06Model::LoadState node count mismatch");
  }
  is.read(reinterpret_cast<char*>(states_.data()), static_cast<std::streamsize>(states_.size() * sizeof(double)));
  is.read(reinterpret_cast<char*>(rates_.data()), static_cast<std::streamsize>(rates_.size() * sizeof(double)));
  is.read(reinterpret_cast<char*>(constants_.data()), static_cast<std::streamsize>(constants_.size() * sizeof(double)));
  if (!is) {
    throw std::runtime_error("TT06Model::LoadState failed to read state data");
  }
}

}  // namespace mono
