#include "ode/StewartPurkinjeModel.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <istream>
#include <ostream>
#include <stdexcept>

#include "ode/stewart_generated.h"

namespace mono {
namespace {

// Gate-state index, x_inf algebraic index, tau algebraic index.
// Source-of-truth: stewart_generated.c, the lines
//   RATES[state_idx] = (ALGEBRAIC[inf_idx] - STATES[state_idx])/ALGEBRAIC[tau_idx]
struct GateMap {
  int state_idx;
  int inf_idx;
  int tau_idx;
};

constexpr GateMap kGateMaps[] = {
    {4,  0, 37},   // y     (HCN/I_f)
    {5,  1, 38},   // Xr1   (I_Kr)
    {6,  2, 39},   // Xr2   (I_Kr)
    {7,  3, 40},   // Xs    (I_Ks)
    {8,  4, 41},   // m     (I_Na)
    {9,  5, 42},   // h     (I_Na)
    {10, 6, 43},   // j     (I_Na)
    {12, 7, 46},   // d     (I_CaL)
    {13, 8, 22},   // f     (I_CaL)
    {14, 9, 23},   // f2    (I_CaL)
    {15, 10, 24},  // fCass (I_CaL)
    {16, 11, 25},  // s     (I_to inactivation)
    {17, 12, 26},  // r     (I_to activation)
};

// Forward-Euler concentrations / RyR.
//   STATES[1]  = K_i,    [2] = Na_i,   [3] = Ca_i,
//   STATES[11] = Ca_ss,  [18]= Ca_SR,  [19]= R_prime
constexpr int kFeStateIdx[] = {1, 2, 3, 11, 18, 19};

// ALGEBRAIC indices that sum to I_ion (TT06 outward-positive convention).
// Source: stewart_generated.c RATES[0] = -1*(... sum of these 14 ...).
constexpr int kIionAlgIdx[] = {
    51,  // i_K1
    58,  // i_to
    60,  // i_sus  (Stewart-specific sustained outward, gated by ALGEBRAIC[59])
    52,  // i_Kr
    53,  // i_Ks
    56,  // i_CaL
    61,  // i_NaK
    54,  // i_Na
    55,  // i_b_Na
    62,  // i_NaCa
    57,  // i_b_Ca
    64,  // i_p_K
    63,  // i_p_Ca
    49,  // i_f  (already i_f_Na + i_f_K)
};

}  // namespace

StewartPurkinjeModel::StewartPurkinjeModel(int n_local_true_dofs)
    : n_nodes_(n_local_true_dofs),
      states_(static_cast<size_t>(n_local_true_dofs) * kNumStates, 0.0),
      rates_(static_cast<size_t>(n_local_true_dofs) * kNumRates, 0.0),
      constants_(static_cast<size_t>(n_local_true_dofs) * kNumConsts, 0.0) {
  if (n_nodes_ <= 0) {
    throw std::runtime_error("StewartPurkinjeModel requires n_local_true_dofs > 0");
  }
}

void StewartPurkinjeModel::InitializeRestState(double v_rest_mv) {
  for (int i = 0; i < n_nodes_; ++i) {
    double* c = ConstPtr(i);
    double* r = RatePtr(i);
    double* s = StatePtr(i);
    stewart_initConsts(c, r, s);

    // Codegen sets STATES[0] = -69.137 mV, which is mid-cycle for the
    // Stewart cell (HCN-driven slow drift). When the caller wants a clean
    // start at maximal diastolic potential, override here.
    if (std::isfinite(v_rest_mv)) {
      s[0] = v_rest_mv;
    }
  }
}

void StewartPurkinjeModel::ComputeIion(const mfem::Vector& vm_true,
                                       mfem::Vector& iion_true) const {
  if (vm_true.Size() != n_nodes_) {
    throw std::runtime_error("StewartPurkinjeModel::ComputeIion: size mismatch");
  }
  if (iion_true.Size() != n_nodes_) {
    iion_true.SetSize(n_nodes_);
  }

  // Stack temporaries to keep ComputeIion side-effect-free.
  double local_states[kNumStates];
  double local_rates[kNumRates];
  double local_alg[kNumAlg];

  for (int i = 0; i < n_nodes_; ++i) {
    const double* s_stored = StatePtr(i);
    const double* c = ConstPtr(i);

    std::memcpy(local_states, s_stored, sizeof(local_states));
    std::memcpy(local_rates,
                rates_.data() + static_cast<size_t>(i) * kNumRates,
                sizeof(local_rates));

    local_states[0] = vm_true[i];
    stewart_computeVariables(0.0, const_cast<double*>(c), local_rates,
                             local_states, local_alg);

    double iion = 0.0;
    for (const int idx : kIionAlgIdx) {
      iion += local_alg[idx];
    }
    iion_true[i] = iion;
  }
}

double StewartPurkinjeModel::RushLarsenUpdate(double x, double x_inf,
                                              double tau, double dt) {
  if (tau <= 1e-12 || !std::isfinite(tau)) {
    return x_inf;
  }
  const double expo = std::exp(-dt / tau);
  return x_inf - (x_inf - x) * expo;
}

void StewartPurkinjeModel::AdvanceStates(double dt_pde_ms, double dt_ode_ms,
                                         const mfem::Vector& vm_next_true) {
  if (vm_next_true.Size() != n_nodes_) {
    throw std::runtime_error("StewartPurkinjeModel::AdvanceStates: size mismatch");
  }
  if (dt_pde_ms <= 0.0 || dt_ode_ms <= 0.0) {
    throw std::runtime_error("StewartPurkinjeModel::AdvanceStates: dt must be > 0");
  }

  const int n_sub = static_cast<int>(std::ceil(dt_pde_ms / dt_ode_ms));
  const double dt = dt_pde_ms / static_cast<double>(n_sub);

  double local_alg[kNumAlg];

  for (int node = 0; node < n_nodes_; ++node) {
    double* s = StatePtr(node);
    double* r = RatePtr(node);
    double* c = ConstPtr(node);

    for (int sub = 0; sub < n_sub; ++sub) {
      s[0] = vm_next_true[node];
      stewart_computeRates((sub + 1) * dt, c, r, s, local_alg);

      // Rush-Larsen for 13 gates.
      for (const auto& gm : kGateMaps) {
        s[gm.state_idx] = RushLarsenUpdate(s[gm.state_idx],
                                            local_alg[gm.inf_idx],
                                            local_alg[gm.tau_idx], dt);
      }

      // Forward Euler for concentrations / RyR.
      for (const int idx : kFeStateIdx) {
        s[idx] += dt * r[idx];
      }

      // Positivity / range clamps so single-node solver doesn't blow up.
      s[1]  = std::max(s[1],  1e-3);            // K_i
      s[2]  = std::max(s[2],  1e-3);            // Na_i
      s[3]  = std::max(s[3],  1e-9);            // Ca_i
      s[11] = std::max(s[11], 1e-9);            // Ca_ss
      s[18] = std::max(s[18], 1e-9);            // Ca_SR
      s[19] = std::clamp(s[19], 0.0, 1.0);      // R_prime ∈ [0,1]
    }

    s[0] = vm_next_true[node];
  }
}

void StewartPurkinjeModel::SaveState(std::ostream& os) const {
  const int32_t n = n_nodes_;
  os.write(reinterpret_cast<const char*>(&n), sizeof(n));
  os.write(reinterpret_cast<const char*>(states_.data()),
           static_cast<std::streamsize>(states_.size() * sizeof(double)));
  os.write(reinterpret_cast<const char*>(rates_.data()),
           static_cast<std::streamsize>(rates_.size() * sizeof(double)));
  os.write(reinterpret_cast<const char*>(constants_.data()),
           static_cast<std::streamsize>(constants_.size() * sizeof(double)));
}

void StewartPurkinjeModel::LoadState(std::istream& is) {
  int32_t n = 0;
  is.read(reinterpret_cast<char*>(&n), sizeof(n));
  if (n != n_nodes_) {
    throw std::runtime_error("StewartPurkinjeModel::LoadState size mismatch");
  }
  is.read(reinterpret_cast<char*>(states_.data()),
          static_cast<std::streamsize>(states_.size() * sizeof(double)));
  is.read(reinterpret_cast<char*>(rates_.data()),
          static_cast<std::streamsize>(rates_.size() * sizeof(double)));
  is.read(reinterpret_cast<char*>(constants_.data()),
          static_cast<std::streamsize>(constants_.size() * sizeof(double)));
  if (!is) {
    throw std::runtime_error("StewartPurkinjeModel::LoadState read failed");
  }
}

}  // namespace mono
