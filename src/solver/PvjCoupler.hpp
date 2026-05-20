#pragma once

#include <vector>

#include "mfem.hpp"

#include "config/SimulationConfig.hpp"

namespace mono {

class PurkinjeCableSolver;

// Bidirectional gap-junction coupling between Purkinje terminal nodes and
// nearest-neighbor myocardial true DOFs. Owned by main.cpp; passed into
// MonodomainStepper as a non-owning pointer.
//
// Sign convention: the heart-side current uses the same I_ion semantics used
// elsewhere in the solver (positive outward). Coupling current at a mapped
// myocardial DOF is
//     i_pvj_heart = pvj_g * (V_m - V_p) * pvj_current_scale
// so when the Purkinje terminal is more depolarized than the myocardium, this
// term is negative (depolarizing the myocardium), matching the I_stim sign.
class PvjCoupler {
 public:
  PvjCoupler(const SimulationConfig& cfg,
             const mfem::ParFiniteElementSpace& pfes,
             PurkinjeCableSolver& cable);

  // Fill heart_current_true (size = local true DOFs) with i_pvj at mapped DOFs;
  // zero elsewhere. Reads V_m at the mapped tdofs from vm_true.
  void BuildHeartCouplingCurrent(const mfem::Vector& vm_true,
                                 mfem::Vector& heart_current_true) const;

  // Advance the embedded Purkinje cable solver across one PDE step. Samples
  // V_m at terminal mapping and updates the cable's view of myocardial state
  // (with optional pvj_delay_ms ring buffer).
  void AdvancePurkinje(double dt_pde_ms, const mfem::Vector& vm_true,
                       double t_mid_ms);

  PurkinjeCableSolver& Cable() { return cable_; }
  const PurkinjeCableSolver& Cable() const { return cable_; }

  int GlobalNumMappedPvj() const { return global_num_mapped_; }
  double GlobalMaxMappedDistMm() const { return global_max_dist_mm_; }

 private:
  struct LocalPvjLink {
    int purkinje_node = -1;
    int heart_tdof = -1;
    int owner_rank = -1;
    double dist_mm = 0.0;
    // Smear weight for the heart-side injection. When pvj_smear_radius_mm
    // is > 0, one terminal expands into multiple LocalPvjLink entries on
    // the owner rank, each carrying weight = 1/N_local_in_radius so the
    // per-terminal total injection magnitude is unchanged. The first link
    // for each terminal (with weight=anchor_weight) is also used by
    // AdvancePurkinje to sample V_m back to the cable.
    double weight = 1.0;
    bool is_anchor = true;
  };

  void BuildMapping(const mfem::ParFiniteElementSpace& pfes);
  void AppendDelayedSample(double t_mid_ms, const std::vector<double>& vm_at_terms);
  std::vector<double> ReadDelayedSample(double t_mid_ms) const;

  const SimulationConfig& cfg_;
  PurkinjeCableSolver& cable_;
  MPI_Comm comm_;
  int rank_ = 0;
  int world_size_ = 1;

  // For each terminal index in the cable's terminal list, the global mapping
  // owner rank and (if owner) the local heart tdof + distance.
  std::vector<int> terminal_owner_rank_;
  std::vector<double> terminal_dist_mm_;
  std::vector<LocalPvjLink> local_links_;

  int global_num_mapped_ = 0;
  double global_max_dist_mm_ = 0.0;

  // Optional delay buffer: ring of (t_ms, vm_at_terminals_global).
  struct DelaySample {
    double t_ms;
    std::vector<double> vm;
  };
  mutable std::vector<DelaySample> delay_buf_;
};

}  // namespace mono
