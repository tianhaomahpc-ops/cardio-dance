#pragma once

#include <string>
#include <utility>
#include <vector>

#include "mfem.hpp"

#include "config/SimulationConfig.hpp"

namespace mono {

// Lightweight 1D Purkinje graph with terminal PVJ links into myocardial true dofs.
class PurkinjeSystem {
 public:
  PurkinjeSystem(const SimulationConfig& cfg, const mfem::ParFiniteElementSpace& pfes);

  void Initialize(double v_init_mv);

  // Build equivalent myocardial stimulus current at true dofs from current Purkinje voltage.
  void BuildHeartCouplingCurrent(const mfem::Vector& vm_true, mfem::Vector& heart_current_true) const;

  // Advance Purkinje voltages over one PDE step using explicit substepping.
  void Advance(double dt_ms, const mfem::Vector& vm_true, double t_mid_ms);

  const mfem::Vector& Vm() const { return vp_; }
  int NumNodes() const { return n_nodes_; }
  int GlobalNumMappedPvj() const { return global_num_mapped_pvj_; }
  double GlobalMaxMappedPvjDistMm() const { return global_max_mapped_pvj_dist_mm_; }

 private:
  struct Node {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    bool terminal = false;
  };

  struct LocalPvjLink {
    int node = -1;
    int tdof = -1;
  };

  void LoadNetwork(const std::string& path);
  void BuildAdjacency();
  void BuildPvjMapping(const mfem::ParFiniteElementSpace& pfes);
  void BuildNodeVmSamples(const mfem::Vector& vm_true, mfem::Vector& node_vm) const;

  const SimulationConfig& cfg_;
  MPI_Comm comm_;
  int rank_ = 0;
  int world_size_ = 1;

  int n_nodes_ = 0;
  std::vector<Node> nodes_;
  std::vector<std::pair<int, int>> edges_;
  std::vector<double> edge_weight_;
  std::vector<std::vector<std::pair<int, double>>> adjacency_;

  std::vector<LocalPvjLink> local_pvj_links_;
  std::vector<int> mapped_terminal_owner_;
  std::vector<double> mapped_terminal_dist_mm_;
  int global_num_mapped_pvj_ = 0;
  double global_max_mapped_pvj_dist_mm_ = 0.0;

  std::vector<int> stim_nodes_;

  mfem::Vector vp_;
  mfem::Vector dvdt_;
  mutable mfem::Vector node_vm_local_;
  mutable mfem::Vector node_vm_global_;
};

}  // namespace mono
