#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "mfem.hpp"

#include "config/SimulationConfig.hpp"
#include "ode/IonicModel.hpp"

namespace mono {

// 1D Purkinje cable on a graph of nodes connected by edges. Each edge is
// optionally subdivided into purkinje_cable_subdivision sub-segments, producing
// FE nodes per edge that are spliced together at branch points.
//
// Numerics: same operator-splitting as MonodomainStepper.
//   M_p dV/dt + K_p V = -M_p (I_ion + I_pvj + I_stim)
// where M_p, K_p are mass / stiffness matrices on the 1D graph network.
//
// Supplied an IonicModel managing the FE node count exactly once at construction.
class PurkinjeCableSolver {
 public:
  struct Node {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    bool terminal = false;
  };

  PurkinjeCableSolver(const SimulationConfig& cfg, IonicModel& ionic);

  // Load network from a text .network file:
  //   <num_nodes> <num_edges>
  //   <x> <y> <z> <terminal_flag>      (num_nodes lines)
  //   <i> <j> [edge_g_mS_per_mm]       (num_edges lines)
  // Edge weight, when omitted, defaults to cfg.purkinje_edge_g_mS divided by
  // segment length.
  void LoadNetwork(const std::string& path);

  // Allocate FE per-edge subdivision and initialize Vm to v_init_mv.
  void Initialize(double v_init_mv);

  // Advance over [t_mid - dt/2, t_mid + dt/2] using internal sub-steps.
  // myocardial_vm_at_terms is sampled at each terminal node so the cable feels
  // the bidirectional coupling current.
  void Advance(double dt_pde_ms, double t_mid_ms,
               const std::vector<double>& myocardial_vm_at_terms);

  // Inject a per-FE-node external current (e.g., direct His-bundle pacing).
  // The current is added in the next reaction step; cleared after each Advance.
  void AddExternalCurrent(int fe_node, double i_uA_per_uF);

  // Lookup Purkinje voltage at terminal index t_idx (0-based); returns the FE
  // node value associated with that graph terminal node.
  double TerminalVoltage(int t_idx) const;

  // Voltage averaged over the FE nodes belonging to a graph node (1 for non-
  // branch, may be >1 at branch points).
  double GraphNodeVoltage(int graph_idx) const;

  int NumGraphNodes() const { return static_cast<int>(graph_nodes_.size()); }
  int NumEdges() const { return static_cast<int>(graph_edges_.size()); }
  int NumFeNodes() const { return n_fe_nodes_; }
  int NumTerminals() const { return static_cast<int>(terminal_graph_indices_.size()); }
  int TerminalGraphIndex(int t_idx) const { return terminal_graph_indices_.at(t_idx); }
  const Node& GraphNode(int idx) const { return graph_nodes_.at(idx); }

  const mfem::Vector& Vm() const { return vp_; }

 private:
  // FE-side connectivity: for each FE node, its Cartesian coordinate and the
  // list of (neighbor_fe_node, segment_length_mm, axial_g_mS_per_mm).
  struct FeConnection {
    int neighbor = -1;
    double length_mm = 0.0;
    double g_axial = 0.0;  // per-edge axial conductance contribution
  };

  void BuildFeNodes();
  void BuildSparseMatrices();
  void ApplyDiffusionStep(double dt);

  const SimulationConfig& cfg_;
  IonicModel& ionic_;

  std::vector<Node> graph_nodes_;
  std::vector<std::pair<int, int>> graph_edges_;
  std::vector<double> graph_edge_g_mS_;
  std::vector<int> terminal_graph_indices_;

  // Each graph node maps to one FE node index (graph_to_fe_).
  std::vector<int> graph_to_fe_;
  // For each FE node, the originating coordinate (used for diagnostics).
  std::vector<Node> fe_nodes_;
  // For each FE node, its 1D adjacency list of segment connections.
  std::vector<std::vector<FeConnection>> fe_adj_;
  int n_fe_nodes_ = 0;

  // Sparse mass and stiffness on the FE graph; we use lumped mass for speed.
  mfem::SparseMatrix K_p_;
  mfem::Vector mass_lumped_;  // diagonal mass entries (mm)

  // Crank-Nicolson amplification systems for diffusion step.
  std::unique_ptr<mfem::SparseMatrix> A_p_;  // (M + 0.5*dt*K)
  std::unique_ptr<mfem::SparseMatrix> B_p_;  // (M - 0.5*dt*K)
  double cached_dt_ms_ = -1.0;
  std::unique_ptr<mfem::CGSolver> cg_;
  std::unique_ptr<mfem::DSmoother> prec_;

  mfem::Vector vp_;
  mfem::Vector iion_p_;
  mfem::Vector i_ext_p_;          // external (PVJ + stim) currents per FE node
  mfem::Vector rhs_p_;
  mfem::Vector tmp_p_;

  std::vector<int> stim_fe_nodes_;
};

}  // namespace mono
