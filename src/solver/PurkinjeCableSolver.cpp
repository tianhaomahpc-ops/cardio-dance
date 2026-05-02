#include "solver/PurkinjeCableSolver.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace mono {
namespace {

double SegmentLength(const PurkinjeCableSolver::Node& a,
                     const PurkinjeCableSolver::Node& b) {
  const double dx = a.x - b.x;
  const double dy = a.y - b.y;
  const double dz = a.z - b.z;
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

}  // namespace

PurkinjeCableSolver::PurkinjeCableSolver(const SimulationConfig& cfg, IIonicModel& ionic)
    : cfg_(cfg), ionic_(ionic) {}

void PurkinjeCableSolver::LoadNetwork(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("PurkinjeCableSolver: cannot open " + path);
  }
  int n_nodes = 0;
  int n_edges = 0;
  in >> n_nodes >> n_edges;
  if (!in || n_nodes <= 0 || n_edges < 0) {
    throw std::runtime_error("PurkinjeCableSolver: invalid header in " + path);
  }
  graph_nodes_.resize(static_cast<size_t>(n_nodes));
  graph_edges_.clear();
  graph_edges_.reserve(static_cast<size_t>(n_edges));
  graph_edge_g_mS_.assign(static_cast<size_t>(n_edges), 0.0);

  std::string rest_line;
  std::getline(in, rest_line);

  for (int i = 0; i < n_nodes; ++i) {
    std::string line;
    if (!std::getline(in, line)) {
      throw std::runtime_error("PurkinjeCableSolver: missing node line " +
                               std::to_string(i));
    }
    std::istringstream iss(line);
    int term = 0;
    iss >> graph_nodes_[i].x >> graph_nodes_[i].y >> graph_nodes_[i].z >> term;
    if (!iss) {
      throw std::runtime_error("PurkinjeCableSolver: bad node line " +
                               std::to_string(i));
    }
    graph_nodes_[i].terminal = (term != 0);
  }

  for (int e = 0; e < n_edges; ++e) {
    std::string line;
    if (!std::getline(in, line)) {
      throw std::runtime_error("PurkinjeCableSolver: missing edge line " +
                               std::to_string(e));
    }
    std::istringstream iss(line);
    int a = 0, b = 0;
    iss >> a >> b;
    if (!iss || a < 0 || a >= n_nodes || b < 0 || b >= n_nodes || a == b) {
      throw std::runtime_error("PurkinjeCableSolver: bad edge line " +
                               std::to_string(e));
    }
    double g = 0.0;
    if (iss >> g) {
      graph_edge_g_mS_[e] = g;
    } else {
      graph_edge_g_mS_[e] = -1.0;
    }
    graph_edges_.emplace_back(a, b);
  }

  terminal_graph_indices_.clear();
  for (int i = 0; i < n_nodes; ++i) {
    if (graph_nodes_[i].terminal) {
      terminal_graph_indices_.push_back(i);
    }
  }
}

void PurkinjeCableSolver::BuildFeNodes() {
  const int n_graph = static_cast<int>(graph_nodes_.size());
  const int subdivision = std::max(1, cfg_.purkinje_cable_subdivision);

  graph_to_fe_.assign(static_cast<size_t>(n_graph), -1);
  fe_nodes_.clear();
  fe_adj_.clear();

  for (int g = 0; g < n_graph; ++g) {
    fe_nodes_.push_back(graph_nodes_[g]);
    graph_to_fe_[g] = static_cast<int>(fe_nodes_.size()) - 1;
  }
  fe_adj_.assign(fe_nodes_.size(), {});

  for (size_t e = 0; e < graph_edges_.size(); ++e) {
    const int a = graph_edges_[e].first;
    const int b = graph_edges_[e].second;
    const double total_len = SegmentLength(graph_nodes_[a], graph_nodes_[b]);
    const double seg_len = total_len / static_cast<double>(subdivision);

    double g_axial_per_mm = graph_edge_g_mS_[e];
    if (g_axial_per_mm <= 0.0) {
      g_axial_per_mm = cfg_.purkinje_edge_g_mS_per_mm;
    }
    const double g_seg = g_axial_per_mm / std::max(seg_len, 1e-9);

    int prev_fe = graph_to_fe_[a];
    for (int s = 1; s < subdivision; ++s) {
      Node mid;
      const double t = static_cast<double>(s) / static_cast<double>(subdivision);
      mid.x = graph_nodes_[a].x + t * (graph_nodes_[b].x - graph_nodes_[a].x);
      mid.y = graph_nodes_[a].y + t * (graph_nodes_[b].y - graph_nodes_[a].y);
      mid.z = graph_nodes_[a].z + t * (graph_nodes_[b].z - graph_nodes_[a].z);
      mid.terminal = false;
      fe_nodes_.push_back(mid);
      fe_adj_.push_back({});
      const int cur_fe = static_cast<int>(fe_nodes_.size()) - 1;
      fe_adj_[prev_fe].push_back({cur_fe, seg_len, g_seg});
      fe_adj_[cur_fe].push_back({prev_fe, seg_len, g_seg});
      prev_fe = cur_fe;
    }
    const int last_fe = graph_to_fe_[b];
    fe_adj_[prev_fe].push_back({last_fe, seg_len, g_seg});
    fe_adj_[last_fe].push_back({prev_fe, seg_len, g_seg});
  }

  n_fe_nodes_ = static_cast<int>(fe_nodes_.size());
}

void PurkinjeCableSolver::BuildSparseMatrices() {
  // Lumped mass: each FE node accumulates half the length of incident segments
  // multiplied by Cm.
  mass_lumped_.SetSize(n_fe_nodes_);
  mass_lumped_ = 0.0;
  // Stiffness graph: K(i,i) = sum_j g_ij; K(i,j) = -g_ij.
  K_p_ = mfem::SparseMatrix(n_fe_nodes_, n_fe_nodes_);

  for (int i = 0; i < n_fe_nodes_; ++i) {
    double diag = 0.0;
    for (const auto& con : fe_adj_[i]) {
      mass_lumped_[i] += 0.5 * con.length_mm * cfg_.purkinje_cm_uF_per_mm2;
      diag += con.g_axial;
      // Off-diagonal added once per edge end; SparseMatrix Add may insert
      // duplicates, so we'll Finalize below to compress.
      if (con.neighbor != i) {
        K_p_.Add(i, con.neighbor, -con.g_axial);
      }
    }
    K_p_.Add(i, i, diag);
  }
  K_p_.Finalize();

  cached_dt_ms_ = -1.0;
  A_p_.reset();
  B_p_.reset();
  cg_.reset();
  prec_.reset();
}

void PurkinjeCableSolver::Initialize(double v_init_mv) {
  if (graph_nodes_.empty()) {
    throw std::runtime_error("PurkinjeCableSolver::Initialize called before LoadNetwork");
  }
  BuildFeNodes();
  if (ionic_.NumNodes() != n_fe_nodes_) {
    throw std::runtime_error(
        "PurkinjeCableSolver: ionic model FE-node count mismatch (got " +
        std::to_string(ionic_.NumNodes()) + ", expected " +
        std::to_string(n_fe_nodes_) + ")");
  }
  BuildSparseMatrices();

  vp_.SetSize(n_fe_nodes_);
  iion_p_.SetSize(n_fe_nodes_);
  i_ext_p_.SetSize(n_fe_nodes_);
  rhs_p_.SetSize(n_fe_nodes_);
  tmp_p_.SetSize(n_fe_nodes_);

  vp_ = v_init_mv;
  iion_p_ = 0.0;
  i_ext_p_ = 0.0;
  ionic_.InitializeRestState(v_init_mv);

  // Resolve graph-level stim nodes to FE-node indices for fast pacing.
  stim_fe_nodes_.clear();
  stim_fe_nodes_.reserve(cfg_.purkinje_stim_nodes.size());
  const int n_graph = static_cast<int>(graph_nodes_.size());
  for (int gnode : cfg_.purkinje_stim_nodes) {
    if (gnode < 0 || gnode >= n_graph) {
      throw std::runtime_error("purkinje_stim_nodes contains out-of-range index");
    }
    stim_fe_nodes_.push_back(graph_to_fe_[gnode]);
  }
}

void PurkinjeCableSolver::ApplyDiffusionStep(double dt) {
  // Build (M + 0.5*dt*K) and (M - 0.5*dt*K) once per dt.
  if (cached_dt_ms_ != dt || !A_p_) {
    A_p_ = std::make_unique<mfem::SparseMatrix>(K_p_);
    *A_p_ *= 0.5 * dt;
    B_p_ = std::make_unique<mfem::SparseMatrix>(*A_p_);
    *B_p_ *= -1.0;
    // Add lumped mass to diagonal of A and B.
    for (int i = 0; i < n_fe_nodes_; ++i) {
      A_p_->Set(i, i, A_p_->Elem(i, i) + mass_lumped_[i]);
      B_p_->Set(i, i, B_p_->Elem(i, i) + mass_lumped_[i]);
    }
    A_p_->Finalize();
    B_p_->Finalize();

    prec_ = std::make_unique<mfem::DSmoother>(*A_p_);
    cg_ = std::make_unique<mfem::CGSolver>();
    cg_->SetOperator(*A_p_);
    cg_->SetPreconditioner(*prec_);
    cg_->SetMaxIter(500);
    cg_->SetRelTol(1e-9);
    cg_->SetAbsTol(0.0);
    cg_->SetPrintLevel(-1);
    cached_dt_ms_ = dt;
  }

  B_p_->Mult(vp_, rhs_p_);
  // Reaction: subtract dt * M * (I_ion + I_ext) from RHS.
  for (int i = 0; i < n_fe_nodes_; ++i) {
    tmp_p_[i] = mass_lumped_[i] * (iion_p_[i] + i_ext_p_[i]);
  }
  rhs_p_.Add(-dt, tmp_p_);

  cg_->Mult(rhs_p_, vp_);
}

void PurkinjeCableSolver::Advance(double dt_pde_ms, double t_mid_ms,
                                  const std::vector<double>& myocardial_vm_at_terms) {
  if (n_fe_nodes_ == 0) return;

  const double pdt = dt_pde_ms;
  const double pdt_sub = std::min(pdt, std::max(cfg_.purkinje_dt_ms, 1e-6));
  const int n_sub = static_cast<int>(std::ceil(pdt / pdt_sub));
  const double dt = pdt / static_cast<double>(n_sub);

  // Build PVJ injection currents on the cable side (Purkinje sees i_pvj_p =
  // pvj_g * (V_p - V_m) at terminals; positive when Purkinje more depolarized
  // -- this is repolarizing, so I_ion semantics positive.)
  std::vector<double> i_pvj_p(n_fe_nodes_, 0.0);
  if (!myocardial_vm_at_terms.empty()) {
    const int nt = static_cast<int>(terminal_graph_indices_.size());
    for (int t = 0; t < nt; ++t) {
      const int fe = graph_to_fe_[terminal_graph_indices_[t]];
      const double Vm = myocardial_vm_at_terms[t];
      i_pvj_p[fe] = cfg_.pvj_g_mS * cfg_.pvj_current_scale * (vp_[fe] - Vm);
    }
  }

  for (int sub = 0; sub < n_sub; ++sub) {
    // 1) compute I_ion at current Vp.
    ionic_.ComputeIion(vp_, iion_p_);

    // 2) update gating/state using current Vp (mirrors monodomain order).
    ionic_.AdvanceStates(dt, dt, vp_);

    // 3) build external current (PVJ + direct stimulus on root nodes).
    i_ext_p_ = 0.0;
    for (int i = 0; i < n_fe_nodes_; ++i) i_ext_p_[i] += i_pvj_p[i];
    if (t_mid_ms >= cfg_.purkinje_stim_start_ms &&
        t_mid_ms <= cfg_.purkinje_stim_end_ms &&
        cfg_.purkinje_stim_amp_uA_per_uF != 0.0) {
      for (int fe : stim_fe_nodes_) {
        i_ext_p_[fe] += cfg_.purkinje_stim_amp_uA_per_uF;
      }
    }

    // 4) Crank-Nicolson diffusion + reaction step.
    ApplyDiffusionStep(dt);
  }
}

void PurkinjeCableSolver::AddExternalCurrent(int fe_node, double i_uA_per_uF) {
  if (fe_node < 0 || fe_node >= n_fe_nodes_) return;
  i_ext_p_[fe_node] += i_uA_per_uF;
}

double PurkinjeCableSolver::TerminalVoltage(int t_idx) const {
  if (t_idx < 0 || t_idx >= static_cast<int>(terminal_graph_indices_.size())) {
    return 0.0;
  }
  const int fe = graph_to_fe_[terminal_graph_indices_[t_idx]];
  return vp_[fe];
}

double PurkinjeCableSolver::GraphNodeVoltage(int graph_idx) const {
  if (graph_idx < 0 || graph_idx >= static_cast<int>(graph_nodes_.size())) {
    return 0.0;
  }
  return vp_[graph_to_fe_[graph_idx]];
}

}  // namespace mono
