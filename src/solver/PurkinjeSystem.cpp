#include "solver/PurkinjeSystem.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace mono {
namespace {

std::string Trim(std::string s) {
  const auto not_space = [](unsigned char c) { return !std::isspace(c); };
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
  s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
  return s;
}

bool IsFinite(double v) {
  return std::isfinite(v) != 0;
}

std::string NextDataLine(std::istream& in, int& line_no) {
  std::string line;
  while (std::getline(in, line)) {
    ++line_no;
    const auto hash = line.find('#');
    if (hash != std::string::npos) {
      line = line.substr(0, hash);
    }
    line = Trim(line);
    if (!line.empty()) {
      return line;
    }
  }
  return "";
}

}  // namespace

PurkinjeSystem::PurkinjeSystem(const SimulationConfig& cfg, const mfem::ParFiniteElementSpace& pfes)
    : cfg_(cfg), comm_(pfes.GetComm()) {
  MPI_Comm_rank(comm_, &rank_);
  MPI_Comm_size(comm_, &world_size_);

  LoadNetwork(cfg_.purkinje_network_path);
  BuildAdjacency();
  BuildPvjMapping(pfes);

  vp_.SetSize(n_nodes_);
  dvdt_.SetSize(n_nodes_);
  node_vm_local_.SetSize(n_nodes_);
  node_vm_global_.SetSize(n_nodes_);

  stim_nodes_ = cfg_.purkinje_stim_nodes;
  if (stim_nodes_.empty() && n_nodes_ > 0) {
    stim_nodes_.push_back(0);
  }
  for (const int node : stim_nodes_) {
    if (node < 0 || node >= n_nodes_) {
      throw std::runtime_error("purkinje_stim_nodes contains out-of-range node index");
    }
  }
}

void PurkinjeSystem::Initialize(double v_init_mv) {
  vp_ = v_init_mv;
}

void PurkinjeSystem::LoadNetwork(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Cannot open Purkinje network file: " + path);
  }

  int line_no = 0;
  const std::string header = NextDataLine(in, line_no);
  if (header.empty()) {
    throw std::runtime_error("Purkinje network file is empty: " + path);
  }

  std::istringstream hs(header);
  int n_edges = 0;
  hs >> n_nodes_ >> n_edges;
  if (!hs || n_nodes_ <= 0 || n_edges < 0) {
    throw std::runtime_error("Purkinje header must be: <num_nodes> <num_edges>");
  }

  nodes_.assign(static_cast<size_t>(n_nodes_), Node{});
  edges_.clear();
  edge_weight_.clear();
  edges_.reserve(static_cast<size_t>(n_edges));
  edge_weight_.reserve(static_cast<size_t>(n_edges));

  for (int i = 0; i < n_nodes_; ++i) {
    const std::string line = NextDataLine(in, line_no);
    if (line.empty()) {
      throw std::runtime_error("Purkinje file ended while reading node list");
    }
    std::istringstream ls(line);
    Node node;
    int terminal = 0;
    ls >> node.x >> node.y >> node.z >> terminal;
    if (!ls) {
      throw std::runtime_error("Invalid Purkinje node line at input line " + std::to_string(line_no));
    }
    node.terminal = (terminal != 0);
    nodes_[static_cast<size_t>(i)] = node;
  }

  for (int e = 0; e < n_edges; ++e) {
    const std::string line = NextDataLine(in, line_no);
    if (line.empty()) {
      throw std::runtime_error("Purkinje file ended while reading edge list");
    }

    std::istringstream ls(line);
    int a = -1;
    int b = -1;
    double w = -1.0;
    ls >> a >> b;
    if (!ls) {
      throw std::runtime_error("Invalid Purkinje edge line at input line " + std::to_string(line_no));
    }
    if (ls >> w) {
      // Optional edge weight provided.
    } else {
      w = -1.0;
    }

    if (a < 0 || a >= n_nodes_ || b < 0 || b >= n_nodes_ || a == b) {
      throw std::runtime_error("Purkinje edge has invalid node indices at input line " +
                               std::to_string(line_no));
    }
    edges_.emplace_back(a, b);
    edge_weight_.push_back(w);
  }
}

void PurkinjeSystem::BuildAdjacency() {
  adjacency_.clear();
  adjacency_.resize(static_cast<size_t>(n_nodes_));

  for (size_t e = 0; e < edges_.size(); ++e) {
    const int a = edges_[e].first;
    const int b = edges_[e].second;

    double w = edge_weight_[e];
    if (!(w > 0.0)) {
      const Node& na = nodes_[static_cast<size_t>(a)];
      const Node& nb = nodes_[static_cast<size_t>(b)];
      const double dx = na.x - nb.x;
      const double dy = na.y - nb.y;
      const double dz = na.z - nb.z;
      const double len = std::sqrt(dx * dx + dy * dy + dz * dz);
      if (!(len > 0.0) || !IsFinite(len)) {
        throw std::runtime_error("Purkinje edge length must be finite and > 0");
      }
      w = 1.0 / len;
    }

    const double g = cfg_.purkinje_edge_g_mS * w;
    adjacency_[static_cast<size_t>(a)].push_back({b, g});
    adjacency_[static_cast<size_t>(b)].push_back({a, g});
  }
}

void PurkinjeSystem::BuildPvjMapping(const mfem::ParFiniteElementSpace& pfes) {
  mapped_terminal_owner_.assign(static_cast<size_t>(n_nodes_), -1);
  mapped_terminal_dist_mm_.assign(static_cast<size_t>(n_nodes_), std::numeric_limits<double>::infinity());
  local_pvj_links_.clear();

  const mfem::ParMesh* pmesh = pfes.GetParMesh();
  if (pmesh == nullptr) {
    throw std::runtime_error("PurkinjeSystem requires valid ParMesh");
  }

  mfem::Array<int> vdofs;
  const int nv = pmesh->GetNV();

  for (int node = 0; node < n_nodes_; ++node) {
    if (!nodes_[static_cast<size_t>(node)].terminal) {
      continue;
    }

    double local_best_dist2 = std::numeric_limits<double>::infinity();
    int local_best_tdof = -1;

    const Node& n = nodes_[static_cast<size_t>(node)];
    for (int v = 0; v < nv; ++v) {
      pfes.GetVertexDofs(v, vdofs);
      if (vdofs.Size() < 1) {
        continue;
      }

      const int ldof = (vdofs[0] >= 0) ? vdofs[0] : (-1 - vdofs[0]);
      const int tdof = pfes.GetLocalTDofNumber(ldof);
      if (tdof < 0) {
        continue;
      }

      const double* x = pmesh->GetVertex(v);
      const double dx = x[0] - n.x;
      const double dy = x[1] - n.y;
      const double dz = x[2] - n.z;
      const double d2 = dx * dx + dy * dy + dz * dz;
      if (d2 < local_best_dist2) {
        local_best_dist2 = d2;
        local_best_tdof = tdof;
      }
    }

    struct {
      double dist2;
      int rank;
    } in{local_best_dist2, rank_}, out{0.0, -1};

    MPI_Allreduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MINLOC, comm_);

    const bool has_global_candidate = IsFinite(out.dist2);
    const double best_dist_mm = has_global_candidate ? std::sqrt(std::max(out.dist2, 0.0))
                                                     : std::numeric_limits<double>::infinity();
    if (!has_global_candidate || best_dist_mm > cfg_.pvj_max_dist_mm) {
      continue;
    }

    mapped_terminal_owner_[static_cast<size_t>(node)] = out.rank;
    mapped_terminal_dist_mm_[static_cast<size_t>(node)] = best_dist_mm;

    if (rank_ == out.rank && local_best_tdof >= 0) {
      local_pvj_links_.push_back({node, local_best_tdof});
    }
  }

  int local_count = static_cast<int>(local_pvj_links_.size());
  MPI_Allreduce(&local_count, &global_num_mapped_pvj_, 1, MPI_INT, MPI_SUM, comm_);

  double local_max_dist = 0.0;
  for (const auto& link : local_pvj_links_) {
    const double d = mapped_terminal_dist_mm_[static_cast<size_t>(link.node)];
    local_max_dist = std::max(local_max_dist, d);
  }
  MPI_Allreduce(&local_max_dist, &global_max_mapped_pvj_dist_mm_, 1, MPI_DOUBLE, MPI_MAX, comm_);
}

void PurkinjeSystem::BuildNodeVmSamples(const mfem::Vector& vm_true, mfem::Vector& node_vm) const {
  if (vm_true.Size() <= 0) {
    throw std::runtime_error("PurkinjeSystem::BuildNodeVmSamples requires vm_true");
  }

  node_vm_local_ = 0.0;
  for (const auto& link : local_pvj_links_) {
    if (link.tdof < 0 || link.tdof >= vm_true.Size()) {
      continue;
    }
    node_vm_local_[link.node] = vm_true[link.tdof];
  }

  node_vm_global_.SetSize(n_nodes_);
  MPI_Allreduce(node_vm_local_.GetData(), node_vm_global_.GetData(), n_nodes_, MPI_DOUBLE, MPI_SUM,
                comm_);
  node_vm = node_vm_global_;
}

void PurkinjeSystem::BuildHeartCouplingCurrent(const mfem::Vector& vm_true,
                                               mfem::Vector& heart_current_true) const {
  if (heart_current_true.Size() != vm_true.Size()) {
    heart_current_true.SetSize(vm_true.Size());
  }
  heart_current_true = 0.0;

  if (cfg_.pvj_g_mS <= 0.0 || local_pvj_links_.empty()) {
    return;
  }

  const double g = cfg_.pvj_current_scale * cfg_.pvj_g_mS;
  for (const auto& link : local_pvj_links_) {
    if (link.tdof < 0 || link.tdof >= vm_true.Size()) {
      continue;
    }
    const double dvm = std::clamp(vp_[link.node] - vm_true[link.tdof], -200.0, 200.0);
    const double i_p_to_m = g * dvm;
    // Monodomain convention uses negative I_stim as depolarizing.
    heart_current_true[link.tdof] += -i_p_to_m;
  }
}

void PurkinjeSystem::Advance(double dt_ms, const mfem::Vector& vm_true, double t_mid_ms) {
  if (dt_ms <= 0.0 || n_nodes_ <= 0) {
    return;
  }

  BuildNodeVmSamples(vm_true, node_vm_global_);

  const int n_sub = static_cast<int>(std::ceil(dt_ms / cfg_.purkinje_dt_ms));
  const double h = dt_ms / static_cast<double>(std::max(n_sub, 1));
  const bool stim_on = (t_mid_ms >= cfg_.purkinje_stim_start_ms) && (t_mid_ms <= cfg_.purkinje_stim_end_ms) &&
                       (cfg_.purkinje_stim_amp != 0.0);

  std::vector<uint8_t> stim_mask(static_cast<size_t>(n_nodes_), 0);
  if (stim_on) {
    for (const int i : stim_nodes_) {
      stim_mask[static_cast<size_t>(i)] = 1;
    }
  }

  for (int sub = 0; sub < n_sub; ++sub) {
    for (int i = 0; i < n_nodes_; ++i) {
      double rhs = 0.0;

      rhs += -cfg_.purkinje_leak_g_mS * (vp_[i] - cfg_.purkinje_rest_mv);

      if (stim_mask[static_cast<size_t>(i)] != 0) {
        rhs += cfg_.purkinje_stim_amp;
      }

      for (const auto& nbr : adjacency_[static_cast<size_t>(i)]) {
        rhs += nbr.second * (vp_[nbr.first] - vp_[i]);
      }

      if (mapped_terminal_owner_[static_cast<size_t>(i)] >= 0 && cfg_.pvj_g_mS > 0.0) {
        const double dvm = std::clamp(vp_[i] - node_vm_global_[i], -200.0, 200.0);
        rhs += -cfg_.pvj_g_mS * dvm;
      }

      dvdt_[i] = rhs / cfg_.purkinje_cm_uF_per_mm;
    }

    vp_.Add(h, dvdt_);
    for (int i = 0; i < n_nodes_; ++i) {
      if (!IsFinite(vp_[i])) {
        throw std::runtime_error("PurkinjeSystem produced non-finite voltage");
      }
      vp_[i] = std::clamp(vp_[i], -120.0, 60.0);
    }
  }

  // Optional diagnostic: set PURKINJE_DEBUG=1 to print Vp summary statistics
  // and the first three nodes' adjacency every 20 calls. Useful for tuning
  // purkinje_edge_g_mS / pvj_g_mS for a new mesh.
  static int log_call = 0;
  static const char* dbg_env = std::getenv("PURKINJE_DEBUG");
  if (dbg_env && *dbg_env == '1') {
    log_call += 1;
    if (log_call == 1 && rank_ == 0) {
      for (int probe : {0, 1, 2}) {
        if (probe >= n_nodes_) break;
        std::cout << "[purkinje:adj] node " << probe << " has "
                  << adjacency_[probe].size() << " neighbors:";
        for (const auto& nbr : adjacency_[probe]) {
          std::cout << " (" << nbr.first << ", g=" << nbr.second << ")";
        }
        std::cout << "\n";
      }
      std::cout << "[purkinje:adj] edges=" << edges_.size() << ", nodes=" << n_nodes_ << "\n";
    }
    if (log_call % 20 == 1 && rank_ == 0) {
      double vp_min = vp_[0], vp_max = vp_[0];
      int n_above_50 = 0, n_above_zero = 0;
      for (int i = 0; i < n_nodes_; ++i) {
        if (vp_[i] < vp_min) vp_min = vp_[i];
        if (vp_[i] > vp_max) vp_max = vp_[i];
        if (vp_[i] > -50.0) ++n_above_50;
        if (vp_[i] > 0.0) ++n_above_zero;
      }
      std::cout << "[purkinje@t=" << t_mid_ms << "] Vp=[" << vp_min << ", "
                << vp_max << "] n>-50=" << n_above_50 << " n>0=" << n_above_zero
                << "/" << n_nodes_ << " stim_on=" << (stim_on ? 1 : 0) << "\n";
    }
  }
}

}  // namespace mono
