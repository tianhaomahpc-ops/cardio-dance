#include "solver/PvjCoupler.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "solver/PurkinjeCableSolver.hpp"

namespace mono {
namespace {

void GatherTrueDofCoordinates(const mfem::ParFiniteElementSpace& pfes,
                              std::vector<double>& x,
                              std::vector<double>& y,
                              std::vector<double>& z) {
  const int n = pfes.GetTrueVSize();
  x.assign(n, 0.0);
  y.assign(n, 0.0);
  z.assign(n, 0.0);
  mfem::ParGridFunction coord_gf(const_cast<mfem::ParFiniteElementSpace*>(&pfes));
  mfem::ParMesh* pmesh = pfes.GetParMesh();

  // Project coordinate components into a P1 grid function. P1 element ordering
  // matches true DOFs after GetTrueDofs.
  mfem::Vector tdof(n);
  for (int comp = 0; comp < 3; ++comp) {
    mfem::FunctionCoefficient coef(
        [comp](const mfem::Vector& p) { return p[comp]; });
    coord_gf.ProjectCoefficient(coef);
    coord_gf.GetTrueDofs(tdof);
    auto& dst = (comp == 0) ? x : (comp == 1) ? y : z;
    for (int i = 0; i < n; ++i) dst[i] = tdof[i];
  }
  (void)pmesh;
}

}  // namespace

PvjCoupler::PvjCoupler(const SimulationConfig& cfg,
                       const mfem::ParFiniteElementSpace& pfes,
                       PurkinjeCableSolver& cable)
    : cfg_(cfg), cable_(cable) {
  comm_ = pfes.GetComm();
  MPI_Comm_rank(comm_, &rank_);
  MPI_Comm_size(comm_, &world_size_);
  BuildMapping(pfes);
}

void PvjCoupler::BuildMapping(const mfem::ParFiniteElementSpace& pfes) {
  const int n_term = cable_.NumTerminals();
  if (n_term == 0) {
    global_num_mapped_ = 0;
    global_max_dist_mm_ = 0.0;
    return;
  }

  std::vector<double> tx, ty, tz;
  GatherTrueDofCoordinates(pfes, tx, ty, tz);
  const int n_local_tdof = static_cast<int>(tx.size());

  terminal_owner_rank_.assign(n_term, -1);
  terminal_dist_mm_.assign(n_term, std::numeric_limits<double>::infinity());

  // Per-terminal: find the closest local true DOF; then Allreduce the (dist,
  // rank) pair so that the global owner rank is determined.
  std::vector<double> local_dist(n_term, std::numeric_limits<double>::infinity());
  std::vector<int> local_tdof(n_term, -1);

  for (int t = 0; t < n_term; ++t) {
    const int g = cable_.TerminalGraphIndex(t);
    const auto& gn = cable_.GraphNode(g);
    double best = std::numeric_limits<double>::infinity();
    int best_tdof = -1;
    for (int i = 0; i < n_local_tdof; ++i) {
      const double dx = tx[i] - gn.x;
      const double dy = ty[i] - gn.y;
      const double dz = tz[i] - gn.z;
      const double d2 = dx * dx + dy * dy + dz * dz;
      if (d2 < best) {
        best = d2;
        best_tdof = i;
      }
    }
    local_dist[t] = (best_tdof >= 0) ? std::sqrt(best) : std::numeric_limits<double>::infinity();
    local_tdof[t] = best_tdof;
  }

  // Reduce min distance across ranks.
  struct DistRank { double d; int r; };
  std::vector<DistRank> in(n_term);
  std::vector<DistRank> out(n_term);
  for (int t = 0; t < n_term; ++t) {
    in[t].d = local_dist[t];
    in[t].r = (local_tdof[t] >= 0) ? rank_ : world_size_;
  }
  MPI_Allreduce(in.data(), out.data(), n_term, MPI_DOUBLE_INT, MPI_MINLOC, comm_);

  local_links_.clear();
  global_num_mapped_ = 0;
  global_max_dist_mm_ = 0.0;
  const double smear_r = cfg_.pvj_smear_radius_mm;
  const double smear_r2 = smear_r * smear_r;
  int n_smear_total = 0;

  for (int t = 0; t < n_term; ++t) {
    const double d = out[t].d;
    const int owner = out[t].r;
    terminal_dist_mm_[t] = d;
    terminal_owner_rank_[t] = owner;
    if (d > cfg_.pvj_max_dist_mm) continue;
    if (owner < 0 || owner >= world_size_) continue;
    ++global_num_mapped_;
    global_max_dist_mm_ = std::max(global_max_dist_mm_, d);

    // Anchor link: owner rank's nearest DOF, used by AdvancePurkinje for
    // V_m feedback to the cable.
    if (owner == rank_ && local_tdof[t] >= 0) {
      LocalPvjLink anchor;
      anchor.purkinje_node = t;
      anchor.heart_tdof = local_tdof[t];
      anchor.owner_rank = owner;
      anchor.dist_mm = d;
      anchor.weight = 1.0;
      anchor.is_anchor = true;

      if (smear_r <= 0.0) {
        local_links_.push_back(anchor);
      } else {
        // Scan owner's local DOFs within the smear ball around the
        // anchor coordinate, weighted uniformly so the per-terminal
        // total injection magnitude equals one anchor injection.
        const double ax = tx[anchor.heart_tdof];
        const double ay = ty[anchor.heart_tdof];
        const double az = tz[anchor.heart_tdof];
        std::vector<int> in_ball;
        for (int i = 0; i < n_local_tdof; ++i) {
          const double dx = tx[i] - ax;
          const double dy = ty[i] - ay;
          const double dz = tz[i] - az;
          if (dx * dx + dy * dy + dz * dz <= smear_r2) {
            in_ball.push_back(i);
          }
        }
        if (in_ball.empty()) {
          local_links_.push_back(anchor);
        } else {
          const double w = 1.0 / static_cast<double>(in_ball.size());
          n_smear_total += static_cast<int>(in_ball.size());
          for (size_t k = 0; k < in_ball.size(); ++k) {
            LocalPvjLink l;
            l.purkinje_node = t;
            l.heart_tdof = in_ball[k];
            l.owner_rank = owner;
            l.dist_mm = d;
            l.weight = w;
            // The first link kept marks the anchor used by AdvancePurkinje.
            l.is_anchor = (in_ball[k] == anchor.heart_tdof);
            local_links_.push_back(l);
          }
        }
      }
    }
  }

  if (smear_r > 0.0) {
    int global_smear_total = 0;
    // Collective: every rank must call MPI_Reduce.
    MPI_Reduce(&n_smear_total, &global_smear_total, 1, MPI_INT, MPI_SUM, 0,
               comm_);
    (void)global_smear_total;
  }
}

void PvjCoupler::BuildHeartCouplingCurrent(const mfem::Vector& vm_true,
                                           mfem::Vector& heart_current_true) const {
  if (heart_current_true.Size() != vm_true.Size()) {
    heart_current_true.SetSize(vm_true.Size());
  }
  heart_current_true = 0.0;
  // i_pvj_heart = weight * pvj_g * (V_m - V_p) summed over smeared DOFs.
  // I_ion outward-positive convention: when V_p > V_m the contribution is
  // negative, depolarizing the local myocardium.
  for (const auto& lk : local_links_) {
    const double Vp = cable_.TerminalVoltage(lk.purkinje_node);
    const double Vm = vm_true[lk.heart_tdof];
    heart_current_true[lk.heart_tdof] +=
        lk.weight * cfg_.pvj_g_mS * cfg_.pvj_current_scale * (Vm - Vp);
  }
}

void PvjCoupler::AppendDelayedSample(double t_mid_ms,
                                     const std::vector<double>& vm_at_terms) {
  if (cfg_.pvj_delay_ms <= 0.0) return;
  delay_buf_.push_back({t_mid_ms, vm_at_terms});
  // Trim history: keep only samples within delay window + small slack.
  const double cutoff = t_mid_ms - 4.0 * cfg_.pvj_delay_ms;
  while (delay_buf_.size() > 1 && delay_buf_.front().t_ms < cutoff) {
    delay_buf_.erase(delay_buf_.begin());
  }
}

std::vector<double> PvjCoupler::ReadDelayedSample(double t_mid_ms) const {
  const int n_term = cable_.NumTerminals();
  if (cfg_.pvj_delay_ms <= 0.0 || delay_buf_.empty()) {
    return {};
  }
  const double t_target = t_mid_ms - cfg_.pvj_delay_ms;
  // Find closest sample at or before t_target; if none, return earliest.
  const DelaySample* picked = &delay_buf_.front();
  for (const auto& s : delay_buf_) {
    if (s.t_ms <= t_target) {
      picked = &s;
    }
  }
  if (static_cast<int>(picked->vm.size()) != n_term) return {};
  return picked->vm;
}

void PvjCoupler::AdvancePurkinje(double dt_pde_ms, const mfem::Vector& vm_true,
                                 double t_mid_ms) {
  const int n_term = cable_.NumTerminals();
  std::vector<double> vm_at_terms_local(n_term, 0.0);
  std::vector<double> vm_at_terms_global(n_term, 0.0);

  // Sample V_m only at the anchor (owner rank's single nearest DOF) so the
  // MPI sum below recovers exactly one owner's value per terminal even
  // when smearing inflates the local link list.
  for (const auto& lk : local_links_) {
    if (lk.is_anchor) {
      vm_at_terms_local[lk.purkinje_node] = vm_true[lk.heart_tdof];
    }
  }
  if (n_term > 0) {
    MPI_Allreduce(vm_at_terms_local.data(), vm_at_terms_global.data(),
                  n_term, MPI_DOUBLE, MPI_SUM, comm_);
  }

  std::vector<double> sample_to_use = vm_at_terms_global;
  if (cfg_.pvj_delay_ms > 0.0) {
    AppendDelayedSample(t_mid_ms, vm_at_terms_global);
    auto delayed = ReadDelayedSample(t_mid_ms);
    if (!delayed.empty()) {
      sample_to_use = std::move(delayed);
    }
  }

  cable_.Advance(dt_pde_ms, t_mid_ms, sample_to_use);
}

}  // namespace mono
