#include "ode/RegionalIonicModel.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <istream>
#include <ostream>
#include <set>
#include <stdexcept>

#include "ode/Grandi2011Model.hpp"
#include "ode/PassiveModel.hpp"
#include "ode/TT06Model.hpp"

namespace mono {
namespace {

constexpr int kNumRegions = 4;

}  // namespace

RegionalIonicModel::RegionalIonicModel(const SimulationConfig& cfg,
                                       const mfem::ParFiniteElementSpace& pfes)
    : n_nodes_(pfes.GetTrueVSize()),
      dof_region_(static_cast<size_t>(pfes.GetTrueVSize()),
                  static_cast<int>(Region::Ventricle)),
      per_region_count_({0, 0, 0, 0}) {
  if (n_nodes_ <= 0) {
    throw std::runtime_error("RegionalIonicModel requires positive true DOF count");
  }

  BuildDofRegionMap(pfes, cfg);

  // All four child models size to full DOF count; per-region calls only
  // touch DOFs whose dof_region_ matches. State stored only where used; the
  // unused per-DOF entries remain at rest and never feed back into the system.
  ventricle_model_ = std::make_unique<TT06Model>(n_nodes_);
  atria_model_     = std::make_unique<Grandi2011Model>(n_nodes_);
  avdelay_model_   = std::make_unique<PassiveModel>(
      n_nodes_, cfg.av_delay_leak_g_mS_per_uF, cfg.passive_v_rest_mv);
  fibrosis_model_  = std::make_unique<PassiveModel>(
      n_nodes_, cfg.fibrosis_leak_g_mS_per_uF, cfg.passive_v_rest_mv);
}

RegionalIonicModel::~RegionalIonicModel() = default;

void RegionalIonicModel::BuildDofRegionMap(const mfem::ParFiniteElementSpace& pfes,
                                           const SimulationConfig& cfg) {
  std::set<int> atria(cfg.atria_volume_attrs.begin(), cfg.atria_volume_attrs.end());
  std::set<int> ventricle(cfg.ventricle_volume_attrs.begin(),
                          cfg.ventricle_volume_attrs.end());
  std::set<int> avdelay(cfg.av_delay_volume_attrs.begin(),
                        cfg.av_delay_volume_attrs.end());
  std::set<int> fibrosis(cfg.fibrosis_volume_attrs.begin(),
                         cfg.fibrosis_volume_attrs.end());

  const auto attr_to_region = [&](int attr) -> Region {
    if (avdelay.count(attr))   return Region::AvDelay;
    if (fibrosis.count(attr))  return Region::Fibrosis;
    if (atria.count(attr))     return Region::Atria;
    if (ventricle.count(attr)) return Region::Ventricle;
    return Region::Ventricle;  // default fallback
  };

  // Walk elements and tag DOFs by their owning element's attribute.
  // Multiple attributes touching one DOF: most-specific wins (AV-delay >
  // fibrosis > atria > ventricle).
  std::vector<int> dof_priority(static_cast<size_t>(n_nodes_), -1);

  const int num_elements = pfes.GetParMesh()->GetNE();
  mfem::Array<int> dofs;
  for (int e = 0; e < num_elements; ++e) {
    const int attr = pfes.GetParMesh()->GetAttribute(e);
    const Region r = attr_to_region(attr);
    const int prio = static_cast<int>(r);  // higher index = higher priority
    pfes.GetElementDofs(e, dofs);
    for (int i = 0; i < dofs.Size(); ++i) {
      const int local = dofs[i] >= 0 ? dofs[i] : -1 - dofs[i];
      const int tdof = pfes.GetLocalTDofNumber(local);
      if (tdof < 0) continue;  // not owned by this rank
      if (prio > dof_priority[tdof]) {
        dof_priority[tdof] = prio;
        dof_region_[tdof] = static_cast<int>(r);
      }
    }
  }

  // Bucket DOFs per region for fast gather/scatter.
  for (int i = 0; i < kNumRegions; ++i) {
    region_indices_[i].clear();
  }
  for (int tdof = 0; tdof < n_nodes_; ++tdof) {
    region_indices_[dof_region_[tdof]].push_back(tdof);
  }
  for (int i = 0; i < kNumRegions; ++i) {
    per_region_count_[i] = static_cast<int>(region_indices_[i].size());
  }
}

void RegionalIonicModel::InitializeRestState(double v_rest_mv) {
  ventricle_model_->InitializeRestState(v_rest_mv);
  atria_model_->InitializeRestState(v_rest_mv);
  avdelay_model_->InitializeRestState(v_rest_mv);
  fibrosis_model_->InitializeRestState(v_rest_mv);
}

void RegionalIonicModel::GatherForRegion(Region r, const mfem::Vector& src,
                                         mfem::Vector& dst) const {
  const auto& idx = region_indices_[static_cast<int>(r)];
  if (dst.Size() != n_nodes_) {
    dst.SetSize(n_nodes_);
  }
  // Pad with rest-state-equivalent values (V_m doesn't matter for non-region
  // DOFs since their I_ion outputs are discarded). Use zeros for simplicity.
  dst = 0.0;
  for (int i : idx) {
    dst[i] = src[i];
  }
}

void RegionalIonicModel::ScatterFromRegion(Region r, const mfem::Vector& src,
                                           mfem::Vector& dst) const {
  const auto& idx = region_indices_[static_cast<int>(r)];
  for (int i : idx) {
    dst[i] = src[i];
  }
}

void RegionalIonicModel::ComputeIion(const mfem::Vector& vm_true,
                                     mfem::Vector& iion_true) const {
  if (vm_true.Size() != n_nodes_) {
    throw std::runtime_error("RegionalIonicModel::ComputeIion size mismatch");
  }
  if (iion_true.Size() != n_nodes_) {
    iion_true.SetSize(n_nodes_);
  }

  if (vm_local_.Size() != n_nodes_) {
    vm_local_.SetSize(n_nodes_);
    iion_local_.SetSize(n_nodes_);
  }

  // We pass full-length vectors to children but only consume their values at
  // the DOFs belonging to that region.
  iion_true = 0.0;

  auto run_region = [&](Region r, IIonicModel& m) {
    if (per_region_count_[static_cast<int>(r)] == 0) return;
    // Use vm_true directly; child writes a full-length vector.
    m.ComputeIion(vm_true, iion_local_);
    ScatterFromRegion(r, iion_local_, iion_true);
  };

  run_region(Region::Ventricle, *ventricle_model_);
  run_region(Region::Atria,     *atria_model_);
  run_region(Region::AvDelay,   *avdelay_model_);
  run_region(Region::Fibrosis,  *fibrosis_model_);
}

void RegionalIonicModel::AdvanceStates(double dt_pde_ms, double dt_ode_ms,
                                       const mfem::Vector& vm_next_true) {
  // Advance every child; only DOFs owned by their region matter, but the
  // operations are O(N_dof) each, so total cost = O(4 * N_dof). For small
  // regions this could be optimized later.
  ventricle_model_->AdvanceStates(dt_pde_ms, dt_ode_ms, vm_next_true);
  atria_model_->AdvanceStates(dt_pde_ms, dt_ode_ms, vm_next_true);
  avdelay_model_->AdvanceStates(dt_pde_ms, dt_ode_ms, vm_next_true);
  fibrosis_model_->AdvanceStates(dt_pde_ms, dt_ode_ms, vm_next_true);
}

void RegionalIonicModel::SaveState(std::ostream& os) const {
  ventricle_model_->SaveState(os);
  atria_model_->SaveState(os);
  avdelay_model_->SaveState(os);
  fibrosis_model_->SaveState(os);
  const int32_t n = n_nodes_;
  os.write(reinterpret_cast<const char*>(&n), sizeof(n));
  os.write(reinterpret_cast<const char*>(dof_region_.data()),
           static_cast<std::streamsize>(dof_region_.size() * sizeof(int)));
}

void RegionalIonicModel::LoadState(std::istream& is) {
  ventricle_model_->LoadState(is);
  atria_model_->LoadState(is);
  avdelay_model_->LoadState(is);
  fibrosis_model_->LoadState(is);
  int32_t n = 0;
  is.read(reinterpret_cast<char*>(&n), sizeof(n));
  if (n != n_nodes_) {
    throw std::runtime_error("RegionalIonicModel::LoadState DOF count mismatch");
  }
  is.read(reinterpret_cast<char*>(dof_region_.data()),
          static_cast<std::streamsize>(dof_region_.size() * sizeof(int)));
}

}  // namespace mono
