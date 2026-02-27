#include "ode/RegionalIonicModel.hpp"

#include <algorithm>
#include <cstdint>
#include <string>
#include <stdexcept>
#include <unordered_set>

namespace mono {
namespace {

std::unordered_set<int> ToSet(const std::vector<int>& v) {
  std::unordered_set<int> out;
  out.reserve(v.size());
  for (const int x : v) {
    out.insert(x);
  }
  return out;
}

bool Intersects(const std::unordered_set<int>& a, const std::unordered_set<int>& b) {
  for (const int x : a) {
    if (b.find(x) != b.end()) {
      return true;
    }
  }
  return false;
}

}  // namespace

RegionalIonicModel::RegionalIonicModel(const SimulationConfig& cfg,
                                       const mfem::ParFiniteElementSpace& pfes)
    : n_nodes_(pfes.GetTrueVSize()) {
  AssignRegionsFromMesh(cfg, pfes);

  atria_model_ = std::make_unique<Grandi2011Model>(static_cast<int>(atria_nodes_.size()));
  ventricles_model_ = std::make_unique<TT06Model>(static_cast<int>(ventricles_nodes_.size()));
  fibrosis_model_ =
      std::make_unique<PassiveModel>(static_cast<int>(fibrosis_nodes_.size()), -85.0, 6.0643e-4);

  vm_atria_.SetSize(static_cast<int>(atria_nodes_.size()));
  vm_ventricles_.SetSize(static_cast<int>(ventricles_nodes_.size()));
  vm_fibrosis_.SetSize(static_cast<int>(fibrosis_nodes_.size()));
  iion_atria_.SetSize(static_cast<int>(atria_nodes_.size()));
  iion_ventricles_.SetSize(static_cast<int>(ventricles_nodes_.size()));
  iion_fibrosis_.SetSize(static_cast<int>(fibrosis_nodes_.size()));
}

void RegionalIonicModel::AssignRegionsFromMesh(const SimulationConfig& cfg,
                                               const mfem::ParFiniteElementSpace& pfes) {
  if (n_nodes_ < 0) {
    throw std::runtime_error("RegionalIonicModel requires n_nodes >= 0");
  }

  const auto atria_set = ToSet(cfg.atria_volume_attrs);
  const auto ventricles_set = ToSet(cfg.ventricles_volume_attrs);
  const auto fibrosis_set = ToSet(cfg.fibrosis_volume_attrs);

  if (atria_set.empty() || ventricles_set.empty() || fibrosis_set.empty()) {
    throw std::runtime_error("RegionalIonicModel requires non-empty atria/ventricles/fibrosis attrs");
  }
  if (Intersects(atria_set, ventricles_set) || Intersects(atria_set, fibrosis_set) ||
      Intersects(ventricles_set, fibrosis_set)) {
    throw std::runtime_error("RegionalIonicModel attr sets must be disjoint");
  }

  region_of_node_.assign(static_cast<size_t>(n_nodes_), Region::Unknown);
  const mfem::ParMesh* pmesh = pfes.GetParMesh();
  if (pmesh == nullptr) {
    throw std::runtime_error("RegionalIonicModel requires valid ParMesh");
  }

  mfem::Array<int> vdofs;
  for (int e = 0; e < pmesh->GetNE(); ++e) {
    const int attr = pmesh->GetAttribute(e);
    Region reg = Region::Unknown;
    if (ventricles_set.find(attr) != ventricles_set.end()) {
      reg = Region::Ventricles;
    } else if (atria_set.find(attr) != atria_set.end()) {
      reg = Region::Atria;
    } else if (fibrosis_set.find(attr) != fibrosis_set.end()) {
      reg = Region::Fibrosis;
    } else {
      throw std::runtime_error("RegionalIonicModel found element attribute outside configured sets: " +
                               std::to_string(attr));
    }

    pfes.GetElementVDofs(e, vdofs);
    for (int j = 0; j < vdofs.Size(); ++j) {
      const int ldof = std::abs(vdofs[j]);
      const int tdof = pfes.GetLocalTDofNumber(ldof);
      if (tdof < 0) {
        continue;
      }
      region_of_node_[static_cast<size_t>(tdof)] =
          PriorityMax(region_of_node_[static_cast<size_t>(tdof)], reg);
    }
  }

  atria_nodes_.clear();
  ventricles_nodes_.clear();
  fibrosis_nodes_.clear();
  atria_nodes_.reserve(region_of_node_.size());
  ventricles_nodes_.reserve(region_of_node_.size());
  fibrosis_nodes_.reserve(region_of_node_.size());

  for (int i = 0; i < n_nodes_; ++i) {
    const Region reg = region_of_node_[static_cast<size_t>(i)];
    switch (reg) {
      case Region::Atria:
        atria_nodes_.push_back(i);
        break;
      case Region::Ventricles:
        ventricles_nodes_.push_back(i);
        break;
      case Region::Fibrosis:
        fibrosis_nodes_.push_back(i);
        break;
      case Region::Unknown:
      default:
        throw std::runtime_error("RegionalIonicModel found unassigned true dof");
    }
  }

  int local_counts[3] = {static_cast<int>(atria_nodes_.size()),
                         static_cast<int>(ventricles_nodes_.size()),
                         static_cast<int>(fibrosis_nodes_.size())};
  int global_counts[3] = {0, 0, 0};
  MPI_Allreduce(local_counts, global_counts, 3, MPI_INT, MPI_SUM, pfes.GetComm());
  if (global_counts[0] <= 0 || global_counts[1] <= 0 || global_counts[2] <= 0) {
    throw std::runtime_error("RegionalIonicModel requires non-empty global node sets for all regions");
  }
}

void RegionalIonicModel::InitializeRestState(double v_rest_mv) {
  atria_model_->InitializeRestState(v_rest_mv);
  ventricles_model_->InitializeRestState(v_rest_mv);
  fibrosis_model_->InitializeRestState(v_rest_mv);
}

void RegionalIonicModel::ComputeIion(const mfem::Vector& vm_true, mfem::Vector& iion_true) const {
  if (vm_true.Size() != n_nodes_) {
    throw std::runtime_error("RegionalIonicModel::ComputeIion vm_true size mismatch");
  }
  if (iion_true.Size() != n_nodes_) {
    iion_true.SetSize(n_nodes_);
  }
  iion_true = 0.0;

  for (int i = 0; i < static_cast<int>(atria_nodes_.size()); ++i) {
    vm_atria_[i] = vm_true[atria_nodes_[static_cast<size_t>(i)]];
  }
  atria_model_->ComputeIion(vm_atria_, iion_atria_);
  for (int i = 0; i < static_cast<int>(atria_nodes_.size()); ++i) {
    iion_true[atria_nodes_[static_cast<size_t>(i)]] = iion_atria_[i];
  }

  for (int i = 0; i < static_cast<int>(ventricles_nodes_.size()); ++i) {
    vm_ventricles_[i] = vm_true[ventricles_nodes_[static_cast<size_t>(i)]];
  }
  ventricles_model_->ComputeIion(vm_ventricles_, iion_ventricles_);
  for (int i = 0; i < static_cast<int>(ventricles_nodes_.size()); ++i) {
    iion_true[ventricles_nodes_[static_cast<size_t>(i)]] = iion_ventricles_[i];
  }

  for (int i = 0; i < static_cast<int>(fibrosis_nodes_.size()); ++i) {
    vm_fibrosis_[i] = vm_true[fibrosis_nodes_[static_cast<size_t>(i)]];
  }
  fibrosis_model_->ComputeIion(vm_fibrosis_, iion_fibrosis_);
  for (int i = 0; i < static_cast<int>(fibrosis_nodes_.size()); ++i) {
    iion_true[fibrosis_nodes_[static_cast<size_t>(i)]] = iion_fibrosis_[i];
  }
}

void RegionalIonicModel::AdvanceStates(double dt_pde_ms,
                                       double dt_ode_ms,
                                       const mfem::Vector& vm_next_true) {
  if (vm_next_true.Size() != n_nodes_) {
    throw std::runtime_error("RegionalIonicModel::AdvanceStates vm_next_true size mismatch");
  }

  for (int i = 0; i < static_cast<int>(atria_nodes_.size()); ++i) {
    vm_atria_[i] = vm_next_true[atria_nodes_[static_cast<size_t>(i)]];
  }
  atria_model_->AdvanceStates(dt_pde_ms, dt_ode_ms, vm_atria_);

  for (int i = 0; i < static_cast<int>(ventricles_nodes_.size()); ++i) {
    vm_ventricles_[i] = vm_next_true[ventricles_nodes_[static_cast<size_t>(i)]];
  }
  ventricles_model_->AdvanceStates(dt_pde_ms, dt_ode_ms, vm_ventricles_);

  for (int i = 0; i < static_cast<int>(fibrosis_nodes_.size()); ++i) {
    vm_fibrosis_[i] = vm_next_true[fibrosis_nodes_[static_cast<size_t>(i)]];
  }
  fibrosis_model_->AdvanceStates(dt_pde_ms, dt_ode_ms, vm_fibrosis_);
}

void RegionalIonicModel::SaveState(std::ostream& os) const {
  const int32_t version = 1;
  const int32_t n = n_nodes_;
  os.write(reinterpret_cast<const char*>(&version), sizeof(version));
  os.write(reinterpret_cast<const char*>(&n), sizeof(n));
  for (const Region reg : region_of_node_) {
    const uint8_t v = static_cast<uint8_t>(reg);
    os.write(reinterpret_cast<const char*>(&v), sizeof(v));
  }
  atria_model_->SaveState(os);
  ventricles_model_->SaveState(os);
  fibrosis_model_->SaveState(os);
}

void RegionalIonicModel::LoadState(std::istream& is) {
  int32_t version = 0;
  int32_t n = 0;
  is.read(reinterpret_cast<char*>(&version), sizeof(version));
  is.read(reinterpret_cast<char*>(&n), sizeof(n));
  if (version != 1) {
    throw std::runtime_error("RegionalIonicModel::LoadState unsupported version");
  }
  if (n != n_nodes_) {
    throw std::runtime_error("RegionalIonicModel::LoadState node count mismatch");
  }
  for (int i = 0; i < n_nodes_; ++i) {
    uint8_t v = 0;
    is.read(reinterpret_cast<char*>(&v), sizeof(v));
    if (static_cast<Region>(v) != region_of_node_[static_cast<size_t>(i)]) {
      throw std::runtime_error("RegionalIonicModel::LoadState region map mismatch");
    }
  }
  atria_model_->LoadState(is);
  ventricles_model_->LoadState(is);
  fibrosis_model_->LoadState(is);
  if (!is) {
    throw std::runtime_error("RegionalIonicModel::LoadState failed to read data");
  }
}

}  // namespace mono
