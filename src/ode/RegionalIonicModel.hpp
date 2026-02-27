#pragma once

#include <cstdint>
#include <iosfwd>
#include <memory>
#include <vector>

#include "config/SimulationConfig.hpp"
#include "mfem.hpp"
#include "ode/Grandi2011Model.hpp"
#include "ode/IonicModel.hpp"
#include "ode/PassiveModel.hpp"
#include "ode/TT06Model.hpp"

namespace mono {

// Regional ionic dispatch:
//   atria -> Grandi2011
//   ventricles -> TT06
//   fibrosis -> Passive
class RegionalIonicModel : public IonicModel {
 public:
  RegionalIonicModel(const SimulationConfig& cfg, const mfem::ParFiniteElementSpace& pfes);

  void InitializeRestState(double v_rest_mv) override;
  void ComputeIion(const mfem::Vector& vm_true, mfem::Vector& iion_true) const override;
  void AdvanceStates(double dt_pde_ms, double dt_ode_ms, const mfem::Vector& vm_next_true) override;

  void SaveState(std::ostream& os) const override;
  void LoadState(std::istream& is) override;

  int NumNodes() const override { return n_nodes_; }
  const char* ModelTag() const override { return "REGIONAL_V1"; }

 private:
  enum class Region : uint8_t { Unknown = 0, Fibrosis = 1, Atria = 2, Ventricles = 3 };

  static Region PriorityMax(Region a, Region b) {
    return (static_cast<int>(a) >= static_cast<int>(b)) ? a : b;
  }

  void AssignRegionsFromMesh(const SimulationConfig& cfg, const mfem::ParFiniteElementSpace& pfes);

  int n_nodes_ = 0;
  std::vector<Region> region_of_node_;
  std::vector<int> atria_nodes_;
  std::vector<int> ventricles_nodes_;
  std::vector<int> fibrosis_nodes_;

  std::unique_ptr<Grandi2011Model> atria_model_;
  std::unique_ptr<TT06Model> ventricles_model_;
  std::unique_ptr<PassiveModel> fibrosis_model_;

  mutable mfem::Vector vm_atria_;
  mutable mfem::Vector vm_ventricles_;
  mutable mfem::Vector vm_fibrosis_;
  mutable mfem::Vector iion_atria_;
  mutable mfem::Vector iion_ventricles_;
  mutable mfem::Vector iion_fibrosis_;
};

}  // namespace mono
