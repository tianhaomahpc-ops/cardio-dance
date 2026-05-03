#pragma once

#include <iosfwd>
#include <memory>
#include <vector>

#include "mfem.hpp"

#include "config/SimulationConfig.hpp"
#include "ode/IIonicModel.hpp"

namespace mono {

class TT06Model;
class Grandi2011Model;
class PassiveModel;

// Per-true-DOF dispatcher that routes I_ion / state advance to a region-
// specific child IIonicModel. Region tags are derived from the parallel mesh
// element attributes mapped to true DOFs at construction time.
//
// Region IDs are stable enum values; their mapping to mesh attributes is
// supplied by SimulationConfig (atria/ventricle/av_delay/fibrosis_volume_attrs).
// All children own state for the FULL DOF count for simplicity; the dispatcher
// only invokes the relevant per-region child on each DOF, so cost scales with
// number of regions, not number of children.
class RegionalIonicModel : public IIonicModel {
 public:
  enum class Region : int {
    Ventricle = 0,
    Atria = 1,
    AvDelay = 2,
    Fibrosis = 3,
  };

  RegionalIonicModel(const SimulationConfig& cfg,
                     const mfem::ParFiniteElementSpace& pfes);
  ~RegionalIonicModel() override;

  void InitializeRestState(double v_rest_mv) override;
  void ComputeIion(const mfem::Vector& vm_true, mfem::Vector& iion_true) const override;
  void AdvanceStates(double dt_pde_ms, double dt_ode_ms,
                     const mfem::Vector& vm_next_true) override;

  void SaveState(std::ostream& os) const override;
  void LoadState(std::istream& is) override;

  int NumNodes() const override { return n_nodes_; }
  std::string ModelId() const override { return "Regional"; }

  // Diagnostics: number of true DOFs per region (this rank).
  int LocalDofCount(Region r) const { return per_region_count_[static_cast<int>(r)]; }
  // True-DOF -> region tag (length = TrueVSize()).
  const std::vector<int>& DofRegions() const { return dof_region_; }

 private:
  void BuildDofRegionMap(const mfem::ParFiniteElementSpace& pfes,
                         const SimulationConfig& cfg);
  // Filter vm_true / iion_true into per-region scratch vectors and back.
  void GatherForRegion(Region r, const mfem::Vector& src,
                       mfem::Vector& dst) const;
  void ScatterFromRegion(Region r, const mfem::Vector& src,
                         mfem::Vector& dst) const;

  int n_nodes_;
  std::vector<int> dof_region_;             // size n_nodes_
  std::array<std::vector<int>, 4> region_indices_;
  std::array<int, 4> per_region_count_;

  std::unique_ptr<TT06Model> ventricle_model_;
  std::unique_ptr<Grandi2011Model> atria_model_;
  std::unique_ptr<PassiveModel> avdelay_model_;
  std::unique_ptr<PassiveModel> fibrosis_model_;

  mutable mfem::Vector vm_local_;
  mutable mfem::Vector iion_local_;
};

}  // namespace mono
