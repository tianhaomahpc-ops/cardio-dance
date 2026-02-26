#pragma once

#include <string>

#include "mfem.hpp"

#include "config/SimulationConfig.hpp"
#include "ode/TT06Model.hpp"

namespace mono {

// Rank-sharded checkpoint I/O for Vm and TT06 internal states.
// Metadata file tracks step/time and MPI world size for restart validation.
class CheckpointIO {
 public:
  explicit CheckpointIO(const SimulationConfig& cfg, MPI_Comm comm = MPI_COMM_WORLD);

  // Write latest checkpoint snapshot for current rank.
  void SaveLatest(int step, double t_ms, const mfem::ParGridFunction& vm, const TT06Model& tt06) const;
  // Load latest checkpoint if present and compatible with current MPI size.
  bool LoadLatest(int& step, double& t_ms, mfem::ParGridFunction& vm, TT06Model& tt06) const;

 private:
  std::string MetaPath() const;
  std::string VmShardPath(int rank) const;
  std::string Tt06ShardPath(int rank) const;

  MPI_Comm comm_;
  int rank_ = 0;
  int world_size_ = 1;
  std::string checkpoint_dir_;
};

}  // namespace mono
