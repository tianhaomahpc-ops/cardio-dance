#pragma once

#include <vector>

#include "mfem.hpp"

#include "config/SimulationConfig.hpp"
#include "solver/TorsoPotentialSolver.hpp"
#include "space/Assembler.hpp"

namespace mono {

// 构建并应用单向耦合映射：ue(heart) -> torso 约束 true dof 的 Dirichlet 采样值。
class InterfaceMapper {
 public:
  InterfaceMapper(const SimulationConfig& cfg,
                  const Assembler& heart,
                  const TorsoPotentialSolver& torso,
                  MPI_Comm comm);

  void MapHeartToTorso(const mfem::Vector& ue_heart_true, mfem::Vector& torso_bc_true) const;

  const mfem::Array<int>& TorsoConstrainedDofs() const { return torso_constrained_tdofs_; }
  int GlobalNumConstrainedDofs() const { return global_num_constrained_; }
  double GlobalMaxConstrainedDistMm() const { return global_max_constrained_dist_mm_; }

 private:
  static void BuildTrueDofCoordinates(const mfem::ParFiniteElementSpace& fes,
                                      mfem::Vector& x_true,
                                      mfem::Vector& y_true,
                                      mfem::Vector& z_true);

  const SimulationConfig& cfg_;
  const Assembler& heart_;
  const TorsoPotentialSolver& torso_;
  MPI_Comm comm_;
  int rank_ = 0;
  int world_size_ = 1;

  mfem::Array<int> heart_bdr_tdofs_local_;
  std::vector<int> heart_counts_;
  std::vector<int> heart_displs_;

  std::vector<double> heart_x_global_;
  std::vector<double> heart_y_global_;
  std::vector<double> heart_z_global_;
  mutable std::vector<double> heart_values_global_;

  mfem::Array<int> torso_constrained_tdofs_;
  mfem::Array<int> torso_to_heart_sample_idx_;
  bool use_parent_vertex_direct_map_ = false;
  mfem::Array<int> heart_tdofs_direct_;
  mfem::Array<int> torso_tdofs_direct_;
  int global_num_constrained_ = 0;
  double global_max_constrained_dist_mm_ = 0.0;
};

}  // namespace mono
