#include "coupling/InterfaceMapper.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace mono {

void InterfaceMapper::BuildTrueDofCoordinates(const mfem::ParFiniteElementSpace& fes,
                                              mfem::Vector& x_true,
                                              mfem::Vector& y_true,
                                              mfem::Vector& z_true) {
  // 用投影方式在 true dof 上提取坐标场，避免手写几何遍历的映射误差。
  mfem::ParGridFunction gx(const_cast<mfem::ParFiniteElementSpace*>(&fes));
  mfem::ParGridFunction gy(const_cast<mfem::ParFiniteElementSpace*>(&fes));
  mfem::ParGridFunction gz(const_cast<mfem::ParFiniteElementSpace*>(&fes));

  mfem::FunctionCoefficient cx([](const mfem::Vector& x) { return x[0]; });
  mfem::FunctionCoefficient cy([](const mfem::Vector& x) { return x.Size() > 1 ? x[1] : 0.0; });
  mfem::FunctionCoefficient cz([](const mfem::Vector& x) { return x.Size() > 2 ? x[2] : 0.0; });
  gx.ProjectCoefficient(cx);
  gy.ProjectCoefficient(cy);
  gz.ProjectCoefficient(cz);

  gx.GetTrueDofs(x_true);
  gy.GetTrueDofs(y_true);
  gz.GetTrueDofs(z_true);
}

InterfaceMapper::InterfaceMapper(const SimulationConfig& cfg,
                                 Assembler& heart,
                                 const TorsoPotentialSolver& torso,
                                 MPI_Comm comm)
    : cfg_(cfg), heart_(heart), torso_(torso), comm_(comm) {
  MPI_Comm_rank(comm_, &rank_);
  MPI_Comm_size(comm_, &world_size_);

  if (cfg_.use_conforming_wholebody) {
    // conforming 路径：heart/torso 都来自同一个 parent mesh。
    // 通过 parent vertex id 做一一对应，避免全局最近邻搜索。
    const auto* heart_sub = dynamic_cast<const mfem::ParSubMesh*>(heart_.PFES().GetParMesh());
    const auto* torso_sub = dynamic_cast<const mfem::ParSubMesh*>(torso_.PFES().GetParMesh());
    if (heart_sub == nullptr || torso_sub == nullptr) {
      throw std::runtime_error(
          "conforming wholebody mode requires both heart and torso meshes to be ParSubMesh");
    }
    if (heart_sub->GetParent() != torso_sub->GetParent()) {
      throw std::runtime_error(
          "conforming wholebody mode requires heart/torso submeshes from the same parent mesh");
    }

    const auto& heart_parent_vid = heart_sub->GetParentVertexIDMap();
    const auto& torso_parent_vid = torso_sub->GetParentVertexIDMap();
    std::unordered_map<int, int> parent_to_heart_owned_tdof;
    parent_to_heart_owned_tdof.reserve(static_cast<size_t>(heart_parent_vid.Size()));
    // 仅收集本 rank 拥有的 heart true dof（LocalTDofNumber >= 0）。
    for (int hv = 0; hv < heart_parent_vid.Size(); ++hv) {
      mfem::Array<int> h_vdofs;
      heart_.PFES().GetVertexDofs(hv, h_vdofs);
      if (h_vdofs.Size() < 1) {
        continue;
      }

      const int h_ldof = (h_vdofs[0] >= 0) ? h_vdofs[0] : (-1 - h_vdofs[0]);
      const int h_tdof = heart_.PFES().GetLocalTDofNumber(h_ldof);
      if (h_tdof < 0) {
        continue;
      }
      parent_to_heart_owned_tdof.emplace(heart_parent_vid[hv], h_tdof);
    }

    std::vector<std::pair<int, int>> heart_owned_pairs;
    heart_owned_pairs.reserve(parent_to_heart_owned_tdof.size());
    for (const auto& kv : parent_to_heart_owned_tdof) {
      heart_owned_pairs.push_back(kv);
    }
    // 按 parent vid 排序，保证 Allgatherv 后的 sample 索引稳定可复现。
    std::sort(heart_owned_pairs.begin(), heart_owned_pairs.end(), [](const auto& a, const auto& b) {
      return a.first < b.first;
    });

    std::vector<int> heart_parent_vid_local;
    heart_parent_vid_local.reserve(heart_owned_pairs.size());
    heart_tdofs_direct_.SetSize(0);
    for (const auto& kv : heart_owned_pairs) {
      heart_parent_vid_local.push_back(kv.first);
      heart_tdofs_direct_.Append(kv.second);
    }

    const int heart_local_count = static_cast<int>(heart_parent_vid_local.size());
    heart_counts_.assign(world_size_, 0);
    MPI_Allgather(&heart_local_count, 1, MPI_INT, heart_counts_.data(), 1, MPI_INT, comm_);

    heart_displs_.assign(world_size_, 0);
    int heart_global_count = 0;
    for (int r = 0; r < world_size_; ++r) {
      heart_displs_[r] = heart_global_count;
      heart_global_count += heart_counts_[r];
    }
    if (heart_global_count <= 0) {
      throw std::runtime_error(
          "conforming wholebody map found zero owned heart parent-vertex interface dofs");
    }

    std::vector<int> heart_parent_vid_global(heart_global_count, -1);
    MPI_Allgatherv(heart_parent_vid_local.data(),
                   heart_local_count,
                   MPI_INT,
                   heart_parent_vid_global.data(),
                   heart_counts_.data(),
                   heart_displs_.data(),
                   MPI_INT,
                   comm_);
    heart_values_global_.assign(heart_global_count, 0.0);

    std::unordered_map<int, int> parent_to_heart_sample;
    parent_to_heart_sample.reserve(static_cast<size_t>(heart_global_count));
    for (int i = 0; i < heart_global_count; ++i) {
      parent_to_heart_sample.emplace(heart_parent_vid_global[i], i);
    }

    // 遍历 torso 顶点：若 parent vid 在 heart 接口样本中存在，则建立直连约束。
    torso_tdofs_direct_.SetSize(0);
    torso_to_heart_sample_idx_.SetSize(0);
    for (int tv = 0; tv < torso_parent_vid.Size(); ++tv) {
      const int parent_vid = torso_parent_vid[tv];
      const auto it = parent_to_heart_sample.find(parent_vid);
      if (it == parent_to_heart_sample.end()) {
        continue;
      }

      mfem::Array<int> t_vdofs;
      torso_.PFES().GetVertexDofs(tv, t_vdofs);
      if (t_vdofs.Size() < 1) {
        continue;
      }

      const int t_ldof = (t_vdofs[0] >= 0) ? t_vdofs[0] : (-1 - t_vdofs[0]);
      const int t_tdof = torso_.PFES().GetLocalTDofNumber(t_ldof);
      if (t_tdof < 0) {
        continue;
      }

      torso_tdofs_direct_.Append(t_tdof);
      torso_to_heart_sample_idx_.Append(it->second);
    }

    torso_constrained_tdofs_ = torso_tdofs_direct_;
    use_parent_vertex_direct_map_ = true;
    const int local_constrained = torso_tdofs_direct_.Size();
    MPI_Allreduce(&local_constrained, &global_num_constrained_, 1, MPI_INT, MPI_SUM, comm_);
    global_max_constrained_dist_mm_ = 0.0;
    if (global_num_constrained_ <= 0) {
      throw std::runtime_error(
          "conforming wholebody map produced zero shared parent-vertex constraints");
    }
    return;
  }

  // 非 conforming 回退路径：在心脏接口 true dof 上做全局最近邻搜索。
  const mfem::ParMesh* heart_mesh = heart_.PFES().GetParMesh();
  const int bdr_attr_max = heart_mesh->bdr_attributes.Max();
  if (bdr_attr_max <= 0) {
    throw std::runtime_error("InterfaceMapper requires heart mesh boundary attributes");
  }

  mfem::Array<int> heart_bdr_marker(bdr_attr_max);
  heart_bdr_marker = 0;
  if (cfg_.heart_interface_bdr_attrs.empty()) {
    heart_bdr_marker = 1;
  } else {
    for (const int attr : cfg_.heart_interface_bdr_attrs) {
      if (attr < 1 || attr > bdr_attr_max) {
        throw std::runtime_error("heart_interface_bdr_attrs contains invalid attribute");
      }
      heart_bdr_marker[attr - 1] = 1;
    }
  }
  heart_.PFES().GetEssentialTrueDofs(heart_bdr_marker, heart_bdr_tdofs_local_);

  mfem::Vector xh, yh, zh;
  BuildTrueDofCoordinates(heart_.PFES(), xh, yh, zh);

  const int heart_local_count = heart_bdr_tdofs_local_.Size();
  std::vector<double> heart_x_local(heart_local_count, 0.0);
  std::vector<double> heart_y_local(heart_local_count, 0.0);
  std::vector<double> heart_z_local(heart_local_count, 0.0);
  for (int i = 0; i < heart_local_count; ++i) {
    const int tdof = heart_bdr_tdofs_local_[i];
    heart_x_local[i] = xh[tdof];
    heart_y_local[i] = yh[tdof];
    heart_z_local[i] = zh[tdof];
  }

  heart_counts_.assign(world_size_, 0);
  MPI_Allgather(&heart_local_count, 1, MPI_INT, heart_counts_.data(), 1, MPI_INT, comm_);

  heart_displs_.assign(world_size_, 0);
  int heart_global_count = 0;
  for (int r = 0; r < world_size_; ++r) {
    heart_displs_[r] = heart_global_count;
    heart_global_count += heart_counts_[r];
  }
  if (heart_global_count <= 0) {
    throw std::runtime_error("InterfaceMapper found zero heart interface dofs");
  }

  heart_x_global_.assign(heart_global_count, 0.0);
  heart_y_global_.assign(heart_global_count, 0.0);
  heart_z_global_.assign(heart_global_count, 0.0);
  heart_values_global_.assign(heart_global_count, 0.0);

  MPI_Allgatherv(heart_x_local.data(),
                 heart_local_count,
                 MPI_DOUBLE,
                 heart_x_global_.data(),
                 heart_counts_.data(),
                 heart_displs_.data(),
                 MPI_DOUBLE,
                 comm_);
  MPI_Allgatherv(heart_y_local.data(),
                 heart_local_count,
                 MPI_DOUBLE,
                 heart_y_global_.data(),
                 heart_counts_.data(),
                 heart_displs_.data(),
                 MPI_DOUBLE,
                 comm_);
  MPI_Allgatherv(heart_z_local.data(),
                 heart_local_count,
                 MPI_DOUBLE,
                 heart_z_global_.data(),
                 heart_counts_.data(),
                 heart_displs_.data(),
                 MPI_DOUBLE,
                 comm_);

  mfem::Vector xt, yt, zt;
  BuildTrueDofCoordinates(torso_.PFES(), xt, yt, zt);
  torso_constrained_tdofs_.SetSize(0);
  torso_to_heart_sample_idx_.SetSize(0);
  const double dist_tol = cfg_.interface_map_max_dist_mm;

  // 对每个 torso true dof，找最近的 heart 接口样本，且距离不超过阈值。
  double local_max_dist = 0.0;
  for (int tdof = 0; tdof < xt.Size(); ++tdof) {
    const double tx = xt[tdof];
    const double ty = yt[tdof];
    const double tz = zt[tdof];

    double best_d2 = std::numeric_limits<double>::infinity();
    int best_i = -1;
    for (int i = 0; i < heart_global_count; ++i) {
      const double dx = tx - heart_x_global_[i];
      const double dy = ty - heart_y_global_[i];
      const double dz = tz - heart_z_global_[i];
      const double d2 = dx * dx + dy * dy + dz * dz;
      if (d2 < best_d2) {
        best_d2 = d2;
        best_i = i;
      }
    }
    const double best_dist = std::sqrt(best_d2);
    if (best_i >= 0 && best_dist <= dist_tol) {
      torso_constrained_tdofs_.Append(tdof);
      torso_to_heart_sample_idx_.Append(best_i);
      local_max_dist = std::max(local_max_dist, best_dist);
    }
  }

  const int local_constrained = torso_constrained_tdofs_.Size();
  MPI_Allreduce(&local_constrained,
                &global_num_constrained_,
                1,
                MPI_INT,
                MPI_SUM,
                comm_);
  MPI_Allreduce(&local_max_dist,
                &global_max_constrained_dist_mm_,
                1,
                MPI_DOUBLE,
                MPI_MAX,
                comm_);
  if (global_num_constrained_ <= 0) {
    throw std::runtime_error("InterfaceMapper produced zero constrained torso dofs");
  }
}

void InterfaceMapper::MapHeartToTorso(const mfem::Vector& ue_heart_true, mfem::Vector& torso_bc_true) const {
  if (use_parent_vertex_direct_map_) {
    // conforming 直连路径：按预构建的 sample 索引进行 gather + 赋值。
    const int heart_local_count = heart_tdofs_direct_.Size();
    std::vector<double> heart_values_local(heart_local_count, 0.0);
    for (int i = 0; i < heart_local_count; ++i) {
      const int tdof = heart_tdofs_direct_[i];
      if (tdof < 0 || tdof >= ue_heart_true.Size()) {
        throw std::runtime_error("InterfaceMapper direct map heart tdof out of range");
      }
      heart_values_local[i] = ue_heart_true[tdof];
    }

    MPI_Allgatherv(heart_values_local.data(),
                   heart_local_count,
                   MPI_DOUBLE,
                   heart_values_global_.data(),
                   heart_counts_.data(),
                   heart_displs_.data(),
                   MPI_DOUBLE,
                   comm_);

    torso_bc_true.SetSize(torso_.PFES().GetTrueVSize());
    torso_bc_true = 0.0;
    for (int i = 0; i < torso_tdofs_direct_.Size(); ++i) {
      const int t = torso_tdofs_direct_[i];
      const int sample = torso_to_heart_sample_idx_[i];
      if (t < 0 || t >= torso_bc_true.Size() || sample < 0 ||
          sample >= static_cast<int>(heart_values_global_.size())) {
        throw std::runtime_error("InterfaceMapper direct parent-vertex map index out of range");
      }
      torso_bc_true[t] = heart_values_global_[sample];
    }
    return;
  }

  // 最近邻回退路径：接口样本 gather 后再给 torso 约束 dof 赋值。
  const int heart_local_count = heart_bdr_tdofs_local_.Size();
  std::vector<double> heart_values_local(heart_local_count, 0.0);
  for (int i = 0; i < heart_local_count; ++i) {
    const int tdof = heart_bdr_tdofs_local_[i];
    if (tdof < 0 || tdof >= ue_heart_true.Size()) {
      throw std::runtime_error("InterfaceMapper local heart tdof out of range");
    }
    heart_values_local[i] = ue_heart_true[tdof];
  }

  MPI_Allgatherv(heart_values_local.data(),
                 heart_local_count,
                 MPI_DOUBLE,
                 heart_values_global_.data(),
                 heart_counts_.data(),
                 heart_displs_.data(),
                 MPI_DOUBLE,
                 comm_);

  torso_bc_true.SetSize(torso_.PFES().GetTrueVSize());
  torso_bc_true = 0.0;
  for (int i = 0; i < torso_constrained_tdofs_.Size(); ++i) {
    const int tdof = torso_constrained_tdofs_[i];
    const int sample = torso_to_heart_sample_idx_[i];
    torso_bc_true[tdof] = heart_values_global_[sample];
  }
}

}  // namespace mono
