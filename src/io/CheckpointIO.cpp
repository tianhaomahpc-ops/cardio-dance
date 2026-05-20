#include "io/CheckpointIO.hpp"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace mono {

CheckpointIO::CheckpointIO(const SimulationConfig& cfg, MPI_Comm comm)
    : comm_(comm), checkpoint_dir_(cfg.checkpoint_dir) {
  MPI_Comm_rank(comm_, &rank_);
  MPI_Comm_size(comm_, &world_size_);
  std::filesystem::create_directories(checkpoint_dir_);
}

std::string CheckpointIO::MetaPath() const {
  return checkpoint_dir_ + "/latest.meta";
}

std::string CheckpointIO::VmShardPath(int rank) const {
  std::ostringstream oss;
  oss << checkpoint_dir_ << "/vm_rank" << std::setw(6) << std::setfill('0') << rank << ".gf";
  return oss.str();
}

std::string CheckpointIO::Tt06ShardPath(int rank) const {
  // Filename retained for backwards compatibility; payload is now whatever
  // IIonicModel::SaveState writes for the active model.
  std::ostringstream oss;
  oss << checkpoint_dir_ << "/cell_rank" << std::setw(6) << std::setfill('0') << rank << ".bin";
  return oss.str();
}

void CheckpointIO::SaveLatest(int step,
                              double t_ms,
                              const mfem::ParGridFunction& vm,
                              const IIonicModel& ionic) const {
  // Each rank writes local shard independently.
  {
    std::ofstream vm_out(VmShardPath(rank_));
    vm.Save(vm_out);
  }

  {
    std::ofstream cell_out(Tt06ShardPath(rank_), std::ios::binary);
    ionic.SaveState(cell_out);
  }

  MPI_Barrier(comm_);

  if (rank_ == 0) {
    // CHKPT_V3 format adds a model_id token so cross-model restarts fail
    // loudly instead of silently misinterpreting binary state.
    //   line 1: CHKPT_V3
    //   line 2: <world_size>
    //   line 3: <step>
    //   line 4: <t_ms>
    //   line 5: <model_id>
    std::ofstream meta(MetaPath());
    meta << "CHKPT_V3\n";
    meta << world_size_ << "\n";
    meta << step << "\n";
    meta << t_ms << "\n";
    meta << ionic.ModelId() << "\n";
  }

  MPI_Barrier(comm_);
}

bool CheckpointIO::LoadLatest(int& step, double& t_ms, mfem::ParGridFunction& vm,
                              IIonicModel& ionic) const {
  constexpr int kMetaUnknown = -1;
  constexpr int kMetaLegacy = 0;
  constexpr int kMetaV2 = 2;
  constexpr int kMetaV3 = 3;

  int meta_version = kMetaUnknown;
  int checkpoint_world_size = 0;
  int checkpoint_step = 0;
  double checkpoint_t_ms = 0.0;
  std::string checkpoint_model_id;
  // For V3 broadcast we need a fixed-size buffer.
  constexpr int kModelIdBufSize = 64;
  char model_id_buf[kModelIdBufSize] = {0};

  if (rank_ == 0) {
    std::ifstream meta(MetaPath());
    if (!meta) {
      meta_version = kMetaUnknown;
    } else {
      std::string head;
      if (meta >> head) {
        if (head == "CHKPT_V3") {
          meta_version = kMetaV3;
          meta >> checkpoint_world_size >> checkpoint_step >> checkpoint_t_ms;
          meta >> checkpoint_model_id;
          if (!meta) meta_version = kMetaUnknown;
        } else if (head == "CHKPT_V2") {
          meta_version = kMetaV2;
          meta >> checkpoint_world_size >> checkpoint_step >> checkpoint_t_ms;
          if (!meta) meta_version = kMetaUnknown;
        } else {
          try {
            checkpoint_step = std::stoi(head);
            meta >> checkpoint_t_ms;
            if (meta) {
              checkpoint_world_size = 1;
              meta_version = kMetaLegacy;
            } else {
              meta_version = kMetaUnknown;
            }
          } catch (...) {
            meta_version = kMetaUnknown;
          }
        }
      }
    }
    if (meta_version == kMetaV3 &&
        checkpoint_model_id.size() < static_cast<size_t>(kModelIdBufSize)) {
      std::strncpy(model_id_buf, checkpoint_model_id.c_str(),
                   kModelIdBufSize - 1);
    }
  }

  MPI_Bcast(&meta_version, 1, MPI_INT, 0, comm_);
  if (meta_version == kMetaUnknown) {
    return false;
  }
  MPI_Bcast(&checkpoint_world_size, 1, MPI_INT, 0, comm_);
  MPI_Bcast(&checkpoint_step, 1, MPI_INT, 0, comm_);
  MPI_Bcast(&checkpoint_t_ms, 1, MPI_DOUBLE, 0, comm_);
  MPI_Bcast(model_id_buf, kModelIdBufSize, MPI_CHAR, 0, comm_);

  if (checkpoint_world_size != world_size_) {
    return false;
  }

  if (meta_version == kMetaV3) {
    const std::string ckpt_model = model_id_buf;
    const std::string runtime_model = ionic.ModelId();
    if (ckpt_model != runtime_model) {
      if (rank_ == 0) {
        std::cerr << "[checkpoint] model mismatch: checkpoint=" << ckpt_model
                  << " runtime=" << runtime_model
                  << " -- refusing to load" << std::endl;
      }
      return false;
    }
  } else {
    // Legacy / V2 checkpoints carry no model_id. Only TT06 was supported at
    // those format versions; accept the load only if runtime is also TT06.
    if (ionic.ModelId() != "TT06") {
      if (rank_ == 0) {
        std::cerr << "[checkpoint] CHKPT_V2 / legacy meta found but runtime"
                  << " model is " << ionic.ModelId()
                  << " (only TT06 was supported pre-V3) -- refusing to load"
                  << std::endl;
      }
      return false;
    }
  }

  std::string vm_path;
  std::string cell_path;
  if (meta_version == kMetaV3 || meta_version == kMetaV2) {
    vm_path = VmShardPath(rank_);
    cell_path = Tt06ShardPath(rank_);
  } else {
    if (world_size_ != 1) {
      return false;
    }
    vm_path = checkpoint_dir_ + "/vm.gf";
    cell_path = checkpoint_dir_ + "/tt06.bin";
  }

  {
    std::ifstream vm_in(vm_path);
    if (!vm_in) {
      return false;
    }
    mfem::ParGridFunction vm_src(vm.ParFESpace()->GetParMesh(), vm_in);
    if (vm_src.Size() != vm.Size()) {
      return false;
    }
    vm = vm_src;
  }

  {
    std::ifstream cell_in(cell_path, std::ios::binary);
    if (!cell_in) {
      return false;
    }
    ionic.LoadState(cell_in);
  }

  step = checkpoint_step;
  t_ms = checkpoint_t_ms;
  return true;
}

}  // namespace mono
