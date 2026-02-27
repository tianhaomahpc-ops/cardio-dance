#include "io/CheckpointIO.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

namespace mono {

CheckpointIO::CheckpointIO(const SimulationConfig& cfg, MPI_Comm comm) : comm_(comm), checkpoint_dir_(cfg.checkpoint_dir) {
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
  std::ostringstream oss;
  oss << checkpoint_dir_ << "/tt06_rank" << std::setw(6) << std::setfill('0') << rank << ".bin";
  return oss.str();
}

std::string CheckpointIO::IonicShardPath(int rank) const {
  std::ostringstream oss;
  oss << checkpoint_dir_ << "/ionic_rank" << std::setw(6) << std::setfill('0') << rank << ".bin";
  return oss.str();
}

void CheckpointIO::SaveLatest(int step,
                              double t_ms,
                              const mfem::ParGridFunction& vm,
                              const IonicModel& ionic) const {
  // Each rank writes local shard independently.
  {
    std::ofstream vm_out(VmShardPath(rank_));
    vm.Save(vm_out);
  }

  {
    std::ofstream cell_out(IonicShardPath(rank_), std::ios::binary);
    ionic.SaveState(cell_out);
  }

  MPI_Barrier(comm_);

  if (rank_ == 0) {
    // Metadata is single-writer and tiny; write after shards are durable.
    std::ofstream meta(MetaPath());
    meta << "CHKPT_V3\n";
    meta << world_size_ << "\n";
    meta << step << "\n";
    meta << t_ms << "\n";
    meta << ionic.ModelTag() << "\n";
  }

  MPI_Barrier(comm_);
}

bool CheckpointIO::LoadLatest(int& step,
                              double& t_ms,
                              mfem::ParGridFunction& vm,
                              IonicModel& ionic) const {
  constexpr int kMetaUnknown = -1;
  constexpr int kMetaLegacy = 0;
  constexpr int kMetaV2 = 2;
  constexpr int kMetaV3 = 3;

  int meta_version = kMetaUnknown;
  int checkpoint_world_size = 0;
  int checkpoint_step = 0;
  double checkpoint_t_ms = 0.0;
  std::string checkpoint_model_tag;

  if (rank_ == 0) {
    // Support CHKPT_V3, CHKPT_V2 and a simple legacy format for backwards compatibility.
    std::ifstream meta(MetaPath());
    if (!meta) {
      meta_version = kMetaUnknown;
    } else {
      std::string head;
      if (meta >> head) {
        if (head == "CHKPT_V3") {
          meta_version = kMetaV3;
          meta >> checkpoint_world_size >> checkpoint_step >> checkpoint_t_ms >> checkpoint_model_tag;
          if (!meta) {
            meta_version = kMetaUnknown;
          }
        } else if (head == "CHKPT_V2") {
          meta_version = kMetaV2;
          meta >> checkpoint_world_size >> checkpoint_step >> checkpoint_t_ms;
          if (!meta) {
            meta_version = kMetaUnknown;
          }
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
  }

  MPI_Bcast(&meta_version, 1, MPI_INT, 0, comm_);
  if (meta_version == kMetaUnknown) {
    return false;
  }
  MPI_Bcast(&checkpoint_world_size, 1, MPI_INT, 0, comm_);
  MPI_Bcast(&checkpoint_step, 1, MPI_INT, 0, comm_);
  MPI_Bcast(&checkpoint_t_ms, 1, MPI_DOUBLE, 0, comm_);
  int model_tag_len = static_cast<int>(checkpoint_model_tag.size());
  MPI_Bcast(&model_tag_len, 1, MPI_INT, 0, comm_);
  checkpoint_model_tag.resize(static_cast<size_t>(std::max(model_tag_len, 0)));
  if (model_tag_len > 0) {
    MPI_Bcast(checkpoint_model_tag.data(), model_tag_len, MPI_CHAR, 0, comm_);
  }

  if (checkpoint_world_size != world_size_) {
    // Cross-size restart is not yet supported because shards are rank-local.
    return false;
  }

  std::string vm_path;
  std::string ionic_path;
  if (meta_version == kMetaV3) {
    if (checkpoint_model_tag != ionic.ModelTag()) {
      return false;
    }
    vm_path = VmShardPath(rank_);
    ionic_path = IonicShardPath(rank_);
  } else if (meta_version == kMetaV2) {
    if (std::string(ionic.ModelTag()) != "TT06") {
      return false;
    }
    vm_path = VmShardPath(rank_);
    ionic_path = Tt06ShardPath(rank_);
  } else {
    if (std::string(ionic.ModelTag()) != "TT06") {
      return false;
    }
    if (world_size_ != 1) {
      return false;
    }
    vm_path = checkpoint_dir_ + "/vm.gf";
    ionic_path = checkpoint_dir_ + "/tt06.bin";
  }

  {
    std::ifstream vm_in(vm_path);
    if (!vm_in) {
      return false;
    }
    // Read via stream metadata and copy into destination FE space.
    mfem::ParGridFunction vm_src(vm.ParFESpace()->GetParMesh(), vm_in);
    if (vm_src.Size() != vm.Size()) {
      return false;
    }
    vm = vm_src;
  }

  {
    std::ifstream cell_in(ionic_path, std::ios::binary);
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
