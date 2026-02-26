#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include "mfem.hpp"

#include "config/SimulationConfig.hpp"
#include "io/CheckpointIO.hpp"
#include "ode/TT06Model.hpp"
#include "solver/LinearSolverFactory.hpp"
#include "solver/MonodomainStepper.hpp"
#include "space/Assembler.hpp"

namespace {

// Build a compact benchmark config used by restart equivalence test.
mono::SimulationConfig MakeBaseConfig(const std::filesystem::path& checkpoint_dir) {
  const std::filesystem::path repo_root = std::filesystem::path(__FILE__).parent_path().parent_path();
  mono::SimulationConfig cfg;
  cfg.mesh_path = (repo_root / "benchmarks/niederer/niederer_benchmark.mesh").string();
  cfg.use_fiber_gf = true;
  cfg.fiber_f_path = (repo_root / "benchmarks/niederer/fiber_f.gf").string();
  cfg.fiber_s_path = (repo_root / "benchmarks/niederer/fiber_s.gf").string();
  cfg.fiber_n_path = (repo_root / "benchmarks/niederer/fiber_n.gf").string();
  cfg.dt_pde_ms = 0.01;
  cfg.dt_ode_ms = 0.01;
  cfg.stim_start_ms = 0.0;
  cfg.stim_end_ms = 2.0;
  cfg.output_stride = 1000000;
  cfg.checkpoint_stride = 10;
  cfg.output_dir = (checkpoint_dir / "unused_output").string();
  cfg.checkpoint_dir = checkpoint_dir.string();
  return cfg;
}

mfem::Vector RunContinuous(const mono::SimulationConfig& cfg) {
  // Reference run without restart.
  mono::Assembler assembler(cfg, MPI_COMM_WORLD);
  mono::TT06Model tt06(assembler.TrueVSize());
  tt06.InitializeRestState(-85.23);
  mono::LinearSystemSolver linear_solver(cfg, MPI_COMM_WORLD);
  mono::MonodomainStepper stepper(cfg, assembler, tt06, linear_solver);

  stepper.InitializeVm(-85.23);
  stepper.Bootstrap();
  while (stepper.TimeMs() < cfg.t_end_ms) {
    stepper.StepNoCorrection();
  }

  mfem::Vector vm_final(stepper.VmTrue());
  return vm_final;
}

void RunAndSaveCheckpoint(const mono::SimulationConfig& cfg) {
  // Run prefix and persist periodic checkpoints.
  mono::Assembler assembler(cfg, MPI_COMM_WORLD);
  mono::TT06Model tt06(assembler.TrueVSize());
  tt06.InitializeRestState(-85.23);
  mono::LinearSystemSolver linear_solver(cfg, MPI_COMM_WORLD);
  mono::MonodomainStepper stepper(cfg, assembler, tt06, linear_solver);
  mono::CheckpointIO checkpoint(cfg);

  stepper.InitializeVm(-85.23);
  stepper.Bootstrap();
  if (stepper.StepCount() % cfg.checkpoint_stride == 0) {
    checkpoint.SaveLatest(stepper.StepCount(), stepper.TimeMs(), assembler.Vm(), tt06);
  }

  while (stepper.TimeMs() < cfg.t_end_ms) {
    stepper.StepNoCorrection();
    if (stepper.StepCount() % cfg.checkpoint_stride == 0) {
      checkpoint.SaveLatest(stepper.StepCount(), stepper.TimeMs(), assembler.Vm(), tt06);
    }
  }
}

mfem::Vector RunRestartFromLatest(const mono::SimulationConfig& cfg) {
  // Resume from latest checkpoint and continue to target time.
  mono::Assembler assembler(cfg, MPI_COMM_WORLD);
  mono::TT06Model tt06(assembler.TrueVSize());
  tt06.InitializeRestState(-85.23);
  mono::LinearSystemSolver linear_solver(cfg, MPI_COMM_WORLD);
  mono::MonodomainStepper stepper(cfg, assembler, tt06, linear_solver);
  mono::CheckpointIO checkpoint(cfg);

  int restart_step = 0;
  double restart_t_ms = 0.0;
  if (!checkpoint.LoadLatest(restart_step, restart_t_ms, assembler.Vm(), tt06)) {
    throw std::runtime_error("failed to load latest checkpoint");
  }
  stepper.InitializeFromCurrentVm(restart_step, restart_t_ms);

  mfem::Vector vm_restart_true;
  assembler.Vm().GetTrueDofs(vm_restart_true);
  if (vm_restart_true.Size() != tt06.NumNodes()) {
    throw std::runtime_error("restart state size mismatch: vm_true=" +
                             std::to_string(vm_restart_true.Size()) +
                             ", tt06_nodes=" + std::to_string(tt06.NumNodes()));
  }

  while (stepper.TimeMs() < cfg.t_end_ms) {
    stepper.StepNoCorrection();
  }

  mfem::Vector vm_final(stepper.VmTrue());
  return vm_final;
}

}  // namespace

int main(int argc, char* argv[]) {
  mfem::Mpi::Init(argc, argv);
  const int rank = mfem::Mpi::WorldRank();
  const int world_size = mfem::Mpi::WorldSize();

  try {
    const std::filesystem::path checkpoint_dir =
        std::filesystem::temp_directory_path() / "mono_restart_resume_test";
    if (rank == 0) {
      std::filesystem::remove_all(checkpoint_dir);
      std::filesystem::create_directories(checkpoint_dir);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Compare full run vs restart-resume run at same end time.
    mono::SimulationConfig cfg_ref = MakeBaseConfig(checkpoint_dir);
    cfg_ref.t_end_ms = 0.40;
    const mfem::Vector vm_ref = RunContinuous(cfg_ref);

    mono::SimulationConfig cfg_pre = MakeBaseConfig(checkpoint_dir);
    cfg_pre.t_end_ms = 0.20;
    RunAndSaveCheckpoint(cfg_pre);

    mono::SimulationConfig cfg_post = MakeBaseConfig(checkpoint_dir);
    cfg_post.t_end_ms = 0.40;
    const mfem::Vector vm_restart = RunRestartFromLatest(cfg_post);

    int local_size_ok = (vm_ref.Size() == vm_restart.Size()) ? 1 : 0;
    int global_size_ok = 0;
    MPI_Allreduce(&local_size_ok, &global_size_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (global_size_ok == 0) {
      if (rank == 0) {
        std::cerr << "size mismatch across restart" << std::endl;
      }
      return 1;
    }

    double local_max_abs_diff = 0.0;
    for (int i = 0; i < vm_ref.Size(); ++i) {
      local_max_abs_diff = std::max(local_max_abs_diff, std::abs(vm_ref[i] - vm_restart[i]));
    }
    double global_max_abs_diff = 0.0;
    MPI_Allreduce(&local_max_abs_diff, &global_max_abs_diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    // Iterative linear solves can diverge slightly across a restart boundary.
    if (global_max_abs_diff > 1e-4) {
      if (rank == 0) {
        std::cerr << "restart mismatch: max |dVm| = " << global_max_abs_diff << std::endl;
      }
      return 2;
    }
  } catch (const std::exception& ex) {
    std::cerr << "[rank " << rank << "/" << world_size << "] " << ex.what() << std::endl;
    return 3;
  }

  return 0;
}
