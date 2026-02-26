#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "mfem.hpp"

#include "config/SimulationConfig.hpp"
#include "coupling/InterfaceMapper.hpp"
#include "io/CheckpointIO.hpp"
#include "io/OutputManager.hpp"
#include "ode/TT06Model.hpp"
#include "solver/ExtracellularRecoverySolver.hpp"
#include "solver/LinearSolverFactory.hpp"
#include "solver/MonodomainStepper.hpp"
#include "solver/TorsoPotentialSolver.hpp"
#include "space/Assembler.hpp"

int main(int argc, char* argv[]) {
  mfem::Mpi::Init(argc, argv);
  // BoomerAMG/Hypre 相关对象依赖全局初始化，程序入口统一初始化。
  mfem::Hypre::Init();

  const int rank = mfem::Mpi::WorldRank();
  const int world_size = mfem::Mpi::WorldSize();
  bool petsc_initialized = false;
  bool hypre_finalized = false;

  try {
    // 仅解析少量命令行覆盖项；主要参数仍来自 options 文件。
    std::string config_path = "config/default.options";
    bool restart_from_checkpoint = false;
    bool benchmark_probes = false;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--config" && i + 1 < argc) {
        config_path = argv[++i];
      } else if (arg == "--restart" && i + 1 < argc) {
        restart_from_checkpoint = (std::stoi(argv[++i]) != 0);
      } else if (arg == "--benchmark-probes" && i + 1 < argc) {
        benchmark_probes = (std::stoi(argv[++i]) != 0);
      }
    }

    mono::SimulationConfig cfg = mono::LoadConfigFile(config_path);
    mono::OverrideFromArgs(argc, argv, cfg);

#ifdef MFEM_USE_PETSC
    // PETSc 生命周期按 use_petsc 显式控制。
    if (cfg.use_petsc) {
      mfem::MFEMInitializePetsc();
      petsc_initialized = true;
    }
#endif

    if (rank == 0) {
      std::cout << "[monodomain] configuration\n" << mono::ToString(cfg) << std::endl;
      std::cout << "[monodomain] MPI size = " << world_size << std::endl;
    }

    {
      // 核心对象：网格/装配器/电生理模型/线性求解器/wholebody 耦合组件。
      std::unique_ptr<mfem::Mesh> wholebody_serial_mesh;
      std::unique_ptr<mfem::ParMesh> wholebody_parent_mesh;
      std::unique_ptr<mfem::ParMesh> heart_pmesh_override;
      std::unique_ptr<mfem::ParMesh> torso_pmesh_override;
      if (cfg.enable_wholebody && cfg.use_conforming_wholebody) {
        // 单一 conforming wholebody 网格拆分成心脏与躯干两个 ParSubMesh。
        wholebody_serial_mesh = std::make_unique<mfem::Mesh>(cfg.wholebody_mesh_path.c_str(), 1, 1);
        wholebody_parent_mesh =
            std::make_unique<mfem::ParMesh>(MPI_COMM_WORLD, *wholebody_serial_mesh);

        mfem::Array<int> heart_attrs;
        heart_attrs.SetSize(static_cast<int>(cfg.heart_volume_attrs.size()));
        for (int i = 0; i < heart_attrs.Size(); ++i) {
          heart_attrs[i] = cfg.heart_volume_attrs[static_cast<size_t>(i)];
        }
        mfem::Array<int> torso_attrs;
        torso_attrs.SetSize(static_cast<int>(cfg.torso_volume_attrs.size()));
        for (int i = 0; i < torso_attrs.Size(); ++i) {
          torso_attrs[i] = cfg.torso_volume_attrs[static_cast<size_t>(i)];
        }

        auto heart_sub = mfem::ParSubMesh::CreateFromDomain(*wholebody_parent_mesh, heart_attrs);
        auto torso_sub = mfem::ParSubMesh::CreateFromDomain(*wholebody_parent_mesh, torso_attrs);
        heart_pmesh_override = std::make_unique<mfem::ParSubMesh>(std::move(heart_sub));
        torso_pmesh_override = std::make_unique<mfem::ParSubMesh>(std::move(torso_sub));

        if (rank == 0) {
          std::cout << "[wholebody-conforming] parent NE="
                    << wholebody_parent_mesh->GetNE() << ", heart NE="
                    << heart_pmesh_override->GetNE() << ", torso NE="
                    << torso_pmesh_override->GetNE() << std::endl;
        }
      }

      std::unique_ptr<mono::Assembler> assembler_ptr;
      if (heart_pmesh_override) {
        assembler_ptr =
            std::make_unique<mono::Assembler>(cfg, MPI_COMM_WORLD, std::move(heart_pmesh_override));
      } else {
        assembler_ptr = std::make_unique<mono::Assembler>(cfg, MPI_COMM_WORLD);
      }
      auto& assembler = *assembler_ptr;

      mono::TT06Model tt06(assembler.TrueVSize());
      tt06.InitializeRestState(-85.23);

      mono::LinearSystemSolver linear_solver(cfg, MPI_COMM_WORLD, &assembler.PFES());
      mono::MonodomainStepper stepper(cfg, assembler, tt06, linear_solver);

      std::unique_ptr<mono::ExtracellularRecoverySolver> ue_solver;
      std::unique_ptr<mono::TorsoPotentialSolver> torso_solver;
      std::unique_ptr<mono::InterfaceMapper> interface_mapper;
      mfem::Vector torso_bc_true;
      if (cfg.enable_wholebody) {
        ue_solver = std::make_unique<mono::ExtracellularRecoverySolver>(cfg, assembler, MPI_COMM_WORLD);
        if (torso_pmesh_override) {
          torso_solver = std::make_unique<mono::TorsoPotentialSolver>(
              cfg, MPI_COMM_WORLD, std::move(torso_pmesh_override));
        } else {
          torso_solver = std::make_unique<mono::TorsoPotentialSolver>(cfg, MPI_COMM_WORLD);
        }
        interface_mapper =
            std::make_unique<mono::InterfaceMapper>(cfg, assembler, *torso_solver, MPI_COMM_WORLD);
        torso_solver->SetConstrainedDofs(interface_mapper->TorsoConstrainedDofs());
        torso_bc_true.SetSize(torso_solver->PFES().GetTrueVSize());
        torso_bc_true = 0.0;
        if (rank == 0) {
          std::cout << "[wholebody] constrained torso dofs="
                    << interface_mapper->GlobalNumConstrainedDofs()
                    << ", max map dist(mm)="
                    << interface_mapper->GlobalMaxConstrainedDistMm() << std::endl;
        }
      }

      mono::OutputManager output(cfg, assembler, ue_solver.get(), torso_solver.get());
      mono::CheckpointIO checkpoint(cfg, MPI_COMM_WORLD);
      const bool output_enabled = (cfg.output_stride > 0);
      const bool checkpoint_enabled = (cfg.checkpoint_stride > 0);
      std::ofstream ksp_log;
      std::ofstream timing_log;

      if (rank == 0) {
        std::filesystem::create_directories(cfg.output_dir);
        const std::filesystem::path ksp_log_path =
            std::filesystem::path(cfg.output_dir) / "ksp_history.csv";
        const std::filesystem::path timing_log_path =
            std::filesystem::path(cfg.output_dir) / "timing_breakdown.csv";
        const bool append_mode = restart_from_checkpoint && std::filesystem::exists(ksp_log_path);
        ksp_log.open(ksp_log_path, append_mode ? std::ios::app : std::ios::trunc);
        if (!ksp_log) {
          throw std::runtime_error("failed to open KSP log file: " + ksp_log_path.string());
        }
        timing_log.open(timing_log_path, append_mode ? std::ios::app : std::ios::trunc);
        if (!timing_log) {
          throw std::runtime_error("failed to open timing log file: " + timing_log_path.string());
        }
        if (!append_mode) {
          ksp_log << "step,time_ms,iterations,final_norm,solver\n";
          timing_log << "step,time_ms,solver_step_total_ms,iion_ms,rhs_ms,linear_solve_ms,"
                        "sync_vm_ms,ode_advance_ms,other_ms,ue_solve_ms,map_ms,torso_solve_ms,"
                        "ue_iterations,torso_iterations,output_ms,checkpoint_ms\n";
        }
        ksp_log << std::setprecision(16);
        timing_log << std::setprecision(16);
      }

      // 采用 rank 间 max 归约，记录并行瓶颈时间（最慢 rank）。
      auto reduce_max = [](double local_value) {
        double global_value = 0.0;
        MPI_Allreduce(&local_value, &global_value, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        return global_value;
      };
      auto reduce_max_int = [](int local_value) {
        int global_value = 0;
        MPI_Allreduce(&local_value, &global_value, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        return global_value;
      };

      // 每一步写入收敛信息和分项耗时，便于 ASM/Boomer 等方案对比。
      auto log_step_metrics = [&](double output_ms_local,
                                  double checkpoint_ms_local,
                                  double ue_solve_ms_local,
                                  double map_ms_local,
                                  double torso_solve_ms_local,
                                  int ue_iter_local,
                                  int torso_iter_local) {
        const mono::StepTimingBreakdown& t = stepper.LastTiming();
        double other_ms_local =
            t.step_total_ms -
            (t.iion_ms + t.rhs_ms + t.linear_solve_ms + t.sync_vm_ms + t.ode_advance_ms);
        if (other_ms_local < 0.0) {
          other_ms_local = 0.0;
        }

        const double step_total_ms = reduce_max(t.step_total_ms);
        const double iion_ms = reduce_max(t.iion_ms);
        const double rhs_ms = reduce_max(t.rhs_ms);
        const double linear_solve_ms = reduce_max(t.linear_solve_ms);
        const double sync_vm_ms = reduce_max(t.sync_vm_ms);
        const double ode_advance_ms = reduce_max(t.ode_advance_ms);
        const double other_ms = reduce_max(other_ms_local);
        const double ue_solve_ms = reduce_max(ue_solve_ms_local);
        const double map_ms = reduce_max(map_ms_local);
        const double torso_solve_ms = reduce_max(torso_solve_ms_local);
        const int ue_iter = reduce_max_int(ue_iter_local);
        const int torso_iter = reduce_max_int(torso_iter_local);
        const double output_ms = reduce_max(output_ms_local);
        const double checkpoint_ms = reduce_max(checkpoint_ms_local);

        if (rank == 0 && ksp_log.is_open()) {
          ksp_log << stepper.StepCount() << "," << stepper.TimeMs() << ","
                  << linear_solver.LastNumIterations() << "," << linear_solver.LastFinalNorm()
                  << "," << (cfg.use_petsc ? "petsc" : "cg") << "\n";
        }
        if (rank == 0 && timing_log.is_open()) {
          timing_log << stepper.StepCount() << "," << stepper.TimeMs() << "," << step_total_ms
                     << "," << iion_ms << "," << rhs_ms << "," << linear_solve_ms << ","
                     << sync_vm_ms << "," << ode_advance_ms << "," << other_ms << ","
                     << ue_solve_ms << "," << map_ms << "," << torso_solve_ms << ","
                     << ue_iter << "," << torso_iter << ","
                     << output_ms << "," << checkpoint_ms << "\n";
        }
      };

      auto solve_wholebody_potentials = [&](double* ue_solve_ms = nullptr,
                                            double* map_ms = nullptr,
                                            double* torso_solve_ms = nullptr,
                                            int* ue_iter = nullptr,
                                            int* torso_iter = nullptr) {
        if (!cfg.enable_wholebody) {
          return;
        }
        using Clock = std::chrono::steady_clock;
        auto wb_t0 = Clock::now();
        ue_solver->Solve(stepper.VmTrue());
        const double ue_ms =
            std::chrono::duration<double, std::milli>(Clock::now() - wb_t0).count();
        if (ue_solve_ms != nullptr) {
          *ue_solve_ms = ue_ms;
        }
        if (ue_iter != nullptr) {
          *ue_iter = ue_solver->LastNumIterations();
        }

        wb_t0 = Clock::now();
        interface_mapper->MapHeartToTorso(ue_solver->UeTrue(), torso_bc_true);
        const double map_elapsed_ms =
            std::chrono::duration<double, std::milli>(Clock::now() - wb_t0).count();
        if (map_ms != nullptr) {
          *map_ms = map_elapsed_ms;
        }

        wb_t0 = Clock::now();
        torso_solver->Solve(torso_bc_true);
        const double torso_elapsed_ms =
            std::chrono::duration<double, std::milli>(Clock::now() - wb_t0).count();
        if (torso_solve_ms != nullptr) {
          *torso_solve_ms = torso_elapsed_ms;
        }
        if (torso_iter != nullptr) {
          *torso_iter = torso_solver->LastNumIterations();
        }
      };

      struct ActivationProbe {
        const char* name = "";
        std::array<double, 3> xyz{};
        int local_vdof = -1;
        double local_best_dist2 = std::numeric_limits<double>::infinity();
        bool local_active = false;
        double prev_vm = -85.23;
        bool activated = false;
        double activation_ms = std::numeric_limits<double>::quiet_NaN();
      };

      std::vector<ActivationProbe> probes;
      if (benchmark_probes) {
        probes = {
            {"P1", {0.0, 0.0, 0.0}},
            {"P2", {20.0, 0.0, 0.0}},
            {"P3", {0.0, 7.0, 0.0}},
            {"P4", {0.0, 0.0, 3.0}},
            {"P5", {20.0, 7.0, 0.0}},
            {"P6", {20.0, 0.0, 3.0}},
            {"P7", {0.0, 7.0, 3.0}},
            {"P8", {20.0, 7.0, 3.0}},
            {"P9", {10.0, 3.5, 1.5}},
        };

        const mfem::ParMesh* pmesh = assembler.PFES().GetParMesh();
        for (auto& probe : probes) {
          for (int v = 0; v < pmesh->GetNV(); ++v) {
            const double* x = pmesh->GetVertex(v);
            const double dx = x[0] - probe.xyz[0];
            const double dy = x[1] - probe.xyz[1];
            const double dz = x[2] - probe.xyz[2];
            const double d2 = dx * dx + dy * dy + dz * dz;
            if (d2 >= probe.local_best_dist2) {
              continue;
            }

            mfem::Array<int> vdofs;
            assembler.PFES().GetVertexDofs(v, vdofs);
            if (vdofs.Size() < 1) {
              continue;
            }

            const int dof = (vdofs[0] >= 0) ? vdofs[0] : (-1 - vdofs[0]);
            probe.local_best_dist2 = d2;
            probe.local_vdof = dof;
          }

          double global_best_dist2 = 0.0;
          MPI_Allreduce(
              &probe.local_best_dist2, &global_best_dist2, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
          const double tol =
              1e-14 * std::max(1.0, std::sqrt(std::max(global_best_dist2, 0.0)));
          probe.local_active =
              (probe.local_vdof >= 0) &&
              (std::abs(std::sqrt(std::max(probe.local_best_dist2, 0.0)) -
                        std::sqrt(std::max(global_best_dist2, 0.0))) <= tol);

          if (rank == 0) {
            std::cout << "[benchmark] probe " << probe.name
                      << " nearest-distance(mm)=" << std::sqrt(std::max(global_best_dist2, 0.0))
                      << std::endl;
          }
        }
      }

      auto sample_probe_vm = [&]() {
        std::vector<double> values(probes.size(), std::numeric_limits<double>::quiet_NaN());
        for (size_t i = 0; i < probes.size(); ++i) {
          double local_sum = 0.0;
          int local_count = 0;
          if (probes[i].local_active && probes[i].local_vdof >= 0) {
            local_sum = assembler.Vm()(probes[i].local_vdof);
            local_count = 1;
          }

          double global_sum = 0.0;
          int global_count = 0;
          MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
          MPI_Allreduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
          if (global_count > 0) {
            values[i] = global_sum / static_cast<double>(global_count);
          }
        }
        return values;
      };

      auto update_activation_times =
          [&](double t_prev_ms, double t_curr_ms, const std::vector<double>& vm_now) {
            for (size_t i = 0; i < probes.size(); ++i) {
              if (!std::isfinite(vm_now[i])) {
                continue;
              }
              if (!probes[i].activated && probes[i].prev_vm < 0.0 && vm_now[i] >= 0.0) {
                double t_act = t_curr_ms;
                const double dv = vm_now[i] - probes[i].prev_vm;
                if (std::abs(dv) > 1e-14) {
                  t_act = t_prev_ms + (-probes[i].prev_vm) * (t_curr_ms - t_prev_ms) / dv;
                }
                probes[i].activated = true;
                probes[i].activation_ms = t_act;
              }
              probes[i].prev_vm = vm_now[i];
            }
          };

      if (restart_from_checkpoint) {
        int restart_step = 0;
        double restart_t_ms = 0.0;
        if (!checkpoint.LoadLatest(restart_step, restart_t_ms, assembler.Vm(), tt06)) {
          throw std::runtime_error("restart requested but checkpoint is missing or incompatible with current MPI size");
        }
        stepper.InitializeFromCurrentVm(restart_step, restart_t_ms);
        if (rank == 0) {
          std::cout << "[monodomain] restart loaded at step = " << restart_step
                    << ", t = " << restart_t_ms << " ms" << std::endl;
        }

        if (benchmark_probes) {
          const auto vm0 = sample_probe_vm();
          for (size_t i = 0; i < probes.size(); ++i) {
            if (std::isfinite(vm0[i])) {
              probes[i].prev_vm = vm0[i];
            }
          }
        }

        solve_wholebody_potentials();
      } else {
        // 冷启动：从静息态先做一次 bootstrap。
        using Clock = std::chrono::steady_clock;
        stepper.InitializeVm(-85.23);

        if (benchmark_probes) {
          const auto vm0 = sample_probe_vm();
          for (size_t i = 0; i < probes.size(); ++i) {
            if (std::isfinite(vm0[i])) {
              probes[i].prev_vm = vm0[i];
            }
          }
        }

        const double t_prev_ms = stepper.TimeMs();
        stepper.Bootstrap();
        if (benchmark_probes) {
          update_activation_times(t_prev_ms, stepper.TimeMs(), sample_probe_vm());
        }

        double output_ms_local = 0.0;
        double checkpoint_ms_local = 0.0;
        double ue_solve_ms_local = 0.0;
        double map_ms_local = 0.0;
        double torso_solve_ms_local = 0.0;
        int ue_iter_local = 0;
        int torso_iter_local = 0;

        const bool solve_wholebody_bootstrap =
            cfg.enable_wholebody && (output_enabled || cfg.wholebody_solve_every_step);
        if (solve_wholebody_bootstrap) {
          solve_wholebody_potentials(&ue_solve_ms_local,
                                     &map_ms_local,
                                     &torso_solve_ms_local,
                                     &ue_iter_local,
                                     &torso_iter_local);
        }

        if (output_enabled) {
          auto io_t0 = Clock::now();
          output.Save(stepper.StepCount(),
                      stepper.TimeMs(),
                      stepper.IionTrue(),
                      (cfg.enable_wholebody ? &ue_solver->UeTrue() : nullptr),
                      (cfg.enable_wholebody ? &torso_solver->UTTrue() : nullptr));
          output_ms_local = std::chrono::duration<double, std::milli>(Clock::now() - io_t0).count();
        }

        if (checkpoint_enabled && stepper.StepCount() % cfg.checkpoint_stride == 0) {
          auto io_t0 = Clock::now();
          checkpoint.SaveLatest(stepper.StepCount(), stepper.TimeMs(), assembler.Vm(), tt06);
          checkpoint_ms_local =
              std::chrono::duration<double, std::milli>(Clock::now() - io_t0).count();
        }
        log_step_metrics(
            output_ms_local,
            checkpoint_ms_local,
            ue_solve_ms_local,
            map_ms_local,
            torso_solve_ms_local,
            ue_iter_local,
            torso_iter_local);
      }

      while (stepper.TimeMs() < cfg.t_end_ms) {
        using Clock = std::chrono::steady_clock;
        const double t_prev_ms = stepper.TimeMs();
        stepper.StepNoCorrection();
        if (benchmark_probes) {
          update_activation_times(t_prev_ms, stepper.TimeMs(), sample_probe_vm());
        }
        double output_ms_local = 0.0;
        double checkpoint_ms_local = 0.0;
        double ue_solve_ms_local = 0.0;
        double map_ms_local = 0.0;
        double torso_solve_ms_local = 0.0;
        int ue_iter_local = 0;
        int torso_iter_local = 0;
        const bool save_output_frame = output_enabled && stepper.StepCount() % cfg.output_stride == 0;
        const bool solve_wholebody_this_step =
            cfg.enable_wholebody && (cfg.wholebody_solve_every_step || save_output_frame);

        if (solve_wholebody_this_step) {
          solve_wholebody_potentials(&ue_solve_ms_local,
                                     &map_ms_local,
                                     &torso_solve_ms_local,
                                     &ue_iter_local,
                                     &torso_iter_local);
        }

        if (save_output_frame) {
          auto io_t0 = Clock::now();
          output.Save(stepper.StepCount(),
                      stepper.TimeMs(),
                      stepper.IionTrue(),
                      (cfg.enable_wholebody ? &ue_solver->UeTrue() : nullptr),
                      (cfg.enable_wholebody ? &torso_solver->UTTrue() : nullptr));
          output_ms_local = std::chrono::duration<double, std::milli>(Clock::now() - io_t0).count();
        }
        if (checkpoint_enabled && stepper.StepCount() % cfg.checkpoint_stride == 0) {
          auto io_t0 = Clock::now();
          checkpoint.SaveLatest(stepper.StepCount(), stepper.TimeMs(), assembler.Vm(), tt06);
          checkpoint_ms_local =
              std::chrono::duration<double, std::milli>(Clock::now() - io_t0).count();
        }
        log_step_metrics(
            output_ms_local,
            checkpoint_ms_local,
            ue_solve_ms_local,
            map_ms_local,
            torso_solve_ms_local,
            ue_iter_local,
            torso_iter_local);
      }

      struct FieldStats {
        double min = 0.0;
        double max = 0.0;
        double mean = 0.0;
        double l2 = 0.0;
        long long n = 0;
      };

      auto reduce_field_stats = [](const mfem::Vector& local_vec) {
        FieldStats stats;
        double local_min = std::numeric_limits<double>::infinity();
        double local_max = -std::numeric_limits<double>::infinity();
        double local_sum = 0.0;
        double local_sq_sum = 0.0;
        for (int i = 0; i < local_vec.Size(); ++i) {
          const double v = local_vec[i];
          local_min = std::min(local_min, v);
          local_max = std::max(local_max, v);
          local_sum += v;
          local_sq_sum += v * v;
        }
        long long local_n = local_vec.Size();
        double global_min = 0.0;
        double global_max = 0.0;
        double global_sum = 0.0;
        double global_sq_sum = 0.0;
        long long global_n = 0;
        MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_sq_sum, &global_sq_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_n, &global_n, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        stats.n = global_n;
        if (global_n > 0) {
          stats.min = global_min;
          stats.max = global_max;
          stats.mean = global_sum / static_cast<double>(global_n);
          stats.l2 = std::sqrt(global_sq_sum);
        }
        return stats;
      };

      mfem::Vector vm_true;
      assembler.Vm().GetTrueDofs(vm_true);
      const FieldStats vm_stats = reduce_field_stats(vm_true);
      const FieldStats iion_stats = reduce_field_stats(stepper.IionTrue());
      FieldStats ue_stats;
      FieldStats ut_stats;
      if (cfg.enable_wholebody) {
        // whole-body 是 Vm 的后处理；这里补一次终值求解，保证终值统计与最终 Vm 对齐。
        solve_wholebody_potentials();
        ue_stats = reduce_field_stats(ue_solver->UeTrue());
        ut_stats = reduce_field_stats(torso_solver->UTTrue());
      }

      if (rank == 0) {
        std::cout << "[monodomain] finished at t = " << stepper.TimeMs() << " ms" << std::endl;
        if (vm_stats.n > 0) {
          std::cout << "[monodomain] Vm stats: min=" << vm_stats.min << ", max=" << vm_stats.max
                    << ", mean=" << vm_stats.mean << ", l2=" << vm_stats.l2 << std::endl;
          std::cout << "[monodomain] Iion stats: min=" << iion_stats.min
                    << ", max=" << iion_stats.max << ", mean=" << iion_stats.mean
                    << ", l2=" << iion_stats.l2 << std::endl;
          if (cfg.enable_wholebody) {
            std::cout << "[monodomain] ue stats: min=" << ue_stats.min
                      << ", max=" << ue_stats.max << ", mean=" << ue_stats.mean
                      << ", l2=" << ue_stats.l2 << std::endl;
            std::cout << "[monodomain] uT stats: min=" << ut_stats.min
                      << ", max=" << ut_stats.max << ", mean=" << ut_stats.mean
                      << ", l2=" << ut_stats.l2 << std::endl;
          }
        }
        if (benchmark_probes) {
          std::cout << "[benchmark] activation times (first Vm crossing 0 mV)" << std::endl;
          for (const auto& probe : probes) {
            std::cout << "[benchmark] " << probe.name << "=";
            if (probe.activated && std::isfinite(probe.activation_ms)) {
              std::cout << probe.activation_ms << " ms";
            } else {
              std::cout << "NA";
            }
            std::cout << std::endl;
          }
        }
      }
    }

#ifdef MFEM_USE_PETSC
    if (petsc_initialized) {
      mfem::MFEMFinalizePetsc();
    }
#endif
    if (!hypre_finalized) {
      mfem::Hypre::Finalize();
      hypre_finalized = true;
    }
  } catch (const std::exception& ex) {
#ifdef MFEM_USE_PETSC
    if (petsc_initialized) {
      mfem::MFEMFinalizePetsc();
    }
#endif
    if (!hypre_finalized) {
      mfem::Hypre::Finalize();
      hypre_finalized = true;
    }
    if (rank == 0) {
      std::cerr << "[monodomain] fatal: " << ex.what() << std::endl;
    }
    return 1;
  }

  return 0;
}
