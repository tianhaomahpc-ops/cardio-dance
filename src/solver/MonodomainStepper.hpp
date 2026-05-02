#pragma once

#include "config/SimulationConfig.hpp"
#include "ode/IIonicModel.hpp"
#include "solver/LinearSolverFactory.hpp"
#include "space/Assembler.hpp"

namespace mono {

class PvjCoupler;

// Per-step wall-time breakdown for solver and reaction workflow.
struct StepTimingBreakdown {
  double iion_ms = 0.0;
  double rhs_ms = 0.0;
  double linear_solve_ms = 0.0;
  double sync_vm_ms = 0.0;
  double ode_advance_ms = 0.0;
  double pvj_ms = 0.0;
  double step_total_ms = 0.0;
};

// Operator-splitting stepper for monodomain:
// 1) compute I_ion
// 2) build RHS (reaction + stimulus + optional PVJ coupling current)
// 3) solve linear PDE step
// 4) advance ODE states.
class MonodomainStepper {
 public:
  MonodomainStepper(const SimulationConfig& cfg,
                    Assembler& assembler,
                    IIonicModel& ionic,
                    LinearSystemSolver& linear_solver);

  // Wire an optional PVJ coupler whose contribution is added to the heart RHS
  // each step. Pass nullptr to disable.
  void SetPvjCoupler(PvjCoupler* coupler) { pvj_coupler_ = coupler; }

  void InitializeVm(double v_init_mv);
  void InitializeFromCurrentVm(int step, double t_ms);
  void Bootstrap();
  void StepNoCorrection();

  double TimeMs() const { return t_ms_; }
  int StepCount() const { return step_; }

  const mfem::Vector& VmTrue() const { return vm_n_; }
  const mfem::Vector& IionTrue() const { return iion_true_; }
  // Timing from the most recent Bootstrap/StepNoCorrection call.
  const StepTimingBreakdown& LastTiming() const { return last_timing_; }

 private:
  const SimulationConfig& cfg_;
  Assembler& assembler_;
  IIonicModel& ionic_;
  LinearSystemSolver& linear_solver_;
  PvjCoupler* pvj_coupler_ = nullptr;

  double t_ms_ = 0.0;
  int step_ = 0;

  mfem::Vector vm_n_;
  mfem::Vector vm_np1_;
  mfem::Vector iion_true_;
  mfem::Vector rhs_;
  mfem::Vector tmp_;
  mfem::Vector stim_mask_true_;
  mfem::Vector stim_true_;
  mfem::Vector pvj_current_true_;
  StepTimingBreakdown last_timing_;

  void BuildStimulusMask();
  void BuildStimulus(double t_mid_ms);
  void BuildRhs(double t_mid_ms);
  void SyncVmToGridFunction(const mfem::Vector& vm_true);
};

}  // namespace mono
