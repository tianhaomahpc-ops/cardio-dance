#include "solver/MonodomainStepper.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

namespace mono {
namespace {

bool PointInStimulusRegion(const StimulusRegion& reg, const mfem::Vector& x, int dim) {
  if (reg.type == StimulusRegion::Type::Ball) {
    const double dx = x[0] - reg.x1;
    const double dy = (dim >= 2) ? (x[1] - reg.y1) : 0.0;
    const double dz = (dim >= 3) ? (x[2] - reg.z1) : 0.0;
    return (dx * dx + dy * dy + dz * dz) <= reg.x2 * reg.x2;
  }

  const double xmin = std::min(reg.x1, reg.x2);
  const double xmax = std::max(reg.x1, reg.x2);
  const double ymin = std::min(reg.y1, reg.y2);
  const double ymax = std::max(reg.y1, reg.y2);
  const double zmin = std::min(reg.z1, reg.z2);
  const double zmax = std::max(reg.z1, reg.z2);
  const bool in_x = (x[0] >= xmin) && (x[0] <= xmax);
  const bool in_y = (dim < 2) || ((x[1] >= ymin) && (x[1] <= ymax));
  const bool in_z = (dim < 3) || ((x[2] >= zmin) && (x[2] <= zmax));
  return in_x && in_y && in_z;
}

}  // namespace

MonodomainStepper::MonodomainStepper(const SimulationConfig& cfg,
                                     Assembler& assembler,
                                     IonicModel& ionic_model,
                                     LinearSystemSolver& linear_solver)
    : cfg_(cfg),
      assembler_(assembler),
      ionic_model_(ionic_model),
      linear_solver_(linear_solver) {
  const int n = assembler_.TrueVSize();
  vm_n_.SetSize(n);
  vm_np1_.SetSize(n);
  iion_true_.SetSize(n);
  rhs_.SetSize(n);
  tmp_.SetSize(n);
  stim_mask_true_.SetSize(n);
  stim_true_.SetSize(n);

  assembler_.Vm().GetTrueDofs(vm_n_);
  linear_solver_.SetOperator(assembler_.A());
  BuildStimulusMask();
}

void MonodomainStepper::InitializeVm(double v_init_mv) {
  assembler_.Vm() = v_init_mv;
  assembler_.Vm().GetTrueDofs(vm_n_);
  SyncVmToGridFunction(vm_n_);
  t_ms_ = 0.0;
  step_ = 0;
}

void MonodomainStepper::InitializeFromCurrentVm(int step, double t_ms) {
  assembler_.Vm().GetTrueDofs(vm_n_);
  SyncVmToGridFunction(vm_n_);
  step_ = step;
  t_ms_ = t_ms;
}

void MonodomainStepper::BuildStimulusMask() {
  // Precompute spatial support once; only amplitude/time window changes per step.
  stim_mask_true_ = 0.0;

  const mfem::ParMesh* pmesh = assembler_.PFES().GetParMesh();
  mfem::ParGridFunction stim_mask_gf(&assembler_.PFES());
  const int dim = pmesh->Dimension();

  if (!cfg_.stim_regions.empty()) {
    mfem::FunctionCoefficient mask_coeff([this, dim](const mfem::Vector& x) {
      for (const auto& reg : cfg_.stim_regions) {
        if (PointInStimulusRegion(reg, x, dim)) {
          return 1.0;
        }
      }
      return 0.0;
    });
    stim_mask_gf.ProjectCoefficient(mask_coeff);
    stim_mask_gf.GetTrueDofs(stim_mask_true_);
    return;
  }

  const bool use_box = (cfg_.stim_xmax_mm > cfg_.stim_xmin_mm) &&
                       (cfg_.stim_ymax_mm > cfg_.stim_ymin_mm) &&
                       (cfg_.stim_zmax_mm > cfg_.stim_zmin_mm);

  if (use_box) {
    mfem::FunctionCoefficient mask_coeff([this, dim](const mfem::Vector& x) {
      const bool in_x = (x[0] >= cfg_.stim_xmin_mm) && (x[0] <= cfg_.stim_xmax_mm);
      const bool in_y = (dim < 2) || ((x[1] >= cfg_.stim_ymin_mm) && (x[1] <= cfg_.stim_ymax_mm));
      const bool in_z = (dim < 3) || ((x[2] >= cfg_.stim_zmin_mm) && (x[2] <= cfg_.stim_zmax_mm));
      return (in_x && in_y && in_z) ? 1.0 : 0.0;
    });
    stim_mask_gf.ProjectCoefficient(mask_coeff);
    stim_mask_gf.GetTrueDofs(stim_mask_true_);
    return;
  }

  const int nv_local = pmesh->GetNV();
  double x_min_local = std::numeric_limits<double>::infinity();
  double x_max_local = -std::numeric_limits<double>::infinity();
  for (int i = 0; i < nv_local; ++i) {
    const double* v = pmesh->GetVertex(i);
    x_min_local = std::min(x_min_local, v[0]);
    x_max_local = std::max(x_max_local, v[0]);
  }

  const MPI_Comm comm = assembler_.PFES().GetComm();
  double x_min = 0.0;
  double x_max = 0.0;
  MPI_Allreduce(&x_min_local, &x_min, 1, MPI_DOUBLE, MPI_MIN, comm);
  MPI_Allreduce(&x_max_local, &x_max, 1, MPI_DOUBLE, MPI_MAX, comm);

  const double frac = std::clamp(cfg_.stim_fraction, 0.0, 1.0);
  if (frac <= 0.0 || !(x_max > x_min)) {
    return;
  }
  const double x_cut = x_min + frac * (x_max - x_min);
  mfem::FunctionCoefficient mask_coeff([x_cut](const mfem::Vector& x) {
    return (x[0] <= x_cut) ? 1.0 : 0.0;
  });
  stim_mask_gf.ProjectCoefficient(mask_coeff);
  stim_mask_gf.GetTrueDofs(stim_mask_true_);
}

void MonodomainStepper::BuildStimulus(double t_mid_ms) {
  stim_true_ = 0.0;
  if (t_mid_ms < cfg_.stim_start_ms || t_mid_ms > cfg_.stim_end_ms) {
    return;
  }

  // TT06 sign convention: negative stimulus is depolarizing.
  stim_true_ = stim_mask_true_;
  stim_true_ *= cfg_.stim_amp;
}

void MonodomainStepper::BuildRhs(double t_mid_ms) {
  // RHS = B*Vn - chi*Cm*M*(I_ion + I_stim)
  const double react_scale = cfg_.chi_per_mm * cfg_.cm_uF_per_mm2;
  assembler_.B().Mult(vm_n_, rhs_);

  assembler_.M().Mult(iion_true_, tmp_);
  rhs_.Add(-react_scale, tmp_);

  BuildStimulus(t_mid_ms);
  assembler_.M().Mult(stim_true_, tmp_);
  rhs_.Add(-react_scale, tmp_);
}

void MonodomainStepper::SyncVmToGridFunction(const mfem::Vector& vm_true) {
  assembler_.Vm().SetFromTrueDofs(vm_true);
}

void MonodomainStepper::Bootstrap() {
  using Clock = std::chrono::steady_clock;
  auto t0 = Clock::now();

  // 1) Evaluate ionic source at V^n/state^n.
  auto t_begin = Clock::now();
  ionic_model_.ComputeIion(vm_n_, iion_true_);
  auto t_end = Clock::now();
  last_timing_.iion_ms = std::chrono::duration<double, std::milli>(t_end - t_begin).count();

  // 2) Advance ionic states first (ODE -> PDE coupling order).
  t_begin = Clock::now();
  ionic_model_.AdvanceStates(cfg_.dt_pde_ms, cfg_.dt_ode_ms, vm_n_);
  t_end = Clock::now();
  last_timing_.ode_advance_ms = std::chrono::duration<double, std::milli>(t_end - t_begin).count();

  // 3) Build RHS with current ionic source.
  t_begin = Clock::now();
  BuildRhs(0.5 * cfg_.dt_pde_ms);
  t_end = Clock::now();
  last_timing_.rhs_ms = std::chrono::duration<double, std::milli>(t_end - t_begin).count();

  // 4) PDE solve for V^{n+1}.
  t_begin = Clock::now();
  vm_np1_ = vm_n_;
  linear_solver_.Solve(rhs_, vm_np1_);
  t_end = Clock::now();
  last_timing_.linear_solve_ms = std::chrono::duration<double, std::milli>(t_end - t_begin).count();

  // 5) Sync field representation.
  t_begin = Clock::now();
  SyncVmToGridFunction(vm_np1_);
  t_end = Clock::now();
  last_timing_.sync_vm_ms = std::chrono::duration<double, std::milli>(t_end - t_begin).count();

  vm_n_ = vm_np1_;
  t_ms_ = cfg_.dt_pde_ms;
  step_ = 1;
  last_timing_.step_total_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

void MonodomainStepper::StepNoCorrection() {
  using Clock = std::chrono::steady_clock;
  auto t0 = Clock::now();

  // 1) I_ion(V^n, state^n)
  auto t_begin = Clock::now();
  ionic_model_.ComputeIion(vm_n_, iion_true_);
  auto t_end = Clock::now();
  last_timing_.iion_ms = std::chrono::duration<double, std::milli>(t_end - t_begin).count();

  // 2) Advance ionic states first (ODE -> PDE coupling order).
  t_begin = Clock::now();
  ionic_model_.AdvanceStates(cfg_.dt_pde_ms, cfg_.dt_ode_ms, vm_n_);
  t_end = Clock::now();
  last_timing_.ode_advance_ms = std::chrono::duration<double, std::milli>(t_end - t_begin).count();

  const double t_mid = t_ms_ + 0.5 * cfg_.dt_pde_ms;

  // 3) Build split RHS.
  t_begin = Clock::now();
  BuildRhs(t_mid);
  t_end = Clock::now();
  last_timing_.rhs_ms = std::chrono::duration<double, std::milli>(t_end - t_begin).count();

  // 4) Linear PDE solve for V^{n+1}.
  t_begin = Clock::now();
  vm_np1_ = vm_n_;
  linear_solver_.Solve(rhs_, vm_np1_);
  t_end = Clock::now();
  last_timing_.linear_solve_ms = std::chrono::duration<double, std::milli>(t_end - t_begin).count();

  // 5) Sync field representation.
  t_begin = Clock::now();
  SyncVmToGridFunction(vm_np1_);
  t_end = Clock::now();
  last_timing_.sync_vm_ms = std::chrono::duration<double, std::milli>(t_end - t_begin).count();

  vm_n_ = vm_np1_;
  t_ms_ += cfg_.dt_pde_ms;
  ++step_;
  last_timing_.step_total_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

}  // namespace mono
