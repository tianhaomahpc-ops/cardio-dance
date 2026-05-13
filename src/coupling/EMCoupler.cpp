#include "coupling/EMCoupler.hpp"

#include "mechanics/MechanicsSolver.hpp"
#include "ode/IonicModel.hpp"
#include "ode/Land2017Model.hpp"
#include "solver/LinearSolverFactory.hpp"
#include "space/Assembler.hpp"

namespace mono {

EMCoupler::EMCoupler(const SimulationConfig& cfg,
                     Assembler& assembler,
                     IonicModel& ionic,
                     Land2017Model& land,
                     MechanicsSolver& mech,
                     LinearSystemSolver& linear)
    : cfg_(cfg),
      assembler_(assembler),
      ionic_(ionic),
      land_(land),
      mech_(mech),
      linear_(linear) {
  ta_gf_ = std::make_unique<mfem::ParGridFunction>(&assembler_.PFES());
  *ta_gf_ = 0.0;
  lambda_gf_ = std::make_unique<mfem::ParGridFunction>(&assembler_.PFES());
  *lambda_gf_ = 1.0;
  mech_.SetActiveTension(*ta_gf_);
}

bool EMCoupler::OnStep(int step_idx, double t_ms) {
  if (!cfg_.mechanics_enable) return false;

  // 1) Pull [Ca2+]_i.
  if (!ionic_.GetCytosolicCalcium(cai_buf_)) {
    return false;  // model does not expose Ca; cannot couple.
  }

  // 2) Pull current sarcomere stretch lambda from displacement (1.0 if no solve yet).
  mech_.ComputeFiberStretch(*lambda_gf_);
  lambda_gf_->GetTrueDofs(lambda_buf_);
  if (lambda_buf_.Size() == land_.NumNodes()) {
    land_.SetStretch(lambda_buf_);
  }

  // 3) Advance Land 2017 by one EP timestep.
  land_.Advance(cfg_.dt_pde_ms, cfg_.dt_ode_ms, cai_buf_);

  // 4) Refresh T_a grid function.
  land_.GetTension(ta_buf_);
  ta_gf_->SetFromTrueDofs(ta_buf_);

  // 5) Every mech_substep EP steps, solve mechanics and refresh K/A/B + linear op.
  const bool do_mech = (cfg_.mech_substep > 0) &&
                       ((step_idx % cfg_.mech_substep) == 0) && (step_idx > 0);
  if (do_mech) {
    mech_.SetActiveTension(*ta_gf_);
    mech_.Solve();
    // Rebuild K/A/B with deformation-modified diffusion tensor.
    assembler_.BuildSystemMatrices(cfg_.dt_pde_ms);
    linear_.InvalidatePetscOperator();
    linear_.SetOperator(assembler_.A());
    return true;
  }
  return false;
}

}  // namespace mono
