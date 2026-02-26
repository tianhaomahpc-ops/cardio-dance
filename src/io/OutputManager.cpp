#include "io/OutputManager.hpp"

#include <filesystem>

namespace mono {

OutputManager::OutputManager(const SimulationConfig& cfg,
                             Assembler& assembler,
                             ExtracellularRecoverySolver* ue_solver,
                             TorsoPotentialSolver* torso_solver)
    : assembler_(assembler), ue_solver_(ue_solver), torso_solver_(torso_solver) {
  std::filesystem::create_directories(cfg.output_dir);

  iion_gf_ = std::make_unique<mfem::ParGridFunction>(&assembler_.PFES());
  *iion_gf_ = 0.0;
  if (ue_solver_ != nullptr) {
    ue_gf_ = std::make_unique<mfem::ParGridFunction>(&assembler_.PFES());
    *ue_gf_ = 0.0;
  }

  heart_dc_ =
      std::make_unique<mfem::ParaViewDataCollection>("monodomain", assembler_.PFES().GetParMesh());
  heart_dc_->SetPrefixPath((std::filesystem::path(cfg.output_dir) / "heart").string());
  heart_dc_->SetHighOrderOutput(false);
  heart_dc_->SetDataFormat(mfem::VTKFormat::BINARY);
  heart_dc_->SetLevelsOfDetail(1);
  heart_dc_->RegisterField("Vm", &assembler_.Vm());
  heart_dc_->RegisterField("Iion", iion_gf_.get());
  if (ue_gf_) {
    heart_dc_->RegisterField("ue", ue_gf_.get());
  }
  if (assembler_.HasFiberFields()) {
    // Export fibers in byNODES ordering for stable visualization in ParaView.
    const int dim = assembler_.PFES().GetParMesh()->Dimension();
    fiber_out_fes_ = std::make_unique<mfem::ParFiniteElementSpace>(
        assembler_.PFES().GetParMesh(), assembler_.PFES().FEColl(), dim, mfem::Ordering::byNODES);
    fiber_f_out_gf_ = std::make_unique<mfem::ParGridFunction>(fiber_out_fes_.get());
    fiber_s_out_gf_ = std::make_unique<mfem::ParGridFunction>(fiber_out_fes_.get());
    fiber_n_out_gf_ = std::make_unique<mfem::ParGridFunction>(fiber_out_fes_.get());

    mfem::VectorGridFunctionCoefficient f_coeff(assembler_.FiberF());
    mfem::VectorGridFunctionCoefficient s_coeff(assembler_.FiberS());
    mfem::VectorGridFunctionCoefficient n_coeff(assembler_.FiberN());
    fiber_f_out_gf_->ProjectCoefficient(f_coeff);
    fiber_s_out_gf_->ProjectCoefficient(s_coeff);
    fiber_n_out_gf_->ProjectCoefficient(n_coeff);

    heart_dc_->RegisterField("fiber_f", fiber_f_out_gf_.get());
    heart_dc_->RegisterField("fiber_s", fiber_s_out_gf_.get());
    heart_dc_->RegisterField("fiber_n", fiber_n_out_gf_.get());
  }

  if (torso_solver_ != nullptr) {
    ut_gf_ = std::make_unique<mfem::ParGridFunction>(&torso_solver_->PFES());
    *ut_gf_ = 0.0;
    torso_dc_ =
        std::make_unique<mfem::ParaViewDataCollection>("torso_potential", torso_solver_->PFES().GetParMesh());
    torso_dc_->SetPrefixPath((std::filesystem::path(cfg.output_dir) / "torso").string());
    torso_dc_->SetHighOrderOutput(false);
    torso_dc_->SetDataFormat(mfem::VTKFormat::BINARY);
    torso_dc_->SetLevelsOfDetail(1);
    torso_dc_->RegisterField("uT", ut_gf_.get());
  }
}

void OutputManager::Save(int step,
                         double t_ms,
                         const mfem::Vector& iion_true,
                         const mfem::Vector* ue_true,
                         const mfem::Vector* ut_true) {
  // Convert true-dof vector back to grid function for visualization output.
  iion_gf_->SetFromTrueDofs(iion_true);
  if (ue_gf_ && ue_true != nullptr) {
    ue_gf_->SetFromTrueDofs(*ue_true);
  }
  if (ut_gf_ && ut_true != nullptr) {
    ut_gf_->SetFromTrueDofs(*ut_true);
  }

  heart_dc_->SetCycle(step);
  heart_dc_->SetTime(t_ms);
  heart_dc_->Save();
  if (torso_dc_) {
    torso_dc_->SetCycle(step);
    torso_dc_->SetTime(t_ms);
    torso_dc_->Save();
  }
}

}  // namespace mono
