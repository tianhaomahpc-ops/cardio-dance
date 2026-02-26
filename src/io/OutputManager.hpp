#pragma once

#include <memory>

#include "mfem.hpp"

#include "config/SimulationConfig.hpp"
#include "solver/ExtracellularRecoverySolver.hpp"
#include "solver/TorsoPotentialSolver.hpp"
#include "space/Assembler.hpp"

namespace mono {

// Handles ParaView output of Vm/Iion and optional fiber vectors.
class OutputManager {
 public:
  OutputManager(const SimulationConfig& cfg,
                Assembler& assembler,
                ExtracellularRecoverySolver* ue_solver = nullptr,
                TorsoPotentialSolver* torso_solver = nullptr);

  // Save one output frame at cycle/time.
  void Save(int step,
            double t_ms,
            const mfem::Vector& iion_true,
            const mfem::Vector* ue_true = nullptr,
            const mfem::Vector* ut_true = nullptr);

 private:
  Assembler& assembler_;
  ExtracellularRecoverySolver* ue_solver_ = nullptr;
  TorsoPotentialSolver* torso_solver_ = nullptr;
  std::unique_ptr<mfem::ParGridFunction> iion_gf_;
  std::unique_ptr<mfem::ParGridFunction> ue_gf_;
  std::unique_ptr<mfem::ParGridFunction> ut_gf_;
  std::unique_ptr<mfem::ParFiniteElementSpace> fiber_out_fes_;
  std::unique_ptr<mfem::ParGridFunction> fiber_f_out_gf_;
  std::unique_ptr<mfem::ParGridFunction> fiber_s_out_gf_;
  std::unique_ptr<mfem::ParGridFunction> fiber_n_out_gf_;
  std::unique_ptr<mfem::ParaViewDataCollection> heart_dc_;
  std::unique_ptr<mfem::ParaViewDataCollection> torso_dc_;
};

}  // namespace mono
