#pragma once

#include <memory>

#include "mfem.hpp"

#include "config/SimulationConfig.hpp"

namespace mono {

class Assembler;
class IonicModel;
class Land2017Model;
class MechanicsSolver;
class LinearSystemSolver;

// Orchestrates the electromechanical coupling each EP step:
//   * pulls [Ca2+]_i from the ionic model;
//   * advances Land2017 to update active tension T_a;
//   * every 'mech_substep' EP steps, drives MechanicsSolver::Solve();
//   * after a mechanics solve, signals Assembler to rebuild system matrices
//     using the deformation-modified conductivity.
//
// All operations are silent no-ops if mechanics_enable == false.
class EMCoupler {
 public:
  EMCoupler(const SimulationConfig& cfg,
            Assembler& assembler,
            IonicModel& ionic,
            Land2017Model& land,
            MechanicsSolver& mech,
            LinearSystemSolver& linear);

  // Called from MonodomainStepper at the end of each step, after ionic
  // and Purkinje advances. Returns true iff mechanics was re-solved.
  bool OnStep(int step_idx, double t_ms);

  // Active tension grid function exposed for the MechanicsSolver / output.
  mfem::ParGridFunction& ActiveTension() { return *ta_gf_; }

 private:
  const SimulationConfig& cfg_;
  Assembler& assembler_;
  IonicModel& ionic_;
  Land2017Model& land_;
  MechanicsSolver& mech_;
  LinearSystemSolver& linear_;

  std::unique_ptr<mfem::ParGridFunction> ta_gf_;
  std::unique_ptr<mfem::ParGridFunction> lambda_gf_;
  mfem::Vector cai_buf_;
  mfem::Vector ta_buf_;
  mfem::Vector lambda_buf_;
};

}  // namespace mono
