#pragma once

#include <iosfwd>
#include <string>

#include "mfem.hpp"

namespace mono {

// Abstract per-true-DOF ionic model. Concrete implementations (TT06, Stewart,
// Grandi2011, Passive, RegionalIonicModel) plug into MonodomainStepper through
// this contract.
class IIonicModel {
 public:
  virtual ~IIonicModel() = default;

  // Initialize each managed node to its rest state (e.g., V_m = v_rest_mv).
  virtual void InitializeRestState(double v_rest_mv) = 0;

  // Evaluate I_ion at the supplied membrane voltage without mutating state.
  virtual void ComputeIion(const mfem::Vector& vm_true, mfem::Vector& iion_true) const = 0;

  // Advance internal ODE state across one PDE step using sub-stepping <= dt_ode_ms.
  virtual void AdvanceStates(double dt_pde_ms, double dt_ode_ms,
                             const mfem::Vector& vm_next_true) = 0;

  // Binary checkpoint I/O for restart support.
  virtual void SaveState(std::ostream& os) const = 0;
  virtual void LoadState(std::istream& is) = 0;

  virtual int NumNodes() const = 0;

  // Stable string identifier used by CheckpointIO to refuse cross-model
  // restarts. Must be unique per concrete model type.
  virtual std::string ModelId() const = 0;
};

}  // namespace mono
