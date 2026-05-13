#pragma once

#include <iosfwd>

#include "mfem.hpp"

namespace mono {

// Common ionic-model interface used by the monodomain stepper.
class IonicModel {
 public:
  virtual ~IonicModel() = default;

  virtual void InitializeRestState(double v_rest_mv) = 0;
  virtual void ComputeIion(const mfem::Vector& vm_true, mfem::Vector& iion_true) const = 0;
  virtual void AdvanceStates(double dt_pde_ms, double dt_ode_ms, const mfem::Vector& vm_next_true) = 0;

  virtual void SaveState(std::ostream& os) const = 0;
  virtual void LoadState(std::istream& is) = 0;

  virtual int NumNodes() const = 0;
  virtual const char* ModelTag() const = 0;

  // Optional: cytosolic [Ca2+] in mM. Returns false if not exposed.
  // Default no-op so non-EM-capable models compile unchanged.
  virtual bool GetCytosolicCalcium(mfem::Vector& /*cai_true_mM*/) const { return false; }
};

}  // namespace mono
