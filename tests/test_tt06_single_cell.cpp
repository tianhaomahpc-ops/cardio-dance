#include <cmath>
#include <iostream>

#include "mfem.hpp"

#include "ode/TT06Model.hpp"

int main() {
  // Single-cell stability check: ensure Iion remains finite over many updates.
  mono::TT06Model model(1);
  model.InitializeRestState(-85.23);

  mfem::Vector vm(1);
  vm[0] = -85.23;

  mfem::Vector iion(1);

  for (int n = 0; n < 500; ++n) {
    model.ComputeIion(vm, iion);
    if (!std::isfinite(iion[0])) {
      std::cerr << "Iion is not finite at step " << n << std::endl;
      return 1;
    }
    model.AdvanceStates(0.02, 0.01, vm);
  }

  return 0;
}
