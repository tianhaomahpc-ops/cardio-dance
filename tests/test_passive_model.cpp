#include <cmath>
#include <iostream>

#include "mfem.hpp"

#include "ode/PassiveModel.hpp"

int main() {
  try {
    mono::PassiveModel model(3, -85.0, 6.0643e-4);
    model.InitializeRestState(-85.0);

    mfem::Vector vm(3);
    vm[0] = -85.0;
    vm[1] = -75.0;
    vm[2] = -95.0;

    mfem::Vector iion;
    model.ComputeIion(vm, iion);
    if (iion.Size() != 3) {
      return 1;
    }
    if (std::abs(iion[0]) > 1e-12) {
      return 2;
    }
    if (!(iion[1] > 0.0 && iion[2] < 0.0)) {
      return 3;
    }

    model.AdvanceStates(0.02, 0.01, vm);
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << std::endl;
    return 4;
  }

  return 0;
}
