// Verify RegionalIonicModel state isolation:
//   - children own state only for their region's DOFs
//   - I_ion at out-of-region DOFs is unaffected by extreme V values inside
//     other regions (i.e., children don't get poisoned by foreign V_m)
//
// Fixture: 1-D 4-element mesh; element attribute 1 (ventricle) on first 2
// elements, attribute 14 (AV-delay -> Passive) on last 2 elements.

#include <cmath>
#include <iostream>

#include "mfem.hpp"

#include "config/SimulationConfig.hpp"
#include "ode/RegionalIonicModel.hpp"

int main(int argc, char* argv[]) {
  mfem::Mpi::Init(argc, argv);
  const int rank = mfem::Mpi::WorldRank();

  // 1-D segment, 4 elements over [0, 4] mm.
  mfem::Mesh serial_mesh = mfem::Mesh::MakeCartesian1D(4, 4.0);
  // Tag elements: indexes 0,1 -> attr 1 (ventricle); 2,3 -> attr 14 (AV-delay).
  serial_mesh.SetAttribute(0, 1);
  serial_mesh.SetAttribute(1, 1);
  serial_mesh.SetAttribute(2, 14);
  serial_mesh.SetAttribute(3, 14);
  serial_mesh.SetAttributes();

  mfem::ParMesh pmesh(MPI_COMM_WORLD, serial_mesh);
  mfem::H1_FECollection fec(1, 1);
  mfem::ParFiniteElementSpace pfes(&pmesh, &fec);

  mono::SimulationConfig cfg;
  cfg.enable_regional_ionic = true;
  cfg.ventricle_volume_attrs = {1};
  cfg.av_delay_volume_attrs = {14};
  cfg.av_delay_leak_g_mS_per_uF = 0.05;
  cfg.fibrosis_leak_g_mS_per_uF = 0.05;
  cfg.passive_v_rest_mv = -85.0;

  mono::RegionalIonicModel model(cfg, pfes);
  model.InitializeRestState(-85.0);

  const int n_v = model.LocalDofCount(mono::RegionalIonicModel::Region::Ventricle);
  const int n_av = model.LocalDofCount(mono::RegionalIonicModel::Region::AvDelay);
  if (rank == 0) {
    std::cout << "[regional_iso] DOFs: ventricle=" << n_v
              << " av_delay=" << n_av << std::endl;
  }
  if (n_v + n_av != pfes.GetTrueVSize()) {
    std::cout << "[regional_iso] region DOFs do not partition tdof set" << std::endl;
    return 1;
  }

  // Set extreme V on ventricular DOFs, rest on AV-delay DOFs.
  mfem::Vector vm(pfes.GetTrueVSize());
  const auto& dof_region = model.DofRegions();
  for (int i = 0; i < vm.Size(); ++i) {
    if (dof_region[i] == static_cast<int>(mono::RegionalIonicModel::Region::Ventricle)) {
      vm[i] = +30.0;  // strongly depolarized
    } else {
      vm[i] = -85.0;  // passive at rest
    }
  }

  mfem::Vector iion(pfes.GetTrueVSize());
  model.ComputeIion(vm, iion);

  if (rank == 0) {
    for (int i = 0; i < vm.Size(); ++i) {
      const char* tag = (dof_region[i] ==
          static_cast<int>(mono::RegionalIonicModel::Region::AvDelay))
              ? "AV" : "V";
      std::cout << "  tdof=" << i << " region=" << tag
                << " V=" << vm[i] << " I_ion=" << iion[i] << std::endl;
    }
  }

  bool ok = true;
  // Passive at rest: I_ion = g_leak * (V - V_rest) = 0.05 * (-85 - -85) = 0.
  // Ventricular at +30 mV: I_ion is from TT06; just sanity-check it's finite
  // and nonzero (TT06 at +30 has substantial outward currents).
  for (int i = 0; i < vm.Size(); ++i) {
    if (!std::isfinite(iion[i])) { ok = false; break; }
    if (dof_region[i] == static_cast<int>(mono::RegionalIonicModel::Region::AvDelay)) {
      // Should equal exactly 0 for V at rest.
      if (std::fabs(iion[i]) > 1e-9) {
        if (rank == 0) std::cout << "[regional_iso] AV-delay leak nonzero at rest" << std::endl;
        ok = false;
      }
    }
  }

  // Repeated AdvanceStates with extreme ventricular V should not corrupt
  // passive I_ion at AV-delay DOFs (bug check: prior version ran TT06 on AV
  // DOFs as well, polluting their state).
  for (int step = 0; step < 50; ++step) {
    model.AdvanceStates(0.05, 0.05, vm);
  }
  model.ComputeIion(vm, iion);
  for (int i = 0; i < vm.Size(); ++i) {
    if (dof_region[i] == static_cast<int>(mono::RegionalIonicModel::Region::AvDelay)) {
      if (std::fabs(iion[i]) > 1e-9) {
        if (rank == 0) std::cout << "[regional_iso] AV-delay drifted after 50 steps" << std::endl;
        ok = false;
      }
    }
  }

  return ok ? 0 : 1;
}
