// Verify CheckpointIO refuses to load when the runtime ionic model differs
// from the model that wrote the checkpoint.
//
// Procedure:
//   1. Build a tiny MFEM ParMesh + P1 ParFESpace.
//   2. Construct a TT06Model, save a checkpoint via CheckpointIO::SaveLatest.
//   3. Try to LoadLatest into a PassiveModel; expect false (refused).
//   4. Try to LoadLatest into a fresh TT06Model; expect true (matching ID).

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

#include "mfem.hpp"

#include "config/SimulationConfig.hpp"
#include "io/CheckpointIO.hpp"
#include "ode/PassiveModel.hpp"
#include "ode/TT06Model.hpp"

int main(int argc, char* argv[]) {
  mfem::Mpi::Init(argc, argv);
  const int rank = mfem::Mpi::WorldRank();

  // Tiny mesh: 1-D 4 elements -> 5 nodes, P1 -> 5 true DOFs in serial.
  mfem::Mesh serial_mesh = mfem::Mesh::MakeCartesian1D(4, 4.0);
  mfem::ParMesh pmesh(MPI_COMM_WORLD, serial_mesh);
  mfem::H1_FECollection fec(1, 1);
  mfem::ParFiniteElementSpace pfes(&pmesh, &fec);
  mfem::ParGridFunction vm_gf(&pfes);
  vm_gf = -85.23;

  mono::SimulationConfig cfg;
  cfg.checkpoint_dir = "/tmp/test_checkpoint_model_id";
  std::filesystem::remove_all(cfg.checkpoint_dir);

  mono::CheckpointIO ckpt(cfg, MPI_COMM_WORLD);

  const int n_local = pfes.GetTrueVSize();
  if (n_local <= 0) {
    if (rank == 0) std::cout << "[chkpt_id] empty rank, skipping" << std::endl;
    return 0;
  }

  // Save with TT06.
  {
    mono::TT06Model tt06(n_local);
    tt06.InitializeRestState(-85.23);
    ckpt.SaveLatest(/*step*/ 1, /*t_ms*/ 0.5, vm_gf, tt06);
  }

  bool ok = true;

  // Attempt load with Passive: must refuse.
  {
    mono::PassiveModel passive(n_local, 0.05, -85.0);
    mfem::ParGridFunction vm_dst(&pfes);
    int step = 0;
    double t_ms = 0.0;
    const bool loaded = ckpt.LoadLatest(step, t_ms, vm_dst, passive);
    if (loaded) {
      if (rank == 0) std::cout << "[chkpt_id] FAIL: Passive accepted TT06 checkpoint" << std::endl;
      ok = false;
    } else if (rank == 0) {
      std::cout << "[chkpt_id] OK: Passive refused TT06 checkpoint" << std::endl;
    }
  }

  // Attempt load with TT06 again: must succeed.
  {
    mono::TT06Model tt06b(n_local);
    tt06b.InitializeRestState(-85.23);
    mfem::ParGridFunction vm_dst(&pfes);
    int step = 0;
    double t_ms = 0.0;
    const bool loaded = ckpt.LoadLatest(step, t_ms, vm_dst, tt06b);
    if (!loaded) {
      if (rank == 0) std::cout << "[chkpt_id] FAIL: TT06 failed to load TT06 checkpoint" << std::endl;
      ok = false;
    } else if (step == 1 && std::abs(t_ms - 0.5) < 1e-9) {
      if (rank == 0) std::cout << "[chkpt_id] OK: TT06 reload step=" << step
                                << " t_ms=" << t_ms << std::endl;
    } else {
      if (rank == 0) std::cout << "[chkpt_id] FAIL: metadata mismatch step=" << step
                                << " t_ms=" << t_ms << std::endl;
      ok = false;
    }
  }

  return ok ? 0 : 1;
}
