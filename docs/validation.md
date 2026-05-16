# Validation

## Executed checks

- `test_config_parse`: parser and defaults sanity.
- `test_tt06_single_cell`: finite `I_ion` over repeated updates.
- `test_fiber_gf_load`: build temporary `.mesh + f/s/n .gf`, load fibers, run bootstrap step.
- `test_restart_resume`: continuous vs checkpoint-restart regression on Niederer file mesh/fibers,
  `max |dVm| <= 1e-4`.
- `test_wholebody_smoke`: monodomain + recovery + torso pipeline finite-value smoke test.
- Full benchmark run (Hypre backend):
  - command: `mpirun -np 1 ./build/monodomain --config config/niederer_50ms.options`
  - end time reached: `t = 50 ms`
  - output path: `output/niederer_50ms/monodomain`
  - checkpoint path: `checkpoint/niederer_50ms`
- PETSc backend smoke run:
  - command: `mpirun -np 1 ./build/monodomain --config config/niederer_50ms.options --use-petsc 1 --t-end 0.1`
  - end time reached: `t = 0.1 ms`
- Whole-body mesh generation smoke:
  - command:
    `mpirun -np 1 ./build/generate_torso_box --heart-mesh benchmarks/niederer/niederer_benchmark.mesh --out-mesh benchmarks/torso_box/torso_box.mesh --padding-mm 8.0 --h-mm 1.0`
  - generated torso mesh:
    `benchmarks/torso_box/torso_box.mesh`
- Whole-body runtime smoke (`np=1`):
  - command:
    `mpirun -np 1 ./build/monodomain --config config/wholebody_default.options --t-end 0.2`
  - confirmed:
    - KSP2/KSP3 completed each step
    - constrained torso dofs: `562`
    - max mapping distance: `1 mm`
- Whole-body runtime smoke (`np=4`):
  - command:
    `mpirun -np 4 ./build/monodomain --config config/wholebody_default.options --t-end 0.05`
  - confirmed:
    - MPI run completed without errors
    - constrained torso dofs consistent: `562`
- Conforming whole-body mesh generation smoke:
  - command:
    `mpirun -np 1 ./build/generate_conforming_wholebody_case --heart-mesh benchmarks/niederer/niederer_benchmark.mesh --out-mesh benchmarks/wholebody/wholebody_conforming.mesh --padding-mm 8.0 --h-mm 1.0 --heart-attr 1 --torso-attr 2`
  - generated conforming mesh:
    `benchmarks/wholebody/wholebody_conforming.mesh`
- Conforming whole-body runtime smoke (`np=1`):
  - command:
    `mpirun -np 1 ./build/monodomain --config config/wholebody_conforming.options --t-end 0.2`
  - confirmed:
    - parent/heart/torso submesh path works
    - constrained torso dofs: `444`
    - max mapping distance: `0`
- Conforming whole-body runtime smoke (`np=4`):
  - command:
    `mpirun -np 4 ./build/monodomain --config config/wholebody_conforming.options --t-end 0.05`
  - confirmed:
    - MPI run completed without errors
    - constrained torso dofs: `444`
    - max mapping distance: `0`
- Niederer probe baseline (`np=1`, 50 ms):
  - command:
    `mpirun -np 1 ./build/monodomain --config config/niederer_50ms.options --benchmark-probes 1`
  - activation times:
    - `P1=2.56185 ms`
    - `P2=37.9052 ms`
    - `P3=NA`
    - `P4=18.8652 ms`
    - `P5=NA`
    - `P6=37.189 ms`
    - `P7=NA`
    - `P8=NA`
    - `P9=31.0404 ms`
- Niederer probe baseline (`np=1`, 80 ms, same config) for late-activation check:
  - command:
    `mpirun -np 1 ./build/monodomain --config config/niederer_50ms.options --benchmark-probes 1 --t-end 80`
  - activation times:
    - `P1=2.56185 ms`
    - `P2=37.9052 ms`
    - `P3=57.9972 ms`
    - `P4=18.8652 ms`
    - `P5=63.7508 ms`
    - `P6=37.189 ms`
    - `P7=56.7261 ms`
    - `P8=62.5836 ms`
    - `P9=31.0404 ms`
- Niederer literature-fit iteration (`np=1`, tuned conductivity):
  - command:
    `mpirun -np 1 ./build/monodomain --config config/niederer_70ms_litfit.options --benchmark-probes 1`
  - activation times:
    - `P1=2.58368 ms`
    - `P2=28.3215 ms`
    - `P3=36.1022 ms`
    - `P4=12.5861 ms`
    - `P5=41.4696 ms`
    - `P6=28.0168 ms`
    - `P7=35.3561 ms`
    - `P8=40.7543 ms`
    - `P9=21.3074 ms`
- MPI consistency (same Niederer case, Hypre backend):
  - commands: `mpirun -np 1 ...` vs `mpirun -np 4 ...` (isolated output dirs)
  - both runs reached `t = 50 ms`
  - final `Vm` stats matched to printed precision:
    `min=-89.9092, max=26.4197, mean=-29.9092, l2=2689.41`
- Restart consistency (`continuous` vs `checkpoint-restart`, `np=1`):
  - continuous: `0 -> 50 ms`
  - split run: `0 -> 25 ms`, then `--restart 1` to `50 ms`
  - restart load point: `step = 5000`, `t = 25 ms`
  - final `Vm` stats matched continuous run to printed precision:
    `min=-89.9092, max=26.4197, mean=-29.9092, l2=2689.41`
- Multi-rank checkpoint/restart smoke (`np=4`, Niederer case):
  - continuous: `0 -> 0.10 ms`
  - split run: `0 -> 0.05 ms`, then `--restart 1` to `0.10 ms`
  - restart load point: `step = 10`, `t = 0.05 ms`
  - final `Vm` stats matched continuous run to printed precision:
    `min=-85.2562, max=-83.6879, mean=-85.2141, l2=5591.12`
  - checkpoint layout verified in `checkpoint/validation_np4_ckpt_shared`:
    `latest.meta`, `vm_rank000000.gf..vm_rank000003.gf`,
    `tt06_rank000000.bin..tt06_rank000003.bin`

## Mechanics: LV passive inflation (Land 2015 RSPA Problem 3)

Configuration: `config/lv_passive_inflation.options`
  (50x24x4 LV ellipsoid, fibers ±60 deg helical, Holzapfel-Ogden constants
  from Land 2015, no EP, `land_Tref_kPa=0`, `mech_endo_pressure_pa=200`,
  `mech_endo_pressure_ramp_steps=10`).

Run:
```
mpirun -np 4 ./build/monodomain --config config/lv_passive_inflation.options
```

Result at p = 200 Pa (one mechanics solve, 10 ramped Newton sub-solves):

| Quantity                            | Value         | Pass criterion                 |
|-------------------------------------|---------------|--------------------------------|
| `|u|max`                            | 0.91 mm       | finite, < 5% of mesh extent    |
| Mean endo radial displacement       | +0.31 mm (out)| positive = outward inflation   |
| `J` range                           | (0.972, 1.028)| ~ 3% drift, OK for kappa=1000  |
| Apex z-displacement                 | -0.07 mm down | matches Land 2015 elongation   |
| Newton failures                     | 0 / 10        | every ramp step converged      |

Linear extrapolation to Land 2015's reference of 10 kPa gives mean endo
radial displacement ~ 15 mm vs Land 2015 / lifex Africa 2023 reference
of 7-10 mm; the ~2x softness is consistent with using a kappa=1000 kPa
volumetric penalty instead of the incompressible Lagrange multiplier
formulation in Land 2015. Reaching higher pressures (1-10 kPa) currently
needs additional ramping density or a line search in the Newton path.

## Pending checks

- LV passive inflation at full 10 kPa (Land 2015 reference); blocked on
  Newton line search / trust region in the mechanics path.
- LV active contraction quantitative check (Land 2015 twist + ejection
  fraction); blocked on the same Newton robustness fix.

## Notes

- Checkpoint is rank-sharded (`vm_rank%06d.gf`, `tt06_rank%06d.bin`) and works for `np>=1`.
- `--restart` must use the same MPI size as the saved checkpoint.
- Legacy single-rank checkpoint (`vm.gf`, `tt06.bin`) remains loadable with `np=1`.
