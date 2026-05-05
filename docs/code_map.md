# Code Map

Single-page index for fast navigation. Every class lists its file pair and
the test that exercises it. Keep this file in lockstep with the code: any
new public class or test must also be appended here in the same commit
(enforced by `.githooks/commit-msg`).

If a row points at a missing file or test, that's a documentation bug and
should be filed immediately.

---

## Module dependency (top-down)

```
                       main.cpp  (orchestrator)
                          │
       ┌──────────────────┼─────────────────────┐
       │                  │                     │
  config/SimulationConfig │                     │
       │                  ▼                     ▼
       │            space/Assembler        io/CheckpointIO
       │                  │                io/OutputManager
       │                  │
       │           ┌──────┴──────┐
       │           ▼             ▼
       │  ode/IIonicModel   solver/MonodomainStepper
       │   ├─ TT06Model         │
       │   ├─ Stewart…          │
       │   ├─ Grandi2011        ├─→ solver/LinearSolverFactory
       │   ├─ Passive           │
       │   ├─ Regional…(uses 4 above)
       │                        │
       │                        ├─→ solver/PvjCoupler
       │                        │     │
       │                        │     ▼
       │                        │   solver/PurkinjeCableSolver
       │                        │     (uses one IIonicModel)
       │                        │
       │                        ├─→ solver/ExtracellularRecoverySolver
       │                        └─→ solver/TorsoPotentialSolver
       │                              │
       │                              ▼
       │                        coupling/InterfaceMapper
       └─ space/FiberTensorCoefficient  (used by Assembler)
```

Rules of thumb:
- `ode/` knows nothing about meshes; talks only through `IIonicModel`.
- `space/` knows nothing about cell models; produces $M$, $K$, $A$, $B$.
- `solver/` orchestrates time stepping and depends on both, but each solver
  pulls dependencies through narrow interfaces (no `solver/X.cpp` should
  `#include "ode/TT06Model.hpp"` directly — go through `IIonicModel`).

---

## Class index

### `src/config/`

| Class / Free fn | File | Role | Test |
|---|---|---|---|
| `SimulationConfig` (struct) | `SimulationConfig.{hpp,cpp}` | Run-parameter schema (~80 keys) | `test_config_parse` |
| `LoadConfigFile` | same | Strict key validation parser | `test_config_parse` |
| `OverrideFromArgs` | same | CLI flag overrides | — (covered indirectly) |

### `src/space/`

| Class | File | Public methods (key) | Test |
|---|---|---|---|
| `Assembler` | `Assembler.{hpp,cpp}` | `BuildSystemMatrices(dt)`, `M()/K()/A()/B()`, `PFES()`, `Vm()`, `TrueVSize()`, `Fiber{F,S,N}()` | `test_fiber_gf_load`, `test_wholebody_smoke` |
| `FiberTensorCoefficient` | `FiberTensorCoefficient.{hpp,cpp}` | `Eval(K, T, ip)` builds $\sigma_f f\!\otimes\!f + \cdots$ | `test_fiber_gf_load` |

### `src/ode/`

| Class | File | Methods (besides `IIonicModel` overrides) | Test |
|---|---|---|---|
| `IIonicModel` (abstract) | `IIonicModel.hpp` | `Compute/AdvanceStates`, `Save/LoadState`, `ModelId()`, `NumNodes()` | — (interface) |
| `TT06Model` | `TT06Model.{hpp,cpp}` + `tt06_generated.{h,c}` | CellML codegen path | `test_tt06_single_cell`, `test_restart_resume` |
| `StewartPurkinjeModel` | `StewartPurkinjeModel.{hpp,cpp}` | Hand-port (KNOWN BROKEN, see `purkinje_numerics.md` §5.4) | `test_stewart_single_cell` (WILL_FAIL) |
| `Grandi2011Model` | `Grandi2011Model.{hpp,cpp}` | Compact 15-state atrial port | — (TODO) |
| `PassiveModel` | `PassiveModel.{hpp,cpp}` | Leak-only: `SetLeakConductance`, `LeakConductance`, `RestPotential` | `test_pvj_active_sign`, `test_regional_isolation` |
| `RegionalIonicModel` | `RegionalIonicModel.{hpp,cpp}` | `LocalDofCount(Region)`, `DofRegions()`; gather/scatter on V_m | `test_regional_isolation` |

### `src/solver/`

| Class | File | Public methods (key) | Test |
|---|---|---|---|
| `LinearSystemSolver` | `LinearSolverFactory.{hpp,cpp}` | `SetOperator`, `Solve`, `LastNumIterations` | — (used by every stepper) |
| `MonodomainStepper` | `MonodomainStepper.{hpp,cpp}` | `Bootstrap`, `StepNoCorrection`, `SetPvjCoupler`, `VmTrue`, `LastTiming` | `test_restart_resume`, `test_wholebody_smoke` |
| `PurkinjeCableSolver` | `PurkinjeCableSolver.{hpp,cpp}` | `LoadNetwork`, `Initialize`, `Advance`, `TerminalVoltage`, `GraphNodeVoltage`, `Num*` | `test_purkinje_cable_smoke`, `test_purkinje_cable_cv` (WILL_FAIL) |
| `PvjCoupler` | `PvjCoupler.{hpp,cpp}` | `BuildHeartCouplingCurrent`, `AdvancePurkinje`, `Cable`, `GlobalNumMappedPvj`, `GlobalMaxMappedDistMm` | `test_pvj_active_sign` |
| `ExtracellularRecoverySolver` | `ExtracellularRecoverySolver.{hpp,cpp}` | `Solve(vm_true)`, `UeTrue`, `Ue`, `LastNumIterations` | `test_wholebody_smoke` |
| `TorsoPotentialSolver` | `TorsoPotentialSolver.{hpp,cpp}` | `SetConstrainedDofs`, `Solve(bc_true)`, `NumConstrainedDofs` | `test_wholebody_smoke` |

### `src/coupling/`

| Class | File | Methods | Test |
|---|---|---|---|
| `InterfaceMapper` | `InterfaceMapper.{hpp,cpp}` | `MapHeartToTorso`, `TorsoConstrainedDofs`, `GlobalNumConstrainedDofs`, `GlobalMaxConstrainedDistMm` | `test_wholebody_smoke` |

### `src/io/`

| Class | File | Methods | Test |
|---|---|---|---|
| `CheckpointIO` | `CheckpointIO.{hpp,cpp}` | `SaveLatest(step,t,vm,ionic)`, `LoadLatest` (V3+model_id) | `test_checkpoint_model_id`, `test_restart_resume` |
| `OutputManager` | `OutputManager.{hpp,cpp}` | `Save(step,t,…)` ParaView writer | — (visual) |

### `src/main.cpp`

Orchestrator. No test directly; covered end-to-end by running `./monodomain
--config <…>.options`. Layout:

1. MPI / Hypre / PETSc init
2. Load config; build (optional wholebody-conforming split)
3. Construct `Assembler`
4. Construct ionic model (`RegionalIonicModel` if enabled, else `TT06Model`)
5. Construct `LinearSystemSolver` + `MonodomainStepper`
6. Optional: load `.network` file → build `PurkinjeCableSolver` + `PvjCoupler`; attach via `SetPvjCoupler`
7. Optional: build wholebody `ExtracellularRecoverySolver` + `TorsoPotentialSolver` + `InterfaceMapper`
8. Bootstrap; restart if requested
9. Time loop: `Bootstrap` once, then `StepNoCorrection` until `t_end_ms`
   - Periodic output via `OutputManager`
   - Periodic checkpoint via `CheckpointIO`
10. Cleanup

---

## Tools

| Binary | File | Role |
|---|---|---|
| `generate_niederer_case` | `tools/generate_niederer_case.cpp` | Builds Niederer benchmark mesh + fibers (BROKEN on MFEM 4.5: uses protected `GetElementJacobian`) |
| `generate_torso_box` | `tools/generate_torso_box.cpp` | Auto-generated torso bounding box |
| `generate_conforming_wholebody_case` | `tools/generate_conforming_wholebody_case.cpp` | Wholebody mesh w/ heart + torso attributes |
| `generate_mesh_attr_gf` | `tools/generate_mesh_attr_gf.cpp` | Per-element attribute → grid-function field |
| `generate_purkinje_tree.py` | `tools/generate_purkinje_tree.py` | Costabal-style fractal Purkinje tree generator |
| `generate_conforming_wholebody_gmsh.py` | same `.py` | Optional Gmsh-based wholebody mesh |
| `run_niederer_all_scales.sh` | same `.sh` | Niederer benchmark sweep across mesh resolutions |
| `run_wholebody_gmsh_matrix.sh` | same `.sh` | Wholebody scenario matrix |

---

## Tests (`tests/`, registered in `CMakeLists.txt`)

| Test name | File | What it asserts | Status |
|---|---|---|---|
| `tt06_single_cell` | `test_tt06_single_cell.cpp` | TT06 AP shape | pass |
| `config_parse` | `test_config_parse.cpp` | Strict config key parser | pass |
| `fiber_gf_load` | `test_fiber_gf_load.cpp` | Fiber field load + orthogonality | pass |
| `restart_resume` | `test_restart_resume.cpp` | Checkpoint round-trip identity | fail (mesh missing in repo) |
| `wholebody_smoke` | `test_wholebody_smoke.cpp` | One step of wholebody coupled solve | fail (mesh missing) |
| `stewart_single_cell` | `test_stewart_single_cell.cpp` | Stewart AP fits Stewart 2009 Fig. 2 | WILL_FAIL until CellML port lands |
| `purkinje_cable_smoke` | `test_purkinje_cable_smoke.cpp` | Diffusion through 30-node passive cable | pass |
| `purkinje_cable_cv` | `test_purkinje_cable_cv.cpp` | Stewart cable CV in [1.5, 4.0] m/s | WILL_FAIL (depends on Stewart) |
| `pvj_active_sign` | `test_pvj_active_sign.cpp` | $I_{pvj}=g(V_m-V_p)$ sign + magnitude | pass |
| `regional_isolation` | `test_regional_isolation.cpp` | Children only run on their region's DOFs | pass |
| `checkpoint_model_id` | `test_checkpoint_model_id.cpp` | Loading TT06 ckpt into Passive refused | pass |

---

## Where to look first when …

| Problem | First places to read |
|---|---|
| "Conduction is too slow" | `Assembler::InitializeFiberCoefficients`, `SimulationConfig.sigma_*`, `region_sigma_scale_` (AV-delay/fibrosis multiplier) |
| "Cell model doesn't fire" | `<Model>::ComputeIion` and `<Model>::AdvanceStates`, then the corresponding `tests/test_*_single_cell.cpp` |
| "Purkinje signal doesn't reach myocardium" | `PvjCoupler::BuildMapping` (mapping count log), `pvj_max_dist_mm`, `purkinje_numerics.md` §8 |
| "Restart fails / missing checkpoint" | `CheckpointIO::LoadLatest`, `test_checkpoint_model_id.cpp`, `purkinje_numerics.md` §11 |
| "ParaView output looks wrong" | `OutputManager::Save`, `OutputManager.cpp` field projection block |
| "MPI rank N writes nothing" | `local_links_.empty()` in `PvjCoupler`, `LocalDofCount(region)` in `RegionalIonicModel` |
| "Add a new ionic model" | (1) New `src/ode/<Name>Model.{hpp,cpp}` implementing `IIonicModel`; (2) Register in `CMakeLists.txt`; (3) Optional: extend `RegionalIonicModel` if it should be region-routed; (4) Add `tests/test_<name>_single_cell.cpp`; (5) Update this `code_map.md` and `purkinje_numerics.md` §5/6/7 |

---

## Documents

| File | Topic |
|---|---|
| `docs/architecture_full.md` | Top-level project layout (Chinese, pre-Purkinje) |
| `docs/numerics.md` | Monodomain discretization |
| `docs/purkinje_numerics.md` | Purkinje + PVJ + Regional + Checkpoint math source-of-truth |
| `docs/purkinje_numerics.tex` / `.pdf` | Typeset version of `purkinje_numerics.md` (pandoc + xelatex) |
| `docs/validation.md` | Niederer + benchmark validation procedures |
| `docs/wholebody_workflow.md` | Heart-torso run instructions |
| `docs/literature_comparison.md` | Comparison to other monodomain solvers |
| `docs/petsc_asm_geometric_report.md` | PETSc preconditioner study |
| `docs/code_map.md` | **This file** — navigation index |
