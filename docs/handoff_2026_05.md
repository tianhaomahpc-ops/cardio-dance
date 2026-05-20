# Cross-session handoff (2026-05)

This file fixes the project state at HEAD `1f1b38d` so any future session
(human or AI) can resume cleanly without scrolling chat history.

---

## TL;DR

Repository: `tianhaomahpc-ops/cardio-dance` on branch
`claude/review-branch-history-9W03K`. Latest commit `1f1b38d` (doc sweep).

Working pipeline: half-ellipsoid LV (gmsh tet mesh, h=1 mm, 124k tets) +
91-terminal Purkinje + smeared PVJ + 1-lead pseudo-ECG. Solver: PETSc
CG + ASM(0)/ICC(0). Output movies, snapshots and ECG plots committed
under `output/half_ellipsoid_purkinje/`.

---

## Current state of the major modules

| Module | Status | File |
|---|---|---|
| TT06 ventricular cell | CellML codegen, verified | `src/ode/TT06Model.{hpp,cpp}` + `tt06_generated.{h,c}` |
| Stewart 2009 Purkinje cell | CellML codegen (from models.cellml.org), verified to fit Fig. 2 | `src/ode/StewartPurkinjeModel.{hpp,cpp}` + `stewart_generated.{h,c}` |
| Grandi 2011 atrial cell | Compact 15-state hand port, **untested** | `src/ode/Grandi2011Model.{hpp,cpp}` |
| Passive (leak) cell | Trivially correct | `src/ode/PassiveModel.{hpp,cpp}` |
| Regional ionic dispatcher | Per-region state isolation works (test passes) | `src/ode/RegionalIonicModel.{hpp,cpp}` |
| Purkinje 1D cable solver | Crank-Nicolson + Rush-Larsen, ~5 m/s CV verified | `src/solver/PurkinjeCableSolver.{hpp,cpp}` |
| PVJ bidirectional coupling | Sign + magnitude verified algebraically | `src/solver/PvjCoupler.{hpp,cpp}` |
| PVJ smear (3D source-sink fix) | Working | same |
| AV-delay/fibrosis conductivity scaling | Wired in Assembler, **end-to-end untested** | `src/space/Assembler.cpp` |
| Pseudo-ECG far-field probe | Working, units arbitrary | `src/io/PseudoEcg.{hpp,cpp}` |
| Checkpoint with model_id | CHKPT_V3, refuses cross-model loads | `src/io/CheckpointIO.{hpp,cpp}` |
| Half-ellipsoid mesh (gmsh) | Working, smooth surfaces | `tools/half_ellipsoid.geo` + `tools/finalize_half_ellipsoid_mesh.cpp` |
| Fractal Purkinje generator | Working | `tools/generate_purkinje_tree.py` |
| PETSc backend (ASM+ICC) | Working (project default) | `src/solver/LinearSolverFactory.{hpp,cpp}` |

---

## What works end-to-end

```bash
# Activate PETSc options
export PETSC_OPTIONS="$(grep -v '^##\|^$' config/petsc_asm.opts | tr '\n' ' ')"

# Run 100 ms simulation on full-scale half-ellipsoid
./build/monodomain --config config/half_ellipsoid_purkinje.options
# → output/half_ellipsoid_purkinje/{heart, pseudo_ecg.csv, ksp_history.csv}

# Render outputs
xvfb-run -s "-screen 0 1024x768x24" /tmp/venv/bin/python3 \
  tools/plot_vm_movie.py --run output/half_ellipsoid_purkinje \
  --out output/half_ellipsoid_purkinje/vm_movie_clipped.mp4 \
  --purkinje-network config/half_ellipsoid_purkinje.network --clip-y

xvfb-run -s "-screen 0 1024x768x24" /tmp/venv/bin/python3 \
  tools/plot_pseudo_ecg.py \
  --csv output/half_ellipsoid_purkinje/pseudo_ecg.csv \
  --out output/half_ellipsoid_purkinje/pseudo_ecg.png
```

Wall-time on this 4 vCPU sandbox: ~5m52s for 100 ms simulated.

---

## Sandbox-ephemeral state (NOT in git, must rebuild every session)

| Item | Recovery command |
|---|---|
| apt packages (mpi, gmsh, ffmpeg, …) | `apt-get install -y libopenmpi-dev libhypre-dev libmetis-dev liblapack-dev libpetsc-real-dev gmsh ffmpeg xvfb libosmesa6 libegl1` |
| MFEM 4.7 with PETSc | See README.md Path B; ~5 minutes to compile |
| Python visualisation venv | `python3 -m venv /tmp/venv && /tmp/venv/bin/pip install pyvista matplotlib numpy` |
| `build/` directory | `MFEM_DIR=/opt/mfem-petsc-install cmake -B build -S . && cmake --build build -j` |
| `benchmarks/half_ellipsoid/heart.{msh,mesh}` + `fiber_*.gf` | `gmsh -3 tools/half_ellipsoid.geo -o benchmarks/half_ellipsoid/heart.msh -setnumber h_mm 1.0 && ./build/finalize_half_ellipsoid_mesh --emit-fibers` |
| Demo simulation outputs | `./build/monodomain --config config/half_ellipsoid_purkinje.options` |

---

## Known issues / non-trivial caveats

1. **Fibers are constant vectors**, not Streeter helix. CV anisotropy
   ratio is correct (`√(σ_f/σ_t) ≈ 2.75`) but direction is fixed +x
   everywhere. To add real fibers: implement Bayer-Trayanova LDRBM
   (4 Laplace solves + helix rotation, ~200 lines C++). See
   `purkinje_numerics.md` §13.

2. **`mpirun -n > 1` hangs on small mesh + Hypre/PETSc preconditioners**.
   With ~5k DOFs/rank the PC setup costs dominate. Default workflow
   uses single rank. Real benefit of MPI requires ≥ 50k DOFs/rank.

3. **Half-scale geometry not stable across configs**. There were earlier
   commits using half-scale outer 17.5×11×11 — current
   `config/half_ellipsoid_purkinje.options` is set for **full scale**
   35×22×22 (matching the latest mesh / Purkinje network on disk).

4. **`pvj_max_dist_mm = 5` rejects ~50% of Purkinje terminals** in the
   demo (44/91 mapped). Increasing the cutoff or generating denser
   Purkinje + better-fit-to-endocardium tree would help, but with
   current 91-terminal tree at full scale the activation is sufficient
   for a recognisable wave.

5. **Pseudo-ECG has arbitrary units**. The `σ_i / σ_b` ratio is set from
   typical literature values but absolute calibration would need a
   torso conductivity model. Multi-lead ECG morphology is also limited
   by single-LV-only geometry (no atria, no torso anisotropy).

6. **Atria (Grandi 2011) and AV-delay regions untested at the
   integration level**. The dispatcher routes correctly per
   `test_regional_isolation`, but no end-to-end run uses these regions
   with a multi-attribute mesh.

7. **`-DMFEM_DIR=/opt/mfem-petsc-install` must be set** when configuring
   cmake; the apt-installed MFEM lacks PETSc support so it's not the
   right one.

---

## Discipline rules (from earlier review)

- Source-of-truth for math: `docs/purkinje_numerics.md` (and its PDF).
- Source-of-truth for code layout: `docs/code_map.md`.
- Top-level architecture (Chinese): `docs/architecture_full.md`.
- Numerics summary: `docs/numerics.md` (now covers KSP1-4 + pseudo-ECG).
- Per-commit Math/Code/Mapping discipline: enforced by
  `.githooks/commit-msg` for commits touching `src/{ode,solver,space,coupling}/`.
  Activate locally via `git config core.hooksPath .githooks`.

---

## Suggested next steps (any future session can pick up)

| Item | Effort | Why |
|---|---|---|
| Add Bayer-Trayanova LDRBM fiber generator | medium (~200 lines C++) | Realistic T-wave morphology, multi-lead ECG differences |
| Verify Grandi 2011 atrial cell | small (~50 lines test) | Currently untested; needed before atrial demos |
| Multi-rank break-even study (MFEM partition + run on 16+ cores) | small | Quantify when ASM beats single-rank wall-clock |
| Replace pseudo-ECG with full bidomain + torso (Niederer-style) | large | Absolute mV calibration, multi-lead ECG |
| Add atria + AV junction to half-ellipsoid demo | medium | Captures full P-QRS-T morphology |

---

## Appendix: files committed under `output/half_ellipsoid_purkinje/`

These are demo artifacts — kept in git so they show up on GitHub for
review without rerunning the simulation. They are still regenerable.

| File | Size | What |
|---|---|---|
| `vm_movie.mp4` | ~125 KB | External LV view, V_m animation, Purkinje overlay |
| `vm_movie_clipped.mp4` | ~150 KB | y≥0 clipped view exposing endocardial activation |
| `pseudo_ecg.png` | ~45 KB | 1-lead pseudo-ECG plot |
| `snap_t20_fullscale_h1.png` | ~185 KB | Static screenshot at peak activation |
| `gallery/mesh_wireframe_t0.png` | ~390 KB | All 124k tet edges visible at rest |
| `gallery/mesh_apex_closeup.png` | ~245 KB | Apex zoom, single tets countable |
| `gallery/cross_y0_t{5,15,25,40,80}.png` | each ~100 KB | Sagittal-like clip series |
| `gallery/cross_x_mid_t20.png` | ~120 KB | Transverse wall ring |
| `gallery/cross_z0_t20.png` | ~62 KB | Horizontal cut at peak |
| `gallery/cross_y0_z0_t20.png` | ~63 KB | Octant view |
