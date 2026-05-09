# Monodomain TT06 (MFEM + PETSc/Hypre)

## Build

### Path A — spack-installed dependencies (production)

```bash
source ~/spack/share/spack/setup-env.sh
MFEM_PREFIX=$(spack -e cardiac-sim location -i mfem)
PETSC_PREFIX=$(spack -e cardiac-sim location -i petsc)
MPICXX=$(awk -F= '/^MFEM_CXX[ \t]*=/{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2}' "$MFEM_PREFIX/share/mfem/config.mk")
MPICC=$(dirname "$MPICXX")/mpicc

cmake -S . -B build \
  -DCMAKE_C_COMPILER="$MPICC" \
  -DCMAKE_CXX_COMPILER="$MPICXX" \
  -DMFEM_DIR="$MFEM_PREFIX" \
  -DPETSC_DIR="$PETSC_PREFIX"
cmake --build build -j
```

### Path B — apt + source-build MFEM (sandbox / cloud VM)

```bash
# 1. System dependencies
apt-get install -y libopenmpi-dev libhypre-dev libmetis-dev liblapack-dev \
                   libpetsc-real-dev gmsh ffmpeg xvfb libosmesa6 libegl1

# 2. Build MFEM 4.7 from source with PETSc support
curl -sSL -o /tmp/mfem.tar.gz https://github.com/mfem/mfem/archive/v4.7.tar.gz
tar xzf /tmp/mfem.tar.gz -C /opt
cd /opt/mfem-4.7
make config MFEM_USE_MPI=YES MFEM_USE_PETSC=YES MFEM_USE_LAPACK=YES \
            MFEM_SHARED=YES MFEM_INSTALL_DIR=/opt/mfem-petsc-install \
            PETSC_DIR=/usr/lib/petscdir/petsc3.19/x86_64-linux-gnu-real PETSC_ARCH= \
            HYPRE_DIR=/usr HYPRE_LIB="-lHYPRE" HYPRE_OPT="-I/usr/include/hypre" \
            METIS_DIR=/usr METIS_LIB="-lmetis"
make -j install

# 3. Build cardio-dance against it
MFEM_DIR=/opt/mfem-petsc-install cmake -S /path/to/cardio-dance -B build
cmake --build build -j
```

### Path C — Python venv for visualisation

```bash
python3 -m venv /tmp/venv
/tmp/venv/bin/pip install pyvista matplotlib numpy
# tools/plot_vm_movie.py and friends expect /tmp/venv/bin/python3
```

## Run

```bash
mpirun -np 4 ./build/monodomain --config config/default.options
```

## Whole-body (heart + torso)

Generate a torso box mesh that wraps the Niederer heart mesh:

```bash
mpirun -np 1 ./build/generate_torso_box \
  --heart-mesh benchmarks/niederer/niederer_benchmark.mesh \
  --out-mesh benchmarks/torso_box/torso_box.mesh \
  --padding-mm 8.0 \
  --h-mm 1.0
```

Run whole-body coupled workflow (KSP1/KSP2/KSP3):

```bash
mpirun -np 1 ./build/monodomain --config config/wholebody_default.options
```

Short MPI smoke run:

```bash
mpirun -np 4 ./build/monodomain --config config/wholebody_default.options --t-end 0.05
```

Whole-body outputs:

- Heart fields: `output/wholebody_default/heart/monodomain/*` (`Vm`, `Iion`, `ue`)
- Torso fields: `output/wholebody_default/torso/torso_potential/*` (`uT`)
- Timing CSV: `output/wholebody_default/timing_breakdown.csv`

### Conforming whole-body route (single mesh, attribute split)

Generate one conforming whole-body mesh (heart attr + torso attr) from the
Niederer bounding box:

```bash
mpirun -np 1 ./build/generate_conforming_wholebody_case \
  --heart-mesh benchmarks/niederer/niederer_benchmark.mesh \
  --out-mesh benchmarks/wholebody/wholebody_conforming.mesh \
  --padding-mm 8.0 \
  --h-mm 1.0 \
  --heart-attr 1 \
  --torso-attr 2
```

Run conforming whole-body simulation:

```bash
mpirun -np 1 ./build/monodomain --config config/wholebody_conforming.options
```

MPI smoke:

```bash
mpirun -np 4 ./build/monodomain --config config/wholebody_conforming.options --t-end 0.05
```

## Niederer benchmark (50ms)

Generate one MFEM non-structured tetra mesh + three fiber `.gf` files:

```bash
mpirun -np 1 ./build/generate_niederer_case benchmarks/niederer
```

Generated fibers are constant orthonormal axes:

- `fiber_f = (1, 0, 0)`
- `fiber_s = (0, 1, 0)`
- `fiber_n = (0, 0, 1)`

Run 50ms simulation from file-based mesh/fibers:

```bash
mpirun -np 1 ./build/monodomain --config config/niederer_50ms.options
```

Run all three benchmark spatial scales (0.5 / 0.2 / 0.1 mm) in sequence:

```bash
NP=1 BENCHMARK_PROBES=1 ./tools/run_niederer_all_scales.sh
```

This script will:

- generate meshes/fibers into `benchmarks/niederer_h050`, `benchmarks/niederer_h020`, `benchmarks/niederer_h010`
- run with configs:
  - `config/niederer_50ms_h050.options`
  - `config/niederer_50ms_h020.options`
  - `config/niederer_50ms_h010.options`
- write outputs into:
  - `output/niederer_50ms_h050`
  - `output/niederer_50ms_h020`
  - `output/niederer_50ms_h010`

Outputs:

- `output/niederer_50ms/monodomain/*.pvd/.pvtu/.vtu`
- `checkpoint/niederer_50ms/latest.meta`
- `checkpoint/niederer_50ms/vm_rank%06d.gf` (one shard per MPI rank)
- `checkpoint/niederer_50ms/tt06_rank%06d.bin` (one shard per MPI rank)

Optional literature-fit comparison run (conductivity scaled, same anisotropy):

```bash
mpirun -np 1 ./build/monodomain --config config/niederer_70ms_litfit.options --benchmark-probes 1
```

Restart from latest checkpoint:

```bash
mpirun -np 4 ./build/monodomain --config config/niederer_50ms.options --restart 1
```

`--restart` must use the same MPI size as the saved checkpoint.

For fiber fields from `.gf`, set:

- `use_fiber_gf=1`
- `fiber_f_path=/path/to/fiber_f.gf`
- `fiber_s_path=/path/to/fiber_s.gf`
- `fiber_n_path=/path/to/fiber_n.gf`

The three `.gf` files must match the simulation mesh and be vector grid functions with
`vdim=mesh_dim` and `Ordering=byVDIM`.

## Multi stimulus regions

You can define multiple stimulus regions by repeating `stim_region=` in the options file.
All listed regions are unioned into one spatial mask.

- Ball: `stim_region=ball,cx,cy,cz,r`
- Box: `stim_region=box,x1,y1,z1,x2,y2,z2` (two diagonal corners)

Example:

```ini
stim_start_ms=0.0
stim_end_ms=2.0
stim_amp=-15.0
stim_region=box,0.0,0.0,0.0,1.5,1.5,1.5
stim_region=ball,10.0,3.5,1.5,0.8
```

Legacy single-box keys (`stim_xmin_mm`...`stim_zmax_mm`) are still supported.

## Current coupling scheme

This implementation uses decoupled no-correction steps:

- Bootstrap: `I_ion^0 -> w^1 -> V^1`
- Main loop: `I_ion^n -> w^{n+1} -> V^{n+1}`

No in-step correction is applied.

When `enable_wholebody=1`, each PDE step additionally performs:

1. **Recovery (KSP2)**: solve `K1 ue = -K2 Vm` on heart mesh.
2. **Heart-to-torso map**: map `ue` samples to constrained torso true dofs.
3. **Torso solve (KSP3)**: solve `-div(sigma_torso grad(uT)) = 0` with penalty-Dirichlet constraints.

## Solver backend

The Crank-Nicolson diffusion system is symmetric positive-definite. Available
backends, picked by `SimulationConfig`:

| Config flag | Backend |
|---|---|
| `use_petsc=1` (project default for half-ellipsoid demo) | PETSc KSP + ASM(0)/ICC(0) |
| `use_hypre_boomeramg=1` | MFEM CG + HypreBoomerAMG |
| `use_hypre_block_jacobi=1` | MFEM CG + HypreSmoother (l1-Jacobi) |
| all of the above 0 | plain MFEM CG (no preconditioner) |

Default PETSc options live in `config/petsc_asm.opts` (CG + ASM with overlap=0
and ICC(0) sub-PC). Activate them via:

```bash
export PETSC_OPTIONS="$(grep -v '^##\|^$' config/petsc_asm.opts | tr '\n' ' ')"
./build/monodomain --config config/half_ellipsoid_purkinje.options
```

To override at the command line (e.g. switch sub-PC):

```bash
PETSC_OPTIONS="$(cat config/petsc_asm.opts) -mono_sub_pc_type ilu" ./build/monodomain ...
```

`docs/petsc_asm_vs_cg_iter_bench.md` shows the ASM vs plain CG iteration
count comparison: 5 vs 50 iter/step on the 124k-tet half-ellipsoid mesh
(10× reduction).

ASM/GASM subdomain policy (legacy code path):
- `petsc_use_geometric_asm=1` enables explicit subdomain setup from MFEM
  partition ownership (one local subdomain per MPI rank; aligned with MFEM
  / ParMETIS decomposition)

## Whole-body config keys

Set `enable_wholebody=1` and provide:

- `torso_mesh_path`
- `sigma_i_f_mS_per_mm`, `sigma_i_s_mS_per_mm`, `sigma_i_n_mS_per_mm`
- `sigma_e_f_mS_per_mm`, `sigma_e_s_mS_per_mm`, `sigma_e_n_mS_per_mm`
- `sigma_torso_mS_per_mm`
- `interface_map_max_dist_mm`
- `torso_dirichlet_penalty`

Optional:

- `heart_interface_bdr_attrs=...` to select heart boundary attributes used for mapping
- `torso_interface_bdr_attrs=...` reserved for future boundary-attribute coupling path

Reference config: `config/wholebody_default.options`
Conforming reference config: `config/wholebody_conforming.options`

Detailed workflow and literature-comparison checklist:

- `docs/wholebody_workflow.md`
- `docs/literature_comparison.md`
- `docs/petsc_asm_geometric_report.md`

## Navigation

For fast lookup of class → file → test, see `docs/code_map.md`. The
mathematical source-of-truth for the Purkinje + PVJ + Regional stack lives
in `docs/purkinje_numerics.md` (typeset PDF: `docs/purkinje_numerics.pdf`,
rebuilt via `make -C docs pdf`).

## Developer setup: enable repo-tracked git hooks

The repository ships a `commit-msg` hook in `.githooks/` that requires
commits touching `src/{ode,solver,space,coupling}/` to include `Math:` and
`Mapping:` tags in the commit body. This enforces the change-log
discipline described in `docs/purkinje_numerics.md` §12.

Activate the hooks once per clone:

```bash
git config core.hooksPath .githooks
```

To verify the hook is live, run a self-test:

```bash
echo "tweak" > src/ode/__hook_test  # touch an algorithm-bearing path
git add src/ode/__hook_test
git commit -m "no tags"             # expect rejection
# clean up
git restore --staged src/ode/__hook_test && rm src/ode/__hook_test
```

Bypass with `git commit --no-verify` only after explicitly coordinating
with maintainers — abusing it defeats the entire change-log discipline.
