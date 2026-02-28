# Monodomain TT06 (MFEM + PETSc/Hypre)

## Build

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
- `checkpoint/niederer_50ms/ionic_rank%06d.bin` (one shard per MPI rank)

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

## Regional fibers (atria + ventricles)

Generate regional `fiber_f/s/n.gf` on an existing 3D mesh:

```bash
mpirun -np 1 ./build/generate_regional_fibers \
  --mesh benchmarks/wholebody/wholebody_conforming.mesh \
  --out-dir benchmarks/wholebody/fibers_regional \
  --atria-attrs 11 \
  --atria-anchor CT:path,/path/to/ct.csv \
  --atria-anchor BB:path,/path/to/bb.csv \
  --atria-anchor LSPV:ring,/path/to/lspv.vtk \
  --atria-anchor RSPV:ring,/path/to/rspv.vtk \
  --ventricle-attrs 12 \
  --vent-apex-bdr-attrs 1 \
  --vent-base-bdr-attrs 2 \
  --vent-epi-bdr-attrs 3 \
  --vent-lv-bdr-attrs 4 \
  --vent-rv-bdr-attrs 5
```

Notes:

- Atria pipeline follows anchor-tagging -> anchor vector initialization -> volumetric diffusion
  -> defect repair -> final orthonormalization.
- CSV landmarks support optional affine preprocessing via `--csv-scale` and `--csv-shift`.
- Ventricle pipeline follows the cardioid `fiberp` style: four Laplace-like scalar fields
  (`psi_ab`, `phi_epi`, `phi_lv`, `phi_rv`) + bislerp-based orientation synthesis.
- Output files are written to `--out-dir` as:
  - `fiber_f.gf`
  - `fiber_s.gf`
  - `fiber_n.gf`

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

- `use_petsc=0`: MFEM CG
- `use_petsc=0` + `use_hypre_boomeramg=1`: MFEM `CG + HypreBoomerAMG` preconditioner
- `use_petsc=1`: PETSc KSP(CG), preconditioner configurable via `PETSC_OPTIONS`
  with prefix `mono_` (e.g. `-mono_pc_type asm -mono_sub_pc_type jacobi`)
- ASM/GASM subdomain policy:
  - `petsc_use_geometric_asm=1` enables explicit subdomain setup from MFEM partition ownership
    (one local subdomain per MPI rank; aligned with MFEM/ParMETIS decomposition)

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
Conforming regional reference config: `config/wholebody_conforming_regional.options`

Detailed workflow and literature-comparison checklist:

- `docs/wholebody_workflow.md`
- `docs/literature_comparison.md`
- `docs/petsc_asm_geometric_report.md`
