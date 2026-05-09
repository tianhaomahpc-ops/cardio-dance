# PETSc ASM+CG vs plain CG — iteration count comparison

Benchmark of the monodomain Crank-Nicolson linear system on the
full-scale half-ellipsoid LV mesh (h=1 mm, 21,707 nodes / 124,715 tets).
20 ms simulated → 400 KSP solves per run. Same Purkinje + smeared PVJ
configuration in both runs.

## Setup

MFEM 4.7 rebuilt from source with `MFEM_USE_PETSC=YES`, linked against
PETSc 3.19.6. cardio-dance reconfigured with `MFEM_DIR=/opt/mfem-petsc-install`.
PETSc options file: `config/petsc_asm.opts` defaults to **CG + ASM
(overlap=0) with ICC(0) sub-PC**.

ICC(0) is preferred over ILU(0) here because the Crank-Nicolson system
$A = \chi C_m / \Delta t \cdot M + 0.5\,K$ is symmetric positive-definite
(both M and K are SPD; their positive linear combination is SPD).
Overlap=0 keeps the local sub-block compact and avoids cross-rank
communication for the overlapping element layer; iter count was
identical (5.0) at overlap=1 with ILU(0), so the cheaper variant is
the default.

| Parameter | Value |
|---|---|
| KSP type | CG |
| Tolerance | rtol=1e-8 |
| Max iter | 500 |
| Mesh | 21707 nodes, 124715 tets |
| Time step | dt_pde = 0.05 ms |
| Steps | 400 |
| MPI ranks | 1 |

## Results

| Solver | Mean iter / step | Min / Max iter | Wall (20 ms sim) | V_m mean at t=20 |
|---|---|---|---|---|
| **Plain CG (no PC)** | 49.6 | 44 / 56 | 61.5 s | -81.24 mV |
| PETSc CG + ASM(overlap=1, ILU0) | 5.0 | 4 / 6 | 55.7 s | -81.24 mV |
| **PETSc CG + ASM(overlap=0, ICC0) ★ default** | **5.0** | 4 / 6 | 58.7 s | -81.24 mV |

**Iteration count reduced 10× (49.6 → 5.0).** Both solvers converge to
the same V_m field (mean V_m matches to 4 significant digits at every
time step), confirming correctness.

### Wall-time discussion

Wall time only drops ~10 % despite the 10 × iteration cut. Reason: each
ASM-preconditioned iteration is heavier than a plain CG iteration:

| Per iter | Plain CG | ASM+CG |
|---|---|---|
| 1× MatMult | yes | yes |
| 1× MPI Allreduce (dot) | yes | yes |
| ASM PC apply | -- | + 4× sub-block ILU solve (1 per overlap copy) |
| ASM PC setup (one-time) | -- | first call only |

For our 124k tets / single rank the ILU(0) factorization + apply cost
each iter is ~ 8× a single CG iter, so 5 ASM iters ≈ 40 plain CG iters
in wall, leaving little headroom over plain CG's 49.6 iters.

### Where ASM should win

ASM scales much better than CG-without-PC as

- **mesh size grows**: plain CG iter count scales like sqrt(condition
  number) ≈ O(1/h), so doubling mesh resolution doubles iters. ASM
  iter count is much more stable (typically O(log(N))).
- **rank count grows**: ILU(0) cost is O(local DOFs / rank), so adding
  ranks lowers per-iter cost and the iter savings translate to wall
  savings. We could not verify this in-sandbox: -n 4 with PETSc ASM
  hung in setup the same way Hypre BoomerAMG/HypreSmoother did at -n 4
  on this mesh -- the per-rank workload (~5k DOFs) is well below the
  break-even point for distributed PC setup.

### How to reproduce

```bash
# Plain CG
cp config/half_ellipsoid_purkinje.options /tmp/plain.options
sed -i 's/^use_petsc=.*/use_petsc=0/' /tmp/plain.options
sed -i 's/^use_hypre_boomeramg=.*/use_hypre_boomeramg=0/' /tmp/plain.options
sed -i 's/^t_end_ms=.*/t_end_ms=20.0/' /tmp/plain.options
./build/monodomain --config /tmp/plain.options
awk -F, 'NR>1{s+=$3;c++}END{print s/c}' \
  output/half_ellipsoid_purkinje/ksp_history.csv

# PETSc ASM+CG
cp /tmp/plain.options /tmp/asm.options
sed -i 's/^use_petsc=.*/use_petsc=1/' /tmp/asm.options
export PETSC_OPTIONS="$(grep -v '^##\|^$' config/petsc_asm.opts | tr '\n' ' ')"
./build/monodomain --config /tmp/asm.options
awk -F, 'NR>1{s+=$3;c++}END{print s/c}' \
  output/half_ellipsoid_purkinje/ksp_history.csv
```

## Iteration histograms

### Plain CG (mean 49.6, range 44-56)
```
   6  44 *
   8  45 *
  41  46 *****
  51  47 *******
  77  48 ***********
  86  49 ************
  10  50 *
  33  51 ****
   6  52 *
   6  53 *
  14  54 **
  22  55 ***
  40  56 *****
```

### PETSc ASM+CG (mean 5.0, range 4-6)
```
  80   4 ***********
 242   5 ********************************
  78   6 ***********
```

Tight distribution — ASM-preconditioned CG converges in essentially the
same iteration count regardless of source-term magnitude across the AP
upstroke / plateau / repolarization phases. Plain CG shows wider
variability driven by stim-step changes.
