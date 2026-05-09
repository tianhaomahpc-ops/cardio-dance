# Numerics

## KSP1: Monodomain + TT06

Per-step operator splitting (no in-step correction):

1. Compute `I_ion^n = I_ion(V^n, w^n)`.
2. Advance TT06 states using ODE substepping (`dt_ode_ms`), with:
   - Rush-Larsen for gating variables.
   - Forward Euler for concentration-like states.
3. Solve monodomain PDE for `V^{n+1}`:
   `A V^{n+1} = B V^n - chi*Cm*M*I_ion^n - chi*Cm*M*I_stim^{n+1/2}`.

Discrete matrices:

- `A = alpha*M + 0.5*K`
- `B = alpha*M - 0.5*K`
- `alpha = chi*Cm/dt_pde`

where `M` is mass matrix, `K` is diffusion stiffness assembled from effective monodomain conductivity tensor.

## KSP2: Extracellular recovery on heart

When `enable_wholebody=1`, after KSP1 solve:

- PDE model: `-div((sigma_i + sigma_e) grad(ue)) = div(sigma_i grad(vm))`
- FEM discrete form:
  `K1 * ue = -K2 * vm`

with:

- `K1` assembled from `(sigma_i + sigma_e)` tensor.
- `K2` assembled from `sigma_i` tensor.

Implementation details:

- Uses the same heart FE space as `Vm`.
- Uses fiber-aligned tensors `(f,s,n)` with config coefficients
  `sigma_i_*`, `sigma_e_*`.
- Handles Neumann null-space by adding one pinned dof penalty (reference potential).

## KSP3: Torso potential solve

Torso equation:

- `-div(sigma_torso * grad(uT)) = 0`

Current implementation supports two coupling modes.

### Mode A: independent heart/torso meshes

1. Build heart boundary samples (`ue`) on heart true dofs.
2. Build nearest-neighbor map from torso true dofs to heart boundary samples.
3. Select constrained torso dofs by distance threshold `interface_map_max_dist_mm`.
4. Solve torso Laplace with penalty-enforced constraints:
   `(KT + beta*D) uT = beta*D*g`
   where `D` is diagonal mask on constrained dofs, `g` is mapped heart `ue`, and `beta=torso_dirichlet_penalty`.

This keeps the system SPD and works with CG in parallel.

### Mode B: conforming whole-body mesh

When `use_conforming_wholebody=1`:

1. Load one parent whole-body mesh.
2. Build heart and torso `ParSubMesh` from volume attributes
   (`heart_volume_attrs`, `torso_volume_attrs`).
3. Run KSP1/KSP2 on heart submesh and KSP3 on torso submesh.
4. Interface data transfer uses parent-vertex-id direct mapping:
   - constructor stage builds `parent_vertex_id -> heart owned true dof` index once;
   - torso constrained true dofs are those whose parent vertex id exists on heart interface;
   - each time step only allgathers heart interface values (no nearest-neighbor search).

Compared to Mode A, this removes geometric mismatch between heart/torso meshes.

## Time loop with whole-body enabled

At each PDE step:

1. KSP1: solve `Vm`.
2. KSP2: recover `ue`.
3. Interface map: `ue(heart) -> g(torso)`.
4. KSP3: solve `uT`.
5. Output fields (`Vm`, `Iion`, `ue`, `uT`) by `output_stride`.

## KSP4: Purkinje cable (1D FE on graph)

When `enable_purkinje=1`, a separate 1D PDE on the Purkinje graph is
advanced in lockstep with the heart 3D PDE. See `docs/purkinje_numerics.md`
§4 for full derivation; summary below.

Cable PDE on per-edge subdivided FE nodes:
$$M_p \frac{dV_p}{dt} + K_p V_p = -M_p (I_{ion}^p + I_{pvj}^p + I_{stim}^p)$$

with `M_p` lumped and `K_p` sparse on the graph; `I_ion^p` from
`StewartPurkinjeModel` (CellML codegen).

Per outer step `Δt_pde`, substepped with `Δt_p ≤ purkinje_dt_ms` (default
0.005 ms because Stewart's I_Na rise time is ~ 0.05 ms):
1. ComputeIion(V_p)
2. AdvanceStates(V_p) — Rush-Larsen + FE
3. Build I_ext = I_pvj + I_stim_root
4. Crank-Nicolson: `(M_p + Δt/2 K_p) V_p^{n+1} = (M_p − Δt/2 K_p) V_p^n − Δt M_p (I_ion + I_ext)`

Linear system solved by MFEM CG + Jacobi (DSmoother) preconditioner;
matrices cached after first dt.

## PVJ coupling (gap-junction current)

Each terminal Purkinje node maps to nearest myocardial true DOF (MPI
MINLOC global selection). Heart-side current:
$$I_{pvj}^h = g_{pvj}\, s_{pvj}\, (V_m^{d(t)} - V_p^{t})$$
added to KSP1 RHS; same magnitude with reversed sign (different DOF) is
added to KSP4 RHS at the corresponding cable terminal node. Both
contributions follow the I_ion>0=outward convention.

When `pvj_smear_radius_mm > 0`, the heart-side injection is distributed
uniformly over all owner-rank DOFs within radius R of the anchor; this
fixes 3D source-sink mismatch where a single-DOF point source is drained
by surrounding resting tissue before I_Na can establish.

Optional `pvj_delay_ms > 0` activates a ring-buffer transport delay so
the cable sees an older sample of V_m.

## Linear-solver backends (KSP1)

`LinearSystemSolver` chooses one of:

| `SimulationConfig` flag | Backend |
|---|---|
| `use_petsc=1` (project default) | PETSc KSP via `mono_*` options; default `config/petsc_asm.opts` configures CG + ASM(overlap=0) + ICC(0) sub-PC |
| `use_hypre_boomeramg=1` | MFEM CG + HypreBoomerAMG (algebraic multigrid) |
| `use_hypre_block_jacobi=1` | MFEM CG + HypreSmoother (l1-Jacobi) |
| (all the above zero) | Plain MFEM CG, no preconditioner |

The Crank-Nicolson matrix `A = chi*Cm/dt * M + 0.5 * K` is symmetric
positive-definite, hence ICC(0) (incomplete Cholesky) is a valid choice
and ~30% cheaper to apply than ILU(0). On a 124k-tet half-ellipsoid LV
problem, single rank, plain CG averages 50 iter/step while ASM+CG
averages 5 iter/step — a 10× iteration reduction. Wall-time win is
modest at this scale; the gap widens with mesh size and with rank
count. See `docs/petsc_asm_vs_cg_iter_bench.md` for the full benchmark.

## Pseudo-ECG far-field probe

When `enable_pseudo_ecg=1`, `PseudoEcg::Sample(t, vm)` computes for each
configured probe `x_p`:
$$\phi(\vec x_p, t) = \frac{\sigma_i}{4\pi\sigma_b} \sum_e V_e\, \nabla V_m^{(e)} \cdot \frac{\vec x_p - \vec c_e}{|\vec x_p - \vec c_e|^3}$$
using element-centroid quadrature (exact for P1 on tets); MPI_Allreduce
SUM, rank 0 writes `t_ms, lead_0, lead_1, ...` rows to CSV.

Output is in arbitrary units (`σ_i / σ_b` ratio is uncalibrated). For
absolute-millivolt traces a torso bath geometry with realistic
conductivity gradients would be required.

## Known numerical caveats

- Independent heart/torso meshes use nearest-neighbor coupling; this introduces interface projection error.
- Torso Dirichlet constraints are penalty-enforced (approximate), not exact symmetric elimination.
- Conforming whole-body mode reduces interface geometry error but current implementation
  still uses penalty constraints (though interface mapping no longer performs global nearest-neighbor search).
- Fiber `.gf` files emitted by `finalize_half_ellipsoid_mesh --emit-fibers` are constant-vector placeholders;
  rule-based Bayer-Trayanova LDRBM helical fibers are not implemented (see `purkinje_numerics.md` §13).
- Stewart Purkinje natural rest is ~ −74 mV (HCN/I_f-driven autorhythmic), not the −91 mV of paced
  steady-state experiments; see `purkinje_numerics.md` §5.
- `pvj_max_dist_mm` rejects terminals farther than this from any heart DOF — typically 50% of generated
  Purkinje terminals are dropped at the default 5 mm cutoff in our half-ellipsoid demo.
