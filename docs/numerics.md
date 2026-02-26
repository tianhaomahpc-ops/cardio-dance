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

## Known numerical caveats

- Independent heart/torso meshes use nearest-neighbor coupling; this introduces interface projection error.
- Torso Dirichlet constraints are penalty-enforced (approximate), not exact symmetric elimination.
- Conforming whole-body mode reduces interface geometry error but current implementation
  still uses penalty constraints (though interface mapping no longer performs global nearest-neighbor search).
