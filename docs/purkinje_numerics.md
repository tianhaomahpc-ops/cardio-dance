# Purkinje + PVJ + Regional Ionic Numerics

This document is the source-of-truth for the mathematical formulation behind
the Purkinje stack. Every algorithm change must update this document in the
same commit so reviewers can audit math/code drift in one diff.

---

## 1. Domain decomposition

Three coupled discrete systems share state every PDE step:

| System          | Discretization                                 | Unknowns                |
|-----------------|------------------------------------------------|-------------------------|
| Myocardium      | 3-D continuous Galerkin P1 on tetrahedra       | $V_m$ at true DOFs       |
| Purkinje cable  | 1-D linear FE on a graph (subdivided edges)    | $V_p$ at FE nodes        |
| PVJ coupling    | Discrete gap-junction current at terminal ↔ DOF| Per-link current $I_{pvj}$|

Time stepping is operator-split Godunov with Crank-Nicolson on diffusion and
Rush-Larsen + Forward Euler on reaction. Both PDEs march on the same outer
$\Delta t_\mathrm{PDE}$; the cable substeps with $\Delta t_p \le \Delta t_\mathrm{PDE}$.

---

## 2. Sign and unit conventions

All currents in this codebase follow the TT06 / CellML convention:

> **$I_{ion} > 0$ is outward (repolarizing).**
> **$I_{stim} < 0$ is depolarizing.**

Units (consistently throughout):

| Quantity     | Unit             | Notes                                           |
|--------------|------------------|-------------------------------------------------|
| Length       | mm               | mesh coordinates, fiber lengths                 |
| Time         | ms               |                                                 |
| Voltage      | mV               | Cell membrane potential                         |
| Concentration| mM (mmol/L)      | $[Na^+]_i$, $[Ca^{2+}]_{SS}$, etc.              |
| $C_m$        | μF/mm²           | membrane capacitance per unit area              |
| $\chi$       | 1/mm             | surface-to-volume ratio                         |
| $\sigma$     | mS/mm            | tissue conductivity                             |
| $g_\mathrm{ion}$| mS/μF         | normalized cell conductance                     |
| $I_{ion}, I_{stim}$| μA/μF      | normalized membrane current (∝ dV/dt)           |

**Verification:** the monodomain RHS in `MonodomainStepper::BuildRhs` reads
`-chi*Cm*M*(I_ion + I_stim + I_pvj)` — the negative sign and `chi*Cm`
prefactor convert from "current per unit area" semantics to units consistent
with the Crank-Nicolson system $A V^{n+1} = B V^n + \mathrm{RHS}$.

---

## 3. Myocardial monodomain (existing, recapped here)

Continuous form on heart domain $\Omega_h$:
$$\chi C_m \partial_t V_m = \nabla\cdot(\boldsymbol\sigma \nabla V_m) - \chi C_m (I_{ion}(V_m, \mathbf{s}) + I_{stim} + I_{pvj}^h),$$
$$\partial_t \mathbf{s} = \mathbf{f}(V_m, \mathbf{s}).$$

Spatial discretization: P1 CG, mass $M$ and stiffness $K$ assembled with the
fiber-aware tensor $\boldsymbol\sigma = \sigma_f\, \mathbf{f}\!\otimes\!\mathbf{f}
+ \sigma_s\, \mathbf{s}\!\otimes\!\mathbf{s} + \sigma_n\, \mathbf{n}\!\otimes\!\mathbf{n}$.
With **AV-delay / fibrosis regions** the stiffness uses an attribute-wise
multiplier $\alpha(x)$:
$$K_{ij} = \int_{\Omega_h} \alpha(x)\, (\nabla\phi_i)^T \boldsymbol\sigma \nabla\phi_j\,\mathrm{d}x,$$
where $\alpha = $ `av_delay_sigma_scale` (0.02) on AV-delay attributes,
`fibrosis_sigma_scale` (0.1) on fibrosis attributes, 1.0 elsewhere.

**Code mapping:** `Assembler.cpp` builds `region_sigma_scale_` as a
`PWConstCoefficient` indexed by mesh attribute, then wraps the
`FiberTensorCoefficient` in `ScalarMatrixProductCoefficient`. Single
DiffusionIntegrator consumes the product.

---

## 4. Purkinje cable

### 4.1 Continuous form

Each Purkinje fiber is modeled as a 1-D cable. With per-edge axial
conductivity $g_a$ (mS/mm) and uniform $C_m^p$ (μF/mm²),
$$C_m^p \partial_t V_p = \partial_x (g_a \partial_x V_p) - I_{ion}^p(V_p, \mathbf{s}_p) - I_{pvj}^p - I_{stim}^p.$$
At branch points (graph vertices with degree > 2) currents are conserved by
construction: the FE assembly's nodal balance enforces Kirchhoff's law
because each incident edge contributes its own $g_a/L_\mathrm{seg}$ entry to
the same row.

### 4.2 Spatial discretization

Each graph edge of length $L$ is subdivided into $k$ FE sub-segments of
length $L/k$. Linear FE on each sub-segment yield the local stiffness
$$K^e_{\mathrm{loc}} = \frac{g_a}{L/k} \begin{pmatrix} 1 & -1 \\ -1 & 1 \end{pmatrix}$$
and a lumped mass $M^e_{\mathrm{loc}} = \tfrac{C_m^p L/k}{2} I$.
Global $K_p$ and $\mathrm{diag}(M_p)$ are sparse-assembled in
`PurkinjeCableSolver::BuildSparseMatrices`.

**Edge weight semantics:** the third token in `.network` edge lines is the
**axial conductance per mm of edge length**, i.e. $g_a$. If absent, the value
defaults to `cfg.purkinje_edge_g_mS_per_mm`. The per-segment conductance
contribution is then $g_\mathrm{seg} = g_a / L_\mathrm{seg}$.

### 4.3 Time stepping

Operator split per outer $\Delta t_\mathrm{PDE}$:

1. Subdivide into $n_\mathrm{sub} = \lceil \Delta t_\mathrm{PDE} / \Delta t_p\rceil$ steps of $\Delta t = \Delta t_\mathrm{PDE}/n_\mathrm{sub}$.
2. Per substep:
   1. Compute $I_{ion}^p(V_p, \mathbf{s}_p)$ at current $V_p$.
   2. Advance $\mathbf{s}_p$ via Rush-Larsen (gates) + Forward Euler (concentrations) using the current $V_p$.
   3. Build external current $I_\mathrm{ext} = I_{pvj}^p + I_{stim}^p$.
   4. Solve $(M_p + \tfrac{\Delta t}{2} K_p) V_p^{n+1} = (M_p - \tfrac{\Delta t}{2} K_p) V_p^n - \Delta t\, M_p (I_{ion}^p + I_\mathrm{ext})$.

Linear solve: CG with Jacobi (`mfem::DSmoother`) preconditioner; the system
matrices are cached and rebuilt only when $\Delta t$ changes.

### 4.4 Stability / dt selection

Crank-Nicolson is unconditionally A-stable for the diffusion split. Reaction
substep is governed by the fastest gating $\tau$ (I_Na: $\tau_m \approx 0.05$ ms
near upstroke). Practical guidance:
- $\Delta t_p \le 0.01$ ms during upstroke (handled by Rush-Larsen exact gate update; FE on concentrations less sensitive)
- $\Delta t_\mathrm{PDE}/\Delta t_p \le 4$ keeps PVJ delay quantization < 0.005 ms

---

## 5. Purkinje ionic model (Stewart 2009)

### 5.1 Reference

Stewart, P., Aslanidi, O. V., Noble, D., Noble, P. J., Boyett, M. R., &
Zhang, H. (2009). *Mathematical models of the electrical action potential of
Purkinje fibre cells.* Phil. Trans. R. Soc. A 367 (1896), 2225–2255.
DOI 10.1098/rsta.2008.0283.

The Stewart model is itself a Purkinje variant of TT06 with HCN funny
current $I_f$ added and modified $I_{to}, I_{Ks}, I_{K1}$ kinetics.

### 5.2 Currents (target list)

$I_\mathrm{ion} = I_{Na} + I_{CaL} + I_{to} + I_{Kr} + I_{Ks} + I_{K1} + I_f^{Na} + I_f^K + I_{NaCa} + I_{NaK} + I_{pCa} + I_{pK} + I_{bNa} + I_{bCa} + I_\mathrm{sus}$

**I_CaL (TT06/Stewart shifted GHK):**
$$I_{CaL} = g_{CaL}\, d f f_2 f_{Cass}\, \frac{4(V-15)F^2}{RT}\, \frac{0.25\,[Ca]_{SS}\,e^{2(V-15)F/RT} - [Ca]_o}{e^{2(V-15)F/RT} - 1}.$$

**Important:** both the prefactor and the exponential argument use the
shifted voltage $V_\mathrm{shift} = V - 15$ mV. A historical bug in this
codebase mixed $V-15$ in the prefactor with $V$ in the exponent and produced
huge non-physiological CaL drive.

### 5.3 State vector and integration

20 states: `[0]V, [1]m, [2]h, [3]j, [4]xr1, [5]xr2, [6]xs, [7]r, [8]s, [9]d,
[10]f, [11]f2, [12]fCass, [13]y(I_f), [14]Cai, [15]CaSR, [16]CaSS, [17]Nai,
[18]Ki, [19]Rprime`.

Per substep:
- States 1..13 (gates): Rush-Larsen $x \leftarrow x_\infty - (x_\infty - x)\,e^{-\Delta t/\tau}$.
- States 14..19 (concentrations + RyR): Forward Euler $x \leftarrow x + \Delta t\,\dot x$, with positivity clamps on $[Ca^{2+}]_*, [Na^+]_i, [K^+]_i$.

### 5.4 Known status (2026-05)

The current `StewartPurkinjeModel` is a **hand-port from the paper**, not
a CellML codegen output. After fixing the I_CaL exponent bug it produces:

| Quantity   | Paper Fig. 2 | Current code |
|------------|--------------|--------------|
| V_rest     | -91 mV       | -85.6 mV ✓ (loose) |
| V_peak     | +30 mV       | **+119 mV ✗** |
| APD90      | 340 ms       | 272 ms ✓ (loose) |
| Repol      | yes          | yes ✓        |

The peak overshoot is the main outstanding defect. Suspected residual
issues:
1. Stim integration in the test (applied as external V perturbation rather
   than as -dt*I_stim term in the dV balance) — masks model behavior.
2. Possible sign or scaling bug in $I_f^{Na}$ / $I_f^K$ contribution during
   upstroke.
3. Concentration scale factor `Cm_pF * 1e-6 / (V_c * F)` may be off by a
   factor of 10–100, leading to fast Nai/Ki drift that distorts reversal
   potentials.

The single-cell test asserts strict physiological ranges and is marked
`WILL_FAIL` in CMake until the model matches reference.

### 5.5 Importing CellML reference code

Long-term remediation:
1. Download Stewart 2009 from
   `https://models.physiomeproject.org/exposure/...` (CellML 1.0 XML).
2. Run OpenCOR or `cellml-tools` to emit C with `initConsts`,
   `computeRates`, `computeVariables` matching the TT06 generated code style.
3. Replace `src/ode/StewartPurkinjeModel.cpp` body with thin wrappers
   delegating to the generated functions, mirroring the
   `tt06_generated.{h,c}` pattern. Keep the gate/concentration index lists
   (`kGateMaps`) as the only hand-written part.
4. Re-enable the strict physiological asserts in
   `test_stewart_single_cell.cpp` and remove the `WILL_FAIL` marker.

---

## 6. Atrial ionic model (Grandi 2011)

`Grandi2011Model` is a **reduced 15-state compact port** retaining the
dominant atrial currents: $I_{Na}, I_{CaL}^\mathrm{simplified}, I_{Kr},
I_{Ks}, I_{to,f}, I_{Kur}, I_{K1}, I_{NaK}, I_{NaCa}, I_{pCa}, I_{bNa},
I_{bCa}$. The simplified $I_{CaL}$ uses a constant driving force $(V - 60)$
rather than full GHK, matching what is typical for "minimal-model" atrial
sims when only conduction (not Ca dynamics) matters.

**Status (2026-05):** no single-cell verification test exists yet for
Grandi. Use this model only for atrial-region conduction studies, not for
APD/AERP analysis.

**Reference (target):** Grandi, E., Pandit, S. V., Voigt, N., Workman, A. J.,
Dobrev, D., Jalife, J., & Bers, D. M. (2011). *Human atrial action
potential and Ca²⁺ model: sinus rhythm and chronic atrial fibrillation.*
Circulation Research 109 (9), 1055–1066.

---

## 7. Passive (leak) model

$$I_{ion}(V) = g_\mathrm{leak}\, (V - V_\mathrm{rest}).$$

Used in AV-delay and fibrosis regions to suppress active depolarization
while allowing electrotonic spread. State-free, so checkpoint is just the
two parameters $(g_\mathrm{leak}, V_\mathrm{rest})$.

---

## 8. PVJ coupling

### 8.1 Continuous form

For each terminal Purkinje node $t$ mapped to nearest myocardial true DOF
$d(t)$:
$$I_{pvj,t} = g_{pvj}\, [V_p^t - V_m^{d(t)}]\, s_{pvj}.$$
- $g_{pvj}$ in mS/μF (config: `pvj_g_mS`, default 0.5)
- $s_{pvj}$ unitless scale (config: `pvj_current_scale`, default 1.0)

### 8.2 Sign convention

> When $V_p > V_m$ (Purkinje more depolarized), $I_{pvj} > 0$.
> The **myocardial RHS contribution** is $+I_{pvj}$ at DOF $d(t)$, treated
> with the same `+` sign as $I_{ion}$ (outward semantics). So the actual
> contribution to the RHS is $-\chi C_m M (\ldots + I_{pvj})$, i.e. when
> $V_p > V_m$ the negative sign **depolarizes** $V_m$. ✓

> The **Purkinje cable** sees $-I_{pvj}$ at terminal $t$ (Newton's third law:
> charge flows from Purkinje to myocardium, so Purkinje loses positive ion
> flux). Implementation in `PurkinjeCableSolver::Advance`:
> `i_pvj_p[fe] = pvj_g * scale * (vp[fe] - Vm)`. Same sign as
> `BuildHeartCouplingCurrent` because both are added to their respective
> `I_ion` term.

### 8.3 Mapping algorithm

`PvjCoupler::BuildMapping`:
1. Each rank computes nearest local true DOF for each terminal.
2. `MPI_Allreduce(MPI_DOUBLE_INT, MPI_MINLOC)` selects the global owner.
3. Terminal-DOF pairs farther than `pvj_max_dist_mm` are dropped.
4. Each owner appends a `LocalPvjLink {purkinje_node, heart_tdof, dist}`
   to its `local_links_`.

Coordinate extraction uses MFEM `FunctionCoefficient` projected onto the P1
space, then `GetTrueDofs` — guarantees exact match with monodomain DOFs.

### 8.4 Optional anatomical delay

When `pvj_delay_ms > 0`:
- After each Purkinje step, the current $V_m$ at terminals is appended to a
  ring buffer (`PvjCoupler::AppendDelayedSample`).
- The cable advance reads the sample with timestamp $\le t_\mathrm{mid} -
  \mathrm{delay}$ (`ReadDelayedSample`) — implements a transport delay of
  `delay_ms` between heart and Purkinje sampling.

Buffer trim keeps history within $4\times$ delay for safety.

---

## 9. Regional ionic dispatcher

### 9.1 Mapping

Mesh element attributes drive region assignment:

| Region    | Cell model         | Default config tag             |
|-----------|--------------------|--------------------------------|
| Ventricle | TT06               | `ventricle_volume_attrs`       |
| Atria     | Grandi 2011        | `atria_volume_attrs`           |
| AV-delay  | Passive            | `av_delay_volume_attrs`        |
| Fibrosis  | Passive            | `fibrosis_volume_attrs`        |

For each owned true DOF, the region with the **highest priority** among
incident elements wins (priority order: AV-delay > Fibrosis > Atria > Ventricle).
This guards against a DOF on a region boundary getting an inappropriate
active model.

### 9.2 State isolation (resolved 2026-05)

Each child model is constructed with size = `LocalDofCount(region)`, NOT the
full DOF count. `RegionalIonicModel::ComputeIion` and `AdvanceStates` gather
$V_m$ into a region-local buffer using `region_indices_[r]`, invoke the child
on that buffer, and scatter the result back into the global I_ion vector. So
a child's per-DOF state index is **region-local**, never indexed by global
true DOF.

Empty regions still allocate a 1-DOF placeholder so the unique_ptr is
constructible, but that placeholder is never invoked because
`per_region_count_[r] == 0` short-circuits the run path.

Tested in `tests/test_regional_isolation.cpp`: a 1-D 4-element mesh with
ventricle on elements 0-1 (TT06) and AV-delay on elements 2-3 (Passive)
verifies that (a) DOFs partition the tdof set, (b) Passive at V_rest yields
exactly 0 I_ion, (c) repeated AdvanceStates with extreme ventricular V does
not corrupt AV-delay output.

---

## 10. Verification matrix

Required tests (see `tests/`):

| Test                              | Scope                                     | Status   |
|-----------------------------------|-------------------------------------------|----------|
| `test_tt06_single_cell`           | TT06 AP shape (existing)                  | ✓ pass    |
| `test_stewart_single_cell`        | Stewart AP physiological range            | ✗ WILL_FAIL (compact port broken) |
| `test_purkinje_cable_smoke`       | Cable diffusion correctness               | ✓ pass    |
| `test_purkinje_cable_cv`          | Cable CV ~ 2-3 m/s on 50 mm Stewart line  | TODO     |
| `test_pvj_active_sign`            | PVJ depolarizes myocardium when $V_p>V_m$ | TODO     |
| `test_grandi2011_single_cell`     | Atrial AP shape                           | TODO     |
| `test_regional_isolation`         | Regional dispatch state isolation         | TODO     |

End-to-end:
- Full Niederer benchmark with regional + Purkinje (pending mesh tools).
- Calibration sweep over $g_{pvj}, \sigma_\mathrm{AV}$ vs target activation
  times (script: `tools/calibrate_purkinje.py`, TODO).

---

## 11. Checkpoint format

### 11.1 Meta file (CHKPT_V3)

Plain-text, written only by rank 0:

```
CHKPT_V3
<world_size>
<step>
<t_ms>
<model_id>
```

`model_id` is `IIonicModel::ModelId()` of the model that wrote the
checkpoint. `LoadLatest` refuses to proceed if the runtime model's
`ModelId()` does not match (or, for legacy V2/V0 metas without `model_id`,
if the runtime model is not TT06).

### 11.2 Shards

- `vm_rank<NNNNNN>.gf` — `mfem::ParGridFunction::Save` per rank.
- `cell_rank<NNNNNN>.bin` — `IIonicModel::SaveState` per rank. Binary
  layout is model-specific; cross-model loads are caught by the model_id
  check before any binary is read.

### 11.3 Format compatibility

| Meta header | Status                           |
|-------------|----------------------------------|
| CHKPT_V3    | current, supported               |
| CHKPT_V2    | TT06-only legacy; refused otherwise |
| (legacy)    | single-rank TT06 only             |

### 11.4 Verified by

`tests/test_checkpoint_model_id.cpp`: writes a TT06 checkpoint, then asserts
`PassiveModel` LoadLatest returns false and a fresh `TT06Model` LoadLatest
returns true with matching step/t_ms.

---

## 12. Change-log discipline

Every commit touching algorithms in this stack must include a one-line
note in the message body:

```
Math:    <equation/property changed>
Code:    <file/function changed>
Mapping: <how the verification stays sound>
```

If a verification test no longer passes after a change, mark it `WILL_FAIL`
in the same commit and add a TODO line; do not silently relax the
assertion.
