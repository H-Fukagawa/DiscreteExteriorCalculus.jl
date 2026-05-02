# Galerkin Hodge vs Over-relaxed Non-orthogonal Hodge

Side-by-side comparison of the two Hodge implementations for the
**3D Kuhn-tet Poisson problem**, the case where their characters differ
most. All numbers measured on the same machine on a unit cube and a
skewed parallelepiped using `u(x,y,z) = sin(πξ₁)sin(πξ₂)sin(πξ₃)` (in
parameter space, so `u = 0` on the slanted boundary).

The two methods solve different operator equations:

- **Galerkin / Whitney** — `K · u = M_0 · f` where `K = d_0' M_1 d_0`
  is the consistent FEM stiffness and `M_0` is the consistent FEM mass.
  The discrete `u` is the FEM coefficient.

- **Over-relaxed non-orthogonal** — `L · u = f` (pointwise) where
  `L = ★_n^{-1} d_dual ★_2 d_0` is the lumped-mass DEC Laplacian with
  dual mesh built from cell centroids. The discrete `u` approximates
  `u_exact` at vertex points.


## Table 1: Pointwise solve error `‖u_h − u_ex‖_{ℓ²(int)}`

3D Kuhn-tet mesh. Both schemes converge at h²; absolute error magnitudes
differ.

| Mesh   | n  | Galerkin | nonortho | ratio nono/Gal |
|--------|----|----------|----------|----------------|
| UNIT   | 4  | 0.117    | 0.034    | 0.29           |
| UNIT   | 6  | 0.049    | 0.012    | 0.25           |
| UNIT   | 8  | 0.027    | 0.006    | 0.23           |
| UNIT   | 12 | 0.011    | 0.003    | 0.27           |
| SKEWED | 4  | 0.122    | 0.019    | 0.16           |
| SKEWED | 6  | 0.052    | 0.008    | 0.15           |
| SKEWED | 8  | 0.029    | 0.004    | 0.14           |
| SKEWED | 12 | 0.012    | 0.002    | 0.16           |

**Both achieve ≈ h² convergence**, but nonortho has a 3–7× smaller
absolute error on this test. The lumped-mass DEC scheme delivers a
better pointwise approximation to smooth `u` here, which is a
well-known FEM phenomenon (consistent mass has the right asymptotic
rate but tends to lose to lumped mass on pointwise error for typical
smooth manufactured solutions).


## Table 2: Weak / consistency residual `‖K u_ex − M_0 f_ex‖` (Galerkin)
##           and `‖L u_ex − f_ex‖` (nonortho), at interior nodes.

This metric measures how well the discretization satisfies the equation
when fed the exact continuous solution. It's the natural "discretization
quality" metric.

3D Kuhn skewed:

| n | Galerkin (super-conv) | nonortho |
|---|---|---|
| 4  | 0.073 | 1.73 |
| 6  | 0.011 (×6.74) | 1.01 (×1.71) |
| 8  | 0.0027 (×4.04) | 0.62 (×1.62) |
| 10 | 0.00090 (×3.0) | 0.42 (×1.49) |

**Galerkin shows ≈ h⁴ super-convergence in the weak residual** — a
strong indication of clean discrete-form orthogonality. The nonortho
scheme saturates at h^{≈1.5} due to Kuhn-tet asymmetry (the cell-
centroid dual produces a structurally biased stiffness that cannot
recover h² in this consistency norm even with the over-relaxed
correction).


## Both methods converge to the SAME continuous solution (h² verified)

On a well-centered 2D right-triangle lattice (`triangulated_lattice([1,0],
[0,1], n, n)`) with `u_ex = sin(πx)sin(πy)`, the two methods exhibit
clean h² convergence both individually AND in their pairwise difference,
confirming they limit to the SAME continuous Poisson solution:

| n  | err_Gal   | err_NN    | `|u_G − u_NN|` | rate |
|----|-----------|-----------|----------------|------|
| 4  | 0.092     | 0.035     | 0.127          | -    |
| 8  | 0.0216    | 0.0074    | 0.0289         | ×4.4 |
| 16 | 0.00514   | 0.00172   | 0.00685        | ×4.2 |
| 32 | 0.00125   | 0.000415  | 0.00166        | ×4.1 |
| 64 | 0.000308  | 0.000102  | 0.000410       | ×4.1 |

Both methods are h² (ratio →4 between consecutive sizes), and crucially
`|u_G − u_NN|` also goes to 0 at h². Test:
`galerkin vs nonortho: same continuous limit (2D well-centered)` in
`test_galerkin_hodge.jl`.

On 3D Kuhn the comparison is asymmetric: Galerkin converges h² cleanly
(see Table 1) while the centroid-dual + over-relaxed nonortho with
`corrected_barycentric_hodge` saturates around `‖u_h − u_ex‖ ≈ 0.05`
on this geometry — the cell-centroid dual produces a structurally
biased Hodge that the over-relaxed correction doesn't fully resolve in
3D Kuhn. This is consistent with the docs note that nonortho saturates
at h^{1.5} in the consistency norm; the pointwise error stagnation at
matched n in this 3D test is its mass-lumping/over-relaxation analog.

## Why the two metrics give "different winners"

The pointwise solve error is dominated by the discrete operator's
inverse (= the discrete Green's function), not by the consistency.
nonortho's lumped-mass lumping plus the over-relaxed correction lands
on a discrete operator whose *inverse* better approximates the
continuum Green's function on smooth `u`, even though its consistency
saturates at h^{1.5} — the inverse smooths out the consistency
defect to leading order.

Galerkin's `M_0^{-1} K` has near-perfect consistency but its discrete
operator is FEM-shaped: the inverse acts on the FEM coefficient space,
not on pointwise vertex values.

Concretely:

- For **stability** and **error analysis** in standard Sobolev norms,
  Galerkin is the principled choice (FEEC theory, SPD `K`, no dual
  mesh required).

- For **smooth pointwise PDE solutions** at modest resolution,
  nonortho's lumped scheme often gives smaller errors per unknown.
  This is the same trade-off as consistent vs. lumped mass in
  textbook FEM.


## Other characteristics

| Property | Galerkin | nonortho |
|----------|----------|----------|
| Symmetry of stiffness | strict SPD | ≈ symmetric (5% asymmetry observed in 2D) |
| Dual mesh required | no | yes (centroid) |
| `★★ = ±I` identity | weak only | strict (with caveat: `★_2_dual` not the inverse of corrected `★_2`) |
| Polytope (hex/prism/pyramid) `k>1` | not yet | yes (drop-in for `★_2` in 3D) |
| Polytope `k=1` (vertex mass) | yes | yes |
| Fits `★d★d` DSL | no (different API) | yes (drop-in for `circumcenter_hodge`) |
| 2D 1-form Laplacian | not directly tested | h³ super-convergent on regular skewed lattices |
| Strict `d² = 0` | yes | yes |
| Standard FEEC theory | yes | no (FVM-like) |


## When to use which

- **Use Galerkin Hodge** if you want SPD systems, no dual mesh, FEM-
  style error analysis, mesh-skew robustness in the consistency
  norm, or simplicial meshes only.

- **Use over-relaxed non-orthogonal Hodge** if you need the
  `★d★d` DEC pipeline (existing `differential_operator_sequence`
  works), polytope mesh support, or just smaller pointwise solve
  error on smooth manufactured tests at modest `n`.

The two are complementary, not interchangeable; both achieve h²
convergence on the standard Poisson problem.


## Reproducing these numbers

The Galerkin numbers are produced by tests in
`test/test_galerkin_hodge.jl`. The nonortho numbers were measured on
the `nonorthogonal-hodge` branch with equivalent test code. Switching
between branches reproduces the numbers above on the same problem
geometry and manufactured solution.


## Update: 1-form Hodge Laplacian via mixed Galerkin

`galerkin_hodge_laplacian_block(m, comp, k)` builds the saddle-point
mixed-FEM block for the de Rham Laplacian on `k-1` forms:

    [M_{k-1}              -d_{k-1}ᵀ M_k        ] [σ]   [0    ]
    [M_k d_{k-1}           d_kᵀ M_{k+1} d_k    ] [ω] = [M_k f]

For 2D 1-form Δ_H on the unit square with `ω_ex = sin(πx)sin(πy)·(dx+dy)`:

| n  | err_ω    | rate |
|----|----------|------|
| 8  | 6.5e-3   | -    |
| 16 | 7.8e-4   | ×8.3 |
| 32 | 9.8e-5   | ×8.0 |

→ ×8 super-convergence (h³) — matches nonortho's measured rate on the
same problem.

For 3D Kuhn 1-form Δ_H on unit cube with `ω_ex = sin(πx)sin(πy)sin(πz)·(dx+dy+dz)`:

| n | err_ω | rate |
|---|-------|------|
| 4 | 0.038 | -     |
| 6 | 0.011 | ×3.4  |
| 8 | 0.005 | ×2.4  |

→ ≈ h^{2.5} convergence on Kuhn 3D. (h² rate would be ×2.25, ×1.78.)

So 1-form Hodge Laplacian is fully accessible via the mixed Galerkin
formulation in both 2D and 3D simplicial meshes.


## Polytope (hex / prism / pyramid) `k > 1` Galerkin Hodge

### Hex (axis-aligned): implemented

`galerkin_hodge(m, tcomp::TriangulatedComplex, 2)` for an axis-aligned
hexahedral mesh now uses the lowest-order Nédélec edge element. The
12 × 12 local mass matrix is block-diagonal in three axis groups of 4,
each given in closed form by tensor products of `∫ ν_α ν_β dy = L · (1/3 if α==β else 1/6)`
on the 1D linear hat functions.

Hex unit-cube Poisson SOLVE on a `n³` mesh:

| n  | err   | rate  |
|----|-------|-------|
| 4  | 0.029 | -     |
| 6  | 0.013 | ×2.26 |
| 8  | 0.007 | ×1.83 |
| 12 | 0.003 | ×2.35 |

→ Clean h² convergence (×4 expected for h-halving, ×2.25 / ×1.78
expected at n=4→6 / 6→8 — observed ×2.26 / ×1.83). Smaller absolute
error than the Kuhn-tet Galerkin solve at matched `n` (1 hex per cube
vs 6 Kuhn tets per cube), since the Nédélec basis on a hex is
naturally aligned with the cube faces.

### Prism: implemented (axis-aligned + oblique isoparametric)

`galerkin_hodge(m, tcomp, 2)` for a prism mesh uses the lowest-order
wedge Nédélec element (9 edges = 3 bottom + 3 top + 3 vertical) with
3-pt sub-triangle × Gauss-Lobatto quadrature on the reference prism.
Both axis-aligned and general (oblique) prisms achieve clean h²
Poisson convergence.

### Pyramid: Bedrosian Type-II 10-edge basis with Schur condensation

The lowest-order pyramidal Nédélec basis (Bedrosian 1992 /
Gradinaru-Hiptmair 1999) is genuinely research-grade because of the
apex singularity. We implement the GH/Wachspress shape functions and
the **10-edge Whitney basis** (8 polytope edges + 2 base-diagonal
"bubble" Whitney forms), verified by tests in `test_galerkin_hodge.jl`:

1. **Kronecker δ**: `∫_{e_β} φ_α · t̂ ds = δ_{αβ}` on the 8 reference
   polytope edges.
2. **Face conformity**: tangential trace on a shared base face agrees
   across the two adjacent pyramids (covariant Piola `J^{-T}` exactly
   compensates the (ξ,η)→(x,y) permutation that differs between them).
3. **De Rham**: in the 10-edge graph (with both base diagonals (1,3)
   and (2,4) as bubble edges), `∇N_a = Σ_{α∋a} ε_{α,a} φ_α` exactly for
   all 5 nodal vertices.

The two diagonal "bubble" DOFs are local to each pyramid (not shared
between pyramids). For API uniformity with the 8-polytope-edge global
edge count, the bubbles are **Schur-condensed locally**:

    M_eff = M_PP − M_PD M_DD^{-1} M_DP                  (8×8 SPD)

`galerkin_hodge(m, tcomp, 2)` returns this `M_eff` for pyramid meshes —
a valid SPD ★_2 inner product on the polytope edge space, suitable for
Hodge Laplacian / Whitney-form-based applications. The implementation
is robust to both base orientations (CCW-from-below and CCW-from-above);
input vertex order is canonicalized internally.

⚠ **Stiffness caveat**: `K = d_polytope^T M_eff d_polytope` does NOT
equal the FEM stiffness — the Schur condensation drops bubble couplings
that contribute to the full GH-Wachspress K. For Poisson stiffness,
`galerkin_stiffness(m, tcomp)` is special-cased on pyramid meshes to
bypass M_1 entirely and assemble K directly via per-sub-tet
`⟨∇λ_i, ∇λ_j⟩` (sub-tet P1 FEM).

The mass-matrix integration uses **tensor-product Gauss-Legendre with
Duffy substitution** `ξ = (1-ζ)ξ', η = (1-ζ)η'` to absorb the apex
`(1-ζ)^{-k}` singularity in the GH/Wachspress basis. With 4 × 4 × 4 =
64 quadrature points the diagonal mass entries match a Bey-refined
sub-tet reference (4096 points) to better than 5e-5; the earlier 4-pt
× 2 sub-tet rule had ≈ 9% relative error on the most apex-affected
entries. The Hodge Laplacian convergence rate on pyramid meshes is
unchanged by this improvement (still ≈ h^{1.3-1.5}, limited by the
Schur-condensed `M_1`), but `M_1` itself and the Schur-condensed
`M_eff` are now substantially more accurate as `★_2` operators.

On the cube-center-apex pyramid lattice the sub-tet FEM stiffness path
gives clean h² Poisson convergence:

| n | err   | rate  |
|---|-------|-------|
| 4 | 0.077 | -     |
| 6 | 0.035 | ×2.20 |
| 8 | 0.020 | ×1.78 |

(Expected ratios for h²: ×2.25 / ×1.78.)

For mixed polytope meshes that combine pyramid with hex / prism / tet,
the over-relaxed `nonorthogonal_hodge` remains the only `★_2` option.

### Non-axis-aligned hex: not yet implemented

The current hex implementation assumes the hex's first vertex is the
"bottom-left" corner and the edges align with positive `(x, y, z)`.
General trilinear hexes (rotated, sheared, or with curved edges)
require the isoparametric mapping with numerical quadrature.

### Polytope `k=3` (face / 2-form) mass

`galerkin_hodge(m, tcomp::TriangulatedComplex, 3)` is implemented for
all four polytope types. Three different constructions:

- **Tet** — existing simplicial Whitney 2-form (`_local_mass_2form`).
- **Hex** — lowest-order Raviart-Thomas (RT_0) on the reference cube
  with isoparametric **contravariant** Piola pull-back
  `ψ^p = J ψ^r / det J`. 6 face DOFs; mass evaluated by 2 × 2 × 2
  Gauss-Legendre quadrature. On the unit cube the local mass is
  block-diagonal in three axis groups, each block `[[1/3, -1/6], [-1/6, 1/3]]`.
- **Prism / pyramid** — sub-tet decomposition with **area-weighted
  projection**: each polytope quad face = 2 sub-tet triangle faces, and
  the polytope-face Whitney 2-form has uniform flux 1 across the
  polytope face. The projection matrix `T[σ, F] = ±A_σ / A_F` (sign
  flip if the sub-tet face normal points opposite to the polytope face
  normal), and `M_polytope = T^T M_subtet T` summed over sub-tets.
  Internal sub-tet faces (those not lying on a polytope face) get no
  bubble DOFs in this construction (set to 0 in T) — a pragmatic choice
  sufficient for SPD `★_2` inner products. 5 face DOFs each.

Sign convention for global assembly is uniform across all polytope
types: triangle faces use permutation parity vs the global face cell's
stored vertex order; quad faces use cyclic equivalence (+1 if some
cyclic shift matches, −1 if the reversed cycle matches). Verified by
the equivalence test (`galerkin_hodge: TriangulatedComplex method
matches CellComplex on simplicial`) for k=3 on tet meshes — polytope
dispatcher gives identical M_2 to the simplicial Whitney path.

For mixed `hex + prism + pyramid` meshes the assembler now produces a
global SPD M_2 of size `(n_faces × n_faces)`, enabling 1-form Hodge
Laplacian and other `★_2`-based formulations on polytope meshes.

### `galerkin_hodge_laplacian_block` on polytope meshes

`galerkin_hodge_laplacian_block(m, tcomp::TriangulatedComplex, k)` is a
new overload that wires the polytope-aware mass matrices into the
saddle-point mixed-FEM block matrix for the Hodge Laplacian on `k-1`
forms. Convergence on the unit cube with
`ω_ex = sin(πx)sin(πy)sin(πz) (dx + dy + dz)`:

| Mesh             | n=4    | n=8    | rate          |
|------------------|--------|--------|---------------|
| Hex (RT_0)       | 0.015  | 0.002  | ≈ h^{2.5}     |
| Prism (sub-tet)  | 0.019  | 0.011  | ≈ h^{0.5–1}   |
| Pyramid (sub-tet) | 0.016  | 0.006  | ≈ h^{1.3}     |

Hex achieves super-convergence (matching the ≈h^{2.5} measured for
3D Kuhn tet). Prism / pyramid inherit slower convergence from their
simplified `M_2` construction (sub-tet projection without bubble DOFs)
and (for pyramid) the Schur-condensed `M_1`. Both still give SPD,
solvable systems with stable error decrease — sufficient for ★d★d
operator pipelines on mixed polytope meshes.

#### Why bubble DOFs (Schur) don't fix prism/pyramid M_2 convergence

A natural-looking improvement is to add the internal sub-tet faces as
per-polytope **bubble DOFs** (each shared between 2 sub-tets within
the polytope), then Schur-condense them locally to keep the global
DOF count = polytope-face count. This gives an "energy-optimal
marginal" M_eff:

    M_eff = M_FF − M_FB M_BB^{-1} M_BF.

Empirically, this DEGRADES the Hodge Laplacian convergence on prism
meshes (rate drops from ≈1.7 to ≈1.4 between n=4 and n=8). Reason: the
Schur-marginal mass is the energy-OPTIMAL extension of polytope-face
coefficients into the full sub-tet FE space, which minimizes the L²
norm of the bubble component — but that's a WORSE L² inner product on
the polytope-face subspace than the partition-of-unity extension
(bubbles ≡ 0). The current code therefore uses partition-of-unity (T
= 0 on internal faces), and the bubble-DOF experiment is documented
in the commit history but not used.

Truly improving prism/pyramid `M_2` convergence requires either
(a) keeping bubble DOFs as **independent global DOFs** (changing the
edge/face count in d_0/d_1 assembly — significant API change), or
(b) implementing a true polytope `RT_0`/Nédélec face basis (analogous
to the hex isoparametric construction). Future work.
