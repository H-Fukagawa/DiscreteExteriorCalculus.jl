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

### Prism / pyramid: not yet implemented

A wedge-Nédélec basis (prism) and pyramidal-Nédélec basis (pyramid,
known to be tricky due to the apex singularity — see Bedrosian or
Gradinaru-Hiptmair) are next on the list. For mixed polytope meshes
that combine prism / pyramid with tet, the over-relaxed
`nonorthogonal_hodge` remains the only choice.

### Non-axis-aligned hex: not yet implemented

The current hex implementation assumes the hex's first vertex is the
"bottom-left" corner and the edges align with positive `(x, y, z)`.
General trilinear hexes (rotated, sheared, or with curved edges)
require the isoparametric mapping with numerical quadrature.
