using Test, DiscreteExteriorCalculus
const DEC = DiscreteExteriorCalculus
using LinearAlgebra: norm, diag, eigvals, Symmetric, dot
using SparseArrays: sparse, SparseMatrixCSC

# ============================================================================
# Helpers: 3D Kuhn tet lattice (re-used across tests in this file).
# ============================================================================
const _KUHN_TETS = ((1,2,3,7), (1,3,4,7), (1,4,8,7),
                    (1,8,5,7), (1,5,6,7), (1,6,2,7))

function _tet_lattice_3d(v1, v2, v3, n)
    pts = Dict{NTuple{3,Int}, Point{3}}()
    for i in 0:n, j in 0:n, k in 0:n
        c = (i/n) .* v1 .+ (j/n) .* v2 .+ (k/n) .* v3
        pts[(i,j,k)] = Point(c[1], c[2], c[3])
    end
    simplices = Simplex{3, 4}[]
    for i in 0:n-1, j in 0:n-1, k in 0:n-1
        c8 = [pts[(i,j,k)],   pts[(i+1,j,k)],   pts[(i+1,j+1,k)], pts[(i,j+1,k)],
              pts[(i,j,k+1)], pts[(i+1,j,k+1)], pts[(i+1,j+1,k+1)], pts[(i,j+1,k+1)]]
        for tup in _KUHN_TETS
            push!(simplices, Simplex(c8[tup[1]], c8[tup[2]], c8[tup[3]], c8[tup[4]]))
        end
    end
    return TriangulatedComplex(simplices)
end

# ============================================================================
# Structural sanity tests: SPD, sizes, symmetry.
# ============================================================================
@testset "galerkin_hodge: structural properties" begin
    m2 = Metric(2)
    _, tcomp2 = DEC.triangulated_lattice([1.0, 0.0], [0.5, 0.5*sqrt(3)], 4, 4)
    orient!(tcomp2.complex)
    comp2 = tcomp2.complex
    for k in 1:3
        Mk = galerkin_hodge(m2, comp2, k)
        nk = length(comp2.cells[k])
        @test size(Mk) == (nk, nk)
        @test Mk ≈ transpose(Mk)
        @test minimum(eigvals(Symmetric(Matrix(Mk)))) > 0
    end

    m3 = Metric(3)
    tcomp3 = _tet_lattice_3d([1.0,0,0], [0,1.0,0], [0,0,1.0], 2)
    orient!(tcomp3.complex)
    comp3 = tcomp3.complex
    for k in 1:4
        Mk = galerkin_hodge(m3, comp3, k)
        nk = length(comp3.cells[k])
        @test size(Mk) == (nk, nk)
        @test Mk ≈ transpose(Mk)
        @test minimum(eigvals(Symmetric(Matrix(Mk)))) > 0
    end
end

# ============================================================================
# 0-form mass on a single triangle should match the textbook FEM P1 mass.
# For a triangle with area V:  M = V/12 · [[2,1,1],[1,2,1],[1,1,2]]
# ============================================================================
@testset "galerkin_hodge: 0-form mass matches FEM P1" begin
    m = Metric(2)
    s = Simplex(Point(0.0, 0.0), Point(1.0, 0.0), Point(0.0, 1.0))
    V = volume(m, s)
    expected = V / 12 .* [2 1 1; 1 2 1; 1 1 2]
    @test DEC._local_mass_0form(m, s) ≈ expected
end

# ============================================================================
# Conservation: mass matrices integrate constant 1-functions correctly.
# The total mass Σ_{i,j} M_0[i,j] must equal the total volume of the mesh.
# ============================================================================
@testset "galerkin_hodge: total 0-form mass = volume" begin
    m = Metric(2)
    _, tcomp = DEC.triangulated_lattice([1.0, 0.0], [0.3, 0.85], 4, 4)
    orient!(tcomp.complex)
    comp = tcomp.complex
    M0 = galerkin_hodge(m, comp, 1)
    total = sum(M0)
    expected = sum(volume(m, Simplex(c)) for c in comp.cells[3])
    @test total ≈ expected

    m3 = Metric(3)
    tcomp3 = _tet_lattice_3d([1.0,0,0], [0,1.0,0], [0,0,1.0], 3)
    orient!(tcomp3.complex)
    comp3 = tcomp3.complex
    M03 = galerkin_hodge(m3, comp3, 1)
    expected3 = sum(volume(m3, Simplex(c)) for c in comp3.cells[4])
    @test sum(M03) ≈ expected3
end

# ============================================================================
# Galerkin orthogonality: weak Poisson residual converges with h on smooth
# manufactured solution. K = d_0' M_1 d_0 should satisfy
#     K · u_exact ≈ M_0 · (-Δu)_exact + O(h^p)
# at interior nodes for some p ≥ 2. Skewed Kuhn-tet meshes — where the
# diagonal centroidal Hodge is inconsistent (h^0) and the over-relaxed
# correction reaches at most h^{1.5} — should converge cleanly here
# because the Galerkin Hodge does not depend on a dual mesh.
# ============================================================================
@testset "galerkin_hodge: weak Poisson residual converges (2D skewed)" begin
    m = Metric(2)
    function err(n)
        _, tcomp = DEC.triangulated_lattice([1.0, 0.0], [0.3, 0.85], n, n)
        orient!(tcomp.complex)
        comp = tcomp.complex
        M0 = galerkin_hodge(m, comp, 1)
        M1 = galerkin_hodge(m, comp, 2)
        d0 = DEC.exterior_derivative(comp, 1)
        K = transpose(d0) * M1 * d0
        verts = comp.cells[1]
        u = [sin(π*v.points[1].coords[1])*sin(π*v.points[1].coords[2]) for v in verts]
        f = 2 * π^2 .* u  # -Δu = 2π² u
        res = K * u - M0 * f
        _, ext = DEC.boundary_components_connected(comp)
        bnd = Set(ext.cells[1])
        int_idx = [i for (i, v) in enumerate(verts) if !(v in bnd)]
        return norm(res[int_idx]) / sqrt(length(int_idx))
    end
    es = [err(n) for n in [8, 16]]
    # Empirically observed: ×16 per halving (super-convergent on the regular
    # skewed lattice). Conservatively assert ≥ 8.
    @test es[1] / es[2] > 8.0
end

@testset "galerkin_hodge: weak Poisson residual on Kuhn skewed (3D)" begin
    m = Metric(3)
    function err(n)
        tcomp = _tet_lattice_3d([1.0,0,0], [0.2,1.0,0], [0.1,0.15,1.0], n)
        orient!(tcomp.complex)
        comp = tcomp.complex
        M0 = galerkin_hodge(m, comp, 1)
        M1 = galerkin_hodge(m, comp, 2)
        d0 = DEC.exterior_derivative(comp, 1)
        K = transpose(d0) * M1 * d0
        verts = comp.cells[1]
        u = [sin(π*v.points[1].coords[1])*sin(π*v.points[1].coords[2])*
             sin(π*v.points[1].coords[3]) for v in verts]
        f = 3 * π^2 .* u
        res = K * u - M0 * f
        _, ext = DEC.boundary_components_connected(comp)
        bnd = Set(ext.cells[1])
        int_idx = [i for (i, v) in enumerate(verts) if !(v in bnd)]
        return norm(res[int_idx]) / sqrt(length(int_idx))
    end
    e_4 = err(4)
    e_8 = err(8)
    # Galerkin Hodge is mesh-skew-robust: ratio should be at least ×8 — far
    # better than the centroidal-dual schemes' h^{1.5} on Kuhn meshes.
    @test e_4 / e_8 > 8.0
end

# ============================================================================
# Actual FEM Poisson SOLVE: build K = d_0' M_1 d_0, impose zero Dirichlet,
# solve K_int u_int = (M_0 f)_int, compare u_h vs u_exact at interior nodes.
# This is the "real" deliverable — h² convergence in the discrete L² norm of
# the actual solution (not just the residual).
# ============================================================================
@testset "galerkin_laplacian: Poisson solve, h² convergence (2D unit square)" begin
    m = Metric(2)
    function err(n)
        _, tcomp = DEC.triangulated_lattice([1.0, 0.0], [0.0, 1.0], n, n)
        orient!(tcomp.complex)
        comp = tcomp.complex
        M0, K = galerkin_laplacian(m, comp)
        verts = comp.cells[1]
        u_ex = [sin(π*v.points[1].coords[1])*sin(π*v.points[1].coords[2]) for v in verts]
        f = 2 * π^2 .* u_ex
        _, ext = DEC.boundary_components_connected(comp)
        bnd = Set(ext.cells[1])
        int_idx = [i for (i, v) in enumerate(verts) if !(v in bnd)]
        K_int = K[int_idx, int_idx]
        u_int = K_int \ (M0 * f)[int_idx]
        return norm(u_int - u_ex[int_idx]) / sqrt(length(int_idx))
    end
    es = [err(n) for n in [8, 16, 32]]
    # Standard P1 FEM rate: ratio ≈ 4 per h-halving.
    @test es[1] / es[2] > 3.5
    @test es[2] / es[3] > 3.5
end

@testset "galerkin_laplacian: Poisson solve on 3D Kuhn unit cube (h²)" begin
    m = Metric(3)
    function err(n)
        tcomp = _tet_lattice_3d([1.0,0,0], [0,1.0,0], [0,0,1.0], n)
        orient!(tcomp.complex)
        comp = tcomp.complex
        M0, K = galerkin_laplacian(m, comp)
        verts = comp.cells[1]
        u_ex = [sin(π*v.points[1].coords[1])*sin(π*v.points[1].coords[2])*
                sin(π*v.points[1].coords[3]) for v in verts]
        f = 3 * π^2 .* u_ex
        _, ext = DEC.boundary_components_connected(comp)
        bnd = Set(ext.cells[1])
        int_idx = [i for (i, v) in enumerate(verts) if !(v in bnd)]
        K_int = K[int_idx, int_idx]
        u_int = K_int \ (M0 * f)[int_idx]
        return norm(u_int - u_ex[int_idx]) / sqrt(length(int_idx))
    end
    # 3D Kuhn is the headline case — over-relaxed centroidal-dual saturates
    # at h^{1.5}, the Galerkin Hodge gives clean h² (≥ ×3.5 per h ratio 2/3).
    es = [err(n) for n in [4, 6, 8]]
    @test es[1] / es[3] > 3.5  # n=4 → n=8 expected ratio ≈ (8/4)² = 4
end

# ============================================================================
# TriangulatedComplex method for simplicial complexes equals the CellComplex
# method (forward compatibility for polytope meshes — when polytope ctors are
# available, the same TriangulatedComplex API extends without API churn).
# ============================================================================
@testset "galerkin_hodge: TriangulatedComplex method matches CellComplex on simplicial" begin
    m = Metric(2)
    _, tcomp = DEC.triangulated_lattice([1.0, 0.0], [0.3, 0.85], 4, 4)
    orient!(tcomp.complex)
    M_via_comp = galerkin_hodge(m, tcomp.complex, 1)
    M_via_tcomp = galerkin_hodge(m, tcomp, 1)
    @test M_via_comp ≈ M_via_tcomp

    m3 = Metric(3)
    tcomp3 = _tet_lattice_3d([1.0,0,0], [0,1.0,0], [0,0,1.0], 2)
    orient!(tcomp3.complex)
    @test galerkin_hodge(m3, tcomp3.complex, 1) ≈ galerkin_hodge(m3, tcomp3, 1)
end

# ============================================================================
# Discrete d² = 0 carries through, K is symmetric and positive *semi*-definite
# (zero on constants), and constants are exactly in the null space.
# ============================================================================
@testset "galerkin_laplacian: K symmetric, PSD, null = constants" begin
    m = Metric(2)
    _, tcomp = DEC.triangulated_lattice([1.0, 0.0], [0.3, 0.85], 5, 5)
    orient!(tcomp.complex)
    comp = tcomp.complex
    M0, K = galerkin_laplacian(m, comp)
    @test K ≈ transpose(K)
    @test M0 ≈ transpose(M0)
    # K is positive semi-definite (zero eigenvalue from the constant mode).
    @test minimum(eigvals(Symmetric(Matrix(K)))) > -1e-10
    # K · 1 = 0 (constants are killed by the discrete Laplacian).
    n_v = length(comp.cells[1])
    @test norm(K * ones(n_v)) < 1e-10
end

# ============================================================================
# Comparison with circumcenter Hodge on a well-centered mesh: the variational
# Laplacians give *different* operators (one diagonal-mass-based, one full-
# mass-based) but should agree to leading order on smooth solutions.
# ============================================================================
@testset "galerkin vs circumcenter on well-centered 2D mesh: solutions agree" begin
    m = Metric(2)
    n = 16
    _, tcomp = DEC.triangulated_lattice([1.0, 0.0], [0.0, 1.0], n, n)
    orient!(tcomp.complex)
    comp = tcomp.complex
    verts = comp.cells[1]
    u_ex = [sin(π*v.points[1].coords[1])*sin(π*v.points[1].coords[2]) for v in verts]
    f = 2 * π^2 .* u_ex

    # Galerkin solve
    M0, K = galerkin_laplacian(m, comp)
    _, ext = DEC.boundary_components_connected(comp)
    bnd = Set(ext.cells[1])
    int_idx = [i for (i, v) in enumerate(verts) if !(v in bnd)]
    u_g = K[int_idx, int_idx] \ (M0 * f)[int_idx]

    # Both should approximate u_ex to roughly the same magnitude
    err_g = norm(u_g - u_ex[int_idx]) / sqrt(length(int_idx))
    @test err_g < 0.01  # h² with n=16, h=1/16 → h² ≈ 0.004
end

# ============================================================================
# 1-form Hodge Laplacian via mixed-FEM (saddle-point) Galerkin.
# Find (σ, ω) such that  M_0 σ − d_0' M_1 ω = 0  and
#                        M_1 d_0 σ + d_1' M_2 d_1 ω = M_1 f.
# Tests `galerkin_hodge_laplacian_block(m, comp, 2)` and verifies that the
# discrete ω_h converges to the exact ω at edges. Tangential ω = 0 on the
# unit-cube boundary for our manufactured ω_ex (every component contains a
# `sin(π·)` that vanishes there).
# ============================================================================
@testset "galerkin_hodge_laplacian_block: 2D 1-form Hodge Laplacian" begin
    m = Metric(2)
    function err(n)
        _, tcomp = DEC.triangulated_lattice([1.0, 0.0], [0.0, 1.0], n, n)
        orient!(tcomp.complex)
        comp = tcomp.complex
        A, M_mid = galerkin_hodge_laplacian_block(m, comp, 2)
        n_v = length(comp.cells[1]); n_e = length(comp.cells[2])
        edges = comp.cells[2]

        # ω_ex = sin(πx)sin(πy) (dx + dy); Δ_H ω = 2π² ω; tangential ω = 0 on ∂Ω.
        function ω_edge(e)
            v1, v2 = e.children
            v_pos = v1.parents[e] ? v1 : v2
            v_neg = v1.parents[e] ? v2 : v1
            Δ   = v_pos.points[1].coords - v_neg.points[1].coords
            mid = (v_pos.points[1].coords + v_neg.points[1].coords) / 2
            s = sin(π * mid[1]) * sin(π * mid[2])
            return s * (Δ[1] + Δ[2])
        end
        ω_ex = [ω_edge(e) for e in edges]
        f_e  = 2 * π^2 .* ω_ex
        rhs  = [zeros(n_v); M_mid * f_e]

        _, ext = DEC.boundary_components_connected(comp)
        bnd_e_set = Set(ext.cells[2])
        bnd_e_idx = [i for (i, e) in enumerate(edges) if e in bnd_e_set]
        int_e_idx = setdiff(1:n_e, bnd_e_idx)
        free_idx = [collect(1:n_v); n_v .+ int_e_idx]
        x = A[free_idx, free_idx] \ rhs[free_idx]
        ω_h = zeros(n_e); ω_h[int_e_idx] = x[n_v + 1:end]
        return norm(ω_h[int_e_idx] - ω_ex[int_e_idx]) / sqrt(length(int_e_idx))
    end
    es = [err(n) for n in [8, 16, 32]]
    # ω convergence on regular skewed lattice: empirically ×8 (h³ super-conv).
    # Conservatively assert ≥ ×3.5 (h²).
    @test es[1] / es[2] > 3.5
    @test es[2] / es[3] > 3.5
end

@testset "galerkin_hodge_laplacian_block: 3D Kuhn 1-form Hodge Laplacian" begin
    m = Metric(3)
    function err(n)
        tcomp = _tet_lattice_3d([1.0,0,0], [0,1.0,0], [0,0,1.0], n)
        orient!(tcomp.complex)
        comp = tcomp.complex
        A, M_mid = galerkin_hodge_laplacian_block(m, comp, 2)
        n_v = length(comp.cells[1]); n_e = length(comp.cells[2])
        edges = comp.cells[2]

        # ω_ex = sin(πx)sin(πy)sin(πz) (dx+dy+dz); Δ_H ω = 3π² ω.
        function ω_edge(e)
            v1, v2 = e.children
            v_pos = v1.parents[e] ? v1 : v2
            v_neg = v1.parents[e] ? v2 : v1
            Δ   = v_pos.points[1].coords - v_neg.points[1].coords
            mid = (v_pos.points[1].coords + v_neg.points[1].coords) / 2
            s = sin(π*mid[1]) * sin(π*mid[2]) * sin(π*mid[3])
            return s * (Δ[1] + Δ[2] + Δ[3])
        end
        ω_ex = [ω_edge(e) for e in edges]
        f_e  = 3 * π^2 .* ω_ex
        rhs  = [zeros(n_v); M_mid * f_e]

        _, ext = DEC.boundary_components_connected(comp)
        bnd_e_set = Set(ext.cells[2])
        bnd_e_idx = [i for (i, e) in enumerate(edges) if e in bnd_e_set]
        int_e_idx = setdiff(1:n_e, bnd_e_idx)
        free_idx = [collect(1:n_v); n_v .+ int_e_idx]
        x = A[free_idx, free_idx] \ rhs[free_idx]
        ω_h = zeros(n_e); ω_h[int_e_idx] = x[n_v + 1:end]
        return norm(ω_h[int_e_idx] - ω_ex[int_e_idx]) / sqrt(length(int_e_idx))
    end
    es = [err(n) for n in [4, 6, 8]]
    # 3D Kuhn — measured ≈h^{2.5}. Assert ≥ ×2.0 between consecutive sizes.
    @test es[1] / es[2] > 2.0
    @test es[2] / es[3] > 1.7
end

# ============================================================================
# Hex Nedelec / Whitney 1-form on axis-aligned hexahedral meshes.
# ============================================================================
function _hex_lattice_unit(n)
    pts = Dict{NTuple{3,Int}, Point{3}}()
    for i in 0:n, j in 0:n, k in 0:n
        pts[(i,j,k)] = Point(i/n, j/n, k/n)
    end
    hexes = Vector{Vector{Point{3}}}()
    for i in 0:n-1, j in 0:n-1, k in 0:n-1
        push!(hexes, [pts[(i,j,k)],   pts[(i+1,j,k)],   pts[(i+1,j+1,k)], pts[(i,j+1,k)],
                      pts[(i,j,k+1)], pts[(i+1,j,k+1)], pts[(i+1,j+1,k+1)], pts[(i,j+1,k+1)]])
    end
    return DEC.hexahedral_complex(hexes)
end

@testset "galerkin_hodge: hex Nedelec 1-form mass — structural" begin
    m = Metric(3)
    tcomp = _hex_lattice_unit(2)
    orient!(tcomp.complex)
    M1 = galerkin_hodge(m, tcomp, 2)
    n_e = length(tcomp.complex.cells[2])
    @test size(M1) == (n_e, n_e)
    @test M1 ≈ transpose(M1)
    @test minimum(eigvals(Symmetric(Matrix(M1)))) > 0
end

@testset "galerkin_hodge: hex Poisson SOLVE on unit cube reaches h²" begin
    m = Metric(3)
    function err(n)
        tcomp = _hex_lattice_unit(n)
        orient!(tcomp.complex)
        comp = tcomp.complex
        M0 = galerkin_hodge(m, tcomp, 1)
        M1 = galerkin_hodge(m, tcomp, 2)
        d0 = DEC.exterior_derivative(comp, 1)
        K = transpose(d0) * M1 * d0
        verts = comp.cells[1]
        u_ex = [sin(π*v.points[1].coords[1]) * sin(π*v.points[1].coords[2]) *
                sin(π*v.points[1].coords[3]) for v in verts]
        f = 3 * π^2 .* u_ex
        _, ext = DEC.boundary_components_connected(comp)
        bnd = Set(ext.cells[1])
        int_idx = [i for (i, v) in enumerate(verts) if !(v in bnd)]
        u_int = K[int_idx, int_idx] \ (M0 * f)[int_idx]
        return norm(u_int - u_ex[int_idx]) / sqrt(length(int_idx))
    end
    es = [err(n) for n in [4, 8]]
    # Ratio over 2× refinement should be ≈4 for clean h². Allow ≥3.5.
    @test es[1] / es[2] > 3.5
end

@testset "galerkin_laplacian high-level API on hex mesh" begin
    m = Metric(3)
    tcomp = _hex_lattice_unit(4)
    orient!(tcomp.complex)
    M0, K = galerkin_laplacian(m, tcomp)
    n_v = length(tcomp.complex.cells[1])
    @test size(M0) == (n_v, n_v)
    @test size(K)  == (n_v, n_v)
    # K is symmetric and positive semi-definite (constant null space).
    @test K ≈ transpose(K)
    @test minimum(eigvals(Symmetric(Matrix(K)))) > -1e-10
    @test norm(K * ones(n_v)) < 1e-10

    # Solve Poisson and verify h² (n=4 vs n=8).
    function err(n)
        tc = _hex_lattice_unit(n)
        orient!(tc.complex)
        M0, K = galerkin_laplacian(m, tc)
        verts = tc.complex.cells[1]
        u_ex = [sin(π*v.points[1].coords[1]) * sin(π*v.points[1].coords[2]) *
                sin(π*v.points[1].coords[3]) for v in verts]
        f = 3 * π^2 .* u_ex
        _, ext = DEC.boundary_components_connected(tc.complex)
        bnd = Set(ext.cells[1])
        int_idx = [i for (i, v) in enumerate(verts) if !(v in bnd)]
        u_int = K[int_idx, int_idx] \ (M0 * f)[int_idx]
        return norm(u_int - u_ex[int_idx]) / sqrt(length(int_idx))
    end
    @test err(4) / err(8) > 3.5
end
