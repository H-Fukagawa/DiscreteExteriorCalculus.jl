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
