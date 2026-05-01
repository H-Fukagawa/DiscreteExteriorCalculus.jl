using Test, DiscreteExteriorCalculus
const DEC = DiscreteExteriorCalculus
using LinearAlgebra: I, dot, norm
using SparseArrays: sparse, spzeros

# Kuhn (Coxeter–Freudenthal) triangulation of a unit cube into 6 tetrahedra,
# all sharing the diagonal vertex 1 → vertex 7.
const _KUHN_TETS = ((1,2,3,7), (1,3,4,7), (1,4,8,7),
                    (1,8,5,7), (1,5,6,7), (1,6,2,7))

# Build a structured tetrahedral lattice over the parallelepiped spanned by
# v1, v2, v3 with n×n×n boxes. Returns a `TriangulatedComplex{3, 4}`.
function _tet_lattice(v1, v2, v3, n)
    pts = Dict{NTuple{3,Int}, Point{3}}()
    for i in 0:n, j in 0:n, k in 0:n
        c = (i/n) .* v1 .+ (j/n) .* v2 .+ (k/n) .* v3
        pts[(i,j,k)] = Point(c[1], c[2], c[3])
    end
    simplices = Simplex{3, 4}[]
    for i in 0:n-1, j in 0:n-1, k in 0:n-1
        corners = [pts[(i,j,k)],   pts[(i+1,j,k)],   pts[(i+1,j+1,k)], pts[(i,j+1,k)],
                   pts[(i,j,k+1)], pts[(i+1,j,k+1)], pts[(i+1,j+1,k+1)], pts[(i,j+1,k+1)]]
        for tup in _KUHN_TETS
            push!(simplices, Simplex(corners[tup[1]], corners[tup[2]],
                                     corners[tup[3]], corners[tup[4]]))
        end
    end
    return TriangulatedComplex(simplices)
end

@testset "circumcenter_hodge and exterior_derivative" begin
    # setup
    begin
        m = Metric(2)
        n = 10
        _, tcomp = DEC.triangulated_lattice(n * [1,0], n * [.5, .5 * sqrt(3)], n, n)

        comp = tcomp.complex
        @test simplicial(comp)
        @test one_sided(m, comp)
        @test pairwise_delaunay(m, comp)
        @test well_centered(m, comp)
        mesh = Mesh(tcomp, circumcenter(m))
        N, K = 2, 3
        @test typeof(mesh) <: Mesh{N, K}
    end
    # test hodge star
    begin
        ★s = [DEC.circumcenter_hodge(m, mesh, k, true)
            for k in 1:length(mesh.primal.complex.cells)]
        @test map(t -> size(t, 1), ★s) == map(length, mesh.primal.complex.cells)
        @test map(t -> size(t, 2), ★s) == map(length, mesh.primal.complex.cells)
        dual_★s = [DEC.circumcenter_hodge(m, mesh, k, false)
            for k in 1:length(mesh.dual.complex.cells)]
        @test map(t -> size(t, 1), dual_★s) == map(length, mesh.dual.complex.cells)
        @test map(t -> size(t, 2), dual_★s) == map(length, mesh.dual.complex.cells)
        for (m1, m2, s) in zip(★s, reverse(dual_★s), [1,-1,1])
            @test count(!iszero, m1 * m2 - I*s) == 0 # ★★ ∝ I
        end
    end
    # test exterior derivative
    begin
        ds = [DEC.exterior_derivative(mesh.primal.complex, k)
            for k in 1:length(mesh.primal.complex.cells)]
        @test map(t -> size(t, 2), ds) == map(length, mesh.primal.complex.cells)
        @test map(t -> size(t, 1), ds) == [map(length, mesh.primal.complex.cells)[2:end]..., 0]
        @test all([count(!iszero, ds[i+1] * ds[i]) == 0 for i in 1:(length(ds)-1)]) # d² = 0
        dual_ds = [DEC.exterior_derivative(mesh.dual.complex, k)
            for k in 1:length(mesh.dual.complex.cells)]
        @test map(t -> size(t, 2), dual_ds) == map(length, mesh.dual.complex.cells)
        @test map(t -> size(t, 1), dual_ds) == [map(length, mesh.dual.complex.cells)[2:end]..., 0]
        @test all([count(!iszero, dual_ds[i+1] * dual_ds[i]) == 0 for i in 1:(length(dual_ds)-1)]) # d² = 0
        # use children to write an alternative definition of exterior_derivative
        function _exterior_derivative(comp::CellComplex{N, K}, k::Int) where {N, K}
            @assert 1 <= k <= K
            if k == K
                return spzeros(Int, 0, length(comp.cells[k]))
            else
                row_inds, col_inds, vals = Int[], Int[], Int[]
                for (row_ind, cell) in enumerate(comp.cells[k+1])
                    for c in cell.children
                        o = c.parents[cell]
                        col_ind = findfirst(isequal(c), comp.cells[k])
                        push!(row_inds, row_ind); push!(col_inds, col_ind); push!(vals, 2 * o - 1)
                    end
                end
                num_rows = length(comp.cells[k+1])
                num_cols = length(comp.cells[k])
                return sparse(row_inds, col_inds, vals, num_rows, num_cols)
            end
        end
        @test [_exterior_derivative(mesh.primal.complex, k)
            for k in 1:length(mesh.primal.complex.cells)] == ds
        @test [_exterior_derivative(mesh.dual.complex, k)
            for k in 1:length(mesh.dual.complex.cells)] == dual_ds
    end
    # test differential_operator
    begin
        differential_operator_sequence(m, mesh, "★d★d", 1, true) ==
            [dual_★s[3], dual_ds[2],★s[2],ds[1]]
        # 旧コード:
        # @test differential_operator(m, mesh, "★d★d", 1, true) == (dual_★s[3] * dual_ds[2] * ★s[2] * ds[1])

        # 新コード: isapprox で比較する
        A = differential_operator(m, mesh, "★d★d", 1, true)
        B = dual_★s[3] * dual_ds[2] * ★s[2] * ds[1]
        @test isapprox(A, B; rtol=1e-14, atol=1e-14)
        v = ones(length(mesh.primal.complex.cells[1]))
        @test norm(differential_operator(m, mesh, "★d★d", 1, true, v) -
            dual_★s[3] * dual_ds[2] * ★s[2] * ds[1] * v) < 1e-14
        for primal in [true, false]
            for k in 1:2
                @test count(!iszero, differential_operator(m, mesh, "dd", k, primal)) == 0 # d² = 0
            end
            for (k, s) in zip(1:3, [1,-1,1])
                ★★ = differential_operator(m, mesh, "★★", k, primal)
                @test count(!iszero, ★★ - I*s) == 0 # ★★ ∝ I
                ★★ = differential_operator(m, mesh, "★", K-k+1, !primal) *
                    differential_operator(m, mesh, "★", k, primal)
                @test count(!iszero, ★★ - I*s) == 0 # ★★ ∝ I
            end
        end
    end
end

@testset "barycentric_hodge structural tests" begin
    m = Metric(2)
    n = 5
    _, tcomp = DEC.triangulated_lattice(n * [1,0], n * [.5, .5 * sqrt(3)], n, n)
    mesh = Mesh(tcomp, circumcenter(m))
    K = 3

    for k in 1:K
        for primal in [true, false]
            hodge = DEC.barycentric_hodge(m, mesh, k, primal)
            @test isa(hodge, AbstractMatrix)
            @test size(hodge, 1) == size(hodge, 2)
            expected_size = primal ?
                length(mesh.dual.complex.cells[K-k+1]) :
                length(mesh.primal.complex.cells[K-k+1])
            @test size(hodge, 1) == expected_size
        end
    end
    @test DEC.barycentric_hodge(m, mesh, K+1, true) == spzeros(0, 0)
end

@testset "nonorthogonal_hodge: equilateral mesh = barycentric_hodge" begin
    # On equilateral triangulation, centroid coincides with circumcenter, so the
    # over-relaxed correction must reduce to the diagonal Hodge.
    m = Metric(2)
    _, tcomp = DEC.triangulated_lattice([1.0, 0.0], [0.5, 0.5 * sqrt(3)], 6, 6)
    orient!(tcomp.complex)
    mesh = Mesh(tcomp, centroid)

    sN = DEC.nonorthogonal_hodge(m, mesh)
    sD = DEC.barycentric_hodge(m, mesh, 2, true)
    @test isapprox(sN, sD; atol=1e-12)
end

@testset "nonorthogonal_hodge: exact flux for linear u on skewed mesh" begin
    # For linear u, flux through every dual face equals S_e · ∇u exactly.
    m = Metric(2)
    _, tcomp = DEC.triangulated_lattice([1.0, 0.0], [0.3, 0.85], 6, 6)
    orient!(tcomp.complex)
    mesh = Mesh(tcomp, centroid)

    verts = mesh.primal.complex.cells[1]
    edges = mesh.primal.complex.cells[2]
    grad = [2.0, 3.0]
    u_vec = [grad[1] * v.points[1].coords[1] + grad[2] * v.points[1].coords[2]
             for v in verts]
    d0 = DEC.exterior_derivative(mesh.primal.complex, 1)
    omega = d0 * u_vec

    true_flux = [let de = DEC._primal_edge_vector(e),
                     S = DEC._dual_face_area_vector(mesh, e, de)
                     S[1] * grad[1] + S[2] * grad[2]
                 end for e in edges]

    flux_diag = DEC.barycentric_hodge(m, mesh, 2, true) * omega
    flux_nono = DEC.nonorthogonal_hodge(m, mesh) * omega

    # Diagonal Hodge has O(skew) error; corrected should be machine precision.
    @test norm(flux_nono - true_flux) / norm(true_flux) < 1e-12
    @test norm(flux_diag - true_flux) / norm(true_flux) > 0.05
end

@testset "nonorthogonal_hodge: Laplacian consistency on skewed mesh" begin
    # Manufactured solution u = sin(2πx) sin(2πy) on a skewed parallelogram.
    # Δu = -8π² u, so applying the discrete Laplacian to u_vec should give
    # f_ex = -8π² u_vec at interior vertices. The diagonal centroidal Hodge is
    # inconsistent (error stays O(1) under refinement), the over-relaxed
    # correction restores consistency (error decreases with h).
    m = Metric(2)
    function consistency_err(n, hodge_fn)
        _, tcomp = DEC.triangulated_lattice([1.0, 0.0], [0.3, 0.85], n, n)
        orient!(tcomp.complex)
        mesh = Mesh(tcomp, centroid)
        d0 = DEC.exterior_derivative(mesh.primal.complex, 1)
        d1d = DEC.exterior_derivative(mesh.dual.complex, 2)
        sNi = DEC.barycentric_hodge(m, mesh, 3, false)
        L = sNi * d1d * hodge_fn(mesh) * d0

        verts = mesh.primal.complex.cells[1]
        u_vec = [sin(2π * v.points[1].coords[1]) * sin(2π * v.points[1].coords[2])
                 for v in verts]
        f_ex = -8 * π^2 .* u_vec
        _, ext = DEC.boundary_components_connected(mesh.primal.complex)
        bnd = Set(ext.cells[1])
        int_idx = [i for (i, v) in enumerate(verts) if !(v in bnd)]
        return norm((L * u_vec - f_ex)[int_idx]) / sqrt(length(int_idx))
    end

    diag_hodge(mesh) = DEC.barycentric_hodge(m, mesh, 2, true)
    nono_hodge(mesh) = DEC.nonorthogonal_hodge(m, mesh)

    err_diag = [consistency_err(n, diag_hodge) for n in [8, 16]]
    err_nono = [consistency_err(n, nono_hodge) for n in [8, 16]]

    # Diagonal: inconsistent — barely changes under refinement.
    @test err_diag[2] / err_diag[1] > 0.9
    # Over-relaxed: at least 3× reduction when h halves (close to 2nd order).
    @test err_nono[1] / err_nono[2] > 3.0
    # Over-relaxed beats diagonal at every resolution.
    @test err_nono[1] < err_diag[1]
    @test err_nono[2] < err_diag[2]
end

@testset "corrected_barycentric_hodge dispatch (2D)" begin
    m = Metric(2)
    _, tcomp = DEC.triangulated_lattice([1.0, 0.0], [0.3, 0.85], 5, 5)
    orient!(tcomp.complex)
    mesh = Mesh(tcomp, centroid)

    # Dispatches to nonorthogonal_hodge for 2D primal k=2
    @test DEC.corrected_barycentric_hodge(m, mesh, 2, true) ==
        DEC.nonorthogonal_hodge(m, mesh)
    # Falls back to diagonal barycentric_hodge for other (k, primal)
    for (k, primal) in [(1, true), (3, true), (1, false), (2, false), (3, false)]
        @test DEC.corrected_barycentric_hodge(m, mesh, k, primal) ==
            DEC.barycentric_hodge(m, mesh, k, primal)
    end
end

@testset "nonorthogonal_hodge: exact flux for linear u in 3D" begin
    m = Metric(3)
    tcomp = _tet_lattice([1.0, 0.0, 0.0], [0.2, 1.0, 0.0], [0.1, 0.15, 1.0], 4)
    orient!(tcomp.complex)
    mesh = Mesh(tcomp, centroid)

    verts = mesh.primal.complex.cells[1]
    edges = mesh.primal.complex.cells[2]
    grad = [2.0, 3.0, -1.5]
    u_vec = [grad[1] * v.points[1].coords[1] +
             grad[2] * v.points[1].coords[2] +
             grad[3] * v.points[1].coords[3] for v in verts]
    omega = DEC.exterior_derivative(mesh.primal.complex, 1) * u_vec

    true_flux = [let de = DEC._primal_edge_vector(e),
                     S = DEC._dual_face_area_vector(mesh, e, de)
                     dot(S, grad)
                 end for e in edges]

    flux_diag = DEC.barycentric_hodge(m, mesh, 2, true) * omega
    flux_nono = DEC.nonorthogonal_hodge(m, mesh) * omega

    @test norm(flux_nono - true_flux) / norm(true_flux) < 1e-12
    @test norm(flux_diag - true_flux) / norm(true_flux) > 0.05
end

@testset "nonorthogonal_hodge: 3D Laplacian consistency on skewed tet mesh" begin
    # u = sin(πx) sin(πy) sin(πz) on a skewed parallelepiped lattice.
    # Δu = -3π² u. The diagonal Hodge stays inconsistent under refinement,
    # the over-relaxed correction converges (≥1.5× reduction n=4 → n=8).
    m = Metric(3)
    function err_3d(n, hodge_fn)
        tcomp = _tet_lattice([1.0, 0.0, 0.0], [0.2, 1.0, 0.0],
                             [0.1, 0.15, 1.0], n)
        orient!(tcomp.complex)
        mesh = Mesh(tcomp, centroid)
        d0 = DEC.exterior_derivative(mesh.primal.complex, 1)
        d1d = DEC.exterior_derivative(mesh.dual.complex, 3)
        sNi = DEC.barycentric_hodge(m, mesh, 4, false)
        L = sNi * d1d * hodge_fn(mesh) * d0

        verts = mesh.primal.complex.cells[1]
        u_vec = [sin(π * v.points[1].coords[1]) *
                 sin(π * v.points[1].coords[2]) *
                 sin(π * v.points[1].coords[3]) for v in verts]
        f_ex = -3 * π^2 .* u_vec
        _, ext = DEC.boundary_components_connected(mesh.primal.complex)
        bnd = Set(ext.cells[1])
        int_idx = [i for (i, v) in enumerate(verts) if !(v in bnd)]
        return norm((L * u_vec - f_ex)[int_idx]) / sqrt(length(int_idx))
    end

    diag_h(mesh) = DEC.barycentric_hodge(m, mesh, 2, true)
    nono_h(mesh) = DEC.nonorthogonal_hodge(m, mesh)

    err_diag = [err_3d(n, diag_h) for n in [4, 8]]
    err_nono = [err_3d(n, nono_h) for n in [4, 8]]

    # Diagonal: inconsistent — error does not shrink.
    @test err_diag[2] / err_diag[1] > 0.9
    # Over-relaxed converges; require ≥1.5× reduction.
    @test err_nono[1] / err_nono[2] > 1.5
    @test err_nono[1] < err_diag[1]
    @test err_nono[2] < err_diag[2]
end

@testset "corrected_barycentric_hodge dispatch (3D)" begin
    m = Metric(3)
    tcomp = _tet_lattice([1.0, 0.0, 0.0], [0.2, 1.0, 0.0], [0.1, 0.15, 1.0], 3)
    orient!(tcomp.complex)
    mesh = Mesh(tcomp, centroid)

    @test DEC.corrected_barycentric_hodge(m, mesh, 2, true) ==
        DEC.nonorthogonal_hodge(m, mesh)
    for (k, primal) in [(1, true), (3, true), (4, true),
                        (1, false), (2, false), (3, false), (4, false)]
        @test DEC.corrected_barycentric_hodge(m, mesh, k, primal) ==
            DEC.barycentric_hodge(m, mesh, k, primal)
    end
end
