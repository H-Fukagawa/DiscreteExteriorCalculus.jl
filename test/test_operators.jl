using Test, DiscreteExteriorCalculus
const DEC = DiscreteExteriorCalculus
using LinearAlgebra: I, dot, norm
using SparseArrays: sparse, spzeros

# Kuhn (Coxeter–Freudenthal) triangulation of a unit cube into 6 tetrahedra,
# all sharing the diagonal vertex 1 → vertex 7.
const _KUHN_TETS = ((1,2,3,7), (1,3,4,7), (1,4,8,7),
                    (1,8,5,7), (1,5,6,7), (1,6,2,7))

# Cube vertex layout used by all polytope-lattice helpers below:
#   1=(0,0,0), 2=(1,0,0), 3=(1,1,0), 4=(0,1,0),
#   5=(0,0,1), 6=(1,0,1), 7=(1,1,1), 8=(0,1,1)
function _cube_corners(pts, i, j, k)
    return [pts[(i,j,k)],   pts[(i+1,j,k)],   pts[(i+1,j+1,k)], pts[(i,j+1,k)],
            pts[(i,j,k+1)], pts[(i+1,j,k+1)], pts[(i+1,j+1,k+1)], pts[(i,j+1,k+1)]]
end

function _lattice_points(v1, v2, v3, n)
    pts = Dict{NTuple{3,Int}, Point{3}}()
    for i in 0:n, j in 0:n, k in 0:n
        c = (i/n) .* v1 .+ (j/n) .* v2 .+ (k/n) .* v3
        pts[(i,j,k)] = Point(c[1], c[2], c[3])
    end
    return pts
end

# Tet lattice: each cube → 6 Kuhn tets sharing the (1,7) diagonal.
function _tet_lattice(v1, v2, v3, n)
    pts = _lattice_points(v1, v2, v3, n)
    simplices = Simplex{3, 4}[]
    for i in 0:n-1, j in 0:n-1, k in 0:n-1
        c8 = _cube_corners(pts, i, j, k)
        for tup in _KUHN_TETS
            push!(simplices, Simplex(c8[tup[1]], c8[tup[2]], c8[tup[3]], c8[tup[4]]))
        end
    end
    return TriangulatedComplex(simplices)
end

# Hex lattice: each cube → 1 hexahedral cell.
function _hex_lattice(v1, v2, v3, n)
    pts = _lattice_points(v1, v2, v3, n)
    hexes = Vector{Vector{Point{3}}}()
    for i in 0:n-1, j in 0:n-1, k in 0:n-1
        push!(hexes, _cube_corners(pts, i, j, k))
    end
    return DEC.hexahedral_complex(hexes)
end

# Prism lattice: each cube → 2 triangular prisms split along the bottom (1,3) diagonal.
function _prism_lattice(v1, v2, v3, n)
    pts = _lattice_points(v1, v2, v3, n)
    prisms = Vector{Vector{Point{3}}}()
    for i in 0:n-1, j in 0:n-1, k in 0:n-1
        c8 = _cube_corners(pts, i, j, k)
        push!(prisms, [c8[1], c8[2], c8[4], c8[5], c8[6], c8[8]])
        push!(prisms, [c8[2], c8[3], c8[4], c8[6], c8[7], c8[8]])
    end
    return DEC.prismatic_complex(prisms)
end

# Pyramid lattice: each cube → 6 pyramids whose apex is the cube center.
function _pyramid_lattice(v1, v2, v3, n)
    pts = _lattice_points(v1, v2, v3, n)
    pyramids = Vector{Vector{Point{3}}}()
    for i in 0:n-1, j in 0:n-1, k in 0:n-1
        c8 = _cube_corners(pts, i, j, k)
        cc = (i + 0.5)/n .* v1 .+ (j + 0.5)/n .* v2 .+ (k + 0.5)/n .* v3
        ctr = Point(cc[1], cc[2], cc[3])
        # Faces follow _PYRAMID_FACES convention: first 4 are square base, 5th is apex.
        push!(pyramids, [c8[1], c8[4], c8[3], c8[2], ctr])  # bottom -z
        push!(pyramids, [c8[5], c8[6], c8[7], c8[8], ctr])  # top +z
        push!(pyramids, [c8[1], c8[2], c8[6], c8[5], ctr])  # front -y
        push!(pyramids, [c8[2], c8[3], c8[7], c8[6], ctr])  # right +x
        push!(pyramids, [c8[3], c8[4], c8[8], c8[7], ctr])  # back +y
        push!(pyramids, [c8[4], c8[1], c8[5], c8[8], ctr])  # left -x
    end
    return DEC.pyramidal_complex(pyramids)
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
    # Over-relaxed with Diamond scheme: clean h² convergence (≥3.5× reduction).
    @test err_nono[1] / err_nono[2] > 3.5
    # Over-relaxed beats diagonal at every resolution.
    @test err_nono[1] < err_diag[1]
    @test err_nono[2] < err_diag[2]
end

@testset "nonorthogonal_hodge: 2D 1-form Laplacian convergence" begin
    # ω = du for u = sin(πx)sin(πy). Then Δω = d(Δu) = -2π² ω, and δdω = 0
    # discretely (via d² = 0), so the full 1-form Laplacian equals dδω alone.
    # In 2D K=3, dδω is composed as
    #     dδω = d_1 · ★_3_dual · d_dual_2 · ★_2_primal · ω
    # where ★_2_primal is the only non-trivial Hodge (★_3_dual = 1/CV_volume,
    # ★_1 = vertex CV volume — both diagonal). With the corrected ★_2 the
    # operator is exact for linear u and shows fast convergence; the diagonal
    # ★_2 leaves an O(h) per-edge error.
    m = Metric(2)
    function dδ_err(n, hodge_fn)
        _, tcomp = DEC.triangulated_lattice([1.0, 0.0], [0.3, 0.85], n, n)
        orient!(tcomp.complex)
        mesh = Mesh(tcomp, centroid)
        primal = mesh.primal.complex
        s2p = hodge_fn(mesh)
        s3d = DEC.barycentric_hodge(m, mesh, 3, false)
        d1 = DEC.exterior_derivative(primal, 1)
        dd2 = DEC.exterior_derivative(mesh.dual.complex, 2)
        L = d1 * s3d * dd2 * s2p

        verts = primal.cells[1]
        edges = primal.cells[2]
        u_vec = [sin(π * v.points[1].coords[1]) * sin(π * v.points[1].coords[2])
                 for v in verts]
        ω = d1 * u_vec
        f_ex = -2 * π^2 .* ω
        _, ext = DEC.boundary_components_connected(primal)
        bnd = Set(ext.cells[1])
        int_e = [i for (i, e) in enumerate(edges)
                 if all(!(c in bnd) for c in e.children)]
        return norm((L * ω - f_ex)[int_e]) / sqrt(length(int_e))
    end

    diag_h(mesh) = DEC.barycentric_hodge(m, mesh, 2, true)
    nono_h(mesh) = DEC.nonorthogonal_hodge(m, mesh)

    err_diag = [dδ_err(n, diag_h) for n in [8, 16]]
    err_nono = [dδ_err(n, nono_h) for n in [8, 16]]

    # Diagonal: only h¹ from the d_1 wrap (1.87 measured).
    @test 1.5 < err_diag[1] / err_diag[2] < 2.5
    # Corrected: roughly h³ from cancellation in d_1 ∘ L_0form (≥5× per halving).
    @test err_nono[1] / err_nono[2] > 5.0
    # Corrected always wins.
    @test err_nono[1] < 0.2 * err_diag[1]
    @test err_nono[2] < 0.1 * err_diag[2]
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

@testset "nonorthogonal_hodge: convergence across polytope types" begin
    # Compare 0-form Laplacian convergence on hex / prism / pyramid / tet
    # lattices, on the unit cube and on a skewed parallelepiped. Records the
    # different sensitivities of each cell type to mesh skew.
    #
    # Hex and prism are symmetric enough that their cell-LS gradients give
    # clean h² regardless of skew; tet (Kuhn 6-tet decomposition) loses
    # that — the Kuhn diagonal-sharing structure is asymmetric and combines
    # with skew to produce h^≈1.5; pyramid sits between (apex breaks
    # base-rotational symmetry but each cell has more edges than a tet).
    function err_polytope(lattice_fn, n, v1, v2, v3)
        mtr = Metric(3)
        tcomp = lattice_fn(v1, v2, v3, n)
        orient!(tcomp.complex)
        mesh = Mesh(tcomp, centroid)
        d0 = DEC.exterior_derivative(mesh.primal.complex, 1)
        d1d = DEC.exterior_derivative(mesh.dual.complex, 3)
        sNi = DEC.barycentric_hodge(mtr, mesh, 4, false)
        L = sNi * d1d * DEC.nonorthogonal_hodge(mtr, mesh) * d0
        verts = mesh.primal.complex.cells[1]
        u = [sin(π * v.points[1].coords[1]) *
             sin(π * v.points[1].coords[2]) *
             sin(π * v.points[1].coords[3]) for v in verts]
        f_ex = -3 * π^2 .* u
        _, ext = DEC.boundary_components_connected(mesh.primal.complex)
        bnd = Set(ext.cells[1])
        int_idx = [i for (i, v) in enumerate(verts) if !(v in bnd)]
        return norm((L * u - f_ex)[int_idx]) / sqrt(length(int_idx))
    end

    function refinement_ratio(lattice_fn, v1, v2, v3)
        # n=4 → n=8 ratio; for clean h² this is 4.0
        e4 = err_polytope(lattice_fn, 4, v1, v2, v3)
        e8 = err_polytope(lattice_fn, 8, v1, v2, v3)
        return e4 / e8
    end

    @testset "unit cube — all polytope types ≥ near-h²" begin
        v_unit = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
        # All polytope types reach ≥3.5× (≈ h²) on the regular cube grid.
        @test refinement_ratio(_tet_lattice,     v_unit...) > 3.5
        @test refinement_ratio(_hex_lattice,     v_unit...) > 3.5
        @test refinement_ratio(_prism_lattice,   v_unit...) > 3.5
        @test refinement_ratio(_pyramid_lattice, v_unit...) > 3.5
    end

    @testset "skewed parallelepiped — hex/prism robust, tet/pyramid degrade" begin
        v_skew = ([1.0, 0.0, 0.0], [0.2, 1.0, 0.0], [0.1, 0.15, 1.0])
        # Hex and prism keep clean h² on skewed meshes.
        @test refinement_ratio(_hex_lattice,   v_skew...) > 3.5
        @test refinement_ratio(_prism_lattice, v_skew...) > 3.5
        # Tet (Kuhn) drops to ≈h^1.5 — Kuhn diagonal-sharing asymmetry shows.
        r_tet = refinement_ratio(_tet_lattice, v_skew...)
        @test r_tet > 2.0   # well above h¹ (which would be ≈2.0)
        @test r_tet < 3.5   # not yet h² either; document the gap
        # Pyramid sits between tet and hex: apex breaks symmetry but more
        # edges per cell than a tet.
        r_pyr = refinement_ratio(_pyramid_lattice, v_skew...)
        @test r_pyr > 2.5
    end
end
