using Test, DiscreteExteriorCalculus
const DEC = DiscreteExteriorCalculus
using LinearAlgebra: I, norm
using SparseArrays: sparse, spzeros

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
