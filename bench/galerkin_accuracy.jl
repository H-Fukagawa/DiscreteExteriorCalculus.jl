#!/usr/bin/env julia

using DiscreteExteriorCalculus
const DEC = DiscreteExteriorCalculus

using LinearAlgebra: norm
using Printf

const KUHN_TETS = ((1,2,3,7), (1,3,4,7), (1,4,8,7),
                   (1,8,5,7), (1,5,6,7), (1,6,2,7))

function tet_lattice_3d(v1, v2, v3, n)
    pts = Dict{NTuple{3,Int}, Point{3}}()
    for i in 0:n, j in 0:n, k in 0:n
        c = (i/n) .* v1 .+ (j/n) .* v2 .+ (k/n) .* v3
        pts[(i,j,k)] = Point(c[1], c[2], c[3])
    end
    simplices = Simplex{3, 4}[]
    for i in 0:n-1, j in 0:n-1, k in 0:n-1
        c8 = [pts[(i,j,k)],   pts[(i+1,j,k)],   pts[(i+1,j+1,k)], pts[(i,j+1,k)],
              pts[(i,j,k+1)], pts[(i+1,j,k+1)], pts[(i+1,j+1,k+1)], pts[(i,j+1,k+1)]]
        for tup in KUHN_TETS
            push!(simplices, Simplex(c8[tup[1]], c8[tup[2]], c8[tup[3]], c8[tup[4]]))
        end
    end
    return TriangulatedComplex(simplices)
end

function interior_vertex_indices(comp)
    _, ext = DEC.boundary_components_connected(comp)
    bnd = Set(ext.cells[1])
    return [i for (i, v) in enumerate(comp.cells[1]) if !(v in bnd)]
end

function solve_error(comp_or_tcomp, u_exact, f_exact; quad_order=4)
    comp = comp_or_tcomp isa TriangulatedComplex ? comp_or_tcomp.complex : comp_or_tcomp
    verts = comp.cells[1]
    u_nodes = [u_exact(v.points[1]) for v in verts]
    f_nodes = [f_exact(v.points[1]) for v in verts]
    int_idx = interior_vertex_indices(comp)

    m = Metric(length(verts[1].points[1].coords))
    M0, K = galerkin_laplacian(m, comp_or_tcomp)
    Kii = K[int_idx, int_idx]

    rhs_consistent = (M0 * f_nodes)[int_idx]
    rhs_lumped = (galerkin_lumped_mass(M0) * f_nodes)[int_idx]
    rhs_blend_025 = (galerkin_blended_mass(M0, 0.25) * f_nodes)[int_idx]
    rhs_blend_050 = (galerkin_blended_mass(M0, 0.50) * f_nodes)[int_idx]
    rhs_blend_075 = (galerkin_blended_mass(M0, 0.75) * f_nodes)[int_idx]
    rhs_exact = galerkin_load_vector(m, comp_or_tcomp, f_exact; quad_order=quad_order)[int_idx]

    err(rhs) = norm(Kii \ rhs - u_nodes[int_idx]) / sqrt(length(int_idx))
    return (
        consistent = err(rhs_consistent),
        lumped = err(rhs_lumped),
        blend_025 = err(rhs_blend_025),
        blend_050 = err(rhs_blend_050),
        blend_075 = err(rhs_blend_075),
        exact_load = err(rhs_exact),
    )
end

function print_table(title, rows)
    println()
    println(title)
    println("| n | consistent | lumped | blend .25 | blend .50 | blend .75 | exact load |")
    println("|---|------------|--------|-----------|-----------|-----------|------------|")
    for (n, r) in rows
        @printf("| %d | %.6e | %.6e | %.6e | %.6e | %.6e | %.6e |\n",
            n, r.consistent, r.lumped, r.blend_025, r.blend_050, r.blend_075,
            r.exact_load)
    end
end

function run_2d_square(; ns=(8, 16, 32), quad_order=4)
    rows = []
    for n in ns
        _, tcomp = DEC.triangulated_lattice([1.0, 0.0], [0.0, 1.0], n, n)
        orient!(tcomp.complex)
        u(p) = sin(pi * p.coords[1]) * sin(pi * p.coords[2])
        f(p) = 2 * pi^2 * u(p)
        push!(rows, (n, solve_error(tcomp.complex, u, f; quad_order=quad_order)))
    end
    print_table("2D unit-square P1 Poisson", rows)
end

function run_3d_kuhn(; ns=(4, 6, 8), quad_order=4)
    rows = []
    for n in ns
        tcomp = tet_lattice_3d([1.0,0,0], [0,1.0,0], [0,0,1.0], n)
        orient!(tcomp.complex)
        u(p) = sin(pi * p.coords[1]) * sin(pi * p.coords[2]) * sin(pi * p.coords[3])
        f(p) = 3 * pi^2 * u(p)
        push!(rows, (n, solve_error(tcomp.complex, u, f; quad_order=quad_order)))
    end
    print_table("3D Kuhn-tet unit-cube P1 Poisson", rows)
end

quad_order = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 4
run_2d_square(; quad_order=quad_order)
run_3d_kuhn(; quad_order=quad_order)
