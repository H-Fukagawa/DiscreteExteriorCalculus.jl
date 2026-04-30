using Test, DiscreteExteriorCalculus
const DEC = DiscreteExteriorCalculus
using LinearAlgebra: norm

@testset "Mesh obtuse triangle" begin
    m = Metric(2)
    s = Simplex(Point(0,0), Point(2,0), Point(1,.5))
    for center in [centroid, circumcenter(m)]
        primal = TriangulatedComplex([s])
        mesh = Mesh(primal, center)
        @test map(length, mesh.primal.complex.cells) ==
            reverse(map(length, mesh.dual.complex.cells))
        cell_1 = mesh.dual.complex.cells[1][1]
        @test cell_1.K == 1
        @test cell_1.points[1] == Point(center(s))
        @test length(cell_1.parents) == 3
        @test length(cell_1.children) == 0
        cells_2 = mesh.dual.complex.cells[2]
        @test all([length(c.parents) == 2 for c in cells_2])
        @test all([length(c.children) == 1 for c in cells_2])
        cells_3 = mesh.dual.complex.cells[3]
        @test all([length(c.parents) == 0 for c in cells_3])
        @test all([length(c.children) == 2 for c in cells_3])
    end
end

@testset "signed_volume" begin
    points = [Point(0, 0), Point(2, 0), Point(1, .5), Point(1, -5)]
    obtuse = Simplex(points[1:3])
    acute = Simplex(points[1], points[2], points[4])
    m = Metric(2)
    b, _ = circumsphere_barycentric(m, obtuse)
    @test any(b.coords .< 0) # obtuse
    b, _ = circumsphere_barycentric(m, acute)
    @test !any(b.coords .< 0) # acute
    tcomp = TriangulatedComplex([acute, obtuse])
    center = circumcenter(m)
    mesh = Mesh(tcomp, center)
    # check that the length of the edge between the cells is correct
    edge = subcomplex(mesh.primal.complex, [Simplex(points[1:2])]).cells[2][1]
    dual_edge = dual(mesh, edge)
    ccs = [Point(center(x)) for x in [Simplex(edge), obtuse, acute]]
    d1 = norm(m, ccs[1].coords - ccs[2].coords)
    d2 = norm(m, ccs[1].coords - ccs[3].coords)
    @test abs(volume(m, mesh.dual, dual(mesh, edge))) == abs(d1 - d2)
    # check that the choice of sign was correct
    simplices, bools = zip(mesh.dual.simplices[dual_edge]...)
    @test bools[1] != bools[2]
    @test Set(simplices[bools[1] ? 1 : 2].points) == Set([ccs[1], ccs[3]])
    @test Set(simplices[bools[1] ? 2 : 1].points) == Set([ccs[1], ccs[2]])
end

@testset "honeycomb mesh" begin
    n = 5
    _, tcomp = DEC.triangulated_lattice(n * [1,0], n * [.5, .5 * sqrt(3)], n, n)
    m = Metric(2)
    mesh = Mesh(tcomp, centroid)
    for c in mesh.primal.complex.cells[1]
        @test volume(m, mesh.primal, c) == 1
    end
    for c in mesh.primal.complex.cells[2]
        @test volume(m, mesh.primal, c) ≈ 1
    end
    for c in mesh.primal.complex.cells[3]
        @test volume(m, mesh.primal, c) ≈ sqrt(3)/4
    end
    for c in mesh.dual.complex.cells[1]
        @test volume(m, mesh.dual, c) == 1
    end
    for c in mesh.dual.complex.cells[2]
        @test volume(m, mesh.dual, c) ≈ sqrt(3)/6 * length(c.children)
    end
    for c in mesh.dual.complex.cells[3]
        k = length(c.children)
        @test volume(m, mesh.dual, c) ≈ (sqrt(3)/12) * (k == 6 ? k : k-1)
    end
end

@testset "polygonal 2D non-simplex mesh" begin
    m = Metric(2)
    quad_points = [
        Point(0, 0),
        Point(1, 0),
        Point(1, 1),
        Point(0, 1),
    ]

    quad_tcomp = quadrilateral_complex(quad_points)
    @test map(length, quad_tcomp.complex.cells) == [4, 4, 1]
    @test length(quad_tcomp.complex.cells[3][1].points) == 4
    @test length(quad_tcomp.simplices[quad_tcomp.complex.cells[3][1]]) == 2
    @test volume(m, quad_tcomp, quad_tcomp.complex.cells[3][1]) ≈ 1.0

    quad_mesh = Mesh(quad_tcomp, centroid)
    @test map(length, quad_mesh.dual.complex.cells) == [1, 4, 4]

    circum_mesh = Mesh(quad_tcomp, circumcenter(m))
    quad_center = circum_mesh.dual.complex.cells[1][1].points[1]
    @test quad_center.coords ≈ [0.5, 0.5]

    hex_points = [
        Point(1, 0),
        Point(2, 0),
        Point(3, 1),
        Point(2, 2),
        Point(1, 2),
        Point(0, 1),
    ]

    hex_tcomp = hexagonal_complex(hex_points)
    @test map(length, hex_tcomp.complex.cells) == [6, 6, 1]
    @test length(hex_tcomp.simplices[hex_tcomp.complex.cells[3][1]]) == 4
    @test volume(m, hex_tcomp, hex_tcomp.complex.cells[3][1]) ≈ 4.0

    hex_mesh = Mesh(hex_tcomp, centroid)
    @test map(length, hex_mesh.dual.complex.cells) == [1, 6, 6]
end

@testset "shared quadrilateral edge orientation" begin
    points = [
        Point(0, 0), Point(1, 0), Point(2, 0),
        Point(0, 1), Point(1, 1), Point(2, 1),
    ]
    quads = [
        [1, 2, 5, 4],
        [2, 3, 6, 5],
    ]

    tcomp = quadrilateral_complex(points, quads)
    @test map(length, tcomp.complex.cells) == [6, 7, 2]
    @test sum(volume(Metric(2), tcomp, c) for c in tcomp.complex.cells[3]) ≈ 2.0

    shared_edges = filter(c -> length(c.parents) == 2, tcomp.complex.cells[2])
    @test length(shared_edges) == 1
    @test Set(values(shared_edges[1].parents)) == Set([true, false])

    mesh = Mesh(tcomp, centroid)
    @test map(length, mesh.dual.complex.cells) == [2, 7, 6]
end

@testset "polyhedral hex and prism mesh" begin
    m = Metric(3)
    hex_points = [
        Point(0, 0, 0),
        Point(1, 0, 0),
        Point(1, 1, 0),
        Point(0, 1, 0),
        Point(0, 0, 1),
        Point(1, 0, 1),
        Point(1, 1, 1),
        Point(0, 1, 1),
    ]

    hex_tcomp = hexahedral_complex(hex_points)
    @test map(length, hex_tcomp.complex.cells) == [8, 12, 6, 1]
    @test all(length(c.points) == 4 for c in hex_tcomp.complex.cells[3])
    @test length(hex_tcomp.complex.cells[4][1].points) == 8
    @test volume(m, hex_tcomp, hex_tcomp.complex.cells[4][1]) ≈ 1.0

    hex_mesh = Mesh(hex_tcomp, centroid)
    @test map(length, hex_mesh.dual.complex.cells) == [1, 6, 12, 8]

    circum_mesh = Mesh(hex_tcomp, circumcenter(m))
    hex_center = circum_mesh.dual.complex.cells[1][1].points[1]
    @test hex_center.coords ≈ [0.5, 0.5, 0.5]

    prism_points = [
        Point(0, 0, 0),
        Point(1, 0, 0),
        Point(0, 1, 0),
        Point(0, 0, 1),
        Point(1, 0, 1),
        Point(0, 1, 1),
    ]

    prism_tcomp = prismatic_complex(prism_points)
    @test map(length, prism_tcomp.complex.cells) == [6, 9, 5, 1]
    @test count(c -> length(c.points) == 3, prism_tcomp.complex.cells[3]) == 2
    @test count(c -> length(c.points) == 4, prism_tcomp.complex.cells[3]) == 3
    @test length(prism_tcomp.complex.cells[4][1].points) == 6
    @test volume(m, prism_tcomp, prism_tcomp.complex.cells[4][1]) ≈ 0.5

    prism_mesh = Mesh(prism_tcomp, centroid)
    @test map(length, prism_mesh.dual.complex.cells) == [1, 5, 9, 6]
end

@testset "shared hexahedron face orientation" begin
    points = [
        Point(0, 0, 0), Point(1, 0, 0), Point(2, 0, 0),
        Point(0, 1, 0), Point(1, 1, 0), Point(2, 1, 0),
        Point(0, 0, 1), Point(1, 0, 1), Point(2, 0, 1),
        Point(0, 1, 1), Point(1, 1, 1), Point(2, 1, 1),
    ]
    hexes = [
        [1, 2, 5, 4, 7, 8, 11, 10],
        [2, 3, 6, 5, 8, 9, 12, 11],
    ]

    tcomp = hexahedral_complex(points, hexes)
    @test map(length, tcomp.complex.cells) == [12, 20, 11, 2]
    @test sum(volume(Metric(3), tcomp, c) for c in tcomp.complex.cells[4]) ≈ 2.0

    shared_faces = filter(c -> length(c.parents) == 2, tcomp.complex.cells[3])
    @test length(shared_faces) == 1
    @test Set(values(shared_faces[1].parents)) == Set([true, false])
end
