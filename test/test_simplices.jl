using Test, DiscreteExteriorCalculus
const DEC = DiscreteExteriorCalculus
using LinearAlgebra: norm, diag
using Statistics: mean

@testset "Point and Simplex" begin
    p = Point(1,2,3)
    @test typeof(p) == Point{3}
    s = Simplex(Point(1,2,3), Point(4,5,6), Point(1,4,7))
    @test typeof(s) == Simplex{3, 3}
end

@testset "Barycentric and barycentric_projection" begin
    m = Metric(3)
    s = Simplex(Point(0,0,0), Point(2,0,0), Point(1,.5,0))
    for p in [s.points; Point(3,5,7); Point(-12,25,-100); Point(24.3,-3.5,-2.2)]
        b = Barycentric(m, s, p)
        @test norm(Point(b).coords - [p.coords[1:end-1]; 0]) < 1e-14
    end
end

@testset "distance" begin
    m = Metric(3)
    s = Simplex(Point(0,0,0), Point(2,0,0), Point(1.3,.7,0), Point(0,0,1))
    ps = [s.points; Point(3,5,7); Point(-12,25,-100); Point(24.3,-3.5,-2.2)]
    for p1 in ps
        for p2 in ps
            d1 = sqrt(abs(distance_square(m, Barycentric(m, s, p1), p2)))
            d2 = norm(m, p1.coords - p2.coords)
            d3 = norm(p1.coords - p2.coords) # using Euclidean metric
            @test d2 ≈ d3
            @test abs(d1 - d2) < 2e-5
        end
    end
end

@testset "circumsphere" begin
    function is_circumcenter(m::Metric{N}, s::Simplex{N, K}, p::Point{N}) where {N, K}
        if K == 1
            return s.points[1] == p
        else
            dists = [norm(m, p.coords - q.coords) for q in s.points]
            return all([d ≈ dists[1] for d in dists])
        end
    end
    function circumradius_square(m::Metric{N}, s::Simplex{N}) where N
        center, _ = circumsphere(m, s)
        return mean([norm_square(m, center.coords - v.coords) for v in s.points])
    end
    m = Metric(3)
    s = Simplex(Point(1,2,3), Point(-12,3,5), Point(8,3,4), Point(9,25,-2))
    for k in 1:4
        for sub_s in subsimplices(s, k)
            center, radius_square = circumsphere(m, sub_s)
            @test is_circumcenter(m, sub_s, center)
            @test radius_square ≈ circumradius_square(m, sub_s)
        end
    end
end

@testset "barycentric_subspace" begin
    m = Metric(2)
    s = Simplex(Point(0,0), Point(1,0), Point(0,1))
    f = Simplex(s.points[1:2]...)
    p = Point(.5, 3)
    M, v = DEC.barycentric_subspace(m, s, f, p)
    @test M * Barycentric(m, s, p).coords == v
    @test M * [.5, .5, 0] == v
    @test M * Barycentric(m, s, Point(.5, -2)).coords == v
end

@testset "rotate_about_face" begin
    m = Metric(3)
    s = Simplex(Point(-1,0,0), Point(0,0,0), Point(0,1,0))
    r, θ = 3, π/6
    p = Point(r * cos(θ), 0, r * sin(θ))
    b = DEC.rotate_about_face(m, s, p, 1)
    @test norm(Point(b).coords - [r, 0, 0]) < 1e-15

    m = Metric(2)
    s = Simplex(Point(0,0),Point(1,0))
    p = Point(1,1)
    b = DEC.rotate_about_face(m, s, p, 2)
    @test Point(b).coords ≈ [-sqrt(2), 0]
    b = DEC.rotate_about_face(m, s, p, 1)
    @test Point(b).coords ≈ [2, 0]
end

@testset "pairwise_delaunay" begin
    m = Metric(3)
    s1 = Simplex(Point(-.1,.5,0), Point(0,0,0), Point(0,1,0))
    center, _ = circumsphere(m, s1)
    θ = π/6
    for r in [.1, 1, 5, 10]
        p = Point(r * cos(θ), 0, r * sin(θ))
        proj_p = Point(r, 0, 0)
        # using Euclidean metric
        delaunay = norm(center.coords - proj_p.coords)^2 - norm(center.coords - s1.points[1].coords)^2
        s2 = Simplex(s1.points[2:end]..., p)
        @test pairwise_delaunay(m, s1, s2) ≈ delaunay
    end
end

@testset "mixed_center: per-type circumcenter / centroid dispatch" begin
    # On a well-centered triangulation, mixed_center reproduces the
    # circumcenter dual exactly. On Kuhn 3D tets (every tet obtuse), it
    # falls back to centroid for every cell, matching the centroid dual.
    m2 = Metric(2)

    # Equilateral 2D lattice: well-centered.
    _, tcomp_eq = DEC.triangulated_lattice([1.0, 0.0], [0.5, 0.5*sqrt(3)], 4, 4)
    orient!(tcomp_eq.complex)
    @test well_centered(m2, tcomp_eq.complex)
    mesh_circ = Mesh(tcomp_eq, circumcenter(m2))
    mesh_mix  = Mesh(tcomp_eq, mixed_center(m2))
    @test isapprox(diag(DEC.circumcenter_hodge(m2, mesh_circ, 2, true)),
                   diag(DEC.barycentric_hodge(m2, mesh_mix, 2, true));
                   atol=1e-12)

    # Skewed 2D lattice: still well-centered, mixed_center == circumcenter.
    _, tcomp_sk = DEC.triangulated_lattice([1.0, 0.0], [0.3, 0.85], 4, 4)
    orient!(tcomp_sk.complex)
    @test well_centered(m2, tcomp_sk.complex)
    mesh_circ_sk = Mesh(tcomp_sk, circumcenter(m2))
    mesh_mix_sk  = Mesh(tcomp_sk, mixed_center(m2))
    @test isapprox(diag(DEC.circumcenter_hodge(m2, mesh_circ_sk, 2, true)),
                   diag(DEC.barycentric_hodge(m2, mesh_mix_sk, 2, true));
                   atol=1e-12)

    # 3D Kuhn tets (every tet obtuse): mixed_center falls back to centroid.
    m3 = Metric(3)
    KUHN = ((1,2,3,7),(1,3,4,7),(1,4,8,7),(1,8,5,7),(1,5,6,7),(1,6,2,7))
    function _kuhn_tcomp(n)
        pts = Dict{NTuple{3,Int}, Point{3}}()
        for i in 0:n, j in 0:n, k in 0:n
            pts[(i,j,k)] = Point(i/n, j/n, k/n)
        end
        simplices = Simplex{3,4}[]
        for i in 0:n-1, j in 0:n-1, k in 0:n-1
            c8 = [pts[(i,j,k)], pts[(i+1,j,k)], pts[(i+1,j+1,k)], pts[(i,j+1,k)],
                  pts[(i,j,k+1)], pts[(i+1,j,k+1)], pts[(i+1,j+1,k+1)], pts[(i,j+1,k+1)]]
            for t in KUHN
                push!(simplices, Simplex(c8[t[1]], c8[t[2]], c8[t[3]], c8[t[4]]))
            end
        end
        return TriangulatedComplex(simplices)
    end
    tcomp_kuhn = _kuhn_tcomp(3)
    orient!(tcomp_kuhn.complex)
    @test !well_centered(m3, tcomp_kuhn.complex)
    mesh_cent  = Mesh(tcomp_kuhn, centroid)
    mesh_mix3  = Mesh(tcomp_kuhn, mixed_center(m3))
    @test isapprox(diag(DEC.barycentric_hodge(m3, mesh_cent, 2, true)),
                   diag(DEC.barycentric_hodge(m3, mesh_mix3, 2, true));
                   atol=1e-12)
    @test isapprox(DEC.nonorthogonal_hodge(m3, mesh_cent),
                   DEC.nonorthogonal_hodge(m3, mesh_mix3); atol=1e-12)
end

# TODO test with other metrics
