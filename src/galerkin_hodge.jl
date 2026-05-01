using SparseArrays: sparse, SparseMatrixCSC
using LinearAlgebra: dot, inv

# Whitney / Galerkin Hodge stars for simplicial complexes.
#
# Unlike circumcenter/centroid Hodge stars, the Galerkin Hodge does not
# require a dual mesh — it is the Gram matrix of Whitney k-forms on the
# primal complex:
#
#     M_k[α, β] = ∫_M ⟨w_α^k, w_β^k⟩ dV
#
# The result is a sparse SPD matrix. Its diagonal entries are positive and
# off-diagonal entries couple cells that share a top-dim simplex.
#
# Currently supported:
#     k = 1 (0-form mass / P1 FEM)         on 2D and 3D simplicial meshes
#     k = 2 (1-form Whitney mass)          on 2D and 3D simplicial meshes
#     k = 3 (2-form Whitney mass)          on 3D simplicial meshes only
#     k = 4 (3-form / volume mass)         on 3D simplicial meshes only

export galerkin_hodge

"""
    galerkin_hodge(m::Metric{N}, comp::CellComplex{N, K}, k::Int) where {N, K}

Whitney/Galerkin Hodge mass matrix for primal `k-1` forms on a simplicial
complex. Returns a sparse SPD `SparseMatrixCSC{Float64}` of size
`length(comp.cells[k]) × length(comp.cells[k])`.

The Galerkin Hodge is the Gram matrix of Whitney `k-1`-forms; it does not
require a dual mesh and is well-defined on arbitrary (skewed, irregular)
simplicial meshes. The matrix is non-diagonal but sparse.

For the standard 0-form Laplacian via Galerkin Hodge:

    L = inv(galerkin_hodge(m, comp, 1)) * d_0' * galerkin_hodge(m, comp, 2) * d_0

where `d_0 = exterior_derivative(comp, 1)`. (The inverse is usually not
formed explicitly; use `cholesky` and back-solves for production code.)
"""
function galerkin_hodge(m::Metric{N}, comp::CellComplex{N, K}, k::Int) where {N, K}
    @assert simplicial(comp) "galerkin_hodge(::CellComplex, ...) requires a " *
        "simplicial complex; pass a TriangulatedComplex (with k=1) for polytope meshes"
    @assert 1 <= k <= K "k=$k out of range 1:$K"
    if k == 1
        return _assemble_galerkin(m, comp, k, _local_mass_0form)
    elseif k == 2
        return _assemble_galerkin(m, comp, k, _local_mass_1form)
    elseif k == 3 && K == 4
        return _assemble_galerkin(m, comp, k, _local_mass_2form)
    elseif k == K
        return _assemble_galerkin(m, comp, k, _local_mass_topform)
    else
        error("galerkin_hodge not implemented for k=$k in $(N)D (K=$K)")
    end
end

"""
    galerkin_hodge(m::Metric{N}, tcomp::TriangulatedComplex{N, K}, k::Int) where {N, K}

Galerkin Hodge for a possibly non-simplicial mesh. The DOFs are the
polytope-level cells in `tcomp.complex.cells[k]`. Each top-dim primal
cell (hex, prism, pyramid, or tet) contributes via its stored simplex
decomposition `tcomp.simplices[c]`.

Currently:
- `k = 1`: P1 mass matrix on polytope vertices (correct for any
  polytope decomposition since the basis is linear on each sub-tet
  and all sub-tet vertices are polytope vertices).
- `k > 1` on a non-simplicial complex: not implemented — Whitney 1- and
  2-forms on a polytope edge/face require a polytope-specific
  basis (e.g. trilinear-hex Nedelec), which differs from the
  sub-tet Whitney form.
"""
function galerkin_hodge(m::Metric{N}, tcomp::TriangulatedComplex{N, K},
    k::Int) where {N, K}
    if simplicial(tcomp.complex)
        return galerkin_hodge(m, tcomp.complex, k)
    end
    if k == 1
        return _assemble_galerkin_polytope_0form(m, tcomp)
    elseif k == 2 && N == 3 && K == 4
        return _assemble_galerkin_hex_1form(m, tcomp)
    else
        error("galerkin_hodge with TriangulatedComplex supports k=1 " *
              "for any polytope mesh, and k=2 for axis-aligned hex meshes only; " *
              "got k=$k, N=$N, K=$K")
    end
end

# 0-form mass on polytope mesh: each polytope contributes via its sub-tet
# decomposition, summing per-tet local mass at polytope vertex indices.
function _assemble_galerkin_polytope_0form(m::Metric{N},
    tcomp::TriangulatedComplex{N, K}) where {N, K}
    comp = tcomp.complex
    n_v = length(comp.cells[1])
    point_to_idx = Dict{Point{N}, Int}()
    for (i, vc) in enumerate(comp.cells[1])
        point_to_idx[vc.points[1]] = i
    end
    rows, cols, vals = Int[], Int[], Float64[]
    for top in comp.cells[K]
        for (s_simple, _sign) in tcomp.simplices[top]
            s = Simplex(s_simple)
            Mloc = _local_mass_0form(m, s)
            local_idx = [point_to_idx[p] for p in s.points]
            for i in 1:K, j in 1:K
                push!(rows, local_idx[i])
                push!(cols, local_idx[j])
                push!(vals, Mloc[i, j])
            end
        end
    end
    return sparse(rows, cols, vals, n_v, n_v)
end

export galerkin_stiffness, galerkin_laplacian
"""
    galerkin_stiffness(m, comp_or_tcomp) -> SparseMatrixCSC

The stiffness matrix `K = d_0' M_1 d_0` for the Galerkin Poisson
problem. Symmetric positive semi-definite (zero on constants). For a
boundary-value problem solve `K · u = M_0 · f` after applying boundary
conditions; for the eigenproblem solve `K v = λ M_0 v`.

`comp_or_tcomp` may be a `CellComplex` (must be simplicial) or a
`TriangulatedComplex` (uses sub-tet Whitney forms; currently only for
simplicial primal complex, since `M_1` is not yet defined on polytope
edges).
"""
galerkin_stiffness(m::Metric, comp::CellComplex) =
    transpose(exterior_derivative(comp, 1)) * galerkin_hodge(m, comp, 2) *
    exterior_derivative(comp, 1)

galerkin_stiffness(m::Metric, tcomp::TriangulatedComplex) =
    galerkin_stiffness(m, tcomp.complex)

"""
    galerkin_laplacian(m, comp_or_tcomp) -> (M_0, K)

Convenience wrapper returning the mass matrix `M_0` and stiffness matrix
`K = d_0' M_1 d_0` for the standard 0-form Galerkin (FEM) Laplacian.

Solve `K · u = M_0 · f` for the Poisson problem `-Δu = f`, after
imposing boundary conditions on the appropriate rows/cols of `K` and
the corresponding entries of `M_0 f`.
"""
function galerkin_laplacian(m::Metric, comp::CellComplex)
    return (galerkin_hodge(m, comp, 1), galerkin_stiffness(m, comp))
end

galerkin_laplacian(m::Metric, tcomp::TriangulatedComplex) =
    galerkin_laplacian(m, tcomp.complex)

export galerkin_hodge_laplacian_block
"""
    galerkin_hodge_laplacian_block(m::Metric{N}, comp::CellComplex{N, K}, k::Int) where {N, K}

Build the saddle-point (mixed-FEM) block matrix `A` and the corresponding
right-hand-side mass matrix on the second block, for the Hodge Laplacian
on `k-1` forms (the de Rham Laplacian `Δ_H = dδ + δd`).

The mixed system finds `(σ, ω) ∈ V_{k-1} × V_k` such that:

    M_{k-1} σ − d_{k-1}ᵀ M_k ω = 0
    M_k d_{k-1} σ + d_kᵀ M_{k+1} d_k ω = M_k f

i.e. `σ = δω` (auxiliary variable) and `dσ + δdω = f` ⟺ `Δ_H ω = f`.
For `k = 2` (the 1-form Laplacian) this is the Hodge-Laplacian
analogue of `Δω = f`.

Returns `(A, M_k)` where `A` is the block matrix above.

For `k = 1` (0-form Laplacian) the mixed formulation degenerates to the
standard FEM stiffness; use `galerkin_laplacian` instead.
"""
function galerkin_hodge_laplacian_block(m::Metric{N}, comp::CellComplex{N, K},
    k::Int) where {N, K}
    @assert 2 <= k < K (
        "galerkin_hodge_laplacian_block needs M_{k-1}, M_k, M_{k+1} — " *
        "supported range is 2 ≤ k ≤ K-1, got k=$k, K=$K")
    M_lower = galerkin_hodge(m, comp, k - 1)
    M_mid   = galerkin_hodge(m, comp, k)
    M_upper = galerkin_hodge(m, comp, k + 1)
    d_lower = exterior_derivative(comp, k - 1)
    d_upper = exterior_derivative(comp, k)

    n_lower = size(M_lower, 1)
    n_mid   = size(M_mid, 1)
    A = [M_lower                  -transpose(d_lower) * M_mid;
         M_mid * d_lower           transpose(d_upper) * M_upper * d_upper]
    return (A, M_mid)
end

# Barycentric-coordinate gradients on a full-dim simplex.
# Returns a Vector of (K = N+1) gradient SVectors, one per simplex vertex.
# ∇λ_i is the Euclidean gradient of the i-th barycentric coordinate; it is
# constant inside the simplex.
function _barycentric_gradients(s::Simplex{N, K}) where {N, K}
    @assert K == N + 1 "barycentric gradients require a full-dim simplex"
    Vmat = zeros(N, N)
    for i in 1:N
        Vmat[:, i] = s.points[i + 1].coords - s.points[1].coords
    end
    A = transpose(inv(Vmat))  # ∇λ_{i+1} = column i of A, for i=1..N
    grads = Vector{SVector{N, Float64}}(undef, K)
    g0 = zeros(N)
    for i in 1:N
        gi = SVector{N, Float64}(A[:, i])
        grads[i + 1] = gi
        g0 = g0 - A[:, i]
    end
    grads[1] = SVector{N, Float64}(g0)
    return grads
end

# Standard FEM P1 mass matrix on a single simplex of dimension n = K-1
# with volume V:
#     M[i, j] = V (1 + δ_ij) / ((n+1)(n+2))
# Returns a K × K dense matrix.
function _local_mass_0form(m::Metric{N}, s::Simplex{N, K}) where {N, K}
    V = volume(m, s)
    n = K - 1
    base = V / ((n + 1) * (n + 2))
    Mloc = fill(base, K, K)
    for i in 1:K
        Mloc[i, i] = 2 * base
    end
    return Mloc
end

# Whitney 1-form local mass matrix.
# For a full-dim simplex with K = N+1 vertices, there are C(K,2) edges.
# The Whitney form for an edge from vertex `a` to vertex `b` (a, b being
# "negative" and "positive" endpoints in the d_0 sign convention) is
#     w_{ab} = λ_a dλ_b - λ_b dλ_a.
# Inner product expansion + ∫ λ_i λ_j = V/((n+1)(n+2)) (1+δ_ij) gives, for
# edges α=(a,b), β=(c,d):
#     M[α,β] = (V/((n+1)(n+2))) ·
#              [(1+δ_ac) G_bd - (1+δ_ad) G_bc - (1+δ_bc) G_ad + (1+δ_bd) G_ac]
# where G_ij = ∇λ_i · ∇λ_j (Euclidean inner product induced by the metric).
#
# Returns a (E_local × E_local) matrix where E_local = C(K, 2), indexed by
# the canonical (i<j) ordering of vertex pairs in s.points.
function _local_mass_1form(m::Metric{N}, s::Simplex{N, K}) where {N, K}
    @assert K == N + 1 "1-form Whitney mass requires a full-dim simplex"
    V = volume(m, s)
    grads = _barycentric_gradients(s)
    G = zeros(K, K)
    for i in 1:K, j in 1:K
        G[i, j] = inner_product(m, grads[i], grads[j])
    end
    n = K - 1
    base = V / ((n + 1) * (n + 2))

    pairs = [(i, j) for i in 1:(K - 1) for j in (i + 1):K]
    E_loc = length(pairs)
    Mloc = zeros(E_loc, E_loc)
    δ(i, j) = i == j ? 1.0 : 0.0
    for (α, (a, b)) in enumerate(pairs), (β, (c, d)) in enumerate(pairs)
        Mloc[α, β] = base * (
            (1 + δ(a, c)) * G[b, d]
            - (1 + δ(a, d)) * G[b, c]
            - (1 + δ(b, c)) * G[a, d]
            + (1 + δ(b, d)) * G[a, c]
        )
    end
    return Mloc, pairs
end

# Whitney 2-form local mass matrix on a 3D tet (K = 4). Each face triangle
# (i,j,k) of the tet has Whitney 2-form
#     w_{ijk} = 2 (λ_i dλ_j∧dλ_k + λ_j dλ_k∧dλ_i + λ_k dλ_i∧dλ_j)
# with sign chosen by the orientation of (i,j,k). The 4 × 4 local mass
# matrix indexed by the 4 face-triplets uses the identity
#     ⟨dλ_a∧dλ_b, dλ_c∧dλ_d⟩ = (∇λ_a·∇λ_c)(∇λ_b·∇λ_d) − (∇λ_a·∇λ_d)(∇λ_b·∇λ_c)
# combined with ∫ λ_i λ_j = V (1+δ_ij)/((n+1)(n+2)).
function _local_mass_2form(m::Metric{N}, s::Simplex{N, K}) where {N, K}
    @assert N == 3 && K == 4 "2-form Whitney mass implemented for 3D tets"
    V = volume(m, s)
    grads = _barycentric_gradients(s)
    G = zeros(K, K)
    for i in 1:K, j in 1:K
        G[i, j] = inner_product(m, grads[i], grads[j])
    end
    n = K - 1  # = 3
    base = V / ((n + 1) * (n + 2))  # = V / 20

    # Face triplets in canonical (i<j<k) order.
    triplets = [(i, j, k) for i in 1:(K - 2) for j in (i + 1):(K - 1) for k in (j + 1):K]
    F_loc = length(triplets)
    Mloc = zeros(F_loc, F_loc)
    δ(i, j) = i == j ? 1.0 : 0.0

    # ⟨w_{ijk}, w_{lmn}⟩ at a point = 4 · sum over cyclic permutations σ, τ of
    # (i,j,k), (l,m,n) of λ_{σ(1)} λ_{τ(1)} · ⟨dλ_σ(2)∧dλ_σ(3), dλ_τ(2)∧dλ_τ(3)⟩.
    # Integrate λ_a λ_b = base · (1+δ_ab).
    cyc(t) = ((t[1], t[2], t[3]), (t[2], t[3], t[1]), (t[3], t[1], t[2]))
    inner2(a, b, c, d) = G[a, c] * G[b, d] - G[a, d] * G[b, c]

    for (α, t1) in enumerate(triplets), (β, t2) in enumerate(triplets)
        acc = 0.0
        for (p, q, r) in cyc(t1)
            for (a, b, c) in cyc(t2)
                acc += base * (1 + δ(p, a)) * inner2(q, r, b, c)
            end
        end
        Mloc[α, β] = 4 * acc
    end
    return Mloc, triplets
end

# Top-form (n-form) local mass matrix on a single simplex of dim n = K-1.
# The Whitney n-form is constant on each simplex, equal to (n!)/V times
# the unit volume form, so the local mass is a 1×1 scalar = 1/V.
function _local_mass_topform(m::Metric{N}, s::Simplex{N, K}) where {N, K}
    V = volume(m, s)
    return reshape([1.0 / V], 1, 1)
end

# Generic assembly. `local_mass(m, s)` must return either:
#   - a K × K matrix indexed by simplex vertex order (k=1, top-form), OR
#   - a (Mloc, pairs/triplets) tuple where the second element gives the
#     simplex-local indexing of the cells of dim k-1 (for 1-form, 2-form).
function _assemble_galerkin(m::Metric{N}, comp::CellComplex{N, K}, k::Int,
    local_mass) where {N, K}
    @assert k <= K
    n_cells = length(comp.cells[k])
    n_top = length(comp.cells[K])

    # Map points → vertex-cell index, edge cells → index, etc.
    point_to_vertex_idx = Dict{Point{N}, Int}()
    for (i, vc) in enumerate(comp.cells[1])
        point_to_vertex_idx[vc.points[1]] = i
    end
    cell_idx = Dict{Cell{N}, Int}()
    for (i, c) in enumerate(comp.cells[k])
        cell_idx[c] = i
    end

    # For k > 1 we need a way to find the cells[k] entry from a tuple of
    # vertex-indices on a simplex. Build a lookup.
    # cells[k] are simplices of K_cell = k vertices.
    cell_by_vertset = Dict{Set{Int}, Cell{N}}()
    for c in comp.cells[k]
        s = Set(point_to_vertex_idx[p] for p in c.points)
        cell_by_vertset[s] = c
    end

    # Sign convention: for k=2 (edges), local Whitney form `w_{ab}` uses
    # `a = negative endpoint, b = positive endpoint` in the d_0 sense. If
    # the local pair (i, j) i<j matches (neg, pos) of the global edge,
    # sign = +1; if reversed, sign = −1 (since w_{ab} = −w_{ba}).
    # For k=3 (faces in 3D): orientation determined by the sign of the
    # boundary inclusion in d_2; we pick canonical (i<j<k) and flip if the
    # global face's orientation differs.
    function _orient_sign(cells_in_k::Vector{Int}, c::Cell{N})
        @assert k >= 2
        if k == 2
            # cells_in_k = [a_local_vertex_idx, b_local_vertex_idx]
            # Determine global edge's negative endpoint
            for vc in c.children
                if !vc.parents[c]  # negative
                    neg_idx = point_to_vertex_idx[vc.points[1]]
                    return cells_in_k[1] == neg_idx ? +1.0 : -1.0
                end
            end
            error("edge has no negative endpoint")
        elseif k == 3
            # cells_in_k = [v1_idx, v2_idx, v3_idx] sorted ascending
            # Compare with global face's stored vertex order (c.points)
            face_pt_indices = [point_to_vertex_idx[p] for p in c.points]
            return _permutation_sign(cells_in_k, face_pt_indices)
        else
            return 1.0
        end
    end

    rows, cols, vals = Int[], Int[], Float64[]
    for top in comp.cells[K]
        s = Simplex(top)
        if k == 1
            Mloc = local_mass(m, s)
            local_idx = [point_to_vertex_idx[p] for p in s.points]
            for i in 1:K, j in 1:K
                push!(rows, local_idx[i])
                push!(cols, local_idx[j])
                push!(vals, Mloc[i, j])
            end
        elseif k == K  # top-form
            Mloc = local_mass(m, s)
            top_idx = cell_idx[top]
            push!(rows, top_idx); push!(cols, top_idx); push!(vals, Mloc[1, 1])
        else
            Mloc, local_index_groups = local_mass(m, s)
            local_vertex_idx = [point_to_vertex_idx[p] for p in s.points]
            global_indices = Int[]
            signs = Float64[]
            for group in local_index_groups
                vert_set_of_group = Set(local_vertex_idx[i] for i in group)
                cell = cell_by_vertset[vert_set_of_group]
                push!(global_indices, cell_idx[cell])
                if k >= 2
                    cells_in_k = [local_vertex_idx[i] for i in group]
                    push!(signs, _orient_sign(cells_in_k, cell))
                else
                    push!(signs, 1.0)
                end
            end
            n_loc = length(global_indices)
            for i in 1:n_loc, j in 1:n_loc
                push!(rows, global_indices[i])
                push!(cols, global_indices[j])
                push!(vals, signs[i] * signs[j] * Mloc[i, j])
            end
        end
    end

    return sparse(rows, cols, vals, n_cells, n_cells)
end

# Sign of the permutation taking `from` to `to` (both same length, same
# elements). Returns +1 / -1.
function _permutation_sign(from::Vector{Int}, to::Vector{Int})
    @assert length(from) == length(to)
    n = length(from)
    perm = [findfirst(==(t), from) for t in to]
    sign = 1
    for i in 1:n, j in (i + 1):n
        if perm[i] > perm[j]
            sign = -sign
        end
    end
    return Float64(sign)
end

# ============================================================================
# Hex (lowest-order Nedelec) Whitney 1-form mass matrix.
#
# For an axis-aligned hex with side lengths Lx, Ly, Lz, there are 12 edges
# split into 3 axis groups of 4. Within an axis group, the basis function for
# the edge at perpendicular-plane corner (α, β) ∈ {0,1}² is
#
#     φ_e = (1/L_axis) · ν_α(perp1) · ν_β(perp2) · e_axis
#
# where ν_0(t) = 1 - t/L, ν_1(t) = t/L (linear hats over the axis-perpendicular
# coordinate). The line integral ∫_e φ · t = 1 by construction.
#
# The 12 × 12 mass matrix is block-diagonal in the 3 axis groups (different-
# axis edges are orthogonal vectors, so their inner product is zero).
# Each axis block (4 × 4) is given in closed form by tensor products of
# ∫ ν_α ν_β dy = L · (1/3 if α==β else 1/6).
#
# Currently restricted to AXIS-ALIGNED hexes (vertices in standard order
# 1=(0,0,0), 2=(Lx,0,0), 3=(Lx,Ly,0), 4=(0,Ly,0), 5=(0,0,Lz), …). Non-axis-
# aligned hexes need the trilinear isoparametric mapping, which requires
# numerical quadrature and is left as future work.

# Canonical local edge orientation: (from_local_idx, to_local_idx) such that
# the geometric direction goes along +axis.
const _HEX_X_EDGES = ((1, 2), (4, 3), (5, 6), (8, 7))
const _HEX_Y_EDGES = ((1, 4), (2, 3), (5, 8), (6, 7))
const _HEX_Z_EDGES = ((1, 5), (2, 6), (3, 7), (4, 8))

# Perpendicular-plane corner labels (∈ {0,1}²) for each edge in each axis group.
const _HEX_X_CORNERS = ((0, 0), (1, 0), (0, 1), (1, 1))  # (y_corner, z_corner)
const _HEX_Y_CORNERS = ((0, 0), (1, 0), (0, 1), (1, 1))  # (x_corner, z_corner)
const _HEX_Z_CORNERS = ((0, 0), (1, 0), (1, 1), (0, 1))  # (x_corner, y_corner)

_int_hat(α::Int, β::Int, L::Float64) = (α == β) ? L / 3 : L / 6

function _hex_local_mass_1form(::Metric{3}, hex_points::Vector{Point{3}})
    @assert length(hex_points) == 8 "hex must have exactly 8 vertices"
    p1 = hex_points[1].coords
    p2 = hex_points[2].coords
    p4 = hex_points[4].coords
    p5 = hex_points[5].coords
    Lx = p2[1] - p1[1]
    Ly = p4[2] - p1[2]
    Lz = p5[3] - p1[3]
    @assert Lx > 0 && Ly > 0 && Lz > 0 "hex must be axis-aligned with positive side lengths"

    Mloc = zeros(12, 12)
    # x-edges (rows/cols 1..4)
    for i in 1:4, j in 1:4
        y_i, z_i = _HEX_X_CORNERS[i]
        y_j, z_j = _HEX_X_CORNERS[j]
        Mloc[i, j] = (1 / Lx) * _int_hat(y_i, y_j, Ly) * _int_hat(z_i, z_j, Lz)
    end
    # y-edges (5..8)
    for i in 1:4, j in 1:4
        x_i, z_i = _HEX_Y_CORNERS[i]
        x_j, z_j = _HEX_Y_CORNERS[j]
        Mloc[4 + i, 4 + j] = (1 / Ly) * _int_hat(x_i, x_j, Lx) * _int_hat(z_i, z_j, Lz)
    end
    # z-edges (9..12)
    for i in 1:4, j in 1:4
        x_i, y_i = _HEX_Z_CORNERS[i]
        x_j, y_j = _HEX_Z_CORNERS[j]
        Mloc[8 + i, 8 + j] = (1 / Lz) * _int_hat(x_i, x_j, Lx) * _int_hat(y_i, y_j, Ly)
    end
    edge_pairs = vcat(collect(_HEX_X_EDGES), collect(_HEX_Y_EDGES), collect(_HEX_Z_EDGES))
    return Mloc, edge_pairs
end

# Global assembly for axis-aligned hex meshes. Errors out if the mesh
# contains non-hex top-dim cells (mixed polytope mesh).
function _assemble_galerkin_hex_1form(m::Metric{3}, tcomp::TriangulatedComplex{3, 4})
    comp = tcomp.complex
    n_e = length(comp.cells[2])

    # Edge lookup by unordered vertex-Point pair.
    edge_lookup = Dict{Set{Point{3}}, Cell{3}}()
    for e in comp.cells[2]
        edge_lookup[Set(c.points[1] for c in e.children)] = e
    end
    edge_idx = Dict{Cell{3}, Int}()
    for (i, e) in enumerate(comp.cells[2])
        edge_idx[e] = i
    end

    rows, cols, vals = Int[], Int[], Float64[]
    for top in comp.cells[4]
        if length(top.points) != 8
            error("_assemble_galerkin_hex_1form: top-dim cell has $(length(top.points)) " *
                  "vertices — only 8-vertex (hex) cells supported")
        end

        Mloc, edge_pairs = _hex_local_mass_1form(m, top.points)
        # For each local edge, find global edge cell and its sign vs canonical orientation.
        global_idx = Vector{Int}(undef, 12)
        signs = Vector{Float64}(undef, 12)
        for (α, (from_l, to_l)) in enumerate(edge_pairs)
            p_from = top.points[from_l]
            p_to   = top.points[to_l]
            e = edge_lookup[Set([p_from, p_to])]
            global_idx[α] = edge_idx[e]
            # Determine global d_0 orientation
            v_pos = nothing
            v_neg = nothing
            for vc in e.children
                if vc.parents[e]
                    v_pos = vc.points[1]
                else
                    v_neg = vc.points[1]
                end
            end
            # Local canonical: from p_from to p_to.
            # Global d_0: from v_neg to v_pos.
            # Sign +1 if (v_neg, v_pos) == (p_from, p_to), else −1.
            signs[α] = (v_neg == p_from && v_pos == p_to) ? 1.0 : -1.0
        end

        for i in 1:12, j in 1:12
            push!(rows, global_idx[i])
            push!(cols, global_idx[j])
            push!(vals, signs[i] * signs[j] * Mloc[i, j])
        end
    end
    return sparse(rows, cols, vals, n_e, n_e)
end
