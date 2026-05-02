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
        return _assemble_galerkin_polytope_1form(m, tcomp)
    else
        error("galerkin_hodge with TriangulatedComplex supports k=1 " *
              "for any polytope mesh, and k=2 for tet/hex/prism mixes in 3D; " *
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
`TriangulatedComplex`. The `TriangulatedComplex` version dispatches
through `galerkin_hodge(m, tcomp, 2)`, which currently supports
simplicial meshes (any) and axis-aligned hex meshes (Nédélec).
"""
galerkin_stiffness(m::Metric, comp::CellComplex) =
    transpose(exterior_derivative(comp, 1)) * galerkin_hodge(m, comp, 2) *
    exterior_derivative(comp, 1)

# For TriangulatedComplex: use the polytope-aware `galerkin_hodge(m, tcomp, 2)`
# (which dispatches to hex Nédélec for hex meshes, etc.), NOT the simplicial-
# only CellComplex path. d_0 is structural and works on either complex type.
#
# Special case: if the mesh contains pyramid cells (which lack a Nédélec
# 1-form basis on the 8 polytope edges), assemble the stiffness directly via
# per-polytope sub-tet integration. This gives the FEM P1 stiffness on the
# pyramid's simplicial subdivision and provides clean h² Poisson convergence,
# at the cost of not exposing a separate `galerkin_hodge(_, _, 2)` matrix.
function galerkin_stiffness(m::Metric, tcomp::TriangulatedComplex)
    if !simplicial(tcomp.complex) &&
       any(length(top.points) == 5 for top in tcomp.complex.cells[end])
        return _assemble_polytope_stiffness(m, tcomp)
    end
    d0 = exterior_derivative(tcomp.complex, 1)
    return transpose(d0) * galerkin_hodge(m, tcomp, 2) * d0
end

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
    (galerkin_hodge(m, tcomp, 1), galerkin_stiffness(m, tcomp))

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

# Trilinear isoparametric Nédélec mass matrix for a general (possibly
# non-axis-aligned) hex. The reference cube basis is pulled back to physical
# space via the covariant Piola transformation
#     φ^p(x) = J^{-T}(ξ) · φ^r(ξ)
# so the mass entry becomes
#     M[α,β] = ∫_ref (φ_α^r)ᵀ J^{-1} J^{-T} φ_β^r |det J| dξ dη dζ
# Evaluated by 2 × 2 × 2 Gauss-Legendre quadrature on the reference cube
# (exact for axis-aligned hexes, accurate to O(h⁴) for general trilinear maps).

# Reference vertex labelling matching _HEX_FACES / _HEX_REF_VERT_BIT.
const _HEX_REF_VERT_BIT = ((0,0,0), (1,0,0), (1,1,0), (0,1,0),
                           (0,0,1), (1,0,1), (1,1,1), (0,1,1))

@inline _ν(α::Int, t::Float64) = α == 0 ? 1 - t : t
@inline _dν(α::Int) = α == 0 ? -1.0 : 1.0

@inline function _hex_shape_grad(i::Int, ξ::Float64, η::Float64, ζ::Float64)
    a, b, c = _HEX_REF_VERT_BIT[i]
    νξ, νη, νζ = _ν(a, ξ), _ν(b, η), _ν(c, ζ)
    return (_dν(a) * νη * νζ, νξ * _dν(b) * νζ, νξ * νη * _dν(c))
end

# Reference Nédélec basis function value at (ξ, η, ζ). Returns a 3-tuple
# (only one component non-zero). α ∈ 1..12 indexes edges in the order:
# x-edges (1..4), y-edges (5..8), z-edges (9..12), inside each axis using
# _HEX_X_CORNERS / _HEX_Y_CORNERS / _HEX_Z_CORNERS.
@inline function _hex_ref_nedelec(α::Int, ξ::Float64, η::Float64, ζ::Float64)
    if α <= 4
        y_c, z_c = _HEX_X_CORNERS[α]
        return (_ν(y_c, η) * _ν(z_c, ζ), 0.0, 0.0)
    elseif α <= 8
        x_c, z_c = _HEX_Y_CORNERS[α - 4]
        return (0.0, _ν(x_c, ξ) * _ν(z_c, ζ), 0.0)
    else
        x_c, y_c = _HEX_Z_CORNERS[α - 8]
        return (0.0, 0.0, _ν(x_c, ξ) * _ν(y_c, η))
    end
end

# 2-point Gauss-Legendre on [0, 1].
const _GAUSS_2PT  = ((1 - 1/sqrt(3)) / 2, (1 + 1/sqrt(3)) / 2)
const _GAUSS_2PT_W = (0.5, 0.5)

function _hex_local_mass_1form(::Metric{3}, hex_points::Vector{Point{3}})
    @assert length(hex_points) == 8 "hex must have exactly 8 vertices"
    Mloc = zeros(12, 12)
    Jbuf = zeros(3, 3)
    for iξ in 1:2, iη in 1:2, iζ in 1:2
        ξ = _GAUSS_2PT[iξ]; η = _GAUSS_2PT[iη]; ζ = _GAUSS_2PT[iζ]
        w = _GAUSS_2PT_W[iξ] * _GAUSS_2PT_W[iη] * _GAUSS_2PT_W[iζ]

        # Jacobian J at this quadrature point.
        fill!(Jbuf, 0.0)
        for i in 1:8
            dNξ, dNη, dNζ = _hex_shape_grad(i, ξ, η, ζ)
            pi_coords = hex_points[i].coords
            for k in 1:3
                Jbuf[k, 1] += dNξ * pi_coords[k]
                Jbuf[k, 2] += dNη * pi_coords[k]
                Jbuf[k, 3] += dNζ * pi_coords[k]
            end
        end
        detJ = Jbuf[1,1]*(Jbuf[2,2]*Jbuf[3,3] - Jbuf[2,3]*Jbuf[3,2]) -
               Jbuf[1,2]*(Jbuf[2,1]*Jbuf[3,3] - Jbuf[2,3]*Jbuf[3,1]) +
               Jbuf[1,3]*(Jbuf[2,1]*Jbuf[3,2] - Jbuf[2,2]*Jbuf[3,1])
        @assert detJ > 1e-14 "hex Jacobian determinant non-positive at quadrature point ($detJ); check vertex ordering or degenerate hex"
        Jinv = inv(Jbuf)
        # Mc = J^{-1} J^{-T}, the 3 × 3 SPD metric for the inner product of
        # reference 1-forms after Piola transformation.
        Mc = Jinv * transpose(Jinv)

        # Cache the 12 reference basis values at this point.
        φref = ntuple(α -> _hex_ref_nedelec(α, ξ, η, ζ), 12)
        wj = w * detJ
        for α in 1:12, β in 1:12
            va = φref[α]; vb = φref[β]
            Mca1 = Mc[1,1]*va[1] + Mc[1,2]*va[2] + Mc[1,3]*va[3]
            Mca2 = Mc[2,1]*va[1] + Mc[2,2]*va[2] + Mc[2,3]*va[3]
            Mca3 = Mc[3,1]*va[1] + Mc[3,2]*va[2] + Mc[3,3]*va[3]
            Mloc[α, β] += wj * (Mca1 * vb[1] + Mca2 * vb[2] + Mca3 * vb[3])
        end
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

# ============================================================================
# Prism (lowest-order Nédélec wedge element) Whitney 1-form mass matrix.
#
# A right (axis-aligned) triangular prism has 6 vertices with the codebase
# convention: 1,2,3 = bottom triangle, 4,5,6 = top triangle directly above
# 1,2,3 along +z. There are 9 edges:
#   3 bottom triangle edges (1,2), (1,3), (2,3)
#   3 top    triangle edges (4,5), (4,6), (5,6)
#   3 vertical edges        (1,4), (2,5), (3,6)
#
# The Nédélec basis is a tensor-product:
#   - bottom edge e (= 2D Whitney 1-form on triangle):
#       φ_e = (1 − z/L) · w_e^T(x,y)
#   - top edge e:
#       φ_e = (z/L)     · w_e^T(x,y)
#   - vertical edge at vertex a:
#       φ_v = (1/L) · λ_a^T(x,y) · ẑ
#
# Different-axis groups (horizontal vs vertical) are orthogonal in inner
# product, so the 9 × 9 mass is block-diagonal with a 6 × 6 horizontal
# block (further block-2 × 2 in bottom/top) and a 3 × 3 vertical block.
# All blocks reduce to scaled 2D triangle Whitney / P1 masses.
#
# Implementation: trilinear isoparametric mapping with 3-point triangle ×
# 2-point z Gauss-Legendre quadrature on the reference unit prism. The
# integrand is at most degree 2 in each reference variable for axis-aligned
# right prisms, so quadrature is exact there; for oblique prisms (general
# top triangle pose) it is O(h⁴) accurate, sufficient for h² overall.

# Reference 2D Whitney 1-form on the unit triangle for edge (a, b) at
# parametric (ξ, η). Triangle barycentric: λ_1 = 1−ξ−η, λ_2 = ξ, λ_3 = η.
@inline function _ref_whitney_2d(a::Int, b::Int, ξ::Float64, η::Float64)
    λa = a == 1 ? 1 - ξ - η : (a == 2 ? ξ : η)
    λb = b == 1 ? 1 - ξ - η : (b == 2 ? ξ : η)
    ga_x = a == 1 ? -1.0 : (a == 2 ? 1.0 : 0.0)
    ga_y = a == 1 ? -1.0 : (a == 2 ? 0.0 : 1.0)
    gb_x = b == 1 ? -1.0 : (b == 2 ? 1.0 : 0.0)
    gb_y = b == 1 ? -1.0 : (b == 2 ? 0.0 : 1.0)
    return (λa * gb_x - λb * ga_x, λa * gb_y - λb * ga_y)
end

# Reference prism Nédélec basis at (ξ, η, ζ) where (ξ, η) are unit-triangle
# coords (λ_1=1−ξ−η, λ_2=ξ, λ_3=η) and ζ ∈ [0, 1]. Edges 1..3 = bottom
# triangle in (1,2)/(1,3)/(2,3) order, 4..6 = top, 7..9 = vertical.
@inline function _ref_prism_nedelec(α::Int, ξ::Float64, η::Float64, ζ::Float64)
    if α <= 6
        a, b = α == 1 || α == 4 ? (1, 2) : (α == 2 || α == 5 ? (1, 3) : (2, 3))
        wx, wy = _ref_whitney_2d(a, b, ξ, η)
        scale = α <= 3 ? (1 - ζ) : ζ
        return (scale * wx, scale * wy, 0.0)
    else
        i = α - 6           # 1, 2, or 3
        λi = i == 1 ? 1 - ξ - η : (i == 2 ? ξ : η)
        return (0.0, 0.0, λi)
    end
end

# Trilinear shape-function gradients on the reference unit prism.
# χ(ξ, η, ζ) = Σ N_i p_i with
#   N_1 = (1−ξ−η)(1−ζ),  N_2 = ξ(1−ζ),  N_3 = η(1−ζ),
#   N_4 = (1−ξ−η)ζ,      N_5 = ξζ,      N_6 = ηζ.
@inline function _prism_shape_grad(i::Int, ξ::Float64, η::Float64, ζ::Float64)
    if i == 1
        return (-(1-ζ), -(1-ζ), -(1-ξ-η))
    elseif i == 2
        return ((1-ζ), 0.0, -ξ)
    elseif i == 3
        return (0.0, (1-ζ), -η)
    elseif i == 4
        return (-ζ, -ζ, (1-ξ-η))
    elseif i == 5
        return (ζ, 0.0, ξ)
    else  # i == 6
        return (0.0, ζ, η)
    end
end

# 3-point Gauss for the unit triangle (exact for degree 2). Volume 1/2.
const _GAUSS_TRI_3PT   = ((1/6, 1/6), (4/6, 1/6), (1/6, 4/6))
const _GAUSS_TRI_3PT_W = (1/6, 1/6, 1/6)

function _prism_local_mass_1form(::Metric{3}, prism_points::Vector{Point{3}})
    @assert length(prism_points) == 6 "prism must have exactly 6 vertices"
    Mloc = zeros(9, 9)
    Jbuf = zeros(3, 3)
    for tri in 1:3
        ξ_t, η_t = _GAUSS_TRI_3PT[tri]
        w_tri = _GAUSS_TRI_3PT_W[tri]
        for zi in 1:2
            ζ = _GAUSS_2PT[zi]
            w_z = _GAUSS_2PT_W[zi]
            w = w_tri * w_z

            fill!(Jbuf, 0.0)
            for i in 1:6
                dξ, dη, dζ = _prism_shape_grad(i, ξ_t, η_t, ζ)
                pi_coords = prism_points[i].coords
                for k in 1:3
                    Jbuf[k, 1] += dξ * pi_coords[k]
                    Jbuf[k, 2] += dη * pi_coords[k]
                    Jbuf[k, 3] += dζ * pi_coords[k]
                end
            end
            detJ = Jbuf[1,1]*(Jbuf[2,2]*Jbuf[3,3] - Jbuf[2,3]*Jbuf[3,2]) -
                   Jbuf[1,2]*(Jbuf[2,1]*Jbuf[3,3] - Jbuf[2,3]*Jbuf[3,1]) +
                   Jbuf[1,3]*(Jbuf[2,1]*Jbuf[3,2] - Jbuf[2,2]*Jbuf[3,1])
            @assert detJ > 1e-14 "prism Jacobian non-positive ($detJ); check vertex ordering"
            Jinv = inv(Jbuf)
            Mc = Jinv * transpose(Jinv)

            φref = ntuple(α -> _ref_prism_nedelec(α, ξ_t, η_t, ζ), 9)
            wj = w * detJ
            for α in 1:9, β in 1:9
                va = φref[α]; vb = φref[β]
                Mca1 = Mc[1,1]*va[1] + Mc[1,2]*va[2] + Mc[1,3]*va[3]
                Mca2 = Mc[2,1]*va[1] + Mc[2,2]*va[2] + Mc[2,3]*va[3]
                Mca3 = Mc[3,1]*va[1] + Mc[3,2]*va[2] + Mc[3,3]*va[3]
                Mloc[α, β] += wj * (Mca1 * vb[1] + Mca2 * vb[2] + Mca3 * vb[3])
            end
        end
    end
    bot_pairs  = [(1, 2), (1, 3), (2, 3)]
    top_pairs  = [(4, 5), (4, 6), (5, 6)]
    vert_pairs = [(1, 4), (2, 5), (3, 6)]
    edge_pairs = vcat(bot_pairs, top_pairs, vert_pairs)
    return Mloc, edge_pairs
end

# Polytope-aware 1-form mass dispatcher. Replaces the old hex-only
# `_assemble_galerkin_hex_1form` and handles mixed simplicial / hex / prism
# meshes (and tets via the existing simplicial Whitney).
function _polytope_1form_local_mass(m::Metric{3}, top::Cell{3})
    n_pts = length(top.points)
    if n_pts == 4
        # Tet: sub-tet Whitney 1-form (existing simplicial path)
        s = Simplex(top)
        return _local_mass_1form(m, s)
    elseif n_pts == 6
        return _prism_local_mass_1form(m, top.points)
    elseif n_pts == 8
        return _hex_local_mass_1form(m, top.points)
    elseif n_pts == 5
        error("Galerkin Hodge k=2 not implemented for pyramid cells. The " *
              "lowest-order Nédélec basis on a pyramid (Bedrosian or " *
              "Gradinaru-Hiptmair) requires special apex-singularity " *
              "handling and is not yet supported. Use a tet-only mesh or " *
              "the over-relaxed nonorthogonal_hodge instead for pyramid cells.")
    else
        error("Galerkin Hodge k=2: unsupported top-dim cell with $n_pts vertices")
    end
end

# Generic polytope-mesh 1-form mass assembly. Handles tet / prism / hex
# uniformly via `_polytope_1form_local_mass`. (Replaces the hex-only
# `_assemble_galerkin_hex_1form` from the previous commit.)
function _assemble_galerkin_polytope_1form(m::Metric{3},
    tcomp::TriangulatedComplex{3, 4})
    comp = tcomp.complex
    n_e = length(comp.cells[2])

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
        Mloc, edge_pairs = _polytope_1form_local_mass(m, top)
        n_loc = length(edge_pairs)
        global_idx = Vector{Int}(undef, n_loc)
        signs      = Vector{Float64}(undef, n_loc)
        for (α, (from_l, to_l)) in enumerate(edge_pairs)
            p_from = top.points[from_l]
            p_to   = top.points[to_l]
            e = edge_lookup[Set([p_from, p_to])]
            global_idx[α] = edge_idx[e]
            v_pos = nothing; v_neg = nothing
            for vc in e.children
                if vc.parents[e]
                    v_pos = vc.points[1]
                else
                    v_neg = vc.points[1]
                end
            end
            signs[α] = (v_neg == p_from && v_pos == p_to) ? 1.0 : -1.0
        end
        for i in 1:n_loc, j in 1:n_loc
            push!(rows, global_idx[i])
            push!(cols, global_idx[j])
            push!(vals, signs[i] * signs[j] * Mloc[i, j])
        end
    end
    return sparse(rows, cols, vals, n_e, n_e)
end

# ============================================================================
# Pyramid Whitney 1-form mass — pragmatic Wachspress / sub-tet hybrid.
#
# Pyramid lowest-order Nédélec is research-grade because of the apex
# singularity; proper Bedrosian / Gradinaru-Hiptmair bases use rational
# polynomials with apex-singular terms. Here we instead compute the
# pyramid's contribution to the Galerkin FEM stiffness DIRECTLY, using
# the stored sub-tet decomposition `tcomp.simplices[pyramid]`:
#
#   K_pyramid[i, j] = Σ_{sub-tets t} V_t · ⟨∇λ_i^t, ∇λ_j^t⟩
#
# This is the standard FEM P1 stiffness on the pyramid's simplicial
# subdivision (the diagonal-of-base edge is integrated out implicitly).
# The 1-form *mass* matrix on pyramid edges is NOT exposed as a separate
# operator (a true Nédélec basis on the 8 polytope edges would need the
# Bedrosian construction); only the per-pyramid stiffness contribution
# is provided, sufficient for `galerkin_laplacian` / Poisson solves.
#
# For purely pyramid meshes, this gives clean h² Poisson convergence;
# for pyramid meshes mixed with hex / prism / tet, the global stiffness
# is assembled by mixing per-cell-type contributions.

function _pyramid_local_stiffness(m::Metric{3}, pyr_points::Vector{Point{3}})
    @assert length(pyr_points) == 5 "pyramid must have exactly 5 vertices"
    K = zeros(5, 5)
    for tv in ((1, 2, 3, 5), (1, 3, 4, 5))   # _PYRAMID_TETS
        s = Simplex([pyr_points[v] for v in tv])
        V = volume(m, s)
        grads = _barycentric_gradients(s)
        for ti in 1:4, tj in 1:4
            K[tv[ti], tv[tj]] += V * inner_product(m, grads[ti], grads[tj])
        end
    end
    return K
end

# Per-cell stiffness contribution for any polytope via its stored sub-tet
# decomposition. Used as a fallback for pyramid (no edge-basis available)
# and as a uniform path for mixed-polytope meshes that include pyramids.
function _polytope_local_stiffness(m::Metric{3}, top::Cell{3},
    sub_tets::Vector{SignedSimpleSimplex{3}})
    n_pts = length(top.points)
    K = zeros(n_pts, n_pts)
    point_to_local = Dict{Point{3}, Int}()
    for (i, p) in enumerate(top.points)
        point_to_local[p] = i
    end
    for (s_simple, _sign) in sub_tets
        s = Simplex(s_simple)
        V = volume(m, s)
        grads = _barycentric_gradients(s)
        local_idx = [point_to_local[p] for p in s.points]
        for ti in 1:4, tj in 1:4
            K[local_idx[ti], local_idx[tj]] += V *
                inner_product(m, grads[ti], grads[tj])
        end
    end
    return K
end

# Global stiffness via per-polytope sub-tet integration. Works on any
# polytope mesh (tet / hex / prism / pyramid / mixed). For pure simplicial
# meshes this is identical to `transpose(d0) * M_1 * d0` with sub-tet Whitney;
# for hex / prism it differs from the Nédélec stiffness but still gives
# h² Poisson convergence.
function _assemble_polytope_stiffness(m::Metric{3}, tcomp::TriangulatedComplex{3, 4})
    comp = tcomp.complex
    n_v = length(comp.cells[1])
    point_to_idx = Dict{Point{3}, Int}()
    for (i, vc) in enumerate(comp.cells[1])
        point_to_idx[vc.points[1]] = i
    end
    rows, cols, vals = Int[], Int[], Float64[]
    for top in comp.cells[4]
        K_loc = _polytope_local_stiffness(m, top, tcomp.simplices[top])
        local_to_global = [point_to_idx[p] for p in top.points]
        n_loc = length(top.points)
        for i in 1:n_loc, j in 1:n_loc
            push!(rows, local_to_global[i])
            push!(cols, local_to_global[j])
            push!(vals, K_loc[i, j])
        end
    end
    return sparse(rows, cols, vals, n_v, n_v)
end
