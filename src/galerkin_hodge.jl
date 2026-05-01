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
    @assert simplicial(comp) "galerkin_hodge requires a simplicial complex"
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
