using SparseArrays: spdiagm, sparse, spzeros, SparseMatrixCSC
using LinearAlgebra: diag, I, dot, pinv
using StaticArrays: SVector

export differential_operator_sequence, barycentric_hodge, corrected_barycentric_hodge,
    nonorthogonal_hodge
"""
    differential_operator_sequence(m::Metric{N}, mesh::Mesh{N, K}, expr::String,
        k::Int, primal::Bool) where {N, K}

Compute the differential operators defined by the string `expr`. This string must consist
of the characters `d`, `★`, `δ`, and `Δ` indicating the exterior derivative, hodge dual,
codifferential, and Laplace-de Rham operators, respectively. If there are two `★` operators
in a row, a single `★★` operator is computed since this avoids unnecessary calculation.
The dimension of the form on which the operator acts is `k-1` and its primality or duality
is indicated by `primal`.
"""
function differential_operator_sequence(m::Metric{N}, mesh::Mesh{N, K}, expr::String,
    k::Int, primal::Bool) where {N, K}
    ops = SparseMatrixCSC{Float64,Int64}[]
    chars = collect(expr)
    i = length(chars)
    while i > 0
        char = chars[i]
        @assert char in ['d', '★', 'δ', 'Δ']
        if char == 'd'
            comp = primal ? mesh.primal.complex : mesh.dual.complex
            push!(ops, exterior_derivative(comp, k))
            k += 1
        elseif char == '★'
            if (i > 1) && chars[i-1] == '★'
                push!(ops, hodge_square(m, mesh, k, primal))
                i -= 1
            else
                push!(ops, circumcenter_hodge(m, mesh, k, primal))
                primal = !primal
                k = K-k+1
            end
        elseif char == 'δ'
            push!(ops, codifferential(m, mesh, k, primal))
            k -= 1
        elseif char == 'Δ'
            push!(ops, laplace_de_Rham(m, mesh, k, primal))
        end
        i -= 1
    end
    return reverse(ops)
end

export differential_operator
"""
    differential_operator(m::Metric{N}, mesh::Mesh{N}, expr::String, k::Int,
        primal::Bool) where N

Compute the differential operator defined by the string `expr`. This string must consist
of the characters `d`, `★`, `δ`, and `Δ` indicating the exterior derivative, hodge dual,
codifferential, and Laplace-de Rham operators, respectively. The dimension of the form on
which the operator acts is `k-1` and its primality or duality is indicated by `primal`.
"""
function differential_operator(m::Metric{N}, mesh::Mesh{N}, expr::String, k::Int,
    primal::Bool) where N
    ops = differential_operator_sequence(m, mesh, expr, k, primal)
    return reduce(*, ops)
end

"""
    differential_operator(m::Metric{N}, mesh::Mesh{N}, expr::String, k::Int,
        primal::Bool, v::AbstractVector{<:Real}) where N

Apply the differential operator defined by the string `expr` to the vector `v`.
"""
function differential_operator(m::Metric{N}, mesh::Mesh{N}, expr::String, k::Int,
    primal::Bool, v::AbstractVector{<:Real}) where N
    ops = differential_operator_sequence(m, mesh, expr, k, primal)
    return foldr(*, ops, init=v)
end

"""
    circumcenter_hodge(m::Metric{N}, mesh::Mesh{N, K}, k::Int, primal::Bool) where {N, K}

Find the discrete hodge star operator using the circumcentric hodge star construction. If
`primal == true`, this operator takes primal `k-1` forms to dual `K-k+1` forms. Otherwise
it takes dual `k-1` forms to priaml `K-k+1` forms.
"""
function circumcenter_hodge(m::Metric{N}, mesh::Mesh{N, K}, k::Int,
    primal::Bool) where {N, K}
    @assert 1 <= k <= K+1
    if k == K+1
        return spzeros(0,0)
    end
    if primal
        ratios = Float64[]
        for (p_cell, d_cell) in zip(mesh.primal.complex.cells[k],
                mesh.dual.complex.cells[K-k+1])
            p_vol = volume(m, mesh.primal, p_cell)
            d_vol = volume(m, mesh.dual, d_cell)
            @assert p_vol != 0
            push!(ratios, d_vol/p_vol)
        end
    else
        pm = hodge_square_sign(m, K, k)
        v = diag(circumcenter_hodge(m, mesh, K-k+1, true))
        @assert !any(v .== 0)
        ratios = pm ./ v
    end
    return spdiagm(0 => ratios)
end

"""
    barycentric_hodge(m::Metric{N}, mesh::Mesh{N, K}, k::Int, primal::Bool) where {N, K}

Find the discrete hodge star operator using barycentric dual construction. This version
uses barycentric centers instead of circumcenters, which can provide better numerical
stability but lacks orthogonality. If `primal == true`, this operator takes primal 
`k-1` forms to dual `K-k+1` forms. Otherwise it takes dual `k-1` forms to primal `K-k+1` forms.
"""
function barycentric_hodge(m::Metric{N}, mesh::Mesh{N, K}, k::Int,
    primal::Bool) where {N, K}
    @assert 1 <= k <= K+1
    if k == K+1
        return spzeros(0,0)
    end
    if primal
        ratios = Float64[]
        for (p_cell, d_cell) in zip(mesh.primal.complex.cells[k],
                mesh.dual.complex.cells[K-k+1])
            p_vol = volume(m, mesh.primal, p_cell)
            d_vol = volume(m, mesh.dual, d_cell)
            @assert p_vol != 0
            push!(ratios, d_vol/p_vol)
        end
    else
        pm = hodge_square_sign(m, K, k)
        v = diag(barycentric_hodge(m, mesh, K-k+1, true))
        @assert !any(v .== 0)
        ratios = pm ./ v
    end
    return spdiagm(0 => ratios)
end

"""
    nonorthogonal_hodge(m::Metric{N}, mesh::Mesh{N, K}) -> SparseMatrixCSC

Hodge star for primal 1-forms (`★_2`) on a 2D (`N=2, K=3`) or 3D (`N=3, K=4`)
mesh whose dual is built from non-orthogonal centers (e.g. centroids). Uses
the OpenFOAM-style over-relaxed decomposition `S = E + T` with `E ∥ d`:

    flux_e = α_or · (u_N − u_P) + T_e · (1/n_cells) Σ_C g_C

where `d_e` is the primal edge vector, `S_e` the dual face area-vector,
`α_or = (S·S)/(S·d)`, `T_e = S_e − α_or · d_e`, and `g_C` is the per-cell
"Diamond scheme" gradient: the unique vector such that `(v_j − v_i) · g_C`
matches the edge value `ω_{ij}` for each edge of the top-dim cell `C`. The
sum runs over the (up to 2 in 2D / many in 3D) primal top-dim cells incident
to edge `e`. The result is a sparse `n_edges × n_edges` matrix; the diagonal
entry coincides with `barycentric_hodge` on orthogonal meshes, and the full
operator is exact for linear `u` and second-order accurate for smooth `u`
(versus the first-order vertex-LS reconstruction used previously).

In 2D, `S_e` is the rotated chord between the two adjacent triangle centers
(or between `e_center` and the single adjacent triangle for boundary edges).
In 3D, `S_e` accumulates ½(v₁ × v₂) over the triangular elementary duals
`[e_center, t_center, tet_center]` — equivalent to the area integral of the
oriented dual face. In both cases the orientation is fixed so `S_e · d_e > 0`.
"""
function nonorthogonal_hodge(m::Metric{N}, mesh::Mesh{N, K}) where {N, K}
    @assert (N == 2 && K == 3) || (N == 3 && K == 4) (
        "nonorthogonal_hodge supports only 2D (N=2,K=3) and 3D (N=3,K=4)")
    primal_comp = mesh.primal.complex
    edges = primal_comp.cells[2]
    n_edges = length(edges)
    top_cells = primal_comp.cells[K]

    edge_idx = Dict{Cell{N}, Int}()
    sizehint!(edge_idx, n_edges)
    for (i, e) in enumerate(edges)
        edge_idx[e] = i
    end

    # Per top-dim cell: column indices of its edges + linear-interpolant
    # gradient reconstruction matrix (Diamond scheme). For a cell with edge
    # vectors arranged as rows of `mat`, `pinv(mat * m.mat)` is the matrix
    # G such that `g_C = G · ω_{edges of C}` is the constant gradient of
    # the unique linear interpolant on C — exact when ω = du.
    cell_grad = Dict{Cell{N}, Matrix{Float64}}()
    cell_edge_cols = Dict{Cell{N}, Vector{Int}}()
    for c in top_cells
        es = _cell_edges(c)
        cell_edge_cols[c] = [edge_idx[e] for e in es]
        mat = zeros(length(es), N)
        for (i, e) in enumerate(es)
            mat[i, :] = _primal_edge_vector(e)
        end
        cell_grad[c] = pinv(mat * m.mat)
    end

    rows, cols, vals = Int[], Int[], Float64[]
    for (i, e) in enumerate(edges)
        d_e = _primal_edge_vector(e)
        S_e = _dual_face_area_vector(mesh, e, d_e)

        Sd = dot(S_e, d_e)
        @assert abs(Sd) > 1e-14 "edge $i: dual face area-vector ⊥ primal edge"
        α_or = dot(S_e, S_e) / Sd
        T_e = S_e - α_or * d_e

        push!(rows, i); push!(cols, i); push!(vals, α_or)

        cells_for_e = _top_cells_containing(e)
        weight = 1.0 / length(cells_for_e)
        for c in cells_for_e
            G = cell_grad[c]
            cols_c = cell_edge_cols[c]
            coef = vec(transpose(T_e) * G)
            for (j, col) in enumerate(cols_c)
                push!(rows, i); push!(cols, col); push!(vals, weight * coef[j])
            end
        end
    end

    return sparse(rows, cols, vals, n_edges, n_edges)
end

# Edges (1-cells, K=2) belonging to a primal cell, with deduplication.
function _cell_edges(c::Cell{N}) where N
    if c.K == 2
        return Cell{N}[c]
    end
    seen = Set{Cell{N}}()
    function recurse(x)
        if x.K == 2
            push!(seen, x)
        else
            for ch in x.children
                recurse(ch)
            end
        end
    end
    recurse(c)
    return collect(seen)
end

# Primal top-dim cells (K = N+1) that contain a given lower-dim cell.
function _top_cells_containing(c::Cell{N}) where N
    K = N + 1
    if c.K == K
        return Cell{N}[c]
    end
    seen = Set{Cell{N}}()
    function recurse(x)
        if x.K == K
            push!(seen, x)
        else
            for p in keys(x.parents)
                recurse(p)
            end
        end
    end
    recurse(c)
    return collect(seen)
end

"""
    corrected_barycentric_hodge(m::Metric{N}, mesh::Mesh{N, K}, k::Int, primal::Bool)

Hodge star with non-orthogonality correction. For primal `k=2` in 2D or 3D
(the case where centroidal-dual non-orthogonality enters the standard
Laplacian on 0-forms), returns `nonorthogonal_hodge`. For all other
`(k, primal)` combinations, falls back to the diagonal `barycentric_hodge`.
"""
function corrected_barycentric_hodge(m::Metric{N}, mesh::Mesh{N, K}, k::Int,
    primal::Bool) where {N, K}
    if k == 2 && primal && ((N == 2 && K == 3) || (N == 3 && K == 4))
        return nonorthogonal_hodge(m, mesh)
    end
    return barycentric_hodge(m, mesh, k, primal)
end

# Primal edge vector with the d_0 sign convention:
# v_positive_endpoint - v_negative_endpoint, where positive/negative is set by
# v.parents[e] ∈ {true, false}.
function _primal_edge_vector(e::Cell{N}) where N
    @assert e.K == 2
    de = zero(SVector{N, Float64})
    for v in e.children
        s = v.parents[e] ? 1.0 : -1.0
        de = de + s * v.points[1].coords
    end
    return de
end

# Dual face area-vector for a primal edge in 2D, oriented so that S·d > 0.
# Uses signed dual simplices [e_center, parent_center] from
# `mesh.dual.simplices[dual(mesh, e)]`.
function _dual_face_area_vector(mesh::Mesh{2, 3}, e::Cell{2},
    d_e::SVector{2, Float64})
    de_cell = dual(mesh, e)
    elems = mesh.dual.simplices[de_cell]

    if length(elems) == 2
        p1 = elems[1][1].points[2].coords
        p2 = elems[2][1].points[2].coords
        chord = p2 - p1
    elseif length(elems) == 1
        ec = elems[1][1].points[1].coords
        pc = elems[1][1].points[2].coords
        chord = ec - pc
    else
        error("edge has unexpected number of dual simplices: $(length(elems))")
    end

    S = SVector{2, Float64}(-chord[2], chord[1])
    return dot(S, d_e) < 0 ? -S : S
end

# Dual face area-vector for a primal edge in 3D. The dual face is a (possibly
# non-planar) polygon fan-triangulated from `e_center`; each elementary dual is
# the triangle `[e_center, triangle_center, tet_center]`. The sign convention
# stored in `mesh.dual.simplices` is chosen for *volume* sums and does not
# orient cross products consistently, so each triangle's normal is independently
# aligned with `d_e` (which is roughly perpendicular to the dual face) before
# summing. For planar dual faces this is exact; for mildly non-planar duals
# it gives the area vector projected onto the half-space defined by `d_e`.
function _dual_face_area_vector(mesh::Mesh{3, 4}, e::Cell{3},
    d_e::SVector{3, Float64})
    de_cell = dual(mesh, e)
    elems = mesh.dual.simplices[de_cell]
    S = zero(SVector{3, Float64})
    for (s, _) in elems
        @assert length(s.points) == 3
        v1 = s.points[2].coords - s.points[1].coords
        v2 = s.points[3].coords - s.points[1].coords
        cross_v = SVector{3, Float64}(
            v1[2] * v2[3] - v1[3] * v2[2],
            v1[3] * v2[1] - v1[1] * v2[3],
            v1[1] * v2[2] - v1[2] * v2[1],
        )
        if dot(cross_v, d_e) < 0
            cross_v = -cross_v
        end
        S = S + 0.5 * cross_v
    end
    return S
end

"""
    hodge_square_sign(m::Metric, K::Int, k::Int)

There is an identity `★★ = sign(det(metric)) * (-1)^((k-1) * (K-k)) * I` for `k-1` forms in
`K-1` dimensions. Compute the coefficient of `I` in this expression.
"""
hodge_square_sign(m::Metric, K::Int, k::Int) = sign(det(collect(m.mat))) *
    (mod((k-1) * (K-k), 2) == 0 ? 1 : -1)

"""
    hodge_square(m::Metric{N}, mesh::Mesh{N, K}, k::Int, primal::Bool) where {N, K}

Compute `★★` without computing the `★` operators by using the identity
`★★ = sign(det(metric)) * (-1)^((k-1) * (K-k)) * I` for `k-1` forms in `K-1` dimensions.
"""
function hodge_square(m::Metric{N}, mesh::Mesh{N, K}, k::Int, primal::Bool) where {N, K}
    comp = primal ? mesh.primal.complex : mesh.dual.complex
    n = k < K+1 ? length(comp.cells[k]) : 0
    return hodge_square_sign(m, K, k) * sparse(I, n, n)
end

"""
    exterior_derivative(comp::CellComplex{N, K}, k::Int) where {N, K}

Find the discrete exterior derivative operator.
"""
function exterior_derivative(comp::CellComplex{N, K}, k::Int) where {N, K}
    @assert 0 <= k <= K
    if k == 0
        return spzeros(length(comp.cells[k+1]), 0)
    end
    row_inds, col_inds, vals = Int[], Int[], Int[]
    for (col_ind, cell) in enumerate(comp.cells[k])
        for p in keys(cell.parents)
            o = cell.parents[p]
            row_ind = findfirst(isequal(p), comp.cells[k+1])
            push!(row_inds, row_ind); push!(col_inds, col_ind); push!(vals, 2 * o - 1)
        end
    end
    num_rows = k+1 <= K ? length(comp.cells[k+1]) : 0
    num_cols = length(comp.cells[k])
    return sparse(row_inds, col_inds, vals, num_rows, num_cols)
end

"""
    codifferential(m::Metric{N}, mesh::Mesh{N, K}, k::Int, primal::Bool) where {N, K}

Compute the codifferential defined by
`δ = sign(det(collect(m.mat))) * (-1)^((K-1) * (k-2) + 1) * ★d★` for `k-1` forms in `K-1`
dimensions.
"""
function codifferential(m::Metric{N}, mesh::Mesh{N, K}, k::Int, primal::Bool) where {N, K}
    ★d★ = differential_operator(m, mesh, "★d★", k, primal)
    s = sign(det(collect(m.mat)))
    return s * (mod((K-1) * (k-2) + 1, 2) == 0 ? 1 : -1) * ★d★
end


"""
    laplace_de_Rham(m::Metric{N}, mesh::Mesh{N, K}, k::Int, primal::Bool) where {N, K}

Compute the Laplace-de Rham operator defined by `Δ = dδ + δd`.
"""
function laplace_de_Rham(m::Metric{N}, mesh::Mesh{N, K}, k::Int, primal::Bool) where {N, K}
    dδ = differential_operator(m, mesh, "dδ", k, primal)
    δd = differential_operator(m, mesh, "δd", k, primal)
    return dδ + δd
end

export sharp
"""
    sharp(m::Metric{N}, comp::CellComplex{N}, form::AbstractVector{<:Real}) where N

Given a 1-form on a cell complex, approximate a vector of length `N` at each vertex using
least squares.
"""
function sharp(m::Metric{N}, comp::CellComplex{N}, form::AbstractVector{<:Real}) where N
    field = Vector{Float64}[]
    for c in comp.cells[1]
        mat = zeros(length(c.parents), N)
        w = zeros(length(c.parents))
        for (row_ind, e) in enumerate(collect(keys(c.parents)))
            mat[row_ind, :] = sum([x.points[1].coords * (2 * x.parents[e] - 1)
                for x in e.children])
            w[row_ind] = form[findfirst(isequal(e), comp.cells[2])]
        end
        push!(field, (mat * m.mat) \ w)
    end
    return field
end
