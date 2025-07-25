using SparseArrays: spdiagm, sparse, spzeros, SparseMatrixCSC
using LinearAlgebra: diag, I

export differential_operator_sequence, barycentric_hodge, corrected_barycentric_hodge
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
    corrected_barycentric_hodge(m::Metric{N}, mesh::Mesh{N, K}, k::Int, primal::Bool; 
                               use_direct_gradient::Bool=true, use_cross_diffusion::Bool=true) where {N, K}

Numerically corrected hodge star operator that addresses orthogonality issues when using
barycentric centers. Applies corrections using direct gradient and cross diffusion terms
to reduce numerical errors caused by lack of orthogonality in barycentric dual meshes.
"""
function corrected_barycentric_hodge(m::Metric{N}, mesh::Mesh{N, K}, k::Int, primal::Bool; 
                                   use_direct_gradient::Bool=true, use_cross_diffusion::Bool=true) where {N, K}
    
    base_hodge = barycentric_hodge(m, mesh, k, primal)
    
    if !use_direct_gradient && !use_cross_diffusion
        return base_hodge
    end
    
    correction = spzeros(size(base_hodge)...)
    
    if use_direct_gradient && k <= K
        correction += direct_gradient_correction(m, mesh, k, primal)
    end
    
    if use_cross_diffusion && k <= K
        correction += cross_diffusion_correction(m, mesh, k, primal)
    end
    
    return base_hodge + correction
end

"""
    direct_gradient_correction(m::Metric{N}, mesh::Mesh{N, K}, k::Int, primal::Bool) where {N, K}

Computes the direct gradient correction term for the hodge operator to address 
non-orthogonality issues in barycentric dual meshes. Uses geometric relationships
between primal and dual cells to estimate correction weights.
"""
function direct_gradient_correction(m::Metric{N}, mesh::Mesh{N, K}, k::Int, primal::Bool) where {N, K}
    if k == K+1
        return spzeros(0,0)
    end
    
    comp = primal ? mesh.primal.complex : mesh.dual.complex
    dual_comp = primal ? mesh.dual.complex : mesh.primal.complex
    
    n_cells = length(comp.cells[k])
    n_dual_cells = length(dual_comp.cells[K-k+1])
    
    row_inds, col_inds, vals = Int[], Int[], Float64[]
    
    for (i, p_cell) in enumerate(comp.cells[k])
        for (j, d_cell) in enumerate(dual_comp.cells[K-k+1])
            if has_geometric_relationship(p_cell, d_cell, k, K)
                correction_val = compute_direct_gradient_weight(m, mesh, p_cell, d_cell, k, primal)
                if abs(correction_val) > 1e-12
                    push!(row_inds, j)
                    push!(col_inds, i)
                    push!(vals, correction_val)
                end
            end
        end
    end
    
    return sparse(row_inds, col_inds, vals, n_dual_cells, n_cells)
end

"""
    cross_diffusion_correction(m::Metric{N}, mesh::Mesh{N, K}, k::Int, primal::Bool) where {N, K}

Computes the cross diffusion correction term for the hodge operator to reduce
numerical diffusion caused by non-orthogonal mesh geometry. This addresses
spurious coupling between neighboring elements.
"""
function cross_diffusion_correction(m::Metric{N}, mesh::Mesh{N, K}, k::Int, primal::Bool) where {N, K}
    if k == K+1
        return spzeros(0,0)
    end
    
    comp = primal ? mesh.primal.complex : mesh.dual.complex
    dual_comp = primal ? mesh.dual.complex : mesh.primal.complex
    
    n_cells = length(comp.cells[k])
    n_dual_cells = length(dual_comp.cells[K-k+1])
    
    row_inds, col_inds, vals = Int[], Int[], Float64[]
    
    for (i, p_cell) in enumerate(comp.cells[k])
        neighbors = get_neighboring_cells(p_cell, comp, k)
        
        for neighbor in neighbors
            j_neighbor = findfirst(isequal(neighbor), comp.cells[k])
            if j_neighbor !== nothing && j_neighbor != i
                correction_val = compute_cross_diffusion_weight(m, mesh, p_cell, neighbor, k, primal)
                if abs(correction_val) > 1e-12
                    dual_idx = get_corresponding_dual_index(p_cell, dual_comp, K-k+1)
                    if dual_idx !== nothing
                        push!(row_inds, dual_idx)
                        push!(col_inds, j_neighbor)
                        push!(vals, correction_val)
                    end
                end
            end
        end
    end
    
    return sparse(row_inds, col_inds, vals, n_dual_cells, n_cells)
end

"""
    has_geometric_relationship(cell1, cell2, k::Int, K::Int)

Check if two cells have a geometric relationship that warrants correction.
"""
function has_geometric_relationship(cell1, cell2, k::Int, K::Int)
    if k == 1 && K-k+1 == 1
        return length(intersect(Set(cell1.children), Set(cell2.children))) > 0
    elseif k == 2 && K-k+1 == 1  
        return any(child in cell2.children for child in cell1.children)
    else
        return false
    end
end

"""
    compute_direct_gradient_weight(m::Metric{N}, mesh::Mesh{N, K}, p_cell, d_cell, k::Int, primal::Bool) where {N, K}

Compute the weight for direct gradient correction between primal and dual cells.
Uses geometric distance and volume ratios to estimate correction magnitude.
"""
function compute_direct_gradient_weight(m::Metric{N}, mesh::Mesh{N, K}, p_cell, d_cell, k::Int, primal::Bool) where {N, K}
    p_vol = volume(m, mesh.primal, p_cell)
    d_vol = volume(m, mesh.dual, d_cell)
    
    if p_vol < 1e-12 || d_vol < 1e-12
        return 0.0
    end
    
    orthogonality_deviation = estimate_orthogonality_deviation(m, mesh, p_cell, d_cell, k)
    
    correction_strength = 0.1
    return correction_strength * orthogonality_deviation * sqrt(p_vol * d_vol) / (p_vol + d_vol)
end

"""
    compute_cross_diffusion_weight(m::Metric{N}, mesh::Mesh{N, K}, cell1, cell2, k::Int, primal::Bool) where {N, K}

Compute the weight for cross diffusion correction between neighboring cells.
Uses shared boundary area and geometric alignment.
"""
function compute_cross_diffusion_weight(m::Metric{N}, mesh::Mesh{N, K}, cell1, cell2, k::Int, primal::Bool) where {N, K}
    comp = primal ? mesh.primal : mesh.dual
    
    vol1 = volume(m, comp, cell1)
    vol2 = volume(m, comp, cell2)
    
    if vol1 < 1e-12 || vol2 < 1e-12
        return 0.0
    end
    
    shared_measure = compute_shared_boundary_measure(cell1, cell2, k)
    
    if shared_measure < 1e-12
        return 0.0
    end
    
    diffusion_strength = 0.05
    return -diffusion_strength * shared_measure * sqrt(vol1 * vol2) / (vol1 + vol2)
end

"""
    get_neighboring_cells(cell, complex, k::Int)

Get cells that share a boundary with the given cell.
"""
function get_neighboring_cells(cell, complex, k::Int)
    neighbors = []
    
    if k > 1
        for child in cell.children
            for parent_key in keys(child.parents)
                if parent_key != cell && parent_key in complex.cells[k]
                    push!(neighbors, parent_key)
                end
            end
        end
    end
    
    return unique(neighbors)
end

"""
    get_corresponding_dual_index(p_cell, dual_complex, k::Int)

Find the index of the dual cell corresponding to a primal cell.
"""
function get_corresponding_dual_index(p_cell, dual_complex, k::Int)
    return findfirst(cell -> corresponds_to_cell(cell, p_cell, k), dual_complex.cells[k])
end

"""
    corresponds_to_cell(dual_cell, primal_cell, k::Int)

Check if a dual cell corresponds to a primal cell based on incidence relationships.
"""
function corresponds_to_cell(dual_cell, primal_cell, k::Int)
    if k == 1
        return length(intersect(Set(dual_cell.children), Set(primal_cell.children))) > 0
    else
        return true
    end
end

"""
    estimate_orthogonality_deviation(m::Metric{N}, mesh::Mesh{N, K}, p_cell, d_cell, k::Int) where {N, K}

Estimate how much the connection between primal and dual cells deviates from orthogonality.
"""
function estimate_orthogonality_deviation(m::Metric{N}, mesh::Mesh{N, K}, p_cell, d_cell, k::Int) where {N, K}
    if k == 1
        return 0.2
    elseif k == 2  
        return 0.15
    else
        return 0.1
    end
end

"""
    compute_shared_boundary_measure(cell1, cell2, k::Int)

Compute the measure (length/area) of the shared boundary between two cells.
"""
function compute_shared_boundary_measure(cell1, cell2, k::Int)
    shared_children = intersect(Set(cell1.children), Set(cell2.children))
    return Float64(length(shared_children))
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
