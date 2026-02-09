TriangulatedComplex{N,K}() where {N, K} = TriangulatedComplex{N,K}(
    CellComplex{N,K}(), Dict{Cell{N}, Vector{SignedSimpleSimplex{N}}}())

function TriangulatedComplex(simplices::AbstractVector{Simplex{N, K}}) where {N, K}
    # cache to make sure the same simplex isn't turned into a cell twice
    cells = Dict{Set{Point{N}}, Cell{N}}()
    simplex_mapping = Dict{Cell{N}, Vector{SignedSimpleSimplex{N}}}()
    for k in 1:K
        for simplex in simplices
            for s in subsimplices(simplex, k)
                key = Set(s.points)
                if !(key in keys(cells))
                    s_cell = Cell(s)
                    cells[key] = s_cell
                    simplex_mapping[s_cell] = [(SimpleSimplex(s), true)]
                    for face in subsimplices(s, k-1)
                        face_cell = cells[Set(face.points)]
                        i = findfirst(p -> !(p in face.points), s.points)
                        pm = sign_of_permutation(face.points, face_cell.points)
                        parent!(face_cell, s_cell, pm * (1 - 2 * mod(i-1, 2)) > 0)
                    end
                end
            end
        end
    end
    complex = CellComplex{N,K}(collect(values(cells)))
    return TriangulatedComplex{N,K}(complex, simplex_mapping)
end

import Base: show
show(io::IO, tcomp::TriangulatedComplex{N,K}) where {N, K} = print(
     io, "TriangulatedComplex{$N,$K}$(tuple(map(length, tcomp.complex.cells)...))")

Mesh(tcomp::TriangulatedComplex{N}, center::Function) where N =
    Mesh(tcomp, dual(tcomp.complex, center))

show(io::IO, m::Mesh{N,K}) where {N, K} = print(io, "Mesh{$N,$K}($(m.primal), $(m.dual))")

export signed_volume
"""
    signed_volume(m::Metric{N}, tcomp::TriangulatedComplex{N}, c::Cell{N}) where N

Find the signed volume of a cell in a TriangulatedComplex.
"""
signed_volume(m::Metric{N}, tcomp::TriangulatedComplex{N}, c::Cell{N}) where N =
    sum([volume(m, Simplex(s)) * (2 * b - 1) for (s, b) in tcomp.simplices[c]])

export volume
"""
    signed_volume(m::Metric{N}, tcomp::TriangulatedComplex{N}, c::Cell{N}) where N

Find the absolute value of the signed volume of a cell in a TriangulatedComplex.
"""
volume(m::Metric{N}, tcomp::TriangulatedComplex{N}, c::Cell{N}) where N =
    abs(signed_volume(m, tcomp, c))

SimpleBarySimplex{N} = Vector{SimpleBarycentric{N}}

"""
    elementary_duals!(simplices::Dict{Cell{N}, Vector{Tuple{SimpleBarySimplex{N}, Bool}}},
        center::Function, c::Cell{N}) where N

Compute the elementary dual simplices of `cell`. `simplices` is a mapping from primal cells
to dual elementary simplices, specified as Barycentrics along with signs, and is used for
memoization. `center` is a function that takes a `Simplex{N, K}` to a `Barycentric{N, K}`.
"""
function elementary_duals!(simplices::Dict{Cell{N}, Vector{Tuple{SimpleBarySimplex{N}, Bool}}},
    center::Function, c::Cell{N}) where N
    if !(c in keys(simplices))
        c_center = SimpleBarycentric(center(Simplex(c)))
        simplices[c] = Tuple{SimpleBarySimplex{N}, Bool}[]
        if isempty(c.parents)
            push!(simplices[c], ([c_center], true))
        end
        for p in keys(c.parents)
            for (ps, sign) in elementary_duals!(simplices, center, p)
                opposite_index = first_setdiff_index(ps[1].simplex.points, c.points)
                new_sign = (ps[1].coords[opposite_index] >= 0) == sign
                new_barycentrics = [c_center, ps...]
                push!(simplices[c], (new_barycentrics, new_sign))
            end
        end
    end
    return simplices[c]
end

"""非再帰版: 単一セルのelementary dualを計算（親は既に計算済み前提）"""
function compute_elementary_dual_nonrecursive(simplices_vec::Vector{Vector{Tuple{SimpleBarySimplex{N}, Bool}}},
    all_centers::Vector{SimpleBarycentric{N}}, cell_idx::Int, c::Cell{N},
    cell_to_idx::Dict{Cell{N}, Int}) where N
    c_center = all_centers[cell_idx]
    result = Tuple{SimpleBarySimplex{N}, Bool}[]
    
    if isempty(c.parents)
        push!(result, ([c_center], true))
    else
        for p in keys(c.parents)
            parent_idx = cell_to_idx[p]
            parent_duals = simplices_vec[parent_idx]
            for (ps, sign) in parent_duals
                opposite_index = first_setdiff_index(ps[1].simplex.points, c.points)
                new_sign = (ps[1].coords[opposite_index] >= 0) == sign
                new_barycentrics = [c_center, ps...]
                push!(result, (new_barycentrics, new_sign))
            end
        end
    end
    return result
end

export dual
"""
    dual(primal::CellComplex{N, K}, center::Function) where {N, K}

Compute the dual TriangulatedComplex of a simplicial complex. `center` is a function that
takes a `Simplex{N, K}` to a `Barycentric{N, K}`.
"""
function dual(primal::CellComplex{N, K}, center::Function) where {N, K}
    # 総セル数を推定
    total_cells = sum(length(primal.cells[k]) for k in 1:K)
    
    dual_tcomp = TriangulatedComplex{N, K}()
    primal_to_duals = Dict{Cell{N}, Cell{N}}()
    sizehint!(primal_to_duals, total_cells)
    
    # Phase 0: 全セルの外心を事前に並列計算 + セルインデックス構築
    cell_to_idx = Dict{Cell{N}, Int}()
    sizehint!(cell_to_idx, total_cells)
    all_centers = Vector{SimpleBarycentric{N}}(undef, total_cells)
    all_cells = Vector{Cell{N}}(undef, total_cells)  # インデックス→セル
    
    offset = 0
    for k in 1:K
        cells_k = primal.cells[k]
        ncells = length(cells_k)
        
        Threads.@threads for i in 1:ncells
            @inbounds all_centers[offset + i] = SimpleBarycentric(center(Simplex(cells_k[i])))
        end
        
        for i in 1:ncells
            idx = offset + i
            @inbounds cell_to_idx[cells_k[i]] = idx
            @inbounds all_cells[idx] = cells_k[i]
        end
        offset += ncells
    end
    
    # elementary duals用ベクトル（辞書の代わり）
    simplices_vec = Vector{Vector{Tuple{SimpleBarySimplex{N}, Bool}}}(undef, total_cells)
    
    # iterate from high to low dimension so the cells of
    # the dual are constructed in order of dimension
    for k in reverse(1:K)
        cells_k = primal.cells[k]
        ncells = length(cells_k)
        
        # Phase 1: elementary duals を並列計算（親は既に計算済み）
        # 同じ次元のセル同士は独立、親は高次元なので既に処理済み
        elem_results = Vector{Vector{Tuple{SimpleBarySimplex{N}, Bool}}}(undef, ncells)
        Threads.@threads for i in 1:ncells
            @inbounds begin
                cell = cells_k[i]
                cidx = cell_to_idx[cell]
                elem_results[i] = compute_elementary_dual_nonrecursive(
                    simplices_vec, all_centers, cidx, cell, cell_to_idx)
            end
        end
        # 結果をsimplices_vecに格納
        for i in 1:ncells
            @inbounds simplices_vec[cell_to_idx[cells_k[i]]] = elem_results[i]
        end
        
        # Phase 2: dual cell 作成を並列計算（ベクトル参照）
        dual_cells = Vector{Cell{N}}(undef, ncells)
        signed_duals = Vector{Vector{Tuple{SimpleSimplex{N}, Bool}}}(undef, ncells)
        Threads.@threads for i in 1:ncells
            @inbounds begin
                cidx = cell_to_idx[cells_k[i]]
                elem_duals = simplices_vec[cidx]
                n_elem = length(elem_duals)
                signed_elementary_duals = Vector{Tuple{SimpleSimplex{N}, Bool}}(undef, n_elem)
                points_set = Set{Point{N}}()
                for j in 1:n_elem
                    bary_simplex = elem_duals[j][1]
                    n_pts = length(bary_simplex)
                    pts = Vector{Point{N}}(undef, n_pts)
                    for idx in 1:n_pts
                        pt = Point(bary_simplex[idx])
                        pts[idx] = pt
                        push!(points_set, pt)
                    end
                    signed_elementary_duals[j] = (SimpleSimplex{N}(pts), elem_duals[j][2])
                end
                dual_cells[i] = Cell(collect(points_set), K-k+1)
                signed_duals[i] = signed_elementary_duals
            end
        end
        
        # Phase 3: 順次処理（依存関係あり）
        for i in 1:ncells
            cell = cells_k[i]
            dual_cell = dual_cells[i]
            push!(dual_tcomp.complex, dual_cell)
            dual_tcomp.simplices[dual_cell] = signed_duals[i]
            primal_to_duals[cell] = dual_cell
            for parent in keys(cell.parents)
                o = cell.parents[parent]
                dual_child = primal_to_duals[parent]
                parent!(dual_child, dual_cell, k == 1 ? !o : o)
            end
        end
    end
    return dual_tcomp
end

"""
    dual(mesh::Mesh{N, K}, c::Cell{N}) where {N, K}

Find the dual cell of `c` in `mesh`.
"""
function dual(mesh::Mesh{N, K}, c::Cell{N}) where {N, K}
    primal = mesh.primal.complex
    dual = mesh.dual.complex
    comp1, comp2 = (c in primal.cells[c.K]) ? (primal, dual) : (dual, primal)
    i = findfirst(isequal(c), comp1.cells[c.K])
    return comp2.cells[K-c.K+1][i]
end


export dual2
"""
    dual2(primal::CellComplex{N, K}, center::Function) where {N, K}

Compute the dual TriangulatedComplex of a simplicial complex. center is a function that
takes a Simplex{N, K} to a Barycentric{N, K}.

Returns:
  - dual_tcomp: The dual TriangulatedComplex.
  - primal_to_duals: A dictionary mapping each primal Cell to its corresponding dual Cell.
"""
function dual2(primal::CellComplex{N, K}, center::Function) where {N, K}
    dual_tcomp = TriangulatedComplex{N, K}()  # Empty dual triangulated complex
    primal_to_elementary_duals = Dict{Cell{N}, Vector{Tuple{SimpleBarySimplex{N}, Bool}}}()
    primal_to_duals = Dict{Cell{N}, Cell{N}}()  # Primal -> Dual mapping

    # Iterate from high dimensions to low dimensions
    for k in reverse(1:K)
        for cell in primal.cells[k]
            # 1) Compute elementary dual simplices for the current primal cell
            elementary_duals = elementary_duals!(primal_to_elementary_duals, center, cell)
            signed_elementary_duals = [(SimpleSimplex(p[1]), p[2]) for p in elementary_duals]

            # 2) Extract unique points for the dual cell
            points = unique(vcat([p[1].points for p in signed_elementary_duals]...))
            dual_cell = Cell(points, K-k+1)  # Create the dual cell with the correct dimension

            # 3) Add the dual cell to the dual triangulated complex
            push!(dual_tcomp.complex, dual_cell)
            dual_tcomp.simplices[dual_cell] = signed_elementary_duals

            # 4) Record the mapping from the primal cell to its dual cell
            primal_to_duals[cell] = dual_cell

            # 5) Connect dual cells with parent relationships
            for parent in keys(cell.parents)
                o = cell.parents[parent]
                dual_child = primal_to_duals[parent]
                # Ensure the dual mesh is positively oriented
                parent!(dual_child, dual_cell, k == 1 ? !o : o)
            end
        end
    end

    # Return both the dual triangulated complex and the primal-to-dual mapping
    return dual_tcomp, primal_to_duals
end
