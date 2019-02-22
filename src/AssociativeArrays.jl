module AssociativeArrays

# Note: We don't currently use any of SparseArrays.
using SparseArrays, Transducers, SplitApplyCombine, Base.Iterators, ArgCheck
export Assoc, NamedAxis

abstract type AbstractNamedArray{T, N, Td} <: AbstractArray{T, N} end
const ANA = AbstractNamedArray

# Base array methods
Base.size(A::ANA) = size(data(A))
Base.size(A::ANA, dim) = size(data(A), dim)

Base.axes(A::ANA) = axes(data(A))
Base.axes(A::ANA, dim) = axes(data(A), dim)

Base.eltype(A::ANA) = eltype(data(A))
Base.length(A::ANA) = length(data(A))
Base.ndims(A::ANA) = ndims(data(A))

Base.IndexStyle(A::ANA) = IndexStyle(data(A))
Base.IndexStyle(::Type{<:ANA{<:Any, <:Any, Td}}) where {Td} = IndexStyle(Td)

# low-level indexing
Base.getindex(A::ANA, i::Int) = data(A)[i]
Base.getindex(A::ANA{T, N}, I::Vararg{Int, N}) where {T, N} = data(A)[I...]

Base.setindex!(A::ANA, v, i::Int) = setindex!(data(A), v, i)
Base.setindex!(A::ANA{T, N}, v, I::Vararg{Int, N}) where {T, N} = setindex!(data(A), v, I...)

Base.similar(A::ANA) = similar(data(A))
Base.similar(A::ANA, ::Type{S}) where {S} = similar(data(A), S)
Base.similar(A::ANA, ::Type{S}, dims::Dims) where {S} = similar(data(A), S, dims)

# Generic named array convenience constructor for eg. Assoc(data, names_1, names_2, ...)
(::Type{Ta})(data::AbstractArray{T, N}, names::Vararg{AbstractArray, N}) where {Ta <: ANA, T, N} = Ta(data, names)

function named_to_indices(A::ANA{T, N}, ax, I) where {T, N}
    dim = N - length(ax) + 1
    # @show ax I dim
    if N == 1 && length(ax) == 1
        @argcheck(
            length(ax[1]) == prod(size(A)),
            BoundsError("Named linear indexing into an $(N)-d Assoc is not supported.", I)
        )
    end
    to_indices(A, ax, (name_to_index(A, dim, I[1]), Base.tail(I)...))
end

function default_named_getindex(A::ANA{T, N}, I′) where {T, N}
    nd = ndims.(I′)

    # Function for basic argument validation during a named indexing operation.
    # Intended to be called from within user-defined named_getindex methods.
    # (I′ is our convention for "lowered" basic index values)
    @argcheck(
        all(x -> x == 0 || x == 1, nd),
        BoundsError("Multidimensional indexing within a single dimension is not currently supported.", I′)
    )

    @argcheck(
        all(i <= N || n == 0 for (i, n) in enumerate(nd)),
        BoundsError("Trailing indices may not introduce new dimensions because it is not clear what the new names would be.", I′)
    )

    data(A)[I′...]
end

function Base.getindex(A::ANA{T, N}, I...; named=missing) where {T, N}
    # Use `named=true` to force a named_getindex call even with all basic indices .
    # E.g. a[1, named=true] will return a 2d assoc rather than a scalar.
    named_indexing = if ismissing(named)
        # Check whether any indices are intended as names.
        any(
            if I[dim] isa AbstractArray
                any(name -> isnametype(A, dim, name), I[dim])
            else
                isnametype(A, dim, I[dim])
            end
            # Iterate through 1:N since trailing indices can't possibly be named
            for dim in 1:min(length(I), N)
        )
    else
        named::Bool
    end

    I′ = to_indices(A, I)
    if named_indexing
        named_getindex(A, I′)
    else
        data(A)[I′...]
    end
end

include("namedaxis.jl")

struct Assoc{T, N, Td} <: AbstractNamedArray{T, N, Td}
    data::Td
    axes::NTuple{N, NamedAxis}
    function Assoc(data::AbstractArray{T, N}, names::Tuple{Vararg{AbstractVector, N}}) where {T, N}
        @argcheck all(allunique.(names)) "Names must be unique within each dimension."
        @argcheck ndims(data) == length(names) "Each data dimension must be named."
        @argcheck all(size(data) .== length.(names)) "Data and name dimensions must be equal: $(size(data)) $(length.(names))."
        @argcheck all(axes(data) .== axes.(names, 1)) "Names must have the same axis as the corresponding data axis."
        @argcheck !any(T <: Union{Int, AbstractArray, Symbol} for T in eltype.(names)) "Names cannot be of type Int, AbstractArray, or Symbol."
        new{T, N, typeof(data)}(data, NamedAxis.(names))
    end
end

data(A::Assoc) = A.data
unparameterized(::Assoc) = Assoc

macro define_named_to_indices(A, T)
    quote
        Base.to_indices(A::$A, ax, I::Tuple{Union{$T, AbstractArray{<:$T}}, Vararg{Any}}) =
            named_to_indices(A, ax, I)
    end
end

# Allow Symbol here since indexing with one is allowed, though it is not allowed to be a name
const assoc_indices = Union{String, Char, Symbol, Pair}
@define_named_to_indices Assoc assoc_indices

# Count Symbols as a name type so that they pass through to named_getindex
isnametype(A::Assoc, dim, name::assoc_indices) = true
isnametype(A::Assoc, dim, name) = false

name_to_index(A::Assoc, dim, I) = let axis = A.axes[dim]
    isnamedindex(axis, I) ? toindices(axis, I) : []
end
name_to_index(A::Assoc, dim, I::AbstractArray) = toindices(A.axes[dim], I)

descalarize(x::AbstractArray) = x
descalarize(x) = [x]

function named_getindex(A::Assoc{T, N, Td}, I′) where {T, N, Td}
    # Lift scalar indices to arrays so that the result of indexing matches
    # the dimensionality of A. We iterate rather than broadcast to avoid
    # descalarizing trailing dimensions.
    len = length(I′)
    I′′ = if len == N
        descalarize.(I′)
    elseif len > N
        # More indices than dimensions; only lift those <= N
        ntuple(dim -> dim > N ? I′[dim] : descalarize(I′[dim]), length(I′))
    else @assert len < N
        # More dimensions than indices; pad to N with singleton arrays.
        singleton = [1]
        ntuple(
            dim -> if dim > len
                # This means you tried to partially index into an array that has non-singleton trailing dimensions.
                @assert isone(size(A, dim)) "Size in each trailing dimension must be 1."
                singleton
            else
                descalarize(I′[dim])
            end,
            N
        )
    end

    @assert length(I′′) >= N "There should be at least as many (nonscalar) indices as array dimensions"
    value = let value = default_named_getindex(A, I′′)
        # Handle zero-dimensional array indexing; Some zero-dimensional
        # indexing behaviors result in a scalar when an array is expected.
        # Ensure that the value is always an array.
        if N == 0 && !(value isa AbstractArray)
            fill!(similar(Td), value)
        else
            value
        end
    end

    I_condensed = condense_indices(value)
    samesize = all(==(:), I_condensed)

    condensed_value = samesize ? value : value[I_condensed...] #  ? value :
    unparameterized(A)(
        condensed_value,
        ntuple(i -> A.axes[i].names[samesize ? I′′[i] : I′′[i][I_condensed[i]]], ndims(condensed_value))
    )
end

function condense_indices(a::AbstractArray{<:Any, 0})
    I = findall(!iszero, a)
    @assert length(I) <= 1
    # Return a value such that a[val...] preserves dimensionality
    # i.e. does not introduce a dimension
    (isempty(I) ? I : fill(first(I)),)
end

function condense_indices(a::AbstractVector)
    I = findall(!iszero, a)
    rows = sort!(unique(i[1] for i in I))
    (length(rows) == size(a, 1) ? (:) : rows,)
end

function condense_indices(a::AbstractMatrix)
    I = findall(!iszero, a)
    rows = sort!(unique(i[1] for i in I))
    cols = sort!(unique(i[2] for i in I))
    (length(rows) == size(a, 1) ? (:) : rows, length(cols) == size(a, 2) ? (:) : cols)
end

#=
function elementwise_mul_like(A::Assoc2D, B::Assoc2D, *)
    z = zero(eltype(A)) * zero(eltype(B)) # Infer element type by doing a zero-zero test for now.
    T = typeof(z) # promote_type(eltype(A), eltype(B))
    @assert iszero(z) "*(0, 0) == 0 must hold for multiplication-like operators."
    # Note: These fail for eg. 1/0 --> Inf.
    # @assert iszero(zero(T) * one(T)) "*(0, 1) == 0 must hold for multiplication-like operators."
    # @assert iszero(one(T) * zero(T)) "*(1, 0) == 0 must hold for multiplication-like operators."

    na1, nb1, k1 = intersect_names(A, B, 1)
    na2, nb2, k2 = intersect_names(A, B, 2)

    C = Assoc(spzeros(T, length(k1), length(k2)), k1, k2)
    if prod(size(C)) > 0 # check for the case of an empty result
        # without the check above, these fail for mysterious indexing error reasons...
        C[k1, k2] .+= A[k1, k2, named=false]
        C[k1, k2] .*= B[k1, k2, named=false]
    end

    condense(C)
end

function elementwise_add_like(A::Assoc2D, B::Assoc2D, +)
    T = promote_type(eltype(A), eltype(B))
    @assert iszero(zero(T) + zero(T)) "+(0, 0) == 0 must hold for addition-like operators."
    @assert isone(zero(T) + one(T)) "+(0, 1) == 1 must hold for addition-like operators."
    @assert isone(one(T) + zero(T)) "+(1, 0) == 1 must hold for addition-like operators."

    na1, nb1, k1 = union_names(A, B, 1)
    na2, nb2, k2 = union_names(A, B, 2)

    C = Assoc(spzeros(T, length(k1), length(k2)), k1, k2)
    # todo: views
    @show size(k1) size(k2)
    if !isempty(na1) || !isempty(na2)
        @show size(na1) size(na2) size(C[na1, na2])
        # C[na1, na2, named=false] .+= data(A)
        let inds = to_indices(C, (na1, na2))
            data(C)[inds...] .+= data(A)
        end
    end
    if !isempty(nb1) || !isempty(nb2)
        # C[nb1, nb2, named=false] .+= data(B)
        let inds = to_indices(C, (nb1, nb2))
            data(C)[inds...] .+= data(B)
        end
    end

    condense(C)
end
=#

end # module