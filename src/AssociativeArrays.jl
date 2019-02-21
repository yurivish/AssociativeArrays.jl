module AssociativeArrays

using ArgCheck, LinearAlgebra, SparseArrays, Base.Iterators
using SplitApplyCombine, OrderedCollections, PrettyTables
using Base: tail

abstract type AbstractNamedArray{T, N, Td} <: AbstractArray{T, N} end
const ANA = AbstractNamedArray

# Generic convenience constructor for named arrays.
# ex. Assoc(data, names_1, names_2, ...)
(::Type{Ta})(data::AbstractArray{T, N}, names::Vararg{AbstractArray, N}) where {Ta <: ANA, T, N} = Ta(data, names)

# Base array methods

Base.size(A::ANA) = size(data(A))
Base.axes(A::ANA) = axes(data(A))

Base.size(A::ANA, dim) = size(data(A), dim)
Base.axes(A::ANA, dim) = axes(data(A), dim)

Base.eltype(A::ANA) = eltype(data(A))
Base.length(A::ANA) = length(data(A))
Base.ndims(A::ANA) = ndims(data(A))
Base.IndexStyle(::Type{<:ANA{<:Any, <:Any, Td}}) where {Td} = IndexStyle(Td)

# low-level indexing
Base.getindex(A::ANA, i::Int) = data(A)[i]
Base.getindex(A::ANA{T, N}, I::Vararg{Int, N}) where {T, N} = data(A)[I...]

Base.setindex!(A::ANA, v, i::Int) = setindex!(data(A), v, i)
Base.setindex!(A::ANA{T, N}, v, I::Vararg{Int, N}) where {T, N} = setindex!(data(A), v, I...)


# We cannot in general know whether dimension names are preserved,
# so we don't propagate names through `similar`.

Base.similar(A::ANA) = similar(data(A))
Base.similar(A::ANA, ::Type{S}) where {S} = similar(data(A), S)
Base.similar(A::ANA, ::Type{S}, dims::Dims) where {S} = similar(data(A), S, dims)

function named_to_indices(A::ANA{T, N}, ax, I) where {T, N}
    dim = N - length(ax) + 1
    if length(ax) == 1
        @argcheck(
            length(first(ax)) != prod(size(A)),
            BoundsError("Named linear indexing into an $(N)-dimensional named array is not supported " *
                "because it is not a meaningful operation.", I)
        )
    end

    to_indices(A, ax, (name_to_index(A, dim, I[1]), tail(I)...))
end

# Do not require `name_to_index(A, dim)` and leave the handling of missing names
# up to the specific implementations of `name_to_index(A, dim, i)`.
name_to_index(A, dim, I::AbstractArray) = [name_to_index(A, dim, i) for i in I]

# named_getindex(A::ANA, I′) = default_named_getindex(A, I′)

# Base.names(A::ANA, args...) = names(A, args...)
# names(A::ANA, dim) = names(A)[dim]
# names(A::ANA, dim, I) = names(A)[dim][I]

const native_indices = Union{Int, AbstractArray}

function argcheck_constructor(data, names)
    @argcheck all(allunique.(names)) "Names must be unique within each dimension."
    @argcheck ndims(data) == length(names) "Each data dimension must be named."
    @argcheck all(size(data) .== length.(names)) "Data and name dimensions must be equal: $(size(data)) $(length.(names))."
    @argcheck all(axes(data) .== axes.(names, 1)) "Names must have the same axis as the corresponding data axis."
    @argcheck !any(T <: native_indices for T in eltype.(names)) "Names cannot be of type Int or AbstractArray."
end

function argcheck_named_indexing(N, I′, nd=ndims.(I′))
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
end

function Base.getindex(A::ANA{T, N}, I...; named=missing) where {T, N}
    # Use `named=true` to force a named array return value when indexing with basic indices.
    # Note that "named" means to treat it like a named indexing operation, not that it treats your inputs as names.
    # E.g. a[1, named=true] will return a 2d assoc rather than a scalar.
    named_indexing = if ismissing(named)
        # The goal here is to check whether any indices are names.
        # Edge cases: non-basic indexes like Not
        any(
            if I[dim] isa AbstractArray
                any(name -> ofnametype(A, dim, name), I[dim])
            else
                ofnametype(A, dim, I[dim])
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