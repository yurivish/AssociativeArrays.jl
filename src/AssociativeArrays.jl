module AssociativeArrays

using Transducers, SplitApplyCombine, Base.Iterators, ArgCheck
export Assoc

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
    # Use `named=true` to force a named array return value when indexing with basic indices.
    # Note that "named" means to treat it like a named indexing operation, not that it treats your inputs as names.
    # E.g. a[1, named=true] will return a 2d assoc rather than a scalar.
    named_indexing = if ismissing(named)
        # The goal here is to check whether any indices are names.
        # Edge cases: non-basic indexes like Not
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
        Tuple(dim > N ? i : descalarize(i) for (dim, i) in enumerate(I′))
    else @assert len < N
        # More dimensions than indices; pad to N with singleton arrays.
        singleton = [1]
        Tuple(
            if dim > len
                # This means you tried to partially index into an array that has non-singleton trailing dimensions.
                @assert isone(size(A, dim)) "Size in each trailing dimension must be 1."
                singleton
            else
                descalarize(I′[dim])
            end
            for dim in 1:N
        )
    end

    @assert length(I′′) >= N "There should be at least as many (nonscalar) indices as array dimensions"
    value = default_named_getindex(A, I′′)
    M = length(I′′)

    unparameterized(A)(
        if N == 0 && !(value isa AbstractArray)
            # Handle the case of zero-dimensional array indexing; Julia has some
            # inconsistent behaviors that may result in a scalar when an array is expected.
            fill!(similar(Td), value)
        else
            value
        end,
        ntuple(i -> A.axes[i].names[I′′[i]], ndims(value)) # i > M ? () :
    )# |> condense # todo: efficiency
end

end