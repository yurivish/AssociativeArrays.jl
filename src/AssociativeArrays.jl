module AssociativeArrays

using ArgCheck, Base.Iterators, LinearAlgebra, SparseArrays, SplitApplyCombine,
      OrderedCollections, PrettyTables
using Base: tail

export Assoc, Num, Id
export explode, triples, densify

abstract type AbstractNamedArray{T, N, Td} <: AbstractArray{T, N} end
const ANA = AbstractNamedArray

Base.size(A::ANA) = size(data(A))
Base.axes(A::ANA) = axes(data(A))

# Base array methods
# todo: efficient hash functions?

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

# We cannot know in general whether dimension names are preserved -- even
# if the similar array is of the same size, names may move around.
# So always return a "similar" array of the underlying array type.
Base.similar(A::ANA) = similar(data(A))
Base.similar(A::ANA, ::Type{S}) where {S} = similar(data(A), S)
Base.similar(A::ANA, ::Type{S}, dims::Dims) where {S} = similar(data(A), S, dims)

# If we wanted to be adventurous:
# This makes the default wrong in some cases, but right in the majority of cases.
# Base.similar(A::ANA) = unparameterized(A)(similar(data(A)), names(A))
# Base.similar(A::ANA, ::Type{S}) where {S} = unparameterized(A)(similar(data(A), S), names(A))

# Note – AxisArrays defines further specializations for Axis types:
# https://github.com/JuliaArrays/AxisArrays.jl/blob/48ec7350e3a8669dc17ef2e2f34069d86c227975/src/core.jl#L303

function named_to_indices(A::ANA{T, N}, ax, I) where {T, N}
    dim = N - length(ax) + 1
#     if length(I) < N then all of size(A, N-dim:N) must == 1
    # todo: revisit this condition; indexing with any number of
    # indices is valid if the array is size 1 in all trailing dimensions

    @boundscheck begin
        if isempty(ax) || size(A, dim) != length(ax[1]) # <- unclear... why the isempty check?
            # Disallow linear indexing with a single name
            throw(BoundsError("Named indexing expects one name per dimension.", I)) # todo: identify the specific index
        end
    end

    to_indices(A, ax, (name_to_index(A, dim, I[1]), tail(I)...))
end

struct Id{T}
    id::T
end

name_missing(A, dim, i) = throw(ArgumentError("Missing name $i for dimension $dim."))

function name_to_index(A, dim, i)
    get(name_to_index(A, dim), i) do
        name_missing(A, dim, i)
    end
end
# name_to_index(A, dim, i::Id) = name_to_index(A, dim, i.name)
name_to_index(A, dim, I::AbstractArray) = [name_to_index(A, dim, i) for i in I]

function default_named_getindex(A::ANA{T, N}, I′) where {T, N}
    nd = ndims.(I′)

    # TODO: Improve these error messages; specify the index/dimension
    @argcheck(
        all(x -> x == 0 || x == 1, nd),
        BoundsError("Multidimensional indexing within a single dimension is not supported.", I′)
    )

    @argcheck(
        all(i <= N || n == 0 for (i, n) in enumerate(nd)),
        BoundsError("Trailing indices may not introduce new dimensions.", I′)
    )

    # The length may be greater than N in the case of trailing singleton indices.
    # It may be less than one in the case of zero-dimensional indexing into a 1x1x... array.
    @assert length(I′) <= 1 || length(I′) >= N

    data(A)[I′...]
end

named_getindex(A::ANA, I′) = default_named_getindex(A, I′)

isarray(x) = false
isarray(x::AbstractArray) = true

# hasname(A::ANA, dim, name) = haskey(name_to_index(A, dim), name)
ofnametype(A::ANA, dim, name) = name isa keytype(name_to_index(A, dim))

# Fast inference path for common basic indexing patterns
fast_path_scalars = Union{Int, CartesianIndex, Colon}
fast_path_arrays = Union{AbstractVector{<:Union{fast_path_scalars, Bool}}}
Base.getindex(A::ANA, I::Union{fast_path_scalars, fast_path_arrays, AbstractVector{<:fast_path_scalars}}...) =
    data(A)[I...]

function Base.getindex(A::ANA{T, N}, I...; named=missing) where {T, N}
    # Use the `named=true` kwarg to force a named return value (e.g. an Assoc) with basic indices
    # Note that "named" means to treat it like a named indexing operation — not that it treats your inputs as names.
    # E.g. a[1, named=true] will return a 2d assoc rather than a scalar.
    named_indexing = if ismissing(named)
        any(
            if isarray(I[dim])
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

const native_indices = Union{Int, AbstractArray}

# todo: determine whether this causes dynamic dispatch
getnames(A::ANA{T, N}, I::Tuple{}) where {T, N} = ()
getnames(A::ANA{T, N}, I::Tuple{Vararg{native_indices}}) where {T, N} =
    Tuple(Iterators.flatten(Iterators.repeated(names(A)[i][I[i]], ndims(I[i])) for i in 1:N))

Base.names(A::ANA, args...) = names(A, args...)

names(A::ANA, dim) = names(A)[dim]

# These methods should be defined by subtypes:
# names(A)::Tuple
# data(A)::Array{T, N}
# name_to_index(A, dim)::Dict
# unparameterized(::Arr) = Arr
# ^— We expect all subtypes to implement Arr(data, names) and this function finds the constructor.

macro define_named_to_indices(A, T)
    quote
        Base.to_indices(A::$A, ax, I::Tuple{Union{$T, AbstractArray{<:$T}}, Vararg{Any}}) =
            named_to_indices(A, ax, I)
    end
end

@define_named_to_indices ANA Id

function argcheck_constructor(data, names)
    @argcheck all(allunique.(names)) "Names must be unique within each dimension."
    @argcheck ndims(data) == length(names) "Each data dimension must be named."
    @argcheck all(size(data) .== length.(names)) "Data and name dimensions must be equal: $(size(data)) $(length.(names))."
    @argcheck all(axes(data) .== axes.(names, 1)) "Names must have the same axis as the corresponding data axis."
    @argcheck !any(T <: native_indices for T in eltype.(names)) "Names cannot be of type Int or AbstractArray."
end

# Operate on the underlying data of an assoc for e.g. scalar broadcasting,
# which is not defined for an assoc.
withdata(f, A::ANA) = unparameterized(A)(f(A), names(A))
withdata!(f!, A::ANA) = (f!(data(A)); A)

densify(A::ANA) = withdata(Array, A)

##

include("broadcast.jl")

include("namedarray.jl")
include("assoc.jl")
include("num.jl")

end # module

