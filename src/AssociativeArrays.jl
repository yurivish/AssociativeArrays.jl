module AssociativeArrays

using ArgCheck, Base.Iterators, LinearAlgebra, SparseArrays, SplitApplyCombine,
      OrderedCollections, PrettyTables
using Base: tail

export Assoc, Num, Id
export explode, triples, densify, withdata

abstract type AbstractNamedArray{T, N, Td} <: AbstractArray{T, N} end
const ANA = AbstractNamedArray

# Generic named array convenience constructor for eg. Assoc(data, names_1, names_2, ...)
(::Type{Ta})(data::AbstractArray{T, N}, names::Vararg{AbstractArray, N}) where {Ta <: ANA, T, N} = Ta(data, names)

Base.size(A::ANA) = size(data(A))
Base.axes(A::ANA) = axes(data(A))

# Base array methods

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

# Note – AxisArrays defines further specializations for Axis types:
# https://github.com/JuliaArrays/AxisArrays.jl/blob/48ec7350e3a8669dc17ef2e2f34069d86c227975/src/core.jl#L303

function named_to_indices(A::ANA{T, N}, ax, I) where {T, N}
    dim = N - length(ax) + 1
    @argcheck(
        !(N > 1 && length(ax) == 1 && length(ax[1]) == length(A)),
        BoundsError("Named linear indexing into an $(N)-d Assoc is not supported.", I)
    )
    to_indices(A, ax, (name_to_index(A, dim, I[1]), tail(I)...))
end

name_missing(A, dim, i) = throw(ArgumentError("Missing name $i for dimension $dim."))

function name_to_index(A, dim, i)
    get(name_to_index(A, dim), i) do
        name_missing(A, dim, i)
    end
end
name_to_index(A, dim, I::AbstractArray) = [name_to_index(A, dim, i) for i in I]

function default_named_getindex(A::ANA{T, N}, I′) where {T, N}
    nd = ndims.(I′)

    @argcheck(
        all(x -> x == 0 || x == 1, nd),
        BoundsError("Multidimensional indexing within a single dimension is not supported.", I′)
    )

    @argcheck(
        all(i <= N || n == 0 for (i, n) in enumerate(nd)),
        BoundsError("Trailing indices may not introduce new dimensions.", I′)
    )

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
fast_path_indices = Union{fast_path_scalars, fast_path_arrays, AbstractVector{<:fast_path_scalars}}
Base.getindex(A::ANA, I::fast_path_indices...) =
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

Base.view(A::ANA, I::fast_path_indices...) = view(data(A), I...)

const native_indices = Union{Int, AbstractArray}

getnames(A::ANA{T, N}, I::Tuple{Vararg{native_indices, M}}) where {T, N, M} =
    # The ternary condition handles cases with implicit trailing indices.
    Tuple(Iterators.flatten(i > M ? () : Iterators.repeated(names(A, i)[I[i]], ndims(I[i])) for i in 1:N))

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

# Introduce a simple `Id` name type for named indexing with integers
struct Id{T}
    id::T
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
withdata(f, A::ANA) = unparameterized(A)(f(data(A)), names(A))
withdata!(f!, A::ANA) = (f!(data(A)); A)

densify(A::ANA) = withdata(Array, A)

Base.cumsum(A::ANA, args...; kw...) = withdata(x -> cumsum(x, args...; kw...), A)
Base.cumsum!(A::ANA, args...; kw...) = withdata!(x -> cumsum(x, args...; kw...), A)

# What to do about names for the dimensions reduced out?
# function Base.sum(A::Assoc, args...; names, dims, kws...)
#     res = sum(data(A), args...; name=name, dims=dims, kw...)
#     # Assoc(res, Tuple(dims) # names.(Ref(A), size(res))
#     # NOTE: Dims might be e.g :
# end

##

include("broadcast.jl")

include("namedarray.jl")
include("assoc.jl")
include("num.jl")

end # module

