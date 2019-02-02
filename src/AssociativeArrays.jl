module AssociativeArrays

using ArgCheck, Base.Iterators, LinearAlgebra, SparseArrays, SplitApplyCombine,
      OrderedCollections, PrettyTables
using Base: tail

export Assoc, Num, Name
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

Base.map!(f, A::ANA) = (map!(f, data(A)); A)
Base.map(f, A::ANA) = unparameterized(A)(map(f, data(A)), names(A))

# low-level indexing
Base.getindex(A::ANA, i::Int) = data(A)[i]
Base.getindex(A::ANA{T, N}, I::Vararg{Int, N}) where {T, N} = data(A)[I...]

Base.setindex!(A::ANA, v, i::Int) = setindex!(data(A), v, i)
Base.setindex!(A::ANA{T, N}, v, I::Vararg{Int, N}) where {T, N} = setindex!(data(A), v, I...)

# We cannot know in general whether dimension names are preserved -- even
# if the similar array is of the same size, names may move around.
# So always return a "similar" array of the underlying array type.
# Base.similar(A::ANA) = similar(data(A))
# Note: See below; the "But just to see...".
# Base.similar(A::ANA, ::Type{S}) where {S} = similar(data(A), S)
Base.similar(A::ANA, ::Type{S}, dims::Dims) where {S} = similar(data(A), S, dims)

# But just to see what happens...
# This makes the default wrong in some cases, but right in the majority of cases.
Base.similar(A::ANA) = unparameterized(A)(similar(data(A)), names(A))
Base.similar(A::ANA, ::Type{S}) where {S} = unparameterized(A)(similar(data(A), S), names(A))

# Note – AxisArrays defines further specializations for Axis types:
# https://github.com/JuliaArrays/AxisArrays.jl/blob/48ec7350e3a8669dc17ef2e2f34069d86c227975/src/core.jl#L303

# broadcast

import Base.Broadcast: BroadcastStyle, AbstractArrayStyle, DefaultArrayStyle, Broadcasted

struct NamedArrayStyle{Style <: BroadcastStyle} <: AbstractArrayStyle{Any} end
NamedArrayStyle(::S) where {S} = NamedArrayStyle{S}()
NamedArrayStyle(::S, ::Val{N}) where {S, N} = NamedArrayStyle(S(Val(N)))
NamedArrayStyle(::Val{N}) where {N} = NamedArrayStyle{DefaultArrayStyle{N}}()
NamedArrayStyle(::Broadcast.Unknown) = Broadcast.Unknown()

# promotion rules

function BroadcastStyle(::Type{<:ANA{T, N, Td}}) where {T, N, Td}
    NamedArrayStyle(BroadcastStyle(Td))
end

function BroadcastStyle(::NamedArrayStyle{A}, ::NamedArrayStyle{B}) where {A, B}
    NamedArrayStyle(BroadcastStyle(A(), B()))
end

# Define these with DefaultArrayStyle for disambiguation.
function BroadcastStyle(::NamedArrayStyle{A}, ::B) where {A, B <: DefaultArrayStyle}
    NamedArrayStyle(BroadcastStyle(A(), B()))
end

function BroadcastStyle(::A, ::NamedArrayStyle{B}) where {A <: DefaultArrayStyle, B}
    NamedArrayStyle(BroadcastStyle(A(), B()))
end

# Note: If these want to get called there will be an ambiguity error with Base. But they're here temporarily
# to help us figure out what more specific methods we want to add in practice.
function BroadcastStyle(::NamedArrayStyle{A}, ::B) where {A, B <: AbstractArrayStyle}
    NamedArrayStyle(BroadcastStyle(A(), B()))
end

function BroadcastStyle(::A, ::NamedArrayStyle{B}) where {A <: AbstractArrayStyle, B}
    NamedArrayStyle(BroadcastStyle(A(), B()))
end

@inline function Base.copy(bc::Broadcasted{NamedArrayStyle{Style}}) where Style
    # Gather a tuple of all named arrays in this broadcast expression
    As = allnamed(bc)

    # Determine the maximal number of dimensions
    i = argmax(ndims.(As))
    A = As[i]
    noms = names(A)

    # Verify that names match along all dimensions of all named arrays.
    # Use isequal to correctly handle `missing` labels
    @argcheck(
        all(isequal(noms[i], names(A′, i)) for A′ in As for i in 1:ndims(A′)),
        "All names must match to broadcast across multiple named arrays."
    )

    # Compute the broadcast result on unwrapped arrays,
    value = copy(unwrap(bc, nothing))

    # Then re-wrap the result in a named array of the appropriate type.
    unparameterized(A)(value, noms)
end

# Our `copy` was based on:
# https://githucom/JuliaDiffEq/RecursiveArrayTools.jl/blob/e666b741ed713e32494de9f164fec13fc15f8391/src/array_partition.jl#L235
# Note: `copyto!` should look essentially the same as above:
# https://githucom/JuliaDiffEq/RecursiveArrayTools.jl/blob/e666b741ed713e32494de9f164fec13fc15f8391/src/array_partition.jl#L243

# Return a tuple of all named arrays
allnamed(bc::Broadcasted) = allnamed(bc.args)
# ::Tuple -> search it
@inline allnamed(args::Tuple) = (allnamed(args[1])..., allnamed(tail(args))...)
# ::ANA -> keep it
allnamed(a::ANA) = (a,)
# ::EmptyTuple -> discard it
allnamed(args::Tuple{}) = ()
# ::Any -> discard it
allnamed(a::Any) = ()

# Unwrap all of the named arrays within a Broadcasted expression. Note: `param` is currently unused, but is passed down
# to the unwrap(A::ANA, param) method in case we want to control the way an array is unwrapped in the future.
@inline unwrap(bc::Broadcasted{Style}, param) where Style = Broadcasted{Style}(bc.f, unwrap_args(bc.args, param), bc.axes)
@inline unwrap(bc::Broadcasted{NamedArrayStyle{Style}}, param) where Style = Broadcasted{Style}(bc.f, unwrap_args(bc.args, param), bc.axes)
unwrap(x, ::Any) = x
unwrap(A::ANA, param) = data(A)

@inline unwrap_args(args::Tuple, param) = (unwrap(args[1], param), unwrap_args(Base.tail(args), param)...)
unwrap_args(args::Tuple{Any}, param) = (unwrap(args[1], param),)
unwrap_args(args::Tuple{}, ::Any, ) = ()

# todo: copyto!

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

struct Name{T}
    name::T
end

const Id = Name # More concise and makes sense to talk about a "row ID"...

name_missing(A, dim, i) = throw(ArgumentError("Missing name $i for dimension $dim."))

function name_to_index(A, dim, i)
    get(name_to_index(A, dim), i) do
        name_missing(A, dim, i)
    end
end
# name_to_index(A, dim, i::Name) = name_to_index(A, dim, i.name)
name_to_index(A, dim, I::AbstractArray) = [name_to_index(A, dim, i) for i in I]

function default_named_getindex(A::ANA{T, N}, I′) where {T, N}
    nd = ndims.(I′)

    @boundscheck begin
        # TODO: Improve these error messages; specify the index/dimension
        @argcheck(
            all(x -> x == 0 || x == 1, nd),
            BoundsError("Multidimensional indexing within a single dimension is not supported.", I′)
        )

        @argcheck(
            all(i <= N || n == 0 for (i, n) in enumerate(nd)),
            BoundsError("Trailing indices may not introduce new dimensions.", I′)
        )
    end

    # The length may be greater than N in the case of trailing singleton indices.
    # It may be less than one in the case of zero-dimensional indexing into a 1x1x... array.
    @assert length(I′) <= 1 || length(I′) >= N
    value = data(A)[I′...]
    # `reduce` to avoid the sum(()) edge case
    nd_sum = reduce(+, nd, init=0)

    if nd_sum == 0
        # value # We have a scalar; return it.
        unparameterized(A)(value)
    else @assert length(I′) >= N
        unparameterized(A)(value, getnames(A, I′))
    end
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

getnames(A::ANA{T, N}, I::Tuple{Vararg{native_indices}}) where {T, N} =
    Tuple(Iterators.flatten(Iterators.repeated(names(A)[i][I[i]], ndims(I[i])) for i in 1:N))

Base.names(A::ANA, args...) = names(A, args...)

names(A::ANA, dim) = names(A)[dim]

# These methods should be defined by subtypes:
# names(A)::Tuple
# data(A)::Array{T, N}
# name_to_index(A, dim)::Dict
# unparameterized(::Arr) = Arr

macro define_named_to_indices(A, T)
    quote
        Base.to_indices(A::$A, ax, I::Tuple{Union{$T, AbstractArray{<:$T}}, Vararg{Any}}) =
            named_to_indices(A, ax, I)
    end
end

@define_named_to_indices ANA Name

function argcheck_constructor(data, names)
    @argcheck all(allunique.(names)) "Names must be unique within each dimension."
    @argcheck ndims(data) == length(names) "Each data dimension must be named."
    @argcheck all(size(data) .== length.(names)) "Data and name dimensions must be equal: $(size(data)) $(length.(names))."
    @argcheck all(axes(data) .== axes.(names, 1)) "Names must have the same axis as the corresponding data axis."
    @argcheck !any(T <: native_indices for T in eltype.(names)) "Names cannot be of type Int or AbstractArray."
end

densify(A::ANA) = unparameterized(A)(Array(data(A)), names(A))

##

include("namedarray.jl")
include("assoc.jl")
include("num.jl")

end # module

