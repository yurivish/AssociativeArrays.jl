module AssociativeArrays

# idea: work on a 'data-reshaper' user interface

using LinearAlgebra, SparseArrays, Base.Iterators
using Transducers, SplitApplyCombine, ArgCheck
export Assoc, NamedAxis, condense

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
(::Type{Ta})(data::AbstractArray{T, N}, names::Vararg{AbstractVector, N}) where {Ta <: ANA, T, N} = Ta(data, names)

#=
Notational conventions
    I are indices. Might be anything — names, `Not`, arrays, integers.
    I′ are "lowered" indices through `to_indices`.
    I′′ are lowered + descalarized indices suitable for dimensionality-preserving indexing.
=#

function named_to_indices(A::ANA{T, N}, ax, I) where {T, N}
    dim = N - length(ax) + 1
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

    @argcheck(
        all(x -> x == 0 || x == 1, nd),
        BoundsError("Multidimensional indexing within a single dimension is not supported.", I′)
    )

    @argcheck(
        all(i <= N || n == 0 for (i, n) in enumerate(nd)),
        BoundsError("Trailing indices may not introduce new dimensions; it is not clear what the new names would be.", I′)
    )

    data(A)[I′...]
end

function Base.getindex(A::ANA{T, N}, I...; named=missing) where {T, N}
    # Use `named=true` to force a named_getindex call even with all basic indices .
    # E.g. a[1, named=true] will return an assoc rather than a scalar.
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
    naxes::NTuple{N, NamedAxis}

    # Tuple{} for ambiguity resolution with the constructor accepting a tuple of `AbstractVector`s
    function Assoc(data::AbstractArray{T, N}, naxes::Union{Tuple{}, NTuple{N, NamedAxis}}) where {T, N}
        new{T, N, typeof(data)}(data, naxes)
    end
end

function Assoc(data::AbstractArray{T, N}, names::NTuple{N, AbstractVector}) where {T, N}
    # todo: make these more efficient and non-materializing
    begin # @time
        @argcheck all(allunique.(names)) "Names must be unique within each dimension."
        @argcheck ndims(data) == length(names) "Each data dimension must be named."
        @argcheck all(size(data) .== length.(names)) "Data and name dimensions must be equal: $(size(data)) $(length.(names))."
        @argcheck all(axes(data) .== axes.(names, 1)) "Names must have the same axis as the corresponding data axis."
        @argcheck !any(T <: Union{Int, AbstractArray, Symbol} for T in eltype.(names)) "Names cannot be of type Symbol, Int, or AbstractArray."
    end
    Assoc(data, NamedAxis.(names)) # @time
end

data(A::Assoc) = A.data
unparameterized(::Assoc) = Assoc

macro define_named_to_indices(A, T)
    quote
        # One problem with this is that you can't index with a Vector{Any} containing valid indices.
        Base.to_indices(A::$A, ax, I::Tuple{Union{$T, AbstractArray{<:$T}}, Vararg{Any}}) =
            named_to_indices(A, ax, I)
    end
end

# Allow Symbol here since indexing with one is allowed, it is just not allowed to be a name.
const assoc_name_types = Union{String, Char, Symbol, Pair} # , NamedAxis}
@define_named_to_indices Assoc assoc_name_types

# Count Symbols as a name type so that they pass through to named_getindex
isnametype(A::Assoc, dim, name::assoc_name_types) = true
isnametype(A::Assoc, dim, name) = false

# Missing names are handled in two places: here, and in toindices(axis, ::AbstractArray).
# todo: handle non-OneTo axes (by reindexing into axes(data(A), dim))
name_to_index(A::Assoc, dim, I) = let na = A.naxes[dim]
    isnamedindex(na, I) ? toindices(na, I) : []
end
name_to_index(A::Assoc, dim, I::AbstractArray) = toindices(A.naxes[dim], I)

# 0d array => 1d array
descalarize(x::AbstractArray{<:Any, 0}) = reshape(x, length(x))
# keep other n-d arrays for downstream error-checking
descalarize(x::AbstractArray) = x
# scalar => 1d array
descalarize(x) = [x]

# scalar => 0d array
descalarize_0d(x::Int) = fill(x)
# keep everything else for downstream error-checking
descalarize_0d(x) = x

function named_getindex(A::Assoc{T, N, Td}, I′) where {T, N, Td}
    # Lift scalar indices to arrays so that the result of indexing matches
    # the dimensionality of A. We iterate rather than broadcast to avoid
    # descalarizing trailing dimensions.
    len = length(I′)
    I′′ = if len == N == 0
        # Ensure a zero-dimensional result from default_named_getindex.
        # See the comment in the branch below for more details.
        tuple(fill(1))
    elseif len == N
        # This is the most common branch.
        descalarize.(I′)
    elseif len > N
        # More indices than dimensions; lift those <= N to 1d and make trailing indices 0d.
        # This accounts for corner cases with zero-dimensional indexing, ensuring that
        # default_named_getindex returns an array. Compare `fill(1)[1]` and fill(1)[fill(1)].
        ntuple(dim -> dim > N ? descalarize_0d(I′[dim]) : descalarize(I′[dim]), length(I′))
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

    value = default_named_getindex(A, I′′)

    condense(Assoc(value, Tuple(A.naxes[dim][I′′[dim]] for dim in 1:N)))
end

# 0-d
function condense_indices(a::AbstractArray{<:Any, 0})
    I = findall(!iszero, a)
    @assert length(I) <= 1
    # Return a value such that a[val...] preserves dimensionality,
    # i.e. does not introduce a new dimension
    (isempty(I) ? I : fill(first(I)),)
end

# n-d
function condense_indices(a::AbstractArray{<:Any, N}) where N
    isempty(a) && return ntuple(dim -> [], N)

    I = findall(!iszero, a)
    ntuple(
        dim -> let inds = sort!(unique(i[dim] for i in I))
            length(inds) < size(a, N) ? inds : (:)
        end,
        N
    )
end

# We watch out for colons (:) in `condense` because
# they represent opportunities to reuse existing arrays.
function condense(A::Assoc)
    a = data(A)
    I = condense_indices(a)
    all(==(:), I) && return A
    Assoc(a[I...], getindex.(A.naxes, I))
end

function elementwise_mul(A::Assoc{<:Any, N}, B::Assoc{<:Any, N}, * = *) where N
    z = zero(eltype(A)) * zero(eltype(B)) # Infer element type by doing a zero-zero test for now.
    T = typeof(z) # promote_type(eltype(A), eltype(B))
    @assert iszero(z) "*(0, 0) == 0 must hold for multiplication-like operators."

    axs = map(intersect_names, A.naxes, B.naxes) # note: names, not axes
    value = A[axs..., named=false] .* B[axs..., named=false]
    condense(Assoc(value, axs))
end

function elementwise_add(A::Assoc{<:Any, N}, B::Assoc{<:Any, N}, + = +) where N
    T = promote_type(eltype(A), eltype(B))
    @assert iszero(zero(T) + zero(T)) "+(0, 0) == 0 must hold for addition-like operators."
    @assert isone(zero(T) + one(T)) "+(0, 1) == 1 must hold for addition-like operators."
    @assert isone(one(T) + zero(T)) "+(1, 0) == 1 must hold for addition-like operators."

    axs = map(union_names, A.naxes, B.naxes) # note: names, not axes
    z = issparse(data(A)) || issparse(data(B)) ? spzeros : zeros
    C = Assoc(z(T, length.(axs)), axs)

    # dotview does not accept keyword arguments so we can't do @view C[A.naxes..., named=false]
    I_a, I_b = to_indices(C, A.naxes), to_indices(C, B.naxes)
    data(C)[I_a...] .+= data(A)
    data(C)[I_b...] .+= data(B)
    C # if A and B are condensed, C is condensed by construction.
end

const Assoc2D = Assoc{T, 2} where T

# Sometimes it's necessary for the best performance to remove the wrapper before indexing.
# Related issue: https://github.com/JuliaLang/julia/pull/30552
unwrap_sparse_wrapper(x::Adjoint{<:Any, <:AbstractSparseMatrix}) = sparse(x)
unwrap_sparse_wrapper(x::Transpose{<:Any, <:AbstractSparseMatrix}) = sparse(x)
unwrap_sparse_wrapper(x) = x

function Base.:*(A::Assoc2D, B::Assoc2D)
    inds = intersect_names(A.naxes[2], B.naxes[1])
    I_a = to_indices(A, (:, inds))
    I_b = to_indices(B, (inds, :))
    value = unwrap_sparse_wrapper(data(A))[I_a...] * unwrap_sparse_wrapper(data(B))[I_b...]
    condense(Assoc(value, (A.naxes[1], B.naxes[2])))
end

Base.adjoint(A::Assoc2D) = Assoc(adjoint(data(A)), reverse(A.naxes))
Base.transpose(A::Assoc2D) = Assoc(transpose(data(A)), reverse(A.naxes))

end # module