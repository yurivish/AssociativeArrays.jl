module AssociativeArrays

using SparseArrays, Transducers, SplitApplyCombine, Base.Iterators, ArgCheck
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
    axes::NTuple{N, NamedAxis}

    # Tuple{} for ambiguity resolution with the constructor accepting a tuple of `AbstractVector`s
    function Assoc(data::AbstractArray{T, N}, axes::Union{Tuple{}, NTuple{N, NamedAxis}}) where {T, N}
        new{T, N, typeof(data)}(data, axes)
    end
end

function Assoc(data::AbstractArray{T, N}, names::NTuple{N, AbstractVector}) where {T, N}
    # todo: make these more efficient and non-materializing
    @argcheck all(allunique.(names)) "Names must be unique within each dimension."
    @argcheck ndims(data) == length(names) "Each data dimension must be named."
    @argcheck all(size(data) .== length.(names)) "Data and name dimensions must be equal: $(size(data)) $(length.(names))."
    @argcheck all(axes(data) .== axes.(names, 1)) "Names must have the same axis as the corresponding data axis."
    @argcheck !any(T <: Union{Int, AbstractArray, Symbol} for T in eltype.(names)) "Names cannot be of type Symbol, Int, or AbstractArray."
    Assoc(data, NamedAxis.(names))
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
const assoc_name_types = Union{String, Char, Symbol, Pair, NamedAxis}
@define_named_to_indices Assoc assoc_name_types

# Count Symbols as a name type so that they pass through to named_getindex
isnametype(A::Assoc, dim, name::assoc_name_types) = true
isnametype(A::Assoc, dim, name) = false

# Missing names are handled in two places: here, and in toindices(axis, ::AbstractArray).
name_to_index(A::Assoc, dim, I) = let axis = A.axes[dim]
    isnamedindex(axis, I) ? toindices(axis, I) : []
end
name_to_index(A::Assoc, dim, I::AbstractArray) = toindices(A.axes[dim], I)
name_to_index(A::Assoc, dim, I::NamedAxis) = name_to_index(A, dim, names(I))

# turn a 0d array into a 1d array
descalarize(x::AbstractArray{<:Any, 0}) = reshape(x, length(x))
# keep 1+d arrays for downstream error checking
descalarize(x::AbstractArray) = x
# turn a scalar into a 1d array
descalarize(x) = [x]

# turn a scalar into 0d array
descalarize_0d(x::Int) = fill(x)
# keep everything else for downstream error checking
descalarize_0d(x) = x

function named_getindex(A::Assoc{T, N, Td}, I′) where {T, N, Td}
    # Lift scalar indices to arrays so that the result of indexing matches
    # the dimensionality of A. We iterate rather than broadcast to avoid
    # descalarizing trailing dimensions.
    len = length(I′)
    I′′ = if len == N == 0
        # Ensure a zero-dimensional result from default_named_getindex.
        # (See comment in the branch below)
        tuple(fill(1))
    elseif len >= N
        # More indices than dimensions;
        # - lift those <= N to 1d and make trailing indices 0d. This accounts for corner cases
        # with zero-dimensional indexing, ensuring that default_named_getindex returns an array:
        # compare `fill(1)[1]` and fill(1)[fill(1)].
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

    I_condensed = condense_indices(value)
    samesize = all(==(:), I_condensed)
    condensed_value = samesize ? value : value[I_condensed...]
    unparameterized(A)(
        condensed_value,
        ntuple(dim -> names(A.axes[dim], I_condensed[dim] == (:) ? I′′[dim] : I′′[dim][I_condensed[dim]]), N)
    )
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
    I = findall(!iszero, a)
    ntuple(
        dim -> let inds = sort!(unique(i[dim] for i in I))
            length(inds) < size(a, N) ? inds : (:)
        end,
        N
    )
end

# We're always watching out for colons (:) in `condense` because
# they represent opportunities to reuse existing arrays.
function condense(A::Assoc)
    a = data(A)
    I = condense_indices(a)
    all(==(:), I) && return A
    unparameterized(A)(
        a[I...],
        ntuple(
            dim -> I[dim] == (:) ? names(A.axes[dim]) : names(A.axes[dim], I[dim]),
            ndims(A)
        )
    )
end

function elementwise_mul(A::Assoc{<:Any, N}, B::Assoc{<:Any, N}, * = *) where N
    z = zero(eltype(A)) * zero(eltype(B)) # Infer element type by doing a zero-zero test for now.
    T = typeof(z) # promote_type(eltype(A), eltype(B))
    @assert iszero(z) "*(0, 0) == 0 must hold for multiplication-like operators."

    axs = map(intersect_names, A.axes, B.axes)
    value = A[axs..., named=false] .* B[axs..., named=false]
    condense(Assoc(value, axs))
end

function elementwise_add(A::Assoc{<:Any, N}, B::Assoc{<:Any, N}, + = +) where N
    T = promote_type(eltype(A), eltype(B))
    @assert iszero(zero(T) + zero(T)) "+(0, 0) == 0 must hold for addition-like operators."
    @assert isone(zero(T) + one(T)) "+(0, 1) == 1 must hold for addition-like operators."
    @assert isone(one(T) + zero(T)) "+(1, 0) == 1 must hold for addition-like operators."

    axs = map(union_names, A.axes, B.axes)
    z = issparse(data(A)) || issparse(data(B)) ? spzeros : zeros
    C = Assoc(z(T, length.(axs)), axs)

    # dotview does not accept keyword arguments so we can't do C[A.axes..., named=false]
    I_a, I_b = to_indices(C, A.axes), to_indices(C, B.axes)
    data(C)[I_a...] .+= data(A)
    data(C)[I_b...] .+= data(B)
    C # if A and B are condensed, C is condensed by construction.
end

const Assoc2D = Assoc{T, 2} where T

function Base.:*(A::Assoc2D, B::Assoc2D)
    ax = intersect_names(A.axes[2], B.axes[1])
    value = A[:, ax, named=false] * B[ax, :, named=true]
    condense(Assoc(value, (A.axes[1], B.axes[2])))
end

Base.adjoint(A::Assoc2D) = Assoc(adjoint(data(A)), reverse(A.axes))
Base.transpose(A::Assoc2D) = Assoc(transpose(data(A)), reverse(A.axes))

#=

function mul_indices(A, A_i, B, B_i)
    # Is there some sorting-based efficiency gain to be had here?
    da = name_to_index(A, A_i)
    db = name_to_index(B, B_i)
    ks = collect(keys(da) ∩ keys(db))
    ia = [da[k] for k in ks]
    ib = [db[k] for k in ks]
    ia, ib
end

function Base.:*(A::Assoc2D, B::Assoc2D) # nonsparse
    ia, ib = mul_indices(A, 2, B, 1)
    arr = data(A)[:, ia] * data(B)[ib, :]
    condense(Assoc(arr, names(A, 1), names(B, 2)))
end

function Base.:*(A::sparse_assoc_t, B::sparse_assoc_t) # sparse
    ia, ib = mul_indices(A, 2, B, 1)
    arr = sparse(data(A))[:, ia] * sparse(data(B))[ib, :]
    condense(Assoc(dropzeros!(arr), names(A, 1), names(B, 2)))
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

=#

end # module