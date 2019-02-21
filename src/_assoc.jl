struct Assoc{T, N, Td} <: AbstractNamedArray{T, N, Td}
    data::Td
    # note: these peeps want to be concrete
    names::Tuple
    name_to_index::Tuple
    group_indices::Tuple
    function Assoc(data::AbstractArray{T, N}, names::Tuple{Vararg{AbstractArray, N}}) where {T, N}
        argcheck_constructor(data, names)
        # We make these ordered os that the group indices are in the right order. That's the only reason so far.
        name_to_index = Tuple(OrderedDict(ks .=> vs) for (ks, vs) in zip(names, axes(data)))
        group_indices = Tuple(
            # Disallow using native indices as group shortcuts
            # NOTE: This now breaks if you have some pairs and some not.
            # keytype(d) <: Pair && !(keytype(d) <: Pair{<:native_indices, <:Any}) # <-- future me: do not forget that second bit
            if any(x -> x <: Pair, typeof.(keys(d))) #keytype(d) <: Pair && !(keytype(d) <: Pair{<:native_indices, <:Any})
                group(first∘first, last, filter(x -> x[1] isa Pair, collect(d)))
            else
                # Prevent `ofnametype` from matching basic indices in dispatch.
                Dict{Union{}, Union{}}()
            end
            for d in name_to_index
        )
        new{T, N, typeof(data)}(data, Tuple(names), name_to_index, group_indices)
    end
end

names(A::Assoc) = A.names
data(A::Assoc) = A.data
name_to_index(A::Assoc, dim) = A.name_to_index[dim]
unparameterized(::Assoc) = Assoc

struct NameMissing end

struct IndexGroup <: AbstractArray{Int, 1}
    I::Vector{Int}
end
Base.size(x::IndexGroup) = size(x.I)
Base.getindex(x::IndexGroup, i::Int) = x.I[i]

@define_named_to_indices Assoc Union{Symbol, String, Tuple, Pair, NameMissing}

ofnametype(A::Assoc, dim, name) =
    name isa keytype(name_to_index(A, dim)) || name isa keytype(A.group_indices[dim])

name_missing(A::Assoc, dim, i) = if haskey(A.group_indices[dim], i)
    IndexGroup(A.group_indices[dim][i])
else
    NameMissing()
end
name_to_index(A::Assoc, dim, i::NameMissing) = []
function name_to_index(A::Assoc, dim, I::AbstractArray{<:Any, 1})
    I′ = (name_to_index(A, dim, i) for i in I)
    I′ = collect(Iterators.filter(x -> x != NameMissing(), I′))
    # Current hack for indexing with an array of IndexGroups;
    # todo: allow mixed indexing with an IndexGroup and others (flatten the groups and leave the rest).
    eltype(I′) <: IndexGroup ? [i for g in I′ for i in g.I] : I′
end

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
    unparameterized(A)(
        # Handle the annoying case of zero-dimensional array indexing
        if N == 0
            isarray(value) ? value : fill!(similar(Td), value)
        else
            value
        end,
        getnames(A, I′′)
    ) |> condense # todo: efficiency
end

const Assoc1D = Assoc{T, 1, Td} where {T, Td}
const Assoc2D = Assoc{T, 2, Td} where {T, Td}

Base.adjoint(A::Assoc2D) = Assoc(adjoint(data(A)), names(A)[2], names(A)[1])
Base.transpose(A::Assoc2D) = Assoc(transpose(data(A)), names(A)[2], names(A)[1])
# todo: n-d

const ASA = AbstractSparseArray

function mul_indices(A, A_i, B, B_i)
    # Is there some sorting-based efficiency gain to be had here?
    da = name_to_index(A, A_i)
    db = name_to_index(B, B_i)
    ks = collect(keys(da) ∩ keys(db))
    ia = [da[k] for k in ks]
    ib = [db[k] for k in ks]
    ia, ib
end

const SparseAssoc2D = Assoc2D{T, <:ASA} where {T}

sparse_assoc_t = Union{SparseAssoc2D, SparseAssoc2D{<:Any, <:Union{Transpose, Adjoint}}}

function Base.:*(A::sparse_assoc_t, B::sparse_assoc_t) # sparse
    ia, ib = mul_indices(A, 2, B, 1)
    arr = sparse(data(A))[:, ia] * sparse(data(B))[ib, :]
    condense(Assoc(dropzeros!(arr), names(A, 1), names(B, 2)))
end

function Base.:*(A::Assoc2D, B::Assoc2D) # nonsparse
    ia, ib = mul_indices(A, 2, B, 1)
    arr = data(A)[:, ia] * data(B)[ib, :]
    condense(Assoc(arr, names(A, 1), names(B, 2)))
end

union_names(A, B, dim) = let na = names(A, dim), nb = names(B, dim)
    na, nb, na ∪ nb
end

intersect_names(A, B, dim) = let na = names(A, dim), nb = names(B, dim)
    na, nb, na ∩ nb
end

# +ₛ(A::Assoc2D, B::Assoc2D) = elementwise_add_like(A, B, ∪) # problem = zero. zero in this ring is the empty set...

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

# For us:
#     + and ⊕ both mean elementwise addition
#
Base.:+(A::Assoc2D, B::Assoc2D) = elementwise_add_like(A, B, +) # todo: sparse/nonsparse
A::Assoc ⊕ B::Assoc = A + B
#
#     ⊗ means elementwise multiplication
#     * means associative array multiplication on associative arrays
#
A::Assoc2D ⊗ B::Assoc2D = elementwise_mul_like(A, B, *)
#
#     * means elementwise scalar multiplication
#
Base.:*(s::Number, A::Assoc2D) = condense(data(x -> s .* x, A))
Base.:*(A::Assoc2D, s::Number) = condense(data(x -> x .* s, A))

# Give a nice error message to scalar addition
no_scalar_addition() = error("Scalar addition is forbidden as it would result in a near-infinite associative array.")
Base.:+(A::Assoc2D, s::Number) = no_scalar_addition()
Base.:+(s::Number, A::Assoc2D) = no_scalar_addition()

# todo: how are chained a < b < c comparisons parsed/evaluated?
# todo: <=, >=

>(x, y) = Base.:>(x, y)
>(A::Assoc, s::Number) = data(a -> a .* (a .> s), A)
>(s::Number, A::Assoc) = data(a -> a .* (s .> a), A)

<(x, y) = Base.:<(x, y)
<(A::Assoc, s::Number) = data(a -> a .* (a .< s), A)
<(s::Number, A::Assoc) = data(a -> a .* (s .< a), A)

Base.:/(A::Assoc, s::Number) = data(a -> a ./ s, A)

## Utils.

# sparse-zero-preserving divide. Yes, I am having fun with Unicode.
div_zero(x, y) = iszero(x) && iszero(y) ? zero(x/y) : x/y

# function smooth(a, b, d, α)
#     x = a + logical(a, α*d)
#     y = b + logical(b, α)
#     elementwise_mul_like(x, y, div_zero)
# end

export threshold
function threshold(A::Assoc, cutoff)
    # todo: I think this is weird; run the 3x3 example to see why.
    # 1 4 7; 2 5 8; 3 6 9
    # What if we did it this way instead (remove anything without a value of at least x in some cell)
    # observed ⊗ logical(observed > 2)
    # what his is doing also makes sense, i think -- just might want different cutoffs for each axis

    # remove rows and columns whose total is less than the cutoff
    # compute marginals
    m1 = sum(A, dims=2)
    m2 = sum(A, dims=1)
    # @show m1 m2
    # index into the filtered subset
    # cutoff is inclusive
    A[vec(m1 .>= cutoff), vec(m2 .>= cutoff), named=true]
end

# todo: add a named/assoc argument; that way you can use sortperm from one array to index into another
function Base.sortperm(A::Assoc2D; dims, by=sum, rev=false, alg=Base.Sort.DEFAULT_STABLE)
    # using the `by` keyword is slower; does it reapply the function each time?
    names(A, dims)[sortperm(by.(eachslice(data(A); dims=dims)), rev=rev, alg=alg)]
end

function Base.sort(A::Assoc2D; dims, by=sum, rev=false, alg=Base.Sort.DEFAULT_STABLE)
    I = sortperm(A; dims=dims, by=by, rev=rev, alg=alg)
    if dims == 1
        A[I, :]
    else @assert dims == 2
        A[:, I]
    end
end

## /utils

# note: these are both inefficient. and possibly incorrect.

# todo: define condense for regular dense and sparse arrays, so we can reduce the number of
# intermediate assocs being created [note: incoherent thought; you need to know which names to drop too]
# [but maybe we can make a "find the uI and uJ" function]

function condense_indices(a::AbstractSparseMatrix)
    I, J, V = findnz(a)
    uI = sort(unique(I))
    uJ = sort(unique(J))
    uI, uJ
end

function condense(A::SparseAssoc2D)
    # note: not generic enough to handle non-OneTo axes, e.g. OffsetArrays
    # remove fully zero columns and rows
    a = data(A)
    I, J = condense_indices(a)
    unparameterized(A)(A[I, J], names(A, 1, I), names(A, 2, J))
end

function condense(A::Assoc2D)
    I = findall(!iszero, vec(sum(data(A), dims=2)))
    J = findall(!iszero, vec(sum(data(A), dims=1)))
    unparameterized(A)(A[I, J], names(A, 1, I), names(A, 2, J))
end

function triples(A::Assoc2D{<:Any, <:AbstractSparseMatrix})
    # [(row, col, val), ...]
    rownames, colnames = names(A)
    [
        (row=rownames[i], col=colnames[j], val=val)
        for (i, j, val) in zip(findnz(data(A))...)
        if !iszero(val)
    ]
end

function triples(A::Assoc2D)
    rownames, colnames = names(A)
    vec([
        (row=rownames[I[1]], col=colnames[I[2]], i=I[1], j=I[2], val=data(A)[I])
        for I in CartesianIndices(data(A))
    ])
end

function rowtriples(A::Assoc2D)
    (I, J, V) = findnz(data(A))
    groupby(
        first,
        [collect(zip(I, J, V))][sortperm(I)]
    )
end



function sparse_triples(t) # note: currently matches assocs of nonsparse
    mapmany(enumerate(t)) do (i, row)
        ((row=Id(i), col=typeof(v) <: Number ? k => true : Pair(k, v), val= typeof(v) <: Number ? v : 1) for (k, v) in pairs(row))
    end
end

lookup(arr) = Dict(zip(arr, LinearIndices(arr)))

function explode_sparse(t) # ::Vector{NamedTuple{(:row, :col, :val)}}
    @time rk = identity.(OrderedSet(x.row for x in t))
    @time ck = identity.(OrderedSet(x.col for x in t))
    @time eltype(rk) <: Pair && sort!(rk, by=first, alg=Base.Sort.DEFAULT_STABLE)
    @time eltype(ck) <: Pair && sort!(ck, by=first, alg=Base.Sort.DEFAULT_STABLE)
    @time rl, cl = lookup(rk), lookup(ck)
    @time I = [rl[x.row] for x in t]
    @time J = [cl[x.col] for x in t]
    @time K = [x.val for x in t]
    Assoc(sparse(I, J, K, length(rk), length(ck)), rk, ck)
end

explode(t) = explode_sparse(sparse_triples(t))

# Show

const highlight_row_label = Highlighter(
    f=(data, i, j) -> j == 1,
    crayon=crayon"bold"
)

sep(sz...) = repeat(["..."], sz...)

fmt(x::Pair) = "$(first(x))|$(last(x))"
fmt(x::Id) = x.id
fmt(x) = x

function pretty(io::IO, A::Assoc{<:Any, 1})
    pretty(io, unparameterized(A)(reshape(data(A), size(A, 1), 1), names(A, 1), [Id("-")]))
end

function pretty(io::IO, A::Assoc{<:Any, 2})
    arr = data(A)

    # half-width and half-height
    w = 4
    h = 10

    sz = size(arr)

    # Compute the number of separator rows/cols (0/1)
    nr = sz[1] > 2h ? 1 : 0 # new row?
    nc = sz[2] > 2w ? 1 : 0 # new col?

    if nr == 0
        t = sz[1] ÷ 2
        b = sz[1] - t
    else
        t = b = h
    end

    if nc == 0
        l = sz[2] ÷ 2
        r = sz[2] - l
    else
        l = r = w
    end

    # @show t l b r nr nc

    out =  Any[
        Array(@view arr[1:t, 1:l])           sep(t,  nc)   Array(@view arr[1:t, max(end-r+1,1):end])
        [sep(nr, l)                           sep(nr, nc)   sep(nr, r)]
        Array(@view arr[max(end-b+1,1):end, 1:l])   sep(b,  nc)   Array(@view arr[max(end-b+1,1):end, max(end-r+1,1):end])
    ]

    out[[x isa Number && iszero(x) for x in out]] .= ""

    n1, n2 = names(A)

    col_header = fmt.(vcat([""], n2[1:l], sep(nc), n2[max(end-r+1,1):end]))
    row_header = fmt.(vcat(n1[1:t], sep(nr), n1[max(end-b+1,1):end]))
    println(io, join(sz, '×'), " ", typeof(A), ":")
    if sz[1] > 0
        pretty_table(io, hcat(row_header, out), col_header, borderless, highlighters=(highlight_row_label,), alignment=:l)
    else
        println(io, col_header[2:end])
        println(io, "Empty array.")
    end
end

Base.show(io::IO, A::Assoc{<:Any, 1}) = pretty(io, A)
Base.show(io::IO, ::MIME"text/plain", A::Assoc{<:Any, 1}) = pretty(io, A)

Base.show(io::IO, A::Assoc{<:Any, 2}) = pretty(io, A)
Base.show(io::IO, ::MIME"text/plain", A::Assoc{<:Any, 2}) = pretty(io, A)


## hack around a[[true, false], [false, true], named=true] otherwise not working

# Base.getindex(A::AbstractArray, I1::Base.LogicalIndex, I2::Base.LogicalIndex) = A[collect(I1), collecT(I2)]
Base.getindex(A::SparseMatrixCSC, I1::Base.LogicalIndex, I2::Base.LogicalIndex) = A[collect(I1), collect(I2)]
