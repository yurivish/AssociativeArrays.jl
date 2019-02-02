struct Assoc{T, N, Td} <: AbstractNamedArray{T, N, Td}
    data::Td
    # note: these peeps want to be concrete
    names::Tuple
    name_to_index::Tuple
    group_indices::Tuple
    function Assoc(data::AbstractArray{T, N}, names::Tuple{Vararg{AbstractArray, N}}) where {T, N}
        argcheck_constructor(data, names)
        name_to_index = Tuple(Dict(ks .=> vs) for (ks, vs) in zip(names, axes(data)))
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
    )
end

const Assoc1D = Assoc{T, 1, Td} where {T, Td}
const Assoc2D = Assoc{T, 2, Td} where {T, Td}

Base.adjoint(A::Assoc2D) = Assoc(adjoint(data(A)), names(A)[2], names(A)[1])
Base.transpose(A::Assoc2D) = Assoc(transpose(data(A)), names(A)[2], names(A)[1])
# todo: n-d

const ASA = AbstractSparseArray

function matmul_indices(A, A_i, B, B_i)
    # Is there some sorting-based efficiency gain to be had here?
    da = name_to_index(A, A_i)
    db = name_to_index(B, B_i)
    ks = collect(keys(da) ∩ keys(db))
    ia = [da[k] for k in ks]
    ib = [db[k] for k in ks]
    ia, ib
end

const SparseAssoc2D = Assoc2D{T, <:ASA} where {T}

sparse_mul_t = Union{SparseAssoc2D, SparseAssoc2D{<:Any, <:Union{Transpose, Adjoint}}}

function Base.:*(A::sparse_mul_t, B::sparse_mul_t) # sparse
    ia, ib = matmul_indices(A, 2, B, 1)
    arr = sparse(data(A))[:, ia] * sparse(data(B))[ib, :]
    Assoc(dropzeros!(arr), names(A, 1), names(B, 2))
end

function Base.:*(A::Assoc2D, B::Assoc2D) # nonsparse
    ia, ib = matmul_indices(A, 2, B, 1)
    arr = data(A)[:, ia] * data(B)[ib, :]
    Assoc(arr, names(A, 1), names(B, 2))
end

Base.sum(A::Assoc, args...; kws...) =
    sum(data(A), args...; kws...) # unparameterized(A)(, names(A))

##

function triples(A::Assoc2D{<:Any, <:AbstractSparseMatrix})
    # [(row, col, val), ...]
    rownames, colnames = names(A)
    [
        (row=rownames[i], col=colnames[j], i=i, j=j, val=val)
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

function sparse_triples(t) # note: currently matches assocs of nonsparse
    mapmany(enumerate(t)) do (i, row)
        ((row=Id(i), col=typeof(v) <: Number ? k : Pair(k, v), val=typeof(v) <: Number ? v : 1) for (k, v) in pairs(row))
    end
end

lookup(arr) = Dict(arr .=> LinearIndices(arr))

function explode_sparse(t) # ::Vector{NamedTuple{(:row, :col, :val)}}
    rk = identity.(OrderedSet(x.row for x in t))
    ck = identity.(OrderedSet(x.col for x in t))
    eltype(rk) <: Pair && sort!(rk, by=first, alg=Base.Sort.DEFAULT_STABLE)
    eltype(ck) <: Pair && sort!(ck, by=first, alg=Base.Sort.DEFAULT_STABLE)
    rl, cl = lookup(rk), lookup(ck)
    I = [rl[x.row] for x in t]
    J = [cl[x.col] for x in t]
    K = [x.val for x in t]
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
    w = 2
    h = 5

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
        println(io, "No rows.")
    end
end

Base.show(io::IO, A::Assoc{<:Any, 1}) = pretty(io, A)
Base.show(io::IO, ::MIME"text/plain", A::Assoc{<:Any, 1}) = pretty(io, A)

Base.show(io::IO, A::Assoc{<:Any, 2}) = pretty(io, A)
Base.show(io::IO, ::MIME"text/plain", A::Assoc{<:Any, 2}) = pretty(io, A)

