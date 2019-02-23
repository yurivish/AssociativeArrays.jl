const default_group = gensym("default_group")

struct NamedAxis{Tn, Td, Tr}
    names::Tn  # [name1, name2, ...]. These names are all pairs, using default_group if necessary.
    dicts::Td  # (group1=Dict(name => index, ...), group2=...)
    ranges::Tr # (group1=1:3, group2=4:7, ...)
end

function group_to_pair(group)
    # Group is a vector of pairs all with the same first element, which is a Symbol.
    first(first(group)) => Dict(v => i for (i, (_, v)) in enumerate(group))
end

# Compute the named tuple of index ranges into the underlying name vector
function compute_ranges(names, dicts)
    ax = axes(names, 1)
    ls = Int[length(g) for g in dicts]
    cs = cumsum(ls)
    NamedTuple{keys(dicts)}(ax[i == 1 ? (1:l) : (cs[i-1]+1:cs[i])] for (i, l) in enumerate(ls))
end

NamedAxis(names, dicts) = NamedAxis(names, dicts, compute_ranges(names, dicts))

function NamedAxis(names::AbstractVector)
    # The most generic NamedAxis constructor.
    # todo: separte impl for AbstractVector{<:Pair}
    # todo: what's a good sorting algorithm when the number of array values greatly exceeds the number of
    # unique sort values? might want to just do a bucket sort for each unique pair name.

    # Sorts groups lexicographically by name. Note that the default group name needs to be
    # sorted in its proper place relative to the other names so that NamedAxis() on Non-groups
    # creates the same array as NamedAxis() with default_group pairs.
    # Non-group names are represented as belonging to a default group and are
    # stored as pairs in `names` to increase the efficiency of creating sub-arrays
    # during indexing and algebraic operations.
    # We'll need to special-case them in `names()` but that's a less common operation.

    # Filter out pairs and stable-sort them by group name
    I = BitArray(name isa Pair for name in names)
    _pairs = names[I]
    pairs = issorted(_pairs, by=first) ? _pairs : sort!(_pairs, by=first)
    @assert(
        all(pair -> typeof(first(pair)) == Symbol, pairs),
        "For grouping purposes, the first value of every name `Pair` must be a Symbol."
    )
    rest = names[.!I]

    # Note: We should assert that if pairs has default_group entries then rest is empty.
    # we can use searchsortedfirst.

    # Compute the named tuple of (group1=Dict(name => index, ...), ...)
    dicts = merge(
        (; eduction(PartitionBy(first) |> Map(group_to_pair), pairs)...),
        isempty(rest) ? [] : [default_group => Dict(v => i for (i, v) in enumerate(rest))]
    )

    ranges = compute_ranges(names, dicts)

    NamedAxis(vcat(pairs, isempty(rest) ? [] : default_group .=> rest), dicts, ranges)
end

Base.length(na::NamedAxis) = length(na.names)
Base.names(na::NamedAxis) = na.names
Base.names(na::NamedAxis, i) = na.names[i]

# indexing helper functions

toindices(na::NamedAxis, names::AbstractVector) =
    collect(Iterators.flatten(toindices(na, name) for name in names if isnamedindex(na, name)))

toindices(na::NamedAxis, name::Symbol) = gf(na.ranges, name)
toindices(na::NamedAxis, (k, v)::Pair{Symbol, <:Any}) = gf(na.ranges, k)[gf(na.dicts, k)[v]]
toindices(na::NamedAxis, name) = gf(na.ranges, default_group)[gf(na.dicts, default_group)[name]]

isname(na::NamedAxis, (k, v)::Pair) = haskey(gf(na.dicts, k), v)
isname(na::NamedAxis, name) = haskey(gf(na.dicts, default_group), name)

isnamedindex(na::NamedAxis, name::Symbol) = haskey(na.dicts, name)
isnamedindex(na::NamedAxis, name) = isname(na, name)

# set operations

const gf = getfield

function union_names(a::NamedAxis, b::NamedAxis)
    # Union group names
    groupnames = union(keys(a.dicts), keys(b.dicts))

    # Union the names within each group.
    # We rely on type inference to produce an array of Pair;
    # Assocs do not currently support indexing with Any[].
    mapmany(groupnames) do groupname
        if haskey(a.dicts, groupname)
            if haskey(b.dicts, groupname)
                a_keys = keys(gf(a.dicts, groupname))
                b_keys = keys(gf(b.dicts, groupname))
                # This relies on default_group names being stored as pairs
                groupname .=> (a_keys == b_keys ? a_keys : union(a_keys, b_keys))
            else
                a.names[gf(b.ranges, groupname)]
            end
        else
            b.names[gf(b.ranges, groupname)]
        end
    end
end

function intersect_names(a::NamedAxis, b::NamedAxis)
    # Intersect group names
    groupnames = intersect(keys(a.dicts), keys(b.dicts))

    # Intersect names within each group
    mapmany(groupnames) do groupname
        a_keys = keys(gf(a.dicts, groupname))
        b_keys = keys(gf(b.dicts, groupname))
        groupname .=> (a_keys == b_keys ? a_keys : intersect(a_keys, b_keys))
    end
end

function setdiff_names(a::NamedAxis, b::NamedAxis)
    # Difference group names
    groupnames = setdiff(keys(a.dicts), keys(b.dicts))
    names = mapmany(groupnames) do groupname
        a_keys = keys(gf(a.dicts, groupname))
        b_keys = keys(gf(b.dicts, groupname))
        groupname .=> setdiff(a_keys, b_keys)
    end
    names
end

# Should we use an ArrayPartition from RecursiveArrayTools for the names vector, to account for type-instability?
# https://github.com/JuliaDiffEq/RecursiveArrayTools.jl

#=
function csv2axes(csv)
    names = Vector{Pair}[]
    dicts = []
    offset = 0
    # Note: This does not take into account the fact that our axis wants _unique_ names.
    # I wonder if we can construct the sparse array here simultaneously with axes.
    for (groupname, vals) in pairs(columns(csv))
        groupnames = map(val -> groupname => val, vals)
        inds = (1:length(vals)) .+ offset
        dict = Dict(zip(vals, inds))
        push!(names, groupnames)
        push!(dicts, groupname => dict)
        offset += length(vals)
    end
    NamedAxis(ArrayPartition(names...), (; dicts...))
end

t = let t = Table(csv[1:50_000])
    @time csv2axes(t)
end;
=#

# Note: if we use `unique` it preserves order.
#=

offset = 0

using Iterators

for (col, vals) in cols
    uniques = unique(vals)
    groupnames = map(vals) do val
        groupname => val
    end
    len = length(vals)
    inds = 1:len
    groupdict = Dict(zip(vals, inds)
    # global inds = inds .+ offset; that is what we want to use for (I, J, V)
    offset += length(vals)

    I = (Row(i) for i in 1:len)
    J = (groupdict[val] + offset for val in vals)
    V = repeated(true, len)
end

---

function csv2assoc(csv)
    names = Vector[]
    dicts = []
    Js = Vector{Int}[]
    offset = 0

    cols = columns(csv)
    ncols = length(cols)
    for (groupname, vals) in pairs(cols)
        uvals = unique(vals)
        len = length(uvals)

        groupnames = [groupname => val for val in uvals]
        groupdict = Dict(zip(uvals, 1:len))
        J = [groupdict[val] + offset for val in vals]

        push!(names, groupnames)
        push!(dicts, groupdict)
        push!(Js, J)

        offset += len
    end

    col_axis = NamedAxis(ArrayPartition(names...), NamedTuple{keys(cols)}(dicts))
    row_axis = NamedAxis([:row => i for i in 1:length(csv)])
    value = sparse(ArrayPartition([1:length(csv) for _ in 1:ncols]...), ArrayPartition(Js...), 1)
    Assoc(value, (row_axis, col_axis))
end

=#