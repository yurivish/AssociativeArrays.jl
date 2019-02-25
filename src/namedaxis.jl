const default_group = gensym("default_group")

# in progress: translate to 1:n, not arbitrary axis.

struct NamedAxis{Tp <: NamedTuple, Td <: NamedTuple, Tr <: NamedTuple}
    # every name is a pair of groupkey => groupval.
    parts::Tp  # (groupkey=[groupval, ...], group2=...)
    dicts::Td  # (groupkey=Dict(groupval => index, ...), group2=...)
    ranges::Tr # (groupkey=1:3, group2=4:7, ...)
end

# function group_to_pair(group)
#     # Group is a vector of pairs all with the same first element, which is a Symbol.
#     first(first(group)) => Dict(v => i for (i, (_, v)) in enumerate(group))
# end

index_dict(xs) = Dict(x => i for (i, x) in enumerate(xs))

# Compute the named tuple of index ranges into the underlying name vector
function compute_ranges(dicts)
    ls = Int[length(g) for g in dicts]
    cs = cumsum(ls)
    NamedTuple{keys(dicts)}(i == 1 ? Base.OneTo(l) : cs[i-1]+1:cs[i] for (i, l) in enumerate(ls))
end

NamedAxis(parts) = NamedAxis(parts, map(index_dict, parts))
NamedAxis(parts, dicts) = NamedAxis(parts, dicts, compute_ranges(dicts))

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

    NamedAxis(dicts)
end

Base.length(na::NamedAxis) = sum(length, na.ranges) # length(na.names)
# Base.names(na::NamedAxis) = na.names
# Base.names(na::NamedAxis, i) = na.names[i]

Base.getindex(na::NamedAxis, ::Colon) = na
function Base.getindex(na::NamedAxis, I::UnitRange)
    # next steps: give namedaxes a parts for name partitions
    # and then make this function return a named axis with the appropriate parts, dicts, and ranges.

    # if we have this, i think everything works.
    # having this involves having a `parts` of partitions of names by group
    # searchsortedfirst(i ->
    #     na.ranges

    # Identify the groups that overlap I
    rs = na.ranges

    lo, hi = first(I), last(I)

    rangevals = collect(rs) # no tuple methods for searchsorted

    # indices of the groups that I overlaps
    a = searchsortedlast(rangevals, lo, by=first)
    b = searchsortedfirst(rangevals, hi, by=last)


    # whether there are partial groups at either end
    partial_start = lo > first(rs[a])
    partial_end = hi < last(rs[b])

    # @show a b partial_start partial_end

    outkeys = keys(rs)[a:b]

    if partial_start
        rsa = rs[a]
        prefix = [rsa[lo-first(rsa)+1:end]]
        a += 1
    else
        prefix = []
    end

    if partial_end
        rsb = rs[b]
        suffix = [rsb[1:hi-first(rsb)+1]]
        b -= 1
    else
        suffix = []
    end
    # @show prefix
    # @show collect(rs[x] for x in a:b)
    # @show suffix

    NamedTuple{outkeys}((prefix..., (rs[x] for x in a:b)..., suffix...))
end

# assoc indexing helper functions

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

index_dict(set) = Dict(v => i for (i, v) in enumerate(set))

function Base.union(a::NamedAxis, b::NamedAxis)
    # Union group names
    groupnames = union(keys(a.dicts), keys(b.dicts))

    # Union the names within each group.
    # We rely on type inference to produce an array of Pair;
    # Assocs do not currently support indexing with Any[].
    dicts = map(groupnames) do groupname
        if haskey(a.dicts, groupname)
            if haskey(b.dicts, groupname)
                a_dict = gf(a.dicts, groupname)
                b_dict = gf(b.dicts, groupname)
                a_keys, b_keys = keys(a_dict), keys(b_keys)
                a_keys == b_keys ? a_dict : index_dict(union(a_keys, b_keys))
            else
                gf(a.dicts, groupname)
            end
        else
            gf(b.dicts, groupname)
        end
    end
    NamedTuple{groupnames}(dicts)
end

function Base.intersect(a::NamedAxis, b::NamedAxis)
    # Intersect group names
    groupnames = intersect(keys(a.dicts), keys(b.dicts))

    # Intersect names within each group
    dicts = map(groupnames) do groupname
        a_dict = gf(a.dicts, groupname)
        b_dict = gf(b.dicts, groupname)
        a_keys, b_keys = keys(a_dict), keys(b_keys)
        a_keys == b_keys ? a_dict : index_dict(intersect(a_keys, b_keys))
    end

    NamedTuple{groupnames}(dicts)
end

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

"""
    Note that in general, the return value of this function is only valid
    for the two arrays `a` and `b`. This is because the group name is returned
    as an index for identical groups between the two arrays, and that group name
    might index a different set of elements in another array.
"""
function intersect_names(a::NamedAxis, b::NamedAxis)
    # Intersect group names
    groupnames = intersect(keys(a.dicts), keys(b.dicts))

    # Intersect names within each group
    names = map(groupnames) do groupname
        a_dict = gf(a.dicts, groupname)
        b_dict = gf(b.dicts, groupname)
        a_keys, b_keys = keys(a_dict), keys(b_dict)
        a_keys == b_keys ? groupname : groupname .=> intersect(a_keys, b_keys)
    end

    names
end
