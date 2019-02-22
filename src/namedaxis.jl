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

function NamedAxis(names::AbstractVector)
    # The most generic NamedAxis constructor.

    # Sorts groups lexicographically by name. Note that the default group name needs to be
    # sorted in its proper place relative to the other names so that NamedAxis() on Non-groups
    # creates the same array as NamedAxis() with default_group pairs.
    # Non-group names are represented as belonging to a default group and are
    # stored as pairs in `names` to increase the efficiency of creating sub-arrays
    # during indexing and algebraic operations.
    # We'll need to special-case them in `names()` but that's a less common operation.

    # Filter out pairs and stable-sort them by group name
    I = BitArray(name isa Pair for name in names)
    pairs = sort!(names[I], by=first)
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

    # Compute the named tuple of index ranges into the underlying name vector
    ranges = let
        ax = axes(names, 1)
        ls = Int[length(g) for g in dicts]
        cs = cumsum(ls)
        NamedTuple{keys(dicts)}(ax[i == 1 ? (1:l) : (cs[i-1]+1:cs[i])] for (i, l) in enumerate(ls))
    end

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

# Note: The `mapmany` approach used below will always create `NamedAxis` objects
# with 1:N indices. Maybe what we really want to do here is simply return the names, not a NamedAxis?

function Base.union(a::NamedAxis, b::NamedAxis)
    # Union group names
    groupnames = union(keys(a.dicts), keys(b.dicts))

    # Union the names within each group
    names = mapmany(groupnames) do groupname
        if haskey(a.dicts, groupname)
            if haskey(b.dicts, groupname)
                # Note that this relies on default_group names being stored as pairs
                groupname .=> union(keys(gf(a.dicts, groupname)), keys(gf(b.dicts, groupname)))
            else
                a.names[gf(b.ranges, groupname)]
            end
        else
            b.names[gf(b.ranges, groupname)]
        end
    end

    NamedAxis(names)
    # names
end

function Base.intersect(a::NamedAxis, b::NamedAxis)
    # Intersect group names
    groupnames = intersect(keys(a.dicts), keys(b.dicts))

    # Intersect names within each group
    names = mapmany(groupnames) do groupname
        groupname .=> intersect(keys(gf(a.dicts, groupname)), keys(gf(b.dicts, groupname)))
    end

    # These names are already sorted in the appropriate fashion and are all pairs.
    # We can use this to more efficiently construct the returned NamedAxis in the future.
    # Another possible optimization is to check for axis or group equality.
    # We could also directly construct a result NamedTuple rather than using `mapmany`:
    # NamedTuple{groupnames}(... for groupname in groupnames)

    NamedAxis(names)
    # names
end

function Base.setdiff(a::NamedAxis, b::NamedAxis)
    # Difference group names
    groupnames = setdiff(keys(a.dicts), keys(b.dicts))
    names = mapmany(groupnames) do groupname
        groupname .=> setdiff(keys(gf(a.dicts, groupname)), keys(gf(b.dicts, groupname)))
    end

    NamedAxis(names)
    # names
end


# What happens if we have a group where the values are pairs?
# Maybe two types of named axes — one with groups (and some ungrouped), and another that is plain ungrouped?
# Yeah... This is a good question.

# Idea: To get all "pairs" into one big group, store them as tuples instead. But then it's not the same name any more...

# I want to try to go back to storing the plain non-pair names in names, and dealing with default_group only internally.

# Allowing multiple group-representations for the same data feels bad — "ungrouping" should be a transformation done on the names.
# Maybe using tuples instead of pairs. Though of course the names don't line up any more for e.g. mul if you do it that way...
