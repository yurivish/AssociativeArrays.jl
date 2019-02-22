const default_group = gensym("default_group")

struct NamedAxis{Tn, Td, Tr}
    names::Tn  # [name1, name2, ...]
    dicts::Td  # (group1=Dict(name => index, ...), group2=...)
    ranges::Tr # (group1=1:3, group2=4:7, ...)
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
    @assert all(==(Symbol), typeof.(first.(pairs))) "The first value of every name `Pair` must be a Symbol." # todo: why? for uniformity/ui
    rest = names[.!I]
    groups = vcat(default_group .=> rest, pairs)

    # Compute the named tuple of name => index
    dicts = let
        kvs = Tuple(eduction(
            PartitionBy(first) |> Map(group -> first(first(group)) => Dict(reverse.(enumerate(last.(group))))),
            groups
        ))
        NamedTuple{first.(kvs)}(last.(kvs))
    end

    # Compute the named tuple of index ranges

    ranges = isempty(dicts) ? NamedTuple() : let
        ax = axes(names, 1) # Ranges represent index ranges of the underlying name vector
        ls = [length(g) for g in dicts]
        cs = cumsum(ls)
        NamedTuple{keys(dicts)}(ax[i == 1 ? (1:l) : (cs[i-1]+1:cs[i])] for (i, l) in enumerate(ls))
    end

    NamedAxis(groups, dicts, ranges)
end

function Base.union(a::NamedAxis, b::NamedAxis)
    # Union group names
    groupnames = union(keys(a.dicts), keys(b.dicts))

    # Union names within each group
    names = mapmany(groupnames) do groupname
        if haskey(a.dicts, groupname) && haskey(b.dicts, groupname)
            groupname .=> union(keys(getfield(a.dicts, groupname)), keys(getfield(b.dicts, groupname)))
        elseif haskey(a.dicts, groupname)
            @view a.names[getfield(a.ranges, groupname)]
        else @assert haskey(b.dicts, groupname)
            @view b.names[getfield(b.ranges, groupname)]
        end
    end
    NamedAxis(names)
end

function Base.intersect(a::NamedAxis, b::NamedAxis)
    # Intersect group names
    groupnames = intersect(keys(a.dicts), keys(b.dicts))

    # Intersect names within each group
    names = mapmany(groupnames) do groupname
        groupname .=> intersect(keys(getfield(a.dicts, groupname)), keys(getfield(b.dicts, groupname)))
    end

    # These names are already sorted in the appropriate fashion and are all pairs.
    # We can use this to more efficiently construct the returned NamedAxis in the future.
    # Another possible optimization is to check for axis or group equality.
    # We could also directly construct a result NamedTuple rather than using `mapmany`:
    # NamedTuple{groupnames}(... for groupname in groupnames)
    NamedAxis(names)
end

function Base.setdiff(a::NamedAxis, b::NamedAxis)
    # Difference group names
    groupnames = setdiff(keys(a.dicts), keys(b.dicts))
    names = mapmany(groupnames) do groupname
        groupname .=> setdiff(keys(getfield(a.dicts, groupname)), keys(getfield(b.dicts, groupname)))
    end
    NamedAxis(names)
end

# What happens if we have a group where the values are pairs?
# Maybe two types of named axes — one with groups (and some ungrouped), and another that is plain ungrouped?
# Yeah... This is a good question.

# Another wrinkle: How do you disambiguate between a symbol that is a name, and a symbol that is gesturing towards a group?
# Perhaps do not allow symbols as names — only as pair firsts. Otherwise say strings.

toindices(na::NamedAxis, names::AbstractVector) =
    collect(Iterators.flatten(toindices(na, name) for name in names if isnamedindex(na, name)))

const gf = getfield

toindices(na::NamedAxis, name::Symbol) = gf(na.ranges, name)
toindices(na::NamedAxis, (k, v)::Pair) = gf(na.ranges, k)[gf(na.dicts, k)[v]]
toindices(na::NamedAxis, name) = gf(na.ranges, default_group)[gf(na.dicts, default_group)[name]]

isname(na::NamedAxis, (k, v)::Pair) = haskey(gf(na.dicts, k), v)
isname(na::NamedAxis, name) = haskey(gf(na.dicts, default_group), name)

isnamedindex(na::NamedAxis, name::Symbol) = haskey(na.dicts, name)
isnamedindex(na::NamedAxis, name) = isname(na, name)
