using Base.Iterators, Transducers, SplitApplyCombine

struct NamedAxis{Tn, Td, Tr}
    names::Tn  # name vector
    dicts::Td  # named tuple of group name => index within that group
    ranges::Tr # named tuple of index range per group
end

function NamedAxis(names::Vector)
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
    ranges = let
        ls = [length(g) for g in dicts]
        cs = cumsum(ls)
        NamedTuple{keys(dicts)}(i == 1 ? (1:l) : (cs[i-1]+1:cs[i]) for (i, l) in enumerate(ls))
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

na = NamedAxis([1,2,3,4, :foo => :bar, 6, 5, :foo => :baz, :baz => :borz])

nb = intersect(na, na)

@show na.names
@show nb.names
println()
@show na.dicts
@show nb.dicts
println()
@show na.ranges
@show nb.ranges

na == nb # Now similar up to sorting.

