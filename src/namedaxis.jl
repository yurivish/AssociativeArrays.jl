const default_group = gensym("default_group")

# in progress: translate to 1:n, not arbitrary axis.

struct NamedAxis{Tp <: NamedTuple, Td <: NamedTuple, Tr <: NamedTuple}
    # every name is a pair of key => val.
    parts::Tp  # (key1=[val, ...], key2=...)
    dicts::Td  # (key1=Dict(val => index, ...), key2=...)
    ranges::Tr # (key1=1:3, key2=4:7, ...)
end

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

    # Sorts groups lexicographically by the group name. Note that the default group name needs to be
    # sorted in its proper place relative to the other names so that NamedAxis() on Non-groups
    # ^ is this note still correct?
    # creates the same array as NamedAxis() with default_group pairs.
    # Non-group names are represented as belonging to a default group and are
    # stored as pairs in `names` to increase the efficiency of creating sub-arrays
    # during indexing and algebraic operations.
    # We'll need to special-case them in `names()` but that's a less common operation.

    # Filter out pairs and stable-sort them by group name
    I = BitArray(name isa Pair for name in names)
    pairs = names[I]
    !issorted(pairs, by=first) && sort!(pairs, by=first)

    @assert(
        all(pair -> typeof(first(pair)) == Symbol, pairs),
        "For grouping purposes, the first value of every name `Pair` must be a Symbol."
    )

    # Note: We should assert that if pairs has default_group entries then rest is empty.
    # we can use searchsortedfirst.

    # Compute the named tuple of (groupkey=Dict(groupval => index, ...), ...)
    parts = merge(
        (; eduction(PartitionBy(first) |> Map(xs -> first(first(xs)) => last.(xs)), pairs)...),
        count(I) == length(names) ? [] : [default_group => names[.!I]]
    )

    NamedAxis(parts)
end

Base.length(na::NamedAxis) = sum(length, na.ranges) # length(na.names)

Base.getindex(na::NamedAxis, ::Colon) = na
function Base.getindex(na::NamedAxis, I::UnitRange)
    lo, hi = extrema(I)
    rs = na.ranges
    rv = collect(rs) # no tuple methods for searchsorted, so we need to collect.

    # indices of the groups overlapped by I
    a = searchsortedlast(rv, lo, by=first)
    b = searchsortedfirst(rv, hi, by=last)

    NamedAxis(
        NamedTuple{keys(rs)[a:b]}(
            if i == a && lo > first(rs[a])
                # partial start
                na.parts[a][lo-first(rs[a])+1:end]
            elseif i == b && hi < last(rs[b])
                # partial end
                na.parts[b][1:hi-first(rs[b])+1]
            else
                na.parts[i]
            end
            for i in a:b
        )
    )
end

function Base.getindex(na::NamedAxis, I::AbstractArray)
    # todo: can we do away with this check?
    isempty(I) && return NamedAxis([])

    lo, hi = extrema(I)
    rs = na.ranges
    rv = collect(rs) # no tuple methods for searchsorted, so we need to collect.

    # indices of the groups overlapped by I
    a = searchsortedlast(rv, lo, by=first)
    b = searchsortedfirst(rv, hi, by=last)
    # @show a b rs[a] I findall(in(rs[a]), I)

    NamedAxis(
        NamedTuple{keys(rs)[a:b]}(
            let
                rsi = rs[i]
                found = findall(in(rsi), I)
                if found == rsi
                    na.parts[i]
                else
                    na.parts[i][I[found] .- first(rsi) .+ 1]
                end
            end
            for i in a:b
        )
    )
end

# assoc indexing helper functions

function coalesce(arr::Vector{<:AbstractUnitRange})
    # Coalesce arrays of contiguous indices into a compact range.
    # This helps with things like group indexing.
    length(arr) < 2 && return arr
    step_between = first(arr[2]) - last(arr[1])

    # we coalesce if
    # 1. the step size between ranges is positive,
    # 2. the step size between ranges is the same as the step size within each range,
    # 3. all between steps are the same size.
    can_coalesce = step_between > 0 &&
        all(r -> step(r) == step_between, arr) &&
        all(first(arr[i]) - last(arr[i-1]) == step_between for i in 3:length(arr))

    if can_coalesce
        # If `arr` can be represented as a steprange, return that range.
        first(first(arr)):step_between:last(last(arr))
    else
        # Otherwise, flatten ranges into a single array
        collect(Iterators.flatten(arr))
    end

end

function coalesce(arr::Vector{Int})
    # Coalesce arrays of contiguous indices into a compact range.
    length(arr) < 2 && return arr
    stepsize = arr[2] - arr[1]
    if all(arr[i] - arr[i-1] == stepsize, 3:length(arr))
        first(arr):stepsize:last(arr)
    else
        arr
    end
end


function toindices(na::NamedAxis, names::AbstractVector)
    # collect(Iterators.flatten(toindices(na, name) for name in names if isnamedindex(na, name)))
    arr = [toindices(na, name) for name in names if isnamedindex(na, name)]

    # If there is only one element in `arr`, return it. The element might be an array or int.
    isempty(arr) && return []
    length(arr) == 1 && return first(arr)
    @assert length(arr) >= 2

    if eltype(arr) <: AbstractUnitRange
        coalesce(arr)
    elseif eltype(arr) <: Int
        coalesce(arr)
    else
        # Otherwise, flatten and return.
        collect(Iterators.flatten(arr))
    end
end

toindices(na::NamedAxis, name::Symbol) = gf(na.ranges, name)
toindices(na::NamedAxis, (k, v)::Pair{Symbol, <:Any}) = gf(na.ranges, k)[gf(na.dicts, k)[v]]
toindices(na::NamedAxis, name) = gf(na.ranges, default_group)[gf(na.dicts, default_group)[name]]

isname(na::NamedAxis, (k, v)::Pair) = haskey(gf(na.dicts, k), v)
isname(na::NamedAxis, name) = haskey(na.dicts, default_group) && haskey(gf(na.dicts, default_group), name)

isnamedindex(na::NamedAxis, name::Symbol) = haskey(na.dicts, name)
isnamedindex(na::NamedAxis, name) = isname(na, name)

# set operations

const gf = getfield

#=
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
                a_keys, b_keys = keys(a_dict), keys(b_dict)
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
        a_keys, b_keys = keys(a_dict), keys(b_dict)
        a_keys == b_keys ? a_dict : index_dict(intersect(a_keys, b_keys))
    end

    NamedTuple{groupnames}(dicts)
end
=#

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
    Note that the return value of this function is only valid
    for the two arrays `a` and `b`, since when two groups are identical,
    the group name is returned. That group name might index a different
    set of elements in another array.
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
