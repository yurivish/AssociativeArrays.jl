### Plain named array; names must exist

struct NamedAray{T, N, Td} <: AbstractNamedArray{T, N, Td}
    data::Td
    names::Tuple
    name_to_index::Tuple
    function NamedAray(data::AbstractArray{T, N}, names::Tuple{Vararg{AbstractArray, N}}) where {T, N}
        argcheck_constructor(data, names)
        name_to_index = Tuple(Dict(ks .=> vs) for (ks, vs) in zip(names, axes(data)))
        new{T, N, typeof(data)}(data, Tuple(names), name_to_index)
    end
end
NamedAray(data::AbstractArray{T, N}, names::Vararg{AbstractArray, N}) where {T, N} = NamedAray(data, names)

names(A::NamedAray) = A.names
data(A::NamedAray) = A.data
name_to_index(A::NamedAray, dim) = A.name_to_index[dim]
unparameterized(::NamedAray) = NamedAray

@define_named_to_indices NamedAray Union{Symbol, String, Tuple, Pair}
