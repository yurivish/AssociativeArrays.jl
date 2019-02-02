using AssociativeArrays, Test

# zero-dimensional gauntlet; integer indexing
function zero_dimensional_gauntlet(a)
    @test a[] == 1
    @test a[1] == 1
    @test a[1, 1] == 1
    @test a[[1], 1] == [1]
    @test a[1, [1]] == [1]
    @test a[reshape([1], 1, 1, 1)] == reshape([1], 1, 1, 1)
    @test a[:] == [1]
    @test a[:, :] == reshape([1], 1, 1)
    @test a[CartesianIndex(1)] == 1
    @test a[CartesianIndex(1, 1, 1)] == 1
    @test a[[]] == Int[]
    @test a[[1, 1]] == [1, 1]
    @test a[[true]] == [1]
    @test a[[false]] == Int[]
    @test_throws BoundsError a[[true, true]]
    @test_throws BoundsError a[[false, true]]
    @test_throws BoundsError a[[0, 1]]
end

# zero-dimensional-specific tests
let a = Assoc(fill(1))
    @test_throws BoundsError a[fill(1), named=true] == a
    @test_throws BoundsError a[named=true]
    @test_throws BoundsError a[1, named=true]
    @test_throws BoundsError a[1,1,1,named=true]
end

zero_dimensional_gauntlet(fill(1))
zero_dimensional_gauntlet(Assoc(fill(1)))
zero_dimensional_gauntlet(Assoc([1], [:a]))

function one_dimensional_gauntlet(a)
    @test a[fill(1), named=true] == Assoc(fill(1))
    @test a[named=true] == a
    @test a[1, named=true] == a
    @test a[1,1,1,named=true] == a

    @test a[] ==  a[named=false] == 1
    @test a[:a] == a
    @test a[:a, 1] == a
    @test_throws BoundsError a[:a, [1]]
    @test_throws BoundsError a[:a, 1, [1]]
    @test_throws BoundsError a[:a, [1], 1]
    @test_throws BoundsError a[:a, 0]
    @test_throws BoundsError a[:a, 3]
    @test_throws BoundsError a[:a, :]
    @test size(a[:b]) == (0,)
    @test_throws BoundsError a[:b, :c]
    @test a[[:a]] == a
    @test a[[:a, :b, :c]] == a
    @test a[1, named=true] == a
    @test a[1, named=false] == 1
    @test a[named=true] == a
    # This should really throw a better error about mixed indexing being disallowed:
    @test_throws ArgumentError a[[:a, 1]]
end

one_dimensional_gauntlet(Assoc([1], [:a]))

#=

Should be a bug:

julia> td = Assoc(reshape([1 2 3], 1, 3, 1), [:x], [:a, :b, :c], [:e])
julia> td[named=true]
1×1×1 Assoc{Int64,3,Array{Int64,3}}:
[:, :, 1] =
 1

And now is. An Assertion, at least.

=#