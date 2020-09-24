using Test

@testset "3x3M,Me" begin
    include("3x3M,ME.jl")
end

@testset "test orth" begin
    include("orth.jl")
end

@testset "environment" begin
    include("env.jl")
end

@testset "vumpsstep" begin
    include("vumpsstep.jl")
end
