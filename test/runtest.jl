using Test

@testset "M,Me" begin
    include("M,ME.jl")
end

@testset "test orth" begin
    include("orth.jl")
end

@testset "environment" begin
    include("env.jl")
end

@testset "energy" begin
    include("ene.jl")
end

@testset "vumpsstep" begin
    include("vumpsstep.jl")
end

@testset "r=1" begin
    include("r=1.jl")
end