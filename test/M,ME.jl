
using Test

include("nontrivialVUMPS.jl") 

@testset "M ME" begin
    βc = log(1+sqrt(2))/2
    β = 0.1
    M, ME,λM,λME= classicalisingmpo(β; r = 1.0)
    @test M[1,1] ≈ M[1,2] ≈ M[2,1] ≈ M[2,2]
    @test ME[1,1] ≈ ME[1,2]
    @test ME[2,1] ≈ ME[2,2]
end