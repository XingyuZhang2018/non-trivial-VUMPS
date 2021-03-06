using Test

include("..//src//nontrivialVUMPS.jl")

@testset "vumps test β=$β r=$r" for β in [0.1], r in [1]
    M,ME_row, ME_col, λM,λME_row,λME_col = classicalisingmpo(β; r = r)
    Ni = 2
    Nj = 2
    D = 50
    A = Array{Array,2}(undef, Ni, Nj)
    for i = 1:Ni,j = 1:Nj
        Random.seed!(1234)
        A[i,j] = randn(D, 2, D) + im*randn(D, 2, D)
    end
    λ, AL, C, AR, FL3, FR3, FL4, FR4 = vumps(A, M;verbose = true, tol = 1e-12, maxiter = 100)
    eMPS = energy(M, ME_row, ME_col, AL, C, AR, FL3,FL4, FR3, FR4,λM,λME_row, λME_col)
    eMC = -0.20337739109735598
    @test eMPS ≈ eMC atol = 1e-6
end