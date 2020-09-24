using Test

include("..//src//nontrivialVUMPS.jl")

@testset "vumps test β=$β r=$r" for β in [10], r in [1 5 10]
    Ni = 3
    Nj = 3
    D = 50
    A = Array{Array,2}(undef, Ni, Nj)
    for i = 1:Ni,j = 1:Nj
        Random.seed!(1234)
        A[i,j] = randn(D, 2, D) + im*randn(D, 2, D)
    end
    M, ME_row, ME_col,λM,λME_row,λME_col= classicalisingmpo(β; r = r)
    λ, AL, C, AR, FL3, FR3, FL4, FR4 = vumps(A, M;verbose = true, tol = 1e-12, maxiter = 100)
    eMPS = energy(M, ME_row, ME_col, AL, C, AR, FL3,FL4, FR3, FR4,λM,λME_row, λME_col)
    @test eMPS ≈ -(14+4r)/9 atol = 1e-6
end