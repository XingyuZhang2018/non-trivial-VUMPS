using Test

include("..//src//nontrivialVUMPS.jl")

@testset "energy test" begin
    β = 0.1
    M, ME,λM,λME= classicalisingmpo(β; r = 1.0)
    Ni = 2
    Nj = 2
    D = 50
    A = Array{Array,2}(undef, Ni, Nj)
    for i = 1:Ni,j = 1:Nj
        Random.seed!(1234)
        A[i,j] = randn(D, 2, D) + im*randn(D, 2, D)
    end
    AL, L ,λL = leftorth(A)
    R, AR,λR = rightorth(A,L)
    FL3,FR3,λFL3,λFR3 = env3(AL,AR, M)
    FL4,FR4,λFL4,λFR4 = env4(AL,AR, M)
    C = Array{Array,2}(undef, Ni,Nj)
    for i = 1:Ni,j = 1:Nj
        jr = j + 1 - Nj * (j+1>Nj)
        C[i,j] = L[i,j] * R[i,jr]
    end
    e11 = energy_row(M, ME[1,1], AL, C, AR, FL3, FR3,λM,λME,1,1)
    e12 = energy_row(M, ME[1,2], AL, C, AR, FL3, FR3,λM,λME,2,2)
    e21 = energy_col(M, ME[2,1], AL, C, AR, FL4, FR4,λM,λME,1,1)
    e22 = energy_col(M, ME[2,2], AL, C, AR, FL4, FR4,λM,λME,2,2)
#     @show e11 e12 e21 e22
#     @show e11+e12+e21+e22
    @test C[1,1] ≈ C[1,2] ≈ C[2,1] ≈ C[2,2]
    @test e11 ≈ e12
    @test e21 ≈ e22
end