using Test

include("../src/nontrivialVUMPS.jl") 

@testset "vumpsstep" for Ni = [2,3],Nj = [2,3]
    β = 0.1
    M, ME,λM,λME= classicalisingmpo(β; r = 1.0)
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
    λ, AL, C, AR, errL, errR = vumpsstep(AL,C,FL3,FR3,M)
    for i = 1:Ni,j = 1:Nj
        @tensor AL2[b, c] := AL[i,j][a,s,b]*conj(AL[i,j][a,s,c])
        @test Array(AL2) ≈ I(D)
        @tensor AR2[b, c] := AR[i,j][b,s,a]*conj(AR[i,j][c,s,a])
        @test Array(AR2) ≈ I(D)
    end
end