using Test

include("nontrivialVUMPS.jl")

@testset "environment  with (Ni,Nj)=($Ni $Nj)" for Ni in [1,2], Nj in [1,2]
    β = 0.1
    M, ME,λM,λME= classicalisingmpo(β; r = 1.0)
    D = 10
    A = Array{Array,2}(undef, Ni, Nj)
    for i = 1:Ni,j = 1:Nj
        Random.seed!(1234)
        A[i,j] = randn(D, 2, D) + im*randn(D, 2, D)
    end
    AL, L ,λL = leftorth(A)
    R, AR,λR = rightorth(A,L)
    FL3,FR3,λFL3,λFR3 = env3(AL,AR, M)
    FL4,FR4,λFL4,λFR4 = env4(AL,AR, M)
    for i = 1:Ni,j = 1:Nj
        ir = i + 1 - Ni * (i==Ni)
        irr = i + 2 - Ni * (i+2>Ni) - (Ni==1)
        XL3 = FLmap3(AL[i,:], AL[ir,:], M[i,:], FL3[i,j],j)
        @test XL3 ≈ FL3[i,j] * λFL3[i,j]
        XR3 = FRmap3(AR[i,:], AR[ir,:], M[i,:], FR3[i,j],j)
        @test XR3 ≈ FR3[i,j] * λFR3[i,j]
        XL4 = FLmap4(AL[i,:], AL[irr,:], M[i,:],M[ir,:], FL4[i,j],j)
        @test XL4 ≈ FL4[i,j] * λFL4[i,j]
        XR4 = FRmap4(AR[i,:], AR[irr,:], M[i,:],M[ir,:], FR4[i,j],j)
        @test XR4 ≈ FR4[i,j] * λFR4[i,j]
#         @test FL3[1,1] ≈ FL3[1,2] ≈ FL3[2,1] ≈ FL3[2,2]
#         @test FL4[1,1] ≈ FL4[1,2] ≈ FL4[2,1] ≈ FL4[2,2]
    end
end