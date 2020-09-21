using LinearAlgebra, TensorOperations, KrylovKit, Random

# define M,ME
function statmechmpo(β, h, D; r = 2.0)
    I = zeros(D,D,D,D)
    for i = 1:D
        I[i,i,i,i] = 1
    end
 
    X1 = zeros(D,D)
    for j = 1:D, i = 1:D
        X1[i,j] = exp(-β*h(i,j))
    end

    X2 = zeros(D,D)
    for j = 1:D, i = 1:D
        X2[i,j] = exp(-β*r*h(i,j))
    end

    X1sq = sqrt(X1)
    X2sq = sqrt(X2)
#     @show X1 X2 X1sq X2sq
    M = Array{Array,2}(undef, 2, 2)
    @tensor MT[a,b,c,d] := I[a',b',c',d']*X1sq[a',a]*X2sq[b',b]*X2sq[c',c]*X1sq[d',d]
    M[1,1] = MT
    M[1,2] = permutedims(MT,[3,2,1,4])
    M[2,1] = permutedims(MT,[1,4,3,2])
    M[2,2] = permutedims(MT,[3,4,1,2])

    # For computing energy: M2 is a tensor across 2 nearest neighbor sites in the lattice, whose
    # expectation value in the converged fixed point of the transfer matrix represents the energy
    Y1 = zeros(D,D)
    for j = 1:D, i = 1:D
        Y1[i,j] = h(i,j)*exp(-β*h(i,j))
    end
    Y2 = zeros(D,D)
    for j = 1:D, i = 1:D
        Y2[i,j] = r*h(i,j)*exp(-β*r*h(i,j))
    end
    
    ME = Array{Array,2}(undef, 2, 2)
    @tensor ME[1,1][a,b1,b2,c,d2,d1] := I[a',b1',c1,d1']*X1sq[a,a']*X2sq[b1,b1']*X1sq[d1',d1]* Y2[c1,c2]*
                                    I[c2,b2',c',d2']*X2sq[b2,b2']*X1sq[c',c]*X1sq[d2',d2]
    @tensor ME[1,2][a,b1,b2,c,d2,d1] := I[a',b1',c1,d1']*X2sq[a,a']*X1sq[b1,b1']*X2sq[d1',d1]* Y1[c1,c2]*
                                    I[c2,b2',c',d2']*X1sq[b2,b2']*X2sq[c',c]*X2sq[d2',d2]
    @tensor ME[2,1][d1,d2,s2,d3,d4,s1] := I[d1',c1,d4',s1']*X1sq[d1,d1']*X2sq[d4',d4]*X1sq[s1',s1]* Y2[c1,c2]*
                                    I[d2',s2',d3',c2]*X1sq[d2,d2']*X1sq[s2',s2]*X2sq[d3',d3]
    @tensor ME[2,2][d1,d2,s2,d3,d4,s1] := I[d1',c1,d4',s1']*X2sq[d1,d1']*X1sq[d4',d4]*X2sq[s1',s1]* Y1[c1,c2]*
                                    I[d2',s2',d3',c2]*X2sq[d2,d2']*X2sq[s2',s2]*X1sq[d3',d3]
    λM = norm(M)
    λME = norm(ME)
    return M/λM, ME/λME,λM,λME
end

classicalisingmpo(β; J = 1.0, h = 0.,r = 1.0) = statmechmpo(β, (s1,s2)->-J*(-1)^(s1!=s2) - h/2*(s1==1 + s2==1),2;r)

# function to get AL and AR
safesign(x::Number) = iszero(x) ? one(x) : sign(x)

"""
    qrpos(A)

Returns a QR decomposition, i.e. an isometric `Q` and upper triangular `R` matrix, where `R`
is guaranteed to have positive diagonal elements.
"""
qrpos(A) = qrpos!(copy(A))
function qrpos!(A)
    F = qr!(A)
    Q = Matrix(F.Q)
    R = F.R
    phases = safesign.(diag(R))
    rmul!(Q, Diagonal(phases))
    lmul!(Diagonal(conj!(phases)), R)
    return Q, R
end

"""
    lqpos(A)

Returns a LQ decomposition, i.e. a lower triangular `L` and isometric `Q` matrix, where `L`
is guaranteed to have positive diagonal elements.
"""
lqpos(A) = lqpos!(copy(A))
function lqpos!(A)
    F = qr!(Matrix(A'))
    Q = Matrix(Matrix(F.Q)')
    L = Matrix(F.R')
    phases = safesign.(diag(L))
    lmul!(Diagonal(phases), Q)
    rmul!(L, Diagonal(conj!(phases)))
    return L, Q
end

function cellones(Ni,Nj,D)
    Cell = Array{Array,2}(undef, Ni, Nj)
    for i = 1:Ni,j = 1:Nj
        Cell[i,j] = Matrix{Float64}(I, D, D)
    end
    return Cell
end

function ρmap(ρ,Ai,J)
    Nj = size(Ai,1)
    X = Array{Array,1}(undef, Nj+1)
    X[1] = ρ
    for j = 1:Nj
        jr = J+j-1 - (J+j-1 > Nj)*Nj
        @tensor X[j+1][a,b] := X[j][a',b']*Ai[jr][b',s,b]*
            conj(Ai[jr][a',s,a])
    end
    return X[Nj+1]
end
        
"""
    getL(ρ; kwargs...)
    ┌ A1─A2─    ┌       L ─
    ρ │  │    = ρ   =  │
    ┕ A1─A2─    ┕       L'─
ρ=L'*L, return L, where `L`is guaranteed to have positive diagonal elements.
L = cholesky!(ρ).U
If ρ is not exactly positive definite, cholesky will fail
"""
function getL!(A,L; kwargs...)
    Ni,Nj = size(A)
#     L = Array{Array,2}(undef, Ni, Nj)
    D = size(A[1,1],1)
    for i = 1:Ni,j = 1:Nj
        _,ρs,_ = eigsolve(ρ->ρmap(ρ,A[i,:],j), L[i,j]'*L[i,j], 1, :LM; ishermitian = false, maxiter = 1, kwargs...)
        ρ = ρs[1] + ρs[1]'
        ρ ./= tr(ρ)
        F = svd!(ρ)
        Lo = lmul!(Diagonal(sqrt.(F.S)), F.Vt)
        _, L[i,j] = qrpos!(Lo)
    end
    return L
end

"""
    getAL(A)

Given an MPS tensor `A` and `L` ，return a left-canonical MPS tensor `AL`, a gauge transform `R` and
a scalar factor `λ` such that ``λ AR R = L A``
"""
function getAL(A,L)
    Ni,Nj = size(A)
    AL = Array{Array,2}(undef, Ni, Nj)
    Le = Array{Array,2}(undef, Ni, Nj)
    λ = zeros(Ni, Nj)
    for i = 1:Ni,j = 1:Nj
        D, d, = size(A[1,1])
        Q, R = qrpos!(reshape(L[i,j]*reshape(A[i,j], D, d*D), D*d, D))
        AL[i,j] = reshape(Q, D, d, D)
        λ[i,j] = norm(R)
        Le[i,j] = rmul!(R, 1/λ[i,j])
    end
    return AL, Le,λ
end

function Lmap(L, A, AL)
    @tensor X[a,b] := L[a',b']*A[b',s,b]*conj(AL[a',s,a])
end

function getLsped(Le, A, AL; kwargs...)
    Ni,Nj = size(A)
    L = Array{Array,2}(undef, Ni, Nj)
    for i = 1:Ni,j = 1:Nj
        λs, Ls, _ = eigsolve(X->Lmap(X, A[i,j], AL[i,j]), Le[i,j], 1, :LM; ishermitian = false, kwargs...)
        _, L[i,j] = qrpos!(Ls[1])
    end
    return L
end

"""
    leftorth(A, [C]; kwargs...)

Given an MPS tensor `A`, return a left-canonical MPS tensor `AL`, a gauge transform `C` and
a scalar factor `λ` such that ``λ AL L = L A``, where an initial guess for `C` can be
provided.
"""
function leftorth(A,L=cellones(size(A,1),size(A,2),size(A[1,1],1)); tol = 1e-12, maxiter = 100, kwargs...)
    L = getL!(A,L; kwargs...)
    AL,Le,λ= getAL(A,L;kwargs...)
    numiter = 1
    while norm(L.-Le) > tol && numiter < maxiter
        L = getLsped(Le, A, AL; kwargs...)
        AL,Le,λ= getAL(A, L; kwargs...)
        numiter += 1
    end
    L = Le
    return AL, L,λ
end

"""
    rightorth(A, [C]; kwargs...)

Given an MPS tensor `A`, return a gauge transform C, a right-canonical MPS tensor `AR`, and
a scalar factor `λ` such that ``λ C AR^s = A^s C``, where an initial guess for `C` can be
provided.
"""
function rightorth(A,L=cellones(size(A,1),size(A,2),size(A[1,1],1)); tol = 1e-12, maxiter = 100, kwargs...)
    Ni,Nj = size(A)
    Ar = Array{Array,2}(undef, Ni, Nj)
    Lr = Array{Array,2}(undef, Ni, Nj)
    for i = 1:Ni,j = 1:Nj
        Ar[i,j] = permutedims(A[i,j],(3,2,1))
        Lr[i,j] = permutedims(L[i,j],(2,1))
    end
    AL, L,λ = leftorth(Ar,Lr; tol = tol, kwargs...)
    R = Array{Array,2}(undef, Ni, Nj)
    AR = Array{Array,2}(undef, Ni, Nj)
    for i = 1:Ni,j = 1:Nj
        R[i,j] = permutedims(L[i,j],(2,1))
        AR[i,j] = permutedims(AL[i,j],(3,2,1))
    end
    return R, AR,λ
end

# function to get FL and FR
function FLmap3(ALi,ALip, Mi, FL,J)
    Nj = size(ALi,1)
    X = Array{Array,1}(undef, Nj+1)
    X[1] = FL
    for j=1:Nj
        jr = J+j-1 - (J+j-1 > Nj)*Nj
        @tensor X[j+1][α,a,β] := X[j][α',a',β']*ALi[jr][β',s',β]*Mi[jr][a',s,a,s']*conj(ALip[jr][α',s,α])
    end
    return X[Nj+1]
end

function FRmap3(ARi,ARip, Mi, FR,J)
    Nj = size(ARi,1)
    X = Array{Array,1}(undef, Nj+1)
    X[1] = FR
    for j=1:Nj
        jr = J-(j-1) + (J-(j-1) < 1)*Nj
        @tensor X[j+1][α,a,β] := ARi[jr][α,s',α']*X[j][α',a',β']*Mi[jr][a,s,a',s']*conj(ARip[jr][β,s,β'])
    end
    return X[Nj+1]
end

function FLmap4(ALi, ALip, Mi, Mip, FL, J)
    Nj = size(ALi,1)
    X = Array{Array,1}(undef, Nj+1)
    X[1] = FL
    for j=1:Nj
        jr = J+j-1 - (J+j-1 > Nj)*Nj
        @tensor X[j+1][α,a,b,β] := X[j][α',a',b',β']*ALi[jr][β',s',β]*Mip[jr][a',s,a,c]*Mi[jr][b',c,b,s']*conj(ALip[jr][α',s,α])
    end
    return X[Nj+1]
end

function FRmap4(ARi, ARip, Mi, Mip, FR, J)
    Nj = size(ARi,1)
    X = Array{Array,1}(undef, Nj+1)
    X[1] = FR
    for j=1:Nj
        jr = J-(j-1) + (J-(j-1) < 1)*Nj
        @tensor X[j+1][α,a,b,β] := ARi[jr][α,s',α']*X[j][α',a',b',β']*Mi[jr][a,s,a',s']*Mip[jr][b,c,b',s]*conj(ARip[jr][β,c,β'])
    end
    return X[Nj+1]
end

function F3int(Ni,Nj,D,d)
    F3 = Array{Array,2}(undef, Ni, Nj)
    for i=1:Ni, j=1:Nj
        F3[i,j] = randn(Float64, D, d, D)
    end
    return F3
end

function F4int(Ni,Nj,D,d)
    F4 = Array{Array,2}(undef, Ni, Nj)
    for i=1:Ni, j=1:Nj
        F4[i,j] = randn(Float64, D, d, d, D)
    end
    return F4
end
    
"""
    leftenv(A, M, FL; kwargs)

Compute the left environment tensor for MPS A and MPO M, by finding the left fixed point
of A - M - conj(A) contracted along the physical dimension.
"""
function env3(AL,AR, M, FL = F3int(size(AL,1),size(AL,2),size(AL[1,1],1),size(M[1,1],1)),
                    FR = F3int(size(AR,1),size(AR,2),size(AR[1,1],1),size(M[1,1],1)); kwargs...)
    Ni,Nj = size(AL)
    λFL = zeros(Ni,Nj) + im * zeros(Ni,Nj)
    λFR = zeros(Ni,Nj) + im * zeros(Ni,Nj)
    for i=1:Ni,j=1:Nj
        ir = i + 1 - Ni * (i==Ni)
        λL1s, FL1s, = eigsolve(X->FLmap3(AL[i,:], AL[ir,:], M[i,:], X,j), FL[i,j], 1, :LM; ishermitian = false, kwargs...)
        FL[i,j] = FL1s[1]
        λFL[i,j] = λL1s[1]
        λR1s, FR1s, = eigsolve(X->FRmap3(AR[i,:], AR[ir,:], M[i,:], X,j), FR[i,j], 1, :LM; ishermitian = false, kwargs...)
        FR[i,j] = FR1s[1]
        λFR[i,j] = λR1s[1]
    end
    return FL,FR,λFL,λFR
end

function env4(AL,AR, M, FL = F4int(size(AL,1),size(AL,2),size(AL[1,1],1),size(M[1,1],1)),
                    FR = F4int(size(AR,1),size(AR,2),size(AR[1,1],1),size(M[1,1],1)); kwargs...)
    Ni,Nj = size(AL)
    λFL = zeros(Ni,Nj) + im * zeros(Ni,Nj)
    λFR = zeros(Ni,Nj) + im * zeros(Ni,Nj)
    for i=1:Ni,j=1:Nj
        ir = i + 1 - Ni * (i==Ni)
        irr = i + 2 - Ni * (i+2>Ni) - (Ni==1)
        λL1s, FL1s, = eigsolve(X->FLmap4(AL[i,:], AL[irr,:], M[i,:], M[ir,:], X, j), FL[i,j], 1, :LM; ishermitian = false, kwargs...)
        FL[i,j] = FL1s[1]
        λFL[i,j] = λL1s[1]
        λR1s, FR1s, = eigsolve(X->FRmap4(AR[i,:], AR[irr,:], M[i,:], M[ir,:], X, j), FR[i,j], 1, :LM; ishermitian = false, kwargs...)
        FR[i,j] = FR1s[1]
        λFR[i,j] = λR1s[1]
    end
    return FL,FR,λFL,λFR
end

# function to get energy
function energy_row(M, ME, AL, C, AR, FL3, FR3,λM,λME,i,j)
    Ni,Nj = size(M)
    ir = i + 1 - (i==Ni)*Ni
    jr = j + 1 - (j==Nj)*Nj
    @tensor AAC1[α,s1,s2,β] := AL[i,j][α,s1,α']*C[i,j][α',β']*AR[i,jr][β',s2,β]
    @tensor AAC2[α,s1,s2,β] := AL[ir,j][α,s1,α']*C[ir,j][α',β']*AR[ir,jr][β',s2,β]
    @tensor Z2 = scalar(FL3[i,j][α,c,β]*AAC1[β,s1,s2,β']*M[i,j][c,t1,d,s1]*
        M[i,jr][d,t2,c',s2]*FR3[i,jr][β',c',α']*conj(AAC2[α,t1,t2,α']))
    @tensor e = scalar(FL3[i,j][α,c,β]*AAC1[β,s1,s2,β']*ME[c,t1,t2,c',s2,s1]*
        FR3[i,jr][β',c',α']*conj(AAC2[α,t1,t2,α']) / Z2)
    return e/2/λM^2*λME
end

function energy_col(M, ME, AL, C, AR, FL4, FR4,λM,λME, i, j)
    Ni,Nj = size(M)
    jr = j + 1 - (j==Nj)*Nj
    ir = i + 1 - (i==Ni)*Ni
    irr = i + 2 - (i+2>Ni)*Ni - (Ni==1)
    @tensor AC1[β,s,β'] := AL[i,j][β,s,d1]*C[i,j][d1,β']
    @tensor AC2[β,s,β'] := AL[irr,j][β,s,d1]*C[i,j][d1,β']
    @tensor Z2 = scalar(FL4[i,j][α,a,b,β]*AC1[β,s,β']*M[i,j][b,c,b',s]*
        M[ir,j][a,s',a',c]*FR4[i,j][β',b',a',α']*conj(AC2[α,s',α']))
    @tensor e = scalar(FL4[i,j][α,a,b,β]*AC1[β,s,β']*ME[b,a,s',a',b',s]*
        FR4[i,j][β',b',a',α']*conj(AC2[α,s',α']) / Z2)
    return e/2/λM^2*λME
end

function energy(M, ME, AL, C, AR, FL3,FL4, FR3, FR4,λM,λME)
    e11 = energy_row(M, ME[1,1], AL, C, AR, FL3, FR3,λM,λME,1,1)
    e12 = energy_row(M, ME[1,2], AL, C, AR, FL3, FR3,λM,λME,2,2)
    e21 = energy_col(M, ME[2,1], AL, C, AR, FL4, FR4,λM,λME,1,1)
    e22 = energy_col(M, ME[2,2], AL, C, AR, FL4, FR4,λM,λME,2,2)
#     @show e11 e12 e21 e22
    return e11+e12+e21+e22
end

# vumps
function applyH1(ACij, FLj, FRj, Mj, II)
    Ni = size(FLj,1)
    X = Array{Array,1}(undef, Ni+1)
    X[1] = ACij
    for i=1:Ni
        ir = II+i-1 - (II+i-1 > Ni)*Ni
        @tensor X[i+1][α,s,β] := FLj[ir][α,a,α']*X[i][α',s',β']*Mj[ir][a,s,b,s']*FRj[ir][β',b,β]
    end
    return X[Ni+1]
end

function applyH0(Cij, FLjp, FRj, II)
    Ni = size(FLjp,1)
    X = Array{Array,1}(undef, Ni+1)
    X[1] = Cij
    for i=1:Ni
        ir = II+i-1 - (II+i-1 > Ni)*Ni
        @tensor X[i+1][α,β] := FLjp[ir][α,a,α']*X[i][α',β']*FRj[ir][β',a,β]
    end
    return X[Ni+1]
end

"""
    function vumpsstep(AL, C, AR, FL, FR; kwargs...)

Perform one step of the VUMPS algorithm
"""
function vumpsstep(AL,C,FL3,FR3,M;kwargs...)
    Ni,Nj = size(AL)
    D,d, = size(AL[1,1])
    AC = Array{Array,2}(undef, Ni,Nj)
    AR = Array{Array,2}(undef, Ni,Nj)
    λ = zeros(Ni,Nj) + im*zeros(Ni,Nj)
    errL = zeros(Ni,Nj)
    errR = zeros(Ni,Nj)
    
    for i = 1:Ni,j = 1:Nj
        @tensor AC[i,j][a,s,b] := AL[i,j][a,s,c] * C[i,j][c, b]
        jr = j + 1 - (j==Nj) * Nj
        μACs, ACs, = eigsolve(X->applyH1(X, FL3[:,j], FR3[:,j], M[:,j], i), AC[i,j], 1, :LM; ishermitian = false, maxiter = 10, kwargs...)
        μCs, Cs, = eigsolve(X->applyH0(X, FL3[:,jr], FR3[:,j], i), C[i,j], 1, :LM; ishermitian = false, maxiter = 10, kwargs...)
        λ[i,j] = μACs[1]/μCs[1]
        AC[i,j] = ACs[1]
        C[i,j] = Cs[1]

        QAC, RAC = qrpos(reshape(AC[i,j],(D*d, D)))
        QC, RC = qrpos(C[i,j])
        AL[i,j] = reshape(QAC*QC', (D, d, D))
        errL[i,j] = norm(RAC-RC)
    end
    
    for i=1:Ni, j=1:Nj
        jr = j - 1 + (j==1)*Nj
        LAC, QAC = lqpos(reshape(AC[i,j],(D, d*D)))
        LC, QC = lqpos(C[i,jr])
        AR[i,j] = reshape(QC'*QAC, (D, d, D))
        errR[i,j] = norm(LAC-LC)
    end
    return λ, AL, C, AR, errL, errR
end
    
function erro(AL,C,FL3,FR3,M)
    Ni,Nj = size(AL)
    AC = Array{Array,2}(undef, Ni,Nj)
    err = 0
    for i = 1:Ni,j = 1:Nj
        @tensor AC[i,j][a,s,b] := AL[i,j][a,s,c] * C[i,j][c, b]
        MAC = applyH1(AC[i,j], FL3[:,j], FR3[:,j], M[:,j], i)
        @tensor MAC[a,s,b] -= AL[i,j][a,s,b']*(conj(AL[i,j][a',s',b'])*MAC[a',s',b])
        err += norm(MAC)
    end
    return err
end
    
function vumps(A, M;verbose = false, tol = 1e-6, maxiter = 100, kwargs...)
    Ni,Nj = size(A)
    AL, L ,λL = leftorth(A)
    R, AR,λR = rightorth(A,L)

    FL3,FR3,λFL3,λFR3 = env3(AL,AR, M; tol = tol/10, kwargs...)
    
    C = Array{Array,2}(undef, Ni,Nj)
    for i = 1:Ni,j = 1:Nj
        jr = j + 1 - (j+1>Nj) * Nj 
        C[i,j] = L[i,j] * R[i,jr]
    end
    
    λ, AL, C, AR, errL, errR = vumpsstep(AL,C,FL3,FR3,M;tol = tol/10,kwargs...)
    FL3,FR3,λFL3,λFR3 = env3(AL,AR, M, FL3, FR3; tol = tol/10, kwargs...)
#     FR1 ./= @tensor scalar(FL2[c,b,a]*C1[a,a']*conj(C1[c,c'])*FR1[a',b,c']) 
#     FR2 ./= @tensor scalar(FL1[c,b,a]*C2[a,a']*conj(C2[c,c'])*FR2[a',b,c']) # normalize FL and FR: not really necessary

    # Convergence measure: norm of the projection of the residual onto the tangent space
    err = erro(AL,C,FL3,FR3,M)
    
    i = 1
#     λ =  λ1 * λ2
    verbose && println("Step $i: err ≈ $err")

    while err > tol && i < maxiter
        λ, AL, C, AR, errL, errR = vumpsstep(AL,C,FL3,FR3,M;tol = tol/10,kwargs...)
        FL3,FR3,λFL3,λFR3 = env3(AL,AR, M, FL3, FR3; tol = tol/10, kwargs...)
#         FR1 ./= @tensor scalar(FL2[c,b,a]*C1[a,a']*conj(C1[c,c'])*FR1[a',b,c']) 
#         FR2 ./= @tensor scalar(FL1[c,b,a]*C2[a,a']*conj(C2[c,c'])*FR2[a',b,c']) # normalize FL and FR: not really necessary
        err = erro(AL,C,FL3,FR3,M)
        i += 1
#         λ =  λ1 * λ2
        verbose && println("Step $i: err ≈ $err")
    end
    FL4,FR4,λFL4,λFR4 = env4(AL,AR, M; tol = tol/10, kwargs...)
    return λ, AL, C, AR, FL3, FR3, FL4, FR4
end