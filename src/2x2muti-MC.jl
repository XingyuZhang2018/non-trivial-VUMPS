using Random, Statistics, ProgressMeter

#J1-J2-checkboard
function e_dif( array, lattice, x, y, r)
    a , b = mod(x,2), mod(y,2)
    
    top     = array[x, y + 1 - lattice * (y==lattice)] 
    bottom  = array[x,y - 1 + lattice * (y==1)]
    left    = array[x - 1 + lattice * (x==1), y]
    right   = array[x + 1 - lattice * (x==lattice), y]
    
    if a==1
        right *= r
    else 
        left *= r
    end
    if b==1
        top *= r
    else 
        bottom *= r
    end
    
#     if a==1
#         bottom *= r
#     else 
#         top *= r
#     end
#     if b==1
#         right *= r
#     else 
#         left *= r
#     end
    
    return 2 * array[x, y] * (top+bottom+left+right)
end

function onestep!(spin_array, lattice, β, r)
    for i = 1:lattice
        for j = 1:lattice
            e = e_dif(spin_array, lattice, i, j, r)
            if e <= 0
                spin_array[i,j] = -spin_array[i,j];
            elseif exp(-e*β) > rand()
                spin_array[i,j] = -spin_array[i,j];
            end
        end
    end
    return spin_array
end

function energy(spin_array, lattice, r)
    tol_en = 0
    for i = 1:lattice
        for j = 1:lattice
            e = e_dif(spin_array, lattice, i, j, r)
            tol_en += e
        end
    end
    return -tol_en/4
end

function MC(lattice,β,r,Sweeps_heat,Sweeps)
#     spin_array = ones(lattice,lattice)  #初始化自旋全部为1
    spin_array = (bitrand(lattice,lattice).-0.5)*2 #随机初始化自旋
    for j = 1:Sweeps_heat
        spin_array = onestep!(spin_array,lattice ,β,r);#局部更新
    end
#     mag = zeros(1,Sweeps);
    ene = zeros(1,Sweeps)
    Threads.@threads for j = 1:Sweeps
        spin_array = onestep!(spin_array,lattice ,β,r)
#         mag[j] = abs(sum(spin_array)/lattice^2)
        ene[j] = energy(spin_array, lattice,r)/lattice^2
    end
#     mag_ave = sum(mag)/Sweeps;
    ene_ave = sum(ene)/Sweeps;
    return ene_ave
end 

function mutiMC(β,r,lattice,bins,Sweeps_heat,Sweeps) 
    eMC_bins = zeros(bins,1)
    Threads.@threads for j=1:bins
        eMC_bins[j] =  MC(lattice,β,r,Sweeps_heat,Sweeps)
    end
    eMC = mean(eMC_bins)
    return eMC
end

function MutiMC(β,r,lattice,Bins,bins,Sweeps_heat,Sweeps) 
    eMC_Bins = zeros(1,Bins)
    p = Progress(Bins)
    for j = 1:Bins
        eMC_Bins[j] = mutiMC(β,r,lattice,bins,Sweeps_heat,Sweeps)
        update!(p,j)
    end
    eMC_m = mean(eMC_Bins)
    err = 1.96*sqrt(var(eMC_Bins)/Bins)
    return eMC_m,err
end