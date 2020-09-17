using Random, Statistics, PyPlot, ProgressMeter
#J1-J2 MC
function e_dif1( array, lattice, x, y, r)
    top     = array[x,y - 1 + lattice * (y==1)];
    bottom  = array[x, y + 1 - lattice * (y==lattice)];
    left    = r * array[x - 1 + lattice * (x==1), y];
    right   = array[x + 1 - lattice * (x==lattice), y];
    return 2 * array[x, y] * (left+right+top+bottom)
end

function e_dif2( array, lattice, x, y, r)
    top     = array[x,y - 1 + lattice * (y==1)];
    bottom  = array[x, y + 1 - lattice * (y==lattice)];
    left    = array[x - 1 + lattice * (x==1), y];
    right   = r * array[x + 1 - lattice * (x==lattice), y];
    return 2 * array[x, y] * (left+right+top+bottom)
end

function onestep!(spin_array, lattice, β, r)
    for i = 1:lattice
        for j = 1:lattice
            if mod(i,2) == 1
                e = e_dif1(spin_array, lattice, i, j, r)
            else
                e = e_dif2(spin_array, lattice, i, j, r)
            end
            if e <= 0
                spin_array[i,j] = -spin_array[i,j];
            elseif exp(-e*β) > rand()
                spin_array[i,j] = -spin_array[i,j];
            end
        end
    end
    return spin_array
end

function MC(lattice,β,r,Sweeps_heat,Sweeps)
    spin_array = ones(lattice,lattice);  #初始化自旋全部为1
    for j = 1:Sweeps_heat
        spin_array = onestep!(spin_array,lattice ,β,r);#局部更新
    end
#     mag = zeros(1,Sweeps);
    ene = zeros(1,Sweeps);
    for j = 1:Sweeps
        spin_array = onestep!(spin_array,lattice ,β,r)
#         mag[j] = abs(sum(spin_array)/lattice^2)
        ene[j] = energy(spin_array, lattice,r)/lattice^2
    end
#     mag_ave = sum(mag)/Sweeps;
    ene_ave = sum(ene)/Sweeps;
    return ene_ave
end 

function energy(spin_array, lattice, r)
    tol_en = 0
    for i = 1:lattice
        for j = 1:lattice
            if mod(i,2) == 1
                e = e_dif1(spin_array, lattice, i, j, r)
            else
                e = e_dif2(spin_array, lattice, i, j, r)
            end
            tol_en += e
        end
    end
    return -tol_en/4
end

#test4 muti-data
β= 0.1
# β = 0.1
r = 10
lB = 16;                    
lE = 64;                    
ld = 16;      
bins = 100
steps = Int(round((lE-lB)/ld+1));
eMC = zeros(steps,1)
err = zeros(steps,1)
p = Progress(bins)
for i=1:steps
    lattice = lB+ ld*(i-1);
    println("lattice  = $lattice ")
    eMC_bins = zeros(bins,1)
    # l = Threads.SpinLock()
    # update!(p,0)
    jj = Threads.Atomic{Int}(0)
    Threads.@threads for j=1:bins
        eMC_bins[j] =  MC(lattice,β,r,Int(1e4),Int(1e5))
        Threads.atomic_add!(jj, 1)
        Threads.lock(Threads.SpinLock())
        update!(p, jj[])
        Threads.unlock(Threads.SpinLock())
    end
    err[i] = 1.96*sqrt(var(eMC_bins)/bins)
    eMC[i] = mean(eMC_bins)
end

lattice = lB:ld:lE
errorbar(1/lattice,reverse(reshape(eMC,steps,)),yerr = reshape(err,steps,),marker="s", mfc="red",
                 mec="green", ms=20, mew=4)
legend(loc="best")
xlabel("lattice")
ylabel("energy")