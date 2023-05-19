"""
@Authors: Rémi LELUC, François Portier, Johan SEGERS and Aigerim ZHUMAN
"""

####################
# import libraries #
####################
import multiprocessing
import numpy as np

##############
# Parameters #
##############
N_exp = 100
seed_list = list(np.arange(N_exp))
nb_cores = 6

n_val = np.ceil(np.logspace(1,4,20)).astype(np.int64)
d_val = [2,3,4]
phi_ind = [1,2]
densities = ['uniform','normal']
exec(open("synthetic.py").read())

phi_list = [phi1,phi2]

# loop over integrands 𝜑1 and 𝜑2
for phi_num in phi_ind:
    phi = phi_list[phi_num-1]
    density = densities[phi_num-1]
    # loop over dimensions 
    for d in d_val:
        # loop over sample size n
        for n in n_val:
            # run (N_exp=100) replications 
            if __name__ == '__main__':
                pool = multiprocessing.Pool(processes = nb_cores)
                data = pool.map(run, seed_list)
                np.save(str("results/res_phi%d_d%d_n%d.npy" %(phi_num,d,n)),data)
                pool.close()
                pool.join()
