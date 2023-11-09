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
nb_cores = 8

n_val = [500, 1000, 2000, 3000, 5000]

exec(open("script_pricing.py").read())

for n_sample in n_val:
    if __name__ == '__main__':
        pool = multiprocessing.Pool(processes = nb_cores)
        data = pool.map(one_run, seed_list)
        #np.save(str("res_barrier/resUO_n%d.npy" %(n_sample)),data)
        #np.save(str("res_barrier/resUI_n%d.npy" %(n_sample)),data)
        np.save(str("res_barrier/resUO_Heston_n%d.npy" %(n_sample)),data)
        #np.save(str("res_barrier/resUI_Heston_n%d.npy" %(n_sample)),data)
        pool.close()
        pool.join()

