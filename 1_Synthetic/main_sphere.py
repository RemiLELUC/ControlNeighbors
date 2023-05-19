"""
@Authors: Rémi LELUC, François Portier, Johan SEGERS and Aigerim ZHUMAN
"""

####################
# import libraries #
####################
import numpy as np
from sphere import MC_sphere, CVNN_sphere
from sphere import f_3,f_4,f_5
from sphere import f_3_vec,f_4_vec,f_5_vec

# number of replications
N_exp = 100
# sample size from n=10^1 to n=10^4
N_mc = np.ceil(np.logspace(1,4,20)).astype(np.int64)

I_mc = np.zeros((N_exp,len(N_mc)))
I_cv = np.zeros((N_exp,len(N_mc)))

for (i,n) in enumerate(N_mc):
    print('n=',n)
    for j in range(N_exp):
        I_mc[j,i] = MC_sphere(seed=j,f=f_3,n=n)
        I_cv[j,i] = CVNN_sphere(seed=j,f=f_3_vec,n=n)
np.save('results/sphere/I_mc_f3_sphere.npy',I_mc)
np.save('results/sphere/I_cv_f3_sphere.npy',I_cv)

I_mc = np.zeros((N_exp,len(N_mc)))
I_cv = np.zeros((N_exp,len(N_mc)))

for (i,n) in enumerate(N_mc):
    print('n=',n)
    for j in range(N_exp):
        I_mc[j,i] = MC(seed=j,f=f_4,n=n)
        I_cv[j,i] = CVNN_sphere(seed=j,f=f_4_vec,n=n)
np.save('results/sphere/I_mc_f4_sphere.npy',I_mc)
np.save('results/sphere/I_cv_f4_sphere.npy',I_cv)

I_mc = np.zeros((N_exp,len(N_mc)))
I_cv = np.zeros((N_exp,len(N_mc)))

for (i,n) in enumerate(N_mc):
    print('n=',n)
    for j in range(N_exp):
        I_mc[j,i] = MC(seed=j,f=f_5,n=n)
        I_cv[j,i] = CVNN_sphere(seed=j,f=f_5_vec,n=n)
np.save('results/sphere/I_mc_f5_sphere.npy',I_mc)
np.save('results/sphere/I_cv_f5_sphere.npy',I_cv)