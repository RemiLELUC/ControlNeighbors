"""
@Authors: Rémi LELUC, François Portier, Johan SEGERS and Aigerim ZHUMAN
"""

####################
# import libraries #
####################
import numpy as np
from sphere import CVNN_sphere
from sphere import f_prod_cos, f_sum_cos, exp_sphere

# number of replications
N_exp = 100
# sample size from n=10^1 to n=10^4
N_mc = np.ceil(np.logspace(1,4,20)).astype(np.int64)

I_mc = np.zeros((N_exp,len(N_mc)))
I_cv = np.zeros((N_exp,len(N_mc)))

for (i,n) in enumerate(N_mc):
    print('n=',n)
    for j in range(N_exp):
        mcnn, mc = CVNN_sphere(seed=j,f=f_prod_cos,n=n)
        I_mc[j,i] = mc
        I_cv[j,i] = mcnn
np.save('results/sphere/I_mc_prod_cos.npy',I_mc)
np.save('results/sphere/I_cv_prod_cos.npy',I_cv)

I_mc = np.zeros((N_exp,len(N_mc)))
I_cv = np.zeros((N_exp,len(N_mc)))

for (i,n) in enumerate(N_mc):
    print('n=',n)
    for j in range(N_exp):
        mcnn, mc = CVNN_sphere(seed=j,f=f_sum_cos,n=n)
        I_mc[j,i] = mc
        I_cv[j,i] = mcnn
np.save('results/sphere/I_mc_sum_cos.npy',I_mc)
np.save('results/sphere/I_cv_sum_cos.npy',I_cv)

I_mc = np.zeros((N_exp,len(N_mc)))
I_cv = np.zeros((N_exp,len(N_mc)))

for (i,n) in enumerate(N_mc):
    print('n=',n)
    for j in range(N_exp):
        mcnn, mc = CVNN_sphere(seed=j,f=exp_sphere,n=n)
        I_mc[j,i] = mc
        I_cv[j,i] = mcnn
np.save('results/sphere/I_mc_exp.npy',I_mc)
np.save('results/sphere/I_cv_exp.npy',I_cv)