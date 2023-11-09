"""
@Authors: Rémi LELUC, François Portier, Johan SEGERS and Aigerim ZHUMAN
"""

####################
# import libraries #
####################
import numpy as np
from orthogonal import CVNN_mat

# number of replications
N_exp = 100
# sample size from n=10^1 to n=10^4
N_mc = np.ceil(np.logspace(1,4,20)).astype(np.int64)
# initialize results
I_mc = np.zeros((N_exp,len(N_mc)))
I_cv = np.zeros((N_exp,len(N_mc)))

###################################
# Run experiments with d=3 and k=1#
###################################
for (i,n) in enumerate(N_mc):
    print('n=',n)
    for j in range(N_exp):
        mcnn, mc = CVNN_mat(seed=j,k=1,d=3,n=n)
        I_mc[j,i] = mc
        I_cv[j,i] = mcvnn
# Save results
np.save('results/orthogonal/I_mc_O3_k1.npy',I_mc)
np.save('results/orthogonal/I_cv_O3_k1.npy',I_cv)

###################################
# Run experiments with d=3 and k=2#
###################################
I_mc = np.zeros((N_exp,len(N_mc)))
I_cv = np.zeros((N_exp,len(N_mc)))

for (i,n) in enumerate(N_mc):
    print('n=',n)
    for j in range(N_exp):
        mcnn, mc = CVNN_mat(seed=j,k=2,d=3,n=n)
        I_mc[j,i] = mc
        I_cv[j,i] = mcvnn
# Save results
np.save('results/orthogonal/I_mc_O3_k2.npy',I_mc)
np.save('results/orthogonal/I_cv_O3_k2.npy',I_cv)