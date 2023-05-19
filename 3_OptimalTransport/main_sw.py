"""
@Authors: Rémi LELUC, François Portier, Johan SEGERS and Aigerim ZHUMAN
"""

####################
# import libraries #
####################
import numpy as np
from sw import SW_MC, SW_CVNN


# Generate Data
d = 3             # dimension of the problem
n_sample = 2000   # number of samples for start/target distributions
np.random.seed(0)
mu_X = np.random.normal(size=d)  # mean of start gaussian
mu_Y = np.random.normal(size=d) 

𝜎1 = 2 
𝜎2 = 5
cov_X = (𝜎1**2) * np.eye(d) # covariance matrix of start gaussian
cov_Y = (𝜎2**2) * np.eye(d) #covariance matrix of target gaussian
# random generator with fixed random seed
gen = np.random.default_rng(seed=0)
# generate samples from "start" distribution
X_sample = gen.multivariate_normal(mean=mu_X,
                                  cov=cov_X,
                                  size=n_sample)
# generate samples from "target" distribution
Y_sample = gen.multivariate_normal(mean=mu_Y,
                                   cov=cov_Y,
                                   size=n_sample)

# compute true SW
I_true = np.sqrt((np.linalg.norm(mu_X-mu_Y,ord=2)**2)/d + (𝜎1-𝜎2)**2)
print('I_true=',I_true)

n_list = [50,100,250,500,1000]

N_exp = 100

res_mc_total = np.zeros((len(n_list),N_exp))
res_cv_total = np.zeros((len(n_list),N_exp))
# loop over number of random projections
for i,n in enumerate(n_list):
    print('n=',n)
    res_mc = np.zeros(N_exp)
    res_cv = np.zeros(N_exp)
    # loop over 100 different random seeds
    for s in range(N_exp):
        I_mc = SW_MC(X=X_sample,Y=Y_sample,seed=s,L=n,p=2)
        res_mc[s] = I_mc
        I_cv = SW_CVNN(X=X_sample,Y=Y_sample,seed=s,L=n,p=2,Nmc=int(n**2))
        res_cv[s] = I_cv
    res_mc_total[i] = res_mc
    res_cv_total[i] = res_cv
# Save results
np.save('I_true_d'+str(d)+'.npy',I_true)
np.save('sw_d'+str(d)+'.npy',res_mc_total)
np.save('swnn_d'+str(d)+'.npy',res_cv_total)