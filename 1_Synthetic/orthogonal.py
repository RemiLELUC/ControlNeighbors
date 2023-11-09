"""
@Authors: Rémi LELUC, François Portier, Johan SEGERS and Aigerim ZHUMAN
"""

####################
# import libraries #
####################
import numpy as np
from sklearn.neighbors import KDTree
from scipy.stats import ortho_group

def MC_mat(seed,k,d,n):
    """ Run one simulation of Naive-MC to approximate
    \int_{O_d(R)} tr(X)^k dmu(X)
    Params:
    @seed (int): random seed for reproducibility
    @k    (int): sample size
    @d    (int): dimension of the problem
    @n    (int): sample size
    Returns:
    @mc   (float): naive MC estimate
    """
    # Sample uniformly distributed orthonal matrices
    X = ortho_group.rvs(dim=d,size=n,random_state=seed)
    # Evaluate
    eval_f = np.trace(a=X,axis1=1,axis2=2)**k
    # Average
    I_mc = np.mean(eval_f)
    return I_mc

def CVNN_mat(seed,k,d,n):
    """ Run one simulation of CVNN to approximate
    \int_{O_d(R)} tr(X)^k d mu(X)
    Params:
    @seed (int): random seed for reproducibility
    @k    (int): sample size
    @d    (int): dimension of the problem
    @n    (int): sample size
    Returns:
    @mcnn   (float): CVNN estimate
    """
    # Sample uniformly distributed orthonal matrices
    X = ortho_group.rvs(dim=d,size=n,random_state=seed)
    # Evaluate
    eval_f = np.trace(a=X,axis1=1,axis2=2)**k
    # naive MC estimate
    mc = np.mean(eval_f)
    ## Nearest Neighbor part KDTree
    X_mod = np.ravel(X).reshape(n,4)
    # instance of KDTree to compute k-nearest neighbors
    kdt = KDTree(X_mod, leaf_size = 10, metric='euclidean')
    # query the tree for the k nearest neighbors
    mask_nn = kdt.query(X_mod, k=2, return_distance=False)[:,1]
    # compute evaluations 𝑔̂(X1),...,𝑔̂(Xn)
    mchat = np.mean(eval_f[mask_nn])
    # evaluate integral of 𝑔̂ with MC of size N=n^2
    N = int(n**(1 + 2/d))
    Xmc = np.ravel(ortho_group.rvs(dim=d,size=N,random_state=seed)).reshape(N,4)
    mask2 = kdt.query(Xmc, k=1, return_distance = False)
    mchatmc = np.mean(eval_f[mask2])
    # compute CVNN estimate
    mcnn = mc - (mchat - mchatmc)
    return mcnn, mc