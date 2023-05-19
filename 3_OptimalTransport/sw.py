"""
@Authors: Rémi LELUC, François Portier, Johan SEGERS and Aigerim ZHUMAN
"""

####################
# import libraries #
####################
import numpy as np
from sklearn.neighbors import KDTree


def SW_MC(X, Y, seed, L=10, p=2):
    """
    Computes the Sliced-Wasserstein distance between empirical distributions
    Params:
    @X (m x d array): n samples in dimension d of initial distribution
    @Y (m x d array): m samples in dimension d of target distribution
    @seed      (int): random seed for reproducibility
    @L         (int): number of random projections (MC sample size)
    @p         (int): SW distance order (default p=2)
    Returns:
    @mc   (float): naive MC estimate of the SW distance
    """
    _,n_dim = X.shape
    order = p
    # Project data
    theta = np.random.default_rng(seed=seed).normal(size=(L, n_dim))
    theta = theta / (np.sqrt((theta ** 2).sum(axis=1)))[:, None]  # Normalize
    #print('theta.T shape:',(theta.T).shape)
    xproj = np.matmul(X, theta.T)
    yproj = np.matmul(Y, theta.T)
    diff = np.sort(xproj, 0) - np.sort(yproj, 0)
    eval_sw = np.mean(np.abs(diff) ** order,axis=0)
    mc = np.mean(eval_sw) ** (1/order)
    return mc

def SW_CVNN(X, Y,seed, L=10, p=2, Nmc=1000):
    """
    Computes the Sliced-Wasserstein distance between empirical distributions
    with control neighbors control variates
    Params:
    @X (m x d array): n samples in dimension d of initial distribution
    @Y (m x d array): m samples in dimension d of target distribution
    @seed      (int): random seed for reproducibility
    @L         (int): number of random projections (MC sample size)
    @p         (int): SW distance order (default p=2)
    @Nmc       (int): number of particles in CVNN procedure
    Returns:
    @mcnn    (float): CVNN estimate of the SW distance
    """
    _,n_dim = X.shape
    order = p
    # Project data
    theta = np.random.default_rng(seed=seed).normal(size=(L, n_dim))
    theta = theta / (np.sqrt((theta ** 2).sum(axis=1)))[:, None]  # Normalize
    xproj = np.matmul(X, theta.T)
    yproj = np.matmul(Y, theta.T)
    diff = np.sort(xproj, 0) - np.sort(yproj, 0)
    eval_sw = np.mean((np.abs(diff) ** order),axis=0)                     
    mc = eval_sw.mean()
    ## Nearest Neighbor part KDTree
    # instance of KDTree to compute k-nearest neighbors
    kdt = KDTree(theta, leaf_size = 10, metric='euclidean')
    # query the tree for the k nearest neighbors
    mask_nn = kdt.query(theta, k=2, return_distance=False)[:,1]
    # compute evaluations 𝜑̂ (X1),...,𝜑̂ (Xn)
    mchat = np.mean(eval_sw[mask_nn])
    # evaluate integral of 𝜑̂  with MC of size N=n^2
    theta_mc = np.random.default_rng(seed=seed).normal(size=(Nmc, n_dim))
    theta_mc = theta_mc / (np.sqrt((theta_mc ** 2).sum(axis=1)))[:, None]  # Normalize
    #theta_mc = np.transpose(theta_mc, (0, 2, 1))
    mask2 = kdt.query(theta_mc, k=1, return_distance = False)
    mchatmc = np.mean(eval_sw[mask2])
    # compute CVNN estimate
    mcnn = mc - (mchat - mchatmc)
    return mcnn** (1/order) #,mcnn_ols** (1/order)