"""
@Authors: Rémi LELUC, François Portier, Johan SEGERS and Aigerim ZHUMAN
"""

####################
# import libraries #
####################
import numpy as np
from sklearn.neighbors import KDTree

##############################
# Define integrands 𝜑1 and 𝜑2#
##############################
# Sinusoidal function on [0,1]^d
def phi1(x):
    return np.sin(np.pi*(2*np.mean(x,axis=1) - 1))
# Sinusoidal function on R^d
def phi2(x):
    return np.sin(np.pi*np.mean(x,axis=1))


#################
# Main function #
#################
def run(seed):
    """ Run one simulation of Naive-MC against CVNN
    Params:
    @phi (func): integrand
    @seed (int): random seed for reproducibility
    @n    (int): sample size
    @d    (int): dimension of the problem
    Returns:
    @mc   (float): naive MC estimate
    @mcnn (float): CV-NN MC estimate
    """
    # draw n particles X1,...,Xn uniform on [0,1]^d
    if density=='normal':
        X = np.random.default_rng(seed=seed).normal(size=(n,d))
    elif density=='uniform':
        X = np.random.default_rng(seed=seed).uniform(size=(n,d))
    # compute evaluations g(X1),...,g(Xn)
    eval_phi = phi(X)
    # naive MC estimate
    mc = np.mean(eval_phi)
    ## Nearest Neighbor part KDTree
    # instance of KDTree to compute k-nearest neighbors
    kdt = KDTree(X, leaf_size = 20, metric='euclidean')
    # query the tree for the k nearest neighbors
    mask_nn = kdt.query(X, k=2, return_distance=False)[:,1]
    # compute evaluations 𝑔̂(X1),...,𝑔̂(Xn)
    mchat = np.mean(eval_phi[mask_nn])
    # evaluate integral of 𝑔̂ with MC of size N=n^2
    N = n**2
    if density=='normal':
        Xmc = np.random.default_rng(seed=seed).normal(size=(N,d))
    elif density=='uniform':
        Xmc = np.random.default_rng(seed=seed).uniform(size=(N,d))
    mask2 = kdt.query(Xmc, k=1, return_distance = False)
    mchatmc = np.mean(eval_phi[mask2])
    # compute CVNN estimate
    mcnn = mc - (mchat - mchatmc)
    return (mc, mcnn)


#####################################
# Tool function for Voronoi volumes #
#####################################
def intcv(XX):
    """ Given some particles X1,...,Xn compute the sample-extended grid
    and the volume associated to each element of the underlying partition
    The sample-extended grid is a grid of size n^d containing all points that write 
    (x_i1[1],x_i2[2]... x_id[d]) for all indexes i1,i2...id such that  1<=ik<=n
    This grid contains the sample points. Since the element of the grid are aligned with
    respect to the cartesian axis, a natural rectangle partition of [0,1]^d can be deduced
    from this grid.
    Params:
    @XX (array n x d): particles X1,...,Xn
    Returns:
    @pos_order (array): sample-extended grid
    @volumes   (array): volumes of each element of the partition
    """
    n,d = XX.shape
    # initialize grids
    x = np.sort(XX[:,0])
    xmid = (x[:(n-1)] + x[1:])/2
    xmid = np.append(np.append(0,xmid),1)
    xorder = x
    area = np.diff(xmid)
    for k in range(1,d):
        x = np.sort(XX[:,k])
        xmid = (x[:(n-1)] + x[1:])/2
        xmid = np.append(np.append(0,xmid),1)
        area = np.vstack([area,np.diff(xmid)])
        xorder = np.vstack([xorder,x])
    Xmeshorder = np.meshgrid(*xorder)
    areamesh = np.meshgrid(*area)
    # initialize final grids
    pos_order = Xmeshorder[0].ravel()
    volumes = areamesh[0].ravel()
    for i in range(1,d):
        pos_order = np.vstack([pos_order,Xmeshorder[i].ravel()])
        volumes = np.vstack([volumes,areamesh[i].ravel()])
    return (pos_order, volumes)

