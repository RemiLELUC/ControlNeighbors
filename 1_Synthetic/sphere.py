"""
@Authors: Rémi LELUC, François Portier, Johan SEGERS and Aigerim ZHUMAN
"""

####################
# import libraries #
####################
import numpy as np
from sklearn.neighbors import KDTree


def sample_sphere(seed,n):
    X = np.random.default_rng(seed=seed).uniform(size=(n,2))
    phi = np.arccos(2*X[:,0]-1)
    theta = X[:,1]*2*np.pi
    return np.array([x(phi,theta),y(phi,theta),z(phi,theta)]).T

cos = np.cos
sin = np.sin
tanh = np.tanh
exp = np.exp
## Test functions
x = lambda phi,theta: sin(phi)*cos(theta) 
y = lambda phi,theta: sin(phi)*sin(theta)
z = lambda phi,theta: cos(phi)


def f_3(p,t):
    return (1+tanh(-a*(x(p,t)+y(p,t)-z(p,t))))/(1.0*a)
    
def f_4(p,t):
    return (1-np.sign((x(p,t)+(y(p,t)-z(p,t)))))/(1.0*a)

def f_5(p,t):
    return (1-np.sign((np.pi*x(p,t)+y(p,t))))/(1.0*a)

a = 12 # Function parameter
## Exact integral values
f_3_exact = 4*np.pi/a
f_4_exact = 4*np.pi/a
f_5_exact = 4*np.pi/a

def f_3_vec(x_in):
    return (1+tanh(-a*(x_in[:,0]+x_in[:,1]-x_in[:,2])))/(1.0*a)

def f_4_vec(x_in):
    return (1-np.sign((x_in[:,0]+(x_in[:,1]-x_in[:,2]))))/(1.0*a)

def f_5_vec(x_in):
    return (1-np.sign((np.pi*x_in[:,0]+x_in[:,1])))/(1.0*a)

def MC_sphere(seed,f,n=1000):
    """ Run one simulation of naive MC on sphere S^2
    Params:
    @f   (func): integrand
    @n    (int): sample size
    @seed (int): random seed for reproducibility
    Returns:
    @mc (float): naive MC estimate
    """
    X = np.random.default_rng(seed=seed).uniform(size=(n,2))
    phi = np.arccos(2*X[:,0]-1)
    theta = X[:,1]*2*np.pi
    eval_f = f(phi,theta)
    mc = (4*np.pi) * np.mean(eval_f)
    return mc

def CVNN_sphere(seed,f,n):
    """ Run one simulation of CVNN
    Params:
    @f   (func): integrand
    @n    (int): sample size
    @seed (int): random seed for reproducibility
    Returns:
    @mcnn (float): CV-NN MC estimate
    """
    X = sample_sphere(seed=seed,n=n)
    eval_f = f(X)
    # naive MC estimate
    mc = np.mean(eval_f)
    ## Nearest Neighbor part KDTree
    # instance of KDTree to compute k-nearest neighbors
    kdt = KDTree(X, leaf_size = 10, metric='euclidean')
    # query the tree for the k nearest neighbors
    mask_nn = kdt.query(X, k=2, return_distance=False)[:,1]
    # compute evaluations 𝑔̂(X1),...,𝑔̂(Xn)
    mchat = np.mean(eval_f[mask_nn])
    # evaluate integral of 𝑔̂ with MC of size N=n^2
    N = int(n**2)
    Xmc = sample_sphere(seed=seed,n=N)
    mask2 = kdt.query(Xmc, k=1, return_distance = False)
    mchatmc = np.mean(eval_f[mask2])
    # compute CVNN estimate
    mcnn = mc - (mchat - mchatmc)
    return (4*np.pi) * mcnn

