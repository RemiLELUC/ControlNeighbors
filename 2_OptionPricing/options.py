"""
@Authors: Rémi LELUC, François Portier, Johan SEGERS and Aigerim ZHUMAN
"""

####################
# import libraries #
####################
import numpy as np
from analytical import geometric_asian_call
from sklearn.neighbors import KDTree

class Option:
    ''' Base class for the various option classes.'''
    def __init__(self, model, K):
        self.model = model      # Black-Scholes or Meston model
        self.S0 = self.model.S0 # initial stock
        self.K = K
        self.T = self.model.T   # time to maturity
        self.r = self.model.r   # interest rate
        if self.model.𝜎!= None: # volatility
            self.𝜎 = self.model.𝜎 
        self.m = self.model.m   # discretization 
        self.discount = np.exp(-self.r*self.T)
        
    def MC_pricer(self, n, seed):
        '''A function for pricing options using Monte Carlo simulations
        Params:
        @self (Option): an option with a contract (payoff) function
        @n       (int): number of samples (MC size)
        @seed    (int): random seed for reproducibility
        Returns:
        @mc    (float): Monte Carlo estimate of option price
        '''
        # generate prices paths with model (Black-Sholes or Merton)
        paths = self.model.generate_paths(n=n, seed=seed)
        # compute the contracts (payoffs) 
        particles = self.contract(paths)
        # compute evaluataions phi(𝑋_1), ...,phi(𝑋_n) 
        eval_phi = np.clip(a=particles - self.K,a_min=0,a_max=None).ravel()
        # Monte Carlo estimate is average
        mc = np.mean(eval_phi)
        err_mc = np.std(a=eval_phi,ddof=1)
        return self.discount*mc,err_mc
        
        
    def CVNN_pricer(self, n, N, seed):
        '''A function for pricing options using CVNN simulations
        Params:
        @self (Option): an option with a contract (payoff) function
        @n       (int): number of samples (MC size)
        @seed    (int): random seed for reproducibility
        Returns:
        @mcnn    (float): CVNN MC-estimate of option price
        @err_mcnn(float): Estimated standard error or MC simulation
        '''
        # generate prices paths with model (Black-Sholes or Merton)
        paths = self.model.generate_paths(n=n, seed=seed)
        # compute the contracts (payoffs) 
        particles = self.contract(paths)
        # compute evalutaions phi(𝑋_1), ...,phi(𝑋_n) 
        eval_phi = np.clip(a=particles - self.K,a_min=0,a_max=None).ravel()
        # Monte Carlo estimate is average
        mc = np.mean(eval_phi)
        
        ### Nearest Neighbor part ###
        # Construct NN-tree on 
        kdt = KDTree(particles.reshape(-1,1), leaf_size = 10, metric='euclidean')
        # query the tree for the k nearest neighbors
        mask_nn = kdt.query(particles.reshape(-1,1), k=2, return_distance=False)[:,1]
        # compute evaluations 𝑔̂(X1),...,𝑔̂(Xn)
        eval_hat_phi = eval_phi[mask_nn]
        mchat = np.mean(eval_hat_phi)
        # evaluate integral of 𝑔̂ with MC of size N
        # generate prices paths with model (Black-Sholes or Merton)
        paths_mc = self.model.generate_paths(n=N, seed=seed)
        # compute the contracts 
        X_mc = self.contract(paths_mc)
        eval_phi_mc = np.clip(a=X_mc - self.K,a_min=0,a_max=None).ravel()
        mask_mc = kdt.query(X_mc.reshape(-1,1), k=1, return_distance = False)
        # compute MC estimate of 𝜇(𝑔̂)
        mchatmc = np.mean(eval_phi[mask_mc])
        # compute CVNN estimate
        eval_nn = eval_phi - 1*(eval_hat_phi - mchatmc)
        mcnn = np.mean(eval_nn)
        err_mcnn = np.std(eval_nn,ddof=1)
        return self.discount*mcnn, err_mcnn
         
    def CV_pricer(self, n, seed): 
        '''A function for pricing Asian options using Monte Carlo simulations
        with Geometric average control variates
        Params:
        @self (Option): an option with a contract (payoff) function
        @n       (int): number of samples (MC size)
        @seed    (int): random seed for reproducibility
        Returns:
        @cvmc    (float): Control Variates Monte Carlo estimate 
        @err_cvmc(float): Estimated standard error of CVMC simulation
        '''
        # generate prices paths with model (Black-Sholes or Merton)
        paths = self.model.generate_paths(n=n, seed=seed)
        # compute the contracts (payoffs) 
        particles = self.contract(paths)
        # compute evaluations phi(𝑋_1), ...,phi(𝑋_n) 
        eval_phi = np.clip(a=particles - self.K,a_min=0,a_max=None).ravel()
        # Monte Carlo estimate is average
        mc = np.mean(eval_phi)
        
        eval_geo = np.clip(a=self.control_contract(paths)- self.K,a_min=0,a_max=None).ravel()
        BS_geo_val = geometric_asian_call(S0=self.S0, K=self.K, 𝜎=self.𝜎,
                                          r=self.r, T=self.T, m=self.m)
        # compute optimal beta
        beta = np.cov(eval_geo,eval_phi)[0,1]/np.var(eval_geo)
        print('beta=',beta)
        eval_cvmc = eval_phi - beta*(eval_geo-BS_geo_val)
        cvmc = np.mean(eval_cvmc)
        err_cvmc = np.std(eval_cvmc,ddof=1)
        return self.discount*cvmc, err_cvmc
    
    
class European_call_option(Option):
    ''' European call option whose payoff is V(S_T) = (S_T - K)_+ '''
    def __init__(self, model, K):
        Option.__init__(self, model, K)
        
    def contract(self, path):
        return path[:,-1]

    
class Asian_call_option(Option):
    ''' European call option whose payoff is V(S_T) = (A_S - K)_+
    where A_S is the arithmetic average along the path
    '''
    def __init__(self, model, K):
        Option.__init__(self, model, K)
        
    def contract(self, path):
        avg_price = np.mean(path,1) # compute arithmetic mean along path
        return avg_price.ravel()
    
    def control_contract(self,path):
        geo_avg = np.exp(np.mean(np.log(path),1)) # compute geometric mean along path
        return geo_avg.ravel()

    
class Lookback_European_call_option(Option):
    ''' Lookback European call option whose payoff is V(S_T) = (S_T - K)_+ 
    '''
    def __init__(self, model, K):
        Option.__init__(self, model, K)
        
    def contract(self, path):
        max_price = np.max(path,1) # compute max along path
        return max_price.ravel()
    
    
class Barrier_call_option(Option):
    ''' Barrier call option whose payoff is path-dependent and deals
    with the barrier price H 
    up-and-in    (UI): V(S_T) = (S_T - K)_{+} 1{sup S_t > H}
    up-and-out   (UO): V(S_T) = (S_T - K)_{+} 1{sup S_t < H}
    down-and-in  (DI): V(S_T) = (S_T - K)_{+} 1{inf S_t < H}
    down-and-out (DO): V(S_T) = (S_T - K)_{+} 1{inf S_t > H}
    '''
    def __init__(self, model, K, H, ud, io):
        Option.__init__(self, model, K)
        self.K = K   # strike price
        self.H = H   # barrier price
        self.ud = ud # 'up' or 'down' barrier
        self.io = io # 'in' or 'out' barrier
        
    def contract(self, path):
        # compute indicator functions to check it barrier was hit
        if self.ud=='up': # Up Barrier Option
            if self.io=='in': # Up-In Call
                # is worthless unless spot price goes above barrier 
                # during its lifetime
                barrier_test = (path>self.H).any(axis=1)
            elif self.io=='out': # Up-Out Call
                # standard call option
                # unless the underlying spot price hits the barrier
                # in which case it becomes worthless
                barrier_test = (path<self.H).all(axis=1)
        elif self.ud=='down': # Down Barrier Option
            if self.io=='in': # Down-In Call
                # is worthless unless spot price goes below barrier 
                # during its lifetime
                barrier_test = (path<self.H).any(axis=1)
            elif self.io=='out': # Down-Out Call
                # standard call option
                # unless the underlying spot price hits the barrier
                # in which case it becomes worthless
                barrier_test = (path>self.H).all(axis=1)
        barrier_price = path[:,-1].ravel()
        return (barrier_price * barrier_test).ravel()
    
    
