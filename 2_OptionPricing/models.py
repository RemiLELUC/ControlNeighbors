"""
@Authors: Rémi LELUC, François Portier, Johan SEGERS and Aigerim ZHUMAN
"""

####################
# import libraries #
####################
import numpy as np

class Black_Scholes_Model:
    """ Instance of Black Scholes Model for option pricing 
    Params:
    @S0 (float): initial stock/index level
    @T  (float): time to maturity (in year fractions)
    @r  (float): constant risk-free short rate
    @𝜎  (float): constant volatility factor in diffusion term
    @m    (int): grid or granularity for time (number of discretization points)
    """
    def __init__(self, S0, T, r, 𝜎,m):
        self.S0 = S0 # initial stock
        self.T = T   # time to maturity
        self.r = r   # interest rate
        self.𝜎 = 𝜎   # volatility
        self.m = m   # discretization 
        
    def generate_paths(self, n, seed):
        """ Generate n paths (stock prices) of size m
        Params:
        @n       (int): number of paths (MC size)
        @seed    (int): random seed for reproducibility
        Returns
        @paths (array n x m): n paths of stock prices, each one composed of m time steps
        """
        X = np.random.default_rng(seed=seed).normal(size=(n,self.m)) 
        dt = self.T/self.m
        paths = self.S0 * np.cumprod (np.exp ((self.r - 0.5 * self.𝜎 ** 2) * dt +
                                self.𝜎 * np.sqrt(dt) * X), 1)
        return paths
    
class Heston_Model:
    """ Instance of Heston Model for option pricing 
    dS_t =    𝜇 S_t dt + sqrt(v_t) S_t dW_t^S
    dv_t = 𝜅(𝜃-v_t) dt + 𝜉 sqrt(v_t)   dW_t^v
    Params:
    @S0 (float): initial stock/index level
    @T  (float): time to maturity (in year fractions)
    @r  (float): constant risk-free short rate
    @v0 (float): initial variance
    @𝜃  (float): long run average variance
    @𝜅  (float): rate of mean reversion
    @𝜌  (float): instantaneous correlation in [-1;1]
    @𝜉  (float): volatility of volatility
    """
    def __init__(self, S0, T, r, v0, 𝜃, 𝜅 ,𝜌, 𝜉, m):
        self.S0 = S0 # initial stock
        self.T = T   # time to maturity
        self.r = r   # interest rate
        self.v0 = v0 # initial variance
        self.𝜃 = 𝜃   # long run average variance
        self.𝜅 = 𝜅   # mean reversion rate
        self.𝜌 = 𝜌   # correlation
        self.𝜉 = 𝜉   # vol of vol
        self.𝜎 = None
        self.m = m
        
    
    def generate_paths(self, n , seed, return_vol=False):
        dt = self.T/self.m
        size = (n, self.m)
        paths = np.zeros(size)
        sigs = np.zeros(size)
        S_t = self.S0
        v_t = self.v0
        means = np.array([0,0])
        covs = np.array([[1,self.𝜌],
                         [self.𝜌,1]])
        gen = np.random.default_rng(seed=seed)
        # Draw correlated Wiener processes W1 and W2
        WT = np.sqrt(dt) * gen.multivariate_normal(mean=means,
                                                   cov=covs, 
                                                   size=size)
        W1 = WT[:,:,0]
        W2 = WT[:,:,1]
        for t in range(self.m):
            # update stock price of asset
            S_t = S_t*(np.exp( (self.r- 0.5*v_t)*dt+ np.sqrt(v_t) *W1[:,t] ) ) 
            # update variance
            v_t = np.abs(v_t + self.𝜅*(self.𝜃-v_t)*dt + self.𝜉*np.sqrt(v_t)*W2[:,t])
            paths[:, t] = S_t
            sigs[:, t] = v_t
        if return_vol:
            return paths, sigs
        return paths