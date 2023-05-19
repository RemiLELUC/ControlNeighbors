"""
@Authors: Rémi LELUC, François Portier, Johan SEGERS and Aigerim ZHUMAN
"""

####################
# import libraries #
####################
import numpy as np
from scipy.stats import norm
from scipy.special import erf

def euro_vanilla_call(S0, K, T, r, 𝜎):
    """ Closed-form formula for European Call option in BS model
    Params:
    @S0 (float): initial stock/index level
    @K  (float): strike price
    @T  (float): time to maturity (in year fractions)
    @r  (float): constant risk-free short rate
    @𝜎  (float): volatility factor in diffusion term
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * 𝜎 ** 2) * T) / (𝜎 * np.sqrt(T))
    d2 = (np.log(S0 / K) + (r - 0.5 * 𝜎 ** 2) * T) / (𝜎 * np.sqrt(T))
    call = S0 * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    return call


def euro_vanilla_put(S0, K, T, r, 𝜎):
    """ Closed-form formula for European Put option in BS model
    Params:
    @S0 (float): initial stock/index level
    @K  (float): strike price
    @T  (float): time to maturity (in year fractions)
    @r  (float): constant risk-free short rate
    @𝜎  (float): volatility factor in diffusion term
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * 𝜎 ** 2) * T) / (𝜎 * np.sqrt(T))
    d2 = (np.log(S0 / K) + (r - 0.5 * 𝜎 ** 2) * T) / (𝜎 * np.sqrt(T))
    put = S0 * norm.cdf(-d1) + K * np.exp(-r*T) * norm.cdf(-d2)
    return put


def geometric_asian_call(S0, K, T, m, r, 𝜎):
    """ Exact formula (Black-Scholes) for Geometric average Asian Call option
    Params:
    @S0 (float) : initial stock/index level
    @K  (float) : strike price
    @T  (float) : time to maturity (in year fractions)
    @m    (int) : grid or granularity for time (in number of total points)
    @r   (float): constant risk-free short rate
    @𝜎   (float): volatility factor in diffusion term
    Returns:
    @geometric_value (float): true payoff
    """
    sigsqT = ((𝜎 ** 2 * T * (m + 1) * (2 * m + 1))
              / (6 * m * m))
    muT = (0.5 * sigsqT + (r - 0.5 * 𝜎 ** 2)
           * T * (m + 1) / (2 * m))
    d1 = ((np.log(S0 / K) + (muT + 0.5 * sigsqT))
          / np.sqrt(sigsqT))
    d2 = d1 - np.sqrt(sigsqT)
    N1 = 0.5 * (1 + erf(d1 / np.sqrt(2)))
    N2 = 0.5 * (1 + erf(d2 / np.sqrt(2)))
    geometric_value = np.exp(-r*T) * (S0 * np.exp(muT) * N1 - K * N2)
    return geometric_value


def down_call(kn,S0, H, K, T, r, 𝜎):
    """ Closed-form formula for Barrier Down Call option in BS model
    with H < K
    Params:
    @kn   (str): "in" or "out" barrier
    @S0 (float): initial stock/index level
    @H  (float): barrier price
    @K  (float): strike price
    @T  (float): time to maturity (in year fractions)
    @r  (float): constant risk-free short rate
    @𝜎  (float): volatility factor in diffusion term
    """
    nu = (r - 0.5 * 𝜎**2)
    power = 2*nu / (𝜎**2)
    C_plain =  euro_vanilla_call(S0=S0,K=K,T=T,r=r,𝜎=𝜎)
    C = euro_vanilla_call(S0=H**2/S0,K=K,T=T,r=r,𝜎=𝜎)
    if kn=='in':
        call = ((H/S0)**power) * C
    elif kn=='out':
        call = C_plain - ((H/S0)**power) * C
    return call

def up_call(kn,S0, H, K, T, r, 𝜎):
    """ Closed-form formula for Barrier Up Call option in BS model
    with H > K
    Params:
    @kn   (str): "in" or "out" barrier
    @S0 (float): initial stock/index level
    @H  (float): barrier price
    @K  (float): strike price
    @T  (float): time to maturity (in year fractions)
    @r  (float): constant risk-free short rate
    @𝜎  (float): volatility factor in diffusion term
    """
    nu = (r - 0.5 * 𝜎**2)
    power = 2*nu / (𝜎**2)
    R = (H/S0)**power
    
    d_SH = (np.log(S0/H) + nu*T) / (𝜎 * np.sqrt(T))
    d_HS = (np.log(H/S0) + nu*T) / (𝜎 * np.sqrt(T))
    
    A = (H-K)*np.exp(-r*T)*norm.cdf(d_SH)
    B = (H-K)*np.exp(-r*T)*norm.cdf(d_HS)
    C = (H-K)*np.exp(-r*T)*norm.cdf(-d_HS)
    
    
    C_plain_H =  euro_vanilla_call(S0=S0,K=H,T=T,r=r,𝜎=𝜎)
    C_plain_K =  euro_vanilla_call(S0=S0,K=K,T=T,r=r,𝜎=𝜎)
    C_frac_H = euro_vanilla_call(S0=H**2/S0,K=H,T=T,r=r,𝜎=𝜎)
    C_frac_K = euro_vanilla_call(S0=H**2/S0,K=K,T=T,r=r,𝜎=𝜎)
    P_frac_H = euro_vanilla_put(S0=H**2/S0,K=H,T=T,r=r,𝜎=𝜎)
    P_frac_K = euro_vanilla_put(S0=H**2/S0,K=K,T=T,r=r,𝜎=𝜎)
    
    if kn=='in':
        call = R*(P_frac_K-P_frac_H+C) + C_plain_H + A
    elif kn=='out':
        call = C_plain_K-C_plain_H-A- R*(C_frac_K-C_frac_H-B)
    return call


def barrier_call(S0, H, K, T, r, 𝜎, ud, io):
    """ Closed-form formula for Barrier Call option in BS model
    with H > K for Up-barrier, and H<K for Down-barrier
    Params:
    @S0 (float): initial stock/index level
    @H  (float): barrier price
    @K  (float): strike price
    @T  (float): time to maturity (in year fractions)
    @r  (float): constant risk-free short rate
    @𝜎  (float): volatility factor in diffusion term
    @ud   (str): "up" or "down" barrier
    @io   (str): "in" or "out" barrier
    """
    if ud=='up':
        call= up_call(kn=io,S0=S0, H=H, K=K, T=T, r=r, 𝜎=𝜎)
    elif ud=='down':
        call= down_call(kn=io,S0=S0, H=H, K=K, T=T, r=r, 𝜎=𝜎)
    return call


def euro_lookback_call(S0, K, T, r, 𝜎):
    """ Closed-form formula for European Lookback Call option in BS model
    Params:
    @S0 (float): initial stock/index level
    @K  (float): strike price
    @T  (float): time to maturity (in year fractions)
    @r  (float): constant risk-free short rate
    @𝜎  (float): volatility factor in diffusion term
    """
    d = (np.log(S0 / K) + (r + 0.5 * 𝜎 ** 2) * T) / (𝜎 * np.sqrt(T))
    temp = np.exp(r*T) * norm.cdf(d) - (S0/K)**(-2*r/(𝜎 ** 2)) * norm.cdf(d - (2*r/𝜎)*np.sqrt(T))
    call = S0 * norm.cdf(d) \
            - K * np.exp(-r*T) * norm.cdf(d - 𝜎*np.sqrt(T)) \
            + S0 * np.exp(-r*T) * ((𝜎 ** 2)/(2*r)) * temp
    return call