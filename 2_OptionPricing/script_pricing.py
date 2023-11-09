"""
@Authors: Rémi LELUC, François Portier, Johan SEGERS and Aigerim ZHUMAN
"""

####################
# import libraries #
####################
import numpy as np
from options import Barrier_call_option
from models import Black_Scholes_Model, Heston_Model

#####################
# Define Parameters #
#####################
S0 = 100     # Inital stock price
T = 1/6      # Maturity time in year
m = 240      # Time periods until maturity (number of trading days in a year)
r = 0.1      # Interest rate
#𝜎 = 0.3      # Volatility of the stock
K = 100      # Strike Price
up_H= 130

# instance of Black-Scholes Model
BS_model = Black_Scholes_Model(S0=S0, T=T, r=r, 𝜎=𝜎, m=m)


v0 = 0.1     # initial volatility of the stock
𝜃 = 0.02     # long run average variance
𝜅 = 4        # rate of mean reversion
𝜌 = 0.8      # instantaneous correlation
𝜉 = 0.9 
H_model = Heston_Model(S0=S0, T=T, r=r, v0=v0,
                        𝜃=𝜃, 𝜅=𝜅 ,𝜌=𝜌, 𝜉=𝜉, m=m)
# Up options / Change H_model to BS_model
barrier_UI = Barrier_call_option(model=H_model,K=K,H=up_H,
                                        ud='up', io='in')
barrier_UO = Barrier_call_option(model=H_model,K=K,H=up_H,
                                        ud='up', io='out')
#################
# Main function #
#################
def one_run(seed):
    mc = barrier_UO.MC_pricer(n=n_sample,seed=seed)
    mcnn = barrier_UO.CVNN_pricer(n=n_sample,N=int(2e5),seed=seed)
    return (mc,mcnn)
