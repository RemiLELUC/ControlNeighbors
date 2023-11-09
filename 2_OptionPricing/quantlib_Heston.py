"""
@Authors: Rémi LELUC, François Portier, Johan SEGERS and Aigerim ZHUMAN
"""

####################
# import libraries #
####################
import numpy as np
import QuantLib as ql


spot = 100.0
#vol = 0.3
rate, divi_rate = 0.1, 0.0

T = 1/6 # 2 months
today = ql.Date(1, 1, 2023)
expiry_date = today + ql.Period(int(T*365), ql.Days)
day_count = ql.Actual365Fixed()
calendar = ql.NullCalendar()

riskFreeCurve = ql.FlatForward(today, rate, day_count)
diviCurve = ql.FlatForward(today, divi_rate, day_count)

flat_ts = ql.YieldTermStructureHandle(riskFreeCurve)
dividend_ts = ql.YieldTermStructureHandle(diviCurve)

# Creating and pricing the barrier option with analytic engine
#expiry_date = ql.Date(1, 1, 2023)
strike = 100.0
barrier_level = 130.0

initialValue = ql.QuoteHandle(ql.SimpleQuote(spot))
v0 = 0.1
kappa = 4
theta = 0.02
rho = 0.8
sigma = 0.9

barrier_out = ql.BarrierOption(barrierType=ql.Barrier.UpOut,
                               barrier=barrier_level, rebate=0.0,
                               payoff=ql.PlainVanillaPayoff(ql.Option.Call, strike),
                               exercise=ql.EuropeanExercise(expiry_date))

barrier_in = ql.BarrierOption(barrierType=ql.Barrier.UpIn,
                              barrier=barrier_level, rebate=0.0,
                              payoff=ql.PlainVanillaPayoff(ql.Option.Call, strike),
                              exercise=ql.EuropeanExercise(expiry_date))

hestonProcess = ql.HestonProcess(flat_ts, dividend_ts, initialValue, v0, kappa, theta, sigma, rho)
hestonModel = ql.HestonModel(hestonProcess)

engine_heston = ql.FdHestonBarrierEngine(hestonModel)
# price options
barrier_in.setPricingEngine(engine_heston)
barrier_out.setPricingEngine(engine_heston)
# Set initial value 
true_out = barrier_out.NPV()
true_in = barrier_in.NPV()

print('Up-Out Heston price:',true_out)
print('Up-In  Heston price:',true_in)