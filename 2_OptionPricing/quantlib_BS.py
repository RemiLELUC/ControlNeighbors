"""
@Authors: Rémi LELUC, François Portier, Johan SEGERS and Aigerim ZHUMAN
"""

####################
# import libraries #
####################
import numpy as np
import QuantLib as ql


# Parameters
spot = 100.0
vol = 0.3
rate, divi_rate = 0.1, 0.0

T = 1/6 # 2 months
today = ql.Date(1, 1, 2023)
expiry_date = today + ql.Period(int(T*365), ql.Days)
day_count = ql.Actual365Fixed()
calendar = ql.NullCalendar()

volatility = ql.BlackConstantVol(today, calendar, vol, day_count)
riskFreeCurve = ql.FlatForward(today, rate, day_count)
diviCurve = ql.FlatForward(today, divi_rate, day_count)

flat_ts = ql.YieldTermStructureHandle(riskFreeCurve)
dividend_ts = ql.YieldTermStructureHandle(riskFreeCurve)
flat_vol = ql.BlackVolTermStructureHandle(volatility)

# Creating and pricing the barrier option with analytic engine
strike = 100.0
barrier_level = 130.0

barrier_out = ql.BarrierOption(barrierType=ql.Barrier.UpOut,
                               barrier=barrier_level, rebate=0.0,
                               payoff=ql.PlainVanillaPayoff(ql.Option.Call, strike),
                               exercise=ql.EuropeanExercise(expiry_date))

barrier_in = ql.BarrierOption(barrierType=ql.Barrier.UpIn,
                              barrier=barrier_level, rebate=0.0,
                              payoff=ql.PlainVanillaPayoff(ql.Option.Call, strike),
                              exercise=ql.EuropeanExercise(expiry_date))

process = ql.BlackScholesProcess(ql.QuoteHandle(ql.SimpleQuote(spot)), flat_ts, flat_vol)
engine = ql.AnalyticBarrierEngine(process)
# price options
barrier_in.setPricingEngine(engine)
barrier_out.setPricingEngine(engine)
# Set initial value 
true_out = barrier_out.NPV()
true_in = barrier_in.NPV()

print('Up-Out price:',true_out)
print('Up-In  price:',true_in)