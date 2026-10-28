"""European forward pricing and IV inversion; no provider or archive dependencies.

Prices are per underlying unit (index points for SPX). Contract multipliers are
not applied to the inversion: they multiply both prices and payoffs equally.
The supplied discount factor and forward must share the option valuation date.
"""
import math
from numbers import Real
from scipy.optimize import brentq
from scipy.special import ndtr


def _positive(*values):
    return all(isinstance(x, Real) and not isinstance(x, bool) and math.isfinite(x) and x > 0 for x in values)


def equity_forward(spot, dividend_pv, discount):
    """Prepaid-forward approximation using PV of dividends through expiry.

    A broker dividend PV carries that broker's forecasting/discounting model.
    Combining it with our OIS discounting is an explicit approximation, not a
    reconstruction of an undisclosed IBKR model or a cashflow dividend schedule.
    """
    if not _positive(spot, discount) or (not isinstance(dividend_pv, Real) or isinstance(dividend_pv, bool)):
        raise ValueError('invalid_forward_inputs')
    if not math.isfinite(dividend_pv) or not 0 <= dividend_pv < spot:
        raise ValueError('invalid_dividend_pv')
    return (spot - dividend_pv) / discount


def black_price(forward, strike, years, discount, volatility, right='call'):
    """Black-Scholes expressed through F and D, for European calls or puts.

    Evaluate the out-of-the-money side first, then use parity. This avoids
    subtracting two large nearly equal intrinsic-value terms for deep ITM calls.
    At zero volatility the limiting price is discounted forward intrinsic value.
    """
    if not _positive(forward, strike, years, discount) or right not in ('call', 'put'):
        raise ValueError('invalid_black_inputs')
    if not isinstance(volatility, Real) or isinstance(volatility, bool) or not math.isfinite(volatility) or volatility < 0:
        raise ValueError('invalid_volatility')
    sign = 1 if right == 'call' else -1
    intrinsic = discount * max(sign * (forward - strike), 0)
    if volatility == 0:
        return intrinsic
    stddev = volatility * math.sqrt(years)
    d1 = math.log(forward / strike) / stddev + 0.5 * stddev
    d2 = d1 - stddev
    if forward >= strike:
        put = discount * (strike * ndtr(-d2) - forward * ndtr(-d1))
        return put if right == 'put' else put + discount * (forward - strike)
    call = discount * (forward * ndtr(d1) - strike * ndtr(d2))
    return call if right == 'call' else call + discount * (strike - forward)


def black_vega(forward, strike, years, discount, volatility):
    """Price change per 1.0 annualized volatility, not per percentage point."""
    if not _positive(forward, strike, years, discount, volatility):
        raise ValueError('invalid_vega_inputs')
    d1 = math.log(forward / strike) / (volatility * math.sqrt(years)) + 0.5 * volatility * math.sqrt(years)
    return discount * forward * math.exp(-0.5 * d1*d1) / math.sqrt(2*math.pi) * math.sqrt(years)


def implied_volatility(price, forward, strike, years, discount, right='call', *, maximum=5.0):
    """Find the annualized decimal IV, rejecting non-identifiable boundary prices.

    No tiny IV is invented when price equals intrinsic value. A finite search
    bound is a solver policy, not a clipping rule: failure remains a diagnostic.
    """
    if not _positive(price, forward, strike, years, discount, maximum):
        raise ValueError('nonpositive_or_nonfinite_pricing_input')
    lower = black_price(forward, strike, years, discount, 0.0, right)
    upper = discount * (forward if right == 'call' else strike)
    tolerance = 1e-12 * max(1.0, upper)
    if price <= lower + tolerance:
        raise ValueError('price_at_or_below_intrinsic_bound')
    if price >= upper - tolerance:
        raise ValueError('price_at_or_above_upper_bound')
    if black_price(forward, strike, years, discount, maximum, right) < price:
        raise ValueError('iv_above_solver_limit')
    return float(brentq(lambda vol: black_price(forward, strike, years, discount, vol, right) - price,
                        0.0, maximum, xtol=1e-12, rtol=1e-12))
