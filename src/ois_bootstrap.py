"""Offline OIS bootstrapping, isolated from retrieval and the existing OIS reader.

The input is a dated collection of *par swap quotes*, not zero rates. QuantLib
constructs actual schedules and solves discount factors so each swap has zero
PV at its quoted fixed rate. We immediately detach its nodes into plain Python
values: later option calculations cannot accidentally use a changed global
QuantLib evaluation date. ACT/365F below is the curve time axis; the overnight
and fixed coupon accrual convention is ACT/360 for the supported indices.
"""
from dataclasses import dataclass
from datetime import date
import math
import threading

import numpy as np

# QuantLib has global evaluation settings. Serialize this module's bootstraps
# and restore those settings even when an instrument or solver raises an error.
_QL_LOCK = threading.RLock()


@dataclass(frozen=True)
class DiscountCurve:
    """Immutable log-linear discount nodes anchored at the observation date."""
    as_of_date: str
    currency: str
    index: str
    times: tuple
    discounts: tuple
    conventions: dict

    def discount(self, years):
        """Return D(valuation midnight, midnight + years); never extrapolate."""
        if not math.isfinite(years) or years < 0 or years > self.times[-1]:
            raise ValueError('discount_time_outside_curve')
        # Interpolate log D, not par yields or discount factors. Negative rates
        # may produce D > 1; that is valid and must not be clipped away.
        return float(np.exp(np.interp(years, self.times, np.log(self.discounts))))

    def interval_discount(self, start, end):
        """D(start,end)=D(0,end)/D(0,start), including intraday valuation times."""
        if end <= start:
            raise ValueError('nonpositive_discount_interval')
        return self.discount(end) / self.discount(start)


def bootstrap_ois(curve_input, conventions):
    """Build one dated SOFR/ESTR curve and return nodes plus swap repricing.

    Conventions are supplied explicitly by the new calibration JSON. They are
    recorded as assumptions unless verified against the quote source. No rates
    are fetched, no calendar dates are inferred from the computer's clock, and
    no market fixings are downloaded. This single-curve prototype projects and
    discounts the same overnight index; it does not implement a basis curve.
    """
    import QuantLib as ql

    expected = {'USD': 'SOFR', 'EUR': 'ESTR'}
    currency, index_name = curve_input['currency'], curve_input['overnight_index']
    if expected.get(currency) != index_name:
        raise ValueError('unsupported_currency_index_pair')
    if curve_input['quote_type'] != 'spot_starting_ois_par_swap_rate':
        raise ValueError('expected_par_ois_quotes')
    if curve_input['rate_unit'] != 'percent_per_annum':
        raise ValueError('expected_percent_rate_units')
    if conventions['status'] not in ('assumed', 'verified'):
        raise ValueError('missing_convention_status')
    for key in ('spot_lag_business_days', 'payment_lag_business_days'):
        if type(conventions[key]) is not int or conventions[key] < 0:
            raise ValueError('invalid_' + key)
    # Fail explicitly rather than silently interpreting unsupported settings.
    required = {'fixed_payment_frequency': 'Annual', 'day_count': 'ACT/360',
                'calendar': 'overnight_index_calendar',
                'business_day_convention': 'ModifiedFollowing',
                'payment_convention': 'Following', 'end_of_month': False}
    if any(conventions.get(k) != v for k, v in required.items()):
        raise ValueError('unsupported_ois_convention')
    observed = date.fromisoformat(curve_input['as_of_date'])
    points = curve_input['points']
    if not points or len({p['tenor'] for p in points}) != len(points):
        raise ValueError('empty_or_duplicate_ois_tenors')
    for point in points:
        value = point['rate_percent']
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('invalid_ois_rate')

    with _QL_LOCK, ql.SavedSettings():
        qdate = ql.Date(observed.day, observed.month, observed.year)
        ql.Settings.instance().evaluationDate = qdate
        index = ql.Sofr() if index_name == 'SOFR' else ql.Estr()
        calendar = index.fixingCalendar()
        if not calendar.isBusinessDay(qdate):
            raise ValueError('ois_observation_not_business_day')
        helpers = []
        for point in points:
            helper = ql.OISRateHelper(
                conventions['spot_lag_business_days'], ql.Period(point['tenor']),
                point['rate_percent'] / 100.0, index,
                telescopicValueDates=True,
                paymentLag=conventions['payment_lag_business_days'],
                paymentConvention=ql.Following, paymentFrequency=ql.Annual,
                paymentCalendar=calendar, endOfMonth=False,
                fixedPaymentFrequency=ql.Annual, fixedCalendar=calendar,
                convention=ql.ModifiedFollowing)
            helpers.append((point, helper))
        if len({h.pillarDate().serialNumber() for _, h in helpers}) != len(helpers):
            raise ValueError('duplicate_adjusted_curve_pillars')
        curve = ql.PiecewiseLogLinearDiscount(qdate, [h for _, h in helpers], ql.Actual365Fixed())
        # Accessing nodes forces lazy bootstrapping before evaluationDate is
        # restored. The earliest segment supplies the explicitly documented
        # short-end interpolation assumption; no overnight fixing is invented.
        nodes = list(curve.nodes())
        times = tuple((d - qdate) / 365.0 for d, _ in nodes)
        discounts = tuple(float(df) for _, df in nodes)
        if any(not math.isfinite(d) or d <= 0 for d in discounts):
            raise ValueError('nonpositive_bootstrapped_discount_factor')
        diagnostics = []
        for point, helper in helpers:
            implied = helper.impliedQuote()
            error_bp = (implied - point['rate_percent'] / 100.0) * 10000
            if not math.isfinite(error_bp) or abs(error_bp) > 0.001:
                raise ValueError('ois_repricing_failed')
            diagnostics.append(dict(tenor=point['tenor'],
                maturity_date=helper.maturityDate().ISO(), pillar_date=helper.pillarDate().ISO(),
                input_rate_percent=point['rate_percent'], repriced_rate_percent=implied * 100,
                error_basis_points=error_bp))
        node_rows = [dict(date=d.ISO(), years=t, discount_factor=df,
                         zero_rate_continuous=None if t == 0 else -math.log(df) / t)
                     for (d, _), t, df in zip(nodes, times, discounts)]
    return DiscountCurve(curve_input['as_of_date'], currency, index_name,
                         times, discounts, dict(conventions)), node_rows, diagnostics
