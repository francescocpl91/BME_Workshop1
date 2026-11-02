"""Shared offline historical inputs and portfolio arithmetic.

The historical and physical local-volatility engines consume the SAME aligned
log-return conventions. This file has no random simulation, provider calls,
option fitting or output writes. Private aliases remain in monte_carlo.py for
backward compatibility with existing callers and tests.
"""
from copy import deepcopy
import math
import numpy as np
import pandas as pd
import exchange_calendars as xcals
from .statistics import _day, _prepare, _basis_kind

class SimulationUnavailable(ValueError):
    """Valid configuration, but the archived inputs cannot support a simulation."""


class FXUnavailable(SimulationUnavailable):
    """Missing/unusable conversion factors permit an explicit currency fallback."""


def _sessions(calendar,start,end):
    """Exchange sessions, with no artificial weekend/holiday observations."""
    # Extend calendar construction beyond both endpoints. A weekend boundary
    # otherwise lies outside the calendar's first/last session and raises.
    first = (pd.Timestamp(start)-pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    last = (pd.Timestamp(end)+pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    cal = xcals.get_calendar(calendar,start=first,end=last)
    return cal.sessions_in_range(start,end).strftime("%Y-%m-%d").tolist()


def _root_covariance(covariance):
    """PSD factor, including deterministic/singular cases without invented noise."""
    # Symmetric eigen decomposition C = Q diag(lambda) Q.T. This is NOT
    # Cholesky: no lower-triangular matrix is constructed. Both approaches can
    # generate the same Gaussian covariance, but eigen decomposition also
    # supports singular covariance (constant/perfectly dependent factors).
    eigenvalues,eigenvectors = np.linalg.eigh(covariance)
    # Scale the numerical tolerance to the matrix. Materially negative modes
    # are invalid; tiny negative round-off is not treated as economic variance.
    tolerance = max(float(np.max(np.abs(eigenvalues)))*1e-12,1e-24)
    if np.min(eigenvalues) < -tolerance:
        raise ValueError("Calibration covariance is not positive semidefinite.")
    # Retain only non-negligible stochastic modes. R may be smaller than F,
    # including R=0. Do not invent noise to make a singular matrix invertible.
    positive = eigenvalues > tolerance
    # Broadcast sqrt(lambda) across Q's columns: B = Q sqrt(Lambda), (F,R).
    # B B.T reproduces C up to tolerance. Eigenvectors are essential; using
    # only diagonal eigenvalues would discard cross-factor dependence.
    return eigenvectors[:,positive]*np.sqrt(eigenvalues[positive])


def _holding_inputs(processed,cfg,risk_window=None):
    """Build validated per-holding log intervals; ignore any post-COB rows."""
    # Copy composition so later metadata/rounding reconciliation cannot mutate
    # the cleaned input portfolio or its configured weights.
    portfolio = deepcopy(processed.summary["portfolio"])
    all_holdings = portfolio["holdings"]
    if (any(not isinstance(h.get("request_id"),str) or not h["request_id"]
            or type(h.get("weight")) not in (int,float) or not math.isfinite(h["weight"])
            or not 0 <= h["weight"] <= 1 for h in all_holdings)
            or len({h["request_id"] for h in all_holdings}) != len(all_holdings)):
        raise ValueError("Holdings need unique IDs and finite nonnegative weights.")
    # Zero-weight references do not enter simulated holdings. Every positive
    # holding must be present; do not silently redistribute a missing holding.
    holdings = [h for h in all_holdings if h["weight"] > 0]
    if not holdings or not math.isclose(math.fsum(h["weight"] for h in holdings),1,abs_tol=1e-8,rel_tol=0):
        raise ValueError("Positive holding weights must sum to one.")
    if not isinstance(portfolio.get("reporting_currency"),str) or not portfolio["reporting_currency"]:
        raise ValueError("A reporting currency is required.")
    # Baseline ends at COB; named stress uses its own end <= COB. Lookback
    # is measured in calendar years, but observations are exchange intervals.
    cob = _day(processed.cob)
    end = _day(risk_window["end"]) if risk_window else cob
    start = (_day(risk_window["start"]) if risk_window else cfg["calibration"]["start"] or
             (pd.Timestamp(cob)-pd.DateOffset(years=cfg["calibration"]["lookback_years"])).strftime("%Y-%m-%d"))
    if end > cob:
        raise SimulationUnavailable("Stress calibration end must not be after archive COB.")
    if start >= end:
        raise SimulationUnavailable("Calibration start must precede its end/COB.")
    sessions = _sessions(cfg["calendar"],start,end)
    # Only consecutive exchange-session pairs count as daily observations.
    # Check BOTH endpoints: matching end dates alone could mix unequal periods.
    intervals = set(zip(sessions[:-1],sessions[1:]))
    data = processed.tables["selected_prices"]
    parts,metadata = [],[]
    for h in holdings:
        key = h["request_id"]
        rows = data.loc[(data["instrument_id"] == key) & (data["date"] <= end)].copy()
        if rows.empty:
            raise SimulationUnavailable("No selected history for "+key)
        # Require a single denomination/return meaning. Mixing currencies or
        # price-only and total-return histories would invalidate calibration.
        currencies = set(rows["currency"].dropna())
        bases = set(rows["return_basis"].dropna())
        if len(currencies) != 1 or len(bases) != 1:
            raise SimulationUnavailable("Unresolved currency/return basis for "+key)
        if "calendar" in rows and set(rows["calendar"].dropna()) - {cfg["calendar"]}:
            raise SimulationUnavailable("Holding calendar differs from configured simulation calendar: "+key)
        # Shared statistics helper orders observations and derives simple/log
        # returns. Then retain only valid one-session intervals in this window.
        rows = _prepare(rows)
        valid = rows.loc[rows["log_return"].notna()].copy()
        valid = valid.loc[[(a,b) in intervals for a,b in zip(valid["return_start_date"],valid["date"])]]
        factor = "asset::"+key
        # Two-date index aligns equal holding periods. asset:: distinguishes
        # holding factors from FX factors even when their identifiers resemble.
        parts.append(valid.set_index(["return_start_date","date"])["log_return"].rename(factor))
        metadata.append(dict(instrument_id=key,factor=factor,currency=next(iter(currencies)),
                             return_basis=next(iter(bases)),weight=h["weight"]))
    kinds = {_basis_kind(m["return_basis"]) for m in metadata}
    if "unknown" in kinds or len(kinds) != 1:
        raise SimulationUnavailable("Holdings have incompatible return meanings.")
    # Complete-case alignment: every retained row contains every holding.
    # No interpolation, forward filling or artificial zero returns.
    joined = pd.concat(parts,axis=1).dropna().sort_index()
    if len(joined) < cfg["calibration"]["minimum_observations"]:
        raise SimulationUnavailable(f"Only {len(joined)} common holding intervals; "
                                    f"need {cfg['calibration']['minimum_observations']}.")
    return portfolio,metadata,joined,sessions,start


def _fx_returns(processed,definition,sessions):
    """Interpret an explicit base/quote pair; never forward-fill missing fixes."""
    refs = processed.tables.get("selected_references",pd.DataFrame())
    key = definition["request_id"]
    if refs.empty or "instrument_id" not in refs:
        raise FXUnavailable("No selected FX observations for "+key)
    rows = refs.loc[refs["instrument_id"] == key].copy()
    if rows.empty or not {"date","value","data_type"} <= set(rows.columns):
        raise FXUnavailable("No usable FX history for "+key)
    if set(rows["data_type"].dropna()) != {"fx_reference_rate"}:
        raise FXUnavailable("Reference is not an FX rate: "+key)
    # Example: EURUSD is USD per EUR. Units determine the sign of the FX
    # log return needed to express native wealth in the reporting currency.
    expected_unit = definition["quote_currency"]+" per "+definition["base_currency"]
    if "unit" in rows and any(str(u).strip() != expected_unit for u in rows["unit"].dropna().unique()):
        raise FXUnavailable("FX quote units disagree with configuration: "+key)
    rows = rows.loc[rows["date"].between(sessions[0],sessions[-1])]
    if rows["date"].duplicated().any():
        raise FXUnavailable("Duplicate FX dates: "+key)
    values = pd.to_numeric(rows["value"],errors="coerce")
    if (~np.isfinite(values) | (values <= 0)).any():
        raise FXUnavailable("Invalid FX values: "+key)
    # Missing fixes become NaN; a stale fix is never substituted. A return
    # with either endpoint missing cannot enter the common calibration sample.
    prices = pd.Series(values.to_numpy(),index=rows["date"]).reindex(sessions)
    # Decimal log FX growth over one exchange interval. A two-date index
    # ensures exact alignment with the corresponding holding returns.
    returns = np.log(prices/prices.shift())
    frame = pd.DataFrame({"date":sessions,"return_start_date":pd.Series(sessions).shift(),
                          "value":returns.to_numpy()}).dropna()
    return frame.set_index(["return_start_date","date"])["value"].rename("fx::"+key)


def _currency_inputs(processed,cfg,metadata,holding_returns,sessions,target):
    """Jointly calibrate required FX factors; support direct/inverse conversions."""
    # Several holdings may share one FX factor. Deduplicate it so they all
    # experience the SAME exchange-rate shock within a path/date.
    needed,conversion = {},[]
    for holding in metadata:
        native = holding["currency"]
        if native == target:
            conversion.append(None)
            continue
        matches = [fx for fx in cfg["fx_series"] if
                   {fx["base_currency"],fx["quote_currency"]} == {native,target}]
        if len(matches) != 1:
            raise FXUnavailable(f"No FX definition for {native} to {target}.")
        fx = matches[0]
        factor = "fx::"+fx["request_id"]
        needed[factor] = fx
        # Native base -> quote: multiply by FX, therefore add its log return.
        # Native quote -> base: divide by FX, therefore subtract its log return.
        # A USD holding expressed in EUR using USD-per-EUR needs sign -1.
        sign = 1 if fx["base_currency"] == native else -1
        conversion.append((factor,sign))
    joined = holding_returns.copy()
    for fx in needed.values():
        joined = joined.join(_fx_returns(processed,fx,sessions),how="inner")
    joined = joined.dropna()
    if len(joined) < cfg["calibration"]["minimum_observations"]:
        raise FXUnavailable(f"Insufficient common holding/FX intervals for {target}: {len(joined)}.")
    # Conversion matrix A has shape (H,F). Each row selects one asset log
    # return plus/minus its required FX log return. Without FX, A is identity.
    # A @ factor_log_return gives holding growth in the simulation currency.
    transform = np.zeros((len(metadata),len(joined.columns)))
    for i,holding in enumerate(metadata):
        transform[i,joined.columns.get_loc(holding["factor"])] = 1
        if conversion[i]:
            factor,sign = conversion[i]
            transform[i,joined.columns.get_loc(factor)] = sign
    return joined,transform,list(needed.values())


def _portfolio_wealth(holding_values,weights,initial_value):
    """Stable equivalent forms of the fully invested weighted wealth sum.

Near initial wealth, summing gains avoids an artificial loss from rounding
weighted levels. Far below initial wealth, use positive levels directly so
subtracting/re-adding the initial value cannot erase a small positive balance.
Neither branch clips a return or changes portfolio weights.
"""
    # Sum over the last axis of holding wealth. Fixed INITIAL coefficients
    # represent buy-and-hold positions, not daily rebalancing. Shapes may have
    # additional leading path dimensions, which this multiplication preserves.
    levels = holding_values @ weights
    # Equivalent gains form relies on weights summing to one after accepted
    # rounding reconciliation. It prevents tiny artificial losses at flat wealth.
    gains = (holding_values-initial_value) @ weights
    # Choose stable arithmetic element by element. This threshold is not a
    # trading rule or stop-loss. Far below initial wealth, direct positive levels
    # avoid cancellation erasing a small but genuine positive balance.
    return np.where(np.abs(gains) <= .5*initial_value,initial_value+gains,levels)



def moments(joined):
    """Daily log moments on one complete-case sample; sample covariance M-1.

    Rows are matched session intervals; columns are native assets then FX.
    m has shape (F,), C is (F,F): C_ij=sum[(r_di-m_i)*(r_dj-m_j)]/(M-1).
    This is factor covariance, not a portfolio-only variance. Units are per
    session. Historical MC uses these moments each daily step; physical MC
    annualizes them explicitly before calendar-time integration. This helper
    does not impose either simulation clock or fit a volatility surface.
    """
    values = joined.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite calibration log returns.")
    return values.mean(axis=0), np.atleast_2d(np.cov(values, rowvar=False, ddof=1))


def prepare_history(processed, cfg, risk_window=None, forced_currency=None):
    """Resolve native holdings and joint FX factors without simulating paths.

    Currency fallback is shared with historical MC. It never relabels USD as
    EUR and does not replace missing assets. Output arrays retain factor order.
    """
    portfolio, holdings, native, sessions, start = _holding_inputs(processed, cfg, risk_window)
    currency = forced_currency or portfolio['reporting_currency']
    warnings = []
    try:
        joined, transform, fx = _currency_inputs(processed, cfg, holdings, native, sessions, currency)
    except FXUnavailable as exc:
        if forced_currency is not None or currency == cfg['fallback_currency']:
            raise
        warnings.append(f"{currency} unavailable: {exc}. Trying USD fallback.")
        currency = cfg['fallback_currency']
        joined, transform, fx = _currency_inputs(processed, cfg, holdings, native, sessions, currency)
        warnings.append("Results are in USD, not the requested reporting currency.")
    return dict(portfolio=portfolio, holdings=holdings, native=native, sessions=sessions,
                start=start, currency=currency, joined=joined, transform=transform,
                fx=fx, warnings=warnings)
