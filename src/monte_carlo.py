"""Offline historical Monte Carlo on cleaned holdings and optional FX factors.

Entry points:
    simulate_processed(processed, config_file) -> MonteCarloResult (no writes)
    simulate_saved_processing(folder, config_file) -> checksum-verified equivalent
    result.save(folder) -> separate derived tables/report/plots

The historical daily LOG mean is the drift of the simulated log increment.
No Ito correction is subtracted again. The first forecast date is an initial
normalized valuation, not a claim to have observed prices on that date.
Holdings are simulated jointly, and initial weights apply once (buy-and-hold).

Student-t scenarios condition a multivariate t shock on a bounded radial norm,
then rescale its conditional variance to one. They are explicitly TRUNCATED,
not an ordinary t model and not clipped prices. Within each calibration, the same log mean and covariance are used in every
model, including probability-based truncation sensitivity cases. Optional named
stress windows can retain baseline drift while replacing covariance and tails.
"""
# READING MAP
# 1. Validate JSON choices; prepare matched historical holding/FX log returns.
# 2. Estimate mean/covariance; factor covariance; fit joint Student-t tails.
# 3. Generate joint paths; aggregate holdings; calculate risk/shape diagnostics.
# 4. Repeat for optional stress windows; explicitly save separate derived files.
#
# Array notation used throughout these comments:
# H = holdings, F = asset plus required FX factors, R = stochastic matrix rank,
# N = simulated paths, D = forecast dates INCLUDING the initial valuation,
# M = common historical observations. Factor order follows joined.columns.
# X has shape (M,F); mean is (F,); covariance is (F,F); root B is (F,R).
# All calibrated means/covariances are DAILY LOG returns in decimal units.
# No product names, crisis dates or portfolio dimension are hardcoded here.
from copy import deepcopy
from dataclasses import dataclass
from importlib.metadata import version
from io import BytesIO
from pathlib import Path
import hashlib
import json
import math
import re

import exchange_calendars as xcals
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import betainc, gammaln
from scipy.stats import f as f_distribution

from .statistics import _read_json, _day, _prepare, _basis_kind
from .storage import write_json
from .cob import forecast_start_for_cob
from .mc_diagnostics import (daily_distribution, wealth_diagnostics, diagnostic_figures,
                             probability_of_loss, probability_label, confidence_label,
                             LOSS_PROBABILITY_TOLERANCE)


from .mc_common import (SimulationUnavailable, FXUnavailable, _sessions,
    _root_covariance, _holding_inputs, _fx_returns, _currency_inputs,
    _portfolio_wealth, prepare_history, moments)


def load_monte_carlo_config(path):
    """Strict configuration: reject unknown settings rather than ignoring them."""
    # Read the JSON with the shared reader. Validate before calibration so a
    # misspelled field cannot silently change the financial experiment.
    cfg = _read_json(path)
    # Required fields include explanatory instructions; only the named optional
    # groups below may be omitted. Their defaults enter the resolved settings.
    keys = {"schema_version","enabled","models","forecast_start","forecast_end","paths",
            "seed","calendar","initial_value","portfolio_method","calibration",
            "fallback_currency","fx_series","student_t","quantiles","confidence_levels",
            "sample_paths","plots","instructions"}
    optional = {"diagnostics", "stress_calibrations"}
    if not isinstance(cfg,dict) or not keys <= set(cfg) or set(cfg)-keys-optional:
        raise ValueError("Missing or unknown Monte Carlo configuration settings.")
    if type(cfg["schema_version"]) is not int or cfg["schema_version"] != 2:
        raise ValueError("Monte Carlo schema_version must be 2 (fitted df and probability-based cutoffs).")
    for k in ("enabled","plots"):
        if type(cfg[k]) is not bool:
            raise ValueError(k+" must be boolean.")
    # Allow Gaussian, fitted truncated-t, or both; duplicate names would create
    # ambiguous duplicate scenarios. This choice does not split assets apart.
    models = cfg["models"]
    if (not isinstance(models,list) or not models or
            any(not isinstance(m,str) or m not in {"gaussian","student_t_truncated"} for m in models)
            or len(set(models)) != len(models)):
        raise ValueError("Choose distinct gaussian/student_t_truncated models.")
    # A policy marker is allowed, never a user-entered fixed forecast start.
    # The actual date is resolved from the selected archive in _simulate_core.
    if cfg["forecast_start"] != "cob_plus_one":
        raise ValueError("forecast_start must be cob_plus_one; fixed start dates are not supported.")
    _day(cfg["forecast_end"])
    # At least two paths permit sample standard deviations. The integer seed
    # controls repeatable random draws. sample_paths controls exports only.
    for k,minimum in (("paths",2),("seed",0),("sample_paths",0)):
        if type(cfg[k]) is not int or cfg[k] < minimum:
            raise ValueError(k+" has an invalid integer value.")
    if cfg["sample_paths"] > cfg["paths"]:
        raise ValueError("sample_paths cannot exceed paths.")
    # Starting normalized wealth must be positive/finite. For example, 100
    # is a common accounting base, not an observed ETF share price or NAV.
    if (type(cfg["initial_value"]) not in (int,float)
            or not math.isfinite(cfg["initial_value"]) or cfg["initial_value"] <= 0):
        raise ValueError("initial_value must be finite and positive.")
    if not isinstance(cfg["calendar"],str) or not cfg["calendar"]:
        raise ValueError("A forecast exchange calendar is required.")
    if cfg["portfolio_method"] != "buy_and_hold":
        raise ValueError("Only buy_and_hold is implemented.")
    if cfg["fallback_currency"] != "USD":
        raise ValueError("The supported fallback currency is USD.")
    # An explicit start overrides lookback_years; neither sets the forecast.
    # minimum_observations applies after matching holding and FX intervals.
    cal = cfg["calibration"]
    if not isinstance(cal,dict) or set(cal) != {"lookback_years","start","minimum_observations"}:
        raise ValueError("Invalid calibration settings.")
    for k,minimum in (("lookback_years",1),("minimum_observations",2)):
        if type(cal[k]) is not int or cal[k] < minimum:
            raise ValueError("Invalid calibration "+k)
    if cal["start"] is not None:
        _day(cal["start"])
    # Quantiles describe wealth percentiles. Confidence levels describe the
    # upper tail of LOSSES for VaR/ES. Both use fractions: 0.99 means 99%.
    for k in ("quantiles","confidence_levels"):
        values = cfg[k]
        lower = .5 if k == "confidence_levels" else 0
        if (not isinstance(values,list) or not values
                or any(type(v) not in (int,float) or not math.isfinite(v) or not lower < v < 1 for v in values)
                or len(set(values)) != len(values)):
            raise ValueError("Invalid "+k)
    if not isinstance(cfg["fx_series"],list):
        raise ValueError("fx_series must be an array.")
    # Reject duplicate/ambiguous FX definitions. EUR/USD and USD/EUR identify
    # the same currency pair here, while their quote direction is retained.
    pairs,ids = set(),set()
    for fx in cfg["fx_series"]:
        if not isinstance(fx,dict) or set(fx) != {"request_id","base_currency","quote_currency"}:
            raise ValueError("Each FX definition needs request_id, base_currency and quote_currency.")
        if any(not isinstance(v,str) or not v for v in fx.values()):
            raise ValueError("FX definitions require nonempty strings.")
        pair = frozenset((fx["base_currency"],fx["quote_currency"]))
        if len(pair) != 2 or pair in pairs or fx["request_id"] in ids:
            raise ValueError("Duplicate/ambiguous FX definitions.")
        pairs.add(pair); ids.add(fx["request_id"])
    # Fit one JOINT degrees of freedom, not one per asset. Bounds exceed two
    # because the untruncated calibrated family must have finite covariance.
    # There is no fixed-df fallback after a failed fit.
    t = cfg["student_t"]
    if not isinstance(t,dict) or set(t) != {"fit_df","df_bounds","excluded_tail_probability","sensitivity_tail_probabilities"}:
        raise ValueError("Invalid Student-t settings.")
    if t["fit_df"] is not True:
        raise ValueError("Student-t fit_df must be true; fixed degrees of freedom are not supported.")
    bounds = t["df_bounds"]
    if (not isinstance(bounds,list) or len(bounds) != 2
            or any(type(x) not in (int,float) or not math.isfinite(x) for x in bounds)
            or not 2 < bounds[0] < bounds[1]):
        raise ValueError("Student-t df bounds must satisfy 2 < lower < upper.")
    tails = t["sensitivity_tail_probabilities"]
    if not isinstance(tails,list):
        raise ValueError("sensitivity_tail_probabilities must be an array.")
    # Inputs are removed probability masses, not fixed radii. Radius depends
    # on fitted df/rank. Check 1-p too: extreme floats can round it to one.
    probabilities = [t["excluded_tail_probability"],*tails]
    if (any(type(x) not in (int,float) or not math.isfinite(x)
            or not 0 < x < 1 or not 0 < 1-x < 1 for x in probabilities)
            or len(set(probabilities)) != len(probabilities)):
        raise ValueError("Excluded tail probabilities must be distinct, finite and strictly between zero and one, with a representable retained probability.")
    # Optional additions keep existing schema-2 input files usable. Explicit
    # defaults are included in the run identity and exported resolved settings.
    diag = cfg.setdefault("diagnostics", {"enabled":True,"horizons_sessions":[1,5,20],
                                          "sigma_thresholds":[3,4],"histogram_bins":100})
    if not isinstance(diag,dict) or set(diag) != {"enabled","horizons_sessions","sigma_thresholds","histogram_bins"}:
        raise ValueError("Invalid diagnostics settings.")
    if type(diag["enabled"]) is not bool or type(diag["histogram_bins"]) is not int or not 10 <= diag["histogram_bins"] <= 1000:
        raise ValueError("Diagnostics need a boolean enabled and 10..1000 histogram bins.")
    for field in ("horizons_sessions","sigma_thresholds"):
        values = diag[field]
        if (not isinstance(values,list) or not values or
                any(type(v) is not int or v <= 0 for v in values) or len(set(values)) != len(values)):
            raise ValueError("Diagnostics thresholds/horizons must be distinct positive integers.")
    # Stress windows are separate calibrations, not extra baseline history.
    # Unique safe names also become output labels and filename components.
    stresses = cfg.setdefault("stress_calibrations", [])
    if not isinstance(stresses,list):
        raise ValueError("stress_calibrations must be an array.")
    names = set()
    for stress in stresses:
        if not isinstance(stress,dict) or set(stress) != {"name","start","end","drift"}:
            raise ValueError("Each stress calibration needs name, start, end and drift.")
        name = stress["name"]
        if (not isinstance(name,str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,39}",name)
                or name == "baseline" or name in names):
            raise ValueError("Stress names must be unique lowercase identifiers other than baseline.")
        names.add(name)
        if _day(stress["start"]) >= _day(stress["end"]):
            raise ValueError("Stress start must precede end.")
        if stress["drift"] not in ("baseline","stress"):
            raise ValueError("Stress drift must be baseline or stress.")
    # Instructions are explanatory only, and do not affect numerical run identity.
    return {k:deepcopy(v) for k,v in cfg.items() if k != "instructions"}


def _student_parameters(centered,root,settings):
    """Always fit joint df with historical mean/covariance held fixed.

    The untruncated covariance-standardized family is fitted first. A declared
    excluded probability then determines each truncation radius. Neither df
    nor a radius is reused from a prior portfolio or calibration window.
    """
    rank = root.shape[1]
    if rank == 0:
        # Constant data contain no information about tail shape. Preserve zero
        # shocks and label df as unavailable instead of inventing a fixed df.
        return dict(df=None,method="deterministic",rank=0,
                    df_at_boundary=False,whitened_radii=np.zeros(len(centered)))
    # Whiten (M,F) centered data to (M,R) unit-covariance coordinates.
    # The pseudoinverse of B.T handles dependent factors without adding noise.
    white = centered @ np.linalg.pinv(root.T)
    # Squared radius is joint standardized distance, not a percentage loss
    # or separate cap on each asset. One radius describes the whole shock.
    radii2 = np.sum(white*white,axis=1)
    # Fit df=nu by minimizing negative log likelihood. Hold historical mean
    # and covariance fixed. Whitening terms constant in nu can be omitted.
    def objective(nu):
        # Unit covariance requires shape=(nu-2)/nu, not shape=I.
        # Log gamma avoids overflow. A multivariate t with covariance I has
        # SHAPE (nu-2)/nu * I, explaining nu-2 in normalization and radial decay.
        constant = gammaln((nu+rank)/2)-gammaln(nu/2)-rank/2*np.log((nu-2)*np.pi)
        return float(-np.sum(constant-(nu+rank)/2*np.log1p(radii2/(nu-2))))
    # Re-estimate joint df for this portfolio/window. User bounds constrain
    # the numerical search; they are not themselves estimated from data.
    fit = minimize_scalar(objective,bounds=settings["df_bounds"],method="bounded")
    if not fit.success or not math.isfinite(fit.fun):
        raise SimulationUnavailable("Joint Student-t degrees-of-freedom fit failed.")
    df = float(fit.x)
    # Flag a fit near a search boundary: the configured range may be limiting
    # the inferred shape. It remains visible rather than silently replaced.
    lo,hi = settings["df_bounds"]
    return dict(df=df,method="covariance_constrained_untruncated_likelihood",rank=rank,
                df_at_boundary=min(df-lo,hi-df)<1e-3*(hi-lo),
                whitened_radii=np.sqrt(radii2))


def _truncation(rank,df,excluded_tail_probability):
    """Derive a new radius from fitted df, effective rank and excluded mass.

    With d=rank and covariance-standardized t shocks:
        radius^2 = d*(df-2)/df * F(d,df).
    Invert the upper-tail probability, rather than imposing a fixed radius.
    Rank includes stochastic FX factors and excludes deterministic/dependent
    dimensions; it is not simply the configured number of holdings.

    The conditional variance per coordinate has an exact incomplete-beta form:
        I_z(d/2+1, df/2-1) / P(radius <= cutoff),
        z = cutoff^2 / (cutoff^2 + df - 2).
    This avoids integrating over a potentially very wide radial interval.
    """
    target = excluded_tail_probability
    if (type(target) not in (int,float) or not math.isfinite(target)
            or not 0 < target < 1 or not 0 < 1-target < 1):
        raise ValueError("Invalid excluded tail probability.")
    if rank == 0:
        return dict(excluded_tail_probability=target,actual_excluded_probability=0.,
                    effective_dimensions=0,truncation_applied=False,radial_cutoff=0.,
                    retained_probability=1.,variance_before_rescaling=1.,
                    effective_radial_cutoff=0.,upper_f=0.)
    if rank < 0 or df is None or not math.isfinite(df) or df <= 2:
        raise ValueError("Positive dimension and fitted df > 2 are required.")
    # For covariance-standardized t coordinates in R dimensions,
    # radius^2 = R*(df-2)/df * F(R,df). This defines the radial law.
    scale = rank*(df-2)/df
    # Invert upper-tail probability directly, avoiding precision loss from
    # manually computing 1-p for a very small excluded mass.
    upper = float(f_distribution.isf(target,rank,df))
    cutoff_squared = scale*upper
    if not math.isfinite(cutoff_squared) or cutoff_squared <= 0:
        raise ValueError("The requested tail probability has no numerically usable radius.")
    # Check that the computed threshold actually removes the requested mass.
    # Store both requested/actual values for an auditable report.
    actual = float(f_distribution.sf(upper,rank,df))
    probability = float(f_distribution.cdf(upper,rank,df))
    if not math.isclose(actual,target,rel_tol=1e-6,abs_tol=0):
        raise ValueError("The computed radius does not reproduce the requested tail probability.")
    z = cutoff_squared/(cutoff_squared+df-2)
    # Truncation lowers each coordinate's second moment. This incomplete-beta
    # expression gives its conditional variance; later divide retained shocks
    # by sqrt(variance) so the target covariance is restored.
    variance = float(betainc(rank/2+1,df/2-1,z)/probability)
    if not (0 < probability < 1 and math.isfinite(variance) and 0 < variance <= 1+1e-10):
        raise ValueError("Invalid Student-t truncation normalization.")
    # Effective cutoff includes variance restoration and is therefore wider
    # than the pre-rescaling radius. Neither number is an asset return limit.
    cutoff = math.sqrt(cutoff_squared)
    return dict(excluded_tail_probability=target,actual_excluded_probability=actual,
                effective_dimensions=rank,truncation_applied=True,radial_cutoff=cutoff,
                retained_probability=probability,variance_before_rescaling=variance,
                effective_radial_cutoff=cutoff/math.sqrt(variance),upper_f=upper)


def _draw_shocks(rng,size,root,student=None):
    """Generate correlated shocks, including exact singular covariance support."""
    # B has F rows and R stochastic columns. Output is always (size,F),
    # including a deterministic (zero-shock) result when R=0.
    rank = root.shape[1]
    if rank == 0:
        return np.zeros((size,root.shape[0]))
    if student is None:
        # Z contains independent N(0,1) coordinates, (paths,R). Z @ B.T creates
        # correlated factor shocks with covariance B B.T. The caller adds drift.
        return rng.standard_normal((size,rank)) @ root.T
    # For t sampling, separate angle from radius. An isotropic Gaussian
    # vector divided by its length is uniform on the unit sphere.
    direction = rng.standard_normal((size,rank))
    lengths = np.linalg.norm(direction,axis=1)
    # An exactly zero vector cannot be normalized: redraw only such rows.
    # This numerical guard does not clip extreme financial returns.
    while (lengths == 0).any():
        mask = lengths == 0
        direction[mask] = rng.standard_normal((int(mask.sum()),rank))
        lengths = np.linalg.norm(direction,axis=1)
    # [:,None] reshapes (paths,) to (paths,1), broadcasting each row's own
    # length over its coordinates. Resulting directions have unit norm.
    direction /= lengths[:,None]
    # U is uniform over retained CDF mass. Inverse F CDF samples the bounded
    # conditional radial law directly; it never generates and then clips prices.
    # One radius is shared by the coordinates, inducing joint t tail dependence.
    u = rng.random(size)*student["retained_probability"]
    radial2 = rank*(student["df"]-2)/student["df"] * f_distribution.ppf(u,rank,student["df"])
    # Divide the retained radius by sqrt(conditional variance), restoring
    # unit covariance before imposing historical factor covariance.
    radii = np.sqrt(radial2/student["variance_before_rescaling"])
    # After variance restoration, impose factor covariance via B.T. Unlike an
    # unrestricted Student-t log-return law, bounded shocks have finite
    # exponential moments, so the corresponding wealth expectation is finite.
    return (direction*radii[:,None]) @ root.T


def _expected_shortfall(losses,confidence):
    """Mean of the worst exact tail probability, including a fractional boundary."""
    # ES averages the worst (1-confidence) tail of losses, ordered largest
    # first. Negative losses are gains and remain legitimate observations.
    ordered = np.sort(losses)[::-1]
    count = len(ordered)*(1-confidence)
    # A tail size of 2.5 uses two full losses and half the next one, avoiding
    # bias from rounding the requested probability mass up or down.
    whole = int(math.floor(count)); fraction = count-whole
    return float((ordered[:whole].sum()+fraction*ordered[whole])/count)


# In-memory result container; save() is an explicit separate operation.
# This dataclass is mutable, while source inputs were copied for isolation.
# Fields retain source identity, resolved settings, status and named tables.
@dataclass
class MonteCarloResult:
    """Immutable-input snapshot of a simulation; export is an explicit operation."""
    archive: Path
    processing_id: str
    cob: str
    run_id: str
    settings: dict
    summary: dict
    tables: dict[str,pd.DataFrame]
    source_folder: Path | None = None

    # Format existing results as Markdown. This method does not draw shocks,
    # refit models, download data or alter the simulation tables.
    def report(self):
        s = self.summary
        lines = ["# Historical Monte Carlo","",f"Status: {s['status']}",f"Source COB: {self.cob}",
                 f"Processing ID: {self.processing_id}",f"Simulation ID: {self.run_id}","",
                 f"COB-derived forecast: {self.settings['forecast_start']} to {self.settings['forecast_end']}",
                 f"Paths per scenario: {self.settings['paths']:,}",
                 f"Requested reporting currency: {s.get('reporting_currency','unavailable')}",
                 f"Actual simulation currency: {s.get('simulation_currency','unavailable')}",""]
        lines += ["- "+w for w in s["warnings"]]
        # Disabled/unavailable runs still explain why; do not access numerical
        # tables that would only exist after a completed simulation.
        if s["status"] != "complete":
            return "\n".join(lines)+"\n"
        c = s["calibration"]
        lines += ["","## Calibration and conventions","",
                  f"- Requested historical window: {c['requested_start']} through {self.cob}.",
                  f"- Actual common intervals: {c['first_start']} to {c['last_end']}; {c['observations']:,} observations.",
                  f"- Forecast has {s['steps']} exchange-session increments; {s['forecast_dates'][0]} is the initial level, not a simulated return.",
                  "- Drift is the historical DAILY LOG mean on the same common sample as covariance. No extra half-variance subtraction.",
                  "- Initial holdings: "+", ".join(f"{h['instrument_id']} {h['weight']:.1%}" for h in s["holdings"])+".",
                  "- Initial weights apply once, then drift. Adjusted histories represent reinvested wealth; no dividends are added.",
                  "- Values are normalized wealth, not forecasts of unadjusted ETF share prices. NAV is not used.",
                  "- Gaussian and truncated-t scenarios share log means/covariance. Their expected terminal wealth need not coincide.",
                  "- Shocks are independent through time with fixed historical parameters; correlations are contemporaneous.",
                  "- FX factors, when needed and available, are simulated jointly; no constant-FX assumption.",
                  "- No discounting, option pricing, OIS bootstrap, local-volatility or risk-neutral model is performed.","",
                  "## Terminal portfolio outcomes","",
                  "| Scenario | Mean wealth | Median wealth | Probability of loss | Mean MC standard error |",
                  "|---|---:|---:|---:|---:|"]
        # Overview shows portfolio terminal outcomes; individual holding results
        # remain available in the CSV rather than being dropped.
        for row in self.tables["risk_summary"].query("scope == 'portfolio'").to_dict("records"):
            lines.append(f"| {row['model']} | {row['mean_terminal']:.3f} | {row['median_terminal']:.3f} | "
                         f"{row['probability_loss']:.2%} | {row['mean_mc_standard_error']:.4f} |")
        lines += ["","## Historical daily calibration","","All entries below are decimal log-return units, not annual rates.","",
                  "| Calibration | Factor | Simulated mean log return | Standard deviation | Mean estimation standard error |",
                  "|---|---|---:|---:|---:|"]
        # Distinguish simulated mean from risk-window volatility: a stress may
        # intentionally retain baseline drift while using stress covariance.
        for row in self.tables["calibration_parameters"].to_dict("records"):
            lines.append(f"| {row['calibration_set']} | {row['factor']} | {row['mean_daily_log']:.6f} | "
                         f"{row['std_daily_log']:.6f} | {row['mean_estimation_standard_error']:.6f} |")
        lines += ["","Mean estimation standard errors assume independent observations and exclude regime uncertainty."]
        lines += ["","Loss VaR and Expected Shortfall in risk_summary.csv apply to the FULL forecast horizon, not one day.",
                  "They are fractions of initial wealth and can be negative if the threshold is a gain.",
                  "Monte Carlo standard errors measure sampling noise, not drift/model uncertainty.",
                  f"Probability of loss excludes numerical residuals <= {LOSS_PROBABILITY_TOLERANCE:.3g} of initial wealth; paths and other metrics are unchanged.","",
                  "## Student-t specification","",
                  "The t scenarios are radially truncated and variance-renormalized, not unrestricted Student-t or clipped prices.",
                  "Joint degrees of freedom are always fitted to the untruncated covariance-standardized family.",
                  "Excluded probabilities are per joint daily shock, before variance rescaling; they are user assumptions.",
                  "Radii are recalculated from fitted df and effective factor dimensions for every calibration.",
                  "With deterministic factors only, df is not identifiable and actual excluded probability is zero.","",
                  "| Scenario | df | Dimensions | Target excluded | Actual excluded | Derived radius | Variance before rescaling | Effective radius |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        # Only t models carry df/radius metadata. Deterministic fits show n/a
        # instead of inventing an identifiable degrees-of-freedom estimate.
        for model,meta in s["models"].items():
            if "df" in meta:
                df_text = "n/a" if meta["df"] is None else f"{meta['df']:.3f}"
                lines.append(f"| {model} | {df_text} | {meta['effective_dimensions']} | "
                             f"{meta['excluded_tail_probability']:.6%} | {meta['actual_excluded_probability']:.6%} | "
                             f"{meta['radial_cutoff']:.3f} | {meta['variance_before_rescaling']:.6f} | "
                             f"{meta['effective_radial_cutoff']:.3f} |")
        lines += ["","## Named stress calibrations", "",
                  "The baseline above remains unchanged. Stress scenarios use the same forecast, holdings and currency.",
                  "GFC and other labels denote user-selected windows, not official crisis boundaries.",
                  "Keeping baseline LOG drift does not keep expected arithmetic wealth constant when volatility changes.",
                  "Daily shocks remain independent; these comparisons do not reproduce volatility clustering.",
                  "Matching model names share random streams across calibrations to aid comparison."]
        # Include actual stress coverage and unavailable reasons so absent
        # comparisons cannot look like successful calibrations.
        for stress in s.get("stress_calibrations",[]):
            lines += ["",f"- {stress['name']}: {stress['start']} to {stress['end']}; status={stress['status']}; drift={stress['drift']}."]
            if stress["calibration"]:
                c = stress["calibration"]
                lines.append(f"  Actual common intervals: {c['first_start']} to {c['last_end']}; {c['observations']:,} observations.")
            lines += ["  "+warning for warning in stress["warnings"]]
        if self.settings["diagnostics"]["enabled"]:
            lines += ["","## Distribution and horizon diagnostics","",
                      "Daily results pool all simulated path/session observations. They are daily LOG returns, not terminal returns.",
                      "Holding z-scores use the scenario calibration mean/std; historical z-scores use their own window moments.",
                      "Portfolio z-scores use pooled simulated moments because buy-and-hold weights drift.",
                      "Skewness and excess kurtosis are empirical central-moment ratios (normal excess kurtosis = 0).",
                      "Histogram bins cover z=-10 to +10; underflow/overflow counts are retained in the daily diagnostics table.",
                      "The histogram density denominator includes all observations, including overflow.",
                      "Short-horizon losses measure initial wealth to session 1/5/20 (or configured horizons), once per path.",
                      "Unavailable horizons beyond the configured forecast are explicitly labelled; no extrapolation.",
                      "Path diagnostics measure each path's worst daily return and maximum drawdown separately.","",
                      "| Scenario | Portfolio daily excess kurtosis | "+" | ".join(f"P(abs(z)>{v})" for v in self.settings["diagnostics"]["sigma_thresholds"])+" |",
                      "|---|---:|"+"---:|"*len(self.settings["diagnostics"]["sigma_thresholds"])]
            # Undefined metrics are n/a, not zero. Percent formatting applies to
            # probabilities/loss fractions; raw daily log means retain their units.
            def numeric(value,percent=False):
                return "n/a" if value is None or pd.isna(value) else (f"{value:.3%}" if percent else f"{value:.3f}")
            for row in self.tables["daily_distribution_diagnostics"].query("scope == 'portfolio'").to_dict("records"):
                cells = [numeric(row.get(f"probability_abs_z_gt_{v:g}"),True) for v in self.settings["diagnostics"]["sigma_thresholds"]]
                lines.append(f"| {row['model']} | {numeric(row['excess_kurtosis'])} | "+" | ".join(cells)+" |")
            confidence = max(self.settings["confidence_levels"])
            lines += ["",f"| Scenario | Sessions | {confidence_label(confidence)} loss VaR | {confidence_label(confidence)} loss ES | Probability of loss |",
                      "|---|---:|---:|---:|---:|"]
            for row in self.tables["horizon_risk"].query("scope == 'portfolio'").to_dict("records"):
                lines.append(f"| {row['model']} | {row['horizon_sessions']} | "
                             f"{numeric(row.get(f'loss_var_{probability_label(confidence)}'),True)} | {numeric(row.get(f'loss_es_{probability_label(confidence)}'),True)} | "
                             f"{numeric(row.get('probability_loss'),True)} |")
            lines += ["","Full holding-level results: daily_distribution_diagnostics.csv, daily_histograms.csv, horizon_risk.csv and path_risk.csv.",
                      "Path-level worst_daily_return and max_drawdown are retained in terminal_values.csv."]
        lines += ["","Calibration and simulated-factor diagnostics are exported separately.",
                  "The calibration window is configurable; estimated means are uncertain and not guaranteed forecasts."]
        return "\n".join(lines)+"\n"

    def _destination(self,destination):
        """Never write over raw inputs, cleaned tables, or their ancestor folders."""
        # Resolve absolute paths (including existing symlinks) before containment
        # checks so outputs cannot replace raw archives or cleaned inputs.
        dest = Path(destination).resolve()
        archive = self.archive.resolve()
        if dest == archive or dest.is_relative_to(archive) or archive.is_relative_to(dest):
            raise ValueError("Monte Carlo output must be outside raw archives and their ancestors.")
        if self.source_folder and (dest == self.source_folder or self.source_folder.is_relative_to(dest)):
            raise ValueError("Monte Carlo output cannot overwrite the cleaned source folder.")
        return dest

    # Export CSVs/report/figures. Mark the manifest complete only after all
    # requested outputs succeed; interrupted saves remain explicitly incomplete.
    def save(self,destination):
        destination = self._destination(destination)
        destination.mkdir(parents=True,exist_ok=True)
        # Check each file too, not only its parent directory. Reject symlinks
        # that could redirect a write into a protected input location.
        def safe(path):
            self._destination(path)
            if path.is_symlink():
                raise ValueError("Refusing symlink simulation output.")
            return path
        manifest = deepcopy(self.summary)
        manifest.update(complete=False,files={})
        write_json(safe(destination/"manifest.json"),manifest)
        # UTF-8 preserves metadata; undefined cells export blank. Row counts and
        # SHA-256 of exact saved bytes allow later integrity verification.
        for name,table in self.tables.items():
            path = safe(destination/(name+".csv"))
            table.to_csv(path,index=False,encoding="utf-8",na_rep="")
            manifest["files"][name] = dict(file=path.name,rows=len(table),
                                          sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        report = safe(destination/"report.md")
        report.write_text(self.report(),encoding="utf-8")
        manifest["files"]["report"] = dict(file=report.name,sha256=hashlib.sha256(report.read_bytes()).hexdigest())
        if self.settings["plots"] and self.summary["status"] == "complete":
            import matplotlib
            # Headless backend creates files without GUI windows. finally closes
            # each figure even if saving fails, avoiding leaked plotting resources.
            matplotlib.use("Agg")
            from matplotlib import pyplot as plt
            plots = destination/"plots"
            if plots.is_symlink():
                raise ValueError("Refusing symlink plot directory.")
            plots.mkdir(exist_ok=True)
            for model in self.summary["models"]:
                bands = self.tables["bands"].query("model == @model and scope == 'portfolio'")
                terminal = self.tables["terminal_values"].query("model == @model and scope == 'portfolio'")["terminal_value"]
                fig,axes = plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
                dates = pd.to_datetime(bands["date"])
                # Bands are cross-path percentiles at EACH date, not confidence intervals
                # for the mean or one selected path followed through time.
                axes[0].fill_between(dates,bands["p05"],bands["p95"],alpha=.25,label="5th–95th percentile")
                axes[0].plot(dates,bands["p50"],label="Median")
                axes[0].set(title="Portfolio wealth",ylabel=f"Normalized value ({self.summary['simulation_currency']})")
                axes[0].legend()
                # Use one terminal observation per path. The initial-value line separates
                # terminal losses from gains; density is probability per wealth unit.
                axes[1].hist(terminal,bins=60,density=True,alpha=.6)
                axes[1].axvline(self.settings["initial_value"],color="black",linestyle="--",label="Initial value")
                axes[1].set(title="Terminal portfolio distribution",xlabel="Normalized wealth",ylabel="Density")
                axes[1].legend()
                fig.suptitle(model+" | "+self.settings["forecast_start"]+" to "+self.settings["forecast_end"])
                path = safe(plots/(model+".png"))
                try: fig.savefig(path,dpi=130)
                finally: plt.close(fig)
                manifest["files"]["plots/"+model] = dict(file=path.relative_to(destination).as_posix(),
                                                       sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            if self.settings["diagnostics"]["enabled"]:
                for name,fig in diagnostic_figures(self.tables,self.settings):
                    path = safe(plots/(name+".png"))
                    try: fig.savefig(path,dpi=130)
                    finally: plt.close(fig)
                    manifest["files"]["plots/"+name] = dict(file=path.relative_to(destination).as_posix(),
                                                           sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        # Only this final marker declares the output complete; it does not make
        # an unavailable simulation available (status is reported separately).
        manifest["complete"] = True
        write_json(safe(destination/"manifest.json"),manifest)
        return destination


def _run_scenario(model,mean,covariance,root,transform,weights,dates,cfg,student=None):
    """Advance normalized holdings, then aggregate with fixed initial quantities."""
    # N paths, D-1 increments, H holdings. Date 0 is initial wealth.
    # covariance documents the target; root carries the actual factor used for
    # draws. There is no second covariance recalculation during simulation.
    n,steps = cfg["paths"],len(dates)-1
    holdings = len(weights)
    # Check the main wealth tensor size before allocation (+1 for portfolio).
    # Other diagnostic arrays also consume memory beyond this tensor.
    if n*len(dates)*(holdings+1) > 100_000_000:
        raise ValueError("Requested paths/dates/holdings exceed the in-memory simulation limit.")
    # A model-specific seed is stable even if enabled model ordering changes.
    stream = int.from_bytes(hashlib.sha256(model.encode()).digest()[:8],"little")
    # Combine seed and deterministic model-name hash. Matching models in
    # baseline/stress share streams; differently named sensitivities do not.
    rng = np.random.default_rng(np.random.SeedSequence([cfg["seed"],stream]))
    # wealth[path,date,holding_or_portfolio] has shape (N,D,H+1).
    # All holding indices start at the same normalized base; allocations are
    # applied separately. Do not interpret them as observed share prices.
    wealth = np.empty((n,len(dates),holdings+1))
    wealth[:,0,:] = cfg["initial_value"]
    # (N,H) cumulative reporting-currency log growth. Adding daily logs is
    # equivalent to multiplying successive gross returns.
    cumulative = np.zeros((n,holdings))
    # Accumulate factor sums/cross-products to verify simulated moments.
    # These diagnostics do not update or recalibrate future steps.
    sums = np.zeros(len(mean)); cross = np.zeros((len(mean),len(mean)))
    # Draw ALL N paths at one date, then advance the date. To reproduce
    # saved path 0, keep this draw order: drawing one path per day would consume
    # different random numbers after the first date.
    for step in range(1,len(dates)):
        # Gaussian g = mu + B*z (or t analogue). mean (F,) broadcasts across
        # the (N,F) shocks. Calibration is daily: dt=1 exchange session, no /252.
        # Do not subtract variance/2: mu is already the mean of LOG increments.
        increments = mean+_draw_shocks(rng,n,root,student)
        # sum(axis=0) gives totals per factor; g.T @ g gives all uncentered
        # pairwise products for pooled covariance verification after the last step.
        sums += increments.sum(axis=0); cross += increments.T @ increments
        # Map (N,F) asset/FX returns to (N,H) holding returns in one currency.
        # Each holding uses its own asset and the shared appropriate FX draw.
        cumulative += increments @ transform.T
        # W_i,t = W0*exp(sum_{s<=t} g_i,s), equivalent to W_i,t-1*exp(g_i,t).
        # Reject overflow/invalid arithmetic and, below, zero or nonfinite wealth.
        # No return is clipped to conceal a numerical failure.
        with np.errstate(over="raise",invalid="raise",under="ignore"):
            wealth[:,step,:holdings] = cfg["initial_value"]*np.exp(cumulative)
        if not np.isfinite(wealth[:,step,:holdings]).all() or (wealth[:,step,:holdings] <= 0).any():
            raise ValueError("Nonfinite or underflowed simulated wealth; no path was clipped.")
        # Aggregate gains around the initial level. Directly summing weighted
        # levels can create a tiny loss even for constant holdings because of
        # floating-point summation. This also preserves a zero-gain portfolio
        # exactly after reconciling accepted allocation rounding.
        wealth[:,step,holdings] = _portfolio_wealth(wealth[:,step,:holdings],weights,cfg["initial_value"])
    # Pooled factor diagnostics contain N*(D-1) increments. Subtract their
    # mean outer product and divide by count-1 for sample covariance.
    count = n*steps
    sample_mean = sums/count
    sample_cov = (cross-count*np.outer(sample_mean,sample_mean))/(count-1)
    return wealth,sample_mean,sample_cov


def simulate_processed(processed,config_file):
    """Run baseline and independent named stress calibrations from the same archive.

A stress can retain baseline LOG drift while replacing covariance and fitted
Student-t tails. Each stress uses the same holdings, currency and forecast.
Missing stress history is reported without discarding a successful baseline.
"""
    cfg = load_monte_carlo_config(config_file)
    result = _simulate_core(processed,cfg)
    if result.summary["status"] != "complete":
        return result
    # Capture baseline drifts before concatenating stress tables. A stress
    # with baseline drift borrows means/mean SE only, never baseline covariance.
    baseline_parameters = result.tables["calibration_parameters"].set_index("factor").to_dict("index")
    result.summary["stress_calibrations"] = []
    for name,table in result.tables.items():
        table["calibration_set"] = "baseline"
    for meta in result.summary["models"].values():
        meta["calibration_set"] = "baseline"
    for stress in cfg["stress_calibrations"]:
        override = baseline_parameters if stress["drift"] == "baseline" else None
        # Recalibrate on the stress window, enforcing baseline simulation currency.
        # Missing stress FX must not silently produce an incomparable currency.
        compared = _simulate_core(processed,cfg,risk_window=stress,drift_override=override,
                                 forced_currency=result.summary["simulation_currency"])
        record = dict(stress, status=compared.summary["status"],
                      calibration=compared.summary.get("calibration"),warnings=compared.summary["warnings"])
        result.summary["stress_calibrations"].append(record)
        if compared.summary["status"] != "complete":
            result.summary["warnings"].append("Stress "+stress["name"]+" unavailable: "+"; ".join(record["warnings"]))
            continue
        # Prefix output labels AFTER drawing paths, preserving matching random
        # streams for the same model in baseline and stress comparisons.
        prefix = stress["name"]+"__"
        for model,meta in compared.summary["models"].items():
            result.summary["models"][prefix+model] = dict(meta,calibration_set=stress["name"])
        for name,table in compared.tables.items():
            table = table.copy()
            table["calibration_set"] = stress["name"]
            if "model" in table:
                table["model"] = prefix+table["model"]
            result.tables[name] = pd.concat([result.tables[name],table],ignore_index=True)
    return result


def _simulate_core(processed,cfg,risk_window=None,drift_override=None,forced_currency=None):
    """Calibrate and simulate; invalid inputs never trigger API requests.

    A missing-data condition returns a result with status=unavailable and a
    reason, so run.py can retain successful statistics and a reviewable report.
    Invalid configuration or corrupted inputs still raise an explicit error.
    """
    # Copy only relevant cleaned inputs. Raw provider payloads, option quotes
    # and OIS curves are not used by these historical-measure models.
    cfg = deepcopy(cfg)
    cfg["forecast_start"] = forecast_start_for_cob(processed.cob)
    source_tables = {k:processed.tables[k].copy(deep=True) for k in ("selected_prices","selected_references")
                     if k in processed.tables}
    # Identity hashes settings, source tables, code and dependency versions.
    # Source hashes include COMMENTS: documentation edits change future run IDs
    # even with identical mathematics. Previously saved reports are untouched.
    identity = dict(processing_id=processed.run_id,cob=_day(processed.cob),settings=cfg,
                    portfolio=deepcopy(processed.summary["portfolio"]),risk_window=risk_window,drift_override=drift_override,
                    inputs={k:hashlib.sha256(v.to_csv(index=False).encode()).hexdigest()
                            for k,v in source_tables.items()},
                    code={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (Path(__file__),Path(__file__).with_name("statistics.py"),Path(__file__).with_name("mc_diagnostics.py"),Path(__file__).with_name("cob.py"),Path(__file__).with_name("mc_common.py"))},
                    dependencies={k:version(k) for k in ("numpy","pandas","scipy","exchange-calendars","matplotlib")})
    # Sorted JSON makes a stable compact identifier for this exact setup.
    # Explanatory JSON instructions were excluded by the loader above.
    run_id = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
    # Start as disabled/unavailable, retaining a reviewable reason on missing
    # data. Invalid configuration/corrupted inputs still raise explicit errors.
    summary = dict(schema_version=2,run_id=run_id,cob=processed.cob,processing_id=processed.run_id,
                   identity=identity,status="disabled" if not cfg["enabled"] else "unavailable",
                   warnings=[],simulation_performed=False,option_calibration_performed=False,
                   reporting_currency=processed.summary["portfolio"]["reporting_currency"])
    result = MonteCarloResult(Path(processed.archive),processed.run_id,processed.cob,run_id,cfg,summary,{})
    if not cfg["enabled"]:
        return result
    try:
        if cfg["forecast_end"] <= cfg["forecast_start"]:
            raise SimulationUnavailable("Forecast end must follow archive COB + 1 calendar day; update forecast_end.")
        # Preserve the requested calendar-day initial valuation, even on a
        # weekend/holiday. Random daily returns end only on subsequent exchange
        # sessions; no extra weekend shocks or fabricated observed prices.
        sessions_after_start = [d for d in _sessions(cfg["calendar"],cfg["forecast_start"],cfg["forecast_end"])
                                if d > cfg["forecast_start"]]
        dates = [cfg["forecast_start"]] + sessions_after_start
        if not sessions_after_start:
            raise SimulationUnavailable("No exchange session after COB + 1 within the configured forecast horizon.")
        history = prepare_history(processed, cfg, risk_window, forced_currency)
        portfolio, holdings, native, sessions, requested_start = (
            history[k] for k in ('portfolio','holdings','native','sessions','start'))
        currency, joined, transform, fx = (
            history[k] for k in ('currency','joined','transform','fx'))
        summary['warnings'].extend(history['warnings'])
    except SimulationUnavailable as exc:
        summary["warnings"].append(str(exc))
        return result
    # One row per matched daily interval, one column per asset/FX factor.
    # ALL covariance entries use the same common sample, not pairwise subsets.
    x = joined.to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite calibration log returns.")
    # Column averages of LOG returns estimate daily log drift. Preserve this
    # window mean even when stress uses a baseline drift override: tail fitting
    # must center stress observations on their own mean.
    sample_mean, covariance = moments(joined)
    mean = sample_mean.copy()
    if drift_override is not None:
        if set(drift_override) != set(joined.columns):
            raise ValueError("Stress and baseline factors differ; cannot transfer drift.")
        mean = np.array([drift_override[k]["mean_daily_log"] for k in joined.columns])
    # Columns are factors (rowvar=False); sample covariance divides by M-1
    # (ddof=1). atleast_2d preserves matrix shape for a single factor.
    root = _root_covariance(covariance)
    keys = joined.columns.tolist()
    # The input validator already permits rounding within 1e-8 of one.
    # Resolve only that accepted rounding, not missing holdings or incorrect
    # allocations. Keep the configured values in metadata and the identity.
    configured_weights = np.array([h["weight"] for h in holdings])
    configured_total = math.fsum(configured_weights.tolist())
    weights = configured_weights/configured_total
    for holding,configured,effective in zip(holdings,configured_weights,weights):
        holding["configured_weight"] = float(configured)
        holding["weight"] = float(effective)
    if configured_total != 1.:
        summary["warnings"].append(
            f"Configured weights total {configured_total:.17g}; reconciled accepted rounding to a unit allocation for simulation. Original weights are retained in metadata.")
    # Record requested/actual coverage, currency and weights for audit.
    # Warnings expose shortened history or fallback rather than hiding them.
    summary.update(status="complete",simulation_currency=currency,
                   reporting_currency_returns_computed=currency==portfolio["reporting_currency"],
                   currency_fallback_used=currency!=portfolio["reporting_currency"],
                   holdings=holdings,configured_weight_sum=configured_total,
                   weight_rounding_reconciled=configured_total != 1.,
                   fx_factors=fx,forecast_dates=dates,steps=len(dates)-1,
                   calibration=dict(requested_start=requested_start,requested_end=risk_window["end"] if risk_window else _day(processed.cob),
                                    drift_source="baseline" if drift_override is not None else "same_window",first_start=joined.index[0][0],
                                    last_end=joined.index[-1][1],observations=len(joined),
                                    dropped_holding_intervals_for_fx=len(native)-len(joined)),
                   models={},warnings=summary["warnings"])
    if dates[-1] != cfg["forecast_end"]:
        summary["warnings"].append("Forecast end is not a session; last simulated session is "+dates[-1]+".")
    if joined.index[-1][1] != sessions[-1]:
        summary["warnings"].append("Calibration does not include its requested final session; actual last interval is "+joined.index[-1][1]+".")
    if joined.index[0][0] > sessions[0]:
        summary["warnings"].append("Available common history starts after the requested calibration start.")
    # Collect rows then assemble output tables once. Series order follows
    # the wealth tensor: all holdings first, then the portfolio.
    summaries,terminal_rows,bands,samples,diagnostics = [],[],[],[],[]
    daily_rows,histograms,horizon_rows,path_rows = [],[],[],[]
    series = [("holding",h["instrument_id"]) for h in holdings]+[("portfolio","portfolio")]
    # Keep simulated mean and own-window mean separately. Historical mean SE
    # is daily std/sqrt(M), assuming independent observations; it is NOT the
    # terminal-wealth Monte Carlo standard error calculated later.
    calibration_rows = [dict(factor=k,mean_daily_log=float(mean[i]),sample_mean_daily_log=float(sample_mean[i]),
                             drift_source="baseline" if drift_override is not None else "same_window",
                             std_daily_log=float(math.sqrt(covariance[i,i])),
                             mean_estimation_standard_error=(drift_override[k]["mean_estimation_standard_error"] if drift_override is not None
                                                            else float(math.sqrt(covariance[i,i]/len(joined)))),
                             observations=len(joined)) for i,k in enumerate(keys)]
    # Correlation C_ij/sqrt(C_ii*C_jj) is undefined for a constant factor;
    # export None rather than divide by zero or invent correlation.
    cov_rows = [dict(factor_i=a,factor_j=b,covariance_daily_log=float(covariance[i,j]),
                     correlation=float(covariance[i,j]/math.sqrt(covariance[i,i]*covariance[j,j]))
                     if covariance[i,i]>0 and covariance[j,j]>0 else None)
                for i,a in enumerate(keys) for j,b in enumerate(keys)]
    scenarios = []
    if "gaussian" in cfg["models"]:
        scenarios.append(("gaussian",None))
    if "student_t_truncated" in cfg["models"]:
        try:
            # Fit tails on actual window data centered on their own sample mean.
            # Changing simulated drift alone must not change the estimated tail shape.
            fitted = _student_parameters(x-sample_mean,root,cfg["student_t"])
        except SimulationUnavailable as exc:
            summary["status"] = "unavailable"
            summary["warnings"].append(str(exc))
            return result
        if fitted["df_at_boundary"]:
            summary["warnings"].append("Joint Student-t df fit is near a configured bound; inspect tail sensitivities.")
        if fitted["rank"] == 0:
            summary["warnings"].append("All factors are deterministic: df is not identifiable and no tail truncation is applied.")
        targets = [cfg["student_t"]["excluded_tail_probability"],
                   *cfg["student_t"]["sensitivity_tail_probabilities"]]
        # Each removed-mass setting creates a full scenario sharing calibrated
        # mean/covariance/df but with a separately derived truncation radius.
        for i,target in enumerate(targets):
            # Full float representation prevents collisions between nearby
            # probabilities and gives each sensitivity a stable random stream.
            name = "student_t_truncated" if i==0 else "student_t_tail_"+str(float(target))
            params = _truncation(fitted["rank"],fitted["df"],target)
            params.update(df=fitted["df"],df_method=fitted["method"],
                          calibration_outside_cutoff=int(np.sum(fitted["whitened_radii"]>params["radial_cutoff"])))
            scenarios.append((name,params))
    # Holding log moments in reporting currency are A*mu and A*C*A.T.
    # Portfolio LOG moments cannot use the same linear rule: a portfolio is
    # a sum of wealth, and the logarithm of a sum is nonlinear.
    converted_mean = transform @ mean
    # Transform the PSD factor used by the shock generator. Forming A C A.T
    # directly can subtract almost equal variances for a hedged asset/FX pair
    # and leave a negative diagonal due only to floating-point cancellation.
    # A transformed row's squared norm is nonnegative and agrees with the
    # actual simulated covariance. Materially invalid C is still rejected by
    # _root_covariance; this is not blanket clipping of negative variances.
    converted_root = transform @ root
    converted_cov = converted_root @ converted_root.T
    steps = len(dates)-1
    diag = cfg["diagnostics"]
    if diag["enabled"]:
        # Compare historical holding distributions in the SAME currency as paths.
        # Use their own window moments for z-scores; do not fabricate a historical
        # portfolio series for the daily distribution comparison.
        historical = x @ transform.T
        for i,holding in enumerate(holdings):
            info,hist = daily_distribution(historical[:,i],float(historical[:,i].mean()),
                                          float(historical[:,i].std(ddof=1)),diag["sigma_thresholds"],diag["histogram_bins"])
            tags = dict(model="historical",scope="holding",instrument_id=holding["instrument_id"],currency=currency,
                        source="historical",standardization="own_window_sample_mean_std")
            daily_rows.append(dict(tags,**info))
            histograms.append(hist.assign(**tags))
    for model,student in scenarios:
        summary["models"][model] = dict(student or {},distribution="gaussian" if student is None else "radially_truncated_multivariate_student_t")
        # Generate paths once per scenario. Every subsequent diagnostic reads
        # those paths; it does not alter shocks or consume random numbers.
        wealth,sim_mean,sim_cov = _run_scenario(model,mean,covariance,root,transform,weights,dates,cfg,student)
        for i,key in enumerate(keys):
            diagnostics.append(dict(model=model,kind="mean",factor_i=key,factor_j="",
                                    target=float(mean[i]),simulated=float(sim_mean[i])))
            for j,other in enumerate(keys):
                diagnostics.append(dict(model=model,kind="covariance",factor_i=key,factor_j=other,
                                        target=float(covariance[i,j]),simulated=float(sim_cov[i,j])))
        # Gaussian expected terminal holding/portfolio wealth provides a
        # closed-form benchmark independent of the random-number implementation.
        # Gaussian E[W_T] = W0*exp(n*(mu + variance/2)). This is the lognormal
        # EXPECTATION correction, not a change to the simulated drift. Sum holding
        # expectations for portfolio expectation; use this benchmark only for
        # Gaussian output rows, never as an asserted Student-t expectation.
        analytic_holdings = cfg["initial_value"]*np.exp(steps*(converted_mean+.5*np.diag(converted_cov)))
        analytic_means = np.r_[analytic_holdings,_portfolio_wealth(analytic_holdings,weights,cfg["initial_value"])]
        for i,(scope,key) in enumerate(series):
            # Slice (N,D) wealth for this holding/portfolio. All risk measures use
            # its actual path, including the portfolio's changing percentage weights.
            values = wealth[:,:,i]
            worst_daily = np.min(values[:,1:]/values[:,:-1]-1,axis=1)
            if diag["enabled"]:
                tags = dict(model=model,scope=scope,instrument_id=key,currency=currency)
                # Recover log(V_t/V_previous) from wealth. In particular, portfolio log
                # returns are not a fixed-weight sum of holding log returns.
                daily_log = np.diff(np.log(values),axis=1)
                # Portfolio weights drift, so its pooled standardized returns
                # use pooled sample moments, not a fixed weighted log mean.
                location = float(converted_mean[i]) if scope == "holding" else float(daily_log.mean())
                scale = float(math.sqrt(converted_cov[i,i])) if scope == "holding" else float(daily_log.std(ddof=1))
                info,hist = daily_distribution(daily_log,location,scale,diag["sigma_thresholds"],diag["histogram_bins"])
                labels = dict(tags,source="simulated",standardization="calibration_mean_std" if scope == "holding" else "pooled_simulation_mean_std")
                daily_rows.append(dict(labels,**info))
                histograms.append(hist.assign(**labels))
                horizon,path,worst_daily = wealth_diagnostics(values,cfg["initial_value"],diag["horizons_sessions"],cfg["confidence_levels"])
                horizon_rows.extend(dict(tags,**row) for row in horizon)
                path_rows.append(dict(tags,**path))
            # One final value per independent path. Loss = 1-final/W0; gains produce
            # negative losses. Loss probability uses the same machine-precision
            # zero rule as horizon diagnostics; other statistics retain raw values.
            final = values[:,-1]
            losses = 1-final/cfg["initial_value"]
            # Track each path's running peak, including initial wealth. The most
            # negative wealth/peak-1 is the maximum drawdown. Intervening positive days
            # are allowed; it is not a count of consecutive losing days.
            max_drawdown = np.min(values/np.maximum.accumulate(values,axis=1)-1,axis=1)
            # SE of mean terminal wealth = sample std across paths / sqrt(N), in
            # wealth units. It measures simulation noise, not investment dispersion
            # or uncertainty in historical parameters/model selection.
            mc_se = float(final.std(ddof=1)/np.sqrt(cfg["paths"]))
            row = dict(model=model,scope=scope,instrument_id=key,currency=currency,
                       mean_terminal=float(final.mean()),median_terminal=float(np.median(final)),
                       mean_mc_standard_error=mc_se,probability_loss=probability_of_loss(losses),
                       mean_max_drawdown=float(max_drawdown.mean()),
                       max_drawdown_p05=float(np.quantile(max_drawdown,.05)))
            for q in cfg["quantiles"]:
                row[f"terminal_quantile_{probability_label(q)}"] = float(np.quantile(final,q))
            # These VaR/ES values cover the FULL forecast. Short-horizon loss measures
            # are separately exported by wealth_diagnostics.
            for q in cfg["confidence_levels"]:
                row[f"loss_var_{probability_label(q)}"] = float(np.quantile(losses,q))
                row[f"loss_es_{probability_label(q)}"] = _expected_shortfall(losses,q)
            if student is None:
                row["gaussian_expected_terminal"] = float(analytic_means[i])
                row["gaussian_mean_error_in_mc_se"] = float((final.mean()-analytic_means[i])/mc_se) if mc_se>1e-12 else None
            summaries.append(row)
            # Preserve each path's terminal wealth/return and worst-day/drawdown
            # for inspection. Fan bands below summarize cross-path wealth by date.
            terminal_rows.append(pd.DataFrame(dict(model=model,scope=scope,instrument_id=key,currency=currency,
                                                   path=np.arange(cfg["paths"]),terminal_value=final,
                                                   terminal_return=final/cfg["initial_value"]-1,max_drawdown=max_drawdown,worst_daily_return=worst_daily)))
            # Fixed 5/50/95 bands are always available for comparable fan charts;
            # user-requested quantiles are added as separately labelled columns.
            quantile_values = np.quantile(values,[.05,.5,.95],axis=0)
            band = pd.DataFrame(dict(model=model,scope=scope,instrument_id=key,currency=currency,date=dates,
                                     mean=values.mean(axis=0),p05=quantile_values[0],p50=quantile_values[1],p95=quantile_values[2]))
            for q in cfg["quantiles"]:
                band[f"quantile_{probability_label(q)}"] = np.quantile(values,q,axis=0)
            bands.append(band)
            # Save the first requested path IDs, not a newly sampled subset. Every
            # simulated path still contributes to risk statistics and bands.
            for path in range(cfg["sample_paths"]):
                samples.append(pd.DataFrame(dict(model=model,scope=scope,instrument_id=key,currency=currency,
                                                  path=path,date=dates,value=values[path])))
    # Package calibration/results/diagnostics; do not write files here.
    # Export is explicitly requested later through result.save(destination).
    result.tables = dict(
        calibration_returns=joined.reset_index(),calibration_parameters=pd.DataFrame(calibration_rows),
        calibration_covariance=pd.DataFrame(cov_rows),risk_summary=pd.DataFrame(summaries),
        terminal_values=pd.concat(terminal_rows,ignore_index=True),bands=pd.concat(bands,ignore_index=True),
        sample_paths=pd.concat(samples,ignore_index=True) if samples else pd.DataFrame(columns=["model","scope","instrument_id","currency","path","date","value"]),
        simulation_diagnostics=pd.DataFrame(diagnostics),
        daily_distribution_diagnostics=pd.DataFrame(daily_rows),
        daily_histograms=pd.concat(histograms,ignore_index=True) if histograms else pd.DataFrame(),
        horizon_risk=pd.DataFrame(horizon_rows),path_risk=pd.DataFrame(path_rows))
    summary["simulation_performed"] = True
    return result


def simulate_saved_processing(folder,config_file):
    """Read only hash-verified cleaned inputs; no downloader or statistics rerun."""
    from .data_cleaning import ProcessedArchive
    folder = Path(folder).resolve()
    # For saved input, verify completion and each used table. Folder existence
    # alone is insufficient evidence that a cleaned run is valid.
    manifest = _read_json(folder/"manifest.json")
    if manifest.get("complete") is not True:
        raise ValueError("Cleaning output is incomplete.")
    tables = {}
    for name in ("selected_prices","selected_references"):
        if name not in manifest["files"]:
            if name == "selected_prices":
                raise ValueError("Missing selected_prices table.")
            continue
        entry = manifest["files"][name]
        # Resolve manifest references and reject traversal/symlink escape before
        # reading the file. Only contained cleaned tables may become inputs.
        path = (folder/entry["file"]).resolve()
        if not path.is_relative_to(folder) or path == folder:
            raise ValueError("Cleaned input path escapes its source folder.")
        # Verify exact bytes before parsing and row count afterward. Identifier
        # and date columns stay strings to preserve their meaning.
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError("Cleaned input checksum mismatch: "+name)
        table = pd.read_csv(BytesIO(raw),dtype={"instrument_id":str,"date":str,"return_start_date":str})
        if len(table) != entry["rows"]:
            raise ValueError("Cleaned input row-count mismatch: "+name)
        tables[name] = table
    # Reconstruct the cleaned-input object, then call the same entry point as
    # run.py. Remember source_folder to protect it during subsequent export.
    result = simulate_processed(ProcessedArchive(Path(manifest["archive"]),manifest["cob"],manifest["run_id"],tables,manifest),
                                config_file)
    result.source_folder = folder
    return result
