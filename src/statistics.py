"""Offline historical statistics over cleaned prices, with no provider/API calls.

Entry points and calculation flow:
    analyze_processed(processed, modeling_file)
        -> read selected_prices produced by data_cleaning
        -> AssetStatistics(...) for each positive-weight holding
            -> __post_init__() runs automatically because this is a dataclass
                -> _prepare(): validate simple returns and derive log returns
                -> shared price-path metrics, then statistics per return type
        -> _portfolio_series(): construct buy-and-hold wealth and simple returns
            -> AssetStatistics(...) runs the same analysis for the portfolio
        -> assemble a PortfolioStatistics result; save() exports it explicitly
    analyze_saved_processing(folder, modeling_file) verifies a saved cleaning
    output and then delegates to the same analyze_processed() entry point.

Data contract: one row per expected session, sorted by ISO date, with positive
analysis_price or a missing value. Cleaning already supplies simple_return;
missing returns can also represent deliberately excluded intervals. Return
values are decimals (0.01 means 1% for simple returns), not percentage points.
Both return columns share the same valid intervals; return_types selects which
conventions get statistical summaries, not which daily columns are retained.

Adjusted observations represent reinvested wealth; do not add dividends again.
Portfolio aggregation is buy-and-hold, with explicit currency and return-basis
guards. Reference instruments and option quotes are not portfolio holdings.
"""
from copy import deepcopy
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
import hashlib
import json
import math
import warnings

import numpy as np
import pandas as pd

from .json_config import _unique_object, _invalid_constant
from .storage import safe_name, write_json


def _read_json(path):
    """Read strict UTF-8 JSON, rejecting duplicate keys and non-finite constants."""
    return json.loads(Path(path).read_text(encoding="utf-8-sig"),
                      object_pairs_hook=_unique_object, parse_constant=_invalid_constant)


def _day(value):
    """Require canonical ISO dates; never infer dates from capture timestamps."""
    try:
        stamp = pd.Timestamp(value) if isinstance(value, str) else pd.NaT
        if pd.isna(stamp) or stamp.strftime("%Y-%m-%d") != value:
            raise ValueError()
    except (ValueError, TypeError):
        raise ValueError("Statistics dates must use YYYY-MM-DD.") from None
    return value


def load_modeling_config(path):
    """Load the statistics section of modeling.json, or raise ValueError.

    The instructions section is explanatory text; only statistics drives the
    calculations. Validate options here so unsupported settings cannot appear
    to have taken effect in the exported report.
    """
    data = _read_json(path)
    if not isinstance(data, dict) or set(data) - {"schema_version", "statistics", "instructions"}:
        raise ValueError("Unknown modeling configuration fields.")
    if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise ValueError("Modeling schema_version must be 1.")
    cfg = data.get("statistics")
    keys = {"periods_per_year", "minimum_observations", "rolling_windows", "quantiles",
            "stress_periods", "portfolio_method", "base_value", "plots", "fit_student_t", "return_types"}
    if not isinstance(cfg, dict) or set(cfg) != keys:
        raise ValueError("Missing or unknown statistics settings.")
    _return_types(cfg)
    for name in ("periods_per_year", "minimum_observations"):
        if type(cfg[name]) is not int or cfg[name] < 2:
            raise ValueError(name + " must be an integer >= 2.")
    windows = cfg["rolling_windows"]
    if (not isinstance(windows, list) or not windows
            or any(type(x) is not int or x < 2 for x in windows)
            or len(set(windows)) != len(windows)):
        raise ValueError("rolling_windows must contain distinct integers >= 2.")
    quantiles = cfg["quantiles"]
    if (not isinstance(quantiles, list) or not quantiles
            or any(type(x) not in (int, float) or not math.isfinite(x) or not 0 < x < 1 for x in quantiles)
            or len(set(quantiles)) != len(quantiles)):
        raise ValueError("quantiles must contain distinct fractions between 0 and 1.")
    for name in ("plots", "fit_student_t"):
        if type(cfg[name]) is not bool:
            raise ValueError(name + " must be boolean.")
    if cfg["portfolio_method"] != "buy_and_hold":
        raise ValueError("Only buy_and_hold statistics are implemented.")
    if (type(cfg["base_value"]) not in (int, float)
            or not math.isfinite(cfg["base_value"]) or cfg["base_value"] <= 0):
        raise ValueError("base_value must be finite and positive.")
    if not isinstance(cfg["stress_periods"], list):
        raise ValueError("stress_periods must be an array.")
    # Stress bounds are inclusive calendar dates, not necessarily trading days.
    names = set()
    for period in cfg["stress_periods"]:
        if not isinstance(period, dict) or set(period) != {"name", "start", "end"}:
            raise ValueError("Each stress period needs name, start and end.")
        if not isinstance(period["name"], str) or not period["name"] or period["name"] in names:
            raise ValueError("Stress names must be distinct non-empty strings.")
        if _day(period["start"]) > _day(period["end"]):
            raise ValueError("Stress start must not follow end.")
        names.add(period["name"])
    return cfg


def _return_types(cfg):
    """Require explicit, unique return conventions; never silently substitute one."""
    kinds = cfg.get("return_types")
    if (not isinstance(kinds, list) or not kinds
            or any(not isinstance(k, str) or k not in {"simple", "log"} for k in kinds)
            or len(set(kinds)) != len(kinds)):
        raise ValueError("return_types must contain simple and/or log, without duplicates.")
    return kinds


def _return_column(return_type):
    """A single dispatch point keeps all calculations on the labelled convention."""
    if return_type not in {"simple", "log"}:
        raise ValueError("Unknown return type.")
    return return_type + "_return"


def _finite(value):
    """Represent undefined moments as None for strict JSON output."""
    return float(value) if pd.notna(value) and math.isfinite(float(value)) else None


def _prepare(frame):
    """Check selected observations, preserving missing/excluded daily returns.

    The preceding expected row, not the preceding non-missing price, defines the
    interval. Older cleaned CSVs without return_start_date can therefore be read
    without filling gaps. Existing interval metadata is checked when present.
    """
    if not {"date", "analysis_price", "simple_return"} <= set(frame.columns):
        raise ValueError("Selected prices need date, analysis_price and simple_return.")
    # Work on an independent, chronological frame; shift() below means the
    # preceding expected session, including rows whose price is missing.
    out = frame.copy(deep=True).sort_values("date").reset_index(drop=True)
    if out["date"].duplicated().any():
        raise ValueError("Duplicate dates in selected prices.")
    for value in out["date"]:
        _day(value)
    for name in ("analysis_price", "simple_return"):
        original = out[name]
        out[name] = pd.to_numeric(original, errors="coerce")
        if (original.notna() & out[name].isna()).any() or np.isinf(out[name]).any():
            raise ValueError("Non-numeric or infinite selected " + name)
    if (out["analysis_price"].dropna() <= 0).any() or (out["simple_return"].dropna() <= -1).any():
        raise ValueError("Prices must be positive and returns must exceed -100%.")
    # Simple return r_t = P_t / P_previous - 1. This is a consistency check:
    # keep cleaning's supplied values and exclusions rather than replacing them.
    # A missing current/previous price must leave the interval unavailable.
    expected = out["analysis_price"] / out["analysis_price"].shift() - 1
    supplied = out["simple_return"].notna()
    if (supplied & expected.isna()).any() or not np.allclose(
            out.loc[supplied, "simple_return"], expected[supplied], rtol=1e-8, atol=1e-12):
        raise ValueError("Selected returns disagree with adjacent selected prices.")
    # Store both interval endpoints. Matching end dates alone is insufficient
    # when holdings follow different market holiday calendars.
    starts = out["date"].shift().where(supplied)
    if "return_start_date" in out and not out.loc[supplied, "return_start_date"].eq(starts[supplied]).all():
        raise ValueError("Selected return intervals disagree with expected sessions.")
    out["return_start_date"] = starts
    # Log return g_t = ln(1 + r_t) = ln(P_t / P_previous). log1p keeps
    # precision for small returns and preserves missing/excluded intervals.
    out["log_return"] = np.log1p(out["simple_return"])
    return out


def _path_statistics(frame):
    """Shared investment-path metrics, independent of simple versus log returns.

    Growth, CAGR and percentage drawdown always use observed wealth/prices.
    Excluded intervals and interior gaps still withhold full-path metrics.
    """
    result = dict(cumulative_return=None, cagr=None, max_drawdown=None,
                  first_price_date=None, last_price_date=None, interior_missing_sessions=0,
                  path_status="no_prices")
    valid = frame["analysis_price"].notna()
    if valid.any():
        observed = frame.loc[valid]
        # Ignore unavailable history before inception and after the last price,
        # but withhold path metrics for any break between observed endpoints.
        path = frame.loc[observed.index[0]:observed.index[-1]]
        missing = int(path["analysis_price"].isna().sum())
        broken = missing > 0 or path["simple_return"].iloc[1:].isna().any()
        result.update(first_price_date=observed.iloc[0]["date"], last_price_date=observed.iloc[-1]["date"],
                      interior_missing_sessions=missing, path_status="unavailable_gaps" if broken else "available")
        if not broken:
            prices = path["analysis_price"]
            # Endpoint growth compounds the entire observed path. CAGR uses
            # calendar elapsed years, while volatility uses sessions per year.
            # Drawdown compares each price with its running historical peak.
            growth = prices.iloc[-1]/prices.iloc[0]
            years = (pd.Timestamp(path.iloc[-1]["date"])-pd.Timestamp(path.iloc[0]["date"])).days/365.25
            with np.errstate(over="ignore"):
                result.update(cumulative_return=float(growth-1),
                              cagr=_finite(np.expm1(np.log(growth)/years)) if years > 0 else None,
                              max_drawdown=float((prices/prices.cummax()-1).min()))
    return result


def _sample_statistics(frame, cfg, return_type="simple"):
    """Describe the requested daily return column without altering the sample.

    Sample std uses ddof=1; skew/kurtosis are bias corrected. Mean times the
    annualization factor is arithmetic for simple returns and additive log
    growth for log returns. Neither field is the shared investment CAGR.
    """
    returns = frame[_return_column(return_type)].dropna()
    # ddof=1 estimates sample variance using N-1. Undefined small-sample
    # moments become None, while minimum_observations only controls the status.
    n, sd = len(returns), returns.std(ddof=1)
    # sqrt(periods_per_year) is the conventional volatility scaling; it does
    # not estimate serial dependence or promise future annual volatility.
    result = dict(return_type=return_type, observations=n,
                  status="available" if n >= cfg["minimum_observations"] else
                  ("insufficient_history" if n else "no_returns"),
                  mean_daily=_finite(returns.mean()), std_daily=_finite(sd),
                  mean_annualized=_finite(returns.mean()*cfg["periods_per_year"]),
                  volatility_annualized=_finite(sd*math.sqrt(cfg["periods_per_year"])),
                  skewness=_finite(returns.skew()) if n >= 3 and sd > 0 else None,
                  excess_kurtosis=_finite(returns.kurt()) if n >= 4 and sd > 0 else None,
                  worst_daily=_finite(returns.min()), best_daily=_finite(returns.max()))
    for q in cfg["quantiles"]:
        # Keep the full probability in the column name: :g rounds to six
        # significant digits and can silently merge two distinct quantiles.
        result[f"quantile_{q}"] = _finite(returns.quantile(q)) if n else None
    return result


@dataclass
class AssetStatistics:
    """Analytical wrapper over a cleaned holding, distinct from raw PriceSeries.

    Construction is the calculation trigger: dataclasses generates __init__,
    assigns frame/settings, and calls __post_init__ automatically. Callers do
    not need a separate calculate() call. Fields with init=False are outputs.
    The same definitions apply to the derived portfolio wealth series.

    return_statistics["simple"] / ["log"] hold metrics, stress samples,
    rolling volatility and distribution fits for each requested convention.
    path_metrics and stress_paths are shared price/wealth results. The legacy
    metrics/stress/rolling/distribution attributes expose the primary type.
    """
    instrument_id: str
    currency: str | None
    return_basis: str | None
    frame: pd.DataFrame
    settings: dict
    scope: str = "holding"
    metrics: dict = field(init=False)
    stress: list = field(init=False)
    rolling: pd.DataFrame = field(init=False)
    distribution: dict = field(init=False)
    path_metrics: dict = field(init=False)
    stress_paths: list = field(init=False)
    return_statistics: dict = field(init=False)

    def __post_init__(self):
        """Prepare both daily return columns, then compute requested summaries."""
        # Reached automatically from AssetStatistics(...) in analyze_processed
        # and _portfolio_series. _prepare supplies the log-return calculation.
        kinds = _return_types(self.settings)
        self.frame = _prepare(self.frame)
        self.path_metrics = _path_statistics(self.frame)
        # Slice and validate each stress path once, before choosing a return
        # column, so both conventions always describe identical intervals.
        samples, self.stress_paths = [], []
        for period in self.settings["stress_periods"]:
            sample = self.frame.loc[self.frame["date"].between(period["start"],period["end"])].copy()
            # An interval ending inside the window can still begin outside it;
            # remove that return, retaining its endpoint price for path metrics.
            outside = sample["return_start_date"].notna() & (sample["return_start_date"] < period["start"])
            sample.loc[outside,["simple_return","log_return"]] = np.nan
            samples.append((period,sample))
            self.stress_paths.append({**period,**_path_statistics(sample)})
        self.return_statistics = {}
        for kind in kinds:
            metrics = _sample_statistics(self.frame,self.settings,kind)
            stress = [{**period,**_sample_statistics(sample,self.settings,kind)} for period,sample in samples]
            rolling = self.frame[["date"]].copy()
            # Keep session rows rather than dropna(): min_periods=window makes
            # any missing/excluded interval break a rolling volatility window.
            for window in self.settings["rolling_windows"]:
                rolling[f"volatility_{window}"] = (
                    self.frame[_return_column(kind)].rolling(window,min_periods=window).std(ddof=1)
                    * math.sqrt(self.settings["periods_per_year"]))
            self.return_statistics[kind] = dict(metrics=metrics,stress=stress,rolling=rolling,
                                                distribution=self._distribution_fit(kind,metrics))
        # Preserve the original convenience attributes for library callers.
        # Prefer simple when enabled; otherwise use the explicitly labelled log
        # analysis. New callers should address return_statistics[kind] directly.
        primary = self.return_statistics["simple" if "simple" in kinds else kinds[0]]
        self.metrics = {**primary["metrics"],**self.path_metrics,
                        "mean_log_daily":_finite(self.frame["log_return"].mean())}
        self.stress = [{**m,**p} for m,p in zip(primary["stress"],self.stress_paths)]
        self.rolling = primary["rolling"]
        self.distribution = primary["distribution"]

    def _distribution_fit(self, return_type, metrics):
        """Fit original simple/log observations, not standardized Q-Q values."""
        r = self.frame[_return_column(return_type)].dropna()
        out = dict(return_type=return_type,normal_mean=metrics["mean_daily"],normal_std=metrics["std_daily"],
                   student_t_status="disabled",student_t_df=None,student_t_location=None,student_t_scale=None)
        if not self.settings["fit_student_t"]:
            return out
        if len(r) < self.settings["minimum_observations"] or not r.std(ddof=1) > 0:
            out["student_t_status"] = "insufficient_or_constant_returns"
            return out
        from scipy.stats import t
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error",RuntimeWarning)
                # Fit the selected raw daily returns. Student-t scale is a
                # distribution parameter; its standard deviation, when df > 2,
                # is scale * sqrt(df / (df - 2)).
                df, location, scale = t.fit(r.to_numpy())
            if not all(math.isfinite(x) for x in (df,location,scale)) or df <= 0 or scale <= 0:
                raise ValueError("Invalid fitted parameters.")
            out.update(student_t_status="available",student_t_df=float(df),student_t_location=float(location),
                       student_t_scale=float(scale),student_t_variance_exists=bool(df > 2))
        # A numerical fit failure must not discard valid empirical statistics
        # or prevent the other return convention from being analyzed.
        except (ValueError,RuntimeError,RuntimeWarning,FloatingPointError) as exc:
            out.update(student_t_status="fit_failed",diagnostic=str(exc))
        return out


def _basis_kind(value):
    """Group cleaning's detailed basis labels for portfolio compatibility.

    Different total-return conventions remain labelled on individual assets;
    grouping here only distinguishes reinvested wealth from price-only levels.
    Unknown labels cannot be assumed compatible.
    """
    if value in {"split_and_dividend_adjusted_close","gross_total_return_index_level","net_total_return_index_level"}:
        return "total_return"
    if value in {"price_index_level","split_adjusted_close","unadjusted_close"}:
        return "price_return"
    return "unknown"


def _joint_returns(assets, return_type="simple"):
    """Outer-join returns on (start, end); leave unavailable values missing.

    For example, a Thu-to-Mon return cannot match a Fri-to-Mon return even
    though both end on Monday. Callers use dropna() to require every holding
    to have a valid observation for the exact same interval.
    """
    column = _return_column(return_type)
    parts = []
    for key, asset in assets.items():
        part = asset.frame[["date","return_start_date",column]].dropna()
        parts.append(part.set_index(["return_start_date","date"])[column].rename(key))
    if not parts:
        return pd.DataFrame(index=pd.MultiIndex.from_arrays([[],[]],names=["return_start_date","date"]))
    return pd.concat(parts,axis=1).sort_index(level=["date","return_start_date"])


def _dependence(assets, cfg, return_type="simple"):
    """Common-sample covariance/correlation; rolling windows retain missing dates."""
    # Use one common sample across ALL holdings, rather than different samples
    # for each pair. The exported observation count refers to this sample.
    complete = _joint_returns(assets,return_type).dropna()
    covariance = complete.cov() if len(complete) >= 2 else pd.DataFrame(np.nan,index=list(assets),columns=list(assets))
    correlation = complete.corr() if len(complete) >= 2 else covariance.copy()
    rows = [dict(instrument_i=a,instrument_j=b,observations=len(complete),
                 covariance_daily=_finite(covariance.loc[a,b]),
                 covariance_annualized=_finite(covariance.loc[a,b]*cfg["periods_per_year"]),
                 correlation=_finite(correlation.loc[a,b])) for a in assets for b in assets]
    # Restore the union of expected sessions after alignment. Without reindex,
    # rolling() would compress gaps into apparently consecutive observations.
    dates = sorted({d for a in assets.values() for d in a.frame["date"]})
    daily = complete.copy()
    daily.index = complete.index.get_level_values("date")
    daily = daily.reindex(dates)
    rolling, keys = [], list(assets)
    for i,a in enumerate(keys):
        for b in keys[i+1:]:
            for window in cfg["rolling_windows"]:
                values = daily[a].rolling(window,min_periods=window).corr(daily[b])
                rolling.extend(dict(date=d,instrument_i=a,instrument_j=b,window=window,correlation=_finite(v))
                               for d,v in values.items())
    # Prefix IDs so a holding named "date" cannot collide with index columns.
    aligned = complete.rename(columns={key:"return::"+key for key in assets}).reset_index()
    return (pd.DataFrame(rows,columns=["instrument_i","instrument_j","observations","covariance_daily",
                                      "covariance_annualized","correlation"]),
            pd.DataFrame(rolling,columns=["date","instrument_i","instrument_j","window","correlation"]),aligned)


def _portfolio_series(assets, holdings, cfg):
    """Aggregate fixed initial weights, not daily-rebalanced weighted returns.

    All active holdings need observed prices in one currency and a compatible
    return meaning. Missing holdings do not trigger weight renormalization.
    """
    keys = [h["request_id"] for h in holdings]
    if any(k not in assets or not assets[k].frame["analysis_price"].notna().any() for k in keys):
        return None,"unavailable_missing_holdings"
    currencies = {assets[k].currency for k in keys}
    if None in currencies or len(currencies) != 1:
        return None,"unavailable_currency_conversion"
    kinds = {_basis_kind(assets[k].return_basis) for k in keys}
    if "unknown" in kinds or len(kinds) != 1:
        return None,"unavailable_incompatible_return_bases"
    levels = pd.concat([assets[k].frame.set_index("date")["analysis_price"].rename(k) for k in keys],axis=1).sort_index()
    complete = levels.dropna()
    if complete.empty:
        return None,"unavailable_no_common_prices"
    # All sleeves start together on the first date with a price for every
    # holding. Earlier individual history remains in each AssetStatistics.
    levels = levels.loc[complete.index[0]:]
    weights = pd.Series({h["request_id"]:h["weight"] for h in holdings})
    # V_t = base_value * sum_i(weight_i * P_i,t / P_i,start). This fixes the
    # initial allocation once; relative sleeve weights drift as prices move.
    # min_count requires every sleeve: a missing price cannot shrink the
    # portfolio into a partial total or trigger weight renormalization.
    wealth = (levels.div(complete.iloc[0])*weights*cfg["base_value"]).sum(axis=1,min_count=len(keys))
    # Portfolio simple returns come from aggregate wealth. Log returns will
    # be derived by AssetStatistics -> __post_init__ -> _prepare below.
    returns = wealth/wealth.shift()-1
    # Even with both wealth endpoints present, every underlying return must
    # be admissible for the same interval (including cleaning exclusions).
    allowed = set(_joint_returns({k:assets[k] for k in keys}).dropna().index)
    starts = pd.Series(wealth.index,index=wealth.index).shift()
    returns = returns.where([(s,d) in allowed for s,d in zip(starts,wealth.index)])
    frame = pd.DataFrame({"date":wealth.index,"analysis_price":wealth.to_numpy(),"simple_return":returns.to_numpy()})
    return AssetStatistics("portfolio",next(iter(currencies)),next(iter(kinds)),frame,cfg,scope="portfolio"),"available"



@dataclass
class PortfolioStatistics:
    """Container for computed results, with explicit reporting/export methods.

    analyze_processed() populates assets, portfolio, summary and tables; this
    dataclass does not trigger calculations. report() only formats text,
    plots_report() writes PNGs, and save() exports the complete output bundle.
    """
    archive: Path
    processing_id: str
    cob: str
    run_id: str
    settings: dict
    assets: dict[str, AssetStatistics]
    portfolio: AssetStatistics | None
    summary: dict
    tables: dict[str, pd.DataFrame]
    source_folder: Path | None = None

    def report(self):
        """Compare return conventions while reporting each wealth path only once."""
        def fmt(value, percent=False):
            value = _finite(value)
            return "N/A" if value is None else (f"{100*value:.2f}%" if percent else f"{value:.3f}")
        lines = ["# Historical statistics","",f"Archive COB: {self.cob}",
                 f"Processing ID: {self.processing_id}",f"Statistics ID: {self.run_id}","",
                 "Historical description only. No simulation, pricing, FX conversion or API requests.","",
                 f"Portfolio status: {self.summary['portfolio_status']}",
                 f"Portfolio analysis currency: {self.summary['analysis_currency'] or 'unavailable'}",
                 f"Configured reporting currency: {self.summary['reporting_currency']}",
                 "Initial weights: "+", ".join(f"{h['request_id']} {h['weight']:.1%}"
                     for h in self.summary["identity"]["portfolio"]["holdings"]),
                 "Method: buy-and-hold initial weights; sleeve weights drift. No periodic rebalancing.","",
                 "## Shared investment paths","",
                 "These dates can differ between series: compare CAGR only over the same period.",
                 "CAGR, cumulative return and percentage drawdown do not depend on the return convention.","",
                 "| Series | Currency | First price | Last price | Cumulative return | CAGR | Max drawdown | Path status |",
                 "|---|---|---|---|---:|---:|---:|---|"]
        for row in self.tables["path_statistics"].to_dict("records"):
            lines.append(f"| {row['scope']}: {row['instrument_id']} | {row['currency']} | "
                         f"{row['first_price_date']} | {row['last_price_date']} | {fmt(row['cumulative_return'],True)} | "
                         f"{fmt(row['cagr'],True)} | {fmt(row['max_drawdown'],True)} | {row['path_status']} |")
        for kind in self.settings["return_types"]:
            unit = "percentage returns" if kind == "simple" else "log-return units x 100 (not percentage investment gains)"
            lines += ["",f"## {kind.capitalize()} returns","",f"Mean and volatility below are displayed as {unit}.","",
                      "| Series | Daily mean x 100 | Annual volatility x 100 | Skewness | Excess kurtosis | N | Status |",
                      "|---|---:|---:|---:|---:|---:|---|"]
            rows = self.tables["statistics"]
            for row in rows.loc[rows["return_type"] == kind].to_dict("records"):
                lines.append(f"| {row['scope']}: {row['instrument_id']} | {fmt(row['mean_daily']*100 if row['mean_daily'] is not None else None)} | "
                             f"{fmt(row['volatility_annualized']*100 if row['volatility_annualized'] is not None else None)} | "
                             f"{fmt(row['skewness'])} | {fmt(row['excess_kurtosis'])} | {row['observations']} | {row['status']} |")
            lines += ["","### Common-interval correlations","",
                      "| Holding A | Holding B | Correlation | Common intervals |",
                      "|---|---|---:|---:|"]
            dep = self.tables["dependence"]
            seen = set()
            for row in dep.loc[dep["return_type"] == kind].to_dict("records"):
                pair = frozenset((row["instrument_i"],row["instrument_j"]))
                if row["instrument_i"] == row["instrument_j"] or pair in seen:
                    continue
                seen.add(pair)
                lines.append(f"| {row['instrument_i']} | {row['instrument_j']} | {fmt(row['correlation'])} | {row['observations']} |")
            lines += ["","### Stress comparisons","",
                      "| Series | Window | Returns | Annual volatility x 100 | Status |",
                      "|---|---|---:|---:|---|"]
            stress = self.tables["stress_statistics"]
            for row in stress.loc[stress["return_type"] == kind].to_dict("records"):
                lines.append(f"| {row['scope']}: {row['instrument_id']} | {row['name']} | {row['observations']} | "
                             f"{fmt(row['volatility_annualized']*100 if row['volatility_annualized'] is not None else None)} | "
                             f"{row['status']} / {row['coverage']} |")
        lines += ["","## Interpretation","",
                  "- All CSV return values use decimal units and are labelled return_type; shared path tables have no return_type.",
                  "- Simple return = P_t/P_previous - 1; log return = log1p(simple return), using the same valid intervals.",
                  "- Portfolio log returns come from portfolio wealth, never a weighted sum of holding log returns.",
                  "- Annualized mean = daily mean times periods_per_year: arithmetic for simple, additive log growth for log. Neither is CAGR.",
                  "- Annual volatility = sample daily standard deviation times sqrt(periods_per_year).",
                  "- Individual statistics use own histories; covariance and correlation use identical common intervals.",
                  "- Missing sessions are not filled; full-path metrics are withheld across gaps or excluded intervals.",
                  "- Quantiles are empirical. Normal/t curves describe the selected return type; Q-Q values alone are standardized.",
                  "- Student-t scale is not standard deviation. Fitted moment-existence flags are not statements about finite sample moments.",
                  "- Stress price coverage, CAGR and drawdown appear once in stress_path_statistics.csv.",
                  "- Adjusted returns already include distribution adjustments; dividends are not added again.",
                  "- Statistics describe history and do not establish constant volatility/correlation or forecast returns."]
        lines += ["- "+warning for warning in self.summary["warnings"]]
        return "\n".join(lines)+"\n"

    def plots_report(self, destination):
        """Shared wealth/drawdown charts plus explicitly typed return diagnostics."""
        import matplotlib
        # Render PNGs without opening a GUI, including during offline tests.
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        from scipy import stats
        destination = Path(destination)
        self._check_destination(destination)
        if destination.is_symlink():
            raise ValueError("Refusing symlink plot directory.")
        destination.mkdir(parents=True,exist_ok=True)
        files = []

        def save_figure(fig,path):
            self._check_destination(path)
            if path.is_symlink():
                plt.close(fig)
                raise ValueError("Refusing symlink plot output.")
            try:
                fig.savefig(path,dpi=130)
                files.append(path)
            finally:
                plt.close(fig)

        # One pair of path charts: a log transformation must never change
        # historical investment wealth or percentage drawdown.
        for column,name,title,ylabel,factor in (
                ("normalized_value","wealth","Normalized observed wealth / price index","Index level",1),
                ("drawdown","drawdown","Historical drawdown (complete observed paths)","Percent below previous peak",100)):
            fig,ax = plt.subplots(figsize=(11,4),constrained_layout=True)
            for (scope,key),rows in self.tables["daily_statistics"].groupby(["scope","instrument_id"]):
                if rows[column].notna().any():
                    ax.plot(pd.to_datetime(rows["date"]),rows[column]*factor,label=f"{scope}: {key}",linewidth=1)
            ax.set(title=title+("\nEach series starts at its own first observed date" if name == "wealth" else ""),ylabel=ylabel)
            if ax.lines:
                ax.legend()
            else:
                ax.text(.5,.5,"No usable observed paths",ha="center",transform=ax.transAxes)
            save_figure(fig,destination/(name+".png"))

        for kind in self.settings["return_types"]:
            folder = destination/kind
            self._check_destination(folder)
            if folder.is_symlink():
                raise ValueError("Refusing symlink plot directory.")
            folder.mkdir(exist_ok=True)
            units = "Simple return (%)" if kind == "simple" else "Log return x 100"
            volatility_units = "Percent" if kind == "simple" else "Log-return units x 100"
            for a in list(self.assets.values())+([self.portfolio] if self.portfolio else []):
                analysis = a.return_statistics[kind]
                r = a.frame[_return_column(kind)].dropna()
                fig,axes = plt.subplots(1,3,figsize=(16,4.5),constrained_layout=True)
                dates = pd.to_datetime(a.frame["date"])
                for window in self.settings["rolling_windows"]:
                    axes[0].plot(dates,analysis["rolling"][f"volatility_{window}"]*100,
                                 label=f"{window} sessions",linewidth=1)
                axes[0].set(title="Rolling annualized volatility",ylabel=volatility_units)
                axes[0].legend()
                if len(r) >= 2 and r.std(ddof=1) > 0:
                    x = np.linspace(r.min(),r.max(),300)
                    # Density transforms by /100 when the x-axis is multiplied
                    # by 100. Fits use unstandardized observations of this kind.
                    axes[1].hist(r*100,bins=min(70,max(5,int(np.sqrt(len(r))))),density=True,alpha=.45)
                    fit = analysis["distribution"]
                    axes[1].plot(x*100,stats.norm.pdf(x,fit["normal_mean"],fit["normal_std"])/100,label="Normal")
                    if fit["student_t_status"] == "available":
                        axes[1].plot(x*100,stats.t.pdf(x,fit["student_t_df"],
                                     loc=fit["student_t_location"],scale=fit["student_t_scale"])/100,label="Student-t")
                    axes[1].legend()
                    # Standardization is solely a shape diagnostic for the
                    # normal Q-Q panel; it does not replace analyzed returns.
                    stats.probplot((r-r.mean())/r.std(ddof=1),dist="norm",plot=axes[2])
                else:
                    for ax in axes[1:]:
                        ax.text(.5,.5,"Insufficient nonconstant returns",ha="center",transform=ax.transAxes)
                axes[1].set(title=f"Daily {kind}-return distribution",xlabel=units,ylabel="Density")
                axes[2].set(title=f"Normal Q-Q: standardized {kind} returns")
                fig.suptitle(f"{a.scope}: {a.instrument_id} | {kind} returns | {a.currency} | {a.return_basis}")
                save_figure(fig,folder/(safe_name(a.scope+"_"+a.instrument_id)+".png"))

            if len(self.assets) > 1:
                fig,axes = plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
                dep = self.tables["dependence"]
                matrix = dep.loc[dep["return_type"] == kind].pivot(
                    index="instrument_i",columns="instrument_j",values="correlation")
                im = axes[0].imshow(matrix.to_numpy(dtype=float),vmin=-1,vmax=1,cmap="RdBu_r")
                axes[0].set_xticks(range(len(matrix)),matrix.columns,rotation=30,ha="right")
                axes[0].set_yticks(range(len(matrix)),matrix.index)
                axes[0].set_title(f"Common-interval {kind}-return correlation")
                fig.colorbar(im,ax=axes[0])
                rolling = self.tables["rolling_correlations"]
                for (a,b,window),rows in rolling.loc[rolling["return_type"] == kind].groupby(
                        ["instrument_i","instrument_j","window"]):
                    axes[1].plot(pd.to_datetime(rows["date"]),rows["correlation"],label=f"{a}/{b}: {window}",linewidth=.8)
                axes[1].set(title=f"Rolling {kind}-return correlation",ylim=(-1,1))
                axes[1].legend(fontsize=7)
                save_figure(fig,folder/"dependence.png")
        return files

    def _check_destination(self, destination):
        """Prevent writes into/above the raw archive or over cleaned source files."""
        target,archive = Path(destination).resolve(),self.archive.resolve()
        if target == archive or target.is_relative_to(archive) or archive.is_relative_to(target):
            raise ValueError("Statistics outputs must be outside raw archives and their ancestors.")
        if self.source_folder and (target == self.source_folder or self.source_folder.is_relative_to(target)):
            raise ValueError("Statistics outputs cannot replace the cleaned source folder.")

    def save(self, destination):
        """Export derived tables/report/plots with hashes and completion marker."""
        destination = Path(destination).resolve()
        self._check_destination(destination)
        destination.mkdir(parents=True,exist_ok=True)
        # Invalidate any earlier completion marker before writing outputs.
        # If export fails halfway, readers must not treat the folder as complete.
        manifest = {**self.summary,"complete":False,"files":{}}
        def safe(path):
            if path.is_symlink():
                raise ValueError("Refusing symlink statistics output.")
            return path
        write_json(safe(destination/"manifest.json"),manifest)
        for name,table in self.tables.items():
            path = safe(destination/(name+".csv"))
            table.to_csv(path,index=False,encoding="utf-8",na_rep="")
            manifest["files"][name] = {"file":path.name,"rows":len(table),
                                      "sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
        path = safe(destination/"report.md")
        path.write_text(self.report(),encoding="utf-8")
        manifest["files"]["report"] = {"file":path.name,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
        if self.settings["plots"]:
            for path in self.plots_report(destination/"plots"):
                manifest["files"][path.relative_to(destination).as_posix()] = {"file":path.relative_to(destination).as_posix(),
                                                "sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
        # Mark complete only after every requested table, report and plot exists
        # and has a checksum. Missing analytical data is recorded separately.
        manifest["complete"] = True
        write_json(destination/"manifest.json",manifest)
        return destination


def analyze_processed(processed, modeling_file):
    """Analyze an in-memory cleaning result and return PortfolioStatistics.

    processed supplies selected_prices and portfolio holdings/weights/currency;
    modeling_file supplies the validated statistical settings. No files are
    exported here. run.py calls this after cleaning and then calls result.save().

    AssetStatistics construction below performs the per-series calculations
    automatically; the remaining work aligns holdings and assembles tables.
    """
    cfg = load_modeling_config(modeling_file)
    # Snapshot nested weights/metadata: later edits to the input must not change
    # an existing report or its identity without changing its run_id.
    portfolio_input = deepcopy(processed.summary["portfolio"])
    raw_holdings = portfolio_input["holdings"]
    if (any(not isinstance(h["request_id"],str) or not h["request_id"]
            or type(h["weight"]) not in (int,float) or not math.isfinite(h["weight"])
            or not 0 <= h["weight"] <= 1 for h in raw_holdings)
            or len({h["request_id"] for h in raw_holdings}) != len(raw_holdings)):
        raise ValueError("Holdings need unique IDs and finite nonnegative weights.")
    # Zero-weight entries need no history. Every positive-weight entry remains
    # required for aggregation; unavailable holdings are never redistributed.
    holdings = [h for h in raw_holdings if h["weight"] > 0]
    if not holdings or not math.isclose(sum(h["weight"] for h in holdings),1,abs_tol=1e-8,rel_tol=0):
        raise ValueError("Statistics require complete positive portfolio weights summing to one.")
    # Cleaning already calculated simple_return (and may have excluded some
    # intervals). Select configured holdings only; references/options stay out.
    data = processed.tables["selected_prices"].copy(deep=True)
    assets,notices = {},[]
    for h in holdings:
        key = h["request_id"]
        frame = data.loc[data["instrument_id"] == key].copy()
        if frame.empty:
            notices.append(f"{key}: no selected history; portfolio statistics unavailable.")
            continue
        currencies,bases = set(frame["currency"].dropna()),set(frame["return_basis"].dropna())
        if len(currencies) != 1 or len(bases) != 1:
            notices.append(f"{key}: inconsistent currency or return convention; excluded.")
            continue
        # This constructor is the connection to the calculations: Python calls
        # __post_init__(), which calls _prepare() and the statistics helpers.
        assets[key] = AssetStatistics(key,next(iter(currencies)),next(iter(bases)),frame,cfg)
    # Construct portfolio wealth and run the same AssetStatistics pipeline.
    # A failed currency/basis/coverage guard returns None plus an explicit status.
    portfolio,status = _portfolio_series(assets,holdings,cfg)
    # Each convention gets its own covariance, fits and stress statistics.
    # The validated source rows are shared, so complete intervals match exactly.
    dependencies,rolling_correlations,aligned_tables,stress_dependence = [],[],[],[]
    aligned_counts = {}
    for kind in cfg["return_types"]:
        dependence,rolling_corr,aligned = _dependence(assets,cfg,kind)
        dependencies.append(dependence.assign(return_type=kind))
        rolling_correlations.append(rolling_corr.assign(return_type=kind))
        aligned_tables.append(aligned.assign(return_type=kind))
        aligned_counts[kind] = len(aligned)
        common = _joint_returns(assets,kind).dropna()
        for period in cfg["stress_periods"]:
            sample = common.loc[(common.index.get_level_values("return_start_date") >= period["start"])
                                & (common.index.get_level_values("date") <= period["end"])]
            corr = sample.corr() if len(sample) >= 2 else pd.DataFrame(np.nan,index=list(assets),columns=list(assets))
            for a in assets:
                for b in assets:
                    stress_dependence.append(dict(return_type=kind,name=period["name"],start=period["start"],end=period["end"],
                        instrument_i=a,instrument_j=b,observations=len(sample),
                        correlation=_finite(corr.loc[a,b]),
                        status="available" if len(sample) >= cfg["minimum_observations"] else "insufficient_history"))
    reporting = portfolio_input["reporting_currency"]
    currency = portfolio.currency if portfolio else None
    if currency and currency != reporting:
        notices.append(f"Portfolio results are in {currency}, not {reporting}; FX has not been modeled or converted.")
    if status != "available":
        notices.append("Portfolio wealth is unavailable: "+status+". Individual local-currency statistics remain usable.")
    if len(assets) != len(holdings):
        notices.append("Dependence covers available holdings only, not the full portfolio.")
    # Flatten computed objects into export tables: path rows appear once, while
    # return-dependent rows carry return_type and appear for each selected type.
    series = list(assets.values())+([portfolio] if portfolio else [])
    metrics,stresses,daily,rolling,fits,paths,stress_paths = [],[],[],[],[],[],[]
    for a in series:
        tags = dict(scope=a.scope,instrument_id=a.instrument_id,currency=a.currency,return_basis=a.return_basis)
        paths.append({**tags,**a.path_metrics})
        coverage_by_name = {}
        dates = a.frame.loc[a.frame["analysis_price"].notna(),"date"]
        for row in a.stress_paths:
            # History must bracket the calendar window, even when its bounds
            # fall on holidays/weekends. Check expected rows inside it as well:
            # bracketing observations alone can hide missing session prices.
            window_prices = a.frame.loc[a.frame["date"].between(row["start"],row["end"]),"analysis_price"]
            coverage = "unavailable" if row["first_price_date"] is None else (
                "full_window" if not dates.empty and dates.min() <= row["start"] and dates.max() >= row["end"]
                and window_prices.notna().all() else "partial_window")
            coverage_by_name[row["name"]] = coverage
            stress_paths.append({**tags,**row,"coverage":coverage})
        for kind,analysis in a.return_statistics.items():
            metrics.append({**tags,**analysis["metrics"]})
            fits.append({**tags,**analysis["distribution"]})
            if analysis["metrics"]["status"] != "available":
                notices.append(f"{a.scope} {a.instrument_id} ({kind}): {analysis['metrics']['status']}.")
            if analysis["distribution"].get("student_t_variance_exists") is False:
                notices.append(f"{a.scope} {a.instrument_id} ({kind}): fitted Student-t has no finite variance; descriptive fit only.")
            for row in analysis["stress"]:
                stresses.append({**tags,**row,"coverage":coverage_by_name[row["name"]]})
            rolling.append(analysis["rolling"].assign(**tags,return_type=kind))
        # Prices, both return columns and wealth stay together once per date.
        # The portfolio's log column was derived by _prepare from its own wealth.
        values = a.frame[["date","return_start_date","analysis_price","simple_return","log_return"]].copy()
        observed = values["analysis_price"].dropna()
        values["normalized_value"] = values["analysis_price"]/observed.iloc[0]*cfg["base_value"] if len(observed) else np.nan
        values["drawdown"] = (values["normalized_value"]/values["normalized_value"].cummax()-1
                              if a.path_metrics["path_status"] == "available" else np.nan)
        daily.append(values.assign(**tags))
    empty_daily = ["scope","instrument_id","date","return_start_date","analysis_price","simple_return","log_return",
                   "normalized_value","drawdown","currency","return_basis"]
    tables = dict(
        statistics=pd.DataFrame(metrics,columns=list(metrics[0]) if metrics else ["scope","instrument_id","return_type","status"]),
        path_statistics=pd.DataFrame(paths,columns=list(paths[0]) if paths else ["scope","instrument_id","path_status"]),
        stress_statistics=pd.DataFrame(stresses,columns=list(stresses[0]) if stresses else ["scope","instrument_id","return_type","name","status"]),
        stress_path_statistics=pd.DataFrame(stress_paths,columns=list(stress_paths[0]) if stress_paths else ["scope","instrument_id","name","path_status"]),
        # Fits may differ in outcome: retain diagnostics/flags introduced by
        # later rows, rather than restricting columns to the first fit.
        distributions=pd.DataFrame(fits) if fits else pd.DataFrame(columns=["scope","instrument_id","return_type","student_t_status"]),
        daily_statistics=pd.concat(daily,ignore_index=True) if daily else pd.DataFrame(columns=empty_daily),
        rolling_volatility=pd.concat(rolling,ignore_index=True) if rolling else pd.DataFrame(columns=["date","instrument_id","return_type"]),
        dependence=pd.concat(dependencies,ignore_index=True),
        rolling_correlations=pd.concat(rolling_correlations,ignore_index=True),
        aligned_returns=pd.concat(aligned_tables,ignore_index=True),
        stress_dependence=pd.DataFrame(stress_dependence,columns=["return_type","name","start","end","instrument_i",
                                                                 "instrument_j","observations","correlation","status"]))
    # Include actual selected bytes/weights/settings and versions in reproducibility identity.
    # This hash is an audit identity, not a random ID: changing input data,
    # settings, source code (including comments), or dependencies creates a run.
    identity = dict(processing_id=processed.run_id,settings=cfg,portfolio=portfolio_input,
                    selected_prices_sha256=hashlib.sha256(data.to_csv(index=False).encode()).hexdigest(),
                    statistics_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    dependencies={name:version(name) for name in ("numpy","pandas","scipy","matplotlib")})
    run_id = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
    summary = dict(schema_version=2,run_id=run_id,cob=processed.cob,identity=identity,
                   portfolio_status=status,analysis_currency=currency,reporting_currency=reporting,
                   reporting_currency_returns_computed=bool(portfolio and currency == reporting),
                   aligned_observations=next(iter(aligned_counts.values())),
                   aligned_observations_by_return_type=aligned_counts,return_types=cfg["return_types"],warnings=notices,
                   simulation_performed=False,calibration_performed=False)
    return PortfolioStatistics(Path(processed.archive),processed.run_id,processed.cob,run_id,
                               cfg,assets,portfolio,summary,tables)


def analyze_saved_processing(folder, modeling_file):
    """Verify saved cleaning output, then use the in-memory analysis entry point.

    The manifest identifies selected_prices, its checksum, and its row count.
    Verify the exact bytes that will be parsed before trusting the table.
    source_folder subsequently protects the cleaning output from replacement.
    """
    from .data_cleaning import ProcessedArchive
    from io import BytesIO
    folder = Path(folder).resolve()
    manifest = _read_json(folder/"manifest.json")
    if manifest.get("complete") is not True:
        raise ValueError("Cleaning output is incomplete.")
    entry = manifest["files"]["selected_prices"]
    file = (folder/entry["file"]).resolve()
    if not file.is_relative_to(folder) or file == folder:
        raise ValueError("Selected-price file escapes its source folder.")
    raw = file.read_bytes()
    if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
        raise ValueError("Selected-price checksum mismatch.")
    # Cleaning writes missing cells as empty strings. Restrict NA parsing to
    # those cells so valid string IDs such as "NA", "NULL" and "001" survive.
    data = pd.read_csv(BytesIO(raw),dtype={"instrument_id":str,"date":str,"return_start_date":str},
                       keep_default_na=False,na_values=[""])
    if len(data) != entry["rows"]:
        raise ValueError("Selected-price row count mismatch.")
    result = analyze_processed(ProcessedArchive(Path(manifest["archive"]),manifest["cob"],manifest["run_id"],
                                               {"selected_prices":data},manifest),modeling_file)
    result.source_folder = folder
    return result
