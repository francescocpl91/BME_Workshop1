"""One investment portfolio: statistics, simulation, report and plots in one object.

Creating a Portfolio computes its historical statistics automatically. The heavy
calculations stay in statistics.py and monte_carlo.py; this class holds them
together so callers work with a portfolio rather than with separate functions.
"""
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path

from .data_cleaning import ProcessedArchive
from .mc_diagnostics import confidence_label, probability_label
from .monte_carlo import (MonteCarloResult, forecast_end_for_horizon, load_monte_carlo_config,
                          load_saved_processing, simulate_processed)
from .statistics import PortfolioStatistics, _finite, analyze_processed

REPORT_SECTIONS = ("overview", "performance", "correlations", "simulation", "diagnostics", "risk_warnings")


def _number(value):
    value = _finite(value)
    return "N/A" if value is None else f"{value:.3f}"


def _percent(value):
    value = _finite(value)
    return "N/A" if value is None else f"{value:.2%}"


@dataclass
class Portfolio:
    """Holdings and weights from a cleaned run, with their analysis as methods."""
    processed: ProcessedArchive
    modeling_file: Path
    monte_carlo_file: Path
    statistics: PortfolioStatistics = field(init=False, repr=False)
    simulation: MonteCarloResult | None = field(init=False, default=None, repr=False)

    def __post_init__(self):
        self.statistics = analyze_processed(self.processed, self.modeling_file)

    @classmethod
    def from_saved(cls, folder, modeling_file, monte_carlo_file):
        """Build from a saved cleaned run after verifying its checksums."""
        portfolio = cls(load_saved_processing(folder), modeling_file, monte_carlo_file)
        portfolio.statistics.source_folder = Path(folder).resolve()
        return portfolio

    @property
    def holdings(self):
        """Holdings with positive weight; zero-weight entries are not invested."""
        return [dict(h) for h in self.processed.summary["portfolio"]["holdings"] if h["weight"] > 0]

    @property
    def weights(self):
        return {h["request_id"]: h["weight"] for h in self.holdings}

    def monte_carlo(self, *, paths=None, horizon_sessions=None, forecast_end=None, seed=None, models=None,
                    sample_paths=None, instrument=None, assumptions=None, volatility_multiplier=1):
        """Simulate the whole portfolio or one holding (instrument) and return the result.

        Omitted values come from the JSON configuration. horizon_sessions counts
        simulated exchange-session returns, as an alternative to forecast_end.
        assumptions maps a holding to annual_return and/or annual_volatility;
        volatility_multiplier scales holdings without an explicit volatility.
        Only whole-portfolio runs become self.simulation, used by report and plots.
        """
        if horizon_sessions is not None and forecast_end is not None:
            raise ValueError("Use horizon_sessions or forecast_end, not both.")
        overrides = {key: value for key, value in dict(paths=paths, forecast_end=forecast_end, seed=seed,
                                                       models=models, sample_paths=sample_paths).items()
                     if value is not None}
        # A small run may request fewer paths than the configured trajectory export.
        config = load_monte_carlo_config(self.monte_carlo_file)
        if paths is not None and sample_paths is None:
            overrides["sample_paths"] = min(config["sample_paths"], paths)
        if horizon_sessions is not None:
            overrides["forecast_end"] = forecast_end_for_horizon(self.processed.cob, config, horizon_sessions)
        processed = self.processed if instrument is None else self._single_holding(instrument)
        views = None
        if assumptions is not None or volatility_multiplier != 1:
            # The periods convention matches the annualized statistics of this portfolio.
            views = dict(holdings=deepcopy(assumptions or {}), volatility_multiplier=volatility_multiplier,
                         periods_per_year=self.statistics.settings["periods_per_year"])
        result = simulate_processed(processed, self.monte_carlo_file, overrides, views)
        result.source_folder = self.statistics.source_folder
        if instrument is None:
            self.simulation = result
        return result

    def _single_holding(self, instrument):
        """Same cleaned data, invested entirely in one existing holding."""
        chosen = [h for h in self.holdings if h["request_id"] == instrument]
        if not chosen:
            raise ValueError("Instrument is not a holding of this portfolio: "+str(instrument))
        summary = deepcopy(self.processed.summary)
        summary["portfolio"]["holdings"] = [dict(chosen[0], weight=1)]
        return replace(self.processed, summary=summary)

    def plot_monte_carlo(self, **options):
        """Show trajectories and distributions of the latest whole-portfolio simulation."""
        if self.simulation is None:
            raise ValueError("Run monte_carlo() first.")
        return self.simulation.plot(**options)

    def report(self, sections=REPORT_SECTIONS, *, return_type="simple", model=None, confidence=None,
               volatility_limit=.20, drawdown_limit=.20, loss_probability_limit=.25, correlation_limit=.80):
        """Return Markdown; sections choose the content and the limits trigger risk warnings.

        Limits are positive fractions (.20 means 20%). A drawdown limit compares
        the size of the fall. confidence defaults to the highest configured level.
        """
        unknown = set(sections)-set(REPORT_SECTIONS)
        if unknown:
            raise ValueError("Unknown report sections: "+", ".join(sorted(unknown)))
        if return_type not in self.statistics.settings["return_types"]:
            raise ValueError("Return type was not computed: "+str(return_type))
        for limit in (volatility_limit, drawdown_limit, loss_probability_limit, correlation_limit):
            if type(limit) not in (int, float) or limit <= 0:
                raise ValueError("Report limits must be positive numbers.")
        simulation = self._simulation_rows(model, confidence)
        writers = dict(overview=self._overview, performance=lambda: self._performance(return_type),
                       correlations=lambda: self._correlations(return_type),
                       simulation=lambda: self._simulation_section(*simulation),
                       diagnostics=self._diagnostics,
                       risk_warnings=lambda: self._risk_warnings(return_type, simulation[0], volatility_limit,
                                                                 drawdown_limit, loss_probability_limit,
                                                                 correlation_limit))
        lines = ["# Portfolio report", ""]
        for section in sections:
            lines += writers[section]()+[""]
        return "\n".join(lines).rstrip()+"\n"

    def _simulation_rows(self, model, confidence):
        """Baseline portfolio rows; argument errors surface even if not displayed."""
        if self.simulation is None or self.simulation.summary["status"] != "complete":
            if model is not None or confidence is not None:
                raise ValueError("model and confidence require a completed monte_carlo() run.")
            return [], None
        levels = self.simulation.settings["confidence_levels"]
        confidence = max(levels) if confidence is None else confidence
        if confidence not in levels:
            raise ValueError("Confidence level was not simulated: "+str(confidence))
        risk = self.simulation.tables["risk_summary"]
        rows = risk.loc[(risk["scope"] == "portfolio") & (risk["calibration_set"] == "baseline")]
        if model is not None:
            if model not in set(rows["model"]):
                raise ValueError("Unknown baseline model: "+str(model))
            rows = rows.loc[rows["model"] == model]
        return rows.to_dict("records"), confidence

    def _overview(self):
        summary = self.statistics.summary
        allocation = ", ".join(f"{key} {weight:.1%}" for key, weight in self.weights.items())
        return ["## Overview", "",
                f"- Archive COB: {self.processed.cob}",
                f"- Holdings: {allocation}",
                f"- Reporting currency requested: {summary['reporting_currency']}",
                f"- Historical statistics currency: {summary['analysis_currency'] or 'unavailable'}",
                "- Method: buy-and-hold initial weights; no periodic rebalancing."]

    def _performance(self, return_type):
        lines = [f"## Performance ({return_type} returns)", "",
                 "| Series | First price | Last price | Cumulative return | CAGR | Max drawdown | Annual volatility |",
                 "|---|---|---|---:|---:|---:|---:|"]
        tables = self.statistics.tables
        moments = tables["statistics"].loc[tables["statistics"]["return_type"] == return_type]
        volatility = {(row["scope"], row["instrument_id"]): row["volatility_annualized"]
                      for row in moments.to_dict("records")}
        for row in tables["path_statistics"].to_dict("records"):
            key = (row["scope"], row["instrument_id"])
            lines.append(f"| {row['instrument_id']} | {row.get('first_price_date')} | {row.get('last_price_date')} | "
                         f"{_percent(row.get('cumulative_return'))} | {_percent(row.get('cagr'))} | "
                         f"{_percent(row.get('max_drawdown'))} | {_percent(volatility.get(key))} |")
        return lines

    def _pairs(self, return_type):
        """Each holding pair once, from the common-interval dependence table."""
        table = self.statistics.tables["dependence"]
        rows, seen = [], set()
        for row in table.loc[table["return_type"] == return_type].to_dict("records"):
            pair = frozenset((row["instrument_i"], row["instrument_j"]))
            if row["instrument_i"] != row["instrument_j"] and pair not in seen:
                seen.add(pair)
                rows.append(row)
        return rows

    def _correlations(self, return_type):
        lines = [f"## Correlations ({return_type} returns)", ""]
        pairs = self._pairs(return_type)
        if not pairs:
            return lines+["Only one holding: no pair to compare."]
        lines += ["| Holding A | Holding B | Correlation | Common intervals |", "|---|---|---:|---:|"]
        lines += [f"| {row['instrument_i']} | {row['instrument_j']} | {_number(row['correlation'])} | "
                  f"{row['observations']} |" for row in pairs]
        return lines

    def _simulation_section(self, rows, confidence):
        lines = ["## Monte Carlo simulation", ""]
        if self.simulation is None:
            return lines+["Not run. Call monte_carlo() to include this section."]
        summary = self.simulation.summary
        if summary["status"] != "complete":
            return lines+[f"Status: {summary['status']}."]+["- "+warning for warning in summary["warnings"]]
        label, key = confidence_label(confidence), probability_label(confidence)
        settings = self.simulation.settings
        lines += [f"{settings['paths']:,} paths from {summary['forecast_dates'][0]} to {summary['forecast_dates'][-1]} "
                  f"({summary['steps']} sessions), in {summary['simulation_currency']}. "
                  f"Initial value {settings['initial_value']}. VaR and ES cover the full horizon.", "",
                  f"| Scenario | Mean | Median | Probability of loss | {label} VaR | {label} ES |",
                  "|---|---:|---:|---:|---:|---:|"]
        lines += [f"| {row['model']} | {_number(row['mean_terminal'])} | {_number(row['median_terminal'])} | "
                  f"{_percent(row['probability_loss'])} | {_percent(row[f'loss_var_{key}'])} | "
                  f"{_percent(row[f'loss_es_{key}'])} |" for row in rows]
        return lines

    def _diagnostics(self):
        summary = self.statistics.summary
        lines = ["## Diagnostics", "",
                 f"- Portfolio statistics status: {summary['portfolio_status']}",
                 f"- Common return intervals across holdings: {summary['aligned_observations']:,}",
                 f"- Simulation status: {self.simulation.summary['status'] if self.simulation else 'not run'}"]
        notices = summary["warnings"]+(self.simulation.summary["warnings"] if self.simulation else [])
        return lines+["- "+notice for notice in notices]

    def _risk_warnings(self, return_type, simulation_rows, volatility_limit, drawdown_limit,
                       loss_probability_limit, correlation_limit):
        tables = self.statistics.tables
        alerts = []
        moments = tables["statistics"]
        for row in moments.loc[moments["return_type"] == return_type].to_dict("records"):
            volatility = _finite(row["volatility_annualized"])
            if volatility is not None and volatility > volatility_limit:
                alerts.append(f"{row['instrument_id']}: annual volatility {volatility:.2%} exceeds {volatility_limit:.2%}.")
        for row in tables["path_statistics"].to_dict("records"):
            drawdown = _finite(row.get("max_drawdown"))
            if drawdown is not None and -drawdown > drawdown_limit:
                alerts.append(f"{row['instrument_id']}: historical drawdown {drawdown:.2%} exceeds {drawdown_limit:.2%}.")
        for row in self._pairs(return_type):
            correlation = _finite(row["correlation"])
            if correlation is not None and correlation > correlation_limit:
                alerts.append(f"{row['instrument_i']}/{row['instrument_j']}: correlation {correlation:.3f} "
                              f"exceeds {correlation_limit:.3f}; diversification may be limited.")
        for row in simulation_rows:
            probability = _finite(row["probability_loss"])
            if probability is not None and probability > loss_probability_limit:
                alerts.append(f"{row['model']}: simulated probability of loss {probability:.2%} "
                              f"exceeds {loss_probability_limit:.2%}.")
        lines = ["## Risk warnings", ""]
        lines += ["- "+alert for alert in alerts] or ["- No configured limit was exceeded."]
        return lines+["- Results rest on historical data and model assumptions; they do not predict future returns."]

    def plots_report(self, destination, include_simulation=True):
        """Write the historical charts and, when available, the simulation chart as PNG."""
        files = self.statistics.plots_report(destination)
        if include_simulation and self.simulation is not None and self.simulation.summary["status"] == "complete":
            from matplotlib import pyplot as plt
            path = Path(destination)/"monte_carlo.png"
            plt.close(self.simulation.plot(destination=path, show=False))
            files.append(path)
        return files
