"""The Portfolio object and automatic forecast dates, with offline fixture data."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import math
import unittest

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from src import Portfolio
from src.cob import forecast_start_for_cob
from src.data_cleaning import ProcessedArchive
from src.monte_carlo import _sessions, forecast_end_for_horizon, simulate_processed

ROOT = Path(__file__).resolve().parents[1]


class PortfolioModelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.addCleanup(plt.close, "all")
        self.root = Path(self.tmp.name); self.raw = self.root/"raw"; self.raw.mkdir()
        self.cfg = json.loads((ROOT/"config/monte_carlo.json").read_text(encoding="utf-8-sig"))
        self.cfg["stress_calibrations"] = []
        self.cfg["diagnostics"]["enabled"] = False
        self.cfg.update(paths=2000, plots=False, sample_paths=5, forecast_end="2026-10-30")
        self.cfg["calibration"].update(start="2026-06-01", minimum_observations=10)
        self.cfg["student_t"]["sensitivity_tail_probabilities"] = []
        self.mc_file = self.root/"mc.json"; self.write_config()
        modeling = json.loads((ROOT/"config/modeling.json").read_text(encoding="utf-8"))
        modeling["statistics"].update(minimum_observations=2, rolling_windows=[5], fit_student_t=False,
                                      plots=False, stress_periods=[])
        self.modeling_file = self.root/"modeling.json"
        self.modeling_file.write_text(json.dumps(modeling), encoding="utf-8")
        self.dates = _sessions("XNYS", "2026-06-01", "2026-10-08")

    def write_config(self):
        self.mc_file.write_text(json.dumps(self.cfg), encoding="utf-8")

    def processed(self, cob="2026-10-08"):
        rng = np.random.default_rng(321)
        logs = rng.multivariate_normal([.0005, .0001], [[.0001, -.00001], [-.00001, .00002]], len(self.dates)-1)
        frames = []
        for key, values in (("A", logs[:, 0]), ("B", logs[:, 1])):
            frames.append(pd.DataFrame(dict(
                instrument_id=key, date=self.dates, calendar="XNYS", currency="USD",
                return_basis="split_and_dividend_adjusted_close",
                analysis_price=100*np.exp(np.r_[0, np.cumsum(values)]), simple_return=np.r_[np.nan, np.expm1(values)])))
        return ProcessedArchive(self.raw, cob, "fixture", {"selected_prices": pd.concat(frames, ignore_index=True)},
                                {"portfolio": {"reporting_currency": "USD",
                                               "holdings": [{"request_id": "A", "weight": .5},
                                                            {"request_id": "B", "weight": .5}]}})

    def portfolio(self, cob="2026-10-08"):
        return Portfolio(self.processed(cob), self.modeling_file, self.mc_file)

    def parameters(self, result):
        return result.tables["calibration_parameters"].set_index("factor")

    def test_forecast_start_is_the_day_after_any_cob(self):
        self.assertEqual(self.cfg["forecast_start"], "cob_plus_one")
        self.assertEqual(forecast_start_for_cob("2026-10-08"), "2026-10-09")
        result = simulate_processed(self.processed(), self.mc_file)
        self.assertEqual(result.summary["status"], "complete")
        self.assertEqual(result.summary["forecast_dates"][0], "2026-10-09")
        self.assertEqual(result.settings["forecast_start"], "2026-10-09")
        self.assertIn("2026-10-09", result.report())

    def test_weekend_start_stays_initial_level_and_first_return_ends_next_session(self):
        # 8 August 2026 is a Saturday, so the archive COB of Friday gives a Saturday start.
        self.dates = _sessions("XNYS", "2026-03-02", "2026-08-07")
        result = simulate_processed(self.processed("2026-08-07"), self.mc_file)
        self.assertEqual(result.summary["status"], "complete")
        self.assertEqual(result.summary["forecast_dates"][:2], ["2026-08-08", "2026-08-10"])

    def test_cob_after_forecast_end_is_unavailable_with_reason(self):
        self.cfg["forecast_end"] = "2026-10-08"; self.write_config()
        result = simulate_processed(self.processed(), self.mc_file)
        self.assertEqual(result.summary["status"], "unavailable")
        self.assertIn("Forecast end must follow", result.summary["warnings"][0])

    def test_creating_a_portfolio_computes_statistics_only(self):
        portfolio = self.portfolio()
        self.assertEqual(portfolio.weights, {"A": .5, "B": .5})
        self.assertIsNotNone(portfolio.statistics.assets["A"].metrics["std_daily"])
        self.assertIsNone(portfolio.simulation)

    def test_monte_carlo_parameters_replace_configuration_for_this_run(self):
        portfolio = self.portfolio()
        result = portfolio.monte_carlo(paths=500, horizon_sessions=10, seed=3, models=["gaussian"])
        self.assertIs(portfolio.simulation, result)
        self.assertEqual(result.settings["paths"], 500)
        self.assertEqual(result.settings["seed"], 3)
        self.assertEqual(result.summary["steps"], 10)
        self.assertEqual(list(result.summary["models"]), ["gaussian"])
        self.assertEqual(json.loads(self.mc_file.read_text(encoding="utf-8"))["paths"], 2000)
        repeated = self.portfolio().monte_carlo(paths=500, horizon_sessions=10, seed=3, models=["gaussian"])
        self.assertTrue(result.tables["terminal_values"].equals(repeated.tables["terminal_values"]))

    def test_horizon_counts_returns_from_a_weekend_start(self):
        cfg = dict(self.cfg)
        self.assertEqual(forecast_end_for_horizon("2026-08-07", cfg, 1), "2026-08-10")
        with self.assertRaises(ValueError):
            forecast_end_for_horizon("2026-08-07", cfg, 0)

    def test_invalid_parameters_are_rejected(self):
        portfolio = self.portfolio()
        for options in (dict(paths=1), dict(models=["unknown"]), dict(instrument="Z"),
                        dict(horizon_sessions=5, forecast_end="2026-10-30"),
                        dict(assumptions={"Z": {"annual_return": .05}}),
                        dict(assumptions={"A": {"annual_volatility": 0}}),
                        dict(assumptions={"A": {"annual_return": -1}}),
                        dict(assumptions={"A": {"sharpe": 1}}), dict(volatility_multiplier=0)):
            with self.subTest(options), self.assertRaises(ValueError):
                portfolio.monte_carlo(**options)

    def test_single_holding_is_simulated_without_replacing_portfolio_result(self):
        portfolio = self.portfolio()
        whole = portfolio.monte_carlo(paths=500, models=["gaussian"])
        single = portfolio.monte_carlo(instrument="A", paths=500, models=["gaussian"])
        self.assertIs(portfolio.simulation, whole)
        self.assertEqual([h["instrument_id"] for h in single.summary["holdings"]], ["A"])
        self.assertEqual(single.summary["holdings"][0]["weight"], 1)
        self.assertEqual(portfolio.weights, {"A": .5, "B": .5})
        risk = single.tables["risk_summary"].set_index("scope")
        self.assertAlmostEqual(risk.loc["holding", "mean_terminal"], risk.loc["portfolio", "mean_terminal"])

    def test_assumptions_set_mean_and_volatility_and_keep_correlation(self):
        portfolio = self.portfolio()
        base = portfolio.monte_carlo(paths=500, models=["gaussian"])
        views = portfolio.monte_carlo(paths=500, models=["gaussian"],
                                      assumptions={"A": {"annual_return": .10, "annual_volatility": .30}})
        row = self.parameters(views).loc["asset::A"]
        self.assertAlmostEqual(row["mean_daily_log"], math.log1p(.10)/252)
        self.assertAlmostEqual(row["std_daily_log"], .30/math.sqrt(252))
        self.assertEqual(row["drift_source"], "user_assumptions")
        self.assertEqual(self.parameters(views).loc["asset::B", "drift_source"], "same_window")
        unchanged = self.parameters(base).loc["asset::B"]
        self.assertAlmostEqual(self.parameters(views).loc["asset::B", "std_daily_log"], unchanged["std_daily_log"])
        correlation = lambda r: r.tables["calibration_covariance"].set_index(["factor_i", "factor_j"]).loc[("asset::A", "asset::B"), "correlation"]
        self.assertAlmostEqual(correlation(views), correlation(base))
        self.assertTrue(any("assumptions" in w for w in views.summary["warnings"]))

    def test_volatility_multiplier_scales_holdings_and_widens_outcomes(self):
        portfolio = self.portfolio()
        base = portfolio.monte_carlo(paths=2000, models=["gaussian"])
        wider = portfolio.monte_carlo(paths=2000, models=["gaussian"], volatility_multiplier=2)
        for factor in ("asset::A", "asset::B"):
            self.assertAlmostEqual(self.parameters(wider).loc[factor, "std_daily_log"],
                                   2*self.parameters(base).loc[factor, "std_daily_log"])
        spread = lambda r: r.tables["terminal_values"].query("scope == 'portfolio'")["terminal_value"].std()
        self.assertGreater(spread(wider), spread(base))

    def test_assumptions_disable_stress_windows(self):
        self.cfg["stress_calibrations"] = [dict(name="window", start="2026-07-01", end="2026-09-30", drift="stress")]
        self.write_config()
        portfolio = self.portfolio()
        self.assertIn("window__gaussian", portfolio.monte_carlo(paths=200, models=["gaussian"]).summary["models"])
        result = portfolio.monte_carlo(paths=200, models=["gaussian"], volatility_multiplier=1.5)
        self.assertEqual(list(result.summary["models"]), ["gaussian"])

    def test_report_sections_and_limits_are_parameters(self):
        portfolio = self.portfolio()
        before = portfolio.report()
        self.assertIn("## Performance", before)
        self.assertIn("Not run", before)
        portfolio.monte_carlo(paths=500, models=["gaussian"])
        only = portfolio.report(sections=("simulation",), confidence=.95)
        self.assertIn("## Monte Carlo simulation", only)
        self.assertIn("95", only)
        self.assertNotIn("## Performance", only)
        strict = portfolio.report(sections=("risk_warnings",), volatility_limit=.0001, loss_probability_limit=.0001)
        self.assertIn("annual volatility", strict)
        self.assertIn("probability of loss", strict)
        relaxed = portfolio.report(sections=("risk_warnings",), volatility_limit=10, drawdown_limit=10,
                                   loss_probability_limit=10, correlation_limit=10)
        self.assertIn("No configured limit was exceeded", relaxed)
        self.assertIn("do not predict", relaxed)

    def test_invalid_report_arguments_are_rejected(self):
        portfolio = self.portfolio()
        for options in (dict(sections=("unknown",)), dict(return_type="other"), dict(volatility_limit=0),
                        dict(confidence=.95), dict(model="gaussian")):
            with self.subTest(options), self.assertRaises(ValueError):
                portfolio.report(**options)
        portfolio.monte_carlo(paths=200, models=["gaussian"])
        for options in (dict(confidence=.5), dict(model="unknown")):
            with self.subTest(options), self.assertRaises(ValueError):
                portfolio.report(**options)

    def test_plot_draws_requested_trajectories_and_saves(self):
        portfolio = self.portfolio()
        with self.assertRaises(ValueError):
            portfolio.plot_monte_carlo(show=False)
        portfolio.monte_carlo(paths=500, models=["gaussian"])
        path = self.root/"out"/"simulation.png"
        fig = portfolio.plot_monte_carlo(trajectories=3, destination=path, show=False)
        self.assertTrue(path.is_file())
        self.assertEqual(len(fig.axes[0].lines), 3+1)
        holding = portfolio.simulation.plot(instrument_id="A", trajectories=0, show=False)
        self.assertIn("A | gaussian", holding._suptitle.get_text())
        for options in (dict(instrument_id="Z"), dict(model="unknown"), dict(trajectories=-1)):
            with self.subTest(options), self.assertRaises(ValueError):
                portfolio.simulation.plot(show=False, **options)

    def test_plot_refuses_to_write_inside_raw_archive(self):
        portfolio = self.portfolio()
        portfolio.monte_carlo(paths=200, models=["gaussian"])
        with self.assertRaises(ValueError):
            portfolio.plot_monte_carlo(destination=self.raw/"simulation.png", show=False)

    def test_plots_report_adds_simulation_chart_once_simulated(self):
        portfolio = self.portfolio()
        without = {p.name for p in portfolio.plots_report(self.root/"first")}
        self.assertNotIn("monte_carlo.png", without)
        self.assertIn("wealth.png", without)
        portfolio.monte_carlo(paths=200, models=["gaussian"])
        files = portfolio.plots_report(self.root/"second")
        self.assertIn("monte_carlo.png", {p.name for p in files})
        self.assertTrue(all(p.is_file() for p in files))

    def test_from_saved_verifies_and_uses_same_cleaned_tables(self):
        processed = self.processed()
        folder = self.root/"clean"; folder.mkdir()
        import hashlib
        path = folder/"selected_prices.csv"
        processed.tables["selected_prices"].to_csv(path, index=False)
        files = dict(selected_prices=dict(file=path.name, rows=len(processed.tables["selected_prices"]),
                                          sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        manifest = dict(complete=True, archive=str(self.raw), cob=processed.cob, run_id="fixture", files=files,
                        portfolio=processed.summary["portfolio"])
        (folder/"manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        portfolio = Portfolio.from_saved(folder, self.modeling_file, self.mc_file)
        self.assertEqual(portfolio.statistics.source_folder, folder.resolve())
        result = portfolio.monte_carlo(paths=200, models=["gaussian"])
        self.assertEqual(result.summary["status"], "complete")
        self.assertEqual(result.source_folder, folder.resolve())
        path.write_text("corrupted", encoding="utf-8")
        with self.assertRaises(ValueError):
            Portfolio.from_saved(folder, self.modeling_file, self.mc_file)


if __name__ == "__main__":
    unittest.main()
