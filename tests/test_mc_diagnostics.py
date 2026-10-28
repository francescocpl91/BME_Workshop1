"""Exact diagnostic arithmetic, no changed paths, and isolated stress windows."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import hashlib
import json
import socket
import unittest
import matplotlib
# Offline figure tests own their backend; do not depend on test order or Tk.
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from src.data_cleaning import ProcessedArchive
from src.monte_carlo import simulate_processed, load_monte_carlo_config, _sessions
from src.mc_diagnostics import daily_distribution, wealth_diagnostics

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "config"


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg = json.loads((FIXTURES/"monte_carlo.json").read_text(encoding="utf-8"))
        self.cfg.update(paths=200,plots=False,sample_paths=0,forecast_end="2026-11-10")
        self.cfg["calibration"].update(start="2026-07-01",minimum_observations=20)
        self.cfg["student_t"]["sensitivity_tail_probabilities"] = []
        self.cfg["stress_calibrations"] = []
        self.file = self.root/"config.json"
        dates = _sessions("XNYS","2026-01-02","2026-10-08")
        rng = np.random.default_rng(420)
        # Different means AND covariance before July, exposing accidental use
        # of baseline mean to center stress data or inclusion of later rows.
        early = np.array(dates[1:]) < "2026-07-01"
        logs = rng.multivariate_normal([.0005,.0001],[[.0001,.00001],[.00001,.00002]],len(dates)-1)
        logs[early] = logs[early]*3 + [-.004,.002]
        tables = []
        for i,key in enumerate(("A","B")):
            price = 100*np.exp(np.r_[0,logs[:,i].cumsum()])
            tables.append(pd.DataFrame(dict(instrument_id=key,date=dates,calendar="XNYS",currency="USD",
                                           return_basis="split_and_dividend_adjusted_close",analysis_price=price,
                                           simple_return=np.r_[np.nan,np.expm1(logs[:,i])])))
        self.p = ProcessedArchive(self.root/"raw","2026-10-08","fixture",{"selected_prices":pd.concat(tables)},
                                  {"portfolio":{"reporting_currency":"USD","holdings":[{"request_id":"A","weight":.5},{"request_id":"B","weight":.5}]}})

    def run_model(self):
        self.file.write_text(json.dumps(self.cfg),encoding="utf-8")
        with patch.object(socket.socket,"connect",side_effect=AssertionError("Offline only")):
            return simulate_processed(self.p,self.file)

    def stress(self,drift="baseline"):
        self.cfg["stress_calibrations"] = [dict(name="test_crisis",start="2026-01-02",end="2026-06-30",drift=drift)]

    def test_daily_moments_histogram_overflow_and_zero_variance(self):
        x = np.array([-12.,-4.,-1.,0.,1.,4.,12.])
        info,hist = daily_distribution(x,0.,1.,[3,4],100)
        self.assertAlmostEqual(info["skewness"],skew(x,bias=True))
        self.assertAlmostEqual(info["excess_kurtosis"],kurtosis(x,bias=True))
        self.assertAlmostEqual(info["probability_abs_z_gt_3"],4/7)
        self.assertAlmostEqual(info["probability_abs_z_gt_4"],2/7)
        self.assertEqual(hist["count"].sum()+info["histogram_underflow"]+info["histogram_overflow"],len(x))
        self.assertAlmostEqual(float((hist["density"]*(hist["right_z"]-hist["left_z"])).sum()),5/7)
        constant,hist = daily_distribution(np.ones(10),1.,0.,[3,4])
        self.assertIsNone(constant["excess_kurtosis"])
        self.assertIsNone(constant["probability_abs_z_gt_3"])
        self.assertFalse(constant["standardization_available"])
        self.assertTrue(hist.empty)

    def test_short_horizons_and_path_risk_use_wealth_not_weighted_logs(self):
        wealth = np.array([[100,80,110],[100,120,90]],dtype=float)
        rows,path,worst = wealth_diagnostics(wealth,100,[1,2,5],[.75])
        self.assertEqual(rows[0]["probability_loss"],.5)
        self.assertAlmostEqual(rows[0]["loss_var_0.75"],.1)
        self.assertAlmostEqual(rows[0]["loss_es_0.75"],.2)
        self.assertAlmostEqual(rows[1]["mean_return"],0)
        self.assertEqual(rows[2]["status"],"unavailable_horizon")
        np.testing.assert_allclose(worst,[-.2,-.25])
        self.assertAlmostEqual(path["mean_max_drawdown"],-.225)

    def test_diagnostics_do_not_change_paths_or_random_streams(self):
        self.cfg["diagnostics"]["enabled"] = False
        before = self.run_model()
        self.cfg["diagnostics"]["enabled"] = True
        after = self.run_model()
        for table in ("terminal_values","risk_summary","simulation_diagnostics","bands"):
            pd.testing.assert_frame_equal(before.tables[table],after.tables[table])
        self.assertFalse(after.tables["daily_distribution_diagnostics"].empty)
        self.assertEqual(set(after.tables["horizon_risk"]["horizon_sessions"]),{1,5,20})

    def test_stress_baseline_drift_and_covariance_match_independent_calculation(self):
        before = self.run_model(); self.stress(); after = self.run_model()
        baseline = after.tables["terminal_values"].query("calibration_set == 'baseline'")
        pd.testing.assert_frame_equal(before.tables["terminal_values"],baseline.reset_index(drop=True))
        calibration = after.tables["calibration_returns"]
        stress = calibration.query("calibration_set == 'test_crisis'")
        self.assertTrue((stress["return_start_date"]>="2026-01-02").all())
        self.assertTrue((stress["date"]<="2026-06-30").all())
        raw = self.p.tables["selected_prices"]
        price = raw.pivot(index="date",columns="instrument_id",values="analysis_price").loc["2026-01-02":"2026-06-30"]
        expected = np.log(price/price.shift()).dropna()
        np.testing.assert_allclose(stress[["asset::A","asset::B"]],expected,atol=1e-14)
        parameters = after.tables["calibration_parameters"]
        a = parameters.query("calibration_set == 'baseline'").set_index("factor")
        b = parameters.query("calibration_set == 'test_crisis'").set_index("factor")
        np.testing.assert_allclose(a["mean_daily_log"],b["mean_daily_log"])
        np.testing.assert_allclose(b["sample_mean_daily_log"],expected.mean())
        np.testing.assert_allclose(b["std_daily_log"],expected.std(ddof=1))
        covariance = after.tables["calibration_covariance"].query("calibration_set == 'test_crisis'")
        matrix = covariance.pivot(index="factor_i",columns="factor_j",values="covariance_daily_log")
        np.testing.assert_allclose(matrix,expected.cov(),atol=1e-14)
        # Changing only simulated drift must not change fitted stress tails.
        self.stress("stress"); own_drift = self.run_model()
        c = own_drift.tables["calibration_parameters"].query("calibration_set == 'test_crisis'")
        np.testing.assert_allclose(c["mean_daily_log"],expected.mean())
        self.assertEqual(after.summary["models"]["test_crisis__student_t_truncated"],
                         own_drift.summary["models"]["test_crisis__student_t_truncated"])
        self.assertNotEqual(after.run_id,own_drift.run_id)

    def test_unavailable_and_future_stress_do_not_remove_baseline(self):
        for start,end in (("2008-01-01","2009-06-30"),("2026-01-02","2026-10-09")):
            self.cfg["stress_calibrations"] = [dict(name="missing",start=start,end=end,drift="baseline")]
            result = self.run_model()
            self.assertEqual(result.summary["status"],"complete")
            self.assertEqual(result.summary["stress_calibrations"][0]["status"],"unavailable")
            self.assertEqual(set(result.summary["models"]),{"gaussian","student_t_truncated"})
            self.assertTrue(any("unavailable" in w for w in result.summary["warnings"]))

    def test_stress_cannot_silently_switch_simulation_currency(self):
        self.p.summary["portfolio"]["reporting_currency"] = "EUR"
        # Only the baseline has FX data. Stress must stay EUR or be unavailable.
        dates = _sessions("XNYS","2026-07-01","2026-10-08")
        self.p.tables["selected_references"] = pd.DataFrame(dict(instrument_id="EURUSD",date=dates,
            data_type="fx_reference_rate",unit="USD per EUR",value=1.1*np.exp(np.arange(len(dates))*.0001)))
        self.stress(); result = self.run_model()
        self.assertEqual(result.summary["simulation_currency"],"EUR")
        self.assertEqual(result.summary["stress_calibrations"][0]["status"],"unavailable")

    def test_invalid_config_and_legacy_defaults(self):
        from copy import deepcopy
        original = deepcopy(self.cfg)
        invalid = [dict(name="../bad",start="2008-01-01",end="2009-06-30",drift="baseline"),
                   dict(name="baseline",start="2008-01-01",end="2009-06-30",drift="baseline"),
                   dict(name="bad",start="2009-06-30",end="2008-01-01",drift="baseline"),
                   dict(name="bad",start="2008-01-01",end="2009-06-30",drift="invented")]
        for stress in invalid:
            self.cfg["stress_calibrations"] = [stress]
            with self.assertRaises(ValueError): self.run_model()
        self.cfg = deepcopy(original)
        self.cfg["diagnostics"]["horizons_sessions"] = [True]
        with self.assertRaises(ValueError): self.run_model()
        self.cfg = deepcopy(original)
        del self.cfg["stress_calibrations"]; del self.cfg["diagnostics"]
        self.file.write_text(json.dumps(self.cfg),encoding="utf-8")
        cfg = load_monte_carlo_config(self.file)
        self.assertEqual(cfg["stress_calibrations"],[])
        self.assertTrue(cfg["diagnostics"]["enabled"])

    def test_export_plots_and_checksums_with_unavailable_horizon(self):
        self.cfg.update(plots=True,paths=50,forecast_end="2026-10-12")
        self.stress(); result = self.run_model()
        output = result.save(self.root/"diagnostics")
        manifest = json.loads((output/"manifest.json").read_text(encoding="utf-8"))
        for entry in manifest["files"].values():
            self.assertEqual(hashlib.sha256((output/entry["file"]).read_bytes()).hexdigest(),entry["sha256"])
        self.assertTrue(any(k.startswith("plots/daily_tails_") for k in manifest["files"]))
        self.assertIn("plots/horizon_risk_test_crisis",manifest["files"])
        self.assertIn("baseline LOG drift",(output/"report.md").read_text(encoding="utf-8"))
        self.assertTrue((result.tables["horizon_risk"].query("horizon_sessions == 20")["status"]=="unavailable_horizon").all())

    def test_daily_diagnostics_distinguish_equal_variance_gaussian_and_t_shocks(self):
        from src.monte_carlo import _draw_shocks, _truncation
        root = np.array([[.01]])
        parameters = _truncation(1,4,.0005); parameters["df"] = 4
        gaussian = _draw_shocks(np.random.default_rng(82),100000,root)[:,0]
        student = _draw_shocks(np.random.default_rng(83),100000,root,parameters)[:,0]
        g,_ = daily_distribution(gaussian,0.,.01,[3,4])
        t,_ = daily_distribution(student,0.,.01,[3,4])
        self.assertAlmostEqual(g["std_daily_log"],t["std_daily_log"],delta=.0003)
        self.assertGreater(t["excess_kurtosis"],g["excess_kurtosis"]+1)
        self.assertGreater(t["probability_abs_z_gt_4"],10*g["probability_abs_z_gt_4"])

    def test_deterministic_diagnostics_and_all_horizons_unavailable_are_explicit(self):
        for key,rows in self.p.tables["selected_prices"].groupby("instrument_id"):
            self.p.tables["selected_prices"].loc[self.p.tables["selected_prices"]["instrument_id"]==key,"analysis_price"] = 100.
        self.p.tables["selected_prices"]["simple_return"] = self.p.tables["selected_prices"]["simple_return"].where(
            self.p.tables["selected_prices"]["simple_return"].isna(),0.)
        self.cfg.update(paths=10,plots=True,forecast_end="2026-10-12")
        self.cfg["diagnostics"]["horizons_sessions"] = [5,20]
        self.cfg["diagnostics"]["sigma_thresholds"] = [2,5]
        result = self.run_model()
        self.assertEqual(result.summary["status"],"complete")
        self.assertTrue((result.tables["horizon_risk"]["status"]=="unavailable_horizon").all())
        self.assertFalse(result.tables["daily_distribution_diagnostics"]["standardization_available"].any())
        report = result.report()
        self.assertIn("P(abs(z)>5)",report)
        self.assertNotIn("P(abs(z)>3)",report)
        output=result.save(self.root/"constant_output")
        self.assertTrue((output/"plots/path_risk_baseline.png").is_file())
        self.assertTrue((output/"plots/horizon_risk_baseline.png").is_file())

    def test_offsetting_fx_variance_roundoff_does_not_abort_diagnostics(self):
        self.cfg.update(paths=20,forecast_end="2026-10-12")
        self.p.summary["portfolio"]["reporting_currency"] = "EUR"
        self.p.summary["portfolio"]["holdings"] = [{"request_id":"A","weight":1.}]
        dates = sorted(self.p.tables["selected_prices"]["date"].unique())
        rng = np.random.default_rng(0)
        fx = np.exp(np.r_[0,np.cumsum(rng.normal(0,.01,len(dates)-1))])
        prices = 100*fx
        self.p.tables["selected_prices"] = pd.DataFrame(dict(instrument_id="A",date=dates,calendar="XNYS",currency="USD",
            return_basis="split_and_dividend_adjusted_close",analysis_price=prices,
            simple_return=np.r_[np.nan,prices[1:]/prices[:-1]-1]))
        self.p.tables["selected_references"] = pd.DataFrame(dict(instrument_id="EURUSD",date=dates,
            data_type="fx_reference_rate",unit="USD per EUR",value=fx))
        result = self.run_model()
        self.assertEqual(result.summary["status"],"complete")
        self.assertEqual(result.summary["simulation_currency"],"EUR")
        np.testing.assert_allclose(result.tables["terminal_values"]["terminal_value"],100,rtol=0,atol=1e-10)
        diagnostics = result.tables["daily_distribution_diagnostics"].query("scope == 'holding' and source == 'simulated'")
        self.assertTrue(np.isfinite(diagnostics["std_daily_log"]).all())
        self.assertFalse(diagnostics["standardization_available"].any())
        # Genuine invalid covariance must still raise, rather than be clipped.
        from src.monte_carlo import _root_covariance
        with self.assertRaisesRegex(ValueError,"positive semidefinite"):
            _root_covariance(np.array([[1.,2.],[2.,1.]]))

    def test_accepted_weight_rounding_has_no_constant_path_loss_or_input_mutation(self):
        from copy import deepcopy
        self.cfg.update(paths=10,forecast_end="2026-10-16")
        prices = self.p.tables["selected_prices"]
        prices["analysis_price"] = 100.
        prices["simple_return"] = prices["simple_return"].where(prices["simple_return"].isna(),0.)
        self.stress()
        for second_weight in (.499999999,.500000001,.5):
            self.p.summary["portfolio"]["holdings"] = [{"request_id":"A","weight":.5},{"request_id":"B","weight":second_weight}]
            original = deepcopy(self.p.summary["portfolio"])
            result = self.run_model()
            self.assertEqual(result.summary["status"],"complete")
            self.assertEqual(self.p.summary["portfolio"],original)
            self.assertEqual(result.summary["identity"]["portfolio"],original)
            rows = result.tables["risk_summary"].query("scope == 'portfolio'")
            np.testing.assert_array_equal(rows["probability_loss"],0.)
            np.testing.assert_array_equal(rows["mean_max_drawdown"],0.)
            np.testing.assert_array_equal(rows["mean_terminal"],100.)
            np.testing.assert_array_equal(result.tables["bands"].query("scope == 'portfolio'")["mean"],100.)
            self.assertEqual(result.summary["weight_rounding_reconciled"],second_weight != .5)
            self.assertEqual([h["configured_weight"] for h in result.summary["holdings"]],[.5,second_weight])
            np.testing.assert_allclose([h["weight"] for h in result.summary["holdings"]],
                                       np.array([.5,second_weight])/(.5+second_weight))

    def test_material_weight_errors_remain_rejected(self):
        for weight in (.49,.51):
            self.p.summary["portfolio"]["holdings"][1]["weight"] = weight
            with self.assertRaisesRegex(ValueError,"sum to one"):
                self.run_model()

    def test_stable_portfolio_aggregation_preserves_tiny_positive_wealth(self):
        from src.monte_carlo import _portfolio_wealth
        weights = np.array([1/3,1/3,1/3])
        levels = np.array([[100.,100.,100.],[1e-100,2e-100,3e-100],[90.,100.,110.],[1e100,2e100,3e100]])
        actual = _portfolio_wealth(levels,weights,100.)
        self.assertEqual(actual[0],100.)
        self.assertEqual(actual[2],100.)
        np.testing.assert_allclose(actual[[1,3]],levels[[1,3]] @ weights,rtol=1e-15,atol=0)
        self.assertTrue((actual>0).all())


    def test_loss_probability_roundoff_tolerance_preserves_real_small_losses(self):
        from src.mc_diagnostics import probability_of_loss, LOSS_PROBABILITY_TOLERANCE
        tol = LOSS_PROBABILITY_TOLERANCE
        losses = np.array([-1e-12, -tol, 0., tol/2, tol, 2*tol, 1e-12, .1])
        self.assertEqual(probability_of_loss(losses), 3/8)
        # Same classification at different initial wealth scales, with actual
        # small losses of 1e-12 retained. No wealth or return array is modified.
        for initial in (1., 100., 1e9):
            wealth = initial*np.column_stack([np.ones(len(losses)), 1-losses])
            original = wealth.copy()
            rows,_,_ = wealth_diagnostics(wealth, initial, [1], [.99])
            self.assertEqual(rows[0]['probability_loss'], 3/8)
            np.testing.assert_array_equal(wealth, original)

    def test_exact_fx_offset_has_no_false_loss_across_models_and_horizons(self):
        self.cfg.update(paths=30,forecast_end='2027-12-17')
        self.stress('stress')
        self.p.summary['portfolio']['reporting_currency'] = 'EUR'
        self.p.summary['portfolio']['holdings'] = [{'request_id':'A','weight':1.}]
        dates = sorted(self.p.tables['selected_prices']['date'].unique())
        # Seed 2 produces negative cancellation residuals, unlike seed 0 of
        # the older variance test. More than one day exposes accumulation.
        fx = np.exp(np.r_[0,np.cumsum(np.random.default_rng(2).normal(0,.01,len(dates)-1))])
        prices = 100*fx
        self.p.tables['selected_prices'] = pd.DataFrame(dict(
            instrument_id='A',date=dates,calendar='XNYS',currency='USD',
            return_basis='split_and_dividend_adjusted_close',analysis_price=prices,
            simple_return=np.r_[np.nan,prices[1:]/prices[:-1]-1]))
        self.p.tables['selected_references'] = pd.DataFrame(dict(
            instrument_id='EURUSD',date=dates,data_type='fx_reference_rate',unit='USD per EUR',value=fx))
        result = self.run_model()
        self.assertEqual(result.summary['status'], 'complete')
        self.assertEqual(result.summary['stress_calibrations'][0]['status'], 'complete')
        self.assertEqual(len(result.summary['models']), 4)
        np.testing.assert_array_equal(result.tables['risk_summary']['probability_loss'], 0.)
        np.testing.assert_array_equal(result.tables['horizon_risk']['probability_loss'], 0.)
        np.testing.assert_allclose(result.tables['terminal_values']['terminal_value'],100.,rtol=0,atol=2e-12)
        # Classification is corrected without snapping saved wealth to 100.
        self.assertTrue((result.tables['terminal_values']['terminal_value'] != 100.).any())

    def test_close_probabilities_keep_distinct_columns_report_and_plots(self):
        from src.mc_diagnostics import probability_label, confidence_label, diagnostic_figures
        from matplotlib import pyplot as plt
        self.cfg.update(paths=20,models=['gaussian'],quantiles=[.95,.9500001],confidence_levels=[.99,.9900001])
        result = self.run_model()
        self.assertEqual(probability_label(.95), '0.95')
        self.assertEqual(probability_label(.9500001), '0.9500001')
        self.assertNotEqual(probability_label(.95), probability_label(np.nextafter(.95,1.)))
        self.assertEqual(confidence_label(.99), '99%')
        self.assertEqual(confidence_label(.9900001), '99.00001%')
        terminal = result.tables['terminal_values'].query("scope == 'portfolio'")['terminal_value'].to_numpy()
        risk = result.tables['risk_summary'].query("scope == 'portfolio'").iloc[0]
        for q in self.cfg['quantiles']:
            label = probability_label(q)
            self.assertEqual(risk['terminal_quantile_'+label], np.quantile(terminal,q))
            self.assertIn('quantile_'+label, result.tables['bands'])
        for q in self.cfg['confidence_levels']:
            label = probability_label(q)
            self.assertEqual(risk['loss_var_'+label], np.quantile(1-terminal/100,q))
            self.assertIn('loss_es_'+label, result.tables['risk_summary'])
            self.assertIn('loss_var_'+label, result.tables['horizon_risk'])
            self.assertIn('loss_es_'+label, result.tables['horizon_risk'])
        self.assertIn('99.00001% loss VaR',result.report())
        self.assertIn('numerical residuals',result.report())
        found = False
        for name,fig in diagnostic_figures(result.tables,result.settings):
            try:
                if name == 'horizon_risk_baseline':
                    found = True
                    self.assertIn('99.00001%',fig.axes[0].get_title())
                    expected = result.tables['horizon_risk'].query("scope == 'portfolio' and status == 'complete'").sort_values('horizon_sessions')
                    np.testing.assert_array_equal(fig.axes[0].lines[0].get_ydata(),100*expected['loss_var_0.9900001'])
            finally:
                plt.close(fig)
        self.assertTrue(found)
