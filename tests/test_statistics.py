"""Financial invariants for the offline statistics stage (no live providers)."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import hashlib
import json
import math
import socket
import unittest

import numpy as np
import pandas as pd

from src.data_cleaning import ProcessedArchive
from src.statistics import AssetStatistics, analyze_processed, analyze_saved_processing, load_modeling_config

ROOT = Path(__file__).resolve().parents[1]


def frame(key, values, dates=None, currency="USD", basis="split_and_dividend_adjusted_close"):
    """Expected-session rows preserve NaNs; never use pct_change's fill defaults."""
    dates = dates or pd.bdate_range("2026-01-05",periods=len(values)).strftime("%Y-%m-%d").tolist()
    prices = pd.Series(values,dtype=float)
    return pd.DataFrame(dict(instrument_id=key,date=dates,analysis_price=prices,
                             simple_return=prices/prices.shift()-1,currency=currency,return_basis=basis))


class StatisticsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.raw = self.root/"raw"; self.raw.mkdir()
        self.config = json.loads((ROOT/"config/modeling.json").read_text(encoding="utf-8"))
        self.cfg = self.config["statistics"]
        self.cfg.update(minimum_observations=2,rolling_windows=[2,3],fit_student_t=False,plots=False,
                        return_types=["simple"])
        self.cfg["stress_periods"] = [{"name":"example","start":"2026-01-06","end":"2026-01-08"}]
        self.file = self.root/"modeling.json"
        self.write_config()

    def write_config(self):
        self.file.write_text(json.dumps(self.config),encoding="utf-8")

    def processed(self, frames, holdings=None, reporting="USD"):
        if holdings is None:
            holdings = [{"request_id":f.iloc[0]["instrument_id"],"weight":1/len(frames)} for f in frames]
        return ProcessedArchive(self.raw,"2026-01-09","fixture",
            {"selected_prices":pd.concat(frames,ignore_index=True)},
            {"portfolio":{"holdings":holdings,"reporting_currency":reporting}})

    def test_sample_moments_compounding_and_no_dividend_double_count(self):
        f = frame("GENERIC",[100,110,99,108.9,108.9])
        f["dividends"] = 200  # Already adjusted; must have no effect.
        a = AssetStatistics("GENERIC","USD","split_and_dividend_adjusted_close",f,self.cfg)
        r = np.array([.1,-.1,.1,0])
        self.assertAlmostEqual(a.metrics["mean_daily"],r.mean())
        self.assertAlmostEqual(a.metrics["std_daily"],r.std(ddof=1))
        self.assertAlmostEqual(a.metrics["volatility_annualized"],r.std(ddof=1)*math.sqrt(252))
        self.assertAlmostEqual(a.metrics["mean_annualized"],r.mean()*252)
        self.assertAlmostEqual(a.metrics["cumulative_return"],.089)
        self.assertAlmostEqual(a.metrics["max_drawdown"],-.1)
        self.assertAlmostEqual(a.metrics["skewness"],pd.Series(r).skew())
        self.assertAlmostEqual(a.metrics["excess_kurtosis"],pd.Series(r).kurt())

    def test_cagr_uses_elapsed_calendar_time_not_arithmetic_annualization(self):
        a = AssetStatistics("A","USD","price_return",
            frame("A",[100,110],dates=["2025-01-01","2026-01-01"]),self.cfg)
        self.assertAlmostEqual(a.metrics["cagr"],1.1**(365.25/365)-1)
        self.assertNotAlmostEqual(a.metrics["cagr"],a.metrics["mean_annualized"])

    def test_buy_and_hold_weights_drift_without_daily_rebalancing(self):
        p = self.processed([frame("A",[100,200,200]),frame("B",[100,100,200])])
        result = analyze_processed(p,self.file)
        a = result.portfolio
        np.testing.assert_allclose(a.frame["analysis_price"],[100,150,200])
        np.testing.assert_allclose(a.frame["simple_return"].dropna(),[.5,1/3])
        # Daily-rebalanced would grow to 225, which is not this model.
        self.assertAlmostEqual(a.metrics["cumulative_return"],1)
        self.assertAlmostEqual(result.tables["dependence"].iloc[0]["covariance_daily"],.5)

    def test_missing_price_breaks_daily_returns_rolling_and_path_metrics(self):
        a = AssetStatistics("A","USD","price_return",frame("A",[100,110,np.nan,121,133.1]),self.cfg)
        self.assertEqual(a.metrics["observations"],2)
        self.assertIsNone(a.metrics["cumulative_return"])
        self.assertIsNone(a.metrics["max_drawdown"])
        self.assertEqual(a.metrics["path_status"],"unavailable_gaps")
        self.assertTrue(a.rolling["volatility_2"].isna().all())

    def test_missing_late_inception_does_not_become_an_interior_gap(self):
        result = analyze_processed(self.processed([frame("A",[100,110,121,133.1]),
                                                   frame("B",[np.nan,100,100,100])]),self.file)
        self.assertEqual(result.portfolio.metrics["first_price_date"],"2026-01-06")
        np.testing.assert_allclose(result.portfolio.frame["analysis_price"],[100,105,110.5])
        self.assertEqual(result.portfolio.metrics["path_status"],"available")

    def test_cross_calendar_intervals_and_rolling_do_not_bridge(self):
        us = frame("US",[100,101,104,105],["2026-07-01","2026-07-02","2026-07-06","2026-07-07"])
        uk = frame("UK",[100,101,102,104,105],["2026-07-01","2026-07-02","2026-07-03","2026-07-06","2026-07-07"])
        result = analyze_processed(self.processed([us,uk]),self.file)
        self.assertEqual(result.tables["aligned_returns"]["date"].tolist(),["2026-07-02","2026-07-07"])
        self.assertEqual(result.portfolio.metrics["observations"],2)
        self.assertIsNone(result.portfolio.metrics["max_drawdown"])
        self.assertTrue(result.tables["rolling_correlations"]["correlation"].isna().all())

    def test_stress_window_excludes_return_starting_before_window(self):
        result = analyze_processed(self.processed([frame("A",[100,110,99,108.9,109])]),self.file)
        s = result.assets["A"].stress[0]
        self.assertEqual(s["observations"],2)
        self.assertAlmostEqual(s["cumulative_return"],108.9/110-1)
        self.assertEqual(s["first_price_date"],"2026-01-06")
        self.assertEqual(result.tables["stress_dependence"].iloc[0]["observations"],2)

    def test_partial_and_empty_stress_windows_are_explicit(self):
        self.cfg["stress_periods"] = [
            {"name":"partial","start":"2025-12-01","end":"2026-01-07"},
            {"name":"absent","start":"2008-01-01","end":"2008-12-31"}]
        self.write_config()
        result = analyze_processed(self.processed([frame("A",[100,110,120])]),self.file)
        table = result.tables["stress_statistics"]
        self.assertEqual(set(table.loc[table["name"]=="partial","coverage"]),{"partial_window"})
        self.assertEqual(set(table.loc[table["name"]=="absent","coverage"]),{"unavailable"})
        self.assertEqual(set(table.loc[table["name"]=="absent","status"]),{"no_returns"})

    def test_fx_absence_and_mixed_bases_do_not_create_false_portfolio(self):
        a,b = frame("A",[100,110,120]),frame("B",[100,101,102],currency="EUR")
        result = analyze_processed(self.processed([a,b]),self.file)
        self.assertIsNone(result.portfolio)
        self.assertEqual(result.summary["portfolio_status"],"unavailable_currency_conversion")
        self.assertEqual(len(result.assets),2)
        b["currency"] = "USD"; b["return_basis"] = "price_index_level"
        result = analyze_processed(self.processed([a,b]),self.file)
        self.assertEqual(result.summary["portfolio_status"],"unavailable_incompatible_return_bases")

    def test_usd_results_are_not_reported_as_eur_and_references_are_excluded(self):
        p = self.processed([frame("A",[100,110,120]),frame("REFERENCE",[500,550,400])],
                           holdings=[{"request_id":"A","weight":1}],reporting="EUR")
        result = analyze_processed(p,self.file)
        self.assertEqual(set(result.assets),{"A"})
        self.assertEqual(result.summary["analysis_currency"],"USD")
        self.assertFalse(result.summary["reporting_currency_returns_computed"])
        self.assertIn("not EUR",result.report())

    def test_missing_holding_is_not_dropped_and_weights_not_renormalized(self):
        p = self.processed([frame("A",[100,110,120])],
                          holdings=[{"request_id":"A","weight":.5},{"request_id":"MISSING","weight":.5}])
        result = analyze_processed(p,self.file)
        self.assertIsNone(result.portfolio)
        self.assertEqual(result.summary["portfolio_status"],"unavailable_missing_holdings")

    def test_zero_weight_missing_asset_and_reserved_column_names(self):
        p = self.processed([frame("date",[100,110,120])],
                          holdings=[{"request_id":"date","weight":1},{"request_id":"MISSING","weight":0}])
        result = analyze_processed(p,self.file)
        self.assertIsNotNone(result.portfolio)
        self.assertIn("return::date",result.tables["aligned_returns"].columns)

    def test_no_usable_holdings_and_constant_or_short_samples(self):
        p = self.processed([frame("OTHER",[100,110])],holdings=[{"request_id":"MISSING","weight":1}])
        result = analyze_processed(p,self.file)
        self.assertTrue(result.tables["statistics"].empty)
        self.assertIn("unavailable_missing_holdings",result.report())
        for values in ([100],[100,100,100,100]):
            a = AssetStatistics("A","USD","price_return",frame("A",values),self.cfg)
            self.assertIsNone(a.metrics["skewness"])
            self.assertIsNone(a.metrics["excess_kurtosis"])

    def test_reject_invalid_weights_prices_intervals_and_config(self):
        for weight in (-1,float("nan"),None,True):
            p = self.processed([frame("A",[100,110])],holdings=[{"request_id":"A","weight":weight}])
            with self.assertRaises((ValueError,TypeError)):
                analyze_processed(p,self.file)
        f = frame("A",[100,110,120]); f.loc[1,"simple_return"] = .9
        with self.assertRaises(ValueError):
            AssetStatistics("A","USD","price_return",f,self.cfg)
        f = frame("A",[100,110,120]); f["return_start_date"] = "1900-01-01"
        with self.assertRaises(ValueError):
            AssetStatistics("A","USD","price_return",f,self.cfg)
        self.cfg["periods_per_year"] = True; self.write_config()
        with self.assertRaises(ValueError):
            load_modeling_config(self.file)

    def test_statistical_fit_is_descriptive_and_failure_isolated(self):
        self.cfg["fit_student_t"] = True
        with patch("scipy.stats.t.fit",return_value=(5,.001,.01)):
            a = AssetStatistics("A","USD","price_return",frame("A",[100,110,99,105]),self.cfg)
        self.assertEqual(a.distribution["student_t_df"],5)
        self.assertEqual(a.distribution["student_t_scale"],.01)
        with patch("scipy.stats.t.fit",side_effect=ValueError("test failure")):
            a = AssetStatistics("A","USD","price_return",frame("A",[100,110,99,105]),self.cfg)
        self.assertEqual(a.distribution["student_t_status"],"fit_failed")
        self.assertEqual(a.metrics["observations"],3)

    def test_saved_input_checksums_offline_exports_and_preservation(self):
        p = self.processed([frame("A",[100,110,99,105]),frame("B",[100,101,104,102])])
        source = self.root/"processed"; source.mkdir()
        csv = source/"selected_prices.csv"; p.tables["selected_prices"].to_csv(csv,index=False)
        manifest = {**p.summary,"complete":True,"archive":str(p.archive),"cob":p.cob,"run_id":p.run_id,
                    "files":{"selected_prices":{"file":csv.name,"rows":len(p.tables["selected_prices"]),
                    "sha256":hashlib.sha256(csv.read_bytes()).hexdigest()}}}
        (source/"manifest.json").write_text(json.dumps(manifest),encoding="utf-8")
        before = {f:(f.read_bytes(),f.stat().st_mtime_ns) for f in source.iterdir()}
        self.cfg.update(plots=True,return_types=["simple","log"]); self.write_config()
        with patch.object(socket.socket,"connect",side_effect=AssertionError("No network")):
            result = analyze_saved_processing(source,self.file)
            output = result.save(source/"statistics"/result.run_id)
        self.assertEqual(before,{f:(f.read_bytes(),f.stat().st_mtime_ns) for f in before})
        saved = json.loads((output/"manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(saved["complete"])
        self.assertEqual(len(list((output/"plots").rglob("*.png"))),10)
        for kind in ("simple","log"):
            self.assertEqual(len(list((output/"plots"/kind).glob("*.png"))),4)
        png_entries = [e["file"] for e in saved["files"].values() if e["file"].endswith(".png")]
        self.assertEqual(len(png_entries),10)
        self.assertEqual(len(set(png_entries)),10)
        self.assertEqual(saved["schema_version"],2)
        self.assertEqual(saved["aligned_observations_by_return_type"],{"simple":3,"log":3})
        for entry in saved["files"].values():
            self.assertEqual(hashlib.sha256((output/entry["file"]).read_bytes()).hexdigest(),entry["sha256"])
        with self.assertRaises(ValueError): result.save(self.raw/"stats")
        with self.assertRaises(ValueError): result.save(source)
        with self.assertRaises(ValueError): result.save(self.root)
        csv.write_text(csv.read_text()+"\n",encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"checksum"):
            analyze_saved_processing(source,self.file)

    def test_deterministic_identity_and_input_immutability(self):
        p = self.processed([frame("A",[100,110,99])])
        before = p.tables["selected_prices"].copy(deep=True)
        one,two = analyze_processed(p,self.file),analyze_processed(p,self.file)
        self.assertEqual(one.run_id,two.run_id)
        pd.testing.assert_frame_equal(before,p.tables["selected_prices"])
        self.cfg["periods_per_year"] = 250; self.write_config()
        self.assertNotEqual(one.run_id,analyze_processed(p,self.file).run_id)


    def enable_both(self):
        self.cfg["return_types"] = ["simple","log"]
        self.write_config()

    def test_log_statistics_use_same_sample_and_exact_transformation(self):
        self.enable_both()
        a = AssetStatistics("A","USD","price_return",frame("A",[100,120,90,108,100,110]),self.cfg)
        logs = np.log(np.array([120/100,90/120,108/90,100/108,110/100]))
        m = a.return_statistics["log"]["metrics"]
        np.testing.assert_allclose(a.frame["log_return"].dropna(),logs)
        self.assertEqual(m["observations"],a.metrics["observations"])
        self.assertAlmostEqual(m["mean_daily"],logs.mean())
        self.assertAlmostEqual(m["std_daily"],logs.std(ddof=1))
        self.assertAlmostEqual(m["mean_annualized"],252*logs.mean())
        self.assertAlmostEqual(m["volatility_annualized"],np.sqrt(252)*logs.std(ddof=1))
        self.assertAlmostEqual(m["skewness"],pd.Series(logs).skew())
        self.assertAlmostEqual(m["excess_kurtosis"],pd.Series(logs).kurt())
        self.assertAlmostEqual(m["quantile_0.01"],np.quantile(logs,.01))
        self.assertAlmostEqual(m["worst_daily"],logs.min())
        self.assertAlmostEqual(m["best_daily"],logs.max())
        self.assertAlmostEqual(np.exp(logs.sum())-1,a.path_metrics["cumulative_return"])
        self.assertNotIn("cagr",m)

    def test_log_portfolio_returns_come_from_wealth_not_weighted_logs(self):
        self.enable_both()
        result = analyze_processed(self.processed([frame("A",[100,200,200]),frame("B",[100,100,200])]),self.file)
        np.testing.assert_allclose(result.portfolio.frame["log_return"].dropna(),np.log([1.5,4/3]))
        self.assertNotAlmostEqual(result.portfolio.frame["log_return"].iloc[1],.5*np.log(2))
        self.assertEqual(len(result.tables["path_statistics"]),3)
        self.assertNotIn("return_type",result.tables["path_statistics"])
        self.assertNotIn("cagr",result.tables["statistics"])
        self.assertEqual(len(result.tables["daily_statistics"]),9)
        self.assertEqual(len(result.tables["statistics"]),6)

    def test_both_return_covariances_and_stresses_match_manual_common_samples(self):
        self.enable_both()
        result = analyze_processed(self.processed([frame("A",[100,110,99,108.9,105]),
                                                   frame("B",[100,102,101,103,100])]),self.file)
        expected = pd.DataFrame({k:a.frame["simple_return"] for k,a in result.assets.items()}).dropna()
        for kind,values in (("simple",expected),("log",np.log1p(expected))):
            dep = result.tables["dependence"].query("return_type == @kind")
            got = dep.pivot(index="instrument_i",columns="instrument_j",values="covariance_daily")
            np.testing.assert_allclose(got,values.cov())
            np.testing.assert_allclose(dep["covariance_annualized"],dep["covariance_daily"]*252)
            corr = dep.pivot(index="instrument_i",columns="instrument_j",values="correlation")
            np.testing.assert_allclose(corr,values.corr())
            aligned = result.tables["aligned_returns"].query("return_type == @kind")
            np.testing.assert_allclose(aligned[["return::A","return::B"]],values)
            a = result.assets["A"].return_statistics[kind]
            manual = values["A"].rolling(2,min_periods=2).std()*np.sqrt(252)
            np.testing.assert_allclose(a["rolling"]["volatility_2"].iloc[1:],manual,equal_nan=True)
            stress = a["stress"][0]
            self.assertEqual(stress["observations"],2)
            self.assertAlmostEqual(stress["mean_daily"],values["A"].iloc[1:3].mean())
            stress_corr = result.tables["stress_dependence"].query(
                "return_type == @kind and instrument_i == 'A' and instrument_j == 'B'").iloc[0]
            self.assertAlmostEqual(stress_corr["correlation"],values.iloc[1:3].corr().loc["A","B"])
        self.assertEqual(result.summary["aligned_observations"],4)
        self.assertEqual(result.summary["aligned_observations_by_return_type"],{"simple":4,"log":4})

    def test_both_conventions_preserve_gaps_and_cross_calendar_alignment(self):
        self.enable_both()
        result = analyze_processed(self.processed([frame("A",[100,110,np.nan,121,133.1])]),self.file)
        a = result.assets["A"]
        for analysis in a.return_statistics.values():
            self.assertEqual(analysis["metrics"]["observations"],2)
            self.assertTrue(analysis["rolling"]["volatility_2"].isna().all())
        self.assertIsNone(a.path_metrics["cagr"])
        us = frame("US",[100,101,104,105],["2026-07-01","2026-07-02","2026-07-06","2026-07-07"])
        uk = frame("UK",[100,101,102,104,105],["2026-07-01","2026-07-02","2026-07-03","2026-07-06","2026-07-07"])
        result = analyze_processed(self.processed([us,uk]),self.file)
        for kind in ("simple","log"):
            rows = result.tables["aligned_returns"].query("return_type == @kind")
            self.assertEqual(rows["date"].tolist(),["2026-07-02","2026-07-07"])
        self.assertTrue(result.tables["rolling_correlations"]["correlation"].isna().all())

    def test_fits_receive_correct_raw_return_type_and_fail_independently(self):
        self.enable_both()
        self.cfg["fit_student_t"] = True
        with patch("scipy.stats.t.fit",side_effect=[ValueError("simple failed"),(1.9,.001,.01)]) as fit:
            a = AssetStatistics("A","USD","price_return",frame("A",[100,110,99,105]),self.cfg)
        simple = a.frame["simple_return"].dropna().to_numpy()
        np.testing.assert_allclose(fit.call_args_list[0].args[0],simple)
        np.testing.assert_allclose(fit.call_args_list[1].args[0],np.log1p(simple))
        self.assertEqual(a.return_statistics["simple"]["distribution"]["student_t_status"],"fit_failed")
        self.assertEqual(a.return_statistics["log"]["distribution"]["student_t_status"],"available")
        self.assertFalse(a.return_statistics["log"]["distribution"]["student_t_variance_exists"])

    def test_log_only_config_and_invalid_return_types(self):
        self.cfg["return_types"] = ["log"]; self.write_config()
        result = analyze_processed(self.processed([frame("A",[100,110,99,105])]),self.file)
        for name in ("statistics","stress_statistics","distributions","rolling_volatility",
                     "dependence","rolling_correlations","aligned_returns","stress_dependence"):
            self.assertEqual(set(result.tables[name]["return_type"]),{"log"} if len(result.tables[name]) else set())
        self.assertEqual(result.assets["A"].metrics["return_type"],"log")
        self.assertIn("Log returns",result.report())
        for invalid in ([],["simple","simple"],["bad"],"log",[True],[{}],None):
            with self.subTest(invalid=invalid):
                self.cfg["return_types"] = invalid; self.write_config()
                with self.assertRaisesRegex(ValueError,"return_types"):
                    load_modeling_config(self.file)

    def test_shared_paths_and_simple_results_unchanged_when_log_enabled(self):
        p = self.processed([frame("A",[100,110,99,105]),frame("B",[100,102,101,103])])
        simple = analyze_processed(p,self.file)
        self.enable_both()
        both = analyze_processed(p,self.file)
        pd.testing.assert_frame_equal(simple.tables["path_statistics"],both.tables["path_statistics"])
        pd.testing.assert_frame_equal(simple.tables["stress_path_statistics"],both.tables["stress_path_statistics"])
        pd.testing.assert_frame_equal(simple.tables["daily_statistics"],both.tables["daily_statistics"])
        for name in ("statistics","stress_statistics","distributions","rolling_volatility",
                     "dependence","rolling_correlations","aligned_returns","stress_dependence"):
            selected = both.tables[name].query("return_type == 'simple'").reset_index(drop=True)
            pd.testing.assert_frame_equal(simple.tables[name],selected)
        self.assertNotEqual(simple.run_id,both.run_id)
        self.assertFalse(both.summary["simulation_performed"])
        self.assertFalse(both.summary["calibration_performed"])
        self.assertIn("Simple returns",both.report())
        self.assertIn("Log returns",both.report())

    def test_empty_and_constant_samples_support_both_conventions(self):
        self.enable_both()
        p = self.processed([frame("OTHER",[100,110])],holdings=[{"request_id":"MISSING","weight":1}])
        result = analyze_processed(p,self.file)
        self.assertTrue(result.tables["statistics"].empty)
        self.assertIn("Log returns",result.report())
        for values in ([100],[100,100,100,100]):
            a = AssetStatistics("A","USD","price_return",frame("A",values),self.cfg)
            for analysis in a.return_statistics.values():
                self.assertIsNone(analysis["metrics"]["skewness"])
                self.assertIsNone(analysis["metrics"]["excess_kurtosis"])


    def test_export_preserves_diagnostic_when_only_log_fit_fails(self):
        self.enable_both()
        self.cfg["fit_student_t"] = True; self.write_config()
        outcomes = [(5,.001,.01),ValueError("log fit failed"),(5,.001,.01),ValueError("log fit failed")]
        with patch("scipy.stats.t.fit",side_effect=outcomes):
            result = analyze_processed(self.processed([frame("A",[100,110,99,105])]),self.file)
        fits = result.tables["distributions"]
        self.assertEqual(fits.query("return_type == 'log'")["diagnostic"].tolist(),
                         ["log fit failed","log fit failed"])
        self.assertTrue(fits.query("return_type == 'simple'")["student_t_variance_exists"].all())

    def test_stress_coverage_accepts_nontrading_boundaries_but_not_missing_prices(self):
        self.cfg["stress_periods"] = [
            {"name":"weekend_bounds","start":"2026-01-03","end":"2026-01-11"}]
        self.write_config()
        dates = ["2026-01-02","2026-01-05","2026-01-06","2026-01-07",
                 "2026-01-08","2026-01-09","2026-01-12"]
        complete = frame("A",[100,101,102,103,104,105,106],dates)
        result = analyze_processed(self.processed([complete]),self.file)
        self.assertEqual(set(result.tables["stress_statistics"]["coverage"]),{"full_window"})
        # An expected session with no price is still incomplete coverage.
        missing = frame("A",[100,np.nan,102,103,104,105,106],dates)
        result = analyze_processed(self.processed([missing]),self.file)
        self.assertEqual(set(result.tables["stress_statistics"]["coverage"]),{"partial_window"})

    def test_saved_input_preserves_na_like_instrument_ids_and_real_gaps(self):
        keys = ["NA","NULL","N/A","001"]
        p = self.processed([frame(key,[100,110,np.nan,121,133.1]) for key in keys])
        expected = analyze_processed(p,self.file)
        source = self.root/"processed"; source.mkdir()
        csv = source/"selected_prices.csv"
        p.tables["selected_prices"].to_csv(csv,index=False)
        manifest = {**p.summary,"complete":True,"archive":str(p.archive),"cob":p.cob,"run_id":p.run_id,
                    "files":{"selected_prices":{"file":csv.name,"rows":len(p.tables["selected_prices"]),
                    "sha256":hashlib.sha256(csv.read_bytes()).hexdigest()}}}
        (source/"manifest.json").write_text(json.dumps(manifest),encoding="utf-8")
        result = analyze_saved_processing(source,self.file)
        self.assertEqual(set(result.assets),set(keys))
        for key in keys:
            pd.testing.assert_frame_equal(result.assets[key].frame,expected.assets[key].frame)
        self.assertEqual(result.portfolio.path_metrics["path_status"],"unavailable_gaps")

    def test_analysis_identity_does_not_alias_input_portfolio(self):
        p = self.processed([frame("A",[100,110,99])])
        result = analyze_processed(p,self.file)
        identity = deepcopy(result.summary["identity"])
        report = result.report()
        p.summary["portfolio"]["holdings"][0]["weight"] = .25
        p.summary["portfolio"]["reporting_currency"] = "EUR"
        self.assertEqual(result.summary["identity"],identity)
        self.assertEqual(result.report(),report)
        output = result.save(self.root/"statistics")
        saved = json.loads((output/"manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["identity"],identity)

    def test_close_quantiles_keep_distinct_values_in_export(self):
        self.cfg["quantiles"] = [.12345671,.12345672]
        self.write_config()
        result = analyze_processed(self.processed([frame("A",[100,120,90,108,100])]),self.file)
        values = result.assets["A"].frame["simple_return"].dropna().to_numpy()
        table = result.tables["statistics"]
        self.assertEqual(len([c for c in table if c.startswith("quantile_")]),2)
        for q in self.cfg["quantiles"]:
            self.assertAlmostEqual(table.iloc[0][f"quantile_{q}"],np.quantile(values,q))
