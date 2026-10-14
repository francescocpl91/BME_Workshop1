"""Offline financial invariants for historical Monte Carlo and FX fallback."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import hashlib
import json
import socket
import unittest

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.stats import f as f_distribution

from src.data_cleaning import ProcessedArchive
from src.monte_carlo import (simulate_processed,simulate_saved_processing,load_monte_carlo_config,
                            _root_covariance,_draw_shocks,_truncation,_expected_shortfall,
                            _sessions,_student_parameters,SimulationUnavailable)

ROOT=Path(__file__).resolve().parents[1]


class MonteCarloTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name); self.raw=self.root/"raw"; self.raw.mkdir()
        self.cfg=json.loads((ROOT/"config/monte_carlo.json").read_text(encoding="utf-8-sig"))
        self.cfg["stress_calibrations"]=[]
        self.cfg["diagnostics"]["enabled"]=False
        self.cfg.update(paths=2000,plots=False,sample_paths=2,forecast_start="2026-10-09",forecast_end="2026-10-16")
        self.cfg["calibration"].update(start="2026-09-01",minimum_observations=10)
        self.cfg["student_t"]["sensitivity_tail_probabilities"]=[]
        self.file=self.root/"mc.json"; self.write_config()
        self.dates=_sessions("XNYS","2026-09-01","2026-10-08")

    def write_config(self):
        self.file.write_text(json.dumps(self.cfg),encoding="utf-8")

    def prices(self,key,logs,currency="USD"):
        price=100*np.exp(np.r_[0,np.cumsum(logs)])
        return pd.DataFrame(dict(instrument_id=key,date=self.dates,calendar="XNYS",
            currency=currency,return_basis="split_and_dividend_adjusted_close",
            analysis_price=price,simple_return=np.r_[np.nan,np.expm1(logs)]))

    def processed(self,reporting="USD",currency="USD"):
        rng=np.random.default_rng(900)
        logs=rng.multivariate_normal([.0005,.0001],[[.0001,-.00001],[-.00001,.00002]],len(self.dates)-1)
        prices=pd.concat([self.prices("A",logs[:,0],currency),self.prices("B",logs[:,1],currency)],ignore_index=True)
        return ProcessedArchive(self.raw,"2026-10-08","fixture",{"selected_prices":prices},
            {"portfolio":{"reporting_currency":reporting,"holdings":[{"request_id":"A","weight":.5},{"request_id":"B","weight":.5}]}})

    def add_fx(self,p,inverse=False):
        rate=1.1*np.exp(np.arange(len(self.dates))*.001)
        p.tables["selected_references"]=pd.DataFrame(dict(instrument_id="EURUSD",date=self.dates,
            data_type="fx_reference_rate",unit="EUR per USD" if inverse else "USD per EUR",
            value=1/rate if inverse else rate))
        if inverse:
            self.cfg["fx_series"][0].update(base_currency="USD",quote_currency="EUR")
            self.write_config()

    def test_gaussian_increment_mean_covariance_and_analytical_terminal_mean(self):
        self.cfg.update(models=["gaussian"],paths=20000); self.write_config()
        result=simulate_processed(self.processed(),self.file)
        self.assertEqual(result.summary["status"],"complete")
        diagnostics=result.tables["simulation_diagnostics"]
        for row in diagnostics.to_dict("records"):
            self.assertAlmostEqual(row["simulated"],row["target"],delta=1e-4 if row["kind"]=="mean" else 3e-6)
        risk=result.tables["risk_summary"]
        self.assertTrue((risk["gaussian_mean_error_in_mc_se"].abs()<5).all())
        self.assertTrue((result.tables["terminal_values"]["terminal_value"]>0).all())

    def test_no_second_ito_correction_and_normalized_start(self):
        p=self.processed()
        p.tables["selected_prices"]=pd.concat([self.prices("A",np.full(len(self.dates)-1,.001)),
                                               self.prices("B",np.full(len(self.dates)-1,.002))])
        result=simulate_processed(p,self.file)
        n=result.summary["steps"]
        terminal=result.tables["terminal_values"]
        for model in result.summary["models"]:
            rows=terminal.query("model == @model")
            np.testing.assert_allclose(rows.query("instrument_id == 'A'")["terminal_value"],100*np.exp(.001*n),rtol=1e-10)
            expected=50*np.exp(.001*n)+50*np.exp(.002*n)
            np.testing.assert_allclose(rows.query("scope == 'portfolio'")["terminal_value"],expected,rtol=1e-10)
        start=result.tables["bands"].query("date == '2026-10-09'")
        np.testing.assert_allclose(start["mean"],100)
        meta=result.summary["models"]["student_t_truncated"]
        self.assertIsNone(meta["df"])
        self.assertFalse(meta["truncation_applied"])
        saved=result.save(self.root/"deterministic_report")
        self.assertIn("n/a",(saved/"report.md").read_text(encoding="utf-8"))

    def test_calendar_start_is_initial_level_and_end_is_exact(self):
        result=simulate_processed(self.processed(),self.file)
        dates=result.summary["forecast_dates"]
        self.assertEqual(dates[0],"2026-10-09")
        self.assertEqual(dates[-1],"2026-10-16")
        self.assertEqual(result.summary["steps"],len(dates)-1)
        self.assertNotIn("2026-10-10",dates)
        self.cfg["forecast_start"]="2026-10-10"; self.write_config()
        self.assertEqual(simulate_processed(self.processed(),self.file).summary["status"],"unavailable")

    def test_start_not_after_cob_is_saved_as_unavailable(self):
        self.cfg["forecast_start"]="2026-10-08"; self.write_config()
        result=simulate_processed(self.processed(),self.file)
        self.assertFalse(result.summary["simulation_performed"])
        output=result.save(self.root/"output")
        manifest=json.loads((output/"manifest.json").read_text())
        self.assertTrue(manifest["complete"])
        self.assertEqual(manifest["status"],"unavailable")

    def test_lookback_filter_and_future_observations_do_not_leak(self):
        p=self.processed()
        original=simulate_processed(p,self.file)
        future=p.tables["selected_prices"].query("date == '2026-10-08'").copy()
        future["date"]="2026-10-12"; future["analysis_price"]=99999; future["simple_return"]=999
        p.tables["selected_prices"]=pd.concat([p.tables["selected_prices"],future])
        with_future=simulate_processed(p,self.file)
        pd.testing.assert_frame_equal(original.tables["calibration_parameters"],with_future.tables["calibration_parameters"])
        self.cfg["calibration"]["start"]="2026-09-15"; self.write_config()
        result=simulate_processed(p,self.file)
        self.assertGreaterEqual(result.summary["calibration"]["first_start"],"2026-09-15")
        self.assertLess(result.summary["calibration"]["observations"],original.summary["calibration"]["observations"])

    def test_fx_missing_falls_back_to_usd_without_relabelling(self):
        result=simulate_processed(self.processed(reporting="EUR"),self.file)
        self.assertEqual(result.summary["simulation_currency"],"USD")
        self.assertTrue(result.summary["currency_fallback_used"])
        self.assertFalse(result.summary["reporting_currency_returns_computed"])

    def test_fx_present_is_jointly_simulated_with_correct_inverse_sign(self):
        p=self.processed(reporting="EUR"); self.add_fx(p)
        result=simulate_processed(p,self.file)
        self.assertEqual(result.summary["simulation_currency"],"EUR")
        self.assertFalse(result.summary["currency_fallback_used"])
        self.assertIn("fx::EURUSD",result.tables["calibration_parameters"]["factor"].tolist())
        self.assertTrue(result.summary["reporting_currency_returns_computed"])
        # The FX path is deterministic at +0.001 log/day in USD per EUR.
        native=simulate_processed(self.processed(),self.file)
        a=native.tables["risk_summary"].query("model == 'gaussian' and instrument_id == 'A'").iloc[0]
        b=result.tables["risk_summary"].query("model == 'gaussian' and instrument_id == 'A'").iloc[0]
        self.assertAlmostEqual(b["gaussian_expected_terminal"],
                               a["gaussian_expected_terminal"]*np.exp(-.001*result.summary["steps"]),places=8)
        self.add_fx(p,inverse=True)
        inverse=simulate_processed(p,self.file)
        row=inverse.tables["risk_summary"].query("model == 'gaussian' and instrument_id == 'A'").iloc[0]
        self.assertAlmostEqual(row["gaussian_expected_terminal"],b["gaussian_expected_terminal"],places=8)

    def test_missing_fx_does_not_allow_unconverted_mixed_currency_portfolio(self):
        p=self.processed(reporting="EUR")
        p.tables["selected_prices"].loc[p.tables["selected_prices"]["instrument_id"]=="B","currency"]="GBP"
        result=simulate_processed(p,self.file)
        self.assertEqual(result.summary["status"],"unavailable")
        self.assertFalse(result.summary["simulation_performed"])

    def test_fx_gaps_units_and_insufficient_overlap_trigger_fallback(self):
        for problem in ("gap","unit","overlap"):
            p=self.processed(reporting="EUR"); self.add_fx(p)
            refs=p.tables["selected_references"]
            if problem=="gap":
                p.tables["selected_references"]=refs.iloc[::3].copy()
            elif problem=="unit":
                refs["unit"]="GBP per USD"
            else:
                p.tables["selected_references"]=refs.tail(3).copy()
            self.assertEqual(simulate_processed(p,self.file).summary["simulation_currency"],"USD")

    def test_missing_asset_is_not_dropped_or_weights_renormalized(self):
        p=self.processed()
        p.tables["selected_prices"]=p.tables["selected_prices"].query("instrument_id == 'A'")
        result=simulate_processed(p,self.file)
        self.assertEqual(result.summary["status"],"unavailable")
        self.assertIn("B",result.summary["warnings"][-1])

    def test_truncated_t_preserves_covariance_and_enforces_declared_radius(self):
        covariance=np.array([[.0001,-.00002],[-.00002,.00004]])
        root=_root_covariance(covariance)
        params=_truncation(2,5,.0005); params["df"]=5
        shocks=_draw_shocks(np.random.default_rng(55),120000,root,params)
        np.testing.assert_allclose(np.cov(shocks,rowvar=False),covariance,rtol=.04,atol=2e-7)
        self.assertLess(np.max(np.abs(shocks.mean(axis=0))),.0001)
        whitened=shocks @ np.linalg.pinv(root.T)
        self.assertLessEqual(np.max(np.linalg.norm(whitened,axis=1)),params["effective_radial_cutoff"]+1e-10)
        self.assertGreater(params["retained_probability"],0)
        self.assertLess(params["retained_probability"],1)

    def test_singular_and_zero_covariances_have_no_invented_risk(self):
        for covariance in (np.zeros((2,2)),np.array([[1.,1.],[1.,1.]])):
            root=_root_covariance(covariance)
            shocks=_draw_shocks(np.random.default_rng(1),100,root)
            np.testing.assert_allclose(shocks[:,0],shocks[:,1])
        with self.assertRaises(ValueError):
            _root_covariance(np.array([[1,2],[2,1]]))

    def test_joint_df_fit_and_cutoff_sensitivity_are_explicit(self):
        self.cfg["student_t"]["sensitivity_tail_probabilities"]=[.002,.00005]; self.write_config()
        result=simulate_processed(self.processed(),self.file)
        self.assertEqual(len(result.summary["models"]),4)
        primary=result.summary["models"]["student_t_truncated"]
        self.assertGreater(primary["df"],2)
        self.assertEqual(primary["df_method"],"covariance_constrained_untruncated_likelihood")
        for model in ("student_t_tail_0.002","student_t_tail_5e-05"):
            self.assertEqual(primary["df"],result.summary["models"][model]["df"])

    def test_exact_fractional_tail_expected_shortfall(self):
        losses=np.array([.1,.2,.3,.4,.5])
        self.assertAlmostEqual(_expected_shortfall(losses,.7),(.5+.5*.4)/1.5)

    def test_seed_reproducibility_and_model_order_independence(self):
        p=self.processed()
        one=simulate_processed(p,self.file)
        two=simulate_processed(p,self.file)
        self.assertEqual(one.run_id,two.run_id)
        pd.testing.assert_frame_equal(one.tables["terminal_values"],two.tables["terminal_values"])
        self.cfg["models"].reverse(); self.write_config()
        three=simulate_processed(p,self.file)
        pd.testing.assert_frame_equal(one.tables["terminal_values"],three.tables["terminal_values"])
        self.cfg["seed"]+=1; self.write_config()
        four=simulate_processed(p,self.file)
        self.assertFalse(np.array_equal(one.tables["terminal_values"]["terminal_value"],four.tables["terminal_values"]["terminal_value"]))

    def test_invalid_config_and_disabled_mode(self):
        for name,value in (("paths",True),("seed",-1),("models",["local_vol"]),("initial_value",float("inf"))):
            original=deepcopy(self.cfg); self.cfg[name]=value; self.write_config()
            with self.assertRaises(ValueError): load_monte_carlo_config(self.file)
            self.cfg=original
        self.cfg["enabled"]=False; self.write_config()
        result=simulate_processed(self.processed(),self.file)
        self.assertEqual(result.summary["status"],"disabled")
        self.assertFalse(result.summary["simulation_performed"])

    def test_saved_reader_exports_hashes_and_keeps_sources_unchanged(self):
        p=self.processed(reporting="EUR")
        source=self.root/"processed"; source.mkdir()
        entries={}
        for name,table in p.tables.items():
            file=source/(name+".csv"); table.to_csv(file,index=False)
            entries[name]=dict(file=file.name,rows=len(table),sha256=hashlib.sha256(file.read_bytes()).hexdigest())
        manifest={**p.summary,"complete":True,"archive":str(p.archive),"cob":p.cob,"run_id":p.run_id,"files":entries}
        (source/"manifest.json").write_text(json.dumps(manifest),encoding="utf-8")
        before={f:(f.read_bytes(),f.stat().st_mtime_ns) for f in source.iterdir()}
        self.cfg.update(plots=True,paths=200); self.write_config()
        with patch.object(socket.socket,"connect",side_effect=AssertionError("No API access")):
            result=simulate_saved_processing(source,self.file)
            output=result.save(source/"monte_carlo"/result.run_id)
        self.assertEqual(before,{f:(f.read_bytes(),f.stat().st_mtime_ns) for f in before})
        saved=json.loads((output/"manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(saved["complete"]); self.assertTrue(saved["simulation_performed"])
        self.assertFalse(saved["option_calibration_performed"])
        for entry in saved["files"].values():
            self.assertEqual(hashlib.sha256((output/entry["file"]).read_bytes()).hexdigest(),entry["sha256"])
        self.assertEqual(len(list((output/"plots").glob("*.png"))),2)
        with self.assertRaises(ValueError): result.save(source)
        with self.assertRaises(ValueError): result.save(self.raw/"output")
        selected=source/"selected_prices.csv"
        selected.write_bytes(selected.read_bytes()+b"\n")
        with self.assertRaisesRegex(ValueError,"checksum"):
            simulate_saved_processing(source,self.file)

    def test_df_fit_failure_is_a_reported_unavailable_result(self):
        with patch("src.monte_carlo._student_parameters",side_effect=SimulationUnavailable("fit failed")):
            result=simulate_processed(self.processed(),self.file)
        self.assertEqual(result.summary["status"],"unavailable")
        self.assertFalse(result.summary["simulation_performed"])
        self.assertIn("fit failed",result.summary["warnings"])

    def test_cob_is_part_of_identity_and_inputs_are_preserved(self):
        p=self.processed()
        prices=p.tables["selected_prices"].copy(deep=True)
        portfolio=deepcopy(p.summary["portfolio"])
        one=simulate_processed(p,self.file)
        pd.testing.assert_frame_equal(prices,p.tables["selected_prices"])
        self.assertEqual(portfolio,p.summary["portfolio"])
        # Direct callers can reuse a processing label. COB still changes the
        # calibration cutoff and must never collide in the simulation archive.
        p.cob="2026-10-07"
        two=simulate_processed(p,self.file)
        self.assertNotEqual(one.run_id,two.run_id)
        self.assertEqual(two.summary["calibration"]["last_end"],"2026-10-07")

    def test_only_fitted_df_and_probability_inputs_are_accepted(self):
        original=deepcopy(self.cfg)
        changes=[("fit_df",False),("fit_df",1),("df",5),("radial_cutoff",8),
                 ("sensitivity_cutoffs",[6,12]),("excluded_tail_probability",0),
                 ("excluded_tail_probability",1),("excluded_tail_probability",True),
                 ("excluded_tail_probability",float("nan")),("excluded_tail_probability",1e-30),
                 ("sensitivity_tail_probabilities",[.0005]),
                 ("sensitivity_tail_probabilities",[.002,.002]),("df_bounds",[2,50])]
        for name,value in changes:
            with self.subTest(name=name,value=value):
                self.cfg=deepcopy(original)
                self.cfg["student_t"][name]=value; self.write_config()
                with self.assertRaises(ValueError):
                    load_monte_carlo_config(self.file)
        self.cfg=deepcopy(original); self.cfg["schema_version"]=1; self.write_config()
        with self.assertRaisesRegex(ValueError,"schema_version"):
            load_monte_carlo_config(self.file)
        self.cfg=original; self.write_config()
        loaded=load_monte_carlo_config(self.file)
        self.assertNotIn("df",loaded["student_t"])
        self.assertNotIn("radial_cutoff",loaded["student_t"])

    def test_dynamic_radii_match_probability_across_dimensions_and_df(self):
        for dimension in (1,2,5,20):
            for df in (2.1,5,30):
                for target in (.00005,.0005,.002,.25):
                    with self.subTest(dimension=dimension,df=df,target=target):
                        params=_truncation(dimension,df,target)
                        scale=dimension*(df-2)/df
                        excluded=f_distribution.sf(params["radial_cutoff"]**2/scale,dimension,df)
                        self.assertAlmostEqual(excluded/target,1,places=7)
                        self.assertEqual(params["effective_dimensions"],dimension)
                        self.assertGreater(params["variance_before_rescaling"],0)
                        self.assertLess(params["variance_before_rescaling"],1)
        self.assertGreater(_truncation(5,5,.0005)["radial_cutoff"],
                           _truncation(2,5,.0005)["radial_cutoff"])
        self.assertNotEqual(_truncation(2,5,.0005)["radial_cutoff"],
                            _truncation(2,15,.0005)["radial_cutoff"])
        self.assertGreater(_truncation(2,5,.00005)["radial_cutoff"],
                           _truncation(2,5,.002)["radial_cutoff"])

    def test_conditional_variance_agrees_with_independent_numerical_integration(self):
        for dimension,df,target in ((1,3,.002),(2,5,.0005),(5,8,.01),(12,30,.002)):
            params=_truncation(dimension,df,target)
            scale=dimension*(df-2)/df
            integral,_=quad(lambda x:x*f_distribution.pdf(x,dimension,df),
                            0,params["upper_f"],epsabs=1e-10)
            expected=scale*integral/(dimension*params["retained_probability"])
            self.assertAlmostEqual(params["variance_before_rescaling"],expected,places=8)

    def test_zero_and_singular_factor_dimensions_are_reported(self):
        centered=np.zeros((100,3))
        fitted=_student_parameters(centered,_root_covariance(np.zeros((3,3))),self.cfg["student_t"])
        self.assertIsNone(fitted["df"])
        params=_truncation(fitted["rank"],fitted["df"],.0005)
        self.assertFalse(params["truncation_applied"])
        self.assertEqual(params["actual_excluded_probability"],0)
        self.assertEqual(params["effective_dimensions"],0)
        np.testing.assert_array_equal(_draw_shocks(np.random.default_rng(3),10,np.zeros((3,0)),params),0)
        # Two perfectly dependent factors have one stochastic dimension.
        cov=np.array([[.0001,.0002],[.0002,.0004]])
        root=_root_covariance(cov)
        params=_truncation(root.shape[1],5,.0005); params["df"]=5
        self.assertEqual(params["effective_dimensions"],1)
        samples=_draw_shocks(np.random.default_rng(9),30000,root,params)
        np.testing.assert_allclose(samples[:,1],2*samples[:,0])
        np.testing.assert_allclose(np.cov(samples,rowvar=False),cov,rtol=.07)

    def test_df_fit_failure_has_no_configured_fallback(self):
        from types import SimpleNamespace
        with patch("src.monte_carlo.minimize_scalar",return_value=SimpleNamespace(success=False,fun=1.)):
            result=simulate_processed(self.processed(),self.file)
        self.assertEqual(result.summary["status"],"unavailable")
        self.assertFalse(result.summary["simulation_performed"])
        self.assertTrue(any("fit failed" in w for w in result.summary["warnings"]))

    def test_new_holding_and_weights_have_the_expected_calibration_effect(self):
        self.cfg.update(paths=100,sample_paths=0); self.write_config()
        p=self.processed()
        original=simulate_processed(p,self.file)
        p.summary["portfolio"]["holdings"][0]["weight"]=.7
        p.summary["portfolio"]["holdings"][1]["weight"]=.3
        weighted=simulate_processed(p,self.file)
        pd.testing.assert_frame_equal(original.tables["calibration_covariance"],weighted.tables["calibration_covariance"])
        self.assertEqual(original.summary["models"]["student_t_truncated"],
                         weighted.summary["models"]["student_t_truncated"])
        rng=np.random.default_rng(51)
        p.tables["selected_prices"]=pd.concat([p.tables["selected_prices"],
            self.prices("C",rng.normal(0,.007,len(self.dates)-1))],ignore_index=True)
        p.summary["portfolio"]["holdings"]=[{"request_id":k,"weight":w} for k,w in (("A",.4),("B",.3),("C",.3))]
        expanded=simulate_processed(p,self.file)
        fitted=expanded.summary["models"]["student_t_truncated"]
        self.assertEqual(fitted["effective_dimensions"],3)
        self.assertEqual(fitted["excluded_tail_probability"],.0005)
        self.assertNotEqual(fitted["radial_cutoff"],original.summary["models"]["student_t_truncated"]["radial_cutoff"])
        self.assertEqual(len(expanded.tables["calibration_covariance"]),9)

    def test_five_to_ten_years_refits_df_and_radius(self):
        self.cfg.update(paths=50,sample_paths=0)
        self.cfg["calibration"].update(start=None,lookback_years=5,minimum_observations=252)
        self.write_config()
        self.dates=_sessions("XNYS","2016-10-08","2026-10-08")
        n=len(self.dates)-1
        rng=np.random.default_rng(161)
        dfs=np.where(np.array(self.dates[1:])<"2021-10-08",2.5,30.)
        logs=rng.normal(size=(n,2))*np.sqrt((dfs-2)/rng.chisquare(dfs))[:,None]*[.01,.005]
        p=self.processed()
        p.tables["selected_prices"]=pd.concat([self.prices("A",logs[:,0]),self.prices("B",logs[:,1])])
        five=simulate_processed(p,self.file)
        self.cfg["calibration"]["lookback_years"]=10; self.write_config()
        ten=simulate_processed(p,self.file)
        self.assertGreater(ten.summary["calibration"]["observations"],five.summary["calibration"]["observations"])
        a=five.summary["models"]["student_t_truncated"]
        b=ten.summary["models"]["student_t_truncated"]
        self.assertGreater(abs(a["df"]-b["df"]),.1)
        self.assertNotEqual(a["radial_cutoff"],b["radial_cutoff"])
        self.assertEqual(a["excluded_tail_probability"],b["excluded_tail_probability"])
        self.assertNotEqual(five.run_id,ten.run_id)
