
"""Offline financial invariants and end-to-end archive preservation."""
from copy import deepcopy
from contextlib import redirect_stdout
from datetime import datetime, timezone
from io import StringIO
import json
from pathlib import Path
import socket
import unittest
from unittest.mock import patch

from src.data_cleaning import (load_cleaning_config, clean_prices, clean_options,
    clean_auxiliary, process_archive, _calendar)

ROOT = Path(__file__).resolve().parents[1]


def price(day, value, provider="yahoo", *, adjusted=True, **extra):
    return dict(instrument_id="CUSTOM_ETF", provider=provider, date=day,
                timestamp=None, source_file=provider+".json", currency="USD", unit="USD/share",
                data_type="etf_price", frequency="1d", price_basis="split_adjusted_close",
                adjusted_close_basis="split_and_dividend_adjusted" if adjusted else None,
                open=value, high=value, low=value, close=value, adj_close=value if adjusted else None,
                volume=0, requested_start="2026-10-05", requested_end="2026-10-09",
                eligible=True, reasons=[], **extra)


def option(**extra):
    row = dict(instrument_id="CUSTOM_CALL", provider="ibkr", timestamp="2026-10-09T15:00:00+00:00",
        source_file="quote.json", date="2026-10-09", currency="USD", eligible=True, reasons=[],
        contract=dict(underlying="CUSTOM_INDEX", expiration="2026-10-16", strike=100,
                      option_type="call", exercise_style="european", settlement="cash",
                      settlement_time="AM", multiplier=100), bid=9.9, ask=10.1,
        **{"modelGreeks.impliedVol":0.2, "modelGreeks.undPrice":101,
           "modelGreeks.optPrice":9.8, "modelGreeks.pvDividend":0, "modelGreeks.vega":None})
    row.update(extra)
    return row


class CleaningTests(unittest.TestCase):
    def setUp(self):
        self.policy = load_cleaning_config(ROOT/"config/cleaning.json")
        self.policy["instruments"] = {}

    def test_adjusted_return_does_not_double_count_cash_distribution(self):
        a, b = price("2026-10-05",100), price("2026-10-06",99)
        a["adj_close"], b["adj_close"] = 99, 99
        candidates, selected, *_ = clean_prices([a,b], self.policy, [])
        self.assertEqual(selected[1]["simple_return"], 0)
        self.assertEqual(selected[1]["close"], 99)
        self.assertEqual(selected[0]["return_basis"], "split_and_dividend_adjusted_close")
        self.assertEqual(len(candidates),2)

    def test_each_adjusted_row_requires_its_own_known_convention(self):
        for allow_price_only in (False, True):
            for convention in (None, "unknown", "split_only"):
                with self.subTest(allow_price_only=allow_price_only, convention=convention):
                    rows = [price("2026-10-05",100), price("2026-10-06",101),
                            price("2026-10-07",102), price("2026-10-08",103)]
                    rows[1]["adjusted_close_basis"] = convention
                    self.policy["prices"]["allow_price_only"] = allow_price_only
                    provider, selected, *_ = clean_prices(rows, self.policy, [])
                    self.assertFalse(provider[1]["eligible"])
                    self.assertIn("incompatible_adjusted_close_basis", provider[1]["reasons"])
                    self.assertNotIn("return_basis", provider[1])
                    self.assertIsNone(selected[1]["analysis_price"])
                    self.assertIsNone(selected[2]["simple_return"])
                    self.assertIsNone(selected[2]["return_start_date"])
                    self.assertAlmostEqual(selected[3]["simple_return"], 103/102-1)
                    self.assertEqual(selected[3]["return_start_date"], "2026-10-07")

    def test_conflicting_currency_cannot_be_primary_or_fill_gaps(self):
        rows = [price("2026-10-05",100, source_currency="EUR"),
                price("2026-10-06",101, source_currency="EUR"),
                price("2026-10-05",100,"ibkr", source_currency="USD")]
        issues = []
        provider, selected, _, selections, _ = clean_prices(rows, self.policy, issues)
        self.assertEqual(selections[0]["provider"], "ibkr")
        self.assertFalse(provider[0]["eligible"])
        self.assertEqual(provider[0]["source_currency"], "EUR")
        self.assertEqual(provider[0]["currency"], "USD")
        self.assertIsNone(selected[1]["analysis_price"])
        self.assertTrue(all(r["code"] == "currency_mismatch" for r in issues))
        self.assertEqual(len(issues), 2)

    def test_absent_matching_and_subunit_currency_metadata(self):
        for source, configured, accepted in [
                (None, "USD", True), ("USD", "USD", True),
                ("GBp", "GBP", False), ("GBp", "GBp", True)]:
            with self.subTest(source=source, configured=configured):
                row = price("2026-10-05",100, source_currency=source)
                row["currency"] = configured
                provider, selected, *_ = clean_prices([row], self.policy, [])
                self.assertEqual(provider[0]["eligible"], accepted)
                self.assertEqual(bool(selected), accepted)

    def test_resolved_contract_currency_conflict_is_preserved_during_extraction(self):
        from src.data_cleaning import _extract
        from types import SimpleNamespace
        item = dict(provider="ibkr", request_id="CALL", ticker="CALL",
                    metadata={"currency":"USD", "resolved_contract":{"currency":"EUR"}},
                    saved_status="response_saved", requested_start=None, requested_end=None,
                    is_option=True, price_bases=[],
                    response_files=[{"file":"quote.json", "role":"observations", "checksum_matches":True}])
        raw = dict(source="quote.json", date="2026-10-09", timestamp=None, values={"bid":1,"ask":2})
        inventory = SimpleNamespace(requests=[item], observations={
            ("ibkr","CALL"):{"snapshot":[raw]}})
        issues = []
        _, _, options, *_ = _extract(inventory, issues)
        self.assertFalse(options[0]["eligible"])
        self.assertEqual(options[0]["resolved_currency"], "EUR")
        self.assertEqual(options[0]["currency"], "USD")
        self.assertIn("currency_mismatch", options[0]["reasons"])
        self.assertEqual(issues[0]["code"], "currency_mismatch")

    def test_no_adjusted_scale_splicing_or_gap_returns(self):
        rows=[price("2026-10-05",100),price("2026-10-07",102),
              price("2026-10-05",200,"ibkr"),price("2026-10-06",202,"ibkr"),
              price("2026-10-07",204,"ibkr")]
        _, merged, *_=clean_prices(rows,self.policy,[])
        self.assertIsNone(merged[1]["analysis_price"])
        self.assertIsNone(merged[2]["simple_return"])
        self.assertFalse(any(r["gap_filled"] for r in merged))

    def test_verified_same_basis_gap_fill_and_no_averaging(self):
        rows=[price("2026-10-05",100,adjusted=False),price("2026-10-07",102,adjusted=False)]
        rows += [price(d,v,"ibkr",adjusted=False) for d,v in
                 [("2026-10-05",100),("2026-10-06",101),("2026-10-07",102)]]
        _, merged, comparisons, *_=clean_prices(rows,self.policy,[])
        self.assertEqual(merged[1]["analysis_price"],101)
        self.assertTrue(merged[1]["gap_filled"])
        self.assertAlmostEqual(merged[2]["simple_return"],102/101-1)
        self.assertEqual(len(comparisons),2)
        # A discrepancy blocks this fallback; it never averages 100 and 110.
        rows[-3]["close"]=110
        rows[-3]["open"]=rows[-3]["low"]=rows[-3]["high"]=110
        _, merged, comparisons, *_=clean_prices(rows,self.policy,[])
        self.assertIsNone(merged[1]["analysis_price"])
        self.assertEqual(merged[0]["analysis_price"],100)
        self.assertTrue(any(r["exceeds_tolerance"] for r in comparisons))

    def test_identical_duplicates_collapse_conflicts_quarantine(self):
        a=price("2026-10-05",100)
        rows=[a,deepcopy(a),price("2026-10-06",101),price("2026-10-06",102)]
        issues=[]
        candidates,merged,*_=clean_prices(rows,self.policy,issues)
        self.assertEqual(len(candidates),3)
        self.assertEqual(candidates[0]["duplicate_count"],1)
        self.assertIsNone(merged[1]["analysis_price"])
        self.assertEqual(sum(not r["eligible"] for r in candidates),2)
        self.assertIn("conflicting_duplicates",[i["code"] for i in issues])

    def test_exchange_holidays_exceptional_closures_and_early_close(self):
        rows=_calendar("XNYS","2001-09-10","2001-09-17",{})
        self.assertIn("2001-09-10",rows)
        self.assertNotIn("2001-09-11",rows)
        self.assertNotIn("2001-09-14",rows)
        self.assertIn("2001-09-17",rows)
        rows=_calendar("XNYS","2026-04-01","2026-11-30",{})
        self.assertNotIn("2026-04-03",rows)
        self.assertTrue(rows["2026-11-27"]["close_utc"].startswith("2026-11-27T18:00:00"))

    def test_friday_monday_is_adjacent_session_but_outside_dates_are_excluded(self):
        a,b=price("2026-10-02",100),price("2026-10-05",101)
        for r in (a,b):r["requested_start"]="2026-10-02"
        outside=price("2026-10-10",90)
        candidates,merged,*_=clean_prices([a,b,outside],self.policy,[])
        self.assertAlmostEqual(merged[1]["simple_return"],.01)
        self.assertIn("outside_requested_window",candidates[-1]["reasons"])

    def test_adjusted_close_delivered_in_close_field_remains_total_return_basis(self):
        rows=[price("2026-10-05",100,"marketdata",adjusted=False),
              price("2026-10-06",101,"marketdata",adjusted=False)]
        for r in rows:r["price_basis"]="split_and_dividend_adjusted_close"
        self.policy["prices"]["allow_price_only"]=False
        _,merged,_,selection,_=clean_prices(rows,self.policy,[])
        self.assertEqual(selection[0]["return_basis"],"split_and_dividend_adjusted_close")
        self.assertAlmostEqual(merged[1]["simple_return"],.01)

    def test_no_options_does_not_block_price_processing(self):
        self.assertEqual(clean_options([],self.policy,[]),([],[]))
        self.policy["prices"]["allow_price_only"]=False
        _,merged,*_=clean_prices([price("2026-10-05",100,adjusted=False)],self.policy,[])
        self.assertEqual(merged,[])

    def test_option_model_group_kept_separate_and_zero_dividend_is_valid(self):
        rows,_=clean_options([option()],self.policy,[])
        r=rows[0]
        self.assertTrue(r["iv_candidate"])
        self.assertEqual(r["midpoint"],10)
        self.assertEqual(r["modelGreeks.optPrice"],9.8)
        self.assertEqual(r["modelGreeks.impliedVol"],.2)
        self.assertEqual(r["modelGreeks.pvDividend"],0)
        self.assertAlmostEqual(r["time_to_expiry"],(7*24-1.5)/(365*24))
        self.assertIn("estimate",r["expiry_time_source"])

    def test_option_missing_fields_wide_spread_expiry_and_negative_vega(self):
        cases=[
            ({"bid":-1},"missing_positive_bid_ask"),
            ({"bid":11},"crossed_market"),
            ({"ask":20},"wide_spread"),
            ({"modelGreeks.impliedVol":None},"missing_positive_modelGreeks.impliedVol"),
            ({"modelGreeks.undPrice":None},"missing_positive_modelGreeks.undPrice"),
            ({"modelGreeks.pvDividend":None},"missing_nonnegative_dividend_pv"),
            ({"modelGreeks.vega":-1e-12},"low_or_negative_model_vega"),
            ({"timestamp":"2026-10-16T15:00:00+00:00"},"expired_at_capture")]
        for changes, reason in cases:
            with self.subTest(reason=reason):
                rows,_=clean_options([option(**changes)],self.policy,[])
                self.assertFalse(rows[0]["iv_candidate"])
                self.assertIn(reason,rows[0]["iv_reasons"])

    def test_contract_expiry_override_takes_precedence_over_underlying_default(self):
        self.policy["instruments"] = {
            "CUSTOM_INDEX": {"expiration_time": "15:00"},
            "CUSTOM_CALL": {"expiration_time": "16:00"}}
        rows,_=clean_options([option()],self.policy,[])
        self.assertEqual(rows[0]["expiry_timestamp"], "2026-10-16T16:00:00-04:00")
        self.assertTrue(rows[0]["iv_candidate"])

    def test_american_option_is_not_a_european_surface_input(self):
        r=option();r["contract"]["exercise_style"]="american"
        rows,_=clean_options([r],self.policy,[])
        self.assertTrue(rows[0]["iv_candidate"])
        self.assertFalse(rows[0]["european_surface_candidate"])

    def test_snapshot_window_is_diagnosed_not_relabeled_to_cob(self):
        a=option()
        b=option(instrument_id="OTHER_CALL",timestamp="2026-10-09T15:10:00+00:00")
        b["contract"]["strike"]=105
        rows,spans=clean_options([a,b],self.policy,[])
        self.assertEqual(len(rows),2)
        self.assertTrue(spans[0]["exceeds_span_limit"])
        self.assertEqual(spans[0]["capture_date"],"2026-10-09")

    def test_negative_rates_valid_and_percent_conversion_explicit(self):
        base=price("2026-10-09",1)
        base.update(instrument_id="RATE",provider="fred",data_type="overnight_reference_rate",
                    unit="Percent",value=-.5)
        curve=deepcopy(base)
        curve.update(instrument_id="CURVE",provider="manual_ois",date="2026-10-10",
                     tenor="1Y",rate_unit="percent_per_annum",rate_percent=-.1)
        _,refs,selected,curves,_=clean_auxiliary([], [base], [curve], [],self.policy,[])
        self.assertTrue(refs[0]["eligible"])
        self.assertEqual(selected[0]["rate_decimal"],-.005)
        self.assertTrue(curves[0]["eligible"])
        self.assertEqual(curves[0]["date"],"2026-10-10")
        self.assertEqual(curves[0]["rate_decimal"],-.001)

    def test_windows_manual_curve_paths_match_portable_source_provenance(self):
        from src.data_cleaning import _extract
        from types import SimpleNamespace
        item=dict(provider="manual_ois",request_id="CURVE",ticker="CURVE",
                  metadata={"currency":"EUR","quote_type":"spot_starting_ois_par_swap_rate",
                            "rate_unit":"percent_per_annum"},
                  saved_status="response_saved",requested_start=None,requested_end=None,
                  is_option=False,price_bases=["par_swap_quote_not_zero_rate"],
                  response_files=[{"file":"ois/euro/curve.json","role":"observations"}])
        row=dict(source="ois"+chr(92)+"euro"+chr(92)+"curve.json", key="1Y",
                 date="2026-10-09",timestamp=None,values={"rate_percent":2})
        inventory=SimpleNamespace(requests=[item],observations={
            ("manual_ois","CURVE"):{"par_ois_quotes":[row]}})
        _,_,_,_,curves,_=_extract(inventory,[])
        self.assertEqual(len(curves),1)
        self.assertTrue(curves[0]["eligible"])
        self.assertEqual(curves[0]["source_file"],"ois/euro/curve.json")

    def test_unreadable_and_bad_checksum_observations_cannot_be_selected(self):
        from src.data_cleaning import _extract
        from types import SimpleNamespace
        item=dict(provider="yahoo",request_id="CUSTOM",ticker="CUSTOM",metadata={},
                  saved_status="response_saved",requested_start=None,requested_end=None,
                  is_option=False,price_bases=["split_adjusted_close"],
                  response_files=[{"file":"bad.json","role":"unreadable","checksum_matches":True},
                                  {"file":"corrupt.json","role":"observations","checksum_matches":False}])
        raw=lambda f:dict(source=f,date="2026-10-09",timestamp=None,values={"close":10})
        inventory=SimpleNamespace(requests=[item],observations={
            ("yahoo","CUSTOM"):{"prices":[raw("bad.json"),raw("corrupt.json")]}})
        prices,*_=_extract(inventory,[])
        self.assertEqual(len(prices),1)
        self.assertFalse(prices[0]["eligible"])
        self.assertIn("checksum_mismatch",prices[0]["reasons"])


class CleaningIntegrationTests(unittest.TestCase):
    def _process_histories(self, histories, *, calendars=None, minimum=2):
        """Build temporary archives and exercise reader -> cleaning -> readiness."""
        from test_preprocessing import InventoryTests
        f = InventoryTests(); f.setUp(); self.addCleanup(f.doCleanups)
        start = min(day for _, points in histories.values() for day, _ in points)
        for ticker, (currency, points) in histories.items():
            stamps = [int(datetime.fromisoformat(day+"T20:00:00+00:00").timestamp())
                      for day, _ in points]
            values = [v for _, v in points]
            f.add("yahoo", ticker, [{"chart":{"result":[{
                "meta":{"exchangeTimezoneName":"UTC", "currency":currency},
                "timestamp":stamps,
                "indicators":{"quote":[{"close":values}],
                              "adjclose":[{"adjclose":values}]}
            }]}}], start=start)
            f.inputs[-1].update(kind="price", currency="USD", unit="USD/share", frequency="1d")
        f.finish()
        f.write(f.archive/"portfolio.json", dict(
            schema_version=1, reporting_currency="USD",
            weight_basis="reporting_currency_market_value", nav=None,
            target_allocation={"equity":1},
            holdings=[{"request_id":key, "bucket":"equity", "weight":1/len(histories)}
                      for key in histories], reference_data=[]))
        policy = load_cleaning_config(ROOT/"config/cleaning.json")
        policy["instruments"] = {key:{"calendar":value} for key,value in (calendars or {}).items()}
        policy["prices"]["minimum_return_observations"] = minimum
        cleaning = f.root/"cleaning.json"; f.write(cleaning, policy)
        before = {p:(p.read_bytes(),p.stat().st_mtime_ns)
                  for p in f.archive.rglob("*") if p.is_file()}
        with patch.object(socket.socket,"connect",side_effect=AssertionError("No network")):
            result = process_archive(f.archive,cleaning)
        after = {p:(p.read_bytes(),p.stat().st_mtime_ns)
                 for p in f.archive.rglob("*") if p.is_file()}
        self.assertEqual(before,after)
        return result

    def test_currency_mismatch_cannot_become_ready_usd_history(self):
        points = [("2026-10-06",100), ("2026-10-07",101), ("2026-10-08",102)]
        result = self._process_histories({"CUSTOM_ETF":("EUR",points)})
        prices = result.tables["provider_prices"]
        self.assertEqual(set(prices["source_currency"]), {"EUR"})
        self.assertEqual(set(prices["currency"]), {"USD"})
        self.assertFalse(prices["eligible"].any())
        self.assertTrue(all("currency_mismatch" in reasons for reasons in prices["reasons"]))
        self.assertTrue(result.tables["selected_prices"].empty)
        self.assertFalse(result.summary["readiness"]["local_history_ready"])
        self.assertEqual(result.summary["readiness"]["missing_holdings"], ["CUSTOM_ETF"])
        self.assertIn("currency_mismatch", set(result.tables["issues"]["code"]))

    def test_different_calendars_exclude_unequal_intervals_from_readiness(self):
        us = [("2026-07-01",100), ("2026-07-02",101),
              ("2026-07-06",104), ("2026-07-07",105)]
        uk = us[:2] + [("2026-07-03",102)] + us[2:]
        result = self._process_histories({"US_ETF":("USD",us), "UK_ETF":("USD",uk)},
                                        calendars={"US_ETF":"XNYS","UK_ETF":"XLON"}, minimum=3)
        aligned = result.tables["aligned_holding_returns"]
        self.assertEqual(aligned["date"].tolist(), ["2026-07-02","2026-07-07"])
        self.assertAlmostEqual(aligned.iloc[1]["US_ETF"],105/104-1)
        self.assertAlmostEqual(aligned.iloc[1]["UK_ETF"],105/104-1)
        selected = result.tables["selected_prices"].set_index(["instrument_id","date"])
        self.assertEqual(selected.loc[("US_ETF","2026-07-06"),"return_start_date"],"2026-07-02")
        self.assertEqual(selected.loc[("UK_ETF","2026-07-06"),"return_start_date"],"2026-07-03")
        readiness = result.summary["readiness"]
        self.assertEqual(readiness["insufficient_history"], [])
        self.assertFalse(readiness["local_history_ready"])
        self.assertEqual(readiness["excluded_return_interval_mismatches"],1)
        issues = result.tables["issues"]
        mismatch = issues[issues["code"]=="unaligned_return_intervals"].iloc[0]
        self.assertEqual(mismatch["date"],"2026-07-06")
        self.assertEqual(mismatch["return_start_dates"],
                         {"US_ETF":"2026-07-02","UK_ETF":"2026-07-03"})

    def test_matching_calendar_returns_keep_original_values(self):
        points = [("2026-07-01",100), ("2026-07-02",101),
                  ("2026-07-06",104), ("2026-07-07",105)]
        result = self._process_histories({"FIRST":("USD",points), "SECOND":("USD",points)}, minimum=3)
        aligned = result.tables["aligned_holding_returns"]
        self.assertEqual(aligned["date"].tolist(), ["2026-07-02","2026-07-06","2026-07-07"])
        self.assertEqual(aligned.columns.tolist(), ["date","FIRST","SECOND"])
        for actual, expected in zip(aligned["FIRST"], [101/100-1,104/101-1,105/104-1]):
            self.assertAlmostEqual(actual,expected)
        self.assertEqual(aligned["FIRST"].tolist(),aligned["SECOND"].tolist())
        self.assertTrue(result.summary["readiness"]["local_history_ready"])
        self.assertEqual(result.summary["readiness"]["excluded_return_interval_mismatches"],0)

    def test_local_run_is_generic_offline_and_preserves_archive(self):
        import run
        from test_preprocessing import InventoryTests
        f=InventoryTests();f.setUp();self.addCleanup(f.doCleanups)
        stamps=[int(datetime(2026,10,d,20,tzinfo=timezone.utc).timestamp()) for d in (6,7,8)]
        f.add("yahoo","CUSTOM_ETF",[{"chart":{"result":[{
            "meta":{"exchangeTimezoneName":"America/New_York","currency":"USD"},
            "timestamp":stamps, "indicators":{"quote":[{"close":[100,99,101],"volume":[10,11,12]}],
                                              "adjclose":[{"adjclose":[99,99,101]}]},
            "events":{"dividends":{"a":{"date":stamps[1],"amount":1}}}
        }]}}],start="2026-10-06")
        f.inputs[0].update(kind="price",currency="USD",unit="USD/share",frequency="1d")
        f.add("alpha_vantage","EMPTY",[{"Time Series (Daily)":{}}],start="2026-10-06")
        f.inputs[-1]["kind"]="price"
        f.write(f.archive/"ois/euro/curve.json",{"usage_policy":"fixed_for_all_cobs","curve":{
            "currency":"EUR","as_of_date":"2026-10-09","quote_type":"spot_starting_ois_par_swap_rate",
            "rate_unit":"percent_per_annum","points":[{"tenor":"1Y","rate_percent":2},
                                                     {"tenor":"2Y","rate_percent":2.5}]}})
        f.rows.append(dict(provider="manual_ois",ticker="CUSTOM_OIS",request_id="CUSTOM_OIS",
                           operation="manual_curve_snapshot",folder="ois/euro",
                           status="response_saved",cob="2026-10-08"))
        f.finish()
        portfolio=dict(schema_version=1,reporting_currency="EUR",
            weight_basis="reporting_currency_market_value",nav=None,target_allocation={"equity":1},
            holdings=[{"request_id":"CUSTOM_ETF","bucket":"equity","weight":1}],reference_data=[])
        archived=deepcopy(portfolio);archived["reporting_currency"]="USD"
        f.write(f.archive/"portfolio.json",archived)
        file=f.root/"portfolio.json";f.write(file,portfolio)
        policy=load_cleaning_config(ROOT/"config/cleaning.json")
        policy["prices"]["minimum_return_observations"]=2
        cleaning=f.root/"cleaning.json";f.write(cleaning,policy)
        before={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in f.archive.rglob("*") if p.is_file()}
        with patch.object(socket.socket,"connect",side_effect=AssertionError("No network")), \
             patch.object(run.MarketDataDownloader,"from_json",side_effect=AssertionError("No APIs")), \
             patch.object(run,"RAW_DUMP_DIR",f.archive.parent), \
             patch.object(run,"INVENTORY_DIR",f.root/"reports"), \
             patch.object(run,"PROCESSED_DIR",f.root/"processed"), \
             patch.object(run,"PORTFOLIO_FILE",file), patch.object(run,"CLEANING_FILE",cleaning), \
             redirect_stdout(StringIO()):
            inv=run.main("08/10/2026",mode="process")
            processed=process_archive(f.archive,cleaning,portfolio_file=file)
        after={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in f.archive.rglob("*") if p.is_file()}
        self.assertEqual(before,after)
        self.assertIsNotNone(inv.observations)
        self.assertNotIn("observations",inv.to_dict())
        self.assertEqual(processed.summary["options"]["iv_candidates"],0)
        self.assertTrue(processed.summary["portfolio"]["changed_from_archived_snapshot"])
        self.assertEqual(processed.summary["readiness"]["aligned_return_observations"],2)
        self.assertEqual(processed.tables["aligned_holding_returns"].iloc[0]["CUSTOM_ETF"],0)
        self.assertEqual(len(processed.tables["corporate_actions"]),1)
        self.assertEqual(len(processed.tables["ois_quotes"]),2)
        self.assertEqual(processed.tables["ois_quotes"].iloc[0]["rate_decimal"],.02)
        self.assertEqual(processed.tables["ois_quotes"].iloc[0]["date"],"2026-10-09")
        status=processed.tables["request_status"].set_index("instrument_id")
        self.assertFalse(status.loc["EMPTY","has_parsed_observations"])
        destination=f.root/"processed"/inv.cob/processed.run_id
        manifest=json.loads((destination/"manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(manifest["complete"])
        self.assertEqual(manifest["run_id"],processed.run_id)
        statistics_manifests = list((destination/"statistics").glob("*/manifest.json"))
        self.assertEqual(len(statistics_manifests), 1)
        stats = json.loads(statistics_manifests[0].read_text(encoding="utf-8"))
        self.assertTrue(stats["complete"])
        self.assertFalse(stats["simulation_performed"])
        self.assertEqual(stats["analysis_currency"], "USD")
        self.assertFalse(stats["reporting_currency_returns_computed"])
        mc_manifests = list((destination/"monte_carlo").glob("*/manifest.json"))
        self.assertEqual(len(mc_manifests), 1)
        mc = json.loads(mc_manifests[0].read_text(encoding="utf-8"))
        self.assertTrue(mc["complete"])
        self.assertEqual(mc["status"], "unavailable")
        self.assertFalse(mc["simulation_performed"])

        with self.assertRaises(ValueError):
            processed.save(f.archive/"processed")
        with self.assertRaises(ValueError):
            processed.save(f.archive.parent)
