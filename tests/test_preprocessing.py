"""Offline archive fixtures test the reader, not provider/API behaviour."""
from contextlib import redirect_stdout
import hashlib
from io import StringIO
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

from src.preprocessing import inspect_archive, _timestamp

COB = "2026-10-08"


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.archive = self.root / "raw" / COB
        self.archive.mkdir(parents=True)
        self.rows, self.inputs = [], []

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def add(self, provider, ticker, bodies=(), *, status="response_saved",
            operation="price_history", params=None, contract=None, start="2000-01-03",
            attempt="attempt_001"):
        folder = self.archive / provider / ticker
        location = folder / attempt
        location.mkdir(parents=True)
        request = dict(requested_start=start, requested_end=COB, data_type="etf_price",
                       configured_currency="USD", configured_unit="USD/share",
                       contract=contract, requested_frequency="1d")
        self.write(location / "request.json", request)
        responses = []
        for i, body in enumerate(bodies, 1):
            csv_body = isinstance(body, str)
            file = f"response_{i:03d}" + (".csv" if csv_body else ".json")
            raw = body if csv_body else json.dumps(body)
            (location / file).write_bytes(raw.encode("utf-8"))
            responses.append(dict(file=file, sha256=hashlib.sha256(raw.encode()).hexdigest(),
                                  parameters=params or {}, received_at_utc="2026-10-09T10:00:00+00:00"))
        self.write(location / "responses.json", responses)
        row = dict(provider=provider, ticker=ticker, request_id=ticker, operation=operation,
                   status=status, folder=str(folder.relative_to(self.archive)),
                   attempt=str(location.relative_to(self.archive)), cob=COB,
                   requested_start=start, requested_end=COB, responses_saved=len(responses))
        self.rows.append(row)
        self.inputs.append(dict(id=ticker, symbols={provider:ticker}, data_type="etf_price"))
        return location

    def finish(self):
        self.write(self.archive / "manifest.json", dict(
            cob=COB, complete=True, request_count=len(self.rows), all_requests_succeeded=False))
        self.write(self.archive / "summary.json", self.rows)
        self.write(self.archive / "resolved_inputs.json", {"requests": self.inputs})

    def inspect(self):
        self.finish()
        with patch.object(socket.socket, "connect", side_effect=AssertionError("No network")):
            return inspect_archive(self.archive)

    def bars(self, values):
        return dict(method="reqHistoricalData", completed=True, request={
            "what_to_show":"TRADES", "bar_size":"1 day", "use_rth":True},
            bars=[dict(date=day, open=close, high=close, low=close, close=close,
                       volume=100, average=close, barCount=2) for day, close in values])

    def test_repeated_chunks_and_conflicts_are_diagnosed_not_removed(self):
        self.add("ibkr", "IEF", [self.bars([("2026-10-07",100),("2026-10-08",101)]),
                                self.bars([("2026-10-07",100),("2026-10-08",102)])],
                 status="limited_history")
        item = self.inspect().requests[0]
        table = item["datasets"]["TRADES"]
        self.assertEqual((table["row_count"],table["unique_keys"],table["duplicate_rows"],table["conflicting_keys"]),
                         (4,2,2,1))
        self.assertEqual(item["saved_status"],"limited_history")
        self.assertEqual(table["fields"]["close"]["numeric"],4)
        self.assertEqual(item["price_bases"],["split_adjusted_close"])

    def test_recorded_attempt_is_used_not_earlier_files_or_archive_identity(self):
        folder=self.add("ibkr","SPY",[self.bars([("2026-10-08",1)])],start="2010-01-01",attempt="attempt_002")
        self.rows[0]["requested_start"]="2000-01-03"
        old=folder.parent/"attempt_001"
        old.mkdir()
        self.write(old/"response_001.json",self.bars([("2001-01-02",7)]))
        item=self.inspect().requests[0]
        self.assertEqual(item["requested_start"],"2010-01-01")
        self.assertEqual(item["datasets"]["TRADES"]["row_count"],1)
        self.assertEqual(item["datasets"]["TRADES"]["first_observation"],COB)

    def test_yahoo_arrays_adjusted_close_dividends_and_missing_values(self):
        self.add("yahoo","IEF",[{"chart":{"result":[{
            "meta":{"exchangeTimezoneName":"America/New_York","symbol":"IEF"},
            "timestamp":[1791379800,1791466200],
            "indicators":{"quote":[{"close":[100,None],"volume":[0,100]}],
                          "adjclose":[{"adjclose":[99,100]}]},
            "events":{"dividends":{"x":{"date":1791379800,"amount":0.2}}}
        }]}}])
        item=self.inspect().requests[0]
        table=item["datasets"]["prices"]
        self.assertEqual(table["first_observation"],"2026-10-07")
        self.assertEqual(table["fields"]["close"]["missing"],1)
        self.assertEqual(table["fields"]["volume"]["zero"],1)
        self.assertEqual(table["fields"]["adj_close"]["numeric"],2)
        self.assertEqual(item["datasets"]["dividends"]["row_count"],1)

    def test_mismatched_arrays_and_outside_window_remain_visible(self):
        self.add("marketdata","SPY",[{"s":"ok","t":[1791466200,1791552600],
                                     "c":[1,2,3],"v":[2]}],
                 params={"adjustsplits":"true","adjustdividends":"false"})
        item=self.inspect().requests[0]
        table=item["datasets"]["prices"]
        self.assertEqual(table["row_count"],3)
        self.assertEqual(table["invalid_date_rows"],1)
        self.assertEqual(table["rows_after_requested_end"],1)
        self.assertIn("array_length_mismatch",[i["code"] for i in item["issues"]])
        self.assertEqual(item["price_bases"],["split_adjusted_close"])

    def test_marketdata_dividend_adjustment_is_preserved(self):
        self.add("marketdata","SPY",[{"s":"ok","t":[1791466200],"c":[1]}],
                 params={"adjustsplits":"true","adjustdividends":"true"})
        self.assertEqual(self.inspect().requests[0]["price_bases"],["split_and_dividend_adjusted_close"])

    def test_alpha_diagnostic_responses_not_treated_as_observations(self):
        self.add("alpha_vantage","SPY",[{"Information":"Full history unavailable"},
            {"Time Series (Daily)":{"2026-10-08":{"4. close":"100","5. volume":"7"}}}],
            status="limited_history")
        item=self.inspect().requests[0]
        self.assertEqual(item["datasets"]["prices"]["row_count"],1)
        self.assertEqual(item["response_files"][0]["role"],"diagnostic")
        self.assertNotIn("adj_close",item["datasets"]["prices"]["fields"])

    def test_massive_millisecond_timestamps_and_extra_fields(self):
        self.add("massive","SPY",[{"results":[{"t":1791432000000,"c":123,"v":4,"vw":122,"n":5}],
                                   "adjusted":True}])
        item=self.inspect().requests[0]
        self.assertEqual(item["datasets"]["prices"]["first_observation"],COB)
        self.assertEqual(item["datasets"]["prices"]["fields"]["vwap"]["numeric"],1)
        self.assertEqual(item["price_bases"],["split_adjusted_close"])

    def test_fred_missing_markers_and_negative_rates(self):
        self.add("fred","SOFR",[{"seriess":[{"id":"SOFR","units":"Percent"}]},
            {"observations":[{"date":"2026-10-07","value":"."},{"date":COB,"value":"-0.2"}],
             "realtime_start":"2026-10-09"}],operation="observations")
        item=self.inspect().requests[0]
        fields=item["datasets"]["observations"]["fields"]["value"]
        self.assertEqual((fields["numeric"],fields["missing"],fields["negative"]),(1,1,1))
        self.assertEqual(item["metadata"]["vintage_start"],"2026-10-09")

    def test_ecb_units_statuses_and_unicode(self):
        self.add("ecb","ESTR",["TIME_PERIOD,OBS_VALUE,OBS_STATUS,UNIT,TITLE\n2026-10-08,-0.5,A,PCPA,\u20acSTR\n"],
                 operation="observations")
        item=self.inspect().requests[0]
        self.assertEqual(item["metadata"]["series_definition"]["TITLE"],"\u20acSTR")
        self.assertEqual(item["datasets"]["observations"]["fields"]["value"]["negative"],1)

    def test_option_sources_sentinels_expiry_and_last_trading_date(self):
        contract={"underlying":"SPX","expiration":"2026-10-16","strike":7800,"settlement_time":"AM"}
        model={"impliedVol":0.2,"optPrice":10,"undPrice":7800,"delta":-1,"gamma":0.01,
               "vega":None,"theta":-2,"pvDividend":0}
        self.add("ibkr","CALL",[{"ticker":{"time":"2026-10-09T10:00:00+00:00",
            "bid":9,"ask":11,"last":-1,"volume":-1,"modelGreeks":model,"bidGreeks":{"impliedVol":0.19}},
            "resolved_contract":{"lastTradeDateOrContractMonth":"20261015"},
            "contract_details":[{"realExpirationDate":"20261016"}],"snapshot_completed":True}],
            operation="option_model_snapshot",contract=contract,status="partial_model_data")
        inv=self.inspect()
        item=inv.requests[0]
        table=item["datasets"]["snapshot"]
        self.assertEqual(table["fields"]["last"]["unavailable"],1)
        self.assertEqual(table["fields"]["modelGreeks.delta"]["numeric"],1)
        self.assertEqual(table["fields"]["modelGreeks.vega"]["missing"],1)
        self.assertEqual(table["quality_counts"]["positive_model_iv"],1)
        self.assertEqual(table["quality_counts"]["positive_ordered_bid_ask"],1)
        self.assertNotIn("all_numeric_model_fields",table["quality_counts"])
        self.assertEqual(item["metadata"]["resolved_contract"]["lastTradeDateOrContractMonth"],"20261015")
        self.assertEqual(inv.option_expiries[0]["expiration"],"2026-10-16")
        self.assertEqual(inv.option_expiries[0]["contracts_with_positive_model_iv"],1)
        self.assertIn("observations_after_cob",[i["code"] for i in item["issues"]])

    def test_historical_option_channels_stay_separate(self):
        body=self.bars([("2026-10-08T14:00:00+00:00",3)])
        ask=self.bars([("2026-10-08T14:00:00+00:00",4)])
        ask["request"]["what_to_show"]="ASK"
        self.add("ibkr","CALL",[body,ask],operation="option_history",
                 contract={"underlying":"SPX","expiration":"2026-10-16","strike":7800})
        item=self.inspect().requests[0]
        self.assertEqual(set(item["datasets"]),{"TRADES","ASK"})
        self.assertNotIn("snapshot",item["datasets"])

    def test_manual_curve_date_tenors_and_negative_par_quotes(self):
        folder=self.archive/"ois"/"euro"
        self.write(folder/"curve.json",{"usage_policy":"fixed_for_all_cobs","curve":{
            "currency":"EUR","as_of_date":"2026-10-09","quote_type":"spot_starting_ois_par_swap_rate",
            "rate_unit":"percent_per_annum","points":[{"tenor":"1Y","rate_percent":-0.1},
                                                    {"tenor":"2Y","rate_percent":0.5}]}})
        self.rows.append(dict(provider="manual_ois",ticker="EUR_ESTR_OIS",request_id="EUR_ESTR_OIS",
                              operation="manual_curve_snapshot",folder="ois/euro",status="response_saved",cob=COB))
        item=self.inspect().requests[0]
        table=item["datasets"]["par_ois_quotes"]
        self.assertEqual((table["unique_keys"],table["unique_dates"]),(2,1))
        self.assertEqual(table["first_observation"],"2026-10-09")
        self.assertEqual(table["fields"]["rate_percent"]["negative"],1)

    def test_failed_request_is_retained(self):
        self.add("ecb","EURUSD",[],operation="observations",status="network_error")
        item=self.inspect().requests[0]
        self.assertEqual(item["saved_status"],"network_error")
        self.assertEqual(item["datasets"],{})
        self.assertIn("no_observations",[x["code"] for x in item["issues"]])

    def test_checksum_failure_is_explicit_and_not_parsed(self):
        folder=self.add("ibkr","IEF",[self.bars([(COB,100)])])
        self.write(folder/"response_001.json",self.bars([(COB,999)]))
        inv=self.inspect()
        self.assertEqual(inv.metadata["requests_with_reader_errors"],1)
        self.assertEqual(inv.requests[0]["datasets"],{})
        self.assertIn("checksum_mismatch",[x["code"] for x in inv.requests[0]["issues"]])

    def test_missing_and_unrecognised_files_do_not_hide_other_requests(self):
        folder=self.add("yahoo","IEF",[{"unrecognised":True}])
        self.add("fred","SOFR",[{"observations":[{"date":COB,"value":"1"}]}],operation="observations")
        (folder/"response_001.json").unlink()
        inv=self.inspect()
        self.assertEqual(len(inv.requests),2)
        self.assertEqual(inv.metadata["requests_with_reader_errors"],1)
        self.assertEqual(inv.requests[1]["datasets"]["observations"]["row_count"],1)

    def test_archive_unchanged_and_reports_outside_it(self):
        self.add("ibkr","IEF",[self.bars([(COB,100)])])
        self.finish()
        def state():
            return {p.relative_to(self.archive).as_posix():(p.read_bytes(),p.stat().st_mtime_ns)
                    for p in self.archive.rglob("*") if p.is_file()}
        before=state()
        inv=inspect_archive(self.archive)
        files=inv.save(self.root/"reports"/COB)
        self.assertTrue(all(p.exists() for p in files))
        self.assertEqual(state(),before)
        with self.assertRaises(ValueError):
            inv.save(self.archive/"reports")

    def test_path_escape_rejected_and_safe_requests_still_inspected(self):
        self.add("ibkr","IEF",[])
        self.rows[0]["attempt"]="../outside"
        item=self.inspect().requests[0]
        self.assertIn("request_read_error",[x["code"] for x in item["issues"]])

    def test_archive_must_exist_be_completed_and_idle(self):
        with self.assertRaises(ValueError):
            inspect_archive(self.root/"missing")
        self.finish()
        (self.archive/".running").write_text("running")
        with self.assertRaises(ValueError):
            inspect_archive(self.archive)
        (self.archive/".running").unlink()
        self.write(self.archive/"manifest.json",{"cob":COB,"complete":False,"request_count":0})
        with self.assertRaises(ValueError):
            inspect_archive(self.archive)

    def test_pre_1970_timestamp_works_without_windows_fromtimestamp(self):
        self.assertEqual(_timestamp(-86400,"UTC")[:10],"1969-12-31")

    def test_inventory_choice_reads_only_archive_and_does_not_prompt_history(self):
        import run
        self.add("ibkr","IEF",[self.bars([(COB,100)])])
        self.finish()
        with patch.object(run,"RAW_DUMP_DIR",self.archive.parent), \
             patch.object(run,"INVENTORY_DIR",self.root/"reports"), \
             patch.object(run,"_process_inventory") as process, \
             patch.object(run.MarketDataDownloader,"from_json",side_effect=AssertionError("No downloader")), \
             patch("builtins.input",side_effect=["1","08/10/2026"]) as prompt, \
             redirect_stdout(StringIO()):
            inv=run.main()
        self.assertEqual(len(inv.requests),1)
        self.assertEqual(prompt.call_count,2)

    def test_inventory_mode_missing_archive_never_falls_back_to_download(self):
        import run
        with patch.object(run,"RAW_DUMP_DIR",self.root/"missing"), \
             patch.object(run.MarketDataDownloader,"from_json",side_effect=AssertionError("No downloader")), \
             redirect_stdout(StringIO()), self.assertRaises(ValueError):
            run.main("08/10/2026",mode="inventory")



    def test_negative_vega_is_numeric_but_flagged_separately(self):
        model = dict(impliedVol=1.02, optPrice=5394, undPrice=7795, delta=1,
                     gamma=3.7e-19, vega=-9.09e-13, theta=-0.28, pvDividend=1.95)
        self.add("ibkr", "CALL", [{"ticker":{"time":"2026-10-09T10:00:00+00:00",
                 "modelGreeks":model}}], operation="option_model_snapshot",
                 contract={"underlying":"SPX","expiration":"2026-10-16","strike":2400},
                 status="partial_model_data")
        inventory=self.inspect()
        item=inventory.requests[0]
        self.assertIn("negative_model_vega",[i["code"] for i in item["issues"]])
        self.assertEqual(item["datasets"]["snapshot"]["fields"]["modelGreeks.vega"]["numeric"],1)
        self.assertEqual(inventory.option_expiries[0]["contracts_with_negative_model_vega"],1)
        self.assertEqual(item["saved_status"],"partial_model_data")

    def test_parallel_and_sequential_inventory_results_match(self):
        self.add("ibkr","IEF",[self.bars([(COB,100)])])
        self.add("fred","SOFR",[{"observations":[{"date":COB,"value":"1"}]}],operation="observations")
        self.finish()
        a=inspect_archive(self.archive,max_workers=1)
        b=inspect_archive(self.archive,max_workers=4)
        self.assertEqual(a.requests,b.requests)
        self.assertEqual(a.option_expiries,b.option_expiries)
        self.assertEqual([x["ticker"] for x in b.requests],["IEF","SOFR"])

    def test_unknown_schema_and_malformed_body_are_explicit(self):
        a=self.add("yahoo","UNKNOWN",[{"unrecognised":True}])
        b=self.add("yahoo","BROKEN",[{"chart":{"result":[{"indicators":None}]}}])
        inventory=self.inspect()
        self.assertEqual(inventory.metadata["requests_with_reader_errors"],2)
        self.assertIn("unrecognised_response",[i["code"] for i in inventory.requests[0]["issues"]])
        self.assertIn("response_read_error",[i["code"] for i in inventory.requests[1]["issues"]])


    def test_mode_menu_reprompts_before_starting_a_workflow(self):
        import run
        with patch("builtins.input",side_effect=["", "wrong", " Inventory "]) as prompt, \
             patch.object(run,"inventory_main",return_value="inventory result") as inventory, \
             patch.object(run,"download_main") as download, redirect_stdout(StringIO()):
            self.assertEqual(run.main("08/10/2026"),"inventory result")
        self.assertEqual(prompt.call_count,3)
        inventory.assert_called_once_with("08/10/2026")
        download.assert_not_called()

    def test_download_choice_routes_dates_and_retry_arguments(self):
        import run
        with patch("builtins.input",return_value="2") as prompt, \
             patch.object(run,"download_main",return_value="download result") as download, \
             patch.object(run,"inventory_main") as inventory, redirect_stdout(StringIO()):
            result=run.main("08/10/2026","03/01/2000",retry_failed=True)
        self.assertEqual(result,"download result")
        prompt.assert_called_once()
        download.assert_called_once_with("08/10/2026","03/01/2000",retry_failed=True)
        inventory.assert_not_called()

    def test_explicit_mode_bypasses_menu(self):
        import run
        with patch("builtins.input",side_effect=AssertionError("No menu expected")), \
             patch.object(run,"inventory_main",return_value="inventory result") as inventory:
            self.assertEqual(run.main("08/10/2026",mode="inventory"),"inventory result")
        inventory.assert_called_once_with("08/10/2026")


    def test_download_inventories_new_and_reused_archives_without_more_prompts(self):
        import run
        from datetime import date
        from types import SimpleNamespace
        import pandas as pd

        self.add("ibkr", "IEF", [self.bars([(COB, 100)])])
        self.add("ecb", "EURUSD", [], status="network_error")
        for reused in (False, True):
            with self.subTest(reused=reused):
                if reused:
                    self.finish()
                result = SimpleNamespace(folder=self.archive, reused=reused,
                    summary=pd.DataFrame([dict(ticker="IEF", provider="ibkr",
                        status="response_saved", retrieval_method="reqHistoricalData")]))
                before = {}
                def download(*args, **kwargs):
                    # The new archive only becomes complete when retrieval finishes.
                    if not reused:
                        self.finish()
                    before.update({p: p.read_bytes() for p in self.archive.rglob("*") if p.is_file()})
                    return result
                output = StringIO()
                with patch.object(run, "london_today", return_value=date(2026, 10, 9)), \
                     patch.object(run, "INVENTORY_DIR", self.root / "reports"), \
                     patch.object(run, "RAW_DUMP_DIR", self.root / "not_the_returned_archive"), \
                     patch.object(run.MarketDataDownloader, "from_json") as factory, \
                     patch.object(run, "inspect_archive", wraps=inspect_archive) as reader, \
                     patch.object(run, "_process_inventory") as process, \
                     patch("builtins.input", side_effect=["2", "08/10/2026", "03/01/2000"]) as prompt, \
                     patch.object(socket.socket, "connect", side_effect=AssertionError("No network")), \
                     redirect_stdout(output):
                    factory.return_value.download_cob.side_effect = download
                    self.assertIs(run.main(), result)
                self.assertEqual(prompt.call_count, 3)
                factory.return_value.download_cob.assert_called_once()
                reader.assert_called_once_with(self.archive, progress=print, include_observations=True)
                process.assert_called_once()
                self.assertEqual(before, {p: p.read_bytes() for p in self.archive.rglob("*") if p.is_file()})
                report = self.root / "reports" / COB
                self.assertTrue((report / "inventory.md").is_file())
                saved = json.loads((report / "inventory.json").read_text(encoding="utf-8"))
                self.assertEqual(saved["cob"], COB)
                self.assertEqual(len(saved["requests"]), 2)
                self.assertEqual(saved["requests"][1]["saved_status"], "network_error")
                self.assertIn("Starting inventory", output.getvalue())

    def test_download_exception_does_not_start_inventory(self):
        import run
        with patch.object(run.MarketDataDownloader, "from_json") as factory, \
             patch.object(run, "_inventory_archive") as inventory, \
             patch("builtins.input", side_effect=AssertionError("No extra prompt")), \
             redirect_stdout(StringIO()):
            factory.return_value.download_cob.side_effect = ValueError("Archive incompatible")
            with self.assertRaisesRegex(ValueError, "Archive incompatible"):
                run.main("08/10/2026", "03/01/2000", mode="download")
        inventory.assert_not_called()

    def test_inventory_failure_keeps_archive_and_does_not_repeat_download(self):
        import run
        from types import SimpleNamespace
        import pandas as pd

        self.finish()
        self.write(self.archive / "manifest.json", dict(cob=COB, complete=False, request_count=0))
        before = {p: p.read_bytes() for p in self.archive.rglob("*") if p.is_file()}
        result = SimpleNamespace(folder=self.archive, reused=False, summary=pd.DataFrame(
            columns=["ticker", "provider", "status", "retrieval_method"]))
        output = StringIO()
        with patch.object(run.MarketDataDownloader, "from_json") as factory, \
             patch.object(run, "INVENTORY_DIR", self.root / "reports"), \
             redirect_stdout(output):
            factory.return_value.download_cob.return_value = result
            with self.assertRaises(ValueError):
                run.main("08/10/2026", "03/01/2000", mode="download")
        factory.return_value.download_cob.assert_called_once()
        self.assertEqual(before, {p: p.read_bytes() for p in self.archive.rglob("*") if p.is_file()})
        self.assertIn("Local processing could not finish", output.getvalue())
        self.assertFalse((self.root / "reports").exists())


    def test_malformed_response_rows_keep_later_responses_and_requests(self):
        yahoo_record = {"meta": {"exchangeTimezoneName": "America/New_York"},
                        "timestamp": [1791466200],
                        "indicators": {"quote": [{"close": [100]}]}}
        for index, invalid in enumerate((None, 7, "invalid record")):
            cases = [
                ("fred", {"observations": [invalid]},
                 {"observations": [{"date": COB, "value": "1.5"}]}, "observations", "value"),
                ("yahoo", {"chart": {"result": [invalid]}},
                 {"chart": {"result": [yahoo_record]}}, "prices", "close"),
                ("ibkr", {"bars": [invalid]}, self.bars([(COB, 100)]), "TRADES", "close"),
                ("alpha_vantage", {"Time Series (Daily)": {COB: invalid}},
                 {"Time Series (Daily)": {COB: {"4. close": "100"}}}, "prices", "close"),
                ("massive", {"results": [invalid]},
                 {"results": [{"t": 1791432000000, "c": 100}]}, "prices", "close"),
            ]
            for provider, malformed, valid, dataset, field in cases:
                self.add(provider, f"BROKEN_{index}_{provider}", [malformed, valid])
        self.add("ecb", "HEALTHY", ["TIME_PERIOD,OBS_VALUE\n2026-10-08,1.2\n"])
        self.finish()
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns)
                  for p in self.archive.rglob("*") if p.is_file()}
        with patch.object(socket.socket, "connect", side_effect=AssertionError("No network")):
            inventory = inspect_archive(self.archive)
            files = inventory.save(self.root / "reports" / COB)
        self.assertEqual(len(inventory.requests), 16)
        self.assertEqual(inventory.metadata["requests_with_reader_errors"], 15)
        for item in inventory.requests[:-1]:
            with self.subTest(ticker=item["ticker"]):
                self.assertEqual(item["saved_status"], "response_saved")
                errors = [i for i in item["issues"] if i["code"] == "response_read_error"]
                self.assertEqual(len(errors), 1)
                self.assertIn("response_001.json: AttributeError", errors[0]["detail"])
                self.assertEqual([r["role"] for r in item["response_files"]],
                                 ["unreadable", "observations"])
                self.assertTrue(all(r["checksum_matches"] for r in item["response_files"]))
                table = next(iter(item["datasets"].values()))
                self.assertEqual(table["row_count"], 1)
                field = "value" if item["provider"] == "fred" else "close"
                self.assertEqual(table["fields"][field]["numeric"], 1)
        self.assertEqual(inventory.requests[-1]["datasets"]["observations"]["row_count"], 1)
        self.assertEqual(inventory.requests[-1]["issues"], [])
        self.assertTrue(all(p.is_file() for p in files))
        self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns)
                                 for p in self.archive.rglob("*") if p.is_file()})

    def test_malformed_snapshot_greeks_keep_later_responses(self):
        self.add("ibkr", "CALL", [
            {"ticker": {"time": "2026-10-09T10:00:00+00:00", "modelGreeks": [1]}},
            {"ticker": {"time": "2026-10-09T10:01:00+00:00", "bid": 9, "ask": 11,
                        "modelGreeks": {"impliedVol": 0.2}}},
        ], operation="option_model_snapshot",
            contract={"underlying": "SPX", "expiration": "2026-10-16", "strike": 7800})
        inventory = self.inspect()
        self.assertEqual(inventory.metadata["requests_with_reader_errors"], 1)
        table = inventory.requests[0]["datasets"]["snapshot"]
        self.assertEqual(table["row_count"], 1)
        self.assertEqual(table["fields"]["modelGreeks.impliedVol"]["numeric"], 1)
        self.assertEqual(inventory.option_expiries[0]["contracts_with_positive_model_iv"], 1)

    def test_malformed_manual_curve_is_isolated_at_request_boundary(self):
        self.write(self.archive / "ois/us/curve.json", {"curve": None})
        self.rows.append(dict(provider="manual_ois", ticker="USD_SOFR_OIS",
                              request_id="USD_SOFR_OIS", operation="manual_curve_snapshot",
                              folder="ois/us", status="response_saved", cob=COB))
        self.add("fred", "SOFR", [{"observations": [{"date": COB, "value": "1.5"}]}])
        inventory = self.inspect()
        self.assertEqual(inventory.metadata["requests_with_reader_errors"], 1)
        self.assertIn("request_read_error", [i["code"] for i in inventory.requests[0]["issues"]])
        self.assertEqual(inventory.requests[1]["datasets"]["observations"]["row_count"], 1)
        self.assertTrue(inventory.report())

    def test_unexpected_runtime_errors_are_not_silenced(self):
        self.add("fred", "SOFR", [{"observations": [{"date": COB, "value": "1.5"}]}])
        with patch("src.preprocessing._RequestReader.parse", side_effect=RuntimeError("unexpected bug")):
            with self.assertRaisesRegex(RuntimeError, "unexpected bug"):
                self.inspect()
