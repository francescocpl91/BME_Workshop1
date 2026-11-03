"""Exercise both application entry points using a synthetic provider response."""
import io
import json
import socket
import unittest
from contextlib import ExitStack, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import run
import test_preprocessing as archive_fixtures
from src.data_cleaning import load_cleaning_config

ROOT = Path(__file__).resolve().parents[1]


class ScenarioWorkflowTests(unittest.TestCase):
    def test_plot_launcher_opens_only_requested_saved_families(self):
        from tempfile import TemporaryDirectory
        from unittest.mock import MagicMock
        from src.plot_windows import launch
        with TemporaryDirectory() as tmp:
            folder=Path(tmp)
            names=["mc_density_0.png","stress_mc_density_0.png","mc_paths_0.png","smiles_0_0.png"]
            (folder/"manifest.json").write_text(json.dumps(dict(complete=True,files={n:"fixture" for n in names})))
            for n in names:
                (folder/n).touch()
            with patch("src.plot_windows.subprocess.Popen") as popen:
                child=MagicMock();child.stdout=io.StringIO("ready\n");popen.return_value=child
                self.assertFalse(launch(folder,[]))
                self.assertFalse(launch(folder,["missing"]))
                popen.assert_not_called()
                self.assertTrue(launch(folder,["mc_paths","mc_density"]))
                self.assertEqual(popen.call_args.args[0][-2:],["mc_paths_0.png","mc_density_0.png"])
                child.stdout=io.StringIO("")
                child.poll.return_value=1
                self.assertFalse(launch(folder,["mc_density"]))


    def test_download_then_archive_replay_produce_the_same_user_comparison(self):
        f = archive_fixtures.InventoryTests()
        f.setUp()
        self.addCleanup(f.doCleanups)
        portfolio = dict(schema_version=1,reporting_currency="USD",weight_basis="reporting_currency_market_value",
                         nav=None,target_allocation={"equity":1},holdings=[dict(request_id="ETF_A",bucket="equity",weight=1)],
                         reference_data=[])
        def download(*args, **kwargs):
            stamps = [int(datetime(2026,10,d,20,tzinfo=timezone.utc).timestamp()) for d in (5,6,7,8)]
            prices = [100,99,102,101]
            f.add("yahoo","ETF_A",[{"chart":{"result":[{"meta":{"exchangeTimezoneName":"America/New_York",
                  "currency":"USD"},"timestamp":stamps,"indicators":{"quote":[{"close":prices,"volume":[10]*4}],
                  "adjclose":[{"adjclose":prices}]}}]}}],start="2026-10-05")
            f.inputs[0].update(kind="price",currency="USD",unit="USD/share",frequency="1d")
            f.finish()
            f.write(f.archive/"portfolio.json",portfolio)
            return SimpleNamespace(folder=f.archive,reused=False,summary=pd.DataFrame(f.rows).assign(retrieval_method="fixture"))

        clean=load_cleaning_config(ROOT/"config/cleaning.json")
        clean["prices"]["minimum_return_observations"]=2
        mc=json.loads((ROOT/"tests/fixtures/config/monte_carlo.json").read_text(encoding="utf-8"))
        mc.update(paths=32,models=["gaussian"],plots=False,sample_paths=2,forecast_end="2026-10-16",
                  stress_calibrations=[],user_scenario=dict(name="custom",periods_per_year=252,
                  holdings={"ETF_A":{"annual_volatility":.25}}))
        mc["calibration"].update(start="2026-10-05",minimum_observations=2)
        mc["diagnostics"]["enabled"]=False
        configs={"MONTE_CARLO_FILE":mc,"PORTFOLIO_FILE":portfolio,"CLEANING_FILE":clean}
        with ExitStack() as stack:
            for name,data in configs.items():
                filename=f.root/(name+".json")
                f.write(filename,data)
                stack.enter_context(patch.object(run,name,filename))
            for name,folder in (("RAW_DUMP_DIR",f.archive.parent),("PROCESSED_DIR",f.root/"processed"),
                                ("INVENTORY_DIR",f.root/"reports")):
                stack.enter_context(patch.object(run,name,folder))
            stack.enter_context(patch.object(run,"_calibrate_available",return_value=None))
            stack.enter_context(patch.object(socket.socket,"connect",side_effect=AssertionError("Network")))
            stack.enter_context(patch.object(run.sys.stdin,"isatty",return_value=False))
            stack.enter_context(redirect_stdout(io.StringIO()))
            with patch.object(run.MarketDataDownloader,"from_json") as factory:
                factory.return_value.download_cob.side_effect=download
                run.main("08/10/2026","05/10/2026",mode="download")
                factory.return_value.download_cob.assert_called_once()
            reports=f.root/"reports"/"2026-10-08"/"final_analysis"
            [target]=list(reports.iterdir())
            summary=json.loads((target/"summary.json").read_text(encoding="utf-8"))
            self.assertEqual({r["category"] for r in summary["comparison"]},{"baseline","user_assumption"})
            self.assertIn("User assumptions on the historical forecast",(target/"report.md").read_text(encoding="utf-8"))
            before=(target/"manifest.json").read_bytes()
            with patch.object(run.MarketDataDownloader,"from_json",side_effect=AssertionError("Archive opened a provider")):
                run.main("08/10/2026",mode="process")
            self.assertEqual(list(reports.iterdir()),[target])
            self.assertEqual((target/"manifest.json").read_bytes(),before)
