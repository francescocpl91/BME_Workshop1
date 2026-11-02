"""Configuration changes must not redefine the functional unit-test fixtures."""
from pathlib import Path
from unittest.mock import patch
import io
import unittest


class FixtureIsolationTests(unittest.TestCase):
    def test_previously_outdated_tests_do_not_read_user_configuration(self):
        root=Path(__file__).resolve().parents[1]/'config'
        original=Path.read_text
        def isolated(path,*args,**kwargs):
            if path.resolve().is_relative_to(root.resolve()):
                raise AssertionError('Unit test attempted to read user configuration: '+str(path))
            return original(path,*args,**kwargs)
        names=[
            'test_fixed_option_inputs.FixedOptionInputTests.test_every_selected_call_is_explicit_consistent_and_enabled',
            'test_json_config.JsonConfigTests.test_fixture_configs_merge_reference_rates_and_ibkr',
            'test_ois.OisTests.test_fixed_curves_are_included_for_every_cob_without_relabelling',
            'test_portfolio.PortfolioTests.test_fixture_portfolio_has_explicit_50_50_weights',
            'test_ibkr_snapshots.SnapshotTests',
            'test_mc_diagnostics.DiagnosticTests.test_accepted_weight_rounding_has_no_constant_path_loss_or_input_mutation',
            'test_mc_diagnostics.DiagnosticTests.test_deterministic_diagnostics_and_all_horizons_unavailable_are_explicit',
            'test_mc_diagnostics.DiagnosticTests.test_export_plots_and_checksums_with_unavailable_horizon',
            'test_mc_diagnostics.DiagnosticTests.test_offsetting_fx_variance_roundoff_does_not_abort_diagnostics',
        ]
        with patch.object(Path,'read_text',isolated):
            suite=unittest.defaultTestLoader.loadTestsFromNames(names)
            report=io.StringIO()
            result=unittest.TextTestRunner(stream=report,verbosity=1).run(suite)
        self.assertEqual(result.testsRun,14)
        self.assertTrue(result.wasSuccessful(),report.getvalue())
