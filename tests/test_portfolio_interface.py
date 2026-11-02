"""Public portfolio contract: delegate engines, preserve gates and saved inputs."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import json
import unittest

import numpy as np
import pandas as pd

import test_monte_carlo as mc_fixture
import test_local_vol_real_world as physical_fixture
import test_simulation_comparison as report_fixture
from src import Portfolio
from src.statistics import analyze_processed
from src.monte_carlo import simulate_processed


class PortfolioInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.fixture = mc_fixture.MonteCarloTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.processed = self.fixture.processed()
        self.portfolio = Portfolio.from_processed(self.processed, source_folder=self.fixture.root)
        self.config = self.fixture.root / 'statistics.json'
        self.config.write_text(json.dumps(dict(schema_version=1, statistics=dict(
            periods_per_year=252, minimum_observations=2, rolling_windows=[2, 3],
            quantiles=[.05, .5, .95], stress_periods=[], portfolio_method='buy_and_hold',
            base_value=100, plots=False, fit_student_t=False, return_types=['simple', 'log']))))

    def test_constructor_is_passive_and_definition_is_defensive(self):
        with patch('src.statistics.analyze_processed') as stats, patch('src.monte_carlo.simulate_processed') as mc:
            portfolio = Portfolio.from_processed(self.processed)
        stats.assert_not_called(); mc.assert_not_called()
        portfolio.definition['holdings'][0]['weight'] = 0
        self.assertEqual(self.processed.summary['portfolio']['holdings'][0]['weight'], .5)
        self.assertEqual(portfolio.assets, {})
        with self.assertRaises(ValueError): portfolio.report()
        with self.assertRaises(ValueError): portfolio.plots_report(self.fixture.root / 'plots')

    def test_statistics_equal_direct_engine_and_are_automatic_per_series(self):
        before = self.processed.tables['selected_prices'].copy(deep=True)
        expected = analyze_processed(self.processed, self.config)
        actual = self.portfolio.analyze(self.config)
        self.assertEqual(actual.run_id, expected.run_id)
        for name in expected.tables:
            pd.testing.assert_frame_equal(actual.tables[name], expected.tables[name])
        for key, asset in self.portfolio.assets.items():
            self.assertAlmostEqual(asset.metrics['mean_daily'], asset.frame.simple_return.mean())
            self.assertAlmostEqual(asset.metrics['std_daily'], asset.frame.simple_return.std(ddof=1))
        self.assertEqual(self.portfolio.report(), actual.report())
        pd.testing.assert_frame_equal(before, self.processed.tables['selected_prices'])
        with self.assertRaises(ValueError): self.portfolio.report(sections=['overview'])

    def test_no_selected_history_leaves_statistics_and_mc_explicitly_unavailable(self):
        self.processed.tables['selected_prices'] = self.processed.tables['selected_prices'].iloc[:0].copy()
        with patch('socket.socket.connect', side_effect=AssertionError('No downloads')):
            statistics = self.portfolio.analyze(self.config)
            simulation = self.portfolio.simulate(self.fixture.file)
        self.assertEqual(statistics.summary['portfolio_status'], 'unavailable_missing_holdings')
        self.assertEqual(self.portfolio.assets, {})
        self.assertEqual(simulation.summary['status'], 'unavailable')
        self.assertFalse(simulation.summary['simulation_performed'])
        self.assertNotIn('terminal_values', simulation.tables)
        self.assertIn('unavailable_missing_holdings', self.portfolio.report())

    def test_historical_and_single_holding_equal_direct_engine(self):
        for single in (False, True):
            p = deepcopy(self.processed)
            if single:
                p.summary['portfolio']['holdings'] = [dict(request_id='A', weight=1.)]
            before = p.tables['selected_prices'].copy(deep=True)
            expected = simulate_processed(p, self.fixture.file)
            portfolio = Portfolio.from_processed(p)
            actual = portfolio.simulate(self.fixture.file)
            self.assertTrue(actual.summary['simulation_performed'])
            self.assertEqual(actual.summary, expected.summary)
            for name in expected.tables:
                pd.testing.assert_frame_equal(actual.tables[name], expected.tables[name])
            pd.testing.assert_frame_equal(before, p.tables['selected_prices'])
            self.assertIs(portfolio.simulations['historical'], actual)

    def test_failed_reruns_do_not_retain_previous_success(self):
        self.portfolio.analyze(self.config)
        self.portfolio.simulate(self.fixture.file)
        with patch('src.statistics.analyze_processed', side_effect=ValueError('bad input')):
            with self.assertRaises(ValueError): self.portfolio.analyze(self.config)
        with patch('src.monte_carlo.simulate_processed', side_effect=ValueError('bad input')):
            with self.assertRaises(ValueError): self.portfolio.simulate(self.fixture.file)
        self.assertIsNone(self.portfolio.statistics)
        self.assertNotIn('historical', self.portfolio.simulations)

    def test_route_and_missing_validation_are_rejected_before_engine(self):
        with patch('src.local_vol_real_world.simulate_processed') as physical:
            with self.assertRaises(ValueError):
                self.portfolio.simulate(self.fixture.file, method='physical_local_vol')
            with self.assertRaises(ValueError):
                self.portfolio.simulate(self.fixture.file, method='risk_neutral')
            with self.assertRaises(ValueError):
                self.portfolio.simulate(self.fixture.file, local_vol=object())
        physical.assert_not_called()

    def test_physical_parity_and_original_risk_neutral_gates(self):
        f = physical_fixture.RealWorldTests(); f.setUp(); self.addCleanup(f.doCleanups)
        p = Portfolio.from_processed(f.p, source_folder=f.root)
        def simulate(validation):
            return p.simulate(f.file, method='physical_local_vol', historical_config=f.fixture.file,
                              local_vol=f.local, surface=f.surface, validation=validation)
        with patch('socket.socket.connect', side_effect=AssertionError('No downloads')):
            expected = f.run_model(); actual = simulate(f.rn)
        self.assertTrue(actual.summary['simulation_performed'])
        self.assertEqual(expected.summary, actual.summary)
        for key in expected.arrays:
            np.testing.assert_array_equal(expected.arrays[key], actual.arrays[key])
        bad = deepcopy(f.rn); bad.manifest['identity']['source_local_id'] = 'other'
        actual = simulate(bad)
        self.assertFalse(actual.summary['simulation_performed'])
        self.assertEqual(actual.summary['reason'], 'risk_neutral_local_identity_mismatch')
        bad = deepcopy(f.rn); bad.tables['cohorts'][0]['fit_pass'] = False
        actual = simulate(bad)
        self.assertFalse(actual.summary['simulation_performed'])
        self.assertIn('risk_neutral', actual.summary['reason'])
        with self.assertRaises(ValueError): simulate(None)
        self.assertNotIn('physical_local_vol', p.simulations)


class PortfolioReportingTests(unittest.TestCase):
    def setUp(self):
        self.fixture = report_fixture.FinalAnalysisTests(); self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        from src.data_cleaning import ProcessedArchive
        p = ProcessedArchive(f.root/'raw', f.cob, 'sample', {}, {'portfolio': f.portfolio})
        self.portfolio = Portfolio.from_processed(p, source_folder=f.processed)

    def test_portfolio_reports_verified_history_without_options(self):
        f = self.fixture
        # Construct an actual completed processing manifest; optional numerical
        # stages are represented by the existing synthetic saved artifacts.
        (f.processed/'manifest.json').write_text(json.dumps(dict(
            cob=f.cob, run_id='sample', complete=True, files={}, portfolio=f.portfolio)))
        result = self.portfolio.build_final_analysis(
            {'processing': f.processed, 'monte_carlo': f.historical()}, f.cfg)
        target = result.save(f.root/'reports')
        markdown = (target/'report.md').read_text(encoding='utf-8')
        self.assertIn('No physical local-volatility comparison', markdown)
        self.assertIn('gaussian', markdown)
        self.assertIn('student_t_truncated', markdown)
        self.assertIn('Mean MC standard error', markdown)
        self.assertIn('report.md', json.loads((target/'manifest.json').read_text())['files'])
        self.assertEqual(self.portfolio.report(), markdown)
        selected = self.portfolio.report(sections=['overview'], include_figures=False)
        self.assertIn('What can this run tell us?', selected)
        self.assertNotIn('What do the portfolio simulations imply?', selected)
        self.assertNotIn('![', selected)
        paths = self.portfolio.plots_report(f.root/'reports')
        self.assertTrue(paths)
        for path in paths:
            self.assertTrue(path.is_file())
            self.assertIn('](' + path.name + ')', markdown)
        with self.assertRaises(ValueError): result.report(sections=['unknown'])
        with self.assertRaises(ValueError): result.report(sections='overview')
        with self.assertRaises(ValueError): result.report(include_figures='yes')
        # Markdown belongs to the immutable verified bundle as much as HTML.
        (target/'report.md').write_text('tampered')
        with self.assertRaises(ValueError): self.portfolio.plots_report(f.root/'reports')

    def test_foreign_processing_folder_or_portfolio_cannot_attach(self):
        f = self.fixture
        with self.assertRaises(ValueError):
            self.portfolio.build_final_analysis({'processing': f.root/'other'}, f.cfg)
        (f.processed/'manifest.json').write_text(json.dumps(dict(
            cob=f.cob, run_id='sample', complete=True, files={}, portfolio=dict(reporting_currency='EUR', holdings=[]))))
        with self.assertRaises(ValueError):
            self.portfolio.build_final_analysis({'processing': f.processed}, f.cfg)
        self.assertIsNone(self.portfolio.final_analysis)

    def test_matching_weights_do_not_allow_another_processing_identity(self):
        f = self.fixture
        (f.processed/'manifest.json').write_text(json.dumps(dict(
            cob=f.cob, run_id='different', complete=True, files={}, portfolio=f.portfolio)))
        with self.assertRaisesRegex(ValueError, 'processing identity'):
            self.portfolio.build_final_analysis({'processing': f.processed}, f.cfg)
        self.assertIsNone(self.portfolio.final_analysis)

    def test_invalid_constructor_input_is_a_clear_validation_error(self):
        for value in (None, {}, object()):
            with self.assertRaisesRegex(ValueError, 'cleaned portfolio'):
                Portfolio.from_processed(value)

    def test_markdown_keeps_physical_status_and_escapes_tables(self):
        f = self.fixture
        result = f.build({'monte_carlo': f.historical(), 'local_vol_real_world': f.physical()})
        result.findings.append('Test <literal> & text')
        result.comparison.loc[0, 'comparison_reason'] = 'First | second'
        page = result.report(include_figures=False)
        self.assertIn('physical_local_vol', page)
        self.assertIn('diagnostic', page)
        self.assertIn('First \\| second', page)
        self.assertIn('Test &lt;literal&gt; &amp; text', page)
        self.assertIn('data:image/svg+xml;base64,', page)


    def _attach(self, stages):
        f = self.fixture
        (f.processed/'manifest.json').write_text(json.dumps(dict(
            cob=f.cob, run_id='sample', complete=True, files={}, portfolio=f.portfolio)), encoding='utf-8')
        return self.portfolio.build_final_analysis({'processing': f.processed, **stages}, f.cfg)

    def test_text_only_report_preserves_risk_tables_and_warnings_before_plots(self):
        f = self.fixture
        result = self._attach({'monte_carlo': f.historical()})
        # This call order exposed the bug: text must be complete even when no
        # save/plot call has happened. Prohibit chart rendering, not just links.
        with patch('src.simulation_comparison._figures', side_effect=AssertionError('No plots')):
            first = self.portfolio.report(include_figures=False)
            repeated = self.portfolio.report(include_figures=False)
        self.assertEqual(first, repeated)
        self.assertEqual(result.figures, {})
        self.assertIn('Probability of specified terminal losses', first)
        self.assertIn('no saved forecast bands', first)
        self.assertEqual(len(result.chart_notes), 2)
        expected = result.loss_thresholds.copy(deep=True)
        gaussian = expected[expected.scenario == 'gaussian'].set_index('loss_threshold')
        self.assertEqual(gaussian.loc[.1, 'probability'], .25)
        target = result.save(f.root/'reports')
        self.assertEqual(first, self.portfolio.report(include_figures=False))
        pd.testing.assert_frame_equal(expected, result.loss_thresholds)
        self.assertEqual(len(result.chart_notes), 2)
        self.assertIn('Probability of specified terminal losses', (target/'report.md').read_text(encoding='utf-8'))

    def test_empty_cob_report_has_no_invented_forecasts_or_risk(self):
        f = self.fixture
        result = self._attach({})
        first = self.portfolio.report(include_figures=False)
        self.assertIn('No usable full-portfolio terminal simulation', first)
        self.assertNotIn('Probability of specified terminal losses', first)
        self.assertTrue(result.comparison.empty)
        self.assertTrue(result.forecast_bands.empty)
        self.assertTrue(result.loss_thresholds.empty)
        target = result.save(f.root/'reports')
        self.assertEqual(first, self.portfolio.report(include_figures=False))
        for name in ('comparison', 'forecast_bands', 'loss_thresholds'):
            self.assertTrue(pd.read_csv(target/(name+'.csv')).empty)

    def test_no_options_keeps_historical_portfolio_and_reports_reason(self):
        f = self.fixture
        calibration = f.stage('option_calibration', dict(cob=f.cob,
            status='unavailable', reason='no_usable_option_observations'))
        result = self._attach({'monte_carlo': f.historical(), 'option_calibration': calibration})
        text = self.portfolio.report(include_figures=False)
        self.assertEqual(set(result.comparison.scenario), {'gaussian', 'student_t_truncated'})
        self.assertIn('no_usable_option_observations', text)
        self.assertIn('No physical local-volatility comparison', text)
        self.assertIn('Probability of specified terminal losses', text)
        result.save(f.root/'reports')
        self.assertEqual(text, self.portfolio.report(include_figures=False))

    def test_wrong_cob_results_are_not_used_to_fill_an_empty_date(self):
        f = self.fixture
        result = self._attach({'monte_carlo': f.historical(cob='2024-01-01')})
        text = self.portfolio.report(include_figures=False)
        self.assertTrue(result.comparison.empty)
        self.assertTrue(result.loss_thresholds.empty)
        self.assertIn('integrity_error', text)
        self.assertIn('Source COB is missing or differs', text)
        self.assertNotIn('Probability of specified terminal losses', text)

    def test_physical_missing_trajectories_keeps_terminal_risk_without_figures(self):
        f = self.fixture
        result = self._attach({'monte_carlo': f.historical(), 'local_vol_real_world': f.physical()})
        text = self.portfolio.report(include_figures=False)
        self.assertIn('trajectory arrays are unavailable', text)
        self.assertIn('physical_local_vol', text)
        self.assertIn('diagnostic', text)
        self.assertEqual(len(result.loss_thresholds), 6)
        result.save(f.root/'reports')
        self.assertEqual(text, self.portfolio.report(include_figures=False))


if __name__ == '__main__':
    unittest.main()
