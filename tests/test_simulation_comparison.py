"""Synthetic offline fixtures: no current portfolio/configuration dependency."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.simulation_comparison import build_analysis, load_config, _metrics, _forecast_data, _publish_directory, _outcome_interpretation, _modeling_context


POLICY = dict(schema_version=1, enabled=True, primary_return_type="simple",
              reference_model="gaussian", confidence_levels=[.95,.99],
              sampling_confidence=.95, allow_approximate_terminal_horizon=True,
              maximum_relative_horizon_difference=.005,
              material_loss_probability_difference=.02, minimum_expected_tail_paths=50,
              maximum_chart_points=100, surface_display_points=100,
              simulation_plots=dict(sample_paths=2,histogram_bins=12,fan_quantiles=[.05,.95],
                                    tail_display_fraction=.1,loss_thresholds=[.1,.2],smiles_per_figure=4))


class FinalAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.processed = self.root / "processed" / "sample"
        self.processed.mkdir(parents=True)
        self.cob = "2024-01-02"
        self.cfg = self.root / "report.json"; self.cfg.write_text(json.dumps(POLICY))
        self.portfolio = dict(reporting_currency="USD", holdings=[dict(request_id="0005",weight=.6),dict(request_id="NA",weight=.4)])
        self.settings = dict(initial_value=100,paths=4,models=["gaussian","student_t_truncated"],
                             portfolio_method="buy_and_hold",forecast_start="2024-01-03",forecast_end="2025-01-03")
        self.calibration = pd.DataFrame({"return_start_date":["2023-12-28","2023-12-29"],
             "date":["2023-12-29","2024-01-02"],"asset::0005":[.01,-.01],"asset::NA":[.02,-.02]})

    def stage(self, name, meta, tables=None, physical=False, arrays=None):
        folder = self.processed / name / "fixture"; folder.mkdir(parents=True,exist_ok=True)
        files = {}
        (folder/'report.md').write_text('Synthetic saved report',encoding='utf-8')
        digest=hashlib.sha256((folder/'report.md').read_bytes()).hexdigest()
        files['report.md' if physical else 'report']=digest if physical else dict(file='report.md',sha256=digest)
        for key, df in (tables or {}).items():
            file = folder / (key+".csv"); df.to_csv(file,index=False)
            digest = hashlib.sha256(file.read_bytes()).hexdigest()
            files[file.name if physical else key] = digest if physical else dict(file=file.name,sha256=digest,rows=len(df))
        if physical:
            (folder/'summary.json').write_text(json.dumps(meta),encoding='utf-8')
            files['summary.json'] = hashlib.sha256((folder/'summary.json').read_bytes()).hexdigest()
            if arrays is not None:
                np.savez(folder/'paths.npz',terminal_wealth=arrays)
                files['paths.npz'] = hashlib.sha256((folder/'paths.npz').read_bytes()).hexdigest()
            manifest = dict(identity=meta['identity'],complete=True,files=files)
        else:
            manifest = dict(meta,complete=True,files=files)
        (folder/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
        return folder

    def historical(self, **changes):
        identity = dict(cob=self.cob,processing_id='sample',settings=copy.deepcopy(self.settings),portfolio=copy.deepcopy(self.portfolio))
        meta = dict(cob=self.cob,identity=identity,status='complete',simulation_performed=True,simulation_currency='USD',steps=252)
        meta.update(changes)
        rows=[]
        for model,values in [('gaussian',[80,100,110,130]),('student_t_truncated',[70,100,110,140])]:
            for i,value in enumerate(values):
                rows.append(dict(model=model,scope='portfolio',instrument_id='portfolio',currency='USD',path=i,
                                 terminal_value=value,calibration_set='baseline'))
        return self.stage('monte_carlo',meta,dict(terminal_values=pd.DataFrame(rows),calibration_returns=self.calibration.assign(calibration_set='baseline')))

    def physical(self, modify=None):
        identity=dict(cob=self.cob,processing_id='sample',mc_config=copy.deepcopy(self.settings),portfolio=copy.deepcopy(self.portfolio))
        meta=dict(identity=identity,status='diagnostic',simulation_performed=True,simulation_currency='USD',
                  clock_comparison=dict(historical_business_years=1,local_vol_elapsed_years=1.001,historical_session_steps=252))
        if modify: modify(meta)
        return self.stage('local_vol_real_world',meta,dict(calibration_returns=self.calibration),physical=True,
                          arrays=np.array([[100,100,75],[100,100,100],[100,100,115],[100,100,140]]))

    def build(self, folders=None, **kwargs):
        # All integration tests explicitly prohibit provider network access.
        with patch('socket.socket.connect',side_effect=AssertionError('No network')):
            return build_analysis(self.cob, folders or {}, self.cfg,requested_portfolio=self.portfolio,**kwargs)

    def test_no_data_is_an_explicit_report_not_an_exception(self):
        result=self.build(); target=result.save(self.root/'reports')
        page=(target/'report.html').read_text(encoding='utf-8')
        self.assertIn('No usable full-portfolio',page);self.assertIn('ACADEMIC PROJECT',page)
        self.assertNotIn('university',page.lower());self.assertNotIn('\u2014',page)
        self.assertTrue((target/'comparison.csv').exists())
        self.assertTrue(pd.read_csv(target/'comparison.csv').empty)
        self.assertIn('probability_loss',pd.read_csv(target/'comparison.csv').columns)

    def test_historical_only_has_metrics_and_embedded_figures(self):
        folder=self.historical(); before={f.name:f.read_bytes() for f in folder.iterdir()}
        result=self.build({'monte_carlo':folder})
        self.assertEqual(len(result.comparison),2)
        target=result.save(self.root/'reports');page=(target/'report.html').read_text(encoding='utf-8')
        self.assertIn('data:image/png;base64,',page);self.assertIn('No physical local-volatility',page)
        self.assertIn((folder/'report.md').as_uri(),page)
        self.assertNotIn((folder/'report.html').as_uri(),page)
        self.assertEqual(before,{f.name:f.read_bytes() for f in folder.iterdir()})
        self.assertEqual(result.save(self.root/'reports'),target)

    def test_physical_comparison_keeps_diagnostic_and_clock_warning(self):
        result=self.build({'monte_carlo':self.historical(),'local_vol_real_world':self.physical()})
        row=result.comparison.iloc[-1]
        self.assertEqual(row.status,'diagnostic');self.assertEqual(row.comparability,'approximate_terminal_only')
        self.assertIn('diagnostic',' '.join(result.findings))

    def test_modeling_context_uses_saved_settings_and_actual_dates(self):
        self.settings.update(calibration=dict(lookback_years=7,start=None,minimum_observations=100),
                             calendar='XLON',seed=42)
        h=self.historical(calibration=dict(requested_start='2017-01-02',requested_end=self.cob,
                         first_start='2020-01-02',last_end=self.cob,observations=700),
                          forecast_dates=['2024-01-03','2025-01-02'])
        # A later edited configuration must not rewrite saved history or dates.
        self.settings['calibration']['lookback_years']=20
        r=self.build({'monte_carlo':h});context=_modeling_context(r)[0]
        self.assertEqual(context['history_policy'],'Trailing 7 calendar years')
        self.assertEqual(context['actual_history_start'],'2020-01-02')
        self.assertEqual(context['matched_daily_intervals'],700)
        self.assertEqual(context['requested_forecast_end'],'2025-01-03')
        self.assertEqual(context['actual_forecast_end'],'2025-01-02')
        self.assertEqual(context['calendar'],'XLON')
        target=r.save(self.root/'reports');page=(target/'report.html').read_text(encoding='utf-8')
        self.assertIn('SDE interpretation',page);self.assertIn('not an exact Brownian SDE',page)
        self.assertIn('700',page);self.assertIn('Trailing 7 calendar years',page)
        self.assertEqual(json.loads((target/'summary.json').read_text())['modeling_context'][0],context)

    def test_explicit_start_overrides_lookback_in_report_context(self):
        self.settings['calibration']=dict(start='2022-01-01',lookback_years=10,minimum_observations=100)
        r=self.build({'monte_carlo':self.historical()});context=_modeling_context(r)[0]
        self.assertEqual(context['history_policy'],'Explicit start 2022-01-01 (overrides lookback)')
        self.assertIsNone(context['actual_history_start'])
        self.assertIsNone(context['actual_forecast_start'])

    def test_unavailable_local_vol_does_not_block_report_or_invent_forecast(self):
        h=self.historical()
        p=self.physical(lambda m:m.update(status='unavailable',simulation_performed=False,reason='missing_options'))
        r=self.build({'monte_carlo':h,'local_vol_real_world':p})
        target=r.save(self.root/'reports');page=(target/'report.html').read_text(encoding='utf-8')
        self.assertEqual(len(r.comparison),2)
        self.assertIn('No physical local-volatility comparison',page)
        self.assertIn('missing_options',page)
        self.assertIn('settings below describe the saved request',page)
        self.assertFalse(_modeling_context(r)[1]['terminal_results_included'])
        self.assertIn('mc_density_0',r.figures)

    def test_missing_vol_inputs_skip_physical_without_stopping_final_hook(self):
        import run
        from types import SimpleNamespace
        from contextlib import redirect_stdout
        from io import StringIO
        with redirect_stdout(StringIO()):
            self.assertIsNone(run._real_world_available(None,None,None,None,self.processed))
        # The orchestration always calls the final report even if a processing
        # stage returns without any optional volatility result.
        with patch.object(run,'_process_inventory_stages',return_value=None),patch.object(run,'_final_analysis') as report:
            run._process_inventory(SimpleNamespace(cob=self.cob))
        report.assert_called_once_with(self.cob,{}, {})

    def test_mismatched_currency_weights_dates_and_clock_are_not_compared(self):
        edits=[lambda m:m.update(simulation_currency='EUR'),
               lambda m:m['identity']['portfolio']['holdings'][0].update(weight=.5),
               lambda m:m['identity']['mc_config'].update(forecast_end='2025-02-03'),
               lambda m:m['clock_comparison'].update(local_vol_elapsed_years=1.1)]
        for edit in edits:
            with self.subTest(edit=edit):
                r=self.build({'monte_carlo':self.historical(),'local_vol_real_world':self.physical(edit)})
                physical=r.comparison[r.comparison.stage=='local_vol_real_world']
                self.assertTrue(physical.empty or physical.iloc[0].comparability=='not_comparable')

    def test_calibration_dates_and_factors_must_match(self):
        h=self.historical();self.calibration.loc[0,'asset::0005']=.7
        r=self.build({'monte_carlo':h,'local_vol_real_world':self.physical()})
        self.assertEqual(r.comparison.iloc[-1].comparability,'not_comparable')

    def test_bad_history_metadata_does_not_discard_valid_physical_result(self):
        for malformed in (None,[],7):
            with self.subTest(settings=malformed):
                h=self.historical();p=self.physical()
                file=h/'manifest.json';meta=json.loads(file.read_text());meta['identity']['settings']=malformed
                file.write_text(json.dumps(meta))
                r=self.build({'monte_carlo':h,'local_vol_real_world':p})
                self.assertNotIn('monte_carlo',r.sources)
                self.assertEqual(r.comparison.scenario.tolist(),['physical_local_vol'])
                self.assertEqual(r.comparison.iloc[0].comparability,'standalone')
                self.assertTrue((r.save(self.root/'reports')/'report.html').exists())

    def test_failed_terminal_validation_does_not_poison_independent_physical_stage(self):
        h=self.historical();p=self.physical()
        file=h/'manifest.json';meta=json.loads(file.read_text());meta['identity']['settings']['paths']=5
        file.write_text(json.dumps(meta))
        r=self.build({'monte_carlo':h,'local_vol_real_world':p})
        self.assertNotIn('monte_carlo',r.sources)
        self.assertEqual(r.comparison.iloc[0].comparability,'standalone')
        self.assertTrue((r.save(self.root/'reports')/'report.html').exists())

    def test_calibration_identity_check_and_rejection_propagation(self):
        for mode in ('matching','mismatched','corrupt'):
            with self.subTest(mode=mode):
                cal_identity=dict(processing_id='sample',version='one')
                cal=self.stage('option_calibration',dict(cob=self.cob,identity=cal_identity))
                surface_identity=dict(cal_identity,version='other') if mode=='mismatched' else cal_identity
                surface=self.stage('volatility_surface',dict(cob=self.cob,identity=dict(processing_id='sample',calibration_identity=surface_identity)))
                local=self.stage('local_volatility',dict(cob=self.cob,identity=dict(processing_id='sample',source_surface_id=surface.name)))
                validation=self.stage('local_vol_mc',dict(cob=self.cob,identity=dict(processing_id='sample',source_local_id=local.name)))
                def identity(m):m['identity'].update(local_id=local.name,surface_id=surface.name,validation_id=validation.name)
                physical=self.physical(identity)
                if mode=='corrupt':(cal/'report.md').write_text('tampered')
                r=self.build(dict(monte_carlo=self.historical(),option_calibration=cal,volatility_surface=surface,
                                  local_volatility=local,local_vol_mc=validation,local_vol_real_world=physical))
                self.assertEqual('physical_local_vol' in r.samples,mode=='matching')
                for stage in ('volatility_surface','local_volatility','local_vol_mc','local_vol_real_world'):
                    self.assertEqual(stage in r.sources,mode=='matching')
                self.assertEqual('gaussian' in r.samples,True)

    def test_rejected_supplied_dependency_excludes_all_descendants(self):
        for corrupt in (False,True):
            with self.subTest(corrupt=corrupt):
                h=self.historical()
                surface=self.stage('volatility_surface',dict(cob=self.cob,identity=dict(processing_id='sample')))
                local=self.stage('local_volatility',dict(cob=self.cob,identity=dict(processing_id='sample',source_surface_id='wrong_surface')))
                validation=self.stage('local_vol_mc',dict(cob=self.cob,identity=dict(processing_id='sample',source_local_id=local.name)))
                def identity(m):
                    m['identity'].update(local_id=local.name,surface_id=surface.name,validation_id=validation.name)
                physical=self.physical(identity)
                if corrupt:(local/'report.md').write_text('tampered')
                r=self.build(dict(monte_carlo=h,volatility_surface=surface,local_volatility=local,
                                  local_vol_mc=validation,local_vol_real_world=physical))
                self.assertEqual(len(r.comparison),2)
                self.assertNotIn('physical_local_vol',r.samples)
                for stage in ('local_volatility','local_vol_mc','local_vol_real_world'):
                    self.assertNotIn(stage,r.sources)
                    self.assertEqual(next(x for x in r.availability if x['stage']==stage)['status'],'integrity_error')
                self.assertTrue((r.save(self.root/'reports')/'report.html').exists())

    def test_scientific_equations_render_offline_without_changing_figures(self):
        import base64,re,xml.etree.ElementTree as ET
        r=self.build({'monte_carlo':self.historical(),'local_vol_real_world':self.physical()})
        with patch('socket.socket.connect',side_effect=AssertionError('No equation downloads')):
            target=r.save(self.root/'reports')
        page=(target/'report.html').read_text(encoding='utf-8')
        equations=re.findall(r'data:image/svg\+xml;base64,([^\"]+)',page)
        self.assertEqual(len(equations),6)
        for data in equations:self.assertTrue(ET.fromstring(base64.b64decode(data)).tag.endswith('svg'))
        self.assertNotIn('<script',page)
        self.assertIn('mc_density_0',r.figures);self.assertIn('mc_tails_0',r.figures)
        self.assertEqual(len(r.comparison),3)

    def test_nonfinite_or_invalid_clocks_never_pass_comparability(self):
        for value in (float('nan'),float('inf'),-1,True,'1'):
            with self.subTest(value=value):
                h=self.historical();p=self.physical(lambda m:m['clock_comparison'].update(local_vol_elapsed_years=value))
                r=self.build({'monte_carlo':h,'local_vol_real_world':p})
                self.assertEqual(r.comparison.iloc[-1].comparability,'not_comparable')

    def test_missing_history_allows_standalone_physical(self):
        result=self.build({'local_vol_real_world':self.physical()})
        self.assertEqual(result.comparison.iloc[0].comparability,'standalone')

    def test_wrong_cob_is_excluded(self):
        folder=self.historical(cob='2023-01-02')
        result=self.build({'monte_carlo':folder})
        self.assertTrue(result.comparison.empty)
        self.assertEqual(result.availability[2]['status'],'integrity_error')

    def test_tampering_does_not_discard_other_stage(self):
        h=self.historical();p=self.physical();(p/'risk.csv').write_text('unrecorded')
        (p/'summary.json').write_text('{}')
        result=self.build({'monte_carlo':h,'local_vol_real_world':p})
        self.assertEqual(len(result.comparison),2)
        self.assertEqual(result.availability[-1]['status'],'integrity_error')

    def test_manifest_traversal_rejected(self):
        h=self.historical(); manifest=json.loads((h/'manifest.json').read_text())
        manifest['files']['escape']=dict(file='../../../../report.json',sha256='unused')
        (h/'manifest.json').write_text(json.dumps(manifest))
        self.assertEqual(self.build({'monte_carlo':h}).availability[2]['status'],'integrity_error')

    def test_missing_paths_cannot_be_treated_as_complete(self):
        h=self.historical();meta=json.loads((h/'manifest.json').read_text());meta['identity']['settings']['paths']=5
        (h/'manifest.json').write_text(json.dumps(meta))
        r=self.build({'monte_carlo':h});self.assertTrue(r.comparison.empty)

    def test_math_loss_and_sampling_intervals(self):
        m=_metrics(np.array([80,100,110,130]),100,POLICY)
        self.assertAlmostEqual(m['mean_return'],.05);self.assertEqual(m['probability_loss'],.25)
        self.assertLess(m['loss_probability_low'],.25);self.assertGreater(m['loss_probability_high'],.25)
        self.assertAlmostEqual(m['loss_es_0.95'],.2)
        flat=_metrics(np.full(100,100),100,POLICY)
        self.assertEqual(flat['probability_loss'],0);self.assertEqual(flat['mean_mc_standard_error'],0)
        gains=_metrics(np.array([110,120,130]),100,POLICY)
        self.assertLess(gains['loss_var_0.95'],0)

    def test_forecast_uses_full_bands_not_display_sample(self):
        folder=self.historical()
        # Enrich the source artifact with full-population bands and deliberately
        # different display paths. Fan statistics must come from the bands.
        bands=[];paths=[]
        for model,terminal in [('gaussian',[80,100,110,130]),('student_t_truncated',[70,100,110,140])]:
            for day,values in [('2024-01-03',[100]*4),('2025-01-03',terminal)]:
                bands.append(dict(model=model,scope='portfolio',date=day,mean=np.mean(values),
                    p50=np.median(values),p05=np.quantile(values,.05),p95=np.quantile(values,.95)))
                paths.append(dict(model=model,scope='portfolio',date=day,path=0,value=values[0]))
        meta=json.loads((folder/'manifest.json').read_text())
        for name,table in [('bands',pd.DataFrame(bands)),('sample_paths',pd.DataFrame(paths))]:
            table.to_csv(folder/(name+'.csv'),index=False)
            meta['files'][name]=dict(file=name+'.csv',sha256=hashlib.sha256((folder/(name+'.csv')).read_bytes()).hexdigest())
        (folder/'manifest.json').write_text(json.dumps(meta))
        r=self.build({'monte_carlo':folder});data=_forecast_data(r,r.comparison.to_dict('records'))
        frame,sample=data['gaussian']
        self.assertEqual(sample.shape,(2,1));self.assertEqual(frame.mean_index.iloc[-1],105)
        self.assertEqual(frame.lower_index.iloc[-1],83)
        r.save(self.root/'reports')
        self.assertTrue(any(k.startswith('mc_paths_') for k in r.figures))
        self.assertTrue(any(k.startswith('mc_density_') for k in r.figures))
        self.assertEqual(len(r.forecast_bands),4)

    def test_physical_path_clock_and_terminal_consistency(self):
        folder=self.physical(lambda m:m.update(start_timestamp='2024-01-03T12:00:00+00:00',
                                               end_timestamp='2025-01-03T12:00:00+00:00'))
        terminal=np.array([[100,100,75],[100,100,100],[100,100,115],[100,100,140]])
        np.savez(folder/'paths.npz',terminal_wealth=terminal,years=np.array([0,366/365]),
                 portfolio_wealth=np.array([[100]*4,terminal[:,-1]]))
        meta=json.loads((folder/'manifest.json').read_text())
        meta['files']['paths.npz']=hashlib.sha256((folder/'paths.npz').read_bytes()).hexdigest()
        (folder/'manifest.json').write_text(json.dumps(meta))
        r=self.build({'local_vol_real_world':folder});data=_forecast_data(r,r.comparison.to_dict('records'))
        frame,sample=data['physical_local_vol']
        # Floating-point year fractions can introduce a few nanoseconds when
        # converted back to calendar time; require agreement within a microsecond.
        self.assertLess(abs(frame.timestamp.iloc[-1].value-pd.Timestamp('2025-01-03T12:00:00').value),1000)
        self.assertEqual(frame.mean_index.iloc[-1],107.5)
        self.assertEqual(sample.shape,(2,2))
        r.sources['local_vol_real_world'].summary['end_timestamp']='2025-02-02T12:00:00+00:00'
        self.assertFalse(_forecast_data(r,r.comparison.to_dict('records')))
        self.assertTrue(any('horizon' in note for note in r.chart_notes))

    def test_missing_trajectory_arrays_preserves_terminal_comparison(self):
        r=self.build({'monte_carlo':self.historical(),'local_vol_real_world':self.physical()})
        r.save(self.root/'reports')
        self.assertEqual(len(r.comparison),3)
        self.assertTrue(r.forecast_bands.empty)
        self.assertEqual(len(r.loss_thresholds),6)
        self.assertEqual(r.loss_thresholds.query("scenario == 'gaussian' and loss_threshold == .1").loss_events.iloc[0],1)
        self.assertTrue(r.chart_notes)

    def test_invalid_plot_policies_are_rejected(self):
        for key,value in [('fan_quantiles',[.6,.9]),('histogram_bins',False),('tail_display_fraction',.9),('loss_thresholds',[-.1])]:
            policy=copy.deepcopy(POLICY);policy['simulation_plots'][key]=value
            self.cfg.write_text(json.dumps(policy))
            with self.assertRaises(ValueError):load_config(self.cfg)

    def test_publication_retries_transient_lock_but_never_overwrites(self):
        start=self.root/'stage';target=self.root/'target';start.mkdir()
        with patch('src.simulation_comparison.os.rename',side_effect=[PermissionError('locked'),None]) as operation,patch('src.simulation_comparison.time.sleep'):
            _publish_directory(start,target)
            self.assertEqual(operation.call_count,2)
        target.mkdir()
        with patch('src.simulation_comparison.os.rename',side_effect=PermissionError('locked')) as operation,patch('src.simulation_comparison.time.sleep'):
            with self.assertRaises(PermissionError):_publish_directory(start,target)
            operation.assert_called_once()

    def test_interpretation_does_not_compare_incompatible_physical_run(self):
        r=self.build({'monte_carlo':self.historical(),'local_vol_real_world':self.physical(lambda m:m.update(simulation_currency='EUR'))})
        self.assertNotIn('Physical local volatility',' '.join(_outcome_interpretation(r)))

    def test_identifier_and_html_escape_preserved(self):
        self.portfolio['holdings'][0]['request_id']='<script>alert(1)</script>'
        r=self.build();page=(r.save(self.root/'reports')/'report.html').read_text()
        self.assertNotIn('<script>alert',page);self.assertIn('&lt;script&gt;',page)

    def test_invalid_policy_does_not_silently_default(self):
        for cfg in [[],dict(POLICY,maximum_relative_horizon_difference=float('nan')),
                    dict(POLICY,confidence_levels=[[.95]]),dict(POLICY,schema_version=True)]:
            self.cfg.write_text(json.dumps(cfg))
            with self.assertRaises(ValueError):load_config(self.cfg)

    def test_tail_metrics_match_existing_fractional_tail_convention(self):
        policy=dict(POLICY,confidence_levels=[.625,.9500001,.95])
        m=_metrics(np.array([60,80,100,120]),100,policy)
        self.assertAlmostEqual(m['loss_es_0.625'],(.4+.5*.2)/1.5)
        self.assertIn('loss_es_0.9500001',m);self.assertIn('loss_es_0.95',m)

    def test_malformed_manifest_and_statistics_are_integrity_errors(self):
        folder=self.stage('statistics',dict(cob=self.cob),dict(statistics=pd.DataFrame({'unexpected':[1]})))
        r=self.build({'statistics':folder})
        self.assertEqual(r.availability[1]['status'],'integrity_error')
        (folder/'manifest.json').write_text('[]')
        self.assertEqual(self.build({'statistics':folder}).availability[1]['status'],'integrity_error')

    def test_saved_processing_does_not_hide_workflow_failure(self):
        folder=self.stage('statistics',dict(cob=self.cob))
        r=self.build({'statistics':folder},issues={'statistics':'Workflow stopped'})
        self.assertEqual(r.availability[1]['status'],'saved_with_execution_error')

    def test_statistics_only_keeps_identifiers_and_needs_no_options(self):
        rows=pd.DataFrame([dict(scope='holding',instrument_id='0005',return_type='simple'),
                           dict(scope='holding',instrument_id='NA',return_type='simple')])
        folder=self.stage('statistics',dict(cob=self.cob,portfolio_status='unavailable'),dict(statistics=rows))
        r=self.build({'statistics':folder});self.assertTrue(r.comparison.empty)
        self.assertEqual(r.sources['statistics'].table('statistics').instrument_id.tolist(),['0005','NA'])
        self.assertTrue((r.save(self.root/'reports')/'report.html').exists())

    def test_current_requested_weights_do_not_relabel_saved_results(self):
        historical=self.historical()
        self.portfolio['holdings'][0]['weight']=.1
        self.portfolio['holdings'][1]['weight']=.9
        r=self.build({'monte_carlo':historical})
        self.assertEqual(r.portfolio['holdings'][0]['weight'],.6)

    def test_expected_statistics_failure_still_runs_independent_mc(self):
        import run
        from types import SimpleNamespace
        from unittest.mock import Mock
        from contextlib import ExitStack,redirect_stdout
        from io import StringIO
        result=Mock(run_id='sample',summary=dict(portfolio=self.portfolio,
                    readiness=dict(aligned_return_observations=0,foreign_currencies=[]),options=dict(iv_candidates=0)))
        result.save.return_value=self.processed
        mc=Mock(run_id='mc',summary=dict(status='unavailable',warnings=[]))
        with ExitStack() as stack:
            stack.enter_context(redirect_stdout(StringIO()))
            stack.enter_context(patch.object(run,'process_inventory',return_value=result))
            stack.enter_context(patch.object(run,'analyze_processed',side_effect=ValueError('insufficient historical observations')))
            simulator=stack.enter_context(patch.object(run,'simulate_processed',return_value=mc))
            for name in ('_calibrate_available','_surface_available','_local_vol_available','_local_vol_mc_available','_real_world_available'):
                stack.enter_context(patch.object(run,name,return_value=None))
            stack.enter_context(patch('src.volatility_workflow.save_overview',return_value=self.root))
            reporter=stack.enter_context(patch.object(run,'_final_analysis'))
            run._process_inventory(SimpleNamespace(cob=self.cob))
        simulator.assert_called_once()
        self.assertIn('monte_carlo',reporter.call_args.args[1])
        self.assertIn('statistics',reporter.call_args.args[2])

    def test_missing_archive_attempts_report_without_downloader(self):
        import run
        with patch.object(run,'inspect_archive',side_effect=ValueError('missing archive')),patch.object(run,'_final_analysis') as report,patch.object(run,'MarketDataDownloader') as downloader:
            with self.assertRaises(ValueError):run._inventory_archive(self.root/self.cob)
        report.assert_called_once();downloader.assert_not_called()

    def test_report_cannot_overwrite_source(self):
        h=self.historical();r=self.build({'monte_carlo':h})
        with self.assertRaises(ValueError):r.save(h/'reports')

    def test_unsafe_output_identity_is_rejected(self):
        r=self.build();r.run_id='../escape'
        with self.assertRaises(ValueError):r.save(self.root/'reports')

    def test_source_edit_between_build_and_save_blocks_publication(self):
        h=self.historical();r=self.build({'monte_carlo':h})
        table=pd.read_csv(h/'terminal_values.csv');table.loc[0,'terminal_value']=81
        table.to_csv(h/'terminal_values.csv',index=False)
        with self.assertRaises(ValueError):r.save(self.root/'reports')
        self.assertFalse((self.root/'reports'/self.cob/'final_analysis'/r.run_id).exists())

    def test_corrupted_existing_report_is_not_overwritten(self):
        r=self.build();target=r.save(self.root/'reports');(target/'report.html').write_text('tampered')
        with self.assertRaises(ValueError):r.save(self.root/'reports')

    def test_final_hook_runs_on_failed_processing_and_preserves_error(self):
        import run
        from types import SimpleNamespace
        with patch.object(run,'_process_inventory_stages',side_effect=RuntimeError('test bug')),patch.object(run,'_final_analysis') as report:
            with self.assertRaisesRegex(RuntimeError,'test bug'):
                run._process_inventory(SimpleNamespace(cob=self.cob))
        self.assertIn('RuntimeError',report.call_args.args[2]['processing'])


if __name__=='__main__':unittest.main()
