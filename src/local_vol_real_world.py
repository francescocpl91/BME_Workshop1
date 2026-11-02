"""Offline physical-drift portfolio simulation with mapped local volatility.

READING MAP
1. Validate configuration and resolve SAVED option_reference relationships.
2. Reuse historical calibration; convert daily log moments to annual SDE units.
3. Evolve all native holding/FX factors jointly, on one ACT/365F time grid.
4. Convert currencies once, aggregate buy-and-hold wealth, and report evidence.

This is a model scenario, not a statistical conversion of Q probabilities to P.
Option-calibrated diffusion is retained; historical arithmetic drift is imposed.
There are no API calls, surface fitting, Student-t shocks or comparison workflow.
"""
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from html import escape
from importlib.metadata import version
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import hashlib
import json
import math

import numpy as np
import pandas as pd

from .mc_common import (SimulationUnavailable, prepare_history, moments,
                       _root_covariance, _portfolio_wealth, _sessions)
from .monte_carlo import load_monte_carlo_config, _expected_shortfall
from .mc_diagnostics import probability_of_loss
from .local_vol_monte_carlo import coefficient, validate_start_date
from .local_volatility import DateSpotLocalVol, LocalVolEvaluator
from .statistics import _basis_kind
from .storage import write_json

YEAR_SECONDS = 365. * 86400.  # ACT/365F is the calibrated evaluator's convention.


def _digest(value):
    return hashlib.sha256(value).hexdigest()


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode('utf-8')


def _code_fingerprints():
    """Conservatively invalidate saved results after any package code change.

    Local volatility calls transitive helpers for surface math, boundaries,
    pricing, statistics and risk diagnostics. A list of direct imports misses
    those dependencies. Hash every Python source in this package recursively.
    Unrelated edits may invalidate a result too; this is safer than reusing
    outdated numerical output. Config/input hashes and dependency versions
    remain separate identity fields. Reading bytes has no import side effects.
    """
    folder = Path(__file__).parent
    return {p.relative_to(folder).as_posix(): _digest(p.read_bytes())
            for p in sorted(folder.rglob('*.py'))}


def load_config(path):
    """Only explicit supported methods; reject misspellings and hidden defaults."""
    cfg = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    keys = {'schema_version','enabled','annualization_periods','steps_per_year',
            'refinement','end_time','end_timezone','methods_by_return_basis',
            'allow_boundary_assumptions','allow_precision_only_review',
            'drift_convention','unmapped_holdings','save_observations',
            'maximum_path_steps','convergence_relative_initial','convergence_z'}
    if not isinstance(cfg, dict) or set(cfg)-{'instructions'} != keys:
        raise ValueError('invalid_real_world_config_keys')
    if type(cfg['schema_version']) is not int or cfg['schema_version'] != 1:
        raise ValueError('invalid_real_world_schema')
    for k in ('enabled','allow_boundary_assumptions','allow_precision_only_review'):
        if type(cfg[k]) is not bool: raise ValueError('invalid_boolean:'+k)
    for k, lo, hi in (('annualization_periods',1,366),('steps_per_year',12,10000),
                     ('refinement',2,8),('save_observations',2,2000),
                     ('maximum_path_steps',1,200000000)):
        if type(cfg[k]) is not int or not lo <= cfg[k] <= hi:
            raise ValueError('invalid_integer:'+k)
    for k in ('convergence_relative_initial','convergence_z'):
        if type(cfg[k]) not in (int,float) or not math.isfinite(cfg[k]) or cfg[k] <= 0:
            raise ValueError('invalid_tolerance:'+k)
    if cfg['drift_convention'] != 'historical_arithmetic' or cfg['unmapped_holdings'] != 'historical':
        raise ValueError('unsupported_real_world_method')
    if not isinstance(cfg['end_time'], str) or len(cfg['end_time']) != 5:
        raise ValueError('end_time_requires_HH_MM')
    time.fromisoformat(cfg['end_time'])
    # ZoneInfoNotFoundError inherits KeyError, not ValueError. Normalize it
    # so run.py can report a configuration problem without an uncaught crash.
    zone = cfg['end_timezone']
    if not isinstance(zone, str) or not zone.strip():
        raise ValueError('invalid_end_timezone: expected an IANA timezone name')
    try:
        ZoneInfo(zone)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError('invalid_end_timezone:'+zone) from exc
    methods = cfg['methods_by_return_basis']
    if not isinstance(methods, dict) or not methods: raise ValueError('missing_proxy_methods')
    for basis, method in methods.items():
        kind = _basis_kind(basis)
        if (method, kind) not in {('total_return_proxy','total_return'),('direct_price','price_return')}:
            raise ValueError('incompatible_proxy_method:'+str(basis))
    return {k:v for k,v in cfg.items() if k != 'instructions'}


def _rn_permission(row, cfg):
    """A precision-only exception is diagnostic, never a relaxed pricing pass."""
    gates = ('fit_pass','pde_pass','convergence_pass','forward_pass')
    if not row or any(row.get(k) is not True for k in gates):
        raise SimulationUnavailable('risk_neutral_pricing_or_forward_validation_required')
    if row.get('status') == 'validated_with_boundary_assumptions' and row.get('precision_pass') is True:
        return False
    if (cfg['allow_precision_only_review'] and row.get('status') == 'review_required'
            and row.get('precision_pass') is False and row.get('failed_checks') == ['precision_pass']):
        return True
    raise SimulationUnavailable('risk_neutral_precision_validation_required')


def resolve_models(processed, history, local, surface, validation, cfg):
    """Use recorded mappings, never ticker names or the latest editable JSON.

    An unmapped holding uses historical volatility. A mapped but unsupported
    holding blocks the hybrid scenario; silently falling back would mislabel it.
    Multiple usable cohorts for one underlying are ambiguous, not first-match.
    """
    if local is None or surface is None or validation is None:
        raise SimulationUnavailable('local_surface_and_risk_neutral_result_required')
    if local.manifest['identity']['source_surface_id'] != surface.run_id:
        raise ValueError('local_surface_identity_mismatch')
    if validation.manifest['identity']['source_local_id'] != local.run_id:
        raise ValueError('risk_neutral_local_identity_mismatch')
    if local.manifest.get('archive_cob') != processed.cob or validation.manifest.get('archive_cob') != processed.cob:
        raise ValueError('processed_local_cob_mismatch')
    records = processed.summary.get('price_selection', [])
    selections = {r['instrument_id']:r for r in records}
    if len(selections) != len(records): raise ValueError('duplicate_saved_price_selection')
    models, rows, warnings = {}, [], []
    origin = None
    for i, holding in enumerate(history['holdings']):
        key = holding['instrument_id']
        if key not in selections: raise SimulationUnavailable('saved_mapping_metadata_missing:'+key)
        reference = selections[key].get('option_reference')
        row = dict(instrument_id=key, factor=holding['factor'], method='historical',
                   option_reference=reference, return_basis=holding['return_basis'])
        if reference:
            method = cfg['methods_by_return_basis'].get(holding['return_basis'])
            if method is None: raise SimulationUnavailable('proxy_method_not_configured:'+key)
            matches = [r for r in local.tables['state_mapping'] if r['underlying'] == reference
                       and r.get('currency') == holding['currency']
                       and r['group_id'] in local.state_evaluators]
            if len(matches) != 1: raise SimulationUnavailable('missing_or_ambiguous_surface:'+key)
            gid = matches[0]['group_id']; adapter = local.state_evaluators[gid]
            evidence = [r for r in validation.tables['cohorts'] if r['group_id'] == gid]
            if len(evidence) != 1: raise SimulationUnavailable('missing_or_ambiguous_validation:'+key)
            diagnostic = _rn_permission(evidence[0], cfg)
            date = validate_start_date(local, surface, adapter)
            if origin is not None and origin != adapter.reference:
                raise SimulationUnavailable('mapped_surfaces_need_same_reference_instant')
            origin = adapter.reference
            if diagnostic: warnings.append(key+': risk-neutral precision-only review retained; exploratory result.')
            if history['joined'][holding['factor']].std(ddof=1) <= np.finfo(float).eps:
                raise SimulationUnavailable('mapped_factor_has_no_historical_dependence_estimate:'+key)
            models[i] = dict(adapter=adapter, method=method, group_id=gid)
            row.update(method=method, group_id=gid, **date, precision_only_review=diagnostic)
        rows.append(row)
    if not models: raise SimulationUnavailable('no_mapped_local_volatility_holdings')
    return models, rows, origin, warnings


def simulation_window(origin, models, mc, cfg):
    """Retain the actual surface start and a configured, timezone-aware end.

    A final AM expiry is not a close-of-day observation. Never truncate a
    requested forecast to fit a shorter surface. Reject DST ambiguity instead
    of silently selecting one occurrence of a repeated/nonexistent clock time.
    """
    zone = ZoneInfo(cfg['end_timezone'])
    naive = datetime.combine(datetime.fromisoformat(mc['forecast_end']).date(), time.fromisoformat(cfg['end_time']))
    end = naive.replace(tzinfo=zone)
    if (end.utcoffset() != end.replace(fold=1).utcoffset()
            or end.astimezone(timezone.utc).astimezone(zone).replace(tzinfo=None) != naive):
        raise ValueError('ambiguous_or_nonexistent_forecast_end')
    end = end.astimezone(timezone.utc)
    horizon = (end-origin).total_seconds()/YEAR_SECONDS
    if horizon <= 0: raise SimulationUnavailable('forecast_end_must_follow_surface_start')
    knots = {0., horizon}
    for model in models.values():
        a = model['adapter']
        if end > a.expiries[-1]: raise SimulationUnavailable('forecast_end_exceeds_surface_horizon')
        knots.update(float(t) for t in a.boundary.times if 0 < t < horizon)
    # Recording endpoints are included in the numerical grid, not interpolated
    # wealth. Calendar time advances on weekends; historical annualization is
    # explicit, and the report distinguishes this from session-based MC.
    knots.update(np.linspace(0., horizon, cfg['save_observations']))
    anchors = sorted(knots); grid = [0.]
    for left, right in zip(anchors[:-1], anchors[1:]):
        n = max(1, math.ceil((right-left)*cfg['steps_per_year']))
        grid.extend(np.linspace(left,right,n+1)[1:])
    if (len(grid)-1)*cfg['refinement']*mc['paths'] > cfg['maximum_path_steps']:
        raise SimulationUnavailable('configured_path_step_budget_exceeded')
    return np.asarray(grid), end


def clock_comparison(origin, end, mc, cfg):
    """Expose calendar versus business-time horizons without changing the model.

    Historical MC initializes at COB+1 and adds a full daily return on each
    later exchange session. Physical MC uses the actual surface instant and
    an explicit end time. N/252 and actual_seconds/(365*86400) need not match.
    Dropping weekends or changing the denominator alone does not preserve the
    calibrated diffusion. This evidence prepares the later comparison stage.
    """
    start = mc['forecast_start']
    sessions = [d for d in _sessions(mc['calendar'], start, mc['forecast_end']) if d > start]
    years = (end-origin).total_seconds()/YEAR_SECONDS
    annual = cfg['annualization_periods']
    return dict(local_vol_day_count='ACT/365F',
                historical_calendar=mc['calendar'],
                historical_annualization_periods=annual,
                local_vol_elapsed_years=years,
                historical_session_steps=len(sessions),
                historical_business_years=len(sessions)/annual,
                local_vol_equivalent_historical_sessions=annual*years,
                historical_initial_date=start,
                historical_last_session=sessions[-1] if sessions else None,
                directly_time_aligned=False,
                comparison_requirement='Reconcile initial/end instants and observation grid; do not replace ACT/365F by 252 in the calibrated evaluator.')


def correlation_factor(covariance):
    """Separate marginal scales from dependence; deterministic factors stay zero.

    An eigen root supports singular dependence, unlike mandatory Cholesky.
    A zero historical variance has no identified correlation, represented by
    a zero row, not invented stochastic noise.
    """
    sigma = np.sqrt(np.diag(covariance))
    denom = np.outer(sigma,sigma)
    correlation = np.divide(covariance,denom,out=np.zeros_like(covariance),where=denom>0)
    return sigma, correlation, _root_covariance(correlation)


def _local_scales(logs, t, models, historical_sigma, allow_boundary):
    """Return per-path scales and audit masks; never cap volatility or states.

    For a total-return holding, B=D*F/I0 and Iproxy=I0*G*B=G*D*F.
    Consequently k=log(Iproxy/F)=log(G)+log(D). This algebra removes no
    dividend from holding wealth; it only constructs the surface price state.
    A price-only holding instead has Iproxy=I0*G and k=log(I0*G/F).
    """
    scales = np.broadcast_to(historical_sigma, logs.shape).copy()
    evidence = {}
    for i, model in models.items():
        a = model['adapter']; f,d = a.curves.at(float(t)); spot,_ = a.curves.at(0.)
        k = logs[:,i]+(math.log(d) if model['method']=='total_return_proxy' else math.log(spot/f))
        v, flags = coefficient(a,float(t),k,allow_boundary)
        scales[:,i] = np.sqrt(v)
        evidence[i] = dict(flags=flags, k=k, variance=v)
    return scales, evidence


def evolve(history, models, grid, mc, cfg, progress=None):
    """Joint log-Euler, coupled coarse/fine and same-grid Gaussian control.

    Shapes: paths N, factors F, holdings H. Independent normals (N,rank) are
    multiplied by a fixed correlation root, then by each path's volatility.
    dlogG=(mu-variance/2)dt+sigma*dW. All scales/drifts are annual; dt ACT/365F.
    Fine Brownian increments sum exactly to the coarse increment. All paths
    are independent (no antithetics), so SE uses N independent differences.
    """
    # m and C are DAILY LOG moments. annual*C gives annual variances.
    # mu=annual*m+historical_variance/2 converts log growth to SDE drift.
    # Freeze mu thereafter. Replacing that variance with local variance too
    # would cancel the Ito correction and impose a different, constant-log-
    # drift model. Here local variance changes the conditional log growth.
    mean, cov = moments(history['joined']); annual = cfg['annualization_periods']
    sigma, corr, root = correlation_factor(cov*annual)
    mu = annual*mean + .5*sigma*sigma
    paths = mc['paths']; factors = len(mean); refine = cfg['refinement']
    fine = np.zeros((paths,factors)); coarse=fine.copy(); control=fine.copy()
    rng = np.random.default_rng(mc['seed'])
    weights=np.array([h['weight'] for h in history['holdings']]);weights/=math.fsum(weights)
    transform=history['transform']; initial=mc['initial_value']
    captures=[fine.copy()]; portfolio=[np.full(paths,initial)]
    running_peak=np.full(paths,initial);drawdown=np.zeros(paths)
    audits={i:dict(wing_states=0,short_states=0,assumed_states=0,evaluations=0,
                  maximum_local_volatility=0.,wing_paths=np.zeros(paths,dtype=bool)) for i in models}
    for step,(left,right) in enumerate(zip(grid[:-1],grid[1:])):
        coarse_scale,_ = _local_scales(coarse,left,models,sigma,cfg['allow_boundary_assumptions'])
        dw_sum=np.zeros_like(fine)
        for sub in range(refine):
            t=left+(right-left)*sub/refine; dt=(right-left)/refine
            scale, evidence=_local_scales(fine,t,models,sigma,cfg['allow_boundary_assumptions'])
            # Z is independent N(0,1), L L'=R. Z @ L.T correlates the shocks;
            # sqrt(dt) gives Brownian increments. Multiplying by scale below
            # yields covariance diag(sigma_t)*R*diag(sigma_t)*dt. This root
            # is correlation, not currency conversion; it stays fixed.
            dw=(rng.standard_normal((paths,root.shape[1])) @ root.T)*math.sqrt(dt)
            # Per-path log-Euler: local equity scales change with state/time;
            # unmapped assets/FX keep historical scales. All factors receive
            # one JOINT vector shock, so IEF is not simulated independently
            # and stitched onto an equity path later.
            fine += (mu-.5*scale*scale)*dt + scale*dw
            control += (mu-.5*sigma*sigma)*dt + sigma*dw
            dw_sum+=dw
            for i,e in evidence.items():
                a=audits[i];flags=e['flags'];a['evaluations']+=paths
                a['wing_states']+=int(np.sum(flags['wing']));a['wing_paths']|=flags['wing']
                a['short_states']+=paths*int(flags['short'])
                a['assumed_states']+=int(np.sum(flags['assumed']))
                a['maximum_local_volatility']=max(a['maximum_local_volatility'],float(np.sqrt(e['variance']).max()))
            # Validate AFTER each step including the terminal point. A bad
            # state rejects the entire run; never discard only the bad paths.
            _,post_evidence=_local_scales(fine,left+(right-left)*(sub+1)/refine,models,sigma,cfg['allow_boundary_assumptions'])
            for i,e in post_evidence.items():
                # Include a first wing visit on the final update, even though
                # no next return uses that terminal coefficient. State counts
                # above count only coefficients actually used in a step.
                audits[i]['wing_paths'] |= e['flags']['wing']
            with np.errstate(over='raise',invalid='raise',under='ignore'):
                # H x F transform selects native asset log growth plus/minus
                # its FX growth once. Columns below are stand-alone wealth,
                # each starting at initial; portfolio weights apply next.
                holdings=initial*np.exp(fine @ transform.T)
                wealth=_portfolio_wealth(holdings,weights,initial)
            if not np.isfinite(wealth).all() or np.any(wealth <= 0): raise ValueError('invalid_simulated_wealth')
            running_peak=np.maximum(running_peak,wealth)
            drawdown=np.minimum(drawdown,wealth/running_peak-1)
        coarse += (mu-.5*coarse_scale*coarse_scale)*(right-left) + coarse_scale*dw_sum
        _local_scales(coarse,right,models,sigma,cfg['allow_boundary_assumptions'])
        if not all(np.isfinite(x).all() for x in (fine,coarse,control)):
            raise ValueError('nonfinite_simulated_log_growth')
        captures.append(fine.copy());portfolio.append(wealth.copy())
        if progress and (step % max(1,(len(grid)-1)//10)==0):
            progress(f'Physical joint MC: {step+1}/{len(grid)-1} coarse intervals')
    # Vector outputs use rows=paths, columns=factors. Portfolio arithmetic
    # applies weights only after native factors and FX have been combined.
    with np.errstate(over='raise',invalid='raise'):
        terminal=initial*np.exp(fine @ transform.T)
        coarse_terminal=initial*np.exp(coarse @ transform.T)
        control_terminal=initial*np.exp(control @ transform.T)
    if any(not np.isfinite(x).all() or np.any(x <= 0) for x in (terminal,coarse_terminal,control_terminal)):
        raise ValueError('invalid_terminal_wealth_no_clipping')
    def combined(values):return np.column_stack((values,_portfolio_wealth(values,weights,initial)))
    terminal,coarse_terminal,control_terminal=map(combined,(terminal,coarse_terminal,control_terminal))
    ids=[h['instrument_id'] for h in history['holdings']]+['PORTFOLIO']
    risk=[];convergence=[]
    for i,key in enumerate(ids):
        values=terminal[:,i];loss=1-values/initial
        row=dict(instrument_id=key,scope='portfolio' if i==len(ids)-1 else 'holding',
                 mean_wealth=float(values.mean()),median_wealth=float(np.median(values)),
                 probability_of_loss=probability_of_loss(loss),mean_mc_standard_error=float(values.std(ddof=1)/math.sqrt(paths)),
                 gaussian_control_mean=float(control_terminal[:,i].mean()))
        for q in mc['quantiles']:row['wealth_quantile_'+str(q)]=float(np.quantile(values,q))
        for q in mc['confidence_levels']:
            row['loss_var_'+str(q)]=float(np.quantile(loss,q));row['loss_es_'+str(q)]=_expected_shortfall(loss,q)
        risk.append(row)
        # Same Brownian paths make this a paired discretization check. Its
        # SE is uncertainty in the mean difference, not investment volatility.
        # Passing does not establish convergence of VaR/ES or physical tails.
        diff=values-coarse_terminal[:,i];se=float(diff.std(ddof=1)/math.sqrt(paths))
        budget=cfg['convergence_relative_initial']*initial+cfg['convergence_z']*se
        convergence.append(dict(instrument_id=key,scope=row['scope'],mean_fine_minus_coarse=float(diff.mean()),
             paired_standard_error=se,tolerance=budget,passed=bool(abs(diff.mean())<=budget)))
    bounds=[dict(instrument_id=history['holdings'][i]['instrument_id'],
                 **{k:v for k,v in a.items() if k!='wing_paths'},paths_visiting_wings=int(a['wing_paths'].sum())) for i,a in audits.items()]
    arrays=dict(years=grid,native_log_growth=np.asarray(captures),portfolio_wealth=np.asarray(portfolio),
                factor_ids=np.array(history['joined'].columns,dtype=str),conversion_matrix=transform,
                weights=weights,terminal_wealth=terminal,gaussian_control_terminal_wealth=control_terminal,
                coarse_terminal_wealth=coarse_terminal,maximum_drawdown=drawdown)
    tables=dict(risk=pd.DataFrame(risk),convergence=pd.DataFrame(convergence),boundaries=pd.DataFrame(bounds),
                calibration=pd.DataFrame(dict(factor=history['joined'].columns,mean_daily_log=mean,
                    historical_annual_volatility=sigma,annual_arithmetic_drift=mu)),
                correlation=pd.DataFrame(corr,index=history['joined'].columns,columns=history['joined'].columns))
    return tables,arrays


@dataclass
class RealWorldResult:
    source_folder: Path
    run_id: str
    summary: dict
    tables: dict = field(default_factory=dict)
    arrays: dict = field(default_factory=dict)

    def save(self):
        """Atomic derived output, checksummed and immutable, separate from raw data."""
        parent=self.source_folder/'local_vol_real_world';target=parent/self.run_id
        if not self.run_id.isalnum() or target.is_symlink() or target.resolve().parent!=parent.resolve():
            raise ValueError('unsafe_real_world_output')
        def verify(folder):
            manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
            if manifest.get('complete') is not True or manifest.get('identity')!=self.summary['identity']:
                raise ValueError('real_world_existing_identity_mismatch')
            expected={'summary.json','report.html'}|{k+'.csv' for k in self.tables}
            if self.arrays:expected|={'paths.npz','portfolio.png'}
            if set(manifest.get('files',{}))!=expected:raise ValueError('real_world_existing_file_set_mismatch')
            for name,digest in manifest['files'].items():
                if _digest((folder/name).read_bytes())!=digest:raise ValueError('real_world_output_checksum_mismatch')
            return folder
        if target.exists():return verify(target)
        parent.mkdir(parents=True,exist_ok=True)
        with TemporaryDirectory(prefix='.rw-',dir=parent) as temporary:
            stage=Path(temporary)/'out';stage.mkdir()
            for name,table in self.tables.items():table.to_csv(stage/(name+'.csv'),index=name=='correlation')
            write_json(stage/'summary.json',self.summary)
            if self.arrays:
                np.savez_compressed(stage/'paths.npz',**self.arrays)
                from matplotlib.figure import Figure
                from matplotlib.backends.backend_agg import FigureCanvasAgg
                fig=Figure(figsize=(10,5));FigureCanvasAgg(fig);ax=fig.subplots()
                wealth=self.arrays['portfolio_wealth'];years=self.arrays['years']
                lo,median,hi=np.quantile(wealth,[.05,.5,.95],axis=1)
                ax.fill_between(years,lo,hi,alpha=.2,label='5%-95% pathwise quantiles')
                ax.plot(years,median,label='Median');ax.plot(years,wealth.mean(axis=1),label='Mean')
                ax.set(xlabel='ACT/365F years from surface observation',ylabel='Portfolio wealth ('+self.summary['simulation_currency']+')')
                ax.legend();fig.tight_layout();fig.savefig(stage/'portfolio.png')
            (stage/'report.html').write_text(report(self),encoding='utf-8')
            files={p.name:_digest(p.read_bytes()) for p in stage.iterdir()}
            write_json(stage/'manifest.json',dict(complete=True,identity=self.summary['identity'],files=files))
            verify(stage)
            if target.exists():return verify(target)
            stage.rename(target)
        return target


def report(result):
    """Compact evidence, not a cross-model interpretation or investment forecast."""
    s=result.summary
    body='<h1>Physical-drift local-volatility portfolio simulation</h1>'
    body+='<p><b>Status:</b> '+escape(s['status'])+'</p>'
    body+='<p>'+escape(s.get('reason') or '')+'</p><ul>'+''.join('<li>'+escape(w)+'</li>' for w in s['warnings'])+'</ul>'
    body+='<p>Historical arithmetic drift; option-calibrated diffusion; fixed historical shock correlation. '
    body+='This is a physical-measure model assumption, not an estimated conversion of risk-neutral probabilities. '
    body+='Total-return proxy: B(t)=D(0,t)F(0,t)/I0; Iproxy=I0 G B. Dividends are not added again to wealth. '
    body+='All factors evolve in calendar time, including weekends. Existing historical MC uses exchange sessions. '
    body+='The Gaussian control here uses the same clock and shocks. The separate historical Gaussian/Student-t reports are not yet time-aligned comparisons: inspect clock_comparison below. No volatility cap, discarded path or horizon extrapolation.</p>'
    if result.arrays:body+='<img src="portfolio.png" style="max-width:100%"><p>paths.npz contains every path at recorded grid endpoints; native_log_growth is [time,path,factor]. '
    if result.arrays:body+='Portfolio drawdowns use the finer internal grid. Wealth is normalized total-return wealth, not an ETF share-price forecast.</p>'
    for name,frame in result.tables.items():
        if name in ('calibration_returns','calibration_covariance'):
            body+='<p><a href="'+name+'.csv">'+escape(name.replace('_',' '))+'.csv</a> ('+str(len(frame))+' rows)</p>'
            continue
        body+='<h2>'+escape(name.replace('_',' '))+'</h2><div class="scroll">'+frame.to_html(index=name=='correlation',escape=True,float_format=lambda x:f'{x:.6g}')+'</div>'
    body+='<h2>Run assumptions and provenance</h2><pre>'+escape(json.dumps({k:v for k,v in s.items() if k!='identity'},indent=2))+'</pre>'
    return '<!doctype html><html><meta charset="utf-8"><title>Physical local-volatility MC</title><style>body{font:16px system-ui;max-width:1200px;margin:32px auto;padding:20px}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccc}.scroll,pre{overflow:auto}</style><body>'+body+'</body></html>'


def simulate_processed(processed, local, surface, validation, mc_file, config_file, *, source_folder, progress=None):
    """Integrated entry point: current objects in, offline result out, no writes."""
    cfg=load_config(config_file);mc=load_monte_carlo_config(mc_file)
    from .cob import forecast_start_for_cob
    mc['forecast_start'] = forecast_start_for_cob(processed.cob)
    identity=dict(config=cfg,mc_config=mc,processing_id=processed.run_id,cob=processed.cob,
        portfolio=processed.summary['portfolio'],mapping=processed.summary.get('price_selection',[]),
        input_tables={k:_digest(v.to_csv(index=False).encode()) for k,v in processed.tables.items()
                      if k in ('selected_prices','selected_references')},
        local_id=getattr(local,'run_id',None),surface_id=getattr(surface,'run_id',None),
        validation_id=getattr(validation,'run_id',None),
        local_identity=getattr(local,'manifest',{}).get('identity'),
        validation_evidence=getattr(validation,'tables',{}).get('cohorts'),
        code=_code_fingerprints(),
        dependencies={n:version(n) for n in ('numpy','pandas','scipy','matplotlib','exchange-calendars')})
    result=RealWorldResult(Path(source_folder).resolve(),_digest(_json_bytes(identity))[:16],
        dict(identity=identity,status='unavailable',warnings=[],simulation_performed=False))
    if not cfg['enabled']:
        result.summary.update(status='disabled',reason='disabled_in_real_world_config');return result
    try:
        if surface is not None:
            declared=surface.manifest['identity']['calibration_identity'].get('source_folder')
            if declared is None or Path(declared).resolve() != result.source_folder:
                raise ValueError('surface_processing_folder_mismatch')
        if local is not None and Path(local.source_folder).resolve() != result.source_folder:
            raise ValueError('local_processing_folder_mismatch')
        history=prepare_history(processed,mc)
        if history['joined'].index[0][0] > history['sessions'][0]:
            result.summary['warnings'].append('Actual common calibration starts later than requested; see calibration coverage.')
        if history['joined'].index[-1][1] != history['sessions'][-1]:
            result.summary['warnings'].append('Calibration does not include its requested final exchange session.')
        models,rows,origin,warnings=resolve_models(processed,history,local,surface,validation,cfg)
        grid,end=simulation_window(origin,models,mc,cfg)
        result.tables['resolved_models']=pd.DataFrame(rows)
        result.summary['warnings'].extend(history['warnings']+warnings)
        result.summary['warnings'].append('Dividend carry, short-end/wing completion and fixed correlation are model assumptions; repricing does not validate physical tail probabilities.')
        tables,arrays=evolve(history,models,grid,mc,cfg,progress)
        result.tables.update(tables);result.arrays=arrays
        result.tables['calibration_returns']=history['joined'].reset_index()
        _,daily_cov=moments(history['joined'])
        result.tables['calibration_covariance']=pd.DataFrame([
            dict(factor_i=a,factor_j=b,covariance_daily_log=float(daily_cov[i,j]))
            for i,a in enumerate(history['joined'].columns) for j,b in enumerate(history['joined'].columns)])
        arrays['holding_ids']=np.array([h['instrument_id'] for h in history['holdings']],dtype=str)
        arrays['initial_value']=np.array(mc['initial_value'])
        # Store curve factors alongside native paths so the proxy level can be
        # reconstructed without reloading a surface or guessing dividend data.
        indices=sorted(models)
        arrays['proxy_holding_indices']=np.array(indices,dtype=int)
        arrays['proxy_level_per_unit_growth']=np.array([[m['adapter'].curves.at(t)[0]*m['adapter'].curves.at(t)[1]
            if m['method']=='total_return_proxy' else m['adapter'].curves.at(0.)[0]
            for i in indices for m in [models[i]]] for t in grid])
        converged=bool(tables['convergence']['passed'].all())
        result.summary.update(status='diagnostic' if warnings or not converged else 'complete_with_model_assumptions',
            simulation_performed=True,convergence_pass=converged,
            reason=None if converged else 'physical_mean_step_convergence_failed',
            simulation_currency=history['currency'],requested_currency=history['portfolio']['reporting_currency'],
            start_timestamp=origin.isoformat(),end_timestamp=end.isoformat(),
            initial_value=mc['initial_value'],paths=mc['paths'],coarse_steps=len(grid)-1,
            fine_steps=(len(grid)-1)*cfg['refinement'],day_count='ACT/365F',
            clock_comparison=clock_comparison(origin,end,mc,cfg),
            calibration=dict(first_interval=list(history['joined'].index[0]),last_interval=list(history['joined'].index[-1]),
                             observations=len(history['joined']),requested_start=history['start']),
            mean_maximum_drawdown=float(arrays['maximum_drawdown'].mean()))
    except (SimulationUnavailable,ValueError,OverflowError,FloatingPointError) as exc:
        # Never publish terminal risk from only surviving paths or early times.
        result.summary.update(status='unavailable',reason=str(exc),simulation_performed=False)
        result.arrays={}
    return result


def _read_stage(folder):
    """Read-only checksum verification, including safe manifest paths."""
    folder=Path(folder).resolve();m=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    if not isinstance(m,dict) or m.get('complete') is not True or not isinstance(m.get('files'),dict):raise ValueError('incomplete_saved_stage')
    files={}
    for key,record in m['files'].items():
        if not isinstance(record,dict) or not isinstance(record.get('file'),str):raise ValueError('invalid_saved_file_record')
        p=(folder/record['file']).resolve()
        if not p.is_relative_to(folder) or p==folder:raise ValueError('saved_stage_path_escape')
        raw=p.read_bytes()
        if _digest(raw)!=record['sha256']:raise ValueError('saved_stage_checksum_mismatch:'+key)
        files[key]=raw
    return m,files


def _csv_records(raw):
    """CSV empty numeric cells must become JSON null, not floating NaN.

    pandas keeps NaN in numeric columns if where(None) is applied before
    casting to object. Completed and skipped cohorts share a CSV schema, so
    absent counts/metrics on skipped cohorts need this explicit conversion.
    """
    # Identity fields must round-trip unchanged: "0005" is not integer 5.
    # Default NA inference would also erase the valid identifier text "NA".
    # Round-trip precision retains the original binary floats written to CSV;
    # the default fast parser can change them by an ulp and perturb a refit/MC.
    # Only empty fields denote missing values. Other columns retain numeric/
    # bool inference for calculations and explicit gate decoding below.
    identifiers = ('group_id', 'underlying', 'currency', 'instrument_id',
                   'option_reference', 'factor', 'record_id', 'request_id')
    frame=pd.read_csv(BytesIO(raw), dtype={key:str for key in identifiers},
                      keep_default_na=False, na_values=[''], float_precision='round_trip').astype(object)
    return frame.where(frame.notna(),None).to_dict('records')


def simulate_saved(processed_folder, local_folder, surface_folder, validation_folder, mc_file, config_file, progress=None):
    """Explicit saved inputs, no latest-folder guessing, downloads or refitting."""
    from .data_cleaning import ProcessedArchive
    from .surface_model import SviSurface, SurfaceModel
    folder=Path(processed_folder).resolve()
    for path in (local_folder,surface_folder,validation_folder):
        if not Path(path).resolve().is_relative_to(folder):raise ValueError('stages_must_belong_to_processing_folder')
    pm,pfiles=_read_stage(folder);lm,lfiles=_read_stage(local_folder)
    sm,_=_read_stage(surface_folder);vm,vfiles=_read_stage(validation_folder)
    tables={k:pd.read_csv(BytesIO(pfiles[k]),dtype={'instrument_id':str,'date':str},float_precision='round_trip')
            for k in ('selected_prices','selected_references') if k in pfiles}
    if pm['run_id'] != folder.name:raise ValueError('processed_folder_identity_mismatch')
    for k,table in tables.items():
        if len(table)!=pm['files'][k]['rows']:raise ValueError('processed_row_count_mismatch')
    processed=ProcessedArchive(Path(pm['archive']),pm['cob'],pm['run_id'],tables,pm)
    local=SimpleNamespace(run_id=Path(local_folder).name,source_folder=folder,manifest=lm,state_evaluators={},
                          tables={'state_mapping':_csv_records(lfiles['state_mapping'])})
    inputs=json.loads(lfiles['inputs'])
    for gid,entry in inputs.items():
        mapping=entry.get('date_spot_mapping')
        if not mapping:continue
        model=(SviSurface if entry['model'].get('model_type')=='raw_svi' else SurfaceModel)(**entry['model'])
        ev=LocalVolEvaluator(model,entry['permission'],entry['minimum_density_factor'],entry['negative_tolerance'])
        local.state_evaluators[gid]=DateSpotLocalVol(ev,mapping['reference_nodes'],mapping['valuation_date'],mapping['pricing'])
    # CSV bool/list fields need explicit decoding: bool('False') is True.
    import ast
    rows=_csv_records(vfiles['cohorts'])
    for row in rows:
        for key in ('fit_pass','pde_pass','convergence_pass','precision_pass','forward_pass'):
            value=row.get(key)
            row[key]=True if value is True or str(value)=='True' else False if value is False or str(value)=='False' else None
        if isinstance(row.get('failed_checks'),str):row['failed_checks']=ast.literal_eval(row['failed_checks'])
    validation=SimpleNamespace(run_id=Path(validation_folder).name,manifest=vm,tables={'cohorts':rows})
    surface=SimpleNamespace(run_id=Path(surface_folder).name,manifest=sm)
    return simulate_processed(processed,local,surface,validation,mc_file,config_file,source_folder=folder,progress=progress)
