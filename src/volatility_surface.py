"""Select, normalize, fit and audit saved option IVs. No provider connections.

The input is the CalibrationResult returned in this execution, not a hardcoded
path or a search for a 'latest' directory. Every rejected observation and failed
cohort remains visible. Numerical fit success, price quality and data timing
are separate judgments. An intraday fit is always an approximation.
"""
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from zoneinfo import ZoneInfo
import csv
import hashlib
import io
import json
import math
import sys
from tempfile import TemporaryDirectory
import uuid

import numpy as np

from .ois_bootstrap import DiscountCurve
from .storage import write_json
from .option_calibration import _curve_time, _read_json
from .surface_model import SurfaceModel, fit_ssvi, fitted_prices, validate_grid, fit_raw_svi
from .surface_checks import (diagnostics, quote_quality, summarize_quality, coverage_statistics,
    stability_decision, aggregate_stability, assess_holdout, compare_derivatives,
    maturity_jumps, refit_assessment, validate_policy as validate_derivative_policy,
    validate_eligibility_policy, assess_local_vol_eligibility)
from .surface_reporting import render_report, render_html



def _bytes(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode('utf-8')


def _hash(value):
    return hashlib.sha256(value).hexdigest()


def load_surface_config(path):
    """Reject malformed user policies as ValueError before touching any data."""
    cfg=_read_json(path)
    sections={
        'timing':{'mode','maximum_capture_minutes','maximum_spot_range_fraction','maximum_carry_dispersion_fraction'},
        'selection':{'minimum_expiries','minimum_unique_strikes','minimum_each_side','minimum_near_forward',
            'near_forward_abs_k','maximum_abs_k','maximum_relative_spread','maximum_expiry_gap_years','maximum_strike_gap_k'},
        'fit':{'initial_rhos','maximum_evaluations'},
        'quality':{'price_error_floor_spot_fraction','historical_relative_error','minimum_inside_spread_fraction','maximum_scaled_price_rmse'},
        'stability':{'enabled','holdout_stride','maximum_holdout_scaled_rmse','maximum_iv_change','seed','narrow_window_minutes'},
        'grid':{'moneyness_points','maturity_points','derivative_tolerance'}}
    # Schema 1 remains the original SSVI policy. Schema 2 explicitly chooses
    # the richer model; schema 3 changes acceptance, not the fitting algorithm.
    # Old callers retain strict-spread acceptance until explicitly migrated.
    schema=cfg.get('schema_version') if isinstance(cfg,dict) else None
    if schema in (2,3):
        sections['fit']|={'model','svi'}
        sections['diagnostics']={'forward_bump_fraction','block_holdout_fraction','check_mid_expiry'}
    if schema==3:
        sections['quality'].remove('minimum_inside_spread_fraction')
        sections['quality']|={'minimum_practical_coverage','maximum_relative_outside_error',
                              'maximum_outside_half_spreads','maximum_outside_iv_error'}
    if isinstance(cfg,dict) and 'eligibility' in cfg:
        validate_eligibility_policy(cfg['eligibility'])
        sections['eligibility']=set(cfg['eligibility'])
    if isinstance(cfg,dict) and isinstance(cfg.get('diagnostics'),dict) and 'extended' in cfg['diagnostics']:
        if 'diagnostics' not in sections:raise ValueError('diagnostics_require_surface_schema_2_or_3')
        if type(cfg['diagnostics']['extended']) is not bool:raise ValueError('invalid_extended_diagnostics')
        sections['diagnostics'].add('extended')
    if isinstance(cfg,dict) and 'derivative_diagnostics' in cfg:
        validate_derivative_policy(cfg['derivative_diagnostics'])
        sections['derivative_diagnostics']=set(cfg['derivative_diagnostics'])
    if (not isinstance(cfg,dict) or set(cfg)!=set(sections)|{'schema_version','enabled','plots','instructions'}
            or type(schema) is not int or schema not in (1,2,3)):
        raise ValueError('invalid_surface_schema')
    # Optional explicit policies preserve the former all-quotes/required-time
    # semantics for callers that have not opted into supported-domain review.
    if isinstance(cfg.get('stability'),dict):
        sections['stability']|=set(cfg['stability']) & {'maturity_holdout_scope','timing_role'}
    for name,keys in sections.items():
        if not isinstance(cfg[name],dict) or set(cfg[name])!=keys:
            raise ValueError('invalid_surface_policy_keys: '+name)
    if not isinstance(cfg['instructions'],dict):raise ValueError('invalid_surface_instructions')
    for obj,key in ((cfg,'enabled'),(cfg,'plots'),(cfg['stability'],'enabled')):
        if type(obj[key]) is not bool:raise ValueError('invalid_surface_boolean: '+key)
    if cfg['timing']['mode'] not in ('intraday_approximation','strict_snapshot'):
        raise ValueError('invalid_surface_timing_mode')
    if cfg['stability'].get('maturity_holdout_scope','all_quotes') not in ('all_quotes','supported_domain'):
        raise ValueError('invalid_maturity_holdout_scope')
    if cfg['stability'].get('timing_role','required') not in ('required','diagnostic'):
        raise ValueError('invalid_stability_timing_role')
    integers={'minimum_expiries':2,'minimum_unique_strikes':5,'minimum_each_side':1,'minimum_near_forward':1,
              'maximum_evaluations':10,'holdout_stride':2,'seed':0,'moneyness_points':7,'maturity_points':3}
    for name in sections:
        if name in ('derivative_diagnostics','eligibility'):continue  # Independently validated nested policy above.
        for key,value in cfg[name].items():
            if key in ('mode','initial_rhos','enabled','model','svi','check_mid_expiry','maturity_holdout_scope','timing_role','extended'):continue
            if key in integers:
                if type(value) is not int or value<integers[key]:raise ValueError('invalid_surface_integer: '+key)
            elif type(value) not in (int,float) or not math.isfinite(value) or value<=0:
                raise ValueError('invalid_surface_positive_policy: '+key)
    # Decimal units, not displayed percentages. Reject accidental 1 or 25
    # instead of 0.01 or 0.0025; legacy schema policies keep their own meaning.
    if schema==3 and (cfg['quality']['maximum_relative_outside_error']>=1
                      or cfg['quality']['maximum_outside_iv_error']>=1):
        raise ValueError('practical_thresholds_require_decimal_fractions')
    rhos=cfg['fit']['initial_rhos']
    if not isinstance(rhos,list) or not rhos or any(type(r) not in (int,float) or not math.isfinite(r) or abs(r)>.95 for r in rhos):
        raise ValueError('invalid_surface_initial_rhos')
    if cfg['quality'].get('minimum_practical_coverage',cfg['quality'].get('minimum_inside_spread_fraction'))>1 or cfg['selection']['near_forward_abs_k']>cfg['selection']['maximum_abs_k']:
        raise ValueError('inconsistent_surface_thresholds')
    if schema in (2,3):
        if cfg['fit']['model'] not in ('ssvi','raw_svi'):raise ValueError('invalid_surface_model')
        keys={'midpoint_weight','constraint_points','audit_points','constraint_iterations','constraint_refinements','arbitrage_tolerance'}
        svi=cfg['fit']['svi']
        if not isinstance(svi,dict) or set(svi)!=keys:raise ValueError('invalid_svi_policy')
        for name,minimum in [('constraint_points',21),('audit_points',101),('constraint_iterations',10),('constraint_refinements',1)]:
            if type(svi[name]) is not int or svi[name]<minimum:raise ValueError('invalid_svi_integer: '+name)
        for name in ('midpoint_weight','arbitrage_tolerance'):
            if type(svi[name]) not in (int,float) or not math.isfinite(svi[name]) or not 0<svi[name]<1:
                raise ValueError('invalid_svi_fraction: '+name)
        if svi['audit_points']<=svi['constraint_points']:raise ValueError('svi_audit_grid_must_be_denser')
        if svi['arbitrage_tolerance']>1e-7:raise ValueError('svi_arbitrage_tolerance_too_large')
        if (type(cfg['diagnostics']['check_mid_expiry']) is not bool
                or not 0<cfg['diagnostics']['block_holdout_fraction']<.4
                or cfg['diagnostics']['forward_bump_fraction']>=.01):raise ValueError('invalid_surface_diagnostics')
    return cfg


def _fit(points,policy):
    # No fallback between algorithms: failed SVI remains a reported failure.
    return fit_raw_svi(points,policy) if policy.get('model')=='raw_svi' else fit_ssvi(points,policy)


def _number(value):
    if isinstance(value,bool):raise ValueError('non_numeric_input')
    try:value=float(value)
    except (TypeError,ValueError) as exc:raise ValueError('missing_numeric_input') from exc
    if not math.isfinite(value):raise ValueError('nonfinite_input')
    return value


def _stamp(value):
    if not isinstance(value,str):raise ValueError('missing_surface_timestamp')
    result=datetime.fromisoformat(value)
    if result.tzinfo is None:raise ValueError('surface_timestamp_without_timezone')
    return result.astimezone(timezone.utc)


def _coverage(points,cfg):
    """Coverage counts independent strikes, not repeated calls/puts/providers."""
    s=cfg['selection']; by_expiry=defaultdict(list); rejected={}
    for p in points:by_expiry[p['expiration']].append(p)
    for expiry,group in by_expiry.items():
        counts=coverage_statistics(group,s)
        if not counts['count_pass']:
            rejected[expiry]='insufficient_strike_coverage'
        elif not counts['gap_pass']:
            rejected[expiry]='large_strike_gap'
    kept=[p for p in points if p['expiration'] not in rejected]
    times=sorted({p['reference_years'] for p in kept})
    if len(times)<s['minimum_expiries']:return kept,rejected,'insufficient_expiry_coverage'
    if max(np.diff(times))>s['maximum_expiry_gap_years']:return kept,rejected,'large_maturity_gap'
    return kept,rejected,None


def _prepare_group(rows,calibration,cfg,group_id):
    """Retain provenance while mapping a bounded capture to reference maturity.

    The original IV, k, F, D and tau are never overwritten. The normalization
    holds IV(k) stable over capture, not the raw option premium. Reference
    dividend yield is only an equivalent carry representation of saved F,D,S.
    """
    selection=[]; points=[]; s=cfg['selection']; quality=cfg['quality']
    for ordinal,row in rows:
        record=dict(record_id=ordinal,group_id=group_id,instrument_id=row.get('instrument_id'),
                    provider=row.get('provider'),expiration=row.get('expiration'),selected=False,reason=None)
        selection.append(record)
        try:
            if row.get('status')!='iv_calculated_with_assumptions':raise ValueError('individual_iv_not_accepted')
            if row.get('exercise_style')!='european' or row.get('settlement') not in ('cash','physical') or not row.get('settlement_time'):
                raise ValueError('missing_or_unsupported_contract_conventions; reprocess_saved_archive')
            if row.get('option_type') not in ('call','put'):raise ValueError('invalid_option_right')
            if row.get('route') not in ('live','historical'):raise ValueError('unknown_option_route')
            p=deepcopy(row);p['record_id']=ordinal
            for key in ('iv','forward','strike','time_to_expiry','discount_factor','option_price','spot','vega_per_unit_volatility'):
                p[key]=_number(row.get(key))
                if p[key]<=0:raise ValueError('nonpositive_'+key)
            p['k']=math.log(p['strike']/p['forward'])
            if abs(p['k'])>s['maximum_abs_k']:raise ValueError('outside_moneyness_selection')
            stamp=_stamp(p['valuation_timestamp']); expiry=_stamp(p['expiry_timestamp'])
            zone=calibration.manifest['identity']['config']['valuation_timezone']
            if stamp.astimezone(ZoneInfo(zone)).date().isoformat()!=p['valuation_date'] or p.get('curve_date')!=p['valuation_date']:
                raise ValueError('surface_date_mismatch')
            actual=(expiry-stamp).total_seconds()/(365*86400)
            if not math.isclose(actual,p['time_to_expiry'],rel_tol=1e-9,abs_tol=1e-12):raise ValueError('inconsistent_option_lifetime')
            floor=p['spot']*quality['price_error_floor_spot_fraction']
            if p['route']=='live':
                p['bid']=_number(row.get('bid'));p['ask']=_number(row.get('ask'))
                if not 0<p['bid']<=p['ask']:raise ValueError('invalid_surface_bid_ask')
                if not math.isclose((p['bid']+p['ask'])/2,p['option_price'],rel_tol=1e-8):raise ValueError('inconsistent_midpoint')
                p['relative_spread']=(p['ask']-p['bid'])/p['option_price']
                if p['relative_spread']>s['maximum_relative_spread']:raise ValueError('wide_surface_spread')
                p['error_scale']=max((p['ask']-p['bid'])/2,floor)
            else:
                p['relative_spread']=None
                p['error_scale']=max(p['option_price']*quality['historical_relative_error'],floor)
            points.append(p)
        except (ValueError,KeyError,TypeError) as exc:record['reason']=str(exc)
    lookup={r['record_id']:r for r in selection}
    if not points:return [],selection,[],{'status':'unavailable','reason':'no_usable_iv_observations'}
    # Conflicting duplicates are rejected; then select the most informative
    # call/put/provider at each strike. Stable tie breaking is reproducible.
    by_contract=defaultdict(list)
    for p in points:by_contract[(p['expiration'],p['strike'],p['option_type'])].append(p)
    unique=[]
    for group in by_contract.values():
        signatures={(p['option_price'],p['forward'],p['discount_factor'],p['valuation_timestamp']) for p in group}
        if len(signatures)>1:
            for p in group:lookup[p['record_id']]['reason']='conflicting_duplicate_contract'
        else:unique.extend(group)
    by_strike=defaultdict(list)
    for p in unique:by_strike[(p['expiration'],p['strike'])].append(p)
    points=[]
    for group in by_strike.values():
        ranked=sorted(group,key=lambda p:(p['error_scale']/p['vega_per_unit_volatility'],p['record_id']))
        points.append(ranked[0])
        for p in ranked[1:]:lookup[p['record_id']]['reason']='less_informative_duplicate_strike'
    if not points:return [],selection,[],{'status':'unavailable','reason':'no_unique_observations'}
    points.sort(key=lambda p:(p['expiration'],p['strike']))
    ordered=sorted(points,key=lambda p:(_stamp(p['valuation_timestamp']),p['record_id']))
    anchor=ordered[(len(ordered)-1)//2]; ref=_stamp(anchor['valuation_timestamp']); spot=anchor['spot']
    spans=[_stamp(p['valuation_timestamp']) for p in points]
    span=(max(spans)-min(spans)).total_seconds()/60
    spot_range=(max(p['spot'] for p in points)-min(p['spot'] for p in points))/float(np.median([p['spot'] for p in points]))
    info=dict(status='prepared',reference_timestamp=ref.isoformat(),reference_spot=spot,
        capture_minutes=span,spot_range_fraction=spot_range,timing_policy=cfg['timing']['mode'],
        timing_quality='intraday_approximation' if cfg['timing']['mode']=='intraday_approximation' else 'bounded_snapshot_approximation')
    reason=None
    if span>cfg['timing']['maximum_capture_minutes']:reason='capture_window_exceeded'
    if spot_range>cfg['timing']['maximum_spot_range_fraction']:reason='underlying_moved_too_far'
    if reason:
        for p in points:lookup[p['record_id']]['reason']=reason
        return [],selection,[],{**info,'status':'unavailable','reason':reason}
    day,currency=anchor['valuation_date'],anchor['currency']
    definition=calibration.manifest.get('curve_definitions',{}).get(day,{}).get(currency)
    if not definition or definition.get('as_of_date')!=day:
        for p in points:lookup[p['record_id']]['reason']='missing_reference_curve'
        return [],selection,[],{**info,'status':'unavailable','reason':'missing_reference_curve'}
    # Saved live quotes arrived asynchronously. The opted-in intraday
    # approximation retains their original pricing inputs for fit residuals
    # and constructs one reference spot/carry for later dynamics. It does NOT
    # reconstruct synchronized exchange quotes. Timing/dispersion gates limit,
    # but cannot eliminate, that uncertainty. Live/historical cohorts stay apart.
    curve=DiscountCurve(**definition)
    references=[]; normalized=[]
    for expiry in sorted({p['expiration'] for p in points}):
        group=[p for p in points if p['expiration']==expiry]
        try:
            ends={p['expiry_timestamp'] for p in group}
            if len(ends)!=1:raise ValueError('conflicting_expiry_timestamps')
            end=_stamp(ends.pop()); t=(end-ref).total_seconds()/(365*86400)
            if t<=0:raise ValueError('expired_at_reference_time')
            d=curve.interval_discount(_curve_time(ref,curve,zone),_curve_time(end,curve,zone))
            yields=np.array([-math.log(p['forward']*p['discount_factor']/p['spot'])/p['time_to_expiry'] for p in group])
            prepaid=np.exp(-yields*t)
            if not np.isfinite(prepaid).all():raise ValueError('invalid_reference_carry')
            dispersion=float(np.ptp(prepaid))
            if dispersion>cfg['timing']['maximum_carry_dispersion_fraction']:raise ValueError('inconsistent_expiry_carry')
            q=float(np.median(yields)); forward=spot*math.exp(-q*t)/d
            references.append(dict(group_id=group_id,expiration=expiry,reference_timestamp=ref.isoformat(),
                expiry_timestamp=end.isoformat(),years=t,spot=spot,discount_factor=d,forward=forward,
                equivalent_dividend_yield=q,carry_dispersion_fraction=dispersion,
                carry_method='median_equivalent_yield_from_original_F_D_S_tau',curve_date=day))
            for p in group:
                p['reference_years']=t;p['group_id']=group_id
                normalized.append(p)
        except (ValueError,OverflowError) as exc:
            for p in group:lookup[p['record_id']]['reason']=str(exc)
    # Persist the actual population on which coverage is decided, including
    # expiries later rejected. This is evidence only; thresholds are unchanged.
    info['coverage_checks']=[dict(group_id=group_id,expiration=expiry,
        population='eligible_before_coverage',years=group[0]['reference_years'],
        **coverage_statistics(group,s))
        for expiry in sorted({p['expiration'] for p in normalized})
        for group in [[p for p in normalized if p['expiration']==expiry]]]
    kept,rejected,error=_coverage(normalized,cfg)
    for p in normalized:
        record=lookup[p['record_id']]
        record.update(k=p['k'],iv=p['iv'],reference_years=p['reference_years'],error_scale=p['error_scale'])
        if p['expiration'] in rejected:record['reason']=rejected[p['expiration']]
        elif error:record['reason']=error
        else:record['selected']=True
    if error:return [],selection,references,{**info,'status':'unavailable','reason':error}
    references=[r for r in references if r['expiration'] in {p['expiration'] for p in kept}]
    return kept,selection,references,info


def _metrics(model,points,quality=None):
    prices,ivs=fitted_prices(model,points)
    residual=np.array([(v-p['option_price'])/p['error_scale'] for v,p in zip(prices,points)])
    rows=[quote_quality(p,float(price),float(iv),quality) for p,price,iv in zip(points,prices,ivs)]
    live=[r['inside_spread'] for p,r in zip(points,rows) if p['route']=='live']
    return dict(scaled_price_rmse=float(np.sqrt(np.mean(residual**2))),
        iv_rmse=float(np.sqrt(np.mean((ivs-np.array([p['iv'] for p in points]))**2))),
        inside_spread_fraction=float(np.mean(live)) if live else None,observations=len(points),
        **summarize_quality(points,rows,quality))


def _model_distance(a,b):
    """Compare IVs only on the common supported domain, never extrapolate."""
    low=max(a.times[0],b.times[0]);high=min(a.times[-1],b.times[-1]);d=[]
    if high<=low:return None
    for t in np.linspace(low,high,12):
        alo,ahi=a.domain(t);blo,bhi=b.domain(t)
        lo=max(alo,blo);hi=min(ahi,bhi)
        if hi<=lo:continue
        k=np.linspace(lo,hi,25)
        d.extend(np.abs(a.evaluate(k,t)['iv']-b.evaluate(k,t)['iv']).tolist())
    return float(max(d)) if d else None


def _stability(model,points,candidates,cfg,progress=None):
    """Diagnostics are bounded, deterministic, and cannot modify source points."""
    if not cfg['stability']['enabled']:return [],'not_checked'
    policy=cfg['stability']; records=deepcopy(getattr(model,'start_checks',[]))
    derivative_policy=cfg.get('derivative_diagnostics',{})
    for cost,other,_ in candidates[1:]:
        if cost<=candidates[0][0]*1.05+1e-8:
            change=_model_distance(model,other)
            record=dict(check='alternative_start',status='checked',max_iv_change=change)
            if derivative_policy.get('enabled'):
                record['derivative_diagnostics']=compare_derivatives(model,other,derivative_policy)
            records.append(record)
    # Hold out interior strikes; preserve extrema so withheld observations are
    # inside the training domain. No bid/ask or synthetic quote is invented.
    hold=[];training=[]
    for expiry in sorted({p['expiration'] for p in points}):
        group=sorted([p for p in points if p['expiration']==expiry],key=lambda p:p['k'])
        for j,p in enumerate(group):
            (hold if j>0 and j<len(group)-1 and j%policy['holdout_stride']==0 else training).append(p)
    subsets=[('held_out_strikes',training)]
    validation_sets={'held_out_strikes':hold}
    if cfg['schema_version']>=2:
        # A contiguous central block is harder to interpolate than alternating
        # held-out strikes. Keep endpoint strikes to avoid wing extrapolation.
        training=[];held=[]
        for expiry in sorted({p['expiration'] for p in points}):
            group=sorted([p for p in points if p['expiration']==expiry],key=lambda p:p['k'])
            width=max(1,int(len(group)*cfg['diagnostics']['block_holdout_fraction']))
            start=max(1,(len(group)-width)//2);end=min(len(group)-1,start+width)
            held.extend(group[start:end]);training.extend(group[:start]+group[end:])
        subsets.append(('held_out_strike_block',training));validation_sets['held_out_strike_block']=held
        expiries=sorted({p['expiration'] for p in points})
        if cfg['diagnostics']['check_mid_expiry'] and len(expiries)>2:
            removed=expiries[len(expiries)//2]
            subsets.append(('held_out_maturity',[p for p in points if p['expiration']!=removed]))
            validation_sets['held_out_maturity']=[p for p in points if p['expiration']==removed]

    rng=np.random.default_rng(policy['seed'])
    noise=rng.uniform(-1,1,len(points))
    for sign in (1,-1):
        changed=deepcopy(points)
        for i,p in enumerate(changed):
            amplitude=(p['ask']-p['bid'])/2 if p['route']=='live' else p['error_scale']
            p['option_price']+=sign*noise[i]*amplitude
        subsets.append(('quote_perturbation_'+str(sign),changed))
    strict=[p for p in points if abs(p['k'])<=cfg['selection']['maximum_abs_k']*.8
            and (p['relative_spread'] is None or p['relative_spread']<=cfg['selection']['maximum_relative_spread']*.5)]
    subsets.append(('stricter_selection',strict))
    if points[0]['route']=='live':
        stamps=np.array([_stamp(p['valuation_timestamp']).timestamp() for p in points])
        width=policy['narrow_window_minutes']*60
        for label,start in zip(('first','middle','last'),(min(stamps),(min(stamps)+max(stamps)-width)/2,max(stamps)-width)):
            subsets.append(('capture_'+label,[p for p,t in zip(points,stamps) if start<=t<=start+width]))
    for name,subset in subsets:
        if progress:progress("Surface stability check: " + name)
        timing=name.startswith('capture_')
        context=dict(category='timing' if timing else 'numerical',
            required=not timing or policy.get('timing_role','required')=='required')
        kept,_,error=_coverage(subset,cfg)
        if error:
            records.append(dict(check=name,status='insufficient_coverage',reason=error,**context));continue
        other,_,_=_fit(kept,cfg['fit'])
        if other is None:
            records.append(dict(check=name,status='optimizer_failed',**context));continue
        record=dict(check=name,status='checked',max_iv_change=_model_distance(model,other),
                    expiries=len(other.times),full_maturity_coverage=other.times==model.times,**context)
        if name in validation_sets:
            supported=(name=='held_out_maturity' and
                       policy.get('maturity_holdout_scope','all_quotes')=='supported_domain')
            valid,evidence=assess_holdout(other,validation_sets[name],cfg['selection'],supported=supported)
            record.update(evidence)
            if valid:record['holdout_scaled_rmse']=_metrics(other,valid)['scaled_price_rmse']
            if not evidence['scope_coverage_complete']:record['status']='insufficient_coverage'
            # Legacy all-quotes mode deliberately omitted one training expiry;
            # all of its quotes must be evaluated to treat that removal as OK.
            if name=='held_out_maturity' and not supported and evidence['scope_coverage_complete']:
                record['full_maturity_coverage']=True
        if derivative_policy.get('enabled'):
            record['derivative_diagnostics']=compare_derivatives(model,other,derivative_policy)
        records.append(record)
    for record in records:
        record.setdefault('category','numerical');record.setdefault('required',True)
        record['decision']=stability_decision(record,policy)
    return records,aggregate_stability(records,policy)



@dataclass
class SurfaceResult:
    folder: Path
    run_id: str
    manifest: dict
    tables: dict
    models: dict

    def _verify_saved(self,target,expected):
        """Completed outputs are immutable: any mismatch still fails closed."""
        saved=_read_json(target/'manifest.json')
        # A syntactically valid JSON value can still have the wrong shape.
        # Report a controlled integrity error instead of leaking AttributeError.
        if not isinstance(saved,dict) or not isinstance(saved.get('files'),dict):
            raise ValueError('surface_existing_output_inconsistent')
        files=saved['files']
        if saved.get('identity')!=self.manifest['identity'] or saved.get('complete') is not True or set(files)!=set(expected):
            raise ValueError('surface_existing_output_inconsistent')
        for name,filename in expected.items():
            rec=files[name]
            if not isinstance(rec,dict) or rec.get('file')!=filename or _hash((target/filename).read_bytes())!=rec.get('sha256'):
                raise ValueError('surface_existing_output_checksum_mismatch')
        return target

    def save(self):
        """Publish a complete directory atomically, preserving older partial work.

        Render/plot failures leave no final directory. A terminated process may
        leave a private staging directory, which cannot block a retry. Older
        incomplete final directories are renamed and preserved, never deleted.
        Completed outputs must still pass all identity/checksum checks.
        """
        if (not isinstance(self.run_id,str) or not self.run_id or self.run_id in ('.','..')
                or any(c in self.run_id for c in '/\\:')):
            raise ValueError('invalid_surface_output_id')
        parent=self.folder/'volatility_surface';target=parent/self.run_id
        if target.is_symlink() or target.resolve().parent!=parent.resolve():
            raise ValueError('unsafe_surface_output_path')
        plots=self.manifest['identity']['config']['plots']
        expected={name:name+'.csv' for name in self.tables}
        expected.update(parameters='parameters.json',eligibility='eligibility.json',report='report.md',report_html='report.html')
        if plots:
            for gid in self.models:
                for kind in ('smiles','surface','diagnostics'):expected[gid+'_'+kind]=gid+'_'+kind+'.png'
                expected[gid+'_interactive']=gid+'_3d.html'
        def complete_or_recoverable():
            if not target.exists():return False
            if not target.is_dir():raise ValueError('surface_output_is_not_directory')
            if (target/'manifest.json').exists():
                saved=_read_json(target/'manifest.json')
                if not isinstance(saved,dict):
                    raise ValueError('surface_existing_output_inconsistent')
                if saved.get('complete') is True:
                    self._verify_saved(target,expected);return True
                if saved.get('complete') is not False or saved.get('identity')!=self.manifest['identity']:
                    raise ValueError('surface_incomplete_output_identity_mismatch')
            return False  # No completion manifest: preserve and rebuild.

        if complete_or_recoverable():return target
        parent.mkdir(parents=True,exist_ok=True)
        # Keep the temporary basename short for Windows/OneDrive path limits.
        # Same filesystem makes final publication a directory rename. Cleanup
        # owns only this newly created child, never any existing result folder.
        with TemporaryDirectory(prefix='.p-',dir=parent) as temporary:
            owned=Path(temporary)
            if owned.resolve().parent!=parent.resolve():raise ValueError('unsafe_surface_staging_path')
            stage=owned/'output';stage.mkdir()
            for name,rows in self.tables.items():
                stream=io.StringIO(newline='');fields=list(dict.fromkeys(k for r in rows for k in r)) or ['status']
                writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
                for r in rows:writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})
                (stage/(name+'.csv')).write_text(stream.getvalue(),encoding='utf-8',newline='')
            write_json(stage/'parameters.json',{gid:asdict(model) for gid,model in self.models.items()})
            write_json(stage/'eligibility.json',{r['group_id']:r['local_vol_eligibility'] for r in self.tables['cohort_status']})
            markdown=render_report(self)
            (stage/'report.md').write_text(markdown,encoding='utf-8')
            (stage/'report.html').write_text(render_html(markdown,self),encoding='utf-8')
            if plots:
                from .surface_reporting import plot_surfaces
                plot_surfaces(self,stage)
                from .surface_view import write_implied_view
                for gid in self.models:write_implied_view(self,gid,stage/(gid+'_3d.html'))
            files={name:dict(file=filename,sha256=_hash((stage/filename).read_bytes())) for name,filename in expected.items()}
            # Another run may have completed while this one rendered charts.
            if complete_or_recoverable():return target
            backup=parent/('.i-'+uuid.uuid4().hex) if target.exists() else None
            manifest={**self.manifest,'files':files,'complete':True}
            if backup is not None:manifest['recovered_incomplete_output']=backup.name
            write_json(stage/'manifest.json',manifest)
            self._verify_saved(stage,expected)
            if backup is not None:target.rename(backup)
            try:
                stage.rename(target)
            except OSError:
                if backup is not None and not target.exists():backup.rename(target)
                if target.exists() and (target/'manifest.json').exists():
                    return self._verify_saved(target,expected)
                raise
        return target


def fit_calibrated(calibration,config_file,progress=None):
    """Entry point used by run.py: calibration result in, audited surfaces out."""
    cfg=load_surface_config(config_file)
    tables={name:[] for name in ('selection','coverage_checks','reference_inputs','cohort_status','optimizer','repricing','expiry_metrics','surface_grid','stability','stability_unsupported','stability_domains','derivative_refits','derivative_refit_metrics','derivative_refit_points','derivative_refit_domains','maturity_jumps','maturity_jump_points','maturity_jump_domains','quote_diagnostics','timing_diagnostics','residual_buckets','interval_feasibility','model_comparison')}
    models={};groups=defaultdict(list)
    for i,row in enumerate(calibration.tables['option_iv']):
        key=tuple(row.get(k) for k in ('underlying','currency','valuation_date','route','exercise_style','settlement','settlement_time','price_type'))
        groups[key].append((i,row))
    for key,rows in sorted(groups.items(),key=lambda pair:str(pair[0])):
        gid=_hash(_bytes(key))[:12]
        base=dict(group_id=gid,underlying=key[0],currency=key[1],valuation_date=key[2],route=key[3])
        if not cfg['enabled']:
            tables['cohort_status'].append({**base,'status':'skipped','reason':'surface_disabled'});continue
        if progress:progress(f"Surface {key[0]} {key[2]}: selecting {len(rows)} observations.")
        points,selection,refs,info=_prepare_group(rows,calibration,cfg,gid)
        tables['selection'].extend(selection);tables['reference_inputs'].extend(refs)
        tables['coverage_checks'].extend(info.pop('coverage_checks',[]))
        if not points:
            tables['cohort_status'].append({**base,**info});continue
        model,attempts,candidates=_fit(points,cfg['fit'])
        tables['optimizer'].extend(dict(group_id=gid,**r) for r in attempts)
        if model is None:
            tables['cohort_status'].append({**base,**info,'status':'unavailable','reason':'fit_or_arbitrage_failure' if cfg['fit'].get('model')=='raw_svi' else 'optimizer_failed'});continue
        models[gid]=model
        if cfg['schema_version']>=2 and cfg['diagnostics'].get('extended',True):
            diagnostic_tables=diagnostics(points,model,cfg['diagnostics'],cfg['selection']['near_forward_abs_k'])
            for name,rows in diagnostic_tables.items():tables[name].extend(dict(group_id=gid,**r) for r in rows)
            # Benchmark on the SAME selected quotes; no independent filtering
            # gives the flexible model an unfair improvement in reported fit.
            if cfg['fit']['model']=='raw_svi':
                baseline,_,_=fit_ssvi(points,cfg['fit'])
                if baseline is not None:
                    for expiry in sorted({p['expiration'] for p in points}):
                        group=[p for p in points if p['expiration']==expiry]
                        for name,candidate in [('ssvi',baseline),('raw_svi',model)]:
                            tables['model_comparison'].append(dict(group_id=gid,expiration=expiry,model=name,**_metrics(candidate,group,cfg['quality'])))
        grid,validation=validate_grid(model,cfg['grid'])
        tables['surface_grid'].extend(dict(group_id=gid,**r) for r in grid)
        prices,ivs=fitted_prices(model,points)
        for p,price,iv in zip(points,prices,ivs):
            tables['repricing'].append(dict(group_id=gid,record_id=p['record_id'],instrument_id=p['instrument_id'],
                provider=p.get('provider'),option_type=p['option_type'],price_type=p.get('price_type'),
                strike=p['strike'],original_spot=p['spot'],original_forward=p['forward'],
                original_discount_factor=p['discount_factor'],original_years=p['time_to_expiry'],
                expiration=p['expiration'],k=p['k'],years=p['reference_years'],observed_iv=p['iv'],fitted_iv=float(iv),
                observed_price=p['option_price'],fitted_price=float(price),price_error=float(price-p['option_price']),
                scaled_price_error=float((price-p['option_price'])/p['error_scale']),
                bid=p.get('bid'),ask=p.get('ask'),bid_iv=p.get('bid_iv'),ask_iv=p.get('ask_iv'),
                **quote_quality(p,float(price),float(iv),cfg['quality']),
                valuation_timestamp=p['valuation_timestamp'],comparison_basis='original_F_D_tau; stable_IV_at_original_k'))
        # Passing-quote coverage is descriptive. Do not use model residuals to
        # prune observations and then certify the same fitted model on a subset.
        repriced={r['record_id']:r for r in tables['repricing'] if r['group_id']==gid}
        for expiry in sorted({p['expiration'] for p in points}):
            group=[p for p in points if p['expiration']==expiry]
            for population,flag in [('strict_passing_diagnostic','inside_spread'),
                                    ('practical_passing_diagnostic','practical_pass')]:
                if group[0]['route']!='live' or (flag=='practical_pass' and cfg['schema_version']!=3):continue
                subset=[p for p in group if repriced[p['record_id']].get(flag) is True]
                tables['coverage_checks'].append(dict(group_id=gid,expiration=expiry,population=population,
                    years=group[0]['reference_years'],**coverage_statistics(subset,cfg['selection'])))
        metrics=[]
        for expiry in sorted({p['expiration'] for p in points}):
            metric=dict(group_id=gid,expiration=expiry,**_metrics(model,[p for p in points if p['expiration']==expiry],cfg['quality']))
            metric['price_rmse_pass']=metric['scaled_price_rmse']<=cfg['quality']['maximum_scaled_price_rmse']
            # Schema 3: exact bid/ask coverage remains visible but is not an
            # acceptance gate. Missing IV bounds stay in the live denominator.
            # Historical-only groups still use RMSE, without invented spreads.
            practical=cfg['schema_version']==3
            metric['spread_coverage_pass']=None if practical else (metric['inside_spread_fraction'] is None or metric['inside_spread_fraction']>=cfg['quality']['minimum_inside_spread_fraction'])
            coverage_pass=(metric['practical_coverage_pass'] if metric['practical_live_observations'] else True) if practical else metric['spread_coverage_pass']
            metric['market_fit_pass']=bool(metric['price_rmse_pass'] and coverage_pass)
            metric['coverage_policy']='practical_intervals' if practical and metric['practical_live_observations'] else 'historical_rmse_only' if not metric['practical_live_observations'] else 'legacy_strict_spread'
            metrics.append(metric);tables['expiry_metrics'].append(metric)
        price_pass=all(m['market_fit_pass'] for m in metrics)
        if progress:progress(f"Surface {key[0]}: fit converged; checking stability and quote repricing.")
        sensitivity,stable=_stability(model,points,candidates,cfg,progress=progress)
        tables['stability'].extend(dict(group_id=gid,**{k:v for k,v in r.items() if k!='derivative_diagnostics'}) for r in sensitivity)
        # Retain excluded evidence as separate inspectable tables, without
        # removing these contracts from the original fitted surface.
        for check in sensitivity:
            tables['stability_unsupported'].extend(dict(group_id=gid,check=check['check'],**r)
                for r in check.get('unsupported_observations',[]))
            tables['stability_domains'].extend(dict(group_id=gid,check=check['check'],**r)
                for r in check.get('validation_domains',[]))
        # Derivative sensitivity is a separate review stage: it cannot silently
        # change the fitted surface, quote selection or original acceptance gates.
        dp=cfg.get('derivative_diagnostics',{})
        derivative_status='not_tested';jump_status='not_tested'
        if dp.get('enabled'):
            derivative_status=refit_assessment(sensitivity)
            for record in sensitivity:
                if record['check']=='slice_alternative_starts':continue
                evidence=record.get('derivative_diagnostics',{})
                context=dict(group_id=gid,check=record['check'],category=record.get('category','numerical'))
                tables['derivative_refits'].append(dict(**context,
                    status=evidence.get('status','incomplete'),refit_status=record['status'],
                    scope=evidence.get('scope','unassessed'),
                    full_time_range=evidence.get('full_time_range'),
                    baseline_time_min=evidence.get('baseline_time_min'),baseline_time_max=evidence.get('baseline_time_max'),
                    common_time_min=evidence.get('common_time_min'),common_time_max=evidence.get('common_time_max')))
                for evidence_name,table in [('metrics','derivative_refit_metrics'),('points','derivative_refit_points'),('domains','derivative_refit_domains')]:
                    tables[table].extend(dict(**context,**r) for r in evidence.get(evidence_name,[]))
            jumps=maturity_jumps(model,dp);jump_status=jumps['status']
            for evidence_name,table in [('metrics','maturity_jumps'),('points','maturity_jump_points'),('domains','maturity_jump_domains')]:
                tables[table].extend(dict(group_id=gid,**r) for r in jumps[evidence_name])
        timing_status=('not_applicable' if key[3]!='live' else
            'not_checked' if not cfg['stability']['enabled'] else 'not_assessable_from_single_capture')
        unsupported=sum(r.get('holdout_unsupported',0) for r in sensitivity)

        numerical=validation['numerical_arbitrage_pass'] and validation['derivative_check_pass']
        status='fitted_with_limitations' if numerical and price_pass and stable=='passed' else 'diagnostic_fit'
        tables['cohort_status'].append({**base,**info,'status':status,'selected_observations':len(points),
            'expiries':len(model.times),'model':model.model_type,'numerical_checks_pass':bool(numerical),'market_fit_pass':bool(price_pass),
            'stability_status':stable,'stability_scope':cfg['stability'].get('maturity_holdout_scope','all_quotes'),
            'timing_status':timing_status,'unsupported_holdout_observations':unsupported,
            'derivative_stability_status':derivative_status,'maturity_jump_status':jump_status,'validation':validation,'local_volatility_implemented':False,
            'market_fit_policy':metrics[0]['coverage_policy'],
            'failed_market_fit_expiries':[m['expiration'] for m in metrics if not m['market_fit_pass']],
            'failed_checks':([] if numerical else ['numerical_arbitrage_or_derivatives'])+([] if price_pass else ['market_price_accuracy'])+([] if stable=='passed' else ['stability_'+stable]),
            'reason':'intraday_or_close_approximation; no_extrapolation_policy; local_volatility_assessed_in_separate_stage'})
    # The final decision is computed for every cohort, including unavailable
    # inputs. It does not trigger a downstream calculation or erase evidence.
    for cohort in tables['cohort_status']:
        decision=assess_local_vol_eligibility(cohort,models.get(cohort['group_id']),tables,cfg)
        cohort['local_vol_eligibility']=decision
        cohort['dupire_input_eligible']=decision['allowed']
        # This flag is scoped to this stage, not the whole library.
        cohort['local_volatility_implemented']=False
        cohort['result_scope']='implied_surface_only'
    source_files=[Path(__file__),Path(__file__).with_name('surface_model.py'),
                  Path(__file__).with_name('surface_checks.py'),Path(__file__).with_name('surface_reporting.py'),Path(__file__).with_name('surface_view.py')]
    identity=dict(engine_version=9,config=cfg,calibration_identity=calibration.manifest['identity'],
        calibration_content_hash=_hash(_bytes(dict(manifest=calibration.manifest,tables=calibration.tables))),
        source_folder=str(calibration.folder.resolve()),code_hashes={p.name:_hash(p.read_bytes()) for p in source_files},
        dependencies={name:version(name) for name in ('numpy','scipy','matplotlib')},python=sys.version.split()[0])
    summary=dict(status='completed' if models else 'skipped',fitted_cohorts=len(models),
        skip_reason=None if models else ('surface_disabled' if not cfg['enabled'] else 'no_option_data' if not groups else 'no_eligible_fit'),
        diagnostic_cohorts=sum(r['status']=='diagnostic_fit' for r in tables['cohort_status']),
        fitted_with_limitations=sum(r['status']=='fitted_with_limitations' for r in tables['cohort_status']),
        selected_observations=sum(r.get('selected',False) for r in tables['selection']),
        eligible_cohorts=sum(r['dupire_input_eligible'] for r in tables['cohort_status']),
        blocked_cohorts=sum(not r['dupire_input_eligible'] for r in tables['cohort_status']),local_volatility_implemented=False)
    limitations=[
        'Eligibility is only for bounded prototype conversion inputs, not a validated local-volatility model or full simulation.',
        'Exact maturity knots, unsupported wings and extrapolation are excluded; derivative stencils must stay in one permitted interval.',
        'Intraday/historical price and upstream curve/carry approximations remain explicit warnings.',
        'Numerical audits and refits are bounded evidence, not a global arbitrage or data-freshness certificate.',
        'No downloads, smoothing, Dupire or new Monte Carlo calculation.']
    return SurfaceResult(calibration.folder.resolve(),_hash(_bytes(identity))[:16],
        dict(identity=identity,archive_cob=calibration.manifest.get('archive_cob'),summary=summary,limitations=limitations),tables,models)
