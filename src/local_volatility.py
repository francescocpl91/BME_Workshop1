"""Bounded Dupire conversion of this execution's saved implied-vol surface.

No fitting, network calls, curve rebuilding, smoothing or Monte Carlo occurs
here. After strict conversion, an optional separate boundary wrapper performs
deterministic PDE repricing; it never changes the strict evaluator permission. Time is ACT/365F years from the upstream surface reference timestamp;
k is log(K/F(0,T)). Rates and carry already entered the calibrated inputs.
The strict evaluator accepts (T,k). DateSpotLocalVol maps aware timestamps and
positive underlying levels through the validated reference forward curve. Its
completion is explicitly assumed; it never changes the strict domain. Results
are annualized decimal volatilities, not portfolio or FX conversions.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from html import escape
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
import csv
import hashlib
import json
import math
import sys
import uuid

import numpy as np

from .storage import write_json
from .surface_checks import allows_surface_point, allows_surface_stencil
from .surface_model import black_prices


def _bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode('utf-8')


def _hash(value):
    return hashlib.sha256(value).hexdigest()


def _number(value):
    # JSON booleans and strings must never become numerical tolerances.
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError('non_numeric_value')
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('nonfinite_value')
    return result


def load_local_vol_config(path):
    """Only numerical/reporting settings; source and domain come from upstream."""
    cfg = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(cfg, dict) or set(cfg) - {'schema_version', 'enabled', 'grid', 'validation', 'plots', 'instructions', 'pricing'}:
        raise ValueError('invalid_local_vol_config')
    if type(cfg.get('schema_version')) is not int or cfg['schema_version'] != 1:
        raise ValueError('invalid_local_vol_schema')
    if any(type(cfg.get(k)) is not bool for k in ('enabled', 'plots')):
        raise ValueError('invalid_local_vol_boolean')
    grid = cfg.get('grid')
    if not isinstance(grid, dict) or set(grid) != {'maturities_per_interval', 'moneyness_points'}:
        raise ValueError('invalid_local_vol_grid')
    if any(type(v) is not int or not 3 <= v <= 501 for v in grid.values()):
        raise ValueError('invalid_local_vol_grid_size')
    val = cfg.get('validation')
    if not isinstance(val, dict) or set(val) != {'step_fraction', 'relative_tolerance', 'absolute_variance_tolerance'}:
        raise ValueError('invalid_local_vol_validation')
    if not 0 < _number(val['step_fraction']) <= .01 or any(
            _number(val[k]) <= 0 for k in ('relative_tolerance', 'absolute_variance_tolerance')):
        raise ValueError('invalid_local_vol_tolerance')
    from .local_vol_pricing import pricing_config
    cfg['pricing'] = pricing_config(cfg.get('pricing'))
    return cfg


class LocalVolEvaluator:
    """Pointwise conversion with a strict, inherited domain contract.

    This object does NOT interpolate the exported grid. It evaluates the
    original SVI/SSVI model and its derivatives at the requested point. A
    caller must also check the conversion report: numerical validation on a
    finite grid is bounded evidence, not a global pricing certificate.
    """
    def __init__(self, model, permission, minimum_density_factor, negative_tolerance):
        self.model = model
        self.permission = deepcopy(permission)
        self.minimum_density = _number(minimum_density_factor)
        self.negative_tolerance = _number(negative_tolerance)
        if self.minimum_density <= 0 or self.negative_tolerance < 0:
            raise ValueError('invalid_inherited_conditioning_policy')
        if (permission.get('purpose') != 'bounded_prototype_dupire_input'
                or permission.get('maturity_knots') != 'excluded'
                or permission.get('extrapolation') != 'forbidden'):
            raise ValueError('invalid_upstream_domain_contract')

    def _point(self, years, k, values):
        """Convert one derivative record; preserve failed points as explicit rows."""
        row = dict(years=years, k=k, status='unavailable', reason=None)
        if not allows_surface_point(self.permission, years, k):
            row['reason'] = 'blocked_or_outside_supported_domain'
            return row
        try:
            w, wk, wkk, wt = (_number(values[name]) for name in ('w', 'w_k', 'w_kk', 'w_t'))
            if years <= 0 or w <= 0:
                raise ValueError('nonpositive_time_or_total_variance')
            # Recompute the density denominator from derivatives, rather than
            # trusting a precomputed local-volatility or density column.
            g = (1-k*wk/(2*w))**2 - .25*wk*wk*(1/w+.25) + .5*wkk
            row.update(w=w, w_k=wk, w_kk=wkk, w_t=wt, density_factor=_number(g))
            if g < self.minimum_density:
                raise ValueError('density_below_inherited_minimum')
            if wt < -self.negative_tolerance:
                raise ValueError('negative_time_derivative')
            # Only roundoff inside the inherited tolerance may become zero.
            # Record it explicitly: this is not a cap/floor for bad surfaces.
            rounded = wt < 0
            variance = _number(max(wt, 0.) / g)
            row.update(status='available', reason=None, implied_volatility=math.sqrt(w/years),
                       local_variance=variance, local_volatility=math.sqrt(variance),
                       roundoff_adjusted=rounded)
        except (ValueError, KeyError, OverflowError, FloatingPointError) as exc:
            row['reason'] = str(exc)
        return row

    def evaluate(self, years, k):
        """Return an available/unavailable record; never clamp an invalid query."""
        try:
            years, k = _number(years), _number(k)
        except ValueError as exc:
            return dict(status='unavailable', reason=str(exc))
        if not allows_surface_point(self.permission, years, k):
            return self._point(years, k, {})
        try:
            values = self.model.evaluate(k, years)
        except (ValueError, OverflowError, FloatingPointError) as exc:
            return dict(years=years, k=k, status='unavailable', reason=str(exc))
        return self._point(years, k, values)



_SECONDS_PER_YEAR = 365. * 86400.


def _utc_timestamp(value):
    """Require an actual instant; never guess midnight or a local timezone.

    A COB date identifies an archive, not the live surface's observation time.
    Requiring an offset also avoids ambiguous/nonexistent local DST times.
    """
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError('timezone_aware_timestamp_required')
    return value.astimezone(timezone.utc)


class DateSpotLocalVol:
    """Calendar/underlying interface to a separately validated completion.

    t = actual seconds since observation / (365*86400), k = log(S)-log(F(t)).
    S is the underlying of this surface, in its quote currency: this does not
    convert SPX into an ETF, or USD into EUR. A vector is many states at ONE
    instant. No random shocks, drift choice, path stepping or downloads occur.

    By default only strict supported points are returned. A caller must opt
    into the configured completion, and receives per-state flags whenever
    short-end, exact-knot or unsupported-strike assumptions are used. Even
    then, the base PDE strike bounds and the final expiry are hard stops, not
    places where a path is clipped. Finite-grid evidence is not a tail-risk
    guarantee. Objects are per-run and not intended for concurrent mutation.
    """
    def __init__(self, evaluator, reference_rows, valuation_date, pricing):
        from .local_vol_pricing import ReferenceCurves, pricing_boundary
        if evaluator.permission.get('allowed') is not True:
            raise ValueError('surface_permission_required_for_state_mapping')
        self.ev = evaluator
        self.boundary = pricing_boundary(evaluator, pricing)
        self.curves = ReferenceCurves(reference_rows, self.boundary.times, valuation_date)
        self.reference = _utc_timestamp(self.curves.rows[0]['reference_timestamp'])
        # Curve/valuation dates are already matched by ReferenceCurves.
        # They are upstream valuation-calendar dates, not necessarily UTC
        # dates; do not reclassify an evening US or early Asian observation.
        self.expiries = [_utc_timestamp(r['expiry_timestamp']) for r in self.curves.rows]
        # Validate the time convention, not merely the order of the nodes.
        # One microsecond permits only timestamp serialization roundoff.
        for r, expiry in zip(self.curves.rows, self.expiries):
            if _utc_timestamp(r['reference_timestamp']) != self.reference:
                raise ValueError('inconsistent_mapping_reference_timestamp')
            if abs((expiry-self.reference).total_seconds()-float(r['years'])*_SECONDS_PER_YEAR) > 1e-6:
                raise ValueError('reference_expiry_year_fraction_mismatch')
        self.bounds = tuple(pricing['log_grid_bounds'])
        self.policy = {k:pricing[k] for k in ('short_end','wings','knot_policy','forward_policy')}

    def years(self, timestamp):
        """No reset to a chosen start date; no extension past the last expiry."""
        stamp = _utc_timestamp(timestamp)
        if stamp < self.reference or stamp > self.expiries[-1]:
            raise ValueError('timestamp_outside_calibrated_horizon')
        # Exact stored expiries use exact model nodes. This chooses the right
        # interval at knots (the last expiry uses its left limit) consistently.
        for expiry, t in zip(self.expiries, self.boundary.times):
            if stamp == expiry:return float(t)
        return (stamp-self.reference).total_seconds()/_SECONDS_PER_YEAR

    def validate_window(self, start, end):
        """Validate future caller dates and expose knots to split time steps.

        A date-only end such as 2027-12-17 is deliberately insufficient: the
        last option may expire at 14:30 UTC, before that calendar day's close.
        This method prepares metadata only and does not simulate anything.
        """
        a, b = _utc_timestamp(start), _utc_timestamp(end)
        ta, tb = self.years(a), self.years(b)
        if a >= b:raise ValueError('window_end_must_follow_start')
        return dict(start_timestamp=a.isoformat(), end_timestamp=b.isoformat(),
                    start_years=ta, end_years=tb, day_count='ACT/365F',
                    knot_timestamps=[x.isoformat() for x in self.expiries if a < x < b],
                    full_simulation_ready=False)

    def evaluate(self, timestamp, spot, *, allow_boundary_assumptions=False):
        """Return values and masks, or explicit unavailability for the query.

        A bad state rejects the query rather than substituting a coefficient.
        Valid scalar/vector queries use identical mathematics. JSON-style
        lists in the record keep outputs serializable and preserve shape.
        """
        try:
            if type(allow_boundary_assumptions) is not bool:
                raise ValueError('invalid_boundary_opt_in')
            t = self.years(timestamp)
            raw = np.asarray(spot)
            if (raw.dtype.kind not in 'iuf' or not raw.size
                    or any(isinstance(x,(bool,np.bool_)) for x in np.asarray(spot,dtype=object).flat)):
                raise ValueError('numeric_positive_underlying_required')
            states = raw.astype(float)
            if not np.isfinite(states).all() or np.any(states <= 0):
                raise ValueError('numeric_positive_underlying_required')
            forward, discount = self.curves.at(t)
            k = np.log(states)-math.log(forward)  # avoids overflow of S/F
            if np.any(k <= self.bounds[0]) or np.any(k >= self.bounds[1]):
                raise ValueError('state_outside_pricing_grid_no_clipping')
            short = bool(t < self.boundary.times[0])
            knot = bool(np.any(t == self.boundary.times))
            j = int(np.clip(np.searchsorted(self.boundary.times,t,side='right')-1,
                            0,len(self.boundary.domains)-1))
            d = self.boundary.domains[j]
            wing = (k < d['k_min']) | (k > d['k_max'])
            supported = ~(wing | short | knot)
            if not allow_boundary_assumptions and not np.all(supported):
                raise ValueError('boundary_assumptions_require_explicit_opt_in')
            # Preserve strict conditioning checks at every supported query;
            # never let completion opt-in bypass a bad interior coefficient.
            values = np.empty_like(k)
            for i in np.ndindex(k.shape):
                if supported[i]:
                    point = self.ev.evaluate(t,float(k[i]))
                    if point['status'] != 'available':
                        raise ValueError('strict_evaluator_rejected:'+str(point['reason']))
                    values[i] = point['local_variance']
            if not np.all(supported):
                values[~supported] = self.boundary.variance(t,k[~supported])
            def plain(v):return np.asarray(v).tolist()
            return dict(status='available', reason=None,
                timestamp=_utc_timestamp(timestamp).isoformat(), years=t,
                reference_timestamp=self.reference.isoformat(), forward=forward,
                discount_factor=discount, underlying=plain(states), k=plain(k),
                local_variance=plain(values), local_volatility=plain(np.sqrt(values)),
                supported_domain=plain(supported), boundary_assumption_used=plain(~supported),
                short_end_assumption=short, maturity_knot_assumption=knot,
                wing_assumption=plain(wing), full_simulation_ready=False)
        except (ValueError, KeyError, TypeError, OverflowError, FloatingPointError) as exc:
            return dict(status='unavailable', reason=str(exc), full_simulation_ready=False)


def _prepare_state_evaluators(surface, evaluators, tables, pricing, inputs):
    """Expose only cohorts passing the existing pricing and numerical gates.

    Deterministic probes validate the adapter plumbing, not a Monte Carlo
    scheme. Inputs and checks are exported with the same immutable result.
    Missing timestamps or failed gates leave a recorded unavailable cohort.
    """
    adapters = {}; tables['state_mapping'] = []; tables['state_mapping_checks'] = []
    for cohort in surface.tables['cohort_status']:
        gid = cohort['group_id']
        row = {k:cohort.get(k) for k in ('group_id','underlying','currency','route')}
        row.update(status='unavailable', full_simulation_ready=False)
        assessment = next((r for r in tables['pricing_summary'] if r['group_id']==gid), {})
        try:
            if (gid not in evaluators or assessment.get('status')!='validated_with_boundary_assumptions'
                    or any(assessment.get(k) is not True for k in ('all_expiries_assessed','numerical_pass',
                        'surface_reproduction_pass','boundary_sensitivity_pass','market_fit_pass'))):
                raise ValueError('validated_pricing_required_for_state_mapping')
            refs = [r for r in surface.tables.get('reference_inputs',[]) if r['group_id']==gid]
            adapter = DateSpotLocalVol(evaluators[gid], refs, cohort['valuation_date'], pricing)
            # Probe origin, short end, every knot and every interval midpoint.
            # At each instant verify scalar versus batch, with one assumed
            # downside and upside state inside the explicit computational box.
            times = sorted(set([0.,float(adapter.boundary.times[0])/2]
                + list(map(float,adapter.boundary.times))
                + list(map(float,(adapter.boundary.times[:-1]+adapter.boundary.times[1:])/2))))
            for t in times:
                index = np.flatnonzero(adapter.boundary.times==t)
                stamp = adapter.expiries[int(index[0])] if len(index) else adapter.reference+timedelta(seconds=t*_SECONDS_PER_YEAR)
                actual_t = adapter.years(stamp); forward,_ = adapter.curves.at(actual_t)
                j = int(np.clip(np.searchsorted(adapter.boundary.times,actual_t,side='right')-1,0,len(adapter.boundary.domains)-1))
                d = adapter.boundary.domains[j]
                ks = np.array([(adapter.bounds[0]+d['k_min'])/2,0.,(adapter.bounds[1]+d['k_max'])/2])
                spots = forward*np.exp(ks)
                batch = adapter.evaluate(stamp,spots,allow_boundary_assumptions=True)
                scalar = [adapter.evaluate(stamp,float(x),allow_boundary_assumptions=True) for x in spots]
                passed = batch['status']=='available' and all(r['status']=='available' for r in scalar)
                if passed:passed = bool(np.allclose(batch['local_variance'],[r['local_variance'] for r in scalar],rtol=1e-10,atol=1e-12))
                tables['state_mapping_checks'].append(dict(group_id=gid,timestamp=stamp.isoformat(),
                    years=actual_t,check='scalar_batch_equivalence',status='passed' if passed else 'failed',
                    reason=None if passed else batch.get('reason') or 'state_mapping_probe_failed'))
                if not passed:raise ValueError('state_mapping_probe_failed')
            row.update(status='prepared_with_boundary_assumptions', reason=None,
                reference_timestamp=adapter.reference.isoformat(),first_expiry=adapter.expiries[0].isoformat(),
                last_expiry=adapter.expiries[-1].isoformat(),day_count='ACT/365F',
                log_state_min=adapter.bounds[0],log_state_max=adapter.bounds[1],
                boundary_opt_in_required=True, **adapter.policy)
            inputs[gid]['date_spot_mapping'] = dict(reference_nodes=deepcopy(refs),
                valuation_date=cohort['valuation_date'],pricing=deepcopy(pricing),
                timestamp_policy='aware_instant_required',no_clipping=True)
            adapters[gid] = adapter
        except (ValueError,KeyError,TypeError,OverflowError,FloatingPointError) as exc:
            row.update(status='unavailable',reason=str(exc))
        tables['state_mapping'].append(row)
    return adapters

def _normalized_calls(model, k, years):
    # Independently price the implied total variance. No analytic w_T, w_k,
    # w_kk or density values enter the finite-difference calculation below.
    w = np.asarray(model.evaluate(k, years)['w'], float)
    if not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError('invalid_total_variance_in_price_check')
    return black_prices(1., np.exp(k), 1., 1., np.sqrt(w), True)


def _price_variance(model, k, years, hk, ht):
    """Normalized-call Dupire: 2*c_T/(c_kk-c_k), using central differences.

    At x=K/F=exp(k), x^2*c_xx = c_kk-c_k. All times are evaluated at the
    SAME k; holding absolute strike fixed here would be the wrong derivative.
    """
    centre = _normalized_calls(model, k, years)
    left = _normalized_calls(model, k-hk, years)
    right = _normalized_calls(model, k+hk, years)
    ct = (_normalized_calls(model, k, years+ht)-_normalized_calls(model, k, years-ht))/(2*ht)
    ck = (right-left)/(2*hk)
    ckk = (right-2*centre+left)/(hk*hk)
    denominator = ckk-ck
    result = np.full(np.shape(centre), np.nan)
    np.divide(2*ct, denominator, out=result, where=denominator > 0)
    return result


def _score(a, b, cfg):
    return abs(a-b)/(cfg['absolute_variance_tolerance'] + cfg['relative_tolerance']*max(abs(a), abs(b)))


def _sample_interval(evaluator, domain, cfg):
    """Sample interiors only; no derivative stencil crosses a knot or wing.

    Computation is vectorized per interval for large, arbitrary portfolios.
    The grid is for inspection and numerical validation, not extrapolation.
    """
    a, b = domain['start_years'], domain['end_years']
    lo, hi = domain['k_min'], domain['k_max']
    times = np.linspace(a, b, cfg['grid']['maturities_per_interval']+2)[1:-1]
    strikes = np.linspace(lo, hi, cfg['grid']['moneyness_points']+2)[1:-1]
    ts, ks = np.meshgrid(times, strikes, indexing='ij')
    ts, ks = ts.ravel(), ks.ravel()
    rows = []
    try:
        values = evaluator.model.evaluate(ks, ts)
        for i, (t, k) in enumerate(zip(ts, ks)):
            rows.append(evaluator._point(float(t), float(k), {name:np.asarray(values[name]).ravel()[i]
                        for name in ('w', 'w_k', 'w_kk', 'w_t')}))
    except (ValueError, OverflowError, FloatingPointError) as exc:
        rows = [dict(years=float(t), k=float(k), status='unavailable', reason=str(exc)) for t,k in zip(ts,ks)]
    # Use the configured coarse step and half that step. Reducing steps is
    # essential: agreement at only one step could be accidental cancellation.
    policy = cfg['validation']
    hk = np.minimum((hi-lo)*policy['step_fraction'], np.minimum(ks-lo, hi-ks)/4)
    ht = np.minimum((b-a)*policy['step_fraction'], np.minimum(ts-a, b-ts)/4)
    valid_stencils = all(allows_surface_stencil(evaluator.permission,
        [(float(t),float(k)), (float(t-dt),float(k)), (float(t+dt),float(k)),
         (float(t),float(k-dk)), (float(t),float(k+dk))]) for t,k,dt,dk in zip(ts,ks,ht,hk))
    error = None
    try:
        if not valid_stencils or np.any(hk <= 0) or np.any(ht <= 0):
            raise ValueError('unsupported_price_check_stencil')
        coarse = _price_variance(evaluator.model, ks, ts, hk, ht)
        fine = _price_variance(evaluator.model, ks, ts, hk/2, ht/2)
    except (ValueError, OverflowError, FloatingPointError) as exc:
        error = str(exc); coarse = fine = np.full(len(rows), np.nan)
    checks = []
    for row, x, y, dt, dk in zip(rows, coarse, fine, ht, hk):
        check = dict(years=row['years'], k=row['k'], time_step=float(dt), k_step=float(dk),
                     status='unavailable', reason=error or row.get('reason'))
        if row['status']=='available' and np.isfinite([x,y]).all() and min(x,y)>=0:
            scores = [_score(row['local_variance'],float(v),policy) for v in (x,y)]
            convergence = _score(float(x),float(y),policy)
            passed = max(*scores,convergence) <= 1
            check.update(status='passed' if passed else 'failed', reason=None if passed else 'price_check_tolerance_exceeded',
                         analytic_variance=row['local_variance'], coarse_variance=float(x), fine_variance=float(y),
                         maximum_agreement_score=max(scores), step_convergence_score=convergence)
        else:
            check['reason'] = check['reason'] or 'nonfinite_or_negative_price_based_variance'
        checks.append(check)
    return rows, checks


@dataclass
class LocalVolResult:
    source_folder: Path
    run_id: str
    manifest: dict
    tables: dict
    evaluator_inputs: dict
    evaluators: dict
    state_evaluators: dict = field(default_factory=dict)

    def save(self):
        """Immutable checksummed publication; incomplete outputs are preserved."""
        parent = self.source_folder/'local_volatility'
        target = parent/self.run_id
        if not self.run_id.isalnum() or target.is_symlink() or target.resolve().parent != parent.resolve():
            raise ValueError('unsafe_local_vol_output_path')
        plot_ids = [r['group_id'] for r in self.tables['cohorts'] if r.get('available_points',0)] if self.manifest['identity']['config']['plots'] else []
        expected = {name:name+'.csv' for name in self.tables}
        expected.update(inputs='evaluator_inputs.json', report='report.html')
        expected.update({'plot_'+gid:_hash(gid.encode())[:12]+'.png' for gid in plot_ids})
        expected.update({'plot3d_'+gid:_hash(gid.encode())[:12]+'_3d.png' for gid in plot_ids})
        expected.update({'view_'+gid:_hash(gid.encode())[:12]+'_3d.html' for gid in plot_ids})
        def verify(folder):
            data = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
            if not isinstance(data, dict):raise ValueError('local_vol_existing_output_inconsistent')
            files = data.get('files',{})
            if not isinstance(files,dict) or data.get('complete') is not True or data.get('identity') != self.manifest['identity'] or set(files) != set(expected):
                raise ValueError('local_vol_existing_output_inconsistent')
            for key,name in expected.items():
                if not isinstance(files[key],dict) or files[key].get('file') != name or _hash((folder/name).read_bytes()) != files[key].get('sha256'):
                    raise ValueError('local_vol_existing_output_checksum_mismatch')
            return folder
        def complete():
            if not target.exists():return False
            if not target.is_dir():raise ValueError('local_vol_output_not_directory')
            if (target/'manifest.json').exists():
                data=json.loads((target/'manifest.json').read_text(encoding='utf-8'))
                if not isinstance(data,dict):raise ValueError('local_vol_existing_output_inconsistent')
                if data.get('complete') is True:verify(target);return True
                if data.get('complete') is not False or data.get('identity') != self.manifest['identity']:
                    raise ValueError('local_vol_incomplete_identity_mismatch')
            return False
        if complete():return target
        parent.mkdir(parents=True,exist_ok=True)
        # Short names avoid Windows path limits in deep OneDrive folders.
        # The context manager owns only this fresh child. No source is deleted.
        with TemporaryDirectory(prefix='.p-',dir=parent) as temp:
            owned=Path(temp)
            if owned.resolve().parent != parent.resolve():raise ValueError('unsafe_local_vol_staging')
            stage=owned/'out';stage.mkdir()
            for name,rows in self.tables.items():
                fields=list(dict.fromkeys(k for row in rows for k in row)) or ['status']
                with (stage/(name+'.csv')).open('w',encoding='utf-8',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
                    for row in rows:writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})
            write_json(stage/'evaluator_inputs.json',self.evaluator_inputs)
            if plot_ids:
                _plot(self,stage,expected,plot_ids)
                from .surface_view import write_local_view
                for gid in plot_ids:write_local_view(self,gid,stage/expected['view_'+gid])
            (stage/'report.html').write_text(_report(self,expected),encoding='utf-8')
            if complete():return target
            backup=parent/('.i-'+uuid.uuid4().hex) if target.exists() else None
            manifest={**self.manifest,'complete':True,'files':{key:dict(file=name,sha256=_hash((stage/name).read_bytes())) for key,name in expected.items()}}
            if backup:manifest['recovered_incomplete_output']=backup.name
            write_json(stage/'manifest.json',manifest);verify(stage)
            if backup:target.rename(backup)
            try:stage.rename(target)
            except OSError:
                if backup and not target.exists():backup.rename(target)
                elif target.exists():return verify(target)
                raise
        return target


def convert_surface(surface, config_file, progress=None):
    """Use this run's surface object, with no file discovery or market requests."""
    cfg=load_local_vol_config(config_file)
    tables=dict(cohorts=[],domains=[],local_vol_grid=[],validation=[])
    inputs={};evaluators={}
    for cohort in surface.tables['cohort_status']:
        gid=cohort['group_id'];permission=deepcopy(cohort.get('local_vol_eligibility',{}))
        context={k:cohort.get(k) for k in ('group_id','underlying','currency','valuation_date','route')}
        summary={**context,'status':'skipped','reason':None,'full_simulation_ready':False,
                 'warnings':permission.get('warnings',[]),'reference_timestamps':permission.get('reference_timestamps',[])}
        if not cfg['enabled']:summary['reason']='local_vol_disabled'
        elif permission.get('allowed') is not True:
            summary['reason']='surface_not_eligible';summary['blockers']=permission.get('blockers',[])
        elif gid not in surface.models:summary['reason']='eligible_surface_model_missing'
        else:
            model=surface.models[gid]
            dp=surface.manifest['identity']['config'].get('derivative_diagnostics',{})
            try:
                evaluator=LocalVolEvaluator(model,permission,dp['minimum_density_factor'],dp['negative_tolerance'])
                if not permission.get('domain'):raise ValueError('empty_supported_domain')
                # Eligibility is data, not permission to invent model support.
                # Check it against the model before allocating any grid.
                for d in permission['domain']:
                    a,b,lo,hi=(_number(d[k]) for k in ('start_years','end_years','k_min','k_max'))
                    mlo,mhi=model.interval_domain(a,b)
                    if not 0<a<b or not mlo<=lo<hi<=mhi or any(a<t<b for t in model.times):
                        raise ValueError('domain_disagrees_with_surface')
                rows=[];checks=[]
                for interval,d in enumerate(permission['domain']):
                    tables['domains'].append({**context,'interval':interval,**d})
                    grid,audit=_sample_interval(evaluator,d,cfg)
                    rows.extend({**context,'interval':interval,**r} for r in grid)
                    checks.extend({**context,'interval':interval,**r} for r in audit)
                tables['local_vol_grid'].extend(rows);tables['validation'].extend(checks)
                available=[r for r in rows if r['status']=='available']
                passed=bool(rows and len(available)==len(rows) and all(r['status']=='passed' for r in checks))
                vals=[r['local_volatility'] for r in available]
                summary.update(status='validated_on_supported_grid' if passed else 'review_required',
                    reason=None if passed else 'conversion_or_numerical_validation_failed',
                    requested_points=len(rows),available_points=len(available),validation_passes=sum(r['status']=='passed' for r in checks),
                    roundoff_adjustments=sum(r.get('roundoff_adjusted',False) for r in rows),
                    minimum_local_volatility=min(vals) if vals else None,maximum_local_volatility=max(vals) if vals else None)
                summary.update(maximum_agreement_score=max((r.get('maximum_agreement_score',0.) for r in checks),default=None),
                               maximum_step_convergence_score=max((r.get('step_convergence_score',0.) for r in checks),default=None))
                # Store enough provenance to reconstruct later without guessing
                # which surface/configuration produced these derived numbers.
                inputs[gid]=dict(model=asdict(model),permission=permission,
                    minimum_density_factor=evaluator.minimum_density,negative_tolerance=evaluator.negative_tolerance)
                if passed:evaluators[gid]=evaluator
            except (ValueError,KeyError,OverflowError,FloatingPointError) as exc:
                summary.update(status='unavailable',reason=str(exc))
        tables['cohorts'].append(summary)
    from .local_vol_pricing import validate_repricing
    tables.update(validate_repricing(surface,evaluators,cfg['pricing'],progress))
    state_evaluators = _prepare_state_evaluators(surface,evaluators,tables,cfg['pricing'],inputs)
    source_code=[Path(__file__).with_name('local_vol_boundary.py'),Path(__file__).with_name('local_vol_pricing.py'),Path(__file__).with_name('surface_view.py'),Path(__file__),Path(__file__).with_name('surface_model.py'),Path(__file__).with_name('surface_checks.py')]
    identity=dict(engine_version=4,config=cfg,source_surface_id=surface.run_id,
        source_identity=surface.manifest['identity'],
        source_content_hash=_hash(_bytes(dict(manifest=surface.manifest,tables=surface.tables,models={k:asdict(v) for k,v in surface.models.items()}))),
        source_folder=str(surface.folder.resolve()),code_hashes={p.name:_hash(p.read_bytes()) for p in source_code},
        dependencies={k:version(k) for k in ('numpy','scipy','matplotlib')},python=sys.version.split()[0])
    summary=dict(cohorts=len(tables['cohorts']),validated_cohorts=len(evaluators),
        skipped_cohorts=sum(r['status']=='skipped' for r in tables['cohorts']),full_simulation_ready=False)
    return LocalVolResult(surface.folder.resolve(),_hash(_bytes(identity))[:16],
        dict(identity=identity,summary=summary,archive_cob=surface.manifest.get('archive_cob')),tables,inputs,evaluators,state_evaluators)


def _plot(result,folder,expected,gids):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for gid in gids:
        rows=[r for r in result.tables['local_vol_grid'] if r['group_id']==gid and r['status']=='available']
        fig,ax=plt.subplots(figsize=(9,4))
        # Discrete markers intentionally avoid painting untested wings, knots
        # or the missing short end as though they had been interpolated.
        try:
            dots=ax.scatter([r['years'] for r in rows],[math.exp(r['k']) for r in rows],
                c=[100*r['local_volatility'] for r in rows],s=12,cmap='viridis')
            ax.set(xlabel='Years from surface observation',ylabel='Underlying level / forward',title='Local volatility on supported interior grid')
            fig.colorbar(dots,ax=ax,label='Annualized local volatility (%)')
            fig.tight_layout();fig.savefig(folder/expected['plot_'+gid],dpi=130)
        finally:plt.close(fig)
        # One independent mesh per supported maturity interval: never connect
        # across knots or paint unobserved short-end/wing assumptions as data.
        # The vertical axis and shared colour scale both show annualized percent.
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        norm=Normalize(vmin=0.,vmax=max(100*r['local_volatility'] for r in rows))
        fig=plt.figure(figsize=(12,7.5))
        try:
            ax=fig.add_subplot(111,projection='3d')
            from .surface_view import local_patches
            for patch in local_patches(result,gid):
                mesh=np.asarray(patch['rows'],float)
                x,y,z=mesh[:,:,0],mesh[:,:,1],mesh[:,:,2]
                ax.plot_surface(x,y,z,cmap='viridis',norm=norm,linewidth=.15,
                    edgecolor=(.15,.2,.25,.22),antialiased=True,rcount=z.shape[0],ccount=z.shape[1])
            ax.set(xlabel='Years from option-surface observation',
                ylabel='Underlying level / forward',
                zlabel='Local volatility (%)',zlim=(0,norm.vmax*1.03))
            # Labels are generic: the cohort name is metadata, never a model input.
            ax.set_title(str(rows[0].get('underlying','Underlying'))+' local volatility | supported domain',pad=20)
            ax.view_init(elev=27,azim=-128);ax.set_box_aspect((1.8,1.15,1.1))
            fig.colorbar(ScalarMappable(norm=norm,cmap='viridis'),ax=ax,shrink=.65,pad=.09,label='Annualized volatility (%)')
            fig.text(.08,.035,'Separate patches preserve maturity knots. Short-end and wing assumptions are not shown.',fontsize=10)
            fig.subplots_adjust(left=.02,right=.90,bottom=.13,top=.92)
            fig.savefig(folder/expected['plot3d_'+gid],dpi=180)
        finally:plt.close(fig)



def _report(result,expected):
    def e(value):return escape(format(value,'.6g') if isinstance(value,float) else str(value))
    def table(rows,columns):
        return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+e(label)+'</th>' for _,label in columns)+'</tr></thead><tbody>'+''.join(
            '<tr>'+''.join('<td>'+e(row.get(key,''))+'</td>' for key,_ in columns)+'</tr>' for row in rows)+'</tbody></table></div>'
    cfg=result.manifest['identity']['config']
    html=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
        '<title>Local-volatility conversion</title><style>body{font:16px/1.55 system-ui;margin:32px auto;padding:0 24px;max-width:1100px;color:#172b3a}h1,h2{line-height:1.2}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:9px;border-bottom:1px solid #d7e0e5;text-align:left}th{background:#edf3f7}.scroll{overflow-x:auto}.notice{background:#fff3d9;padding:16px;border-left:4px solid #b87900}img{max-width:100%}code{background:#edf3f7;padding:2px 4px}</style>',
        '<h1>Local-volatility conversion</h1>',
        '<p>Archive COB: '+e(result.manifest.get('archive_cob'))+' · Analysis: '+e(result.run_id)+'</p>',
        '<p class="notice"><strong>Bounded conversion and explicit pricing assumptions. Full pricing and simulation are not ready unless the separate checks below pass; no Monte Carlo is implemented here.</strong> No new downloads or surface refitting. The strict evaluator still excludes short end, wings and exact knots. The separate pricing/state wrapper extends them only under labelled assumptions.</p>',
        '<p>Formula: local variance = w_T / g, at fixed k = log(K/F(T)). CSV volatility values are annualized decimals. The evaluator uses the original fitted surface; it does not interpolate this exported grid.</p>',
        '<h2>Decisions</h2>',table(result.tables['cohorts'],[('underlying','Underlying'),('valuation_date','Observed'),('route','Route'),('status','Decision'),('available_points','Available points'),('requested_points','Requested'),('validation_passes','Checks passed'),('reason','Reason')]),
        '<h2>Numerical validation</h2>',
        '<p>Independent finite differences of normalized call prices use 2 c_T / (c_kk − c_k). Both the coarse and half-step estimates must agree with the analytic result and with each other. Every score must be ≤ 1; missing evidence fails. This validates sampled interior points, not global arbitrage or option repricing.</p>',
        '<p>Score = |variance A − variance B| / (absolute tolerance + relative tolerance × max(|A|, |B|)).</p>',
        '<p>Absolute variance tolerance: '+e(cfg['validation']['absolute_variance_tolerance'])+'; relative tolerance: '+e(cfg['validation']['relative_tolerance'])+'; coarse step: '+e(cfg['validation']['step_fraction'])+' of interval width, reduced near boundaries.</p>']
    for row in result.tables['cohorts']:
        gid=row['group_id'];data=result.evaluator_inputs.get(gid)
        html+=['<h2>'+e(row.get('underlying'))+' · '+e(row.get('valuation_date'))+' · '+e(gid)+'</h2>',
               '<ul>'+''.join('<li>'+e(w)+'</li>' for w in row.get('warnings',[]))+'</ul>']
        if data:
            html+=['<p>Minimum density factor: '+e(data['minimum_density_factor'])+'; negative time-derivative roundoff tolerance: '+e(data['negative_tolerance'])+'. Both inherited from the surface. Recorded roundoff adjustments: '+e(row.get('roundoff_adjustments',0))+'. No extreme-volatility cap is applied.</p>',
                '<p>Local-volatility range (annualized decimals): '+e(row.get('minimum_local_volatility'))+' to '+e(row.get('maximum_local_volatility'))+'.</p>',
                '<p>Maximum agreement score: '+e(row.get('maximum_agreement_score'))+'; maximum step-convergence score: '+e(row.get('maximum_step_convergence_score'))+'. Each must be ≤ 1, and every requested point must be assessed.</p>',
                table(data['permission']['domain'],[('start_expiry','After expiry'),('end_expiry','Before expiry'),('k_min','Minimum k'),('k_max','Maximum k')])]
        if 'view_'+gid in expected:html+=['<p><a href="'+e(expected['view_'+gid])+'">Open interactive 3D local-volatility surface</a></p>']
        if 'plot3d_'+gid in expected:html+=['<p><a href="'+e(expected['plot3d_'+gid])+'">Download 3D PNG</a></p><img alt="Three-dimensional supported local-volatility surface" src="'+e(expected['plot3d_'+gid])+'">']
        if 'plot_'+gid in expected:html+=['<img alt="Supported local volatility grid" src="'+e(expected['plot_'+gid])+'">']
    html.extend(_pricing_report(result,table,e))
    html += ['<h2>Date and underlying mapping</h2>',
        '<p>Time zero is the actual option-surface observation timestamp, not archive COB or midnight COB+1. Supply a timezone-aware ISO timestamp and the positive underlying level in its quote currency. ACT/365F elapsed time and the saved log-linear forward give k = log(S/F). No ETF proxy or FX conversion is implied.</p>',
        table(result.tables['state_mapping'], [('underlying','Underlying'),('status','Mapping'),('reference_timestamp','Time zero'),('first_expiry','First expiry'),('last_expiry','Final instant'),('reason','Reason')]),
        '<p>Default evaluation accepts the strict supported domain only. Explicit boundary opt-in enables the calibrated short end, wings and right-sided knots (terminal left limit); every use is flagged per state. Queries outside the base PDE log-state bounds or past the exact last expiry are rejected without clipping. A window check exposes intervening knot timestamps for future time-step splitting. Full simulation readiness remains false.</p>',
        '<p>Deterministic scalar/batch mapping checks: '+str(sum(r['status']=='passed' for r in result.tables['state_mapping_checks']))+' / '+str(len(result.tables['state_mapping_checks']))+' passed. These verify the interface, not simulation convergence. Exact checks are in state_mapping_checks.csv; reference nodes and policies are in evaluator_inputs.json.</p>']
    html+=['<h2>Detailed results</h2><ul>'+''.join('<li><a href="'+e(name)+'">'+e(name)+'</a></li>' for name in expected.values() if not name.endswith(('.html','.png')))+'</ul>',
           '<p>Known-answer and domain regressions are in the automated test suite. The PDE checks below test explicit boundary assumptions. Neither a successful conversion nor a pricing experiment approves unrestricted portfolio simulation. Historical Monte Carlo remains a separate workflow.</p></html>']
    return '\n'.join(html)


def _pricing_report(result,table,e):
    """Keep the decision compact; retain every quote/score in checksummed CSVs."""
    cfg=result.manifest['identity']['config']['pricing']
    summary=result.tables['pricing_summary'];metrics=result.tables['pricing_expiries']
    calibrated=cfg['short_end']=='first_smile_total_variance'
    legacy_notice='<p class="notice">Legacy comparison: frozen first-interval local variance and flat wing variance. Alternative short end uses first implied variance/time; alternative wings fade to ATM. These are uncalibrated extensions, not observed support.</p>'
    return ['<h2>Boundary policies and deterministic option repricing</h2>',
        '<p>No Monte Carlo. Forward strike PDE in normalized units c=C/(DF), x=K/F; c_T = 0.5 local_variance (c_kk-c_k). Initial payoff (1-x)+. The solver prices all strikes together. Reference F and D use log-linear interpolation through upstream nodes, anchored at spot and 1 at time zero.</p>',
        '<p class="notice">Configured short end: '+e(cfg['short_end'])+'; wings: '+e(cfg['wings'])+'. For the calibrated policy, w(T,k)=(T/T1)w(T1,k), so the first smile is reached exactly. Wing OTM prices and slopes match at fitted-node joins, with positive power mixtures ordered across maturities. Local variance is derived from these completed prices. The sensitivity runs also anchor the first smile: alternate time power '+e(cfg['short_end_power_alternative'])+' and alternate tail fraction '+e(cfg['wing_tail_fraction_alternative'])+'. All extensions remain assumptions, not additional observed support. Knots use right intervals (terminal left limit); density may jump at wing joins. There is no extrapolation beyond the final expiry.</p>' if calibrated else legacy_notice,
        table(summary,[('underlying','Underlying'),('status','Pricing decision'),('assessed_quotes','Assessed'),('excluded_quotes','Outside supported strips'),('numerical_pass','Grid checks'),('surface_reproduction_pass','Reproduces surface'),('boundary_sensitivity_pass','Boundary sensitivity'),('market_fit_pass','Market check'),('reason','Reason')]),
        '<p>Price allowance = '+e(cfg['absolute_price_forward_fraction'])+' × D × F + '+e(cfg['relative_price_tolerance'])+' × |target price|. Grid convergence and computational-boundary expansion each use '+e(cfg['numerical_budget_fraction'])+' of that allowance. Price-fit and each boundary-policy sensitivity score must be ≤1 at every assessed quote. IV error must be ≤ '+e(100*cfg['maximum_iv_error'])+' volatility percentage points. These explicit prototype budgets are not market conventions.</p>',
        '<p>All expiries must have assessed quotes. Target prices are from the fitted surface at the reference timestamp and reference strike F exp(k). Comparison to original live bid/ask or historical prices transports the PDE IV back to original F,D,tau using the existing stable-IV timing approximation. Historical data never receives invented spreads. The upstream RMSE/practical-coverage policy is reused on this supported subset; excluded quotes remain in option_repricing.csv.</p>',
        table(metrics,[('expiration','Expiry'),('observations','Quotes'),('maximum_price_fit_score','Max price score'),('maximum_convergence_score','Grid score'),('maximum_grid_boundary_score','Wider-grid score'),('maximum_wing_sensitivity_score','Wing score'),('maximum_short_end_sensitivity_score','Short-end score'),('iv_fit_pass','IV fit'),('market_scaled_rmse','Market RMSE'),('practical_coverage_fraction','Practical coverage')]),
        '<p>Pricing uses a harmonic cell-averaged, exponentially fitted strike stencil for calibrated completions; legacy policies retain central differences. The cell integration splits at density joins. This changes numerical integration, not fitted prices or acceptance thresholds.</p>',
        '<h3>Completion calibration and arbitrage audit</h3>' if calibrated else '<h3>Legacy boundary comparison</h3>',
        '<p>The first-expiry anchor is analytic, rather than a numerical fit to a different smile. Cores and transition strips undergo a numerical calendar/density audit; ordering beyond both wing joins follows from the mixture construction. These checks do not certify market realism in unobserved wings. Boundary sensitivity now compares two calibrated completions; it is not a test against the old uncalibrated policy. <strong>Unobserved far-wing local variance can become very large immediately after an expiry knot; price density can jump at a wing join. Passing prices does not validate simulation time stepping or remote tail behaviour.</strong></p>' if calibrated else '<p>Legacy extensions are not calibrated to preserve the first smile. The comparison remains available for reproducibility.</p>',
        table(result.tables.get('boundary_audit',[]),[('variant','Completion'),('status','Audit'),('points','Strike samples'),('minimum_calendar_price_difference','Minimum calendar difference'),('outer_tail_ordering','Outer-tail ordering')]),
        '<h3>Actual error magnitudes on supported quotes</h3>',
        table(metrics,[('expiration','Expiry'),('maximum_absolute_price_error','Largest absolute price error'),('maximum_absolute_iv_error_vol_points','Largest IV error (vol points)'),('surface_reproduction_pass','Surface reproduced'),('boundary_sensitivity_pass','Boundary sensitivity passed'),('market_fit_pass','Market fit passed')]),
        '<p>Any score above one, missing evidence, or failed IV gate prevents validation. The underlying surface and its eligibility are never changed to improve these results. Timing warnings remain inherited. See pricing_reference_nodes.csv for exact forward/discount nodes and option_repricing.csv for all prices, exclusions and errors.</p>']
