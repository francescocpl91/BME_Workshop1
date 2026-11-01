"""Offline risk-neutral diffusion validation, never portfolio simulation.

The state X=S/F has zero arithmetic drift for deterministic carry. We evolve
log(X), so -variance/2 is essential: it preserves the conditional mean of X
for a frozen Euler coefficient. Rates/dividends enter exactly once through
the saved forward and discount curves. No historical drift enters this file.
"""
from dataclasses import dataclass, field
from datetime import timedelta
from zoneinfo import ZoneInfo
from html import escape
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
import csv
import json
import math
import sys

import numpy as np
from scipy.special import ndtr, ndtri

from .local_volatility import _bytes, _hash, _number
from .storage import write_json
from .cob import forecast_start_for_cob
from .surface_model import black_prices
from . import local_vol_mc_reporting as reporting


def load_config(path):
    """Fail early on misspelled settings, booleans-as-numbers and unsafe sizes."""
    cfg = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    keys = {'schema_version', 'enabled', 'paths', 'seed', 'steps_per_year',
            'refinement', 'allow_boundary_assumptions', 'family_error_probability',
            'absolute_price_forward_fraction', 'relative_price_tolerance',
            'maximum_se_forward_fraction', 'maximum_se_relative_price',
            'forward_relative_tolerance'}
    if not isinstance(cfg, dict) or set(cfg)-{'instructions', 'control_variate', 'reporting'} != keys:
        raise ValueError('invalid_local_vol_mc_config_keys')
    if type(cfg['schema_version']) is not int or cfg['schema_version'] != 1:
        raise ValueError('invalid_local_vol_mc_schema')
    for key in ('enabled', 'allow_boundary_assumptions'):
        if type(cfg[key]) is not bool: raise ValueError('invalid_mc_boolean:'+key)
    for key, low, high in (('paths', 100, 2000000), ('seed', 0, 2**32-1),
                           ('steps_per_year', 12, 10000), ('refinement', 2, 8)):
        if type(cfg[key]) is not int or not low <= cfg[key] <= high:
            raise ValueError('invalid_mc_integer:'+key)
    if cfg['paths'] % 2: raise ValueError('antithetic_paths_must_be_even')
    for key in keys-{'schema_version','enabled','paths','seed','steps_per_year',
                    'refinement','allow_boundary_assumptions'}:
        if _number(cfg[key]) <= 0: raise ValueError('invalid_mc_tolerance:'+key)
    if not 1e-8 <= cfg['family_error_probability'] < .5:
        raise ValueError('invalid_mc_family_error_probability')
    if cfg.get('control_variate', 'delta') not in ('delta', 'coupled_black'):
        raise ValueError('invalid_mc_control_variate')
    reporting.settings(cfg)  # validate optional output settings, never numerical tolerances
    return cfg


def pair_samples(values):
    """First/second array halves are antithetic partners, NOT independent paths."""
    values = np.asarray(values, float)
    if values.ndim != 1 or len(values) < 4 or len(values) % 2 or not np.isfinite(values).all():
        raise ValueError('invalid_antithetic_samples')
    half = len(values)//2
    return .5*(values[:half]+values[half:])


def estimate(independent_samples):
    """Sample standard error of the mean of independent PAIR averages."""
    a = np.asarray(independent_samples, float)
    if a.ndim != 1 or len(a) < 2 or not np.isfinite(a).all():
        raise ValueError('invalid_independent_samples')
    return float(a.mean()), float(a.std(ddof=1)/math.sqrt(len(a)))


def coefficient(adapter, t, k, allow_boundary):
    """Vector equivalent of DateSpotLocalVol, without per-path Python records.

    k=log(S/F) is already the simulation state. Inside supported strips we
    retain the strict derivative/density conditions. Elsewhere we call the
    existing completed surface; this does not widen its economic support.
    Invalid states abort the cohort: silently dropping them biases prices.
    """
    k = np.asarray(k, float)
    if not k.size or not np.isfinite(k).all() or np.any(k <= adapter.bounds[0]) or np.any(k >= adapter.bounds[1]):
        raise ValueError('state_outside_pricing_grid_no_clipping')
    b = adapter.boundary
    if not np.isfinite(t) or not 0 <= t <= b.times[-1]:
        raise ValueError('mc_time_outside_calibrated_horizon')
    short = bool(t < b.times[0]); knot = bool(np.any(t == b.times))
    j = int(np.clip(np.searchsorted(b.times, t, side='right')-1, 0, len(b.domains)-1))
    d = b.domains[j]
    wing = (k < d['k_min']) | (k > d['k_max'])
    supported = ~(wing | short | knot)
    if not allow_boundary and not np.all(supported):
        raise ValueError('boundary_assumptions_require_explicit_opt_in')
    v = np.empty_like(k)
    if np.any(supported):
        x = k[supported]
        p = adapter.ev.model.evaluate(x, t)
        w, wk, wkk, wt = (np.asarray(p[name]) for name in ('w','w_k','w_kk','w_t'))
        g = (1-x*wk/(2*w))**2-.25*wk*wk*(1/w+.25)+.5*wkk
        if (not all(np.isfinite(a).all() for a in (w,wk,wkk,wt,g))
                or np.any(w <= 0) or np.any(g < adapter.ev.minimum_density)
                or np.any(wt < -adapter.ev.negative_tolerance)):
            raise ValueError(f'strict_evaluator_rejected_mc_state: t={t:.10g}, '
                f'min_density={float(np.min(g)):.10g}, required={adapter.ev.minimum_density}, '
                f'min_w_t={float(np.min(wt)):.10g}')
        v[supported] = np.maximum(wt, 0.)/g  # inherited roundoff allowance only
    # A smoothed completed price surface is a distinct, labelled time
    # interpolation. Match the PDE coefficient, including supported strips;
    # retaining the old coefficient there would silently simulate a splice.
    # Evaluate it ONCE on the full state vector. Computing a wing subset first
    # would discard that work and invalidate its mesh cache immediately.
    # All original strict interior checks above still execute before this.
    smoothed=bool(allow_boundary and getattr(b,'smoothing_width',0)>0)
    if smoothed:v=b.variance(t,k)
    elif np.any(~supported):v[~supported]=b.variance(t,k[~supported])
    if not np.isfinite(v).all() or np.any(v < 0):
        raise ValueError('invalid_mc_local_variance')
    return v, dict(wing=wing, short=short, knot=knot, assumed=(~supported)|smoothed, time_smoothing=smoothed)


class SimulationStopped(ValueError):
    """A rejected path invalidates the cohort; preserve evidence, not a price."""
    def __init__(self, reason, details=None):
        super().__init__(reason)
        self.details = details or {}
        self.partial = dict(quotes=[], forwards=[], boundaries=[])


def _audit_state(k, adapter, *, t=None, dt=None, level=None, previous=None, variance=None, dw=None):
    # Check AFTER every update, including the terminal update. Keep the first
    # failing path and its coefficient/shock for diagnosis; never clip it.
    bad = ~np.isfinite(k) | (k <= adapter.bounds[0]) | (k >= adapter.bounds[1])
    if np.any(bad):
        i = int(np.flatnonzero(bad)[0])
        details = dict(level=level, years=t, dt=dt, failed_paths=int(bad.sum()),
            first_failed_path=i, bounds=list(adapter.bounds),
            minimum_log_state=float(np.min(k)) if np.isfinite(k).all() else None,
            maximum_log_state=float(np.max(k)) if np.isfinite(k).all() else None,
            log_state_after=float(k[i]) if np.isfinite(k[i]) else None)
        if t is not None:
            details['timestamp']=(adapter.reference+timedelta(days=365*t)).isoformat()
        if previous is not None: details['log_state_before']=float(previous[i])
        if variance is not None: details['annualized_volatility_before']=float(np.sqrt(variance[i]))
        if dw is not None: details['brownian_increment']=float(dw[i])
        raise SimulationStopped('state_outside_pricing_grid_no_clipping', details)


def _quote_statistics(q, xfine, xcoarse, z, cfg, brownian=None):
    """Common-reference payoff, with fixed analytic control and paired errors.

    beta is a deterministic fitted Black delta. Subtracting beta*(X-1) has
    zero theoretical expectation; beta is not estimated on these same paths.
    We also export raw payoffs and separately test the unadjusted forward.
    """
    t, k = float(q['years']), float(q['k'])
    f, discount = float(q['reference_forward']), float(q['reference_discount'])
    target = float(q['target_price']); pde = float(q['pde_price'])
    iv = float(q['fitted_iv']); right = q['option_type']
    if right not in ('call','put') or t <= 0 or iv <= 0:
        raise ValueError('invalid_mc_quote')
    strike_ratio = math.exp(k); sign = 1 if right == 'call' else -1
    beta = float(ndtr((-k+.5*iv*iv*t)/(iv*math.sqrt(t))))-(right == 'put')
    df = discount*f
    # D*E[(S_T-K)+] = D*F*E[(X_T-K/F)+]. Prices come from simulated terminal
    # states. Fitted IV only supplies a known-expectation control variate below
    # to reduce sampling noise; it never replaces this local-vol payoff.
    raw_fine = df*np.maximum(sign*(xfine-strike_ratio), 0.)
    raw_coarse = df*np.maximum(sign*(xcoarse-strike_ratio), 0.)
    # Keep the old fixed-delta estimates as diagnostics and a backwards-
    # compatible policy. No control coefficient is fitted on pricing paths.
    delta_fine = raw_fine-df*beta*(xfine-1)
    delta_coarse = raw_coarse-df*beta*(xcoarse-1)
    mode = cfg.get('control_variate', 'delta')
    control_price = None
    if mode == 'coupled_black':
        # W_t is the SAME accumulated Brownian motion as the local-vol paths.
        # Y=exp(-iv^2*t/2 + iv*W_t) has E[Y]=1 and a known Black option price.
        # H(X)-H(Y)+Black - beta*(X-Y) therefore estimates the ORIGINAL
        # local-vol payoff expectation. The auxiliary payoff is not a model
        # replacement, and Black is calculated independently of target_price.
        # Adding the residual stock control preserves exact call-put parity.
        # Fine/coarse use identical Y: it cancels in their paired difference.
        w = np.asarray(brownian, float)
        if w.shape != xfine.shape or not np.isfinite(w).all():
            raise ValueError('invalid_control_brownian')
        y = np.exp(-.5*iv*iv*t+iv*w)
        if not np.isfinite(y).all():raise ValueError('invalid_control_lognormal')
        auxiliary = df*np.maximum(sign*(y-strike_ratio), 0.)
        control_price = float(black_prices(f,f*strike_ratio,t,discount,iv,right=='call'))
        correction = auxiliary-control_price-df*beta*(y-1)
        fine_pairs = pair_samples(delta_fine-correction)
        coarse_pairs = pair_samples(delta_coarse-correction)
    elif mode == 'delta':
        fine_pairs = pair_samples(delta_fine)
        coarse_pairs = pair_samples(delta_coarse)
    else:
        raise ValueError('invalid_mc_control_variate')
    delta_price, delta_se = estimate(pair_samples(delta_fine))
    price, se = estimate(fine_pairs)
    coarse, coarse_se = estimate(coarse_pairs)
    difference, difference_se = estimate(fine_pairs-coarse_pairs)
    raw_price, raw_se = estimate(pair_samples(raw_fine))
    budget = cfg['absolute_price_forward_fraction']*df+cfg['relative_price_tolerance']*abs(target)
    precision_budget = cfg['maximum_se_forward_fraction']*df+cfg['maximum_se_relative_price']*abs(target)
    return dict(record_id=q['record_id'], expiration=q['expiration'], years=t, k=k,
        option_type=right, reference_strike=f*strike_ratio, reference_forward=f,
        reference_discount=discount, target_price=target, pde_price=pde,
        fine_price=price, standard_error=se, raw_price=raw_price, raw_standard_error=raw_se,
        coarse_price=coarse, coarse_standard_error=coarse_se, control_beta=beta,
        control_variate=mode, control_price=control_price,
        delta_control_price=delta_price, delta_control_standard_error=delta_se,
        variance_reduction_vs_delta=(delta_se/se)**2 if se > 0 else None,
        paired_difference=difference, paired_difference_se=difference_se,
        deterministic_budget=budget, confidence_multiplier=z,
        sampling_half_width=z*se, precision_budget=precision_budget,
        price_error=price-target, pde_error=price-pde,
        fit_pass=bool(abs(price-target) <= budget+z*se),
        pde_pass=bool(abs(price-pde) <= budget+z*se),
        convergence_pass=bool(abs(difference) <= budget+z*difference_se),
        precision_pass=bool(se <= precision_budget))


def simulate_cohort(adapter, quotes, cfg, seed, progress=None, recorder=None):
    """Coupled coarse/fine log Euler; memory is paths, not paths x dates.

    Each expiry interval has its own exact endpoint. No step straddles a
    maturity derivative jump. Fine independent Brownian increments sum to
    the coarse increment; the coarse coefficient stays at its left endpoint.
    Antithetic pairs share opposite increments at every level and time step.
    """
    if not cfg['allow_boundary_assumptions']:
        raise ValueError('short_end_requires_explicit_boundary_opt_in')
    n = cfg['paths']; half = n//2; factor = cfg['refinement']
    times = np.asarray(adapter.boundary.times, float)
    if not quotes: raise ValueError('no_supported_mc_quotes')
    if any(not np.any(np.isclose(float(q['years']), times, rtol=0, atol=1e-10)) for q in quotes):
        raise ValueError('mc_quote_maturity_not_a_reference_node')
    # Bonferroni covers both price comparisons, convergence and both forward
    # means at all expiries. Dependence between strikes does not invalidate
    # the union bound; the normal approximation is still only asymptotic.
    tests = 3*len(quotes)+2*len(times)
    z = float(-ndtri(cfg['family_error_probability']/(2*tests)))
    rng = np.random.default_rng(seed)
    fine = np.zeros(n); coarse = np.zeros(n)
    brownian = np.zeros(n)  # one accumulated W_t per path; no path-by-time matrix
    output = []; forwards = []; boundary_rows = []
    if recorder is not None: recorder.record(0., fine)
    total_steps = 0; previous = 0.
    try:
        for expiry_index, end in enumerate(times):
            steps = max(1, math.ceil((end-previous)*cfg['steps_per_year']))
            edges = adapter.boundary.time_nodes(previous,float(end),cfg['steps_per_year']) if hasattr(adapter.boundary,'time_nodes') else np.linspace(previous,end,steps+1)
            steps=len(edges)-1
            counters = {name:0 for name in ('fine_wing_states','fine_short_states','fine_knot_states',
                                          'coarse_wing_states','fine_evaluations','coarse_evaluations')}
            visited = np.zeros(n, bool); maximum_vol = 0.; maximum_point = {}
            minimum_k = float(min(fine.min(), coarse.min())); maximum_k = float(max(fine.max(), coarse.max()))
            for left, right in zip(edges[:-1], edges[1:]):
                dt = float(right-left)
                vc, flags = coefficient(adapter, float(left), coarse, True)
                counters['coarse_wing_states'] += int(flags['wing'].sum())
                counters['coarse_evaluations'] += n
                index = int(np.argmax(vc)); candidate = float(np.sqrt(vc[index]))
                if candidate > maximum_vol:
                    maximum_vol = candidate
                    maximum_point = dict(maximum_volatility_years=float(left),
                        maximum_volatility_k=float(coarse[index]), maximum_volatility_level='coarse',
                        maximum_volatility_in_wing=bool(flags['wing'][index]))
                dw_total = np.zeros(n)
                subedges = np.linspace(left, right, factor+1)
                for a, b in zip(subedges[:-1], subedges[1:]):
                    h = float(b-a)
                    vf, flags = coefficient(adapter, float(a), fine, True)
                    counters['fine_wing_states'] += int(flags['wing'].sum())
                    counters['fine_short_states'] += n*int(flags['short'])
                    counters['fine_knot_states'] += n*int(flags['knot'])
                    counters['fine_evaluations'] += n
                    visited |= flags['wing']
                    index = int(np.argmax(vf)); candidate = float(np.sqrt(vf[index]))
                    if candidate > maximum_vol:
                        maximum_vol = candidate
                        maximum_point = dict(maximum_volatility_years=float(a),
                            maximum_volatility_k=float(fine[index]), maximum_volatility_level='fine',
                            maximum_volatility_in_wing=bool(flags['wing'][index]))
                    normal = rng.standard_normal(half)
                    dw = np.r_[normal, -normal]*math.sqrt(h)
                    dw_total += dw
                    before = fine.copy()
                    # Under Q, X=S/F(0,t) has dX/X=sigma*dW. We store log X,
                    # hence -variance/2. No physical historical drift appears.
                    # vf is LOCAL variance from the completed evaluator, not
                    # implied volatility read from an option smile.
                    fine += -.5*vf*h+np.sqrt(vf)*dw
                    _audit_state(fine, adapter, t=float(b), dt=h, level='fine',
                                 previous=before, variance=vf, dw=dw)
                    # Read-only snapshots after a valid step; never consume the pricing RNG.
                    if recorder is not None: recorder.record(float(b), fine)
                    minimum_k = min(minimum_k, float(fine.min()))
                    maximum_k = max(maximum_k, float(fine.max()))
                brownian += dw_total
                before = coarse.copy()
                coarse += -.5*vc*dt+np.sqrt(vc)*dw_total
                _audit_state(coarse, adapter, t=float(right), dt=dt, level='coarse',
                             previous=before, variance=vc, dw=dw_total)
                minimum_k = min(minimum_k, float(coarse.min()))
                maximum_k = max(maximum_k, float(coarse.max()))
            xfine, xcoarse = np.exp(fine), np.exp(coarse)
            f, discount = adapter.curves.at(float(end))
            for level, x in (('fine', xfine), ('coarse', xcoarse)):
                mean, se = estimate(pair_samples(x))
                forwards.append(dict(years=float(end), level=level, expected_forward=f,
                    simulated_forward=f*mean, normalized_mean=mean, standard_error=se,
                    confidence_multiplier=z, tolerance=cfg['forward_relative_tolerance']+z*se,
                    passed=bool(abs(mean-1) <= cfg['forward_relative_tolerance']+z*se)))
            selected = [q for q in quotes if abs(float(q['years'])-end) <= 1e-10]
            if not selected: raise ValueError('mc_expiry_without_supported_quotes')
            for q in selected: output.append(_quote_statistics(q, xfine, xcoarse, z, cfg, brownian))
            boundary_rows.append(dict(years=float(end), interval_start=float(previous),
                **counters, fine_paths_visiting_wings=int(visited.sum()), paths=n,
                maximum_evaluated_volatility=maximum_vol, **maximum_point, minimum_log_state=minimum_k,
                maximum_log_state=maximum_k, coarse_steps=steps, fine_steps=steps*factor,
                bounds=list(adapter.bounds), clipped_paths=0, discarded_paths=0))
            total_steps += steps; previous = float(end)
            if progress: progress(f"Risk-neutral MC expiry {expiry_index+1}/{len(times)}: {len(selected)} quotes assessed")
    except (ValueError, OverflowError, FloatingPointError) as exc:
        failure = exc if isinstance(exc, SimulationStopped) else SimulationStopped(str(exc))
        failure.partial = dict(quotes=output, forwards=forwards, boundaries=boundary_rows)
        failure.details.setdefault('expiry_index', expiry_index+1)
        failure.details.setdefault('interval_start', float(previous))
        failure.details.setdefault('interval_end', float(end))
        raise failure from (None if failure is exc else exc)
    return output, forwards, boundary_rows, dict(coarse_steps=total_steps,
        fine_steps=total_steps*factor, independent_pairs=half, confidence_multiplier=z)


@dataclass
class ValidationResult:
    source_folder: Path
    run_id: str
    manifest: dict
    tables: dict
    path_data: dict = field(default_factory=dict)

    def save(self):
        """Publish a new checksummed derived folder; never overwrite sources."""
        parent = self.source_folder/'local_vol_mc'
        target = parent/self.run_id
        if not self.run_id.isalnum() or target.is_symlink() or target.resolve().parent != parent.resolve():
            raise ValueError('unsafe_local_vol_mc_output')
        expected = {name:name+'.csv' for name in self.tables}
        expected['report'] = 'report.html'
        expected.update(reporting.assets(self))
        def verify(folder):
            m = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
            if (not isinstance(m, dict) or m.get('complete') is not True
                    or m.get('identity') != self.manifest['identity']
                    or not isinstance(m.get('files'), dict) or set(m['files']) != set(expected)):
                raise ValueError('mc_existing_output_inconsistent')
            for key, name in expected.items():
                if (not isinstance(m['files'][key], dict) or m['files'][key].get('file') != name
                        or m['files'][key].get('sha256') != _hash((folder/name).read_bytes())):
                    raise ValueError('mc_existing_output_checksum_mismatch')
            return folder
        if target.exists(): return verify(target)
        parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix='.mc-', dir=parent) as temp:
            stage = Path(temp)/'out'; stage.mkdir()
            for name, rows in self.tables.items():
                fields = list(dict.fromkeys(k for row in rows for k in row)) or ['status']
                with (stage/(name+'.csv')).open('w', encoding='utf-8', newline='') as stream:
                    w = csv.DictWriter(stream, fieldnames=fields); w.writeheader(); w.writerows(rows)
            for gid, data in self.path_data.items():
                np.savez_compressed(stage/('paths_'+reporting.token(gid)+'.npz'), **data)
            reporting.plot(self, stage)
            (stage/'report.html').write_text(_report(self), encoding='utf-8')
            m = {**self.manifest, 'complete':True, 'files':{key:dict(file=name,
                sha256=_hash((stage/name).read_bytes())) for key,name in expected.items()}}
            write_json(stage/'manifest.json', m); verify(stage)
            # A concurrent writer must have published identical checked content.
            if target.exists(): return verify(target)
            stage.rename(target)
        return target


def validate_start_date(local, surface, adapter):
    """Enforce COB+1 without shifting a calibrated surface to another day.

    A date is interpreted in the calibration's declared valuation timezone.
    Keep the actual intraday reference instant on that day: moving it to
    midnight would alter option time-to-expiry, forwards and discounting.
    Historical surfaces dated COB therefore cannot price a COB+1 start unless
    matching inputs have independently been supplied and calibrated for COB+1.
    """
    cob = local.manifest.get('archive_cob')
    expected = forecast_start_for_cob(cob)
    if surface.manifest.get('archive_cob') != cob:
        raise ValueError('mc_surface_archive_cob_mismatch')
    zone = surface.manifest['identity']['calibration_identity']['config']['valuation_timezone']
    if adapter.reference.tzinfo is None or adapter.reference.utcoffset() is None:
        raise ValueError('mc_reference_timestamp_must_be_timezone_aware')
    observed = adapter.reference.astimezone(ZoneInfo(zone)).date().isoformat()
    if observed != expected:
        raise ValueError('mc_start_date_mismatch: required=' + expected + '; surface=' + observed
                         + '; matching dated option/spot/curve inputs are required')
    return dict(requested_start_date=expected, start_date=observed, valuation_timezone=zone,
                start_timestamp=adapter.reference.isoformat(), start_policy='cob_plus_one')


def validation_gates(prices, forwards):
    """Every assessed quote/forward must pass. There is no percentage quorum.

    Empty evidence is never a pass. Preserve separate gate names so a precision
    failure cannot be mistaken for a price fit or forward/convergence failure.
    """
    gates = ('fit_pass', 'pde_pass', 'convergence_pass', 'precision_pass')
    values = {key: bool(prices) and all(q[key] for q in prices) for key in gates}
    values['forward_pass'] = bool(forwards) and all(r['passed'] for r in forwards)
    failed = [key for key, passed in values.items() if not passed]
    return values, failed


def validate_local_vol(local, surface, config_file, progress=None):
    """Use this run's validated objects. No archive guessing or new retrieval.

    Surface is used only to recover call/put metadata by record ID. The saved
    PDE assessment supplies reference strike/time/prices; existing archives
    therefore need no mutation just to add an option-type field.
    """
    cfg = load_config(config_file)
    if local.manifest['identity']['source_surface_id'] != surface.run_id:
        raise ValueError('mc_surface_local_identity_mismatch')
    tables = dict(cohorts=[], quotes=[], forwards=[], boundaries=[], expiries=[], failures=[])
    path_data = {}
    rights = {(r['group_id'],r['record_id']):r['option_type'] for r in surface.tables['repricing']}
    for row in local.tables['state_mapping']:
        gid = row['group_id']; context = {k:row.get(k) for k in ('group_id','underlying','currency','route')}
        assessment = dict(**context, status='skipped', reason=None, portfolio_simulation_ready=False,
                          start_policy='cob_plus_one')
        adapter = local.state_evaluators.get(gid)
        if not cfg['enabled']: assessment['reason'] = 'disabled'
        elif adapter is None: assessment['reason'] = 'validated_state_evaluator_unavailable'
        else:
            try:
                assessment.update(validate_start_date(local, surface, adapter))
                quotes = [{**q, 'option_type':rights[(gid,q['record_id'])]} for q in local.tables['option_repricing']
                          if q['group_id']==gid and q['status']=='assessed']
                # Stable per-cohort seeds: changing ordering or adding another
                # underlying does not change this cohort's random sequence.
                seed = np.random.SeedSequence([cfg['seed'], int(_hash(gid.encode())[:8],16)])
                recorder = reporting.PathRecorder(adapter, cfg)
                prices, forwards, bounds, details = simulate_cohort(adapter, quotes, cfg, seed, progress, recorder)
                captured, arrays = recorder.finish()
                for name, rows in captured.items():
                    tables.setdefault(name, []).extend({**context, **r} for r in rows)
                if arrays is not None: path_data[gid] = arrays
                for name, rows in (('quotes',prices),('forwards',forwards),('boundaries',bounds)):
                    tables[name].extend({**context,**r} for r in rows)
                gate_values, failed = validation_gates(prices, forwards)
                assessment.update(status='validated_with_boundary_assumptions' if not failed else 'review_required',
                    reason=None if not failed else 'failed_checks: ' + ', '.join(failed),
                    failed_checks=failed, quotes=len(prices), expiries=len(adapter.expiries),
                    paths=cfg['paths'], **details, **gate_values,
                    end_timestamp=adapter.expiries[-1].isoformat())
            except SimulationStopped as exc:
                for name, rows in exc.partial.items():
                    tables[name].extend({**context,**r} for r in rows)
                tables['failures'].append({**context, 'reason':str(exc), **exc.details})
                assessment.update(status='unavailable', reason=str(exc),
                    completed_expiries=len(exc.partial['boundaries']), expected_expiries=len(adapter.expiries),
                    partial_quotes=len(exc.partial['quotes']), paths=cfg['paths'])
            except (ValueError, KeyError, OverflowError, FloatingPointError) as exc:
                assessment.update(status='unavailable', reason=str(exc))
            # Never interpret the retained early expiries as a complete pass.
            prices=[q for q in tables['quotes'] if q['group_id']==gid]
            gates=('fit_pass','pde_pass','convergence_pass','precision_pass')
            for expiry in sorted({q['expiration'] for q in prices}):
                group = [q for q in prices if q['expiration']==expiry]
                tables['expiries'].append(dict(**context, expiration=expiry, quotes=len(group),
                    **{k:all(q[k] for q in group) for k in gates},
                    **{k+'_count':sum(q[k] for q in group) for k in gates},
                    maximum_absolute_price_error=max(abs(q['price_error']) for q in group),
                    maximum_standard_error=max(q['standard_error'] for q in group),
                    maximum_paired_difference=max(abs(q['paired_difference']) for q in group)))
        tables['cohorts'].append(assessment)
    identity = dict(engine_version=3, config=cfg, source_local_id=local.run_id,
        start_policy='cob_plus_one', archive_cob=local.manifest.get('archive_cob'),
        date_policy_code_hash=_hash(Path(__file__).with_name('cob.py').read_bytes()),
        source_identity=local.manifest['identity'],
        source_content_hash=_hash(_bytes(dict(tables=local.tables, rights=sorted((str(k),v) for k,v in rights.items())))),
        code_hash=_hash(Path(__file__).read_bytes()),
        reporting_code_hash=_hash(Path(reporting.__file__).read_bytes()),
        plotting_version=version('matplotlib') if reporting.settings(cfg)['plots'] else None,
        dependencies={k:version(k) for k in ('numpy','scipy')}, python=sys.version.split()[0])
    summary = dict(cohorts=len(tables['cohorts']),
        validated_cohorts=sum(r['status']=='validated_with_boundary_assumptions' for r in tables['cohorts']),
        portfolio_simulation_ready=False)
    return ValidationResult(local.source_folder, _hash(_bytes(identity))[:16],
        dict(identity=identity,summary=summary,archive_cob=local.manifest.get('archive_cob')),tables,path_data)


def _report(result):
    """Compact readable report; full per-quote evidence stays in CSV files."""
    def table(rows, fields):
        def fmt(x):
            if isinstance(x,float): return f'{x:.6g}'
            return '' if x is None else str(x)
        return '<div class="scroll"><table><tr>'+''.join('<th>'+escape(k.replace('_',' '))+'</th>' for k in fields)+'</tr>'+''.join(
            '<tr>'+''.join('<td>'+escape(fmt(r.get(k)))+'</td>' for k in fields)+'</tr>' for r in rows)+'</table></div>'
    cfg = result.manifest['identity']['config']
    body = ['<h1>Risk-neutral local-volatility Monte Carlo validation</h1>',
        '<p>Saved-data underlying validation only. No portfolio simulation or downloads. '
        'The start date must equal archive COB + one calendar day. The actual observation time on that day is retained; mismatched surfaces are blocked, never redated. '
        'Short-end and wing continuations are model assumptions, not observed market evidence.</p>',
        table(result.tables['cohorts'],['underlying','group_id','status','reason','start_date','start_timestamp','quotes','expiries','forward_pass','fit_pass','pde_pass','convergence_pass','precision_pass']),
        '<h2>Method and acceptance</h2><p>X=S/F; log-Euler with variance correction, antithetic pairs, '
        'configured fixed control variate (delta or coupled Black payoff), coupled coarse/fine Brownian paths. '
        'Standard errors use independent PAIR means. No volatility cap or discarded paths.</p>',
        '<p>The coupled Black control uses an auxiliary lognormal payoff with the same Brownian motion, '
        'subtracts its simulated payoff and adds its independently calculated analytic expectation. '
        'A residual stock control preserves parity. Raw and legacy delta-control prices/errors remain in quotes.csv; '
        'neither control changes the local-volatility dynamics or acceptance budgets.</p>',
        '<p>Each price error and paired step difference must fit its deterministic budget plus z times its own SE. '
        'Bonferroni z uses the configured family error probability; normal intervals are approximate. '
        'Every assessed quote and forward must pass: there is no 80% or other percentage quorum. An additional SE ceiling prevents an imprecise run from passing with arbitrarily wide intervals. '
        'Raw forward means are tested independently of the price control variate. '
        'A pass is evidence at these resolutions, not proof of asymptotic convergence.</p>',
        '<pre>'+escape(json.dumps({k:v for k,v in cfg.items() if k!='instructions'},indent=2))+'</pre>',
        reporting.section(result),
        '<h2>Expiry checks</h2><p>Pass counts are numbers of assessed quotes passing each gate; every quote must pass.</p>',table(result.tables['expiries'],['underlying','expiration','quotes','fit_pass_count','pde_pass_count','convergence_pass_count','precision_pass_count','maximum_absolute_price_error','maximum_standard_error','maximum_paired_difference']),
        '<h2>Forward consistency</h2>',table(result.tables['forwards'],['underlying','years','level','normalized_mean','standard_error','tolerance','passed']),
        '<h2>Boundary usage</h2><p>Counts are evaluated states, not unique paths; wing-path counts are per expiry interval. '
        'The short-end assumption applies to all paths before the first expiry. Any computational-boundary exit aborts the cohort. '
        'Volatility columns are annualized decimals: 1.0 means 100%, not 1%.</p>',
        table(result.tables['boundaries'],['underlying','years','fine_paths_visiting_wings','paths','fine_wing_states','fine_short_states','fine_evaluations','maximum_evaluated_volatility','maximum_volatility_in_wing','maximum_volatility_k','minimum_log_state','maximum_log_state']),
        '<h2>Stopped-path diagnostics</h2><p>An unavailable cohort is NOT validated. Earlier expiry rows, if present, are partial diagnostics only.</p>',
        table(result.tables.get('failures',[]),['underlying','reason','timestamp','level','dt','failed_paths','log_state_before','log_state_after','annualized_volatility_before','brownian_increment','bounds']),
        '<h2>Detailed evidence</h2><ul>'+''.join('<li><a href="'+name+'.csv">'+name+'.csv</a></li>' for name in result.tables)+'</ul>',
        '<p>Portfolio readiness remains false. This stage does not define index-to-holding proxy mapping, other holding dynamics or joint FX/portfolio dependence.</p>']
    return '<!doctype html><html><head><meta charset="utf-8"><title>Local-volatility MC validation</title><style>body{font:16px system-ui;max-width:1450px;margin:35px auto;padding:0 24px;color:#172b4d}h2{margin-top:32px}.scroll{overflow-x:auto}table{border-collapse:collapse;font-size:14px}td,th{padding:9px;border:1px solid #ccd5df;text-align:left}th{background:#edf3fa}pre{background:#f4f6f8;padding:16px;overflow:auto}</style></head><body>'+''.join(body)+'</body></html>'
