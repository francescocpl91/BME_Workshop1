"""Surface checks and the final bounded-input eligibility contract.

Four responsibilities: price quality, refit/derivative stability, permission
and supported domain, plus optional research diagnostics. No I/O or downloads.
"""
from datetime import datetime
import math
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix
from scipy.special import ndtr
from .surface_model import black_prices, fitted_prices


# Price-fit quality
def _finite(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def quote_quality(point, price, iv, policy=None):
    """Evaluate a fitted quote against original price and own-calculated IV bounds.

    Missing/invalid IV intervals are unassessed, not passing or fabricated.
    Historical prices retain observed-close/IV diagnostics but acquire no spread.
    A legacy policy without practical thresholds retains its former acceptance.
    """
    live = point['route'] == 'live'
    observed = float(point['option_price'])
    result = dict(
        inside_spread=None, relative_observed_price_error=None,
        outside_spread_price_error=None, relative_outside_spread_error=None,
        outside_half_spreads=None, absolute_iv_error=abs(float(iv)-float(point['iv'])),
        outside_iv_error=None, practical_price_pass=None, practical_spread_pass=None,
        practical_iv_pass=None, practical_pass=None,
        practical_status='unassessed' if live else 'not_applicable_historical')
    if not live:
        if observed > 0:
            result['relative_observed_price_error'] = abs(price-observed)/observed
        return result

    bid, ask = _finite(point.get('bid')), _finite(point.get('ask'))
    if bid is not None and ask is not None and 0 <= bid <= ask:
        mid, half = (bid+ask)/2, (ask-bid)/2
        outside = max(bid-price, price-ask, 0.)
        # Only floating-point rounding tolerance; strict coverage is diagnostic.
        result['inside_spread'] = bool(bid-1e-10 <= price <= ask+1e-10)
        result['outside_spread_price_error'] = float(outside)
        if mid > 0:
            result['relative_observed_price_error'] = abs(price-mid)/mid
            result['relative_outside_spread_error'] = outside/mid
        if half > 0:
            result['outside_half_spreads'] = outside/half

    bid_iv, ask_iv = _finite(point.get('bid_iv')), _finite(point.get('ask_iv'))
    if bid_iv is not None and ask_iv is not None and 0 < bid_iv <= ask_iv:
        result['outside_iv_error'] = float(max(bid_iv-iv, iv-ask_iv, 0.))

    if policy is None or 'minimum_practical_coverage' not in policy:
        result['practical_status'] = 'not_enabled_legacy_policy'
        return result
    # AND, not OR: neither a tiny percentage of an expensive premium nor a
    # small IV error on a cheap option can mask a breach of another condition.
    for error, limit, flag in (
        ('relative_outside_spread_error', 'maximum_relative_outside_error', 'practical_price_pass'),
        ('outside_half_spreads', 'maximum_outside_half_spreads', 'practical_spread_pass'),
        ('outside_iv_error', 'maximum_outside_iv_error', 'practical_iv_pass')):
        value = result[error]
        if value is not None:
            result[flag] = bool(value <= policy[limit]+1e-12)
    flags = [result[k] for k in ('practical_price_pass', 'practical_spread_pass', 'practical_iv_pass')]
    if all(value is not None for value in flags):
        result['practical_pass'] = all(flags)
        result['practical_status'] = 'passed' if all(flags) else 'failed'
    return result


def summarize_quality(points, rows, policy=None):
    """Expiry statistics with all live observations in the coverage denominator.

    Missing IV bounds must not improve coverage by shrinking the denominator.
    Quantiles exclude missing values and expose their available sample counts.
    Historical-only cohorts have no practical coverage; their existing price
    RMSE gate continues to apply, without inventing quote or IV intervals.
    """
    live = [row for p, row in zip(points, rows) if p['route'] == 'live']
    enabled = policy is not None and 'minimum_practical_coverage' in policy
    assessed = sum(row['practical_pass'] is not None for row in live) if enabled else 0
    passed = sum(row['practical_pass'] is True for row in live) if enabled else 0
    fraction = passed/len(live) if live and enabled else None
    result = dict(
        practical_coverage_fraction=fraction,
        practical_live_observations=len(live),
        practical_assessed_observations=assessed,
        practical_unassessed_observations=len(live)-assessed if enabled else None,
        practical_passed_observations=passed if enabled else None,
        practical_coverage_pass=bool(fraction >= policy['minimum_practical_coverage']) if fraction is not None else None)
    for key in ('relative_observed_price_error', 'relative_outside_spread_error',
                'outside_half_spreads', 'absolute_iv_error', 'outside_iv_error'):
        values = [row[key] for row in rows if row[key] is not None]
        result[key+'_observations'] = len(values)
        for name, q in (('median', .5), ('p95', .95), ('maximum', 1.)):
            result[name+'_'+key] = float(np.quantile(values, q)) if values else None
    return result

# Refit scope and stability
def coverage_statistics(points, policy):
    """Count distinct strikes and original k=log(K/F), exactly as selection does."""
    ks = sorted({p['k'] for p in points})
    gap = max((b-a for a, b in zip(ks, ks[1:])), default=0.)
    near = policy['near_forward_abs_k']
    result = dict(unique_strikes=len({p['strike'] for p in points}),
                  below=sum(k < 0 for k in ks), above=sum(k > 0 for k in ks),
                  near=sum(abs(k) <= near for k in ks),
                  near_below=sum(-near <= k < 0 for k in ks),
                  near_above=sum(0 < k <= near for k in ks), largest_gap_k=gap)
    result['count_pass'] = (result['unique_strikes'] >= policy['minimum_unique_strikes']
        and result['below'] >= policy['minimum_each_side']
        and result['above'] >= policy['minimum_each_side']
        and result['near'] >= policy['minimum_near_forward'])
    result['gap_pass'] = gap <= policy['maximum_strike_gap_k']
    result['coverage_pass'] = result['count_pass'] and result['gap_pass']
    return result


def stability_decision(record, policy):
    """Return PASS / FAIL / INCOMPLETE for the recorded scope, even for diagnostics.

    The caller decides whether a record is required for aggregate acceptance.
    Unassessed, empty and non-finite comparisons can never become passing tests.
    """
    if record['status']=='optimizer_failed':
        return 'FAIL'
    for field, limit in [('max_iv_change','maximum_iv_change'),
                         ('holdout_scaled_rmse','maximum_holdout_scaled_rmse')]:
        value=record.get(field)
        if value is not None and (not math.isfinite(value) or value>policy[limit]):
            return 'FAIL'
    if record['status']!='checked' or record.get('max_iv_change') is None:
        return 'INCOMPLETE'
    if record.get('holdout_requested',0)>0 and (
            not record.get('holdout_evaluated') or record.get('holdout_scaled_rmse') is None):
        return 'INCOMPLETE'
    if record.get('validation_scope')=='supported_domain':
        # The old full-maturity flag remains honest (the training maturity was
        # deliberately removed). The new flag assesses only declared support.
        if record.get('scope_coverage_complete') is not True:
            return 'INCOMPLETE'
    elif record.get('full_maturity_coverage') is False:
        return 'INCOMPLETE'
    return 'PASS'


def aggregate_stability(records, policy):
    """Timing diagnostics remain in the output but do not masquerade as gates."""
    required=[r for r in records if r.get('required',True)]
    decisions=[stability_decision(r,policy) for r in required]
    if 'FAIL' in decisions:return 'failed'
    if not decisions or 'INCOMPLETE' in decisions:return 'incomplete'
    return 'passed'


def assess_holdout(other, held, policy, supported=False):
    """Separate evaluable quotes from unsupported points before repricing.

    Domain checks use the refitted model's actual no-extrapolation policy.
    For a supported maturity test, the remaining points must still meet all
    per-expiry strike-coverage requirements. This prevents a trivial pass on
    one surviving quote or an empty domain. No source quote is discarded.
    """
    valid=[];outside=[];domains={}
    for point in held:
        low=high=None
        try:
            low,high=map(float,other.domain(point['reference_years']))
            within=low<=point['k']<=high
            reason=None if within else 'outside_supported_strike_domain'
        except ValueError:
            within=False;reason='outside_supported_maturity_domain'
        domains[point['expiration']]=dict(expiration=point['expiration'],
            years=point['reference_years'],k_min=low,k_max=high)
        if within:valid.append(point)
        else:outside.append(dict(record_id=point.get('record_id'),instrument_id=point.get('instrument_id'),
            expiration=point['expiration'],strike=point['strike'],k=point['k'],
            years=point['reference_years'],supported_k_min=low,supported_k_max=high,
            reason=reason,status='unvalidated_wing' if low is not None else 'unvalidated_maturity'))
    enough=True
    for expiry,domain in domains.items():
        group=[p for p in valid if p['expiration']==expiry]
        stats=coverage_statistics(group,policy)
        domain.update(supported_observations=len(group),coverage_pass=stats['coverage_pass'],coverage_required=supported,
                      unique_strikes=stats['unique_strikes'],near_forward=stats['near'],
                      below=stats['below'],above=stats['above'],largest_gap_k=stats['largest_gap_k'])
        enough=enough and stats['coverage_pass']
    complete=bool(valid) and (enough if supported else not outside)
    return valid,dict(holdout_requested=len(held),holdout_supported=len(valid),
        holdout_unsupported=len(outside),holdout_evaluated=len(valid),
        validation_scope='supported_domain' if supported else 'all_requested_quotes',
        scope_coverage_complete=complete,unsupported_observations=outside,
        validation_domains=list(domains.values()))

# Derivative refits and maturity jumps
METRICS = ('w_k', 'w_kk', 'w_t', 'density_factor')


def validate_policy(p):
    """Optional for old configurations; if supplied, every field is explicit."""
    keys = {'enabled', 'strike_points', 'relative_tolerance', 'absolute_tolerances',
            'maximum_p95_score', 'maximum_score', 'minimum_density_factor',
            'negative_tolerance', 'jump_step_fraction', 'jump_convergence_score'}
    if not isinstance(p, dict) or set(p) != keys:
        raise ValueError('invalid_derivative_diagnostics_policy')
    if type(p['enabled']) is not bool or type(p['strike_points']) is not int or p['strike_points'] < 7:
        raise ValueError('invalid_derivative_diagnostics_grid')
    if not isinstance(p['absolute_tolerances'], dict) or set(p['absolute_tolerances']) != set(METRICS):
        raise ValueError('invalid_derivative_absolute_tolerances')
    values = [v for k, v in p.items() if k not in ('enabled', 'strike_points', 'absolute_tolerances')]
    values += list(p['absolute_tolerances'].values())
    if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError('invalid_derivative_diagnostics_number')
    if not (p['jump_step_fraction'] < .01 and p['relative_tolerance'] < 1
            and p['maximum_p95_score'] <= p['maximum_score']):
        raise ValueError('inconsistent_derivative_diagnostics_policy')


def _score(a, b, metric, policy):
    # An absolute allowance protects comparisons near zero. Elsewhere the
    # allowance scales with the larger derivative magnitude, symmetrically.
    return abs(b-a)/(policy['absolute_tolerances'][metric]
                     + policy['relative_tolerance']*max(abs(a), abs(b)))


def _support(model, t):
    return tuple(map(float, getattr(model, 'audit_domain', model.domain)(t)))


def _point(metric, a, b, k, t, policy, **extra):
    a, b = float(a), float(b)
    finite = math.isfinite(a) and math.isfinite(b)
    return dict(metric=metric, years=float(t), k=float(k), baseline=a if math.isfinite(a) else None,
        alternative=b if math.isfinite(b) else None, absolute_change=abs(b-a) if finite else None,
        score=_score(a, b, metric, policy) if finite else None, finite=finite, **extra)


def _summary(points, policy, incomplete=False):
    """Summaries retain missing evidence; it can never silently become a pass."""
    records=[]
    for metric in METRICS:
        all_rows=[r for r in points if r['metric']==metric]
        valid=[r for r in all_rows if r.get('finite')]
        result=dict(metric=metric, requested=len(all_rows), evaluated=len(valid))
        if not valid:
            records.append(dict(result, status='INCOMPLETE')); continue
        scores=np.array([r['score'] for r in valid]); changes=np.array([r['absolute_change'] for r in valid])
        worst=max(valid, key=lambda r:r['score'])
        minimum=min(min(r['baseline'],r['alternative']) for r in valid)
        negative=metric in ('w_t','density_factor') and minimum < -policy['negative_tolerance']
        near_zero=metric=='density_factor' and minimum < policy['minimum_density_factor']
        review=(np.quantile(scores,.95)>policy['maximum_p95_score'] or max(scores)>policy['maximum_score']
                or negative or near_zero)
        status='REVIEW_REQUIRED' if review else 'INCOMPLETE' if incomplete or len(valid)!=len(all_rows) else 'PASS'
        records.append(dict(result, status=status, median_absolute_change=float(np.median(changes)),
            p95_absolute_change=float(np.quantile(changes,.95)), maximum_absolute_change=float(max(changes)),
            p95_score=float(np.quantile(scores,.95)), maximum_score=float(max(scores)),
            worst_years=worst['years'], worst_k=worst['k'], minimum_value=minimum,
            negative_value=negative, near_zero_density=near_zero))
    return records


def assessment_status(rows):
    statuses=[r['status'] for r in rows]
    if 'REVIEW_REQUIRED' in statuses:return 'review_required'
    if not statuses or any(s!='PASS' for s in statuses):return 'incomplete'
    return 'passed'


def compare_derivatives(baseline, alternative, policy):
    """Compare complete refits on quarter/mid/three-quarter interval grids.

    Split at the UNION of both sets of maturity nodes. This avoids confusing a
    maturity derivative jump with an interior derivative sensitivity. Missing
    time/strike ranges are separately recorded, never filled by extrapolation.
    A PASS is explicitly limited to the shared domain.
    """
    low=max(baseline.times[0],alternative.times[0]); high=min(baseline.times[-1],alternative.times[-1])
    knots=sorted({float(t) for t in [low,high,*baseline.times,*alternative.times] if low<=t<=high})
    points=[]; domains=[]; incomplete=False
    for left,right in zip(knots[:-1],knots[1:]):
        for fraction in (.25,.5,.75):
            t=left+(right-left)*fraction
            try:
                # A sampled time-domain bound is not valid for the whole
                # interval when an SSVI boundary slopes toward a narrow node.
                # Both model families provide exact conservative interval
                # support; sample derivatives only inside that fixed strip.
                alo,ahi=baseline.interval_domain(left,right)
                blo,bhi=alternative.interval_domain(left,right)
                lo,hi=max(alo,blo),min(ahi,bhi)
                if not np.isfinite([lo,hi]).all() or hi<=lo:raise ValueError('empty_common_domain')
                ks=np.linspace(lo,hi,policy['strike_points'])
                a=baseline.evaluate(ks,t); b=alternative.evaluate(ks,t)
                domains.append(dict(years=t, interval_start=left, interval_end=right, k_min=lo,k_max=hi,
                    domain_scope='whole_interval',
                    baseline_k_min=alo,baseline_k_max=ahi,alternative_k_min=blo,alternative_k_max=bhi,
                    baseline_wings_excluded=lo>alo or hi<ahi,status='checked'))
                for metric in METRICS:
                    points.extend(_point(metric,x,y,k,t,policy,interval_start=left,interval_end=right)
                                  for x,y,k in zip(a[metric],b[metric],ks))
            except (ValueError,FloatingPointError) as exc:
                incomplete=True; domains.append(dict(years=t,interval_start=left,interval_end=right,
                    status='unassessed',reason=str(exc)))
    rows=_summary(points,policy,incomplete=incomplete)
    return dict(status=assessment_status(rows), metrics=rows,points=points,domains=domains,
        baseline_time_min=float(baseline.times[0]),baseline_time_max=float(baseline.times[-1]),
        common_time_min=float(low),common_time_max=float(high),
        full_time_range=low==baseline.times[0] and high==baseline.times[-1],scope='common_supported_domain')


def maturity_jumps(model, policy):
    """Approach each interior node from BOTH sides without crossing another node.

    Evaluate at epsilon, epsilon/10 and epsilon/100, using the same k grid in
    the intersection of all neighboring support. We report the finest left
    and right limits and whether the estimated jump settled as epsilon shrank.
    This measures the existing interpolation; it does not repair or smooth it.
    """
    points=[]; summaries=[]; domains=[]
    for i,t in enumerate(model.times[1:-1],1):
        epsilon=min(t-model.times[i-1],model.times[i+1]-t)*policy['jump_step_fraction']
        try:
            # Include the knot and every approach point: for other supported
            # surface families the domain can narrow at the knot itself.
            bounds=[_support(model,t)]+[_support(model,t+s*epsilon/divisor)
                for divisor in (1,10,100) for s in (-1,1)]
            lo=max(x[0] for x in bounds);hi=min(x[1] for x in bounds)
            if hi<=lo:raise ValueError('empty_two_sided_domain')
            ks=np.linspace(lo,hi,policy['strike_points']);evaluations=[]
            for divisor in (1,10,100):
                h=epsilon/divisor
                evaluations.append((model.evaluate(ks,t-h),model.evaluate(ks,t+h)))
            local=[]
            for metric in METRICS:
                left,right=evaluations[-1]
                for j,k in enumerate(ks):
                    row=_point(metric,left[metric][j],right[metric][j],k,t,policy,epsilon_years=epsilon/100)
                    # Difference of the two finest jump estimates, scaled in
                    # the SAME derivative units as the left/right comparison.
                    previous=evaluations[-2][1][metric][j]-evaluations[-2][0][metric][j]
                    latest=right[metric][j]-left[metric][j]
                    coarse=evaluations[0][1][metric][j]-evaluations[0][0][metric][j]
                    denominator=policy['absolute_tolerances'][metric]+policy['relative_tolerance']*max(abs(float(left[metric][j])),abs(float(right[metric][j])))
                    error=float(abs(latest-previous)/denominator)
                    row.update(coarse_jump=float(coarse) if np.isfinite(coarse) else None,
                        signed_jump=float(latest) if np.isfinite(latest) else None,
                        convergence_score=error if math.isfinite(error) else None)
                    local.append(row)
            records=_summary(local,policy)
            for row in records:
                errors=[r['convergence_score'] for r in local if r['metric']==row['metric']]
                converged=all(e is not None and e<=policy['jump_convergence_score'] for e in errors)
                row.update(years=float(t),epsilon_years=epsilon/100,limit_estimate_converged=converged,
                    maximum_convergence_score=max(errors) if all(e is not None for e in errors) else None)
                if not converged and row['status']=='PASS':row['status']='INCOMPLETE'
            summaries.extend(records);points.extend(local)
            domains.append(dict(years=float(t),k_min=lo,k_max=hi,status='checked'))
        except (ValueError,FloatingPointError) as exc:
            summaries.extend(dict(years=float(t),metric=m,status='INCOMPLETE',reason=str(exc)) for m in METRICS)
            domains.append(dict(years=float(t),status='unassessed',reason=str(exc)))
    return dict(status=assessment_status(summaries),metrics=summaries,points=points,domains=domains)


def refit_assessment(records):
    """Aggregate full-surface refits only; incomplete timing evidence is separate.

    Raw-SVI per-slice competing-start checks retain only IV differences, not
    alternative complete surfaces. Do not claim to have checked their time
    derivatives. This exclusion is explicit in the report's scope statement.
    """
    refits=[r for r in records if r.get('category')!='timing' and
            (r['check'].startswith(('held_out','quote_perturbation','alternative_start')) or r['check']=='stricter_selection')]
    return assessment_status([dict(status={'passed':'PASS','review_required':'REVIEW_REQUIRED'}.get(
        r.get('derivative_diagnostics',{}).get('status'),'INCOMPLETE')) for r in refits])

# Consolidated input eligibility
"""Final input eligibility: a permission contract, never a Dupire calculation."""


def validate_eligibility_policy(policy):
    keys={'allow_intraday_approximation','allow_historical_closes','allow_maturity_jumps','maturity_knots'}
    if not isinstance(policy,dict) or set(policy)!=keys:
        raise ValueError('invalid_surface_eligibility_policy')
    if any(type(policy[k]) is not bool for k in keys-{'maturity_knots'}):
        raise ValueError('invalid_surface_eligibility_boolean')
    # This first contract deliberately avoids choosing a derivative at a jump.
    # A future consumer must keep its complete finite-difference stencil inside
    # an allowed open maturity interval. No silent extrapolation or averaging.
    if policy['maturity_knots']!='exclude':
        raise ValueError('surface_eligibility_requires_excluded_knots')


def assess_local_vol_eligibility(cohort,model,tables,cfg):
    """One consolidated decision per independent underlying/date/route cohort.

    We preserve all existing acceptance thresholds. The only discretionary
    permissions are the explicitly declared data/interpolation approximations.
    Missing evidence blocks permission. A surface can remain a good price fit
    while being blocked here. Eligibility is limited to conversion inputs;
    local-volatility pricing validation and full simulation remain future work.
    """
    gid=cohort['group_id'];rows=lambda name:[r for r in tables.get(name,[]) if r['group_id']==gid]
    policy=cfg.get('eligibility');blockers=[];warnings=[];domains=[];checks=[]
    if policy is None:blockers.append('eligibility_policy_not_configured')
    else:validate_eligibility_policy(policy)
    if model is None:
        blockers.append(cohort.get('reason','no_fitted_surface'))
    fit_ok=(model is not None and cohort.get('market_fit_pass') is True)
    if not fit_ok:blockers.append('usable_inputs_coverage_or_price_fit_failed')
    checks.append(dict(check='Inputs, coverage and price fit',passed=fit_ok))
    numerical=cohort.get('numerical_checks_pass') is True
    if not numerical:blockers.append('arbitrage_or_derivative_accuracy_failed_or_missing')
    checks.append(dict(check='Arbitrage and derivative accuracy',passed=numerical))
    stable=cohort.get('stability_status')=='passed' and cohort.get('derivative_stability_status')=='passed'
    if not stable:blockers.append('refit_or_derivative_stability_failed_or_missing')
    checks.append(dict(check='Refit and derivative stability',passed=stable))
    if cohort.get('route')=='live':
        if not policy or not policy['allow_intraday_approximation']:blockers.append('intraday_approximation_not_permitted')
        warnings.append('Intraday quotes are not a verified simultaneous snapshot; timing stability remains unproven.')
    elif cohort.get('route')=='historical':
        if not policy or not policy['allow_historical_closes']:blockers.append('historical_close_approximation_not_permitted')
        warnings.append('Historical closes use the declared price-error scale; no live bid/ask evidence is implied.')
    else:blockers.append('unsupported_observation_route')
    if model is not None:
        refs=rows('reference_inputs');dates={float(r['years']):r['expiration'] for r in refs}
        refits=[r for r in rows('derivative_refits') if r.get('category')=='numerical']
        tested_domains=[r for r in rows('derivative_refit_domains') if r.get('category')=='numerical']
        # Use a conservative fixed-k strip inside BOTH endpoint smiles. This
        # works for raw SVI and the selectable SSVI family without assuming
        # that a node-only wing has time-derivative support.
        for a,b in zip(model.times[:-1],model.times[1:]):
            try:
                low,high=map(float,model.interval_domain(a,b))
                if not refits:raise ValueError('no_derivative_refit_support')
                for refit in refits:
                    support=[r for r in tested_domains if r['check']==refit['check']
                             and r.get('interval_start',float('inf'))<b
                             and r.get('interval_end',float('-inf'))>a]
                    if refit.get('status')!='passed' or not support or any(
                            r['status']!='checked' or r.get('domain_scope')!='whole_interval' for r in support):
                        raise ValueError('missing_refit_domain_support')
                    # A successful sample in one part of the interval is not
                    # evidence for missing maturity segments elsewhere.
                    reached=a
                    for r in sorted(support,key=lambda r:r['interval_start']):
                        if r['interval_start']>reached:raise ValueError('gap_in_refit_domain_support')
                        reached=max(reached,r['interval_end'])
                    if reached<b:raise ValueError('incomplete_refit_domain_support')
                    low=max(low,max(r['k_min'] for r in support));high=min(high,min(r['k_max'] for r in support))
                if not np.isfinite([a,b,low,high]).all() or not a<b or not low<high:
                    raise ValueError('invalid_supported_interval')
                domains.append(dict(start_years=float(a),end_years=float(b),
                    start_expiry=dates.get(a),end_expiry=dates.get(b),k_min=low,k_max=high,
                    minimum_strike_over_forward=float(np.exp(low)),maximum_strike_over_forward=float(np.exp(high)),
                    maturity_endpoints='excluded',strike_endpoints='included',stencil='stay_inside_this_interval',
                    support='intersection_of_baseline_and_numerical_refit_domains'))
            except (ValueError,FloatingPointError):blockers.append('invalid_supported_interval')
        # Reuse the dense numerical audit and refit conditioning assessment.
        # No additional tuning threshold is introduced for final eligibility.
        grid=rows('surface_grid');dp=cfg.get('derivative_diagnostics',{})
        values=np.asarray([[r.get('w',np.nan),r.get('w_t',np.nan),r.get('density_factor',np.nan)] for r in grid],float)
        conditioning=bool(len(grid) and dp.get('enabled') and np.isfinite(values).all()
            and np.all(values[:,0]>0) and np.all(values[:,1]>=0)
            and np.all(values[:,2]>=dp.get('minimum_density_factor',float('inf'))))
        if not conditioning:blockers.append('nonfinite_negative_or_ill_conditioned_derivatives')
        jumps=rows('maturity_jumps')
        expected={(float(t),m) for t in model.times[1:-1] for m in METRICS}
        observed={(float(r['years']),r['metric']) for r in jumps}
        # Two retained expiries have one interval and no interior knot to
        # assess. An empty jump table is valid ONLY in that case; for three
        # or more expiries it remains missing evidence and blocks permission.
        assessed=bool(observed==expected and len(jumps)==len(expected)
            and all(r.get('limit_estimate_converged') is True and r.get('evaluated',0)==r.get('requested',-1)
                    and r.get('evaluated',0)>0 and r.get('status') in ('PASS','REVIEW_REQUIRED') for r in jumps))
        if not assessed:blockers.append('maturity_boundary_evidence_incomplete')
        # REVIEW_REQUIRED also encodes bad signs/conditioning, not only a
        # large jump. Approximation permission MUST NOT waive those failures.
        # Missing measurements/flags are not a pass. Empty evidence is valid
        # only for the already-assessed two-expiry case with no interior knot.
        boundary_conditions=assessed and all(
            _finite(r.get('minimum_value')) is not None
            and r.get('negative_value') is False and r.get('near_zero_density') is False
            and (r['metric']!='w_t' or r['minimum_value']>=-dp['negative_tolerance'])
            and (r['metric']!='density_factor' or r['minimum_value']>=dp['minimum_density_factor'])
            for r in jumps)
        if not boundary_conditions:blockers.append('invalid_or_missing_maturity_boundary_conditions')
        reviewed=any(r.get('status')=='REVIEW_REQUIRED' for r in jumps)
        if reviewed and (not policy or not policy['allow_maturity_jumps']):blockers.append('maturity_jumps_not_permitted')
        if reviewed:warnings.append('Maturity-derivative jumps exceed the prototype review band; permitted only inside open expiry intervals.')
        domain_ok=bool(domains and len(domains)==len(model.times)-1 and conditioning and boundary_conditions)
    else:domain_ok=False
    if not domain_ok:blockers.append('supported_domain_or_conditioning_unavailable')
    checks.append(dict(check='Supported domain and conditioning',passed=domain_ok))
    if cohort.get('unsupported_holdout_observations',0):
        warnings.append(f"{cohort['unsupported_holdout_observations']} held-out wing observations remain unvalidated; full-wing eligibility is not implied.")
    warnings.append('Upstream curve/convention and carry assumptions remain applicable; no independent verification is added here.')
    blockers=list(dict.fromkeys(blockers));allowed=not blockers
    return dict(status='blocked' if blockers else 'eligible_with_warnings' if warnings else 'eligible',
        allowed=allowed,purpose='bounded_prototype_dupire_input',checks=checks,blockers=blockers,warnings=warnings,
        domain=domains,coordinates='k=log(K/F(T)); T in years from the saved reference timestamp',
        reference_timestamps=sorted({r['reference_timestamp'] for r in rows('reference_inputs')}),
        maturity_knots='excluded',extrapolation='forbidden',stencils='must_remain_inside_one_allowed_interval',
        full_simulation_ready=False,local_volatility_implemented=False)


def allows_surface_point(assessment,years,k):
    """Future consumers can enforce the exported permission without inference.

    Call for EVERY evaluation point in a derivative stencil, not just its
    centre. Exact maturity knots, outside-domain points and blocked cohorts
    are rejected. This helper neither prices nor computes local volatility.
    """
    return allows_surface_stencil(assessment,[(years,k)])


def allows_surface_stencil(assessment,points):
    """All (T,k) stencil coordinates must belong to ONE permitted interval."""
    if assessment.get('allowed') is not True or not points:return False
    for years,k in points:
        if any(isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,float,np.integer,np.floating))
               or not math.isfinite(v) for v in (years,k)):return False
    return any(all(r['start_years']<t<r['end_years'] and r['k_min']<=k<=r['k_max'] for t,k in points)
               for r in assessment.get('domain',[]))

# Optional research diagnostics (not acceptance gates)
def _correlation(first,second):
    if len(first)<3 or np.ptp(first)<1e-12 or np.ptp(second)<1e-12:return None
    return float(np.corrcoef(first,second)[0,1])


def interval_feasibility(points):
    """Test bid/ask compatibility after stable-IV normalization, per expiry.

    Missing IV bounds make this test unavailable; do not manufacture a lower
    bid IV of zero or count a historical assumed error scale as a live spread.
    """
    if any(p['route']!='live' for p in points):return dict(status='not_checked',reason='historical_prices_have_no_observed_spread')
    bounds=[]
    for p in points:
        try:
            bid=float(p['bid_iv']);ask=float(p['ask_iv'])
            if not 0<bid<=ask or not math.isfinite(bid+ask):raise ValueError()
            bounds.append((bid,ask))
        except (KeyError,TypeError,ValueError):return dict(status='not_checked',reason='missing_or_invalid_bid_ask_iv_bounds')
    k=np.array([p['k'] for p in points]);x=np.exp(k);n=len(k);t=points[0]['reference_years']
    if np.any(np.diff(x)<=1e-10):return dict(status='not_checked',reason='duplicate_or_near_duplicate_normalized_strikes')
    lower=black_prices(1.,x,t,1.,np.array(bounds)[:,0],True)
    upper=black_prices(1.,x,t,1.,np.array(bounds)[:,1],True)
    # Digital bounds -1 <= dc/dx <= 0, followed by increasing secant slopes.
    matrix=lil_matrix((3*n-4,n));rhs=np.zeros(3*n-4);row=0
    for j,h in enumerate(np.diff(x)):
        matrix[row,j]=-1;matrix[row,j+1]=1;row+=1
        matrix[row,j]=1;matrix[row,j+1]=-1;rhs[row]=h;row+=1
    for j in range(n-2):
        h1=x[j+1]-x[j];h2=x[j+2]-x[j+1]
        matrix[row,j]=-h2/(h1+h2);matrix[row,j+1]=1;matrix[row,j+2]=-h1/(h1+h2);row+=1
    solution=linprog(np.zeros(n),A_ub=matrix.tocsr(),b_ub=rhs,
        bounds=list(zip(np.maximum(lower,np.maximum(1-x,0)),np.minimum(upper,1))),
        method='highs',options={'primal_feasibility_tolerance':1e-9})
    status='feasible' if solution.success else 'infeasible' if solution.status==2 else 'solver_failed'
    return dict(status=status,solver_status=int(solution.status),reason=str(solution.message),
                scope='discrete_single_expiry_under_intraday_normalization; no_calendar_check')


def diagnostics(points,model,policy,near_forward):
    """Return quote-level conditioning and expiry/bucket summaries.

    Forward sensitivity is FIRST ORDER with volatility held fixed. It shows
    how a declared hypothetical forward perturbation compares with quote
    uncertainty; it is not a revised market forward or a full recalibration.
    """
    prices,_=fitted_prices(model,points)
    quotes=[];expiries=[];buckets=[];feasibility=[]
    bump=policy['forward_bump_fraction']
    for p,price in zip(points,prices):
        call=p['option_type']=='call'
        intrinsic=p['discount_factor']*max((p['forward']-p['strike'])*(1 if call else -1),0.)
        value=p['option_price']-intrinsic
        spread=p['ask']-p['bid'] if p['route']=='live' else None
        sd=p['iv']*np.sqrt(p['time_to_expiry'])
        d1=np.log(p['forward']/p['strike'])/sd+.5*sd
        forward_delta=p['discount_factor']*(ndtr(d1)-(0 if call else 1))
        change=forward_delta*p['forward']*bump
        error=float((price-p['option_price'])/p['error_scale'])
        quotes.append(dict(record_id=p['record_id'],expiration=p['expiration'],k=p['k'],
            capture_timestamp=p['valuation_timestamp'],time_value=float(value),vega=p['vega_per_unit_volatility'],
            spread_over_vega=spread/p['vega_per_unit_volatility'] if spread is not None else None,
            spread_over_time_value=spread/value if spread is not None and value>0 else None,
            scaled_price_error=error,absolute_price_error=float(abs(price-p['option_price'])),
            forward_bump_fraction=bump,first_order_price_change=float(change),
            first_order_change_in_error_scales=float(change/p['error_scale']),
            first_order_equivalent_iv_change=float(change/p['vega_per_unit_volatility'])))
    lookup={r['record_id']:r for r in quotes}
    for expiry in sorted({p['expiration'] for p in points}):
        group=sorted([p for p in points if p['expiration']==expiry],key=lambda p:p['k'])
        stamps=np.array([datetime.fromisoformat(p['valuation_timestamp']).timestamp() for p in group])
        spots=np.array([p['spot'] for p in group]);k=np.array([p['k'] for p in group])
        error=np.array([lookup[p['record_id']]['scaled_price_error'] for p in group])
        expiries.append(dict(expiration=expiry,observations=len(group),capture_minutes=float(np.ptp(stamps)/60),
            recorded_spot_range=float(np.ptp(spots)),strike_time_correlation=_correlation(k,stamps),
            error_time_correlation=_correlation(error,stamps),error_strike_correlation=_correlation(error,k),
            interpretation='correlations_do_not_identify_causation_or_quote_freshness'))
        feasibility.append(dict(expiration=expiry,**interval_feasibility(group)))
        for name,mask in [('downside',k < -near_forward),('near_forward',abs(k)<=near_forward),('upside',k>near_forward)]:
            chosen=[lookup[p['record_id']] for p,keep in zip(group,mask) if keep]
            if not chosen:continue
            buckets.append(dict(expiration=expiry,bucket=name,observations=len(chosen),
                scaled_price_rmse=float(np.sqrt(np.mean([r['scaled_price_error']**2 for r in chosen]))),
                median_absolute_price_error=float(np.median([r['absolute_price_error'] for r in chosen]))))
    return dict(quote_diagnostics=quotes,timing_diagnostics=expiries,residual_buckets=buckets,interval_feasibility=feasibility)
