"""Deterministic forward-Dupire pricing with explicitly assumed boundaries.

Prices a whole strike grid in normalized units c=C/(D*F), x=K/F. The PDE
c_T = 0.5*sigma(T,log x)^2*(c_kk-c_k) evolves from (1-exp(k))_+.
No random numbers, downloads, surface refitting, or implicit repair of bad
coefficients. Boundary extensions are a separate wrapper: the original
LocalVolEvaluator remains strict and never acquires extrapolation permission.
"""
from copy import deepcopy
import math
import numpy as np
from scipy.linalg import solve_banded
from .surface_model import black_prices
from .surface_checks import quote_quality, summarize_quality


DEFAULT_PRICING = dict(enabled=False, short_end='flat_first_local_variance',
    wings='flat_boundary_variance', knot_policy='right_interval',
    forward_policy='log_linear_reference_nodes',
    log_grid_bounds=[-3., 2.], wider_log_grid_bounds=[-4., 3.],
    space_points=801, steps_per_year=365, refinement=2,
    wing_transition_width=.2, interior_time_fraction=1e-7,
    absolute_price_forward_fraction=1e-5, relative_price_tolerance=.001,
    numerical_budget_fraction=.25, maximum_iv_error=.0005, wing_tail_fraction=.5, wing_tail_fraction_alternative=.65,
    short_end_power_alternative=1.2, boundary_audit_points=14001,
    first_wing_tail_weight=0., time_smoothing_fraction=0., time_smoothing_quadrature=12, time_smoothing_time_points=4)


def pricing_config(raw):
    """Resolve all assumptions into the fingerprint, validating JSON types."""
    if raw is None:return deepcopy(DEFAULT_PRICING)
    if not isinstance(raw,dict) or set(raw)-set(DEFAULT_PRICING):
        raise ValueError('invalid_pricing_config')
    cfg={**deepcopy(DEFAULT_PRICING),**raw}
    if type(cfg['enabled']) is not bool:raise ValueError('invalid_pricing_enabled')
    new_policy = cfg['short_end']=='first_smile_total_variance' and cfg['wings']=='calendar_ordered_power_prices'
    if not new_policy and (cfg['short_end']!=DEFAULT_PRICING['short_end'] or cfg['wings']!=DEFAULT_PRICING['wings']):
        raise ValueError('unsupported_or_mixed_boundary_policies')
    for name in ('knot_policy','forward_policy'):
        if cfg[name]!=DEFAULT_PRICING[name]:raise ValueError('unsupported_'+name)
    for name in ('space_points','steps_per_year','refinement'):
        if type(cfg[name]) is not int:raise ValueError('invalid_'+name)
    # A wider domain needs more points to retain spatial resolution; the
    # finite cap still bounds memory/work for arbitrary configured surfaces.
    if not 101<=cfg['space_points']<=8001 or cfg['space_points']%2!=1:
        raise ValueError('invalid_space_points')
    if not 12<=cfg['steps_per_year']<=4000 or not 2<=cfg['refinement']<=4:
        raise ValueError('invalid_pricing_resolution')
    for name in ('log_grid_bounds','wider_log_grid_bounds'):
        a=cfg[name]
        if not isinstance(a,list) or len(a)!=2 or any(type(v) not in (float,int) or not math.isfinite(v) for v in a):
            raise ValueError('invalid_'+name)
        if not -12<=a[0]<0<a[1]<=8:raise ValueError('invalid_'+name)
    a,b=cfg['log_grid_bounds'];lo,hi=cfg['wider_log_grid_bounds']
    if not lo<a<b<hi:raise ValueError('wider_grid_must_enclose_base')
    for name in ('wing_transition_width','interior_time_fraction','absolute_price_forward_fraction',
                 'relative_price_tolerance','numerical_budget_fraction','maximum_iv_error'):
        v=cfg[name]
        if type(v) not in (float,int) or not math.isfinite(v) or v<=0:raise ValueError('invalid_'+name)
    if cfg['interior_time_fraction']>=.01 or cfg['numerical_budget_fraction']>1:
        raise ValueError('invalid_pricing_fraction')
    for key in ('wing_tail_fraction','wing_tail_fraction_alternative'):
        if type(cfg[key]) not in (int,float) or not 0<cfg[key]<1:raise ValueError('invalid_'+key)
    if cfg['wing_tail_fraction']==cfg['wing_tail_fraction_alternative']:
        raise ValueError('wing_sensitivity_requires_different_tail_fraction')
    power=cfg['short_end_power_alternative']
    if type(power) not in (int,float) or not 1<power<=2:raise ValueError('invalid_short_end_power_alternative')
    if type(cfg['boundary_audit_points']) is not int or not 1001<=cfg['boundary_audit_points']<=100001:
        raise ValueError('invalid_boundary_audit_points')
    for name, upper in (('first_wing_tail_weight',1.),('time_smoothing_fraction',.1)):
        value=cfg[name]
        if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<upper:
            raise ValueError('invalid_'+name)
    if type(cfg['time_smoothing_quadrature']) is not int or not 4<=cfg['time_smoothing_quadrature']<=32:
        raise ValueError('invalid_time_smoothing_quadrature')
    if type(cfg['time_smoothing_time_points']) is not int or not 2<=cfg['time_smoothing_time_points']<=32:
        raise ValueError('invalid_time_smoothing_time_points')
    if not new_policy and (cfg['first_wing_tail_weight'] or cfg['time_smoothing_fraction']):
        raise ValueError('completion_stabilization_requires_calibrated_prices')
    return cfg


class BoundaryLocalVol:
    """A labelled extension of a validated interior, NOT new market support.

    Time is right-continuous at knots (the terminal knot uses its left limit).
    A tiny relative inset avoids evaluating undefined exact-knot derivatives.
    Missing intermediate strips cannot be bridged. Wings freeze VARIANCE at
    the nearest supported level; the alternative fades to the ATM variance.
    The first month freezes the first interval's local-variance profile. The
    sensitivity alternative uses the first implied-variance/time profile.
    """
    def __init__(self,evaluator,cfg,variant='baseline'):
        self.ev=evaluator;self.cfg=cfg;self.variant=variant
        if variant not in ('baseline','wing_alternative','short_end_alternative'):
            raise ValueError('unknown_boundary_variant')
        self.domains=sorted(evaluator.permission['domain'],key=lambda d:d['start_years'])
        if not self.domains:raise ValueError('empty_pricing_domain')
        for i,d in enumerate(self.domains):
            if not d['k_min']<0<d['k_max']:raise ValueError('pricing_domain_missing_forward')
            if i and not math.isclose(self.domains[i-1]['end_years'],d['start_years'],abs_tol=1e-12,rel_tol=0):
                raise ValueError('pricing_domain_has_time_gap')
        self.times=np.array([self.domains[0]['start_years']]+[d['end_years'] for d in self.domains])

    def variance(self,t,k):
        if not np.isfinite(t) or t<0 or t>self.times[-1]:raise ValueError('pricing_time_outside_horizon')
        k=np.asarray(k,float)
        if not np.isfinite(k).all():raise ValueError('nonfinite_pricing_state')
        j=int(np.clip(np.searchsorted(self.times,t,side='right')-1,0,len(self.domains)-1))
        d=self.domains[j];a,b=d['start_years'],d['end_years']
        inset=(b-a)*self.cfg['interior_time_fraction']
        query=float(np.clip(t,a+inset,b-inset))
        # Clamping the query coefficient does NOT clamp an underlying path.
        bounded=np.clip(k,d['k_min'],d['k_max'])
        unique,inverse=np.unique(bounded,return_inverse=True)
        values=self.ev.model.evaluate(unique,query)
        w,wk,wkk,wt=(np.asarray(values[n],float) for n in ('w','w_k','w_kk','w_t'))
        g=(1-unique*wk/(2*w))**2-.25*wk*wk*(1/w+.25)+.5*wkk
        if (not all(np.isfinite(v).all() for v in (w,wk,wkk,wt,g)) or np.any(w<=0)
                or np.any(g<self.ev.minimum_density) or np.any(wt < -self.ev.negative_tolerance)):
            raise ValueError('invalid_pricing_coefficient_inside_supported_strip')
        variance=np.maximum(wt,0)/g
        if self.variant=='short_end_alternative' and t<self.times[0]:
            # Alternative assumption only; do not use it to fix failed prices.
            first=self.ev.model.evaluate(unique,float(self.times[0]))['w']
            variance=np.asarray(first,float)/self.times[0]
        result=variance[inverse].reshape(k.shape)
        if self.variant=='wing_alternative':
            centre=self.ev.model.evaluate(0.,query)
            wc,wkc,wkkc,wtc=(float(centre[n]) for n in ('w','w_k','w_kk','w_t'))
            gc=1-.25*wkc*wkc*(1/wc+.25)+.5*wkkc
            if not math.isfinite(gc) or gc<self.ev.minimum_density or wtc < -self.ev.negative_tolerance:
                raise ValueError('invalid_atm_boundary_reference')
            atm=max(wtc,0)/gc
            distance=np.abs(k-bounded)
            blend=1-np.exp(-distance/self.cfg['wing_transition_width'])
            result=(1-blend)*result+blend*atm
        if not np.isfinite(result).all() or np.any(result<0):raise ValueError('invalid_extended_variance')
        return result


def pricing_boundary(evaluator,cfg,variant='baseline'):
    """Keep legacy experiments reproducible; select the configured completion."""
    if cfg['short_end']=='first_smile_total_variance':
        from .local_vol_boundary import CalibratedBoundaryLocalVol
        return CalibratedBoundaryLocalVol(evaluator,cfg,variant)
    return BoundaryLocalVol(evaluator,cfg,variant)


class ReferenceCurves:
    """Deterministic log-linear F,D with F(0)=spot and D(0)=1.

    Nodes are the upstream reference-date values, not medians of asynchronous
    option observations. No OIS quote is bootstrapped again here. Negative
    rates are permitted: discount factors need only be positive, not <=1.
    """
    def __init__(self,rows,times,valuation_date):
        rows=sorted(rows,key=lambda r:r['years'])
        if len(rows)!=len(times):raise ValueError('missing_reference_nodes')
        self.rows=rows
        t=np.array([float(r['years']) for r in rows])
        f=np.array([float(r['forward']) for r in rows]);d=np.array([float(r['discount_factor']) for r in rows])
        spots=np.array([float(r['spot']) for r in rows])
        if not np.allclose(t,times,rtol=0,atol=1e-10) or not np.all(np.diff(t)>0):
            raise ValueError('reference_maturity_mismatch')
        if not all(np.isfinite(a).all() and np.all(a>0) for a in (t,f,d,spots)):
            raise ValueError('invalid_reference_curve_nodes')
        if not np.allclose(spots,spots[0],rtol=1e-10,atol=0):raise ValueError('inconsistent_reference_spot')
        stamps={r['reference_timestamp'] for r in rows}
        if len(stamps)!=1 or any(r['curve_date']!=valuation_date for r in rows):
            raise ValueError('reference_curve_date_mismatch')
        self.times=np.r_[0.,t];self.logf=np.log(np.r_[spots[0],f]);self.logd=np.log(np.r_[1.,d])

    def at(self,t):
        if not np.isfinite(t) or not 0<=t<=self.times[-1]:raise ValueError('reference_curve_extrapolation_forbidden')
        return float(np.exp(np.interp(t,self.times,self.logf))),float(np.exp(np.interp(t,self.times,self.logd)))


def solve_calls(boundary,times,*,bounds,points,steps_per_year):
    """Forward strike PDE: one deterministic solve prices every expiry/strike.

    Calibrated completions use a harmonic cell-averaged, exponentially fitted
    log-strike stencil to resolve wing density jumps. Legacy policies keep their
    central differences. Both use Crank-Nicolson in time with two implicit
    half steps at the start of EACH expiry interval (Rannacher damping).
    Every maturity is an exact endpoint; no step crosses a coefficient knot.
    Lower/upper computational boundaries use 1-exp(k) / 0. They are numerical
    truncations, not market support, and are checked again on a wider grid.
    """
    times=np.asarray(times,float)
    if len(times)==0 or times[0]<=0 or not np.all(np.diff(times)>0):raise ValueError('invalid_pricing_maturities')
    grid=np.linspace(*bounds,points);h=grid[1]-grid[0]
    if not bounds[0]<0<bounds[1] or h>=1:raise ValueError('invalid_pde_mesh')
    values=np.maximum(1-np.exp(grid),0.)
    low=float(1-math.exp(bounds[0]));outputs={};start=0.
    def advance(values,t0,t1,theta):
        dt=t1-t0
        if hasattr(boundary,'cell_variance'):
            v=boundary.cell_variance((t0+t1)/2,grid[1:-1],h)
            upper=.5*v/(h*np.expm1(h));lower=upper*np.exp(h);diag=-lower-upper
        else:
            v=boundary.variance((t0+t1)/2,grid[1:-1])
            lower=.5*v*(1/h**2+1/(2*h));diag=-v/h**2;upper=.5*v*(1/h**2-1/(2*h))
        rhs=values[1:-1]+(1-theta)*dt*(lower*values[:-2]+diag*values[1:-1]+upper*values[2:])
        rhs[0]+=theta*dt*lower[0]*low
        # Right boundary is zero. scipy's band layout stores upper at [0,1:].
        band=np.zeros((3,len(rhs)));band[1]=1-theta*dt*diag
        band[0,1:]=-theta*dt*upper[:-1];band[2,:-1]=-theta*dt*lower[1:]
        out=np.r_[low,solve_banded((1,1),band,rhs,check_finite=False),0.]
        if not np.isfinite(out).all():raise ValueError('nonfinite_pde_solution')
        return out
    for end in times:
        n=max(2,int(math.ceil((end-start)*steps_per_year)))
        nodes=boundary.time_nodes(start,float(end),steps_per_year) if hasattr(boundary,'time_nodes') else np.linspace(start,end,n+1)
        for i,(a,b) in enumerate(zip(nodes[:-1],nodes[1:])):
            if i==0:
                mid=(a+b)/2;values=advance(values,a,mid,1.);values=advance(values,mid,b,1.)
            else:values=advance(values,a,b,.5)
        # Preserve raw values: never clip to arbitrage bounds to manufacture a pass.
        outputs[float(end)]=values.copy();start=float(end)
    return grid,outputs


def implied_vol(c,k,t):
    """Invert normalized call; invalid PDE prices remain unavailable."""
    intrinsic=max(1-math.exp(k),0.)
    if not intrinsic<c<1 or t<=0:return None
    lo,hi=0.,1.
    while float(black_prices(1.,math.exp(k),t,1.,hi,True))<c and hi<64:hi*=2
    if float(black_prices(1.,math.exp(k),t,1.,hi,True))<c:return None
    for _ in range(60):
        mid=(lo+hi)/2
        if float(black_prices(1.,math.exp(k),t,1.,mid,True))<c:lo=mid
        else:hi=mid
    return (lo+hi)/2


def validate_repricing(surface,evaluators,cfg,progress=None):
    """Return explicit assessments, reference nodes and point-level evidence.

    Reprice only quotes in adjacent supported strips. Exclusions remain rows.
    Reference-price comparison is at ONE timestamp. For comparison to original
    live/historical quotes, transport the recovered IV using the SAME stable-IV
    approximation as upstream; never label that as synchronous market evidence.
    """
    tables=dict(pricing_summary=[],pricing_expiries=[],option_repricing=[],pricing_reference_nodes=[],boundary_calibration=[],boundary_audit=[])
    quality=surface.manifest['identity']['config'].get('quality',{})
    for cohort in surface.tables['cohort_status']:
        gid=cohort['group_id'];context={k:cohort.get(k) for k in ('group_id','underlying','currency','valuation_date','route')}
        summary={**context,'status':'skipped','reason':None,'full_simulation_ready':False}
        if not cfg['enabled']:summary['reason']='pricing_disabled'
        elif gid not in evaluators:summary['reason']='validated_local_volatility_unavailable'
        else:
            try:
                ev=evaluators[gid];boundary=pricing_boundary(ev,cfg)
                refs=[r for r in surface.tables.get('reference_inputs',[]) if r['group_id']==gid]
                curves=ReferenceCurves(refs,boundary.times,cohort['valuation_date'])
                quotes=[r for r in surface.tables.get('repricing',[]) if r['group_id']==gid]
                if not quotes:raise ValueError('no_repricing_quotes')
                selection={r['record_id']:r for r in surface.tables.get('selection',[]) if r['group_id']==gid and r.get('selected')}
                if not quality:raise ValueError('missing_upstream_quality_policy')
                for d in boundary.domains:
                    if not cfg['log_grid_bounds'][0]<d['k_min']<d['k_max']<cfg['log_grid_bounds'][1]:
                        raise ValueError('computational_grid_does_not_enclose_supported_domain')
                runs={};factor=cfg['refinement'];fine_n=(cfg['space_points']-1)*factor+1
                h=(cfg['log_grid_bounds'][1]-cfg['log_grid_bounds'][0])/(fine_n-1)
                wide_n=int(math.ceil((cfg['wider_log_grid_bounds'][1]-cfg['wider_log_grid_bounds'][0])/h))+1
                for name,variant,bounds,n,steps in (
                    ('coarse','baseline',cfg['log_grid_bounds'],cfg['space_points'],cfg['steps_per_year']),
                    ('fine','baseline',cfg['log_grid_bounds'],fine_n,cfg['steps_per_year']*factor),
                    ('wider','baseline',cfg['wider_log_grid_bounds'],wide_n,cfg['steps_per_year']*factor),
                    ('wing_alternative','wing_alternative',cfg['log_grid_bounds'],fine_n,cfg['steps_per_year']*factor),
                    ('short_end_alternative','short_end_alternative',cfg['log_grid_bounds'],fine_n,cfg['steps_per_year']*factor)):
                    if progress:progress(f"Local-volatility PDE {cohort.get('underlying')}: {name}")
                    completion=pricing_boundary(ev,cfg,variant)
                    if name in ('fine','wing_alternative','short_end_alternative') and hasattr(completion,'audit'):
                        tables['boundary_calibration'].extend({**context,**r} for r in completion.records)
                        tables['boundary_audit'].append({**context,'variant':variant,**completion.audit})
                    runs[name]=solve_calls(completion,boundary.times,
                        bounds=bounds,points=n,steps_per_year=steps)
                tables['pricing_reference_nodes'].extend({**context,**r} for r in refs)
                rows=[];by_expiry={}
                for q in quotes:
                    t=float(q['years']);k=float(q['k']);record={**context,'record_id':q['record_id'],
                        'expiration':q['expiration'],'years':t,'k':k,'original_strike':q['strike'],'status':'excluded','reason':None}
                    adjacent=[d for d in boundary.domains if math.isclose(t,d['start_years'],abs_tol=1e-10) or math.isclose(t,d['end_years'],abs_tol=1e-10)]
                    if not adjacent or not max(d['k_min'] for d in adjacent)<=k<=min(d['k_max'] for d in adjacent):
                        record['reason']='outside_supported_expiry_strip';rows.append(record);continue
                    if q['record_id'] not in selection:raise ValueError('repricing_quote_not_selected')
                    node=float(boundary.times[np.argmin(abs(boundary.times-t))])
                    if abs(node-t)>1e-10:raise ValueError('quote_maturity_not_reference_node')
                    f,d=curves.at(node);x=math.exp(k);is_call=q['option_type']=='call'
                    if q['option_type'] not in ('call','put'):raise ValueError('unsupported_repricing_right')
                    fitted_iv=math.sqrt(float(ev.model.evaluate(k,node)['w'])/node)
                    target=float(black_prices(f,f*x,node,d,fitted_iv,is_call))
                    values={name:float(np.interp(k,mesh,prices[node])) for name,(mesh,prices) in runs.items()}
                    prices={name:d*f*(c if is_call else c-1+x) for name,c in values.items()}
                    tol=cfg['absolute_price_forward_fraction']*d*f+cfg['relative_price_tolerance']*abs(target)
                    iv=implied_vol(values['fine'],k,node)
                    record.update(status='assessed',reference_strike=f*x,reference_forward=f,reference_discount=d,
                        target_price=target,pde_price=prices['fine'],fitted_iv=fitted_iv,pde_iv=iv,
                        tolerance=tol,price_error=prices['fine']-target,price_fit_score=abs(prices['fine']-target)/tol,
                        convergence_score=abs(prices['fine']-prices['coarse'])/(tol*cfg['numerical_budget_fraction']),
                        grid_boundary_score=abs(prices['fine']-prices['wider'])/(tol*cfg['numerical_budget_fraction']),
                        wing_sensitivity_score=abs(prices['fine']-prices['wing_alternative'])/tol,
                        short_end_sensitivity_score=abs(prices['fine']-prices['short_end_alternative'])/tol,
                        iv_error=None if iv is None else iv-fitted_iv,
                        iv_fit_pass=bool(iv is not None and abs(iv-fitted_iv)<=cfg['maximum_iv_error']),
                        wing_alternative_price=prices['wing_alternative'],short_end_alternative_price=prices['short_end_alternative'])
                    # Market comparison transports IV back to original F,D,tau;
                    # do NOT compare reference-currency prices to asynchronous bids.
                    point=dict(route=cohort['route'],option_price=q['observed_price'],iv=q['observed_iv'],
                        bid=q.get('bid'),ask=q.get('ask'),bid_iv=q.get('bid_iv'),ask_iv=q.get('ask_iv'))
                    if iv is not None:
                        market_price=float(black_prices(q['original_forward'],q['strike'],q['original_years'],q['original_discount_factor'],iv,is_call))
                        record.update(market_comparison_basis='stable_IV_transport_to_original_F_D_tau',
                            market_price=market_price,market_scaled_error=(market_price-q['observed_price'])/selection[q['record_id']]['error_scale'],
                            **quote_quality(point,market_price,iv,quality))
                    else:
                        record.update(practical_pass=None,inside_spread=None,practical_status='unassessed',market_scaled_error=None)
                    rows.append(record);by_expiry.setdefault(q['expiration'],[]).append((point,record))
                assessed=[r for r in rows if r['status']=='assessed']
                if not assessed:raise ValueError('no_supported_repricing_quotes')
                for expiry,pairs in by_expiry.items():
                    points,items=zip(*pairs)
                    metric={**context,'expiration':expiry,'observations':len(items)}
                    scores=('price_fit_score','convergence_score','grid_boundary_score','wing_sensitivity_score','short_end_sensitivity_score')
                    for key in scores:metric['maximum_'+key]=max(r[key] for r in items)
                    metric['maximum_absolute_price_error']=max(abs(r['price_error']) for r in items)
                    metric['maximum_absolute_iv_error_vol_points']=100*max(abs(r['iv_error']) for r in items) if all(r['iv_error'] is not None for r in items) else None
                    metric['iv_fit_pass']=all(r['iv_fit_pass'] for r in items)
                    metric['numerical_pass']=all(metric['maximum_'+k]<=1 for k in ('convergence_score','grid_boundary_score'))
                    metric['surface_reproduction_pass']=metric['maximum_price_fit_score']<=1 and metric['iv_fit_pass']
                    metric['boundary_sensitivity_pass']=all(metric['maximum_'+k]<=1 for k in ('wing_sensitivity_score','short_end_sensitivity_score'))
                    errors=[r['market_scaled_error'] for r in items]
                    metric['market_scaled_rmse']=float(np.sqrt(np.mean(np.square(errors)))) if all(v is not None for v in errors) else None
                    # summarize_quality expects the diagnostic fields even when inversion failed.
                    for r in items:
                        for key in ('relative_observed_price_error','relative_outside_spread_error','outside_half_spreads','absolute_iv_error','outside_iv_error'):
                            r.setdefault(key,None)
                    metric.update(summarize_quality(points,items,quality))
                    metric['market_fit_pass']=bool(metric['market_scaled_rmse'] is not None and metric['market_scaled_rmse']<=quality['maximum_scaled_price_rmse'] and
                        (cohort['route']!='live' or metric['practical_coverage_pass']))
                    tables['pricing_expiries'].append(metric)
                tables['option_repricing'].extend(rows)
                metrics=[r for r in tables['pricing_expiries'] if r['group_id']==gid]
                expected_expiries={r['expiration'] for r in refs}
                complete={r['expiration'] for r in metrics}==expected_expiries
                flags=('numerical_pass','surface_reproduction_pass','boundary_sensitivity_pass','market_fit_pass')
                summary.update(status='validated_with_boundary_assumptions' if complete and all(all(r[k] for k in flags) for r in metrics) else 'review_required',
                    assessed_quotes=len(assessed),excluded_quotes=len(rows)-len(assessed),expiries=len(metrics),expected_expiries=len(refs),
                    all_expiries_assessed=complete,**{k:complete and all(r[k] for r in metrics) for k in flags})
                summary['reason']=None if summary['status']=='validated_with_boundary_assumptions' else 'pricing_or_boundary_checks_failed'
            except (ValueError,KeyError,OverflowError,FloatingPointError) as exc:
                summary.update(status='unavailable',reason=str(exc))
        tables['pricing_summary'].append(summary)
    return tables
