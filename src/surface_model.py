"""Surface models and fitting: normalized Black prices, SSVI and raw SVI.

This module has no archive/provider dependencies. SVI nodes retain their
bounded numerical arbitrage audits; no local volatility is calculated.
"""
from dataclasses import dataclass, field
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares, minimize, minimize_scalar
from scipy.special import ndtr

# SSVI and shared pricing / numerical audits
def eta_limit(rho, gamma):
    """Global sufficient eta limit for our damped-power-law SSVI family.

    theta*phi <= eta. Also theta*phi**2 = eta**2*h(theta), where
    h(theta)=theta**a/(1+theta)**(1+a), a=1-2*gamma in [0,1].
    Differentiation gives max h=a**a/(1+a)**(1+a), attained at theta=a;
    for a=0 the supremum is one as theta approaches zero.
    Combining both sufficient conditions permits the full safe parameter
    range. A fit stays strictly inside it via the fraction parameter <1.
    """
    if not np.isfinite([rho,gamma]).all() or not abs(rho)<1 or not 0<=gamma<=.5:
        raise ValueError('invalid_ssvi_shape_parameters')
    a=1-2*gamma
    maximum=1. if a==0 else a**a/(1+a)**(1+a)
    return min(4/(1+abs(rho)),2/np.sqrt((1+abs(rho))*maximum))


def ssvi(k, theta, rho, eta, gamma):
    """Return w, dw/dk, d2w/dk2, dw/dtheta; vectorized, analytic derivatives."""
    k, theta = np.broadcast_arrays(np.asarray(k, float), np.asarray(theta, float))
    if (not np.all(np.isfinite(k)) or not np.all(np.isfinite(theta))
            or np.any(theta <= 0) or not np.isfinite([rho, eta, gamma]).all()
            or not abs(rho) < 1 or not 0 <= gamma <= .5
            or not 0 <= eta < eta_limit(rho,gamma)):
        raise ValueError('invalid_ssvi_parameters')
    phi = eta / (theta**gamma * (1+theta)**(1-gamma))
    u = phi*k + rho
    root = np.sqrt(u*u + 1-rho*rho)
    shape = 1 + rho*phi*k + root
    w = .5*theta*shape
    wk = .5*theta*phi*(rho+u/root)
    wkk = .5*theta*phi*phi*(1-rho*rho)/root**3
    phi_derivative = phi*(-gamma/theta-(1-gamma)/(1+theta))
    wtheta = .5*shape + .5*theta*phi_derivative*k*(rho+u/root)
    return w, wk, wkk, wtheta


def black_prices(forward, strike, years, discount, iv, is_call):
    """Vector equivalent of option_math.black_price; use OTM side + parity."""
    f, k, t, d, v = np.broadcast_arrays(forward, strike, years, discount, iv)
    if not all(np.all(np.isfinite(x)) and np.all(x > 0) for x in (f,k,t,d,v)):
        raise ValueError('invalid_vector_black_inputs')
    sd = v*np.sqrt(t)
    d1 = np.log(f/k)/sd + .5*sd
    d2 = d1-sd
    put_otm = d*(k*ndtr(-d2)-f*ndtr(-d1))
    call_otm = d*(f*ndtr(d1)-k*ndtr(d2))
    calls = np.where(f >= k, put_otm+d*(f-k), call_otm)
    puts = np.where(f >= k, put_otm, call_otm+d*(k-f))
    return np.where(is_call, calls, puts)


@dataclass
class SurfaceModel:
    """Evaluation is restricted to the fitted maturity and moneyness envelope.

    The parametric wings exist mathematically but are NOT data-supported
    extrapolations. validate_grid uses the same restricted public evaluator.
    """
    times: list
    theta: list
    rho: float
    eta: float
    gamma: float
    k_min: list
    k_max: list

    def __post_init__(self):
        arrays = [np.asarray(x, float) for x in (self.times,self.theta,self.k_min,self.k_max)]
        t, v, lo, hi = arrays
        if (len(t)<2 or any(a.shape != t.shape or not np.isfinite(a).all() for a in arrays)
                or np.any(t<=0) or np.any(np.diff(t)<=0) or np.any(v<=0)
                or np.any(np.diff(v)<0) or np.any(lo>=hi)):
            raise ValueError('invalid_surface_nodes')
        ssvi(0.,v,self.rho,self.eta,self.gamma)  # validates shape constraints too
        self._atm = PchipInterpolator(t,v,extrapolate=False)

    @property
    def model_type(self):
        return 'ssvi'

    def domain(self, years):
        t=np.asarray(years,float)
        if not np.isfinite(t).all() or np.any(t<self.times[0]) or np.any(t>self.times[-1]):
            raise ValueError('surface_maturity_outside_supported_domain')
        return np.interp(t,self.times,self.k_min),np.interp(t,self.times,self.k_max)

    def interval_domain(self, start, end):
        """Conservative support at EVERY maturity in this interval.

        SSVI support boundaries are piecewise linear. Their extrema occur at
        the endpoints or intervening nodes, so no time-grid approximation is
        needed. Including limiting endpoint values is safe for open intervals.
        """
        if not np.isfinite([start,end]).all() or not start<end:
            raise ValueError('invalid_surface_interval')
        nodes=np.r_[start,[t for t in self.times if start<t<end],end]
        lo,hi=self.domain(nodes)
        return float(np.max(lo)),float(np.min(hi))

    def evaluate(self, k, years):
        k,t = np.broadcast_arrays(np.asarray(k,float),np.asarray(years,float))
        if (not np.isfinite(k).all() or not np.isfinite(t).all()
                or np.any(t<self.times[0]) or np.any(t>self.times[-1])):
            raise ValueError('surface_maturity_outside_supported_domain')
        lo=np.interp(t,self.times,self.k_min); hi=np.interp(t,self.times,self.k_max)
        if np.any(k<lo) or np.any(k>hi):
            raise ValueError('surface_moneyness_outside_supported_domain')
        w,wk,wkk,wtheta=ssvi(k,self._atm(t),self.rho,self.eta,self.gamma)
        wt=wtheta*self._atm.derivative()(t)
        # g is the nonnegative-density diagnostic, NOT a local-volatility value.
        g=(1-k*wk/(2*w))**2-.25*wk*wk*(1/w+.25)+.5*wkk
        return dict(w=w,iv=np.sqrt(w/t),w_k=wk,w_kk=wkk,w_t=wt,density_factor=g)


def _decode_ssvi(x, count):
    # Strictly positive increments avoid calendar crossings during optimization.
    # Log bounds below are numerical search bounds; reaching one is reported.
    theta=np.cumsum(np.exp(x[:count]))
    rho,fraction,gamma=x[count:]
    eta=eta_limit(rho,gamma)*fraction
    return theta,rho,eta,gamma


def _arrays(points):
    expiries=sorted({p['expiration'] for p in points})
    ids={e:i for i,e in enumerate(expiries)}
    index=np.array([ids[p['expiration']] for p in points])
    times=np.array([next(p['reference_years'] for p in points if p['expiration']==e) for e in expiries])
    a={key:np.array([p[key] for p in points],float) for key in
       ('k','iv','forward','strike','time_to_expiry','discount_factor','option_price','error_scale')}
    a['is_call']=np.array([p['option_type']=='call' for p in points])
    return expiries,index,times,a


def fitted_prices(model, points):
    """Compare against ORIGINAL F,D,tau; no fabricated simultaneous premium.

    A pooled fit holds the IV smile at original forward moneyness constant
    over the capture interval. reference_years is the common reference horizon;
    original time_to_expiry remains the Black pricing horizon for each quote.
    """
    _,_,_,a=_arrays(points)
    iv=model.evaluate(a['k'],np.array([p['reference_years'] for p in points]))['iv']
    return black_prices(a['forward'],a['strike'],a['time_to_expiry'],a['discount_factor'],iv,a['is_call']),iv


def fit_ssvi(points, policy, starts=None):
    """Joint robust price fit, balancing observations so each expiry has weight.

    Initial theta comes from near-forward IVs. All starts are deterministic;
    only converged finite solutions can be selected. No arbitrary retry method
    or unconstrained fallback is substituted if optimization fails.
    """
    expiries,index,times,a=_arrays(points)
    n=len(expiries)
    if n<2 or np.any(np.diff(times)<=0):
        raise ValueError('insufficient_or_inconsistent_surface_maturities')
    target=[]
    for j in range(n):
        mask=index==j
        nearest=np.argsort(np.abs(a['k'][mask]))[:max(3,int(mask.sum()*.15))]
        target.append(float(np.median(a['iv'][mask][nearest]**2)*times[j]))
    theta=np.maximum.accumulate(target)
    initial=np.log(np.maximum(np.diff(np.r_[0.,theta]),1e-6))
    counts=np.bincount(index,minlength=n)
    balance=np.sqrt(len(points)/(n*counts[index]))
    lower=np.r_[np.full(n,-25.),-.95,0.,0.]
    upper=np.r_[np.full(n,5.),.95,.999,.5]

    def residual(x):
        th,rho,eta,gamma=_decode_ssvi(x,n)
        w=ssvi(a['k'],th[index],rho,eta,gamma)[0]
        iv=np.sqrt(w/times[index])
        prices=black_prices(a['forward'],a['strike'],a['time_to_expiry'],a['discount_factor'],iv,a['is_call'])
        return (prices-a['option_price'])/a['error_scale']*balance

    seeds=starts if starts is not None else [np.r_[initial,r,.25,.25] for r in policy['initial_rhos']]
    records=[]; candidates=[]
    for j,seed in enumerate(seeds):
        solved=least_squares(residual,np.clip(seed,lower+1e-10,upper-1e-10),bounds=(lower,upper),
            loss='soft_l1',f_scale=1.,max_nfev=policy['maximum_evaluations'],
            ftol=1e-8,xtol=1e-8,gtol=1e-8)
        okay=bool(solved.success and np.isfinite(solved.x).all() and np.isfinite(solved.fun).all())
        records.append(dict(start=j,converged=okay,cost=float(solved.cost),evaluations=int(solved.nfev),
                            search_boundary_reached=bool(np.any(solved.active_mask)),message=str(solved.message)))
        if okay:
            th,rho,eta,gamma=_decode_ssvi(solved.x,n)
            model=SurfaceModel(times.tolist(),th.tolist(),float(rho),float(eta),float(gamma),
                [min(p['k'] for p in points if p['expiration']==e) for e in expiries],
                [max(p['k'] for p in points if p['expiration']==e) for e in expiries])
            candidates.append((float(solved.cost),model,solved.x))
    if not candidates:
        return None,records,[]
    candidates.sort(key=lambda x:x[0])
    return candidates[0][1],records,candidates


def validate_grid(model, grid_policy):
    """Analytical shape constraints plus an independent dense numerical audit."""
    ts=np.unique(np.r_[np.linspace(model.times[0],model.times[-1],grid_policy['maturity_points']),model.times])
    rows=[]; price_audits=[]
    for t in ts:
        lo,hi=getattr(model,'audit_domain',model.domain)(t)
        ks=np.linspace(lo,hi,grid_policy['moneyness_points'])
        values=model.evaluate(ks,t)
        # Independently price normalized calls C/(D*F) against x=K/F.
        # Positive D and F only rescale prices/strikes, so bounds, decreasing
        # prices and increasing strike slopes apply to every reference expiry.
        # This check does not rely on our analytic w derivatives or g formula.
        x=np.exp(ks)
        calls=black_prices(1.,x,t,1.,values['iv'],True)
        slopes=np.diff(calls)/np.diff(x)
        price_audits.append(dict(
            bounds=bool(np.all(calls>=np.maximum(1-x,0)-1e-10) and np.all(calls<=1+1e-10)),
            monotone=bool(np.all(slopes<=1e-10) and np.all(slopes>=-1-1e-10)),
            convex=bool(np.all(np.diff(slopes)>=-1e-10))))
        for i,k in enumerate(ks):
            rows.append(dict(years=float(t),k=float(k),k_min=float(ks[0]),k_max=float(ks[-1]),
                             **{key:float(value[i]) for key,value in values.items()}))
    # Central differences at interior points test the implementation, including
    # time interpolation. Avoid maturity knots and domain edges for this audit.
    errors={'w_k':0.,'w_kk':0.,'w_t':0.}
    for t0,t1 in zip(model.times[:-1],model.times[1:]):
        t=(t0+t1)/2; ht=min(1e-5,(t1-t0)/20)
        lo,hi=model.domain(t)
        for k in np.linspace(lo,hi,7)[1:-1]:
            hk=min(1e-4,(hi-lo)/100)
            v=model.evaluate(k,t)
            left=model.evaluate(k-hk,t)['w']; right=model.evaluate(k+hk,t)['w']
            estimates={'w_k':(right-left)/(2*hk),'w_kk':(right-2*v['w']+left)/(hk*hk),
                       'w_t':(model.evaluate(k,t+ht)['w']-model.evaluate(k,t-ht)['w'])/(2*ht)}
            for name,value in estimates.items():
                errors[name]=max(errors[name],float(abs(value-v[name])/(1+abs(v[name]))))
    minimum={key:min(r[key] for r in rows) for key in ('w','w_t','density_factor')}
    price_checks={key:all(r[key] for r in price_audits) for key in ('bounds','monotone','convex')}
    tolerance=getattr(model,'arbitrage_tolerance',1e-10)
    node_audit=model.audit() if hasattr(model,'audit') else {'passed':True,'scope':'analytical_ssvi_conditions'}
    okay=(minimum['w']>0 and minimum['w_t']>=-tolerance and minimum['density_factor']>=-tolerance
          and all(price_checks.values()) and node_audit['passed'])
    return rows,dict(analytical_constraints_enforced=model.model_type=='ssvi',node_audit=node_audit,
        numerical_arbitrage_pass=bool(okay),
        minimum_values=minimum,normalized_call_checks=price_checks,derivative_scaled_errors=errors,
        derivative_check_pass=max(errors.values())<=grid_policy['derivative_tolerance'])


# Raw SVI: independent constrained smiles and normalized-call interpolation
def raw_svi(k, parameters):
    """One expiry: w(k)=a+b*[rho*(k-m)+sqrt((k-m)^2+sigma^2)].

    k=log(K/F) is dimensionless forward moneyness; w=IV^2*T is total variance.
    a controls level, b the wing slope scale, rho asymmetry, m displacement,
    sigma curvature width. This sigma is an SVI shape parameter, NOT IV or
    simulated local volatility. Return w, dw/dk and d2w/dk2 analytically.
    """
    a,b,rho,m,sigma = parameters
    k=np.asarray(k,float)
    u=k-m; root=np.sqrt(u*u+sigma*sigma)
    return a+b*(rho*u+root), b*(rho+u/root), b*sigma*sigma/root**3


def black_partials(k,w):
    """Normalized Black call and partial derivatives at fixed total variance.

    These are *partial* k,w derivatives. A smile's total strike derivatives
    additionally include w_k and w_kk through the chain rule below.
    """
    k,w=np.broadcast_arrays(np.asarray(k,float),np.asarray(w,float))
    sd=np.sqrt(w);x=np.exp(k);d1=-k/sd+.5*sd;d2=d1-sd
    normal1=np.exp(-.5*d1*d1)/np.sqrt(2*np.pi)
    normal2=np.exp(-.5*d2*d2)/np.sqrt(2*np.pi)
    c=black_prices(1.,x,1.,1.,sd,True)
    cw=normal1/(2*sd)
    ck=-x*ndtr(d2)
    ckk=ck+x*normal2/sd
    ckw=-x*normal2*(k/(2*w*sd)-1/(4*sd))
    cww=cw*(-d1*(k/(2*w*sd)+1/(4*sd))-1/(2*w))
    return c,ck,cw,ckk,ckw,cww


def slice_values(k,parameters):
    """Combine raw-SVI and Black derivatives; g diagnoses butterfly density."""
    w,wk,wkk=raw_svi(k,parameters)
    c,pk,pw,pkk,pkw,pww=black_partials(k,w)
    ck=pk+pw*wk
    ckk=pkk+2*pkw*wk+pww*wk*wk+pw*wkk
    g=(1-np.asarray(k)*wk/(2*w))**2-.25*wk*wk*(1/w+.25)+.5*wkk
    return w,wk,wkk,c,ck,ckk,g


def _search_grid(lo,hi,parameters,count):
    """Uniform points plus points resolving even very narrow SVI curvature."""
    _,_,_,m,sigma=parameters
    feature=m+sigma*np.r_[-np.geomspace(.01,20,35),0.,np.geomspace(.01,20,35)]
    return np.unique(np.r_[np.linspace(lo,hi,count),np.clip(feature,lo,hi)])


def _minimum(function,grid):
    """Audit sampled minima and bounded refinements between neighboring points.

    This improves on a coarse grid but remains a numerical test, explicitly
    reported as such. An optimizer failure in the audit is not a pass.
    """
    values=np.asarray(function(grid),float)
    if not np.isfinite(values).all(): return -np.inf,float(grid[0])
    j=int(np.argmin(values)); best=float(values[j]); where=float(grid[j])
    indices=np.flatnonzero((values[1:-1]<=values[:-2])&(values[1:-1]<=values[2:]))+1
    # Constant functions need no hundreds of redundant local optimizations.
    if np.ptp(values)>1e-13:
        for i in indices:
            solution=minimize_scalar(function,bounds=(grid[i-1],grid[i+1]),method='bounded',
                                     options={'xatol':1e-12})
            if not solution.success or not np.isfinite(solution.fun):return -np.inf,float(grid[i])
            if solution.fun<best:best=float(solution.fun);where=float(solution.x)
    return best,where


def audit_slice(parameters,lo,hi,previous=None,previous_domain=None,count=1001):
    """Independent dense/locally refined audit; return violations and locations."""
    grid=_search_grid(lo,hi,parameters,count)
    functions={
        'density':lambda k:slice_values(k,parameters)[6],
        'digital_lower':lambda k:1+slice_values(k,parameters)[4]/np.exp(k),
        'digital_upper':lambda k:-slice_values(k,parameters)[4]/np.exp(k)}
    results={name:_minimum(fn,grid) for name,fn in functions.items()}
    # No unsupported strike from the neighboring expiry is used in a calendar
    # test. The public interpolator applies this same intersection policy.
    if previous is not None:
        left=max(lo,previous_domain[0]);right=min(hi,previous_domain[1])
        if left>=right:results['calendar']=(-np.inf,left)
        else:
            calendar_grid=np.unique(np.r_[_search_grid(left,right,parameters,count),
                                           _search_grid(left,right,previous,count)])
            results['calendar']=_minimum(lambda k:raw_svi(k,parameters)[0]-raw_svi(k,previous)[0],calendar_grid)
    return results


@dataclass
class SviSurface:
    """Node smiles plus price-space time interpolation, with no extrapolation."""
    times: list
    parameters: list
    k_min: list
    k_max: list
    arbitrage_tolerance: float=1e-9
    model_type: str='raw_svi'
    start_checks: list=field(default_factory=list)
    parameter_order: tuple=('a','b','rho','m','sigma')
    time_interpolation: str='linear_normalized_calls'

    def audit_domain(self,years):
        """Domain where both strike AND one-sided time derivatives are supported."""
        t=np.asarray(years,float)
        self.domain(t)
        j=np.clip(np.searchsorted(self.times,t,side='right')-1,0,len(self.times)-2)
        return (np.maximum(np.asarray(self.k_min)[j],np.asarray(self.k_min)[j+1]),
                np.minimum(np.asarray(self.k_max)[j],np.asarray(self.k_max)[j+1]))

    def audit(self):
        """Check full node domains too, beyond the smaller interpolation domain."""
        records=[];passed=True
        for i,p in enumerate(self.parameters):
            checks=audit_slice(p,self.k_min[i],self.k_max[i],self.parameters[i-1] if i else None,
                               (self.k_min[i-1],self.k_max[i-1]) if i else None)
            passed=passed and all(np.isfinite(v[0]) and v[0]>=-self.arbitrage_tolerance for v in checks.values())
            records.append(dict(years=self.times[i],**{name:float(v[0]) if np.isfinite(v[0]) else None for name,v in checks.items()}))
        return dict(passed=bool(passed),node_checks=records,scope='numerical_on_observed_domains',
                    time_interpolation='linear_normalized_calls; one_sided_derivatives_at_nodes',
                    global_wing_certificate=False)

    def __post_init__(self):
        if (isinstance(self.arbitrage_tolerance,bool) or not np.isfinite(self.arbitrage_tolerance)
                or not 0<self.arbitrage_tolerance<=1e-7 or self.model_type!='raw_svi'
                or tuple(self.parameter_order)!=('a','b','rho','m','sigma')
                or self.time_interpolation!='linear_normalized_calls'):
            raise ValueError('invalid_raw_svi_validation_policy')
        t=np.asarray(self.times,float);p=np.asarray(self.parameters,float)
        lo=np.asarray(self.k_min,float);hi=np.asarray(self.k_max,float)
        if (t.ndim!=1 or len(t)<2 or p.shape!=(len(t),5) or lo.shape!=t.shape or hi.shape!=t.shape
                or not all(np.isfinite(x).all() for x in (t,p,lo,hi)) or np.any(t<=0)
                or np.any(np.diff(t)<=0) or np.any(lo>=hi)):
            raise ValueError('invalid_raw_svi_nodes')
        a,b,rho,_,sigma=p.T
        if (np.any(b<=0) or np.any(np.abs(rho)>=1) or np.any(sigma<=0)
                or np.any(a+b*sigma*np.sqrt(1-rho*rho)<=0)):
            raise ValueError('invalid_raw_svi_parameters')
        if np.any(np.maximum(lo[:-1],lo[1:])>=np.minimum(hi[:-1],hi[1:])):
            raise ValueError('no_adjacent_slice_domain_overlap')

    @property
    def theta(self):
        return [float(raw_svi(0.,p)[0]) for p in self.parameters]

    def domain(self,years):
        """At nodes use their data range; between nodes require both ranges."""
        t=np.asarray(years,float)
        if not np.isfinite(t).all() or np.any(t<self.times[0]) or np.any(t>self.times[-1]):
            raise ValueError('surface_maturity_outside_supported_domain')
        j=np.clip(np.searchsorted(self.times,t,side='right')-1,0,len(self.times)-2)
        lo=np.maximum(np.asarray(self.k_min)[j],np.asarray(self.k_min)[j+1])
        hi=np.minimum(np.asarray(self.k_max)[j],np.asarray(self.k_max)[j+1])
        for i,node in enumerate(self.times):
            lo=np.where(t==node,self.k_min[i],lo);hi=np.where(t==node,self.k_max[i],hi)
        return lo,hi

    def interval_domain(self,start,end):
        """Intersect all node ranges supporting the OPEN maturity interval.

        Raw SVI uses the intersection of adjacent smiles between nodes. A
        node's wider standalone range must not authorize time derivatives
        beyond that intersection. This also handles intervals spanning nodes.
        """
        if not np.isfinite([start,end]).all() or not start<end:
            raise ValueError('invalid_surface_interval')
        self.domain(np.array([start,end]))  # reject maturity extrapolation
        indices={i for j in range(len(self.times)-1)
                 if self.times[j]<end and self.times[j+1]>start for i in (j,j+1)}
        return max(self.k_min[i] for i in indices),min(self.k_max[i] for i in indices)

    def evaluate(self,k,years):
        """Return IV/variance derivatives, recording one-sided time derivatives.

        At an exact maturity use the observed slice. At other times blend calls
        and invert normalized Black with a vector bisection bracketed by the two
        endpoint variances. This avoids interpolating raw SVI parameters.
        """
        k,t=np.broadcast_arrays(np.asarray(k,float),np.asarray(years,float))
        lo,hi=self.domain(t)
        if not np.isfinite(k).all() or np.any(k<lo) or np.any(k>hi):
            raise ValueError('surface_moneyness_outside_supported_domain')
        shape=k.shape;k=k.ravel();t=t.ravel()
        output={key:np.empty(len(k)) for key in ('w','iv','w_k','w_kk','w_t','density_factor','time_derivative_at_knot')}
        interval=np.clip(np.searchsorted(self.times,t,side='right')-1,0,len(self.times)-2)
        for j in np.unique(interval):
            mask=interval==j;kk=k[mask];tt=t[mask]
            first=slice_values(kk,self.parameters[j]);second=slice_values(kk,self.parameters[j+1])
            alpha=(tt-self.times[j])/(self.times[j+1]-self.times[j])
            # c=C/(D*F) is a normalized call, x=K/F=exp(k). Blend PRICES
            # at fixed k, not SVI parameters or IV. Invert Black below for
            # w and then IV=sqrt(w/t). c_t is constant inside an interval
            # but changes at its boundaries; this explains maturity jumps
            # in w_t and local variance even though prices stay continuous.
            c=(1-alpha)*first[3]+alpha*second[3]
            ck=(1-alpha)*first[4]+alpha*second[4]
            ckk=(1-alpha)*first[5]+alpha*second[5]
            low=np.minimum(first[0],second[0]);high=np.maximum(first[0],second[0])
            for _ in range(55):
                middle=(low+high)/2
                below=black_prices(1.,np.exp(kk),1.,1.,np.sqrt(middle),True)<c
                low=np.where(below,middle,low);high=np.where(below,high,middle)
            w=(low+high)/2
            w=np.where(alpha==0,first[0],np.where(alpha==1,second[0],w))
            _,pk,pw,pkk,pkw,pww=black_partials(kk,w)
            if np.any(pw<1e-14):raise ValueError('ill_conditioned_surface_inversion')
            # Implicitly differentiate c(k,t)=Black(k,w(k,t)). pw is partial
            # Black/partial w; ck is the FULL smile derivative. Solve the
            # chain rule for w_k,w_kk. Tiny pw would amplify noise, which is
            # why the preceding guard rejects an ill-conditioned inversion.
            wk=(ck-pk)/pw
            wkk=(ckk-pkk-2*pkw*wk-pww*wk*wk)/pw
            # At node-only strikes outside the neighboring data range, a time
            # derivative is unsupported. Mark NaN internally; audit/export
            # grids use the common domain at nodes and never publish that NaN.
            common=(kk>=max(self.k_min[j],self.k_min[j+1]))&(kk<=min(self.k_max[j],self.k_max[j+1]))
            wt=np.where(common,(second[3]-first[3])/((self.times[j+1]-self.times[j])*pw),np.nan)
            # Use exact derivatives at nodes to avoid cancellation in tiny Vega.
            wk=np.where(alpha==0,first[1],np.where(alpha==1,second[1],wk))
            wkk=np.where(alpha==0,first[2],np.where(alpha==1,second[2],wkk))
            g=(1-kk*wk/(2*w))**2-.25*wk*wk*(1/w+.25)+.5*wkk
            values=(w,np.sqrt(w/tt),wk,wkk,wt,g,np.isin(tt,self.times).astype(float))
            for name,value in zip(output,values):output[name][mask]=value
        return {key:value.reshape(shape) for key,value in output.items()}


def _decode_svi(x):
    # Fitting minimum variance instead of a guarantees w>0 for every k.
    v,b,rho,m,sigma=np.exp(x[0]),np.exp(x[1]),x[2],x[3],np.exp(x[4])
    return [v-b*sigma*np.sqrt(1-rho*rho),b,rho,m,sigma]


def fit_raw_svi(points,policy):
    """Sequential constrained fit with deterministic starts and rejection gates.

    Each later smile must lie above its accepted predecessor on their common
    k-domain. Failed constraints do not trigger a fallback to an unconstrained
    smile. Return no surface if any required expiry cannot be fitted safely.
    """
    expiries=sorted({p['expiration'] for p in points});params=[];times=[];lows=[];highs=[];records=[];start_checks=[]
    opt=policy['svi'];tol=opt['arbitrage_tolerance']
    for expiry in expiries:
        group=sorted([p for p in points if p['expiration']==expiry],key=lambda p:p['k'])
        k=np.array([p['k'] for p in group]);t=group[0]['reference_years'];lo=float(min(k));hi=float(max(k))
        arrays={key:np.array([p[key] for p in group],float) for key in
                ('forward','strike','time_to_expiry','discount_factor','option_price','error_scale','iv')}
        calls=np.array([p['option_type']=='call' for p in group])
        live=np.array([p['route']=='live' for p in group])
        bid=np.array([p['bid'] if p['route']=='live' else p['option_price'] for p in group])
        ask=np.array([p['ask'] if p['route']=='live' else p['option_price'] for p in group])
        previous=params[-1] if params else None;previous_domain=(lows[-1],highs[-1]) if params else None
        grid=np.linspace(lo,hi,opt['constraint_points'])
        calendar_grid=(np.linspace(max(lo,lows[-1]),min(hi,highs[-1]),opt['constraint_points']) if params else None)
        if params and calendar_grid[0]>=calendar_grid[-1]:return None,records,[]

        # Each expiry owns five parameters, fitted earliest to latest.
        # Optimizer x=[log(min w),log(b),rho,m,log(sigma)] ensures positive
        # variance via decoding. Additional density/digital/calendar
        # inequalities are imposed separately and audited after optimization.
        def residual(x):
            iv=np.sqrt(raw_svi(k,_decode_svi(x))[0]/t)
            predicted=black_prices(arrays['forward'],arrays['strike'],arrays['time_to_expiry'],arrays['discount_factor'],iv,calls)
            mid=(predicted-arrays['option_price'])/arrays['error_scale']
            # Live fit focuses on the observed interval. Historical observations
            # retain their price objective: an assumed error scale is NOT a spread.
            outside=(predicted-np.clip(predicted,bid,ask))/arrays['error_scale']
            return np.r_[outside,np.sqrt(opt['midpoint_weight'])*np.where(live,mid,0.)]

        def constraints(x):
            p=_decode_svi(x);values=slice_values(grid,p);digital=values[4]/np.exp(grid)
            result=[values[6],1+digital,-digital,np.array([2-p[1]*(1+abs(p[2]))])]
            if previous is not None:result.append(raw_svi(calendar_grid,p)[0]-raw_svi(calendar_grid,previous)[0])
            return np.concatenate(result)

        theta=float(np.median(arrays['iv'][np.argsort(abs(k))[:max(3,len(k)//7)]]**2)*t)
        bounds=([-20.,-12.,-.99,-1.,-10.],[2.,1.,.99,1.,1.])
        accepted=[]
        for start,rho in enumerate(policy['initial_rhos']):
            seed=[np.log(max(theta*.6,1e-8)),np.log(.1),rho,.03*start,np.log(.06+.04*start)]
            # SciPy soft-L1 least_squares supplies a robust starting fit;
            # SLSQP below enforces inequalities if it is infeasible. This is
            # not a QuantLib calibration. Deterministic multiple starts reduce
            # local-optimum sensitivity; only independently audited candidates
            # can win. Better residuals never override an arbitrage failure.
            initial=least_squares(residual,np.clip(seed,np.array(bounds[0])+1e-8,np.array(bounds[1])-1e-8),
                bounds=bounds,loss='soft_l1',max_nfev=policy['maximum_evaluations'])
            x=initial.x;converged=bool(initial.success);message=str(initial.message);refinements=0
            audit={}
            for refinement in range(opt['constraint_refinements']+1):
                if not converged or np.min(constraints(x)) < -tol:
                    objective=lambda value:float(np.mean(2*(np.sqrt(1+residual(value)**2)-1)))
                    solution=minimize(objective,x,method='SLSQP',bounds=list(zip(*bounds)),
                        constraints={'type':'ineq','fun':constraints},
                        options={'maxiter':opt['constraint_iterations'],'ftol':1e-10})
                    x=solution.x;converged=bool(solution.success);message=str(solution.message)
                if not np.isfinite(x).all():
                    converged=False;audit={'nonfinite_optimizer_result':(-np.inf,lo)};break
                p=_decode_svi(x)
                audit=audit_slice(p,lo,hi,previous,previous_domain,count=opt['audit_points'])
                violations={name:pair for name,pair in audit.items() if pair[0]<-tol}
                if converged and not violations and np.min(constraints(x))>=-tol:break
                # Add off-grid violations as HARD constraints before retrying.
                for name,(_,where) in violations.items():
                    if name=='calendar':calendar_grid=np.unique(np.r_[calendar_grid,where])
                    else:grid=np.unique(np.r_[grid,where])
                converged=False;refinements=refinement+1
            valid=converged and all(pair[0]>=-tol for pair in audit.values()) and np.min(constraints(x))>=-tol
            cost=float(np.mean(2*(np.sqrt(1+residual(x)**2)-1))) if np.isfinite(x).all() else None
            records.append(dict(expiration=expiry,start=start,converged=bool(converged),arbitrage_pass=bool(valid),
                cost=cost,evaluations=int(initial.nfev),constraint_refinements=refinements,
                audit_minima={name:float(pair[0]) if np.isfinite(pair[0]) else None for name,pair in audit.items()},message=message,
                search_boundary_reached=bool(np.any(np.isclose(x,bounds[0],atol=1e-6)) or np.any(np.isclose(x,bounds[1],atol=1e-6)))))
            if valid:accepted.append((cost,p))
        if not accepted:return None,records,[]
        cost,best=min(accepted,key=lambda pair:pair[0])
        comparable=[p for value,p in accepted if value<=cost*1.05+1e-8]
        change=max(float(np.max(abs(np.sqrt(raw_svi(k,p)[0]/t)-np.sqrt(raw_svi(k,best)[0]/t)))) for p in comparable)
        start_checks.append(dict(check='slice_alternative_starts',expiration=expiry,
            status='checked' if len(comparable)>1 else 'insufficient_comparable_starts',
            comparable_starts=len(comparable),max_iv_change=change))
        params.append(best);times.append(float(t));lows.append(lo);highs.append(hi)
    model=SviSurface(times,params,lows,highs,tol,start_checks=start_checks)
    # Alternative-start dispersion is reported at slice level in optimizer.csv;
    # later workflow sensitivity tests refit complete constrained surfaces.
    return model,records,[]
