"""Price-consistent completion for the raw-SVI normalized-call interpolator.

This is a pricing assumption, not a wider market-supported domain. Each expiry
keeps its original smile on a core containing the approved strips. Outside it,
positive mixtures of powers match the OTM premium and its first derivative.
The unsmoothed construction uses linear normalized calls between expiries
and scaled first-smile total variance before the first expiry. An optional
positive time average then smooths the completed PRICES, subject to unchanged
fit tolerances. The PDE and opted-in state/MC interfaces use that same model.
The strict public local-volatility evaluator is never modified.
"""
import math
import numpy as np
from scipy.special import ndtr
from .surface_model import SviSurface, raw_svi, slice_values, black_prices, black_partials
from .local_vol_pricing import BoundaryLocalVol


class CalibratedBoundaryLocalVol(BoundaryLocalVol):
    """Complete prices first, then differentiate them to obtain local variance.

    Wing exponents >1 on the put side and >0 on the call side give positive
    density and the correct limiting prices/digitals. Successive mixtures are
    chosen by the exact minimum of successive tail-price differences; legacy
    zero-seed settings retain the older sufficient dominance rule. The core
    and transition strips are audited separately. Price is C1 in strike;
    density can jump at a join. Positive time averaging smooths maturity
    derivatives without assuming additional market support.
    """
    def __init__(self, evaluator, cfg, variant='baseline'):
        super().__init__(evaluator, cfg, variant)
        self.model = evaluator.model
        if not isinstance(self.model, SviSurface):
            raise ValueError('calibrated_boundary_requires_raw_svi_call_interpolation')
        if not np.allclose(self.times, self.model.times, rtol=0, atol=1e-12):
            raise ValueError('calibrated_boundary_requires_all_expiry_intervals')
        self._cache = {}  # bounded cache: only the latest PDE strike mesh
        self.records = []
        self.ends = []
        lo = hi = 0.
        for j, t in enumerate(self.times):
            adjacent = [d for d in self.domains if abs(t-d['start_years']) < 1e-12
                        or abs(t-d['end_years']) < 1e-12]
            # Nested cores avoid moving a wing join back into an earlier core.
            # This may use fitted node strikes outside the approved time strip;
            # it must never extrapolate the fitted node itself.
            lo = min(lo, min(d['k_min'] for d in adjacent))
            hi = max(hi, max(d['k_max'] for d in adjacent))
            if lo < self.model.k_min[j]-1e-12 or hi > self.model.k_max[j]+1e-12:
                raise ValueError('nested_wing_joins_exceed_fitted_node_support')
            pair = []
            for side, join in enumerate((lo, hi)):
                w, wk, _, _, ck, _, g = [float(z) for z in slice_values(join, self.model.parameters[j])]
                premium = float(black_prices(1., math.exp(join), 1., 1., math.sqrt(w), side == 1))
                # Put derivative uses N(-d2), avoiding cancellation of c_k+x.
                _, _, cw, _, _, _ = black_partials(join, w)
                d2 = -join/math.sqrt(w)-.5*math.sqrt(w)
                derivative = math.exp(join)*float(ndtr(-d2))+float(cw)*wk if side == 0 else -ck
                # A mathematically positive far-OTM premium can underflow
                # to zero. Reject that join BEFORE division, so the cohort
                # becomes unavailable instead of aborting the entire run.
                # Never replace the missing premium with an arbitrary floor.
                if not (math.isfinite(premium) and premium > 0 and
                        math.isfinite(derivative) and math.isfinite(g) and
                        g >= evaluator.minimum_density):
                    raise ValueError('invalid_wing_join_price_slope_or_density')
                effective = derivative/premium
                if not (math.isfinite(effective) and effective > (1 if side == 0 else 0)):
                    raise ValueError('invalid_wing_join_price_slope_or_density')
                pair.append(dict(join=join, premium=premium, effective=effective))
            self.ends.append(pair)
        fraction = cfg['wing_tail_fraction_alternative'] if variant == 'wing_alternative' else cfg['wing_tail_fraction']
        exponents = np.array([[p['effective'] for p in pair] for pair in self.ends])
        self.low = np.array([1+fraction*(exponents[:, 0].min()-1), fraction*exponents[:, 1].min()])
        for j, pair in enumerate(self.ends):
            for side, end in enumerate(pair):
                # Every maturity needs the same positive asymptotic component.
                # With weight=0 at the first node but >0 later, the ratio of
                # successive tail prices can explode as k tends to infinity.
                # Price and slope still determine the remaining exponent.
                weight = cfg['first_wing_tail_weight']
                if j:
                    previous = self.ends[j-1][side]
                    distance = (end['join']-previous['join'])*(1 if side == 0 else -1)
                    prior = previous['premium']*(previous['weight']*math.exp(self.low[side]*distance)
                        +(1-previous['weight'])*math.exp(previous['high']*distance))
                    if cfg['first_wing_tail_weight']:
                        weight = self._minimal_ordered_weight(previous,end,side,distance)
                    else:
                        weight = prior/end['premium']
                if not 0 <= weight < 1:
                    raise ValueError('wing_calendar_dominance_not_feasible')
                high = (end['effective']-weight*self.low[side])/(1-weight)
                end.update(weight=weight, high=high)
                self.records.append(dict(years=float(self.times[j]), side='put' if side == 0 else 'call',
                    **end, low=float(self.low[side]), variant=variant, observed_support=False))
        # Width scales with THIS surface's smallest expiry gap, never a
        # hardcoded date, asset or volatility. Zero retains the legacy model.
        self.smoothing_width = cfg['time_smoothing_fraction']*float(np.min(np.diff(np.r_[0.,self.times])))
        self._time_nodes, self._time_weights = np.polynomial.legendre.leggauss(cfg['time_smoothing_quadrature'])
        self.audit = self._audit()
        self.audit.update(calibration=('node_anchors_before_time_average' if self.smoothing_width else 'exact_first_smile_anchor'),
            first_wing_tail_weight=cfg['first_wing_tail_weight'],
            time_smoothing_width_years=self.smoothing_width,
            time_smoothing_arbitrage='positive_average_of_monotone_time_changes',
            common_positive_asymptotic_tail=cfg['first_wing_tail_weight']>0)

    def _minimal_ordered_weight(self, previous, end, side, distance):
        """Smallest slow-tail mass satisfying exact global calendar ordering.

        The old sufficient rule made the NEW slow component dominate BOTH old
        components. That unnecessarily promoted fast-decaying mass into the
        slow tail at every expiry. Instead compare the two mixtures exactly.

        At outward distance x>=0 from the new join, factor out exp(-p*x).
        The price difference is Anew-Aold + Bnew*exp(-u*x)-Bold*exp(-v*x).
        Its minimum occurs at zero, infinity, or its single stationary point.
        Increasing the mixture weight raises every tail price while leaving
        the join price/slope fixed, so a bracketed scalar bisection is valid.
        This uses each node's prices/slopes; there is no asset-specific tuning.
        """
        p=float(self.low[side]); price=end['premium']; effective=end['effective']
        a=previous['premium']*previous['weight']*math.exp(p*distance)
        b=previous['premium']*(1-previous['weight'])*math.exp(previous['high']*distance)
        v=previous['high']-p
        if price < a+b-1e-12*price:raise ValueError('wing_join_calendar_crossing')
        lower=a/price; upper=(a+b)/price
        if not 0<=lower<=upper<1:raise ValueError('wing_calendar_weight_bracket_failed')
        def minimum(weight):
            new_b=price*(1-weight)
            u=(effective-p)/(1-weight)
            value=min(price-a-b,price*weight-a)
            # u>v is the only case with an interior MINIMUM. For u<=v,
            # any stationary point is a maximum and the endpoints suffice.
            if b>0 and v>0 and u>v:
                location=math.log((u*new_b)/(v*b))/(u-v)
                if location>0:
                    value=min(value,price*weight-a+new_b*math.exp(-u*location)-b*math.exp(-v*location))
            return value
        if minimum(lower)>=0:return lower
        for _ in range(64):
            middle=(lower+upper)/2
            if minimum(middle)>=0:upper=middle
            else:lower=middle
        return upper

    def node(self, j, k):
        """OTM premium, k derivative and (c_kk-c_k), without ITM cancellation.

        OTM puts on k<0 and calls on k>=0 share the same time derivative and
        density: intrinsic value has zero time derivative and c_kk-c_k=0.
        All three outputs therefore support a stable Dupire ratio in the tails.
        """
        k = np.asarray(k, float)
        left, right = self.ends[j]
        bounded = np.clip(k, left['join'], right['join'])
        w, wk, wkk = raw_svi(bounded, self.model.parameters[j])
        _, ck, cw, _, _, _ = black_partials(bounded, w)
        premium = black_prices(1., np.exp(bounded), 1., 1., np.sqrt(w), k >= 0)
        d2 = -bounded/np.sqrt(w)-.5*np.sqrt(w)
        slope = np.where(k < 0, np.exp(bounded)*ndtr(-d2), ck)+cw*wk
        g = (1-bounded*wk/(2*w))**2-.25*wk*wk*(1/w+.25)+.5*wkk
        density = 2*cw*g
        for side, end in enumerate((left, right)):
            sign = 1 if side == 0 else -1
            mask = k < end['join'] if side == 0 else k > end['join']
            delta = np.minimum(sign*(k-end['join']), 0.)
            low, high = self.low[side], end['high']
            a = end['premium']*end['weight']*np.exp(low*delta)
            b = end['premium']*(1-end['weight'])*np.exp(high*delta)
            premium = np.where(mask, a+b, premium)
            slope = np.where(mask, sign*(low*a+high*b), slope)
            density = np.where(mask, low*(low-sign)*a+high*(high-sign)*b, density)
        return premium, slope, density

    def _audit(self):
        """Audit cores/transition strips; analytic tail dominance covers infinity.

        Mesh includes joins and raw-SVI curvature centres at multiple scales.
        This remains a numerical audit in the fitted cores, not a symbolic
        guarantee about raw SVI. Material violations block pricing altogether.
        """
        lo, hi = self.cfg['wider_log_grid_bounds']
        mesh = [np.linspace(lo, hi, self.cfg['boundary_audit_points'])]
        for pair, p in zip(self.ends, self.model.parameters):
            mesh.append(np.array([pair[0]['join'], pair[1]['join']]))
            mesh.append(np.clip(p[3]+p[4]*np.linspace(-10, 10, 1001), lo, hi))
        k = np.unique(np.concatenate(mesh))
        previous = None
        minimum_calendar = math.inf
        minimum_density = math.inf
        for j in range(len(self.times)):
            value, slope, density = self.node(j, k)
            digital = np.where(k < 0, slope/np.exp(k), 1+slope/np.exp(k))
            if not all(np.isfinite(a).all() for a in (value, slope, density)) or np.any(value <= 0) or np.any(density <= 0):
                raise ValueError('completed_wing_invalid_density_or_price')
            if np.any(digital < -1e-10) or np.any(digital > 1+1e-10):
                raise ValueError('completed_wing_invalid_digital')
            minimum_density = min(minimum_density, float(density.min()))
            if previous is not None:
                diff = value-previous
                minimum_calendar = min(minimum_calendar, float(diff.min()))
                # Relative test also catches tiny but materially negative tails.
                if np.any(diff < -1e-10*np.maximum(value, previous)):
                    raise ValueError('completed_wing_calendar_crossing')
            previous = value
        return dict(status='passed', points=len(k), minimum_calendar_price_difference=minimum_calendar,
                    minimum_density=minimum_density, outer_tail_ordering=('analytic_stationary_minimum' if self.cfg['first_wing_tail_weight'] else 'analytic_dominance'),
                    core_and_transition_ordering='numerical_audit', calibration='exact_first_smile_anchor',
                    maximum_tail_exponent=max(r['high'] for r in self.records),
                    price_continuity='C1_at_wing_joins_density_may_jump',
                    warning='unobserved_wings_can_have_large_local_variance_near_maturity_knots')

    def _first_variance(self, k):
        """Invert completed OTM premiums once per mesh, with analytic derivatives.

        Vega can be tiny outside observed strikes. Using OTM prices/derivatives
        avoids subtracting almost equal ITM values. Invalid inversions block;
        there is no volatility clipping or numerical replacement.
        """
        premium, slope, density = self.node(0, k)
        # Each point owns its bracket: a large strike elsewhere in a batch
        # must not decide whether this point can be inverted. Expand only the
        # insufficient brackets, and fail explicitly if no bracket is found.
        # The expansion limit is a numerical guard, never a volatility cap.
        k = np.asarray(k, float)
        strike = np.exp(k)
        if (not np.isfinite(premium).all() or np.any(premium <= 0)
                or np.any(premium >= np.minimum(1., strike))):
            raise ValueError('completed_first_smile_invalid_premium')
        low = np.zeros_like(k)
        high = np.maximum(1., 2*np.abs(k))
        for _ in range(32):
            price = black_prices(1., strike, 1., 1., np.sqrt(high), k >= 0)
            insufficient = price < premium
            if not np.any(insufficient):break
            high = np.where(insufficient, 2*high, high)
        else:
            raise ValueError('completed_first_smile_bracket_failed')
        for _ in range(64):
            middle = (low+high)/2
            below = black_prices(1., np.exp(k), 1., 1., np.sqrt(middle), k >= 0) < premium
            low = np.where(below, middle, low)
            high = np.where(below, high, middle)
        w = (low+high)/2
        _, ck, cw, _, ckw, cww = black_partials(k, w)
        sd = np.sqrt(w); d2 = -k/sd-.5*sd
        partial_k = np.where(k < 0, np.exp(k)*ndtr(-d2), ck)
        partial_kk = partial_k+np.exp(k)*np.exp(-.5*d2*d2)/(np.sqrt(2*np.pi)*sd)
        wk = (slope-partial_k)/cw
        wkk = (density+slope-partial_kk-2*ckw*wk-cww*wk*wk)/cw
        # Retain exact raw-SVI derivatives inside the first node's core.
        inside = (k >= self.ends[0][0]['join']) & (k <= self.ends[0][1]['join'])
        exact = raw_svi(k, self.model.parameters[0])
        w, wk, wkk = [np.where(inside, x, y) for x, y in zip(exact, (w, wk, wkk))]
        check = black_prices(1., np.exp(k), 1., 1., np.sqrt(w), k >= 0)
        if not all(np.isfinite(a).all() for a in (w, wk, wkk)) or np.any(abs(check-premium) > 1e-7*premium):
            raise ValueError('completed_first_smile_inversion_failed')
        return w, wk, wkk

    def cell_variance(self, t, centres, h):
        """Harmonic cell average for the non-divergence forward strike PDE.

        Put Lc=c_kk-c_k. Its exponentially fitted difference stencil is
        [c(i+1)-(1+exp(h))*c(i)+exp(h)*c(i-1)]/[h*expm1(h)].
        This equals a positive weighted cell average of Lc, with kernel
        exp(h)-exp(-s) on -h<s<0 and exp(h-s)-1 on 0<s<h.
        As Lc=2*c_T/variance, approximating c_T locally by its node value
        requires the HARMONIC, not arithmetic, average of variance.

        Three-point Gaussian quadrature splits at wing joins and samples their
        short exponential scales. This resolves density jumps without smoothing
        the fitted prices or calibrating a discrete operator to target prices.
        Constants and exp(k) are exact null modes of the matching stencil.
        """
        centres = np.asarray(centres, float)
        if centres.ndim != 1 or not len(centres) or not np.isfinite(h) or h <= 0:
            raise ValueError('invalid_cell_average_grid')
        cached = getattr(self, '_cell_cache', None)
        if cached is None or cached['h'] != h or not np.array_equal(cached['centres'], centres):
            breaks = []
            for pair in self.ends:
                for side, end in enumerate(pair):
                    # These are integration subdivisions, not model parameters.
                    breaks.extend(end['join']+(1 if side else -1)*np.array([0,.25,.5,1,2,4,8])/end['high'])
            breaks = np.unique(breaks)
            gx, gw = np.polynomial.legendre.leggauss(3)
            nodes, weights, indices = [], [], []
            for i, centre in enumerate(centres):
                edges = np.unique(np.r_[centre-h, centre, centre+h,
                    breaks[(breaks > centre-h) & (breaks < centre+h)]])
                for left, right in zip(edges[:-1], edges[1:]):
                    points = (left+right)/2+(right-left)*gx/2
                    offset = points-centre
                    kernel = np.where(offset < 0, np.expm1(h)-np.expm1(-offset), np.expm1(h-offset))
                    nodes.extend(points)
                    weights.extend(gw*(right-left)*kernel/(2*h*np.expm1(h)))
                    indices.extend([i]*3)
            indices = np.array(indices); weights = np.array(weights); nodes = np.array(nodes)
            if np.any(weights < 0):raise ValueError('invalid_cell_quadrature_weights')
            # Nearly coincident joins can create zero-measure quadrature nodes.
            # Drop those before division, avoiding 0/0 at zero local variance.
            positive = weights > 0
            indices, weights, nodes = indices[positive], weights[positive], nodes[positive]
            self._cell_cache = cached = dict(centres=centres.copy(), h=h, nodes=np.array(nodes),
                weights=weights, indices=indices,
                mass=np.bincount(indices, weights=weights, minlength=len(centres)))
        v = self.variance(t, cached['nodes'])
        # A truly zero local variance gives zero harmonic diffusion. Invalid
        # or negative coefficients have already been rejected by variance().
        with np.errstate(divide='ignore'):
            inverse = np.bincount(cached['indices'], weights=cached['weights']/v, minlength=len(centres))
        return cached['mass']/inverse

    def _unsmoothed_variance(self, t, k):
        if not np.isfinite(t) or not 0 <= t <= self.times[-1]:
            raise ValueError('pricing_time_outside_horizon')
        k = np.asarray(k, float)
        if not k.size or not np.isfinite(k).all():raise ValueError('nonfinite_pricing_state')
        # Avoid repeated node evaluation / inversion for thousands of PDE steps.
        if 'k' not in self._cache or not np.array_equal(k, self._cache['k']):
            self._cache = dict(k=k.copy(), nodes=[self.node(j, k) for j in range(len(self.times))])
        if t < self.times[0]:
            if 'first' not in self._cache:self._cache['first'] = self._first_variance(k)
            w1, wk1, wkk1 = self._cache['first']
            power = self.cfg['short_end_power_alternative'] if self.variant == 'short_end_alternative' else 1.
            ratio = t/self.times[0]; a = ratio**power
            derivative = power*ratio**(power-1)/self.times[0]
            # Algebraically expanded g avoids division by zero at t=0.
            g = (1-k*wk1/(2*w1))**2 + a*(.5*wkk1-.25*wk1*wk1/w1) - a*a*wk1*wk1/16
            if np.any(g <= 0) or not np.isfinite(g).all():raise ValueError('invalid_short_end_density')
            result = derivative*w1/g
        else:
            j = int(np.clip(np.searchsorted(self.times, t, side='right')-1, 0, len(self.times)-2))
            width = self.times[j+1]-self.times[j]; a = (t-self.times[j])/width
            first, second = self._cache['nodes'][j:j+2]
            density = (1-a)*first[2]+a*second[2]
            change = second[0]-first[0]
            if np.any(change < -1e-10*np.maximum(first[0], second[0])) or np.any(density <= 0):
                raise ValueError('invalid_completed_call_time_derivative_or_density')
            # Only signed floating-point roundoff may be zeroed, never prices.
            result = 2*np.maximum(change, 0)/(width*density)
        if not np.isfinite(result).all() or np.any(result < 0):raise ValueError('invalid_extended_variance')
        return result


    def _base_price_density_variance(self, t, k):
        """Completed OTM premium, log density and Dupire variance before smoothing.

        Log density avoids 0/0 in far-OTM short-time tails. It is a numerical
        representation of the same positive density, not a density floor.
        """
        variance = self._unsmoothed_variance(t,k)
        if t < self.times[0]:
            w1,wk1,wkk1 = self._cache['first']
            power = self.cfg['short_end_power_alternative'] if self.variant=='short_end_alternative' else 1.
            a = (t/self.times[0])**power
            w,wk,wkk = a*w1,a*wk1,a*wkk1
            g = (1-k*wk/(2*w))**2-.25*wk*wk*(1/w+.25)+.5*wkk
            d1 = -k/np.sqrt(w)+.5*np.sqrt(w)
            # density = phi(d1)*g/sqrt(w), valid for both OTM calls and puts.
            log_density = -.5*d1*d1-.5*math.log(2*math.pi)+np.log(g)-.5*np.log(w)
            premium = black_prices(1.,np.exp(k),1.,1.,np.sqrt(w),k>=0)
        else:
            j=int(np.clip(np.searchsorted(self.times,t,side='right')-1,0,len(self.times)-2))
            alpha=(t-self.times[j])/(self.times[j+1]-self.times[j])
            first,second=self._cache['nodes'][j:j+2]
            premium=(1-alpha)*first[0]+alpha*second[0]
            log_density=np.log((1-alpha)*first[2]+alpha*second[2])
        return premium,log_density,variance

    def _time_window(self,t):
        """Smooth finite window, vanishing at origin and final expiry.

        h=delta*(1-exp(-(t/delta)^2))*(1-exp(-((Tmax-t)/delta)^2)).
        h<min(t,Tmax-t), |h'|<sqrt(2/e)<1. Thus every mapped time
        g_z=t+h*z, -1<=z<=1, stays in the fitted horizon and increases
        with t. Near zero h/t tends to zero, retaining the original limit.
        """
        delta=self.smoothing_width; last=float(self.times[-1])
        u,v=t/delta,(last-t)/delta
        eu,ev=math.exp(-u*u),math.exp(-v*v)
        a,b=-math.expm1(-u*u),-math.expm1(-v*v)
        return delta*a*b, 2*u*eu*b-2*v*ev*a

    def completed_values(self,t,k):
        """Positive uniform time average of completed option prices.

        C_new(t,k)=1/2 integral[-1,1] C_old(t+h(t)*z,k) dz.
        Each slice is convex in strike. Every mapped time is increasing, so
        calendar monotonicity also survives. Density is the positive average
        of old densities; C_t averages old C_t*(1+h'*z), also nonnegative.
        Split integration at EVERY expiry: a quadrature rule may not straddle
        a derivative jump. Linear-price intervals integrate exactly; only the
        nonlinear short end needs Gaussian quadrature. No variance cap is used.

        Smoothing can slightly move prices at intermediate quoted expiries.
        The unchanged fit/PDE/IV budgets must accept those moves; node matching
        is not falsely claimed to remain exact after time smoothing.
        """
        t=float(t);k=np.asarray(k,float)
        if not np.isfinite(t) or not 0<t<=self.times[-1] or not k.size or not np.isfinite(k).all():
            raise ValueError('invalid_completed_price_query')
        if self.smoothing_width==0 or t==self.times[-1]:
            p,ld,v=self._base_price_density_variance(t,k)
            return dict(premium=p,density=np.exp(ld),time_derivative=.5*np.exp(ld)*v,variance=v)
        h,hp=self._time_window(t)
        if h<=np.finfo(float).eps*t:
            p,ld,v=self._base_price_density_variance(t,k)
            return dict(premium=p,density=np.exp(ld),time_derivative=.5*np.exp(ld)*v,variance=v)
        edges=np.r_[-1.,(self.times[(self.times>t-h)&(self.times<t+h)]-t)/h,1.]
        samples=[]
        for left,right in zip(edges[:-1],edges[1:]):
            mid=(left+right)/2; half=(right-left)/2
            if t+h*mid < self.times[0]:
                nodes=mid+half*self._time_nodes; weights=.5*half*self._time_weights
            else:
                nodes=np.array([mid]);weights=np.array([half])
            for z,weight in zip(nodes,weights):
                p,ld,v=self._base_price_density_variance(t+h*z,k)
                samples.append((float(weight),p,ld,v*(1+hp*z)))
        log_scale=np.maximum.reduce([item[2] for item in samples])
        mass=np.zeros_like(k); numerator=np.zeros_like(k); premium=np.zeros_like(k)
        for weight,p,ld,v in samples:
            scaled=weight*np.exp(ld-log_scale)
            mass+=scaled; numerator+=scaled*v; premium+=weight*p
        variance=numerator/mass
        if not np.isfinite(variance).all() or np.any(variance<0) or np.any(mass<=0):
            raise ValueError('invalid_smoothed_completion')
        density=np.exp(log_scale)*mass
        return dict(premium=premium,density=density,time_derivative=.5*density*variance,variance=variance)

    def variance(self,t,k):
        """Use the SAME completed coefficient for PDE, state lookup and MC."""
        if self.smoothing_width==0 or t==0:
            return self._unsmoothed_variance(t,k)
        return self.completed_values(t,k)['variance']


    def time_nodes(self,start,end,steps_per_year):
        """Resolve the smoothing window instead of stepping over it.

        Additional nodes depend only on this surface's expiries/width and the
        requested numerical resolution. Doubling steps_per_year also doubles
        the local refinement; fine/coarse PDE tests remain meaningful.
        """
        n=max(2,int(math.ceil((end-start)*steps_per_year)))
        nodes=[np.linspace(start,end,n+1)]
        if self.smoothing_width:
            delta=self.smoothing_width
            resolution=max(2,int(math.ceil(self.cfg['time_smoothing_time_points']*steps_per_year/365)))
            for knot in np.r_[0.,self.times]:
                a=max(start,float(knot)-3*delta);b=min(end,float(knot)+3*delta)
                if a<b:
                    count=max(1,int(math.ceil((b-a)/(delta/resolution))))
                    nodes.append(np.linspace(a,b,count+1))
        return np.unique(np.concatenate(nodes))
