"""Price-consistent completion for the raw-SVI normalized-call interpolator.

This is a pricing assumption, not a wider market-supported domain. Each expiry
keeps its original smile on a core containing the approved strips. Outside it,
positive mixtures of powers match the OTM premium and its first derivative.
Between expiries we retain the original linear normalized-call interpolation.
Before the first expiry we scale its completed total variance from zero to one.
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
    chosen so the slow component alone dominates the preceding entire tail
    beyond the new join. The remaining transition strip is audited separately.
    Continuity is C1 in strike; density can jump at a join. Exact maturity knots
    retain the right-interval convention, with a left limit at the terminal.
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
                weight = 0.
                if j:
                    previous = self.ends[j-1][side]
                    distance = (end['join']-previous['join'])*(1 if side == 0 else -1)
                    prior = previous['premium']*(previous['weight']*math.exp(self.low[side]*distance)
                        +(1-previous['weight'])*math.exp(previous['high']*distance))
                    weight = prior/end['premium']
                if not 0 <= weight < 1:
                    raise ValueError('wing_calendar_dominance_not_feasible')
                high = (end['effective']-weight*self.low[side])/(1-weight)
                end.update(weight=weight, high=high)
                self.records.append(dict(years=float(self.times[j]), side='put' if side == 0 else 'call',
                    **end, low=float(self.low[side]), variant=variant, observed_support=False))
        self.audit = self._audit()

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
                    minimum_density=minimum_density, outer_tail_ordering='analytic_dominance',
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

    def variance(self, t, k):
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
