"""Read-only diagnostics of simulated wealth; never draw or modify shocks.

Daily distributions pool path/session observations. Short horizons use each
path once, measured from forecast start (not overlapping rolling windows).
Portfolio daily log returns come from portfolio wealth, not weighted log returns.
"""
# Read-only path diagnostics: no random generator and no model fitting.
# Wealth arrays are (paths, dates), including date 0 initial valuation.
# Returns/losses are decimal fractions until charts explicitly multiply by 100.
import math
from decimal import Decimal
import numpy as np
import pandas as pd


# Relative loss is dimensionless (1 - terminal/initial). A narrow machine-
# precision tolerance prevents cancellation noise in exactly offsetting FX
# positions being classified as an economic loss. At initial wealth 100 this
# corresponds to approximately 1.42e-12 wealth units, NOT a trading threshold.
# It applies only to loss classification: paths, returns, VaR/ES and moments
# retain their computed values. Keep terminal and horizon reporting identical.
LOSS_PROBABILITY_TOLERANCE = 64 * np.finfo(np.float64).eps


def probability_of_loss(losses):
    """Fraction of paths with loss greater than numerical zero, in return units."""
    return float(np.mean(np.asarray(losses) > LOSS_PROBABILITY_TOLERANCE))


def probability_label(value):
    """Round-trip float label: distinct validated probabilities keep distinct keys.

    General formatting (:g) retains only six significant digits by default,
    so e.g. 0.95 and 0.9500001 previously overwrote the same output column.
    repr preserves the shortest decimal that recovers the exact input float,
    while keeping ordinary existing labels such as 0.95 and 0.99 unchanged.
    """
    return repr(float(value))


def confidence_label(value):
    """Exact decimal percent display for the same probability used by the key."""
    # Decimal scaling avoids rounding 99.00001% to 99% or introducing binary
    # multiplication noise in report headings. Strip only fractional zeros.
    label = format(Decimal(probability_label(value)) * 100, 'f')
    return (label.rstrip('0').rstrip('.') if '.' in label else label) + '%'


def _tail_mean(losses, confidence):
    # Sort losses largest first. ES averages exactly N*(1-confidence) tail
    # observations, including a fractional boundary instead of rounding count.
    ordered = np.sort(losses)[::-1]
    count = len(ordered) * (1-confidence)
    whole = int(math.floor(count))
    return float((ordered[:whole].sum()+(count-whole)*ordered[whole])/count)


def daily_distribution(values, mean, std, thresholds, bins=100):
    """Return moments and a fixed standardized histogram, with overflow counts.

Mean/std are calibration parameters for holdings. For the drifting buy-and-hold
portfolio they are pooled simulated moments, explicitly labelled by the caller.
No finite standardization is claimed when the standard deviation is zero.
"""
    # Pool all path/day returns for daily shape diagnostics, not terminal
    # wealth. Buy-and-hold portfolio daily distributions may change with time.
    x = np.asarray(values, dtype=float).ravel()
    # Empirical shape moments use their own mean. The variance below divides
    # by N for central-moment ratios; reported sample std uses N-1.
    centered = x-x.mean()
    variance = float(np.mean(centered**2))
    # Near-constant data cannot identify standardized shape. Leave undefined
    # skew/kurtosis as None instead of pretending they are zero/normal.
    stochastic = variance > 1e-24
    # Skew=m3/m2^(3/2); excess kurtosis=m4/m2^2-3 (normal excess is zero).
    # These are empirical ratios, not fitted PDFs or unbiased small-sample estimates.
    info = dict(observations=len(x), mean_daily_log=float(x.mean()),
                std_daily_log=float(x.std(ddof=1)) if len(x)>1 else None,
                skewness=float(np.mean(centered**3)/variance**1.5) if stochastic else None,
                excess_kurtosis=float(np.mean(centered**4)/variance**2-3) if stochastic else None,
                minimum_daily_log=float(x.min()), maximum_daily_log=float(x.max()))
    # Fixed DAILY LOG-return diagnostic quantiles are separate from configured
    # terminal wealth percentiles. They are not percentages of initial capital.
    for q in (.001,.01,.05,.5,.95,.99,.999):
        info[f"log_quantile_{probability_label(q)}"] = float(np.quantile(x,q))
    # A constant series has no meaningful z-score. Preserve raw statistics
    # but return an empty standardized histogram and unavailable tail measures.
    if std is None or not np.isfinite(std) or std <= 1e-12:
        for threshold in thresholds:
            info[f"probability_abs_z_gt_{threshold:g}"] = None
        info.update(histogram_underflow=0,histogram_overflow=0,standardization_available=False)
        return info, pd.DataFrame(columns=["left_z","right_z","count","density"])
    # Remove location/scale to compare shape: z=-4 is four std below mean,
    # not -4%. Caller chooses calibration or pooled simulation moments.
    z = (x-mean)/std
    # Count BOTH tails outside +/- threshold, with strict inequality.
    # A threshold of three is a sigma distance, not a confidence percentile.
    for threshold in thresholds:
        info[f"probability_abs_z_gt_{threshold:g}"] = float(np.mean(np.abs(z)>threshold))
    # Keep the same bins across scenarios. Outside observations are counted,
    # not clipped or silently dropped from the histogram denominator.
    edges = np.linspace(-10,10,bins+1)
    counts,_ = np.histogram(z,edges)
    info.update(histogram_underflow=int(np.sum(z<edges[0])),
                histogram_overflow=int(np.sum(z>edges[-1])),standardization_available=True)
    # Density=count/(all observations * bin width); AREA is probability.
    # Overflow is included in the denominator, so visible area may be <1.
    # Plot range does not clip shocks or renormalize away extreme observations.
    hist = pd.DataFrame(dict(left_z=edges[:-1],right_z=edges[1:],count=counts,
                             density=counts/(len(x)*np.diff(edges))))
    return info,hist


def wealth_diagnostics(values, initial_value, horizons, confidence_levels):
    """Use exact simulated wealth, including actual evolving portfolio weights."""
    # Simple return per transition. min(axis=1) selects each path's worst
    # single day, distinct from its potentially multi-day peak-to-trough loss.
    daily = values[:,1:]/values[:,:-1]-1
    worst_daily = daily.min(axis=1)
    # Running maximum includes initial wealth. Wealth/running_peak - 1 is
    # nonpositive; its path minimum is the signed maximum drawdown.
    drawdown = np.min(values/np.maximum.accumulate(values,axis=1)-1,axis=1)
    rows = []
    # Column h is h exchange sessions from initial valuation. Use one
    # observation per path/horizon, not overlapping rolling intervals.
    for horizon in horizons:
        # A horizon beyond the simulation is explicitly unavailable; do not
        # extrapolate a return or silently substitute a shorter horizon.
        if horizon >= values.shape[1]:
            rows.append(dict(horizon_sessions=horizon,status="unavailable_horizon",observations=0))
            continue
        returns = values[:,horizon]/initial_value-1
        # Positive losses mean wealth below initial value; gains are negative
        # losses. Treat only machine-precision residual losses as numerical zero.
        losses = -returns
        row = dict(horizon_sessions=horizon,status="complete",observations=len(returns),
                   mean_return=float(returns.mean()),probability_loss=probability_of_loss(losses),
                   return_p01=float(np.quantile(returns,.01)),return_p05=float(np.quantile(returns,.05)))
        # Upper loss quantile is VaR; ES averages the exact worst tail mass.
        # These are h-session measures, daily only when h=1.
        for confidence in confidence_levels:
            row[f"loss_var_{probability_label(confidence)}"] = float(np.quantile(losses,confidence))
            row[f"loss_es_{probability_label(confidence)}"] = _tail_mean(losses,confidence)
        rows.append(row)
    # Summarize across paths their worst day and maximum drawdown. A negative
    # p01 is a severe return/drawdown, not a positive loss magnitude.
    path = dict(mean_worst_daily_return=float(worst_daily.mean()),
                worst_daily_return_p01=float(np.quantile(worst_daily,.01)),
                worst_daily_return_p05=float(np.quantile(worst_daily,.05)),
                mean_max_drawdown=float(drawdown.mean()),
                max_drawdown_p01=float(np.quantile(drawdown,.01)),
                max_drawdown_p05=float(np.quantile(drawdown,.05)))
    return rows,path,worst_daily


def diagnostic_figures(tables, settings):
    """Yield figures to the caller, which owns safe filenames, saving and closing."""
    # Caller owns backend, saving, checksums and closing each yielded figure.
    # Plotting only reads completed tables and cannot change simulation draws.
    from matplotlib import pyplot as plt
    import hashlib
    histograms = tables["daily_histograms"]
    # Each calibration has its own historical comparison. FX-adjusted holding
    # distributions use the same units as the simulated holding paths.
    for (calibration,scope,key),data in histograms.groupby(["calibration_set","scope","instrument_id"],sort=False):
        fig,axes = plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
        for model,rows in data.groupby("model",sort=False):
            # Lines join histogram bin centres, not a fitted smooth PDF or dates.
            # Both panels use exactly the same density values.
            x = (rows["left_z"]+rows["right_z"])/2
            for ax in axes:
                # Zero-density bins have no finite logarithmic height. Mask them rather
                # than insert an invented small density; there is no smoothing.
                ax.plot(x,rows["density"].where(rows["density"]>0),label=model,linewidth=1.2)
        # Left uses linear y and central x range [-5,5]. Right uses log Y to
        # reveal rare tails over all [-10,10] bins. Log density is an axis scale,
        # NOT another logarithm applied to daily log returns.
        axes[0].set(title="Daily log returns: standardized density",xlabel="Standardized daily log return",ylabel="Density",xlim=(-5,5))
        axes[1].set(title="Daily tails (log density)",xlabel="Standardized daily log return",ylabel="Density (log scale)",yscale="log")
        axes[0].legend(fontsize=7)
        fig.suptitle(f"{calibration} | {key} | historical vs simulated daily returns")
        # Hash arbitrary identifiers into a safe filename; retain readable
        # product/calibration names in the figure title.
        token = hashlib.sha256((calibration+"|"+scope+"|"+key).encode()).hexdigest()[:12]
        yield "daily_tails_"+token,fig
    # Overview risk charts use the highest requested confidence; detailed
    # tables retain results for every configured confidence level.
    confidence = max(settings["confidence_levels"])
    horizon = tables["horizon_risk"]
    for calibration,data in horizon.loc[horizon["scope"]=="portfolio"].groupby("calibration_set",sort=False):
        fig,axes = plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
        for model,rows in data.groupby("model",sort=False):
            # Sort available horizons and skip unavailable ones. No interpolation
            # or inferred observations are added to the table.
            rows = rows.loc[rows["status"]=="complete"].sort_values("horizon_sessions")
            if rows.empty:
                continue
            for ax,metric in zip(axes,("loss_var_","loss_es_")):
                # Multiply decimal losses by 100 for percentage labels only. Negative
                # VaR/ES is valid when the chosen loss-tail threshold is still a gain.
                ax.plot(rows["horizon_sessions"],100*rows[metric+f"{probability_label(confidence)}"],marker="o",label=model)
        for ax,title in zip(axes,("VaR","Expected Shortfall")):
            ax.set(title=f"{confidence_label(confidence)} {title}",xlabel="Exchange sessions from forecast start",ylabel="Loss (% of initial wealth)")
            ax.set_xticks(settings["diagnostics"]["horizons_sessions"])
        if axes[0].get_legend_handles_labels()[0]:
            axes[0].legend(fontsize=7)
        else:
            for ax in axes:
                ax.text(.5,.5,"Configured horizons exceed forecast",ha="center",transform=ax.transAxes)
        fig.suptitle(calibration+" | Portfolio short-horizon risk")
        yield "horizon_risk_"+calibration,fig
    terminal = tables["terminal_values"]
    for calibration,data in terminal.loc[terminal["scope"]=="portfolio"].groupby("calibration_set",sort=False):
        fig,axes = plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
        for model,rows in data.groupby("model",sort=False):
            for ax,metric in zip(axes,("worst_daily_return","max_drawdown")):
                # Negate signed worst returns/drawdowns into positive loss magnitudes,
                # then sort for an empirical cumulative distribution, not a histogram.
                loss = np.sort(-100*rows[metric].to_numpy())
                # The kth sorted loss has empirical cumulative probability k/N: the
                # fraction of paths with risk magnitude at or below that x value.
                ax.plot(loss,np.arange(1,len(loss)+1)/len(loss),label=model)
        for ax,title in zip(axes,("Worst daily loss along each path","Maximum drawdown along each path")):
            ax.set(title=title,xlabel="Loss magnitude (%)",ylabel="Empirical cumulative probability")
        axes[0].legend(fontsize=7)
        fig.suptitle(calibration+" | Portfolio path-risk distributions")
        yield "path_risk_"+calibration,fig
