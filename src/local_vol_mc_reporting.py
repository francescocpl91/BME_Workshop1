"""Presentation and path snapshots for risk-neutral underlying validation.

This module never draws simulation shocks or changes local volatility. Snapshots
are taken AFTER existing fine-grid updates, without adding simulation time steps.
All path summaries use every path; displayed trajectories are an independent,
fixed random selection. Saved paths are observations, not every Euler increment.
"""
from datetime import timedelta
from html import escape
from pathlib import Path
import hashlib
import math
import numpy as np


def settings(cfg):
    """Optional reporting keeps older callers/configurations compatible."""
    value=cfg.get('reporting')
    if 'reporting' not in cfg:
        return dict(plots=False,sample_paths=0,time_points=250,save_all_paths=False)
    if not isinstance(value,dict) or set(value)!={'plots','sample_paths','time_points','save_all_paths'}:
        raise ValueError('invalid_mc_reporting_keys')
    for key in ('plots','save_all_paths'):
        if type(value[key]) is not bool:raise ValueError('invalid_mc_reporting_boolean:'+key)
    for key,low,high in (('sample_paths',0,500),('time_points',2,1000)):
        if type(value[key]) is not int or not low<=value[key]<=high:
            raise ValueError('invalid_mc_reporting_integer:'+key)
    return dict(value)


class PathRecorder:
    """Bounded observation storage; no hidden change to the pricing RNG.

    Snapshot targets are evenly spaced in elapsed time, rounded FORWARD to the
    next existing fine node. Every expiry and time zero are included. Hence the
    requested time_points is a target, not a promise of exactly that many rows.
    A separate fixed generator selects displayed IDs and cannot alter dynamics.
    """
    def __init__(self,adapter,cfg):
        self.adapter=adapter;self.cfg=settings(cfg);self.n=cfg['paths']
        self.enabled=bool(self.cfg['plots'] or self.cfg['sample_paths'] or self.cfg['save_all_paths'])
        if not self.enabled:return
        possible=self.cfg['time_points']+len(adapter.boundary.times)+1
        if self.cfg['save_all_paths'] and self.n*possible>20_000_000:
            raise ValueError('mc_path_storage_too_large_reduce_time_points_or_disable_save_all_paths')
        count=min(self.n,self.cfg['sample_paths'])
        self.ids=np.sort(np.random.default_rng(104729).choice(self.n,count,replace=False))
        self.targets=np.linspace(0,float(adapter.boundary.times[-1]),self.cfg['time_points'])
        self.next=0;self.last=None;self.bands=[];self.samples=[];self.logs=[]
        self.last_k=None

    def record(self,t,k):
        if not self.enabled:return
        due=self.next<len(self.targets) and t>=self.targets[self.next]-1e-12
        expiry=bool(np.any(np.isclose(t,self.adapter.boundary.times,atol=1e-12,rtol=0)))
        if not (due or expiry) or self.last==t:return
        while self.next<len(self.targets) and self.targets[self.next]<=t+1e-12:self.next+=1
        k=np.asarray(k,float)
        if k.shape!=(self.n,) or not np.isfinite(k).all():raise ValueError('invalid_mc_reporting_state')
        f,d=self.adapter.curves.at(float(t));x=np.exp(k);s=f*x
        if not np.isfinite(s).all():raise ValueError('invalid_mc_reporting_price')
        q=np.quantile(s,[0,.01,.05,.5,.95,.99,1])
        timestamp=(self.adapter.reference+timedelta(days=365*t)).isoformat()
        self.bands.append(dict(years=float(t),timestamp=timestamp,forward=float(f),discount=float(d),
            paths=self.n,mean=float(s.mean()),minimum=float(q[0]),p01=float(q[1]),p05=float(q[2]),
            median=float(q[3]),p95=float(q[4]),p99=float(q[5]),maximum=float(q[6])))
        self.samples.extend(dict(years=float(t),timestamp=timestamp,path_id=int(i),
            normalized_underlying=float(x[i]),underlying_price=float(s[i])) for i in self.ids)
        if self.cfg['save_all_paths']:self.logs.append(k.copy())
        self.last_k=k.copy();self.last=float(t)

    def finish(self):
        """Only called after a COMPLETE successful trajectory simulation.

        Passing numerical acceptance is separate: a complete review-required
        run still gets plots, but a stopped cohort never masquerades as complete.
        """
        if not self.enabled:return {},None
        if not self.bands or not math.isclose(self.last,float(self.adapter.boundary.times[-1]),abs_tol=1e-12):
            raise ValueError('incomplete_mc_path_capture')
        f=self.bands[-1]['forward'];x=np.exp(self.last_k)
        tables=dict(path_summary=self.bands,sample_paths=self.samples,terminal_paths=[
            dict(path_id=i,years=self.last,timestamp=self.bands[-1]['timestamp'],
                 normalized_underlying=float(v),underlying_price=float(f*v)) for i,v in enumerate(x)])
        data=None
        if self.cfg['save_all_paths']:
            data=dict(years=np.array([r['years'] for r in self.bands]),
                timestamps=np.array([r['timestamp'] for r in self.bands]),
                forward=np.array([r['forward'] for r in self.bands]),
                path_ids=np.arange(self.n,dtype=np.int64),log_normalized_paths=np.stack(self.logs))
        return tables,data


def token(gid):return hashlib.sha256(str(gid).encode()).hexdigest()[:12]


def groups(result):
    return sorted({r['group_id'] for r in result.tables.get('path_summary',[])})


def assets(result):
    """Fixed safe filenames, independent of ticker characters or path inputs."""
    files={}
    for gid in sorted(result.path_data):files['paths_'+token(gid)]='paths_'+token(gid)+'.npz'
    if settings(result.manifest['identity']['config'])['plots']:
        for gid in groups(result):
            for name in ('paths','terminal','validation'):
                files[name+'_plot_'+token(gid)]=name+'_'+token(gid)+'.png'
    return files


def plot(result,folder):
    """Generate offline PNGs. Red score=1 lines are existing limits, not new gates."""
    if not settings(result.manifest['identity']['config'])['plots']:return
    from datetime import datetime
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    for gid in groups(result):
        rows=[r for r in result.tables['path_summary'] if r['group_id']==gid]
        context=next(r for r in result.tables['cohorts'] if r['group_id']==gid)
        label=f"{context.get('underlying')} ({context.get('currency')}) · {context['status']}"
        dates=[datetime.fromisoformat(r['timestamp']) for r in rows]
        def values(key):return np.array([r[key] for r in rows],float)
        fig,ax=plt.subplots(figsize=(11,6),layout='constrained')
        ax.fill_between(dates,values('minimum'),values('maximum'),color='#e7eaf0',label='All-path min–max range')
        ax.fill_between(dates,values('p01'),values('p99'),color='#bdd9f1',label='1st–99th percentiles')
        ax.fill_between(dates,values('p05'),values('p95'),color='#78b0db',label='5th–95th percentiles')
        sampled=[r for r in result.tables.get('sample_paths',[]) if r['group_id']==gid]
        paths={}
        for r in sampled:paths.setdefault(r['path_id'],[]).append(r['underlying_price'])
        for path in paths.values():ax.plot(dates,path,color='#42647a',alpha=.17,linewidth=.6)
        ax.plot(dates,values('median'),color='#12304a',lw=2,label='Median')
        ax.plot(dates,values('forward'),color='#bb671a',lw=1.5,ls='--',label='Forward / theoretical mean')
        ax.set(title=f"Risk-neutral paths: {label}\n{rows[0]['paths']:,} paths in bands; {len(paths)} displayed trajectories",ylabel='Underlying price in quote currency',xlabel='Simulation date',ylim=(0,None))
        locator=mdates.AutoDateLocator(minticks=4,maxticks=7);ax.xaxis.set_major_locator(locator);ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.legend(fontsize=8,loc='upper left');ax.grid(alpha=.15)
        fig.savefig(Path(folder)/('paths_'+token(gid)+'.png'),dpi=145);plt.close(fig)
        terminal=np.array([r['underlying_price'] for r in result.tables['terminal_paths'] if r['group_id']==gid])
        fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
        for ax,log in zip(axes,(False,True)):
            ax.hist(terminal,bins=80,color='#5a9dcb',alpha=.8,log=log)
            ax.axvline(rows[-1]['forward'],color='#bb671a',ls='--',label='Forward')
            ax.axvline(float(np.median(terminal)),color='#12304a',label='Median')
            ax.set(xlabel='Terminal underlying price',ylabel='Path count'+(' (log scale)' if log else ''),title='Tail view' if log else 'Full distribution');ax.legend(fontsize=8)
        fig.suptitle('Risk-neutral terminal distribution: '+label,fontsize=11)
        fig.savefig(Path(folder)/('terminal_'+token(gid)+'.png'),dpi=145);plt.close(fig)
        quotes=[r for r in result.tables['quotes'] if r['group_id']==gid]
        expiries=sorted({r['expiration'] for r in quotes});idx=np.arange(len(expiries))
        fig,axes=plt.subplots(2,1,figsize=(11,7),layout='constrained')
        calculations=(('Fitted-price agreement',lambda q:abs(q['price_error'])/(q['deterministic_budget']+q['sampling_half_width'])),
            ('PDE-price agreement',lambda q:abs(q['pde_error'])/(q['deterministic_budget']+q['sampling_half_width'])),
            ('Step convergence',lambda q:abs(q['paired_difference'])/(q['deterministic_budget']+q['confidence_multiplier']*q['paired_difference_se'])),
            ('Sampling precision',lambda q:q['standard_error']/q['precision_budget']))
        for name,calc in calculations:
            axes[0].plot(idx,[max(calc(q) for q in quotes if q['expiration']==e) for e in expiries],marker='o',ms=3,label=name)
        for level in ('coarse','fine'):
            fw=[r for r in result.tables['forwards'] if r['group_id']==gid and r['level']==level]
            axes[1].plot(np.arange(len(fw)),[abs(r['normalized_mean']-1)/r['tolerance'] for r in fw],marker='o',label=level)
        for ax in axes:
            ax.axhline(1,color='#b53a35',ls='--',label='Existing pass limit');ax.set_xticks(idx,expiries,rotation=40,ha='right');ax.set_ylabel('Error / allowed budget');ax.set_ylim(bottom=0);ax.grid(alpha=.15);ax.legend(fontsize=8)
        axes[0].set_title('Worst option-check score per expiry; ≤ 1 passes')
        axes[1].set_title('Forward-expectation check; ≤ 1 passes')
        fig.suptitle('Validation diagnostics: '+label,fontsize=11)
        fig.savefig(Path(folder)/('validation_'+token(gid)+'.png'),dpi=145);plt.close(fig)


def section(result):
    """Readable explanation of what is stored and what a plot does NOT certify."""
    if not groups(result):return '<h2>Path plots</h2><p>No completed path capture is available. Reporting may be disabled, or the cohort was skipped/stopped.</p>'
    cfg=settings(result.manifest['identity']['config'])
    text=['<h2>Paths and validation plots</h2><p>Every launch runs Monte Carlo. The fixed seed makes identical inputs reproducible; saving outputs is not simulation caching. Existing identical result folders are verified, not overwritten.</p>',
          '<p>Bands use every simulated path. Thin trajectories are a separate random sample. The grey envelope is the pointwise min–max, not one path. Terminal histograms use every final path. These are risk-neutral underlying distributions, not portfolio or real-world forecasts. A review-required cohort is still shown with its failed checks.</p>',
          '<p>Snapshots include time zero, every expiry and selected existing fine-grid nodes. No extra time steps or shocks are introduced for plotting. sample_paths.csv stores displayed trajectories; path_summary.csv stores all-path bands; terminal_paths.csv stores every final underlying value.</p>']
    for gid in groups(result):
        c=next(r for r in result.tables['cohorts'] if r['group_id']==gid)
        text.append('<h3>'+escape(str(c.get('underlying')))+' · '+escape(str(c['status']))+'</h3>')
        if cfg['plots']:
            for name in ('paths','terminal','validation'):
                file=name+'_'+token(gid)+'.png';text.append('<a href="'+file+'"><img style="max-width:100%;height:auto" src="'+file+'" alt="'+name+'"></a>')
        if gid in result.path_data:
            file='paths_'+token(gid)+'.npz'
            text.append('<p><a href="'+file+'">All paths at the saved observation times (NumPy NPZ)</a>. Arrays: years, timestamps, forward, path_ids, log_normalized_paths. Rows are times; columns are paths. Recover prices with exp(log_normalized_paths) × forward[:, None]. This file contains every path at the saved times, not every Euler step. Path IDs in its two halves are antithetic partners.</p>')
    return ''.join(text)
