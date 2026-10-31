"""Concise decision report, safe local HTML and optional scientific plots.

The report summarizes recorded decisions. Detailed numerical evidence stays
in CSV/JSON outputs, not repeated report appendices. Rendering never fits data.
"""
from html import escape
from urllib.parse import quote
import math
import re
import numpy as np

"""Short decision report. Full measurements remain in the checksummed CSVs."""


def _value(value):
    if value is None:return 'Not assessed'
    if isinstance(value,bool):return 'PASS' if value else 'BLOCKED'
    if isinstance(value,float):return f'{value:.5g}'
    return str(value)


def _table(headers,rows):
    def cell(v):return _value(v).replace('|','\\|').replace('\n',' ')
    return ['', '| '+' | '.join(headers)+' |','|'+'|'.join('---' for _ in headers)+'|']+[
        '| '+' | '.join(cell(v) for v in row)+' |' for row in rows]+['']


def render_report(result):
    """Report the actual eligibility decision; never infer a different one."""
    cfg=result.manifest['identity']['config'];q=cfg['quality']
    lines=['# Implied-volatility surface — decision', '',
        f'Archive COB: **{result.manifest.get("archive_cob","not available")}**. Analysis: `{result.run_id}`.', '',
        '## Outcome', '',
        'An eligible result supplies inputs for the separate bounded Dupire stage. '
        'This report assesses the implied surface only; consult the downstream local_volatility report for conversion and pricing results. Full Monte Carlo use is not approved.', '']
    cohorts=result.tables['cohort_status']
    if not cohorts:lines += ['Blocked: '+str(result.manifest['summary'].get('skip_reason','no_option_data')), '']
    for cohort in cohorts:
        gid=cohort['group_id'];rows=lambda n:[r for r in result.tables.get(n,[]) if r['group_id']==gid]
        decision=cohort.get('local_vol_eligibility',{})
        if gid not in result.models:
            reasons=sorted({str(r.get('reason')).replace('_',' ') for r in rows('selection')
                            if r.get('reason')})
            lines += [f'## Excluded inputs — {cohort.get("underlying")} ({cohort.get("valuation_date")})', '',
                f'{len(rows("selection"))} input records. **Blocked:** '
                +str(cohort.get('reason','no fitted surface')).replace('_',' ')+'. '
                'No fitted surface is available for this group.', '']
            if reasons:lines += ['Input exclusions: '+', '.join(reasons)+'.', '']
            continue
        lines += [f'## {cohort.get("underlying")} — {cohort.get("valuation_date")} ({cohort.get("route")})', '']
        lines += _table(['Decision','Next-step permission','Price-fit status'],[
            (decision.get('status','blocked'),'Allowed inside stated domain' if decision.get('allowed') else 'Not allowed',cohort['status'])])
        if decision.get('blockers'):
            lines += ['**Blocking reasons:** '+', '.join(str(x).replace('_',' ') for x in decision['blockers'])+'.', '']
        lines += ['### Essential checks', '']
        lines += _table(['Requirement','Result'],[(r['check'],r['passed']) for r in decision.get('checks',[])])
        metrics=rows('expiry_metrics');live=[r for r in metrics if r.get('practical_coverage_fraction') is not None]
        lines += [f'{sum(r["market_fit_pass"] for r in metrics)}/{len(metrics)} retained expiries pass their market-fit checks. '
            'Strict bid/ask coverage remains diagnostic; the configured practical band determines live quote coverage.', '']
        lines += ['### Warnings and limits', '']
        lines += ['- '+w for w in decision.get('warnings',[])]
        lines += ['- No extrapolation before the first expiry, '
                  'after the last expiry, or outside the domain below.',
                  '- Exact expiry knots are excluded. Every point in a numerical derivative stencil must '
                  'stay inside the SAME open maturity interval.', '']
        refits=[r for r in rows('derivative_refits') if r.get('category')!='timing']
        jumps=[r for r in rows('maturity_jumps') if r['metric']=='w_t']
        lines += _table(['Diagnostic','Result'],[
            ('Derivative refits',f'{sum(r["status"]=="passed" for r in refits)}/{len(refits)} pass'),
            ('Time-derivative jumps',f'{sum(r["status"]=="REVIEW_REQUIRED" for r in jumps)}/{len(jumps)} require review'),
            ('Unsupported holdout observations',cohort.get('unsupported_holdout_observations',0)),
            ('Timing evidence',cohort.get('timing_status','Not assessed'))])
        lines += ['### Supported conversion domain', '',
            'Dates identify expiry boundaries, not execution dates. T is measured from the saved reference '
            'timestamp; x=K/F(T), k=log(x). Both maturity endpoints are excluded. The K/F ranges below '
            'are conservative supported strips, not permission to use all strikes of each expiry.', '']
        lines += ['Reference timestamp: '+', '.join(decision.get('reference_timestamps',[]))+'.', '']
        lines += _table(['After expiry','Before expiry','T start / end (years)','Minimum K/F','Maximum K/F'],[
            (r['start_expiry'],r['end_expiry'],f'{r["start_years"]:.6g} / {r["end_years"]:.6g}',
             r['minimum_strike_over_forward'],r['maximum_strike_over_forward']) for r in decision.get('domain',[])])
        if not decision.get('allowed'):lines+=['These are candidate domain bounds only: this cohort is BLOCKED.', '']
        lines += ['### Price fit by expiry', '']
        lines += _table(['Expiry','Scaled RMSE','Practical coverage','Strict bid/ask coverage','Result'],[
            (r['expiration'],r['scaled_price_rmse'],
             f'{r["practical_coverage_fraction"]:.1%}' if r.get('practical_coverage_fraction') is not None else 'Not applicable',
             f'{r["inside_spread_fraction"]:.1%}' if r.get('inside_spread_fraction') is not None else 'Not applicable',r['market_fit_pass']) for r in metrics])
        if cfg['plots']:
            lines += [f'Interactive chart: [3D implied-volatility surface]({gid}_3d.html).', '', f'Charts: [IV surface]({gid}_surface.png), [expiry smiles]({gid}_smiles.png), '
                      f'[fit diagnostics]({gid}_diagnostics.png).', '']
    lines += ['## Policy and evidence', '',
        f'Price RMSE limit: **{q["maximum_scaled_price_rmse"]:g}**. '
        +(f'Practical live coverage: **{q["minimum_practical_coverage"]:.0%}**. '
          f'Each passing quote meets all three outside-spread limits: **{q["maximum_relative_outside_error"]:.1%}** '
          f'of premium, **{q["maximum_outside_half_spreads"]:g} half-spread(s)** and '
          f'**{100*q["maximum_outside_iv_error"]:g} IV percentage points**. '
          if cfg['schema_version']==3 else 'Legacy strict-spread policy applies. ')
        +'Historical cohorts use only the RMSE gate, without invented bid/ask intervals.', '',
        'Derivative scores use absolute and relative prototype allowances; exceeding the maturity-jump '
        'review band is a warning only when the JSON explicitly permits it. Failed or missing essential '
        'checks always block eligibility. Disabling stability does not produce a passed assessment.', '',
        'Machine-readable permission and exact bounds: [eligibility.json](eligibility.json). '
        'Every configuration value, code fingerprint and file checksum: [manifest.json](manifest.json).', '',
        'Detailed evidence: [selection](selection.csv), [coverage](coverage_checks.csv), '
        '[price errors](repricing.csv), [refit tests](stability.csv), '
        '[derivative comparisons](derivative_refit_metrics.csv), [maturity jumps](maturity_jumps.csv), '
        '[unsupported wings](stability_unsupported.csv). Additional point-level CSVs remain in this folder.', '',
        'Saved data only. No downloads, smoothing, Dupire calculation or new simulation.']
    return '\n'.join(lines)+'\n'


def _slug(text):
    return re.sub(r'\s','-',re.sub(r'[^\w\s-]','',text.lower()).strip())


def _inline(text):
    """Recognize simple generated emphasis, code and safe local links only."""
    pattern=r'(`[^`]+`|\*\*[^*]+\*\*|\[[^\]]+\]\([^)]+\))'
    pieces=[]
    for token in re.split(pattern,text):
        if token.startswith('`') and token.endswith('`'):
            pieces.append('<code>'+escape(token[1:-1])+'</code>')
        elif token.startswith('**') and token.endswith('**'):
            pieces.append('<strong>'+_inline(token[2:-2])+'</strong>')
        elif token.startswith('[') and (match:=re.fullmatch(r'\[([^\]]+)\]\(([^)]+)\)',token)):
            label,url=match.groups()
            # Fragments and single evidence filenames cannot navigate to a
            # remote service or execute a javascript:/data:/file: URL.
            safe=bool(re.fullmatch(r'#[\w-]+|[\w.-]+\.(?:csv|json|md|png|html)',url))
            pieces.append(f'<a href="{escape(url,quote=True)}">{escape(label)}</a>' if safe else escape(label))
        else:pieces.append(escape(token))
    return ''.join(pieces)


def _cell(text):
    text=text.strip().replace('\\|','|')
    labels={'PASS':'pass','PASSED':'pass','FAIL':'fail','INCOMPLETE':'warn','NOT_RUN':'muted',
            'NOT IMPLEMENTED':'muted','Not implemented':'muted','NOT ACCEPTED':'warn',
            'PASS WITH LIMITATIONS':'warn','DIAGNOSTIC':'muted','BLOCKED':'fail',
            'blocked':'fail','eligible_with_warnings':'warn','eligible':'pass',
            'REVIEW_REQUIRED':'warn','review_required':'warn','passed':'pass','incomplete':'warn'}
    if text in labels:
        label={'eligible_with_warnings':'Eligible with warnings','eligible':'Eligible','blocked':'Blocked',
               'review_required':'Review required','REVIEW_REQUIRED':'Review required',
               'passed':'Passed','incomplete':'Incomplete'}.get(text,text)
        return f'<span class="badge {labels[text]}">{escape(label)}</span>'
    readable={'supported_domain':'Supported domain','all_quotes':'All requested quotes',
        'not_assessable_from_single_capture':'Not assessable from a single capture',
        'unvalidated_wing':'Unvalidated wing','unvalidated_maturity':'Unvalidated maturity',
        'not_tested':'Not tested','insufficient_coverage':'Insufficient coverage','checked':'Checked'}
    text=readable.get(text,text)
    return _inline(text)


def _cells(line):
    return re.split(r'(?<!\\)\|',line.strip()[1:-1])


def render_html(markdown,result):
    """Render the same decisions and evidence as report.md with readable tables."""
    lines=markdown.splitlines();body=[];toc=[];i=0;ids={}
    while i<len(lines):
        line=lines[i]
        if not line.strip():i+=1;continue
        if line.startswith('#'):
            match=re.match(r'^(#{1,6})\s+(.+)$',line)
            if match:
                level=len(match[1]);text=match[2];base=_slug(text)
                number=ids.get(base,0);ids[base]=number+1
                anchor=base+(f'-{number}' if number else '')
                body.append(f'<h{level} id="{escape(anchor)}">{_inline(text)}</h{level}>')
                if level==2:toc.append((anchor,text))
                i+=1;continue
        if line.startswith('|') and i+1<len(lines) and re.fullmatch(r'[| :\-]+',lines[i+1]):
            headers=_cells(line);i+=2
            body.append('<div class="table-wrap"><table><thead><tr>'+''.join('<th scope="col">'+_inline(h.strip())+'</th>' for h in headers)+'</tr></thead><tbody>')
            while i<len(lines) and lines[i].startswith('|'):
                cells=_cells(lines[i]);body.append('<tr>'+''.join('<td>'+_cell(c)+'</td>' for c in cells)+'</tr>');i+=1
            body.append('</tbody></table></div>');continue
        bullet=re.match(r'^(?:- |\d+\. )(.+)',line)
        if bullet:
            numbered=not line.startswith('- ');tag='ol' if numbered else 'ul'
            start=int(line.split('.')[0]) if numbered else None
            body.append(f'<{tag}'+(f' start="{start}"' if numbered else '')+'>')
            while i<len(lines):
                match=re.match(r'^(?:\d+\. )(.+)' if numbered else r'^- (.+)',lines[i])
                if not match:break
                content=match[1];i+=1
                while i<len(lines) and lines[i].startswith('  '):content+=' '+lines[i].strip();i+=1
                body.append('<li>'+_inline(content)+'</li>')
                while i<len(lines) and not lines[i].strip():i+=1
            body.append(f'</{tag}>');continue
        paragraph=[line];i+=1
        while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','- ')):
            paragraph.append(lines[i]);i+=1
        body.append('<p>'+_inline(' '.join(paragraph))+'</p>')
    figures=[]  # Charts are linked in the concise report, not embedded repeatedly.
    nav=''.join(f'<a href="#{escape(anchor)}">{escape(title)}</a>' for anchor,title in toc)
    # Inline styles make the report work when opened directly from Explorer.
    # No active script or remote resource is included; CSP reinforces this.
    return '''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; img-src 'self' data:; base-uri 'none'; form-action 'none'">
<title>Surface input eligibility</title>
<style>
:root{color-scheme:light;--ink:#183149;--muted:#5c7083;--line:#dbe4ed;--nav:#102c43}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:24px}
body{margin:0;background:#eef3f7;color:var(--ink);font:16px/1.65 "Segoe UI",Arial,sans-serif}
aside{position:fixed;inset:0 auto 0 0;width:240px;padding:28px 22px;background:var(--nav);color:#fff;overflow:auto}
aside strong{display:block;font-size:20px;line-height:1.3;margin-bottom:22px}aside a{display:block;color:#d7e7f5;text-decoration:none;padding:9px 0;border-bottom:1px solid #345065;font-size:14px}
main{margin:32px 32px 64px 272px;max-width:1220px;background:white;padding:34px 44px 60px;border:1px solid var(--line);border-radius:12px}
h1{font-size:32px;line-height:1.2;margin:0 0 24px;letter-spacing:-.6px}h2{font-size:24px;margin:42px 0 18px;padding-top:16px;border-top:2px solid var(--line)}
h3{font-size:19px;margin:28px 0 12px}h4{font-size:17px;margin:24px 0 10px}p{max-width:95ch;margin:12px 0 18px}li{margin:8px 0}a{color:#075e9e}code{background:#edf3f8;padding:2px 5px;border-radius:4px;overflow-wrap:anywhere;font-size:.9em}
.notice{padding:16px 20px;border-left:4px solid #c47b11;background:#fff7e8;border-radius:6px;margin:0 0 24px}.notice strong{display:block}
.table-wrap{overflow-x:auto;margin:16px 0 26px;border:1px solid var(--line);border-radius:7px}table{border-collapse:collapse;width:100%;font-size:14px;line-height:1.45}
th{text-align:left;background:#eaf1f7;color:#23435e;padding:12px 14px;vertical-align:bottom}td{padding:10px 14px;border-top:1px solid var(--line);vertical-align:top;overflow-wrap:anywhere}tbody tr:nth-child(even){background:#f7fafc}tbody tr:hover{background:#eef6fc}
.badge{display:inline-block;font-size:11px;font-weight:700;letter-spacing:.3px;padding:4px 8px;border-radius:4px;white-space:nowrap}.pass{color:#146239;background:#e5f4eb}.fail{color:#a32927;background:#fdebea}.warn{color:#87550b;background:#fff0ce}.muted{color:#506170;background:#eaf0f4}
figure{margin:24px 0}img{max-width:100%;height:auto;border:1px solid var(--line);border-radius:6px}figcaption{color:var(--muted);font-size:14px}footer{margin-top:35px;color:var(--muted);font-size:13px}
@media(max-width:960px){aside{position:static;width:auto}aside a{display:inline-block;margin-right:16px}main{margin:16px;padding:24px}h1{font-size:27px}}
@media print{aside{display:none}main{margin:0;border:0;padding:0;max-width:none}body{background:white;font-size:11pt}.table-wrap{overflow:visible}thead{display:table-header-group}tr,figure{break-inside:avoid}h2,h3,h4{break-after:avoid}a{color:inherit}.notice{background:white}}
</style></head><body><aside><strong>Surface review</strong>'''+nav+'''</aside><main>
<div class="notice"><strong>Prototype validation — bounded domain</strong>A bounded permission decision for future local-volatility input. No Dupire calculation or full-simulation approval.</div>
'''+''.join(body)+''.join(figures)+'''<footer>Generated from the same recorded results as <a href="report.md">report.md</a>. <a href="manifest.json">Integrity manifest</a>. All resources are local.</footer></main></body></html>
'''


# Scientific figures (opened separately from the concise report)
def plot_surfaces(result,folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for gid,model in result.models.items():
        rows=[r for r in result.tables['repricing'] if r['group_id']==gid]
        status=next(r for r in result.tables['cohort_status'] if r['group_id']==gid)
        label=f"{status['underlying']} | {status['valuation_date']} | {status['status']}"
        expiries=sorted({r['expiration'] for r in rows})
        fig,axes=plt.subplots(math.ceil(len(expiries)/3),3,figsize=(15,3.2*math.ceil(len(expiries)/3)),squeeze=False)
        for ax,expiry in zip(axes.flat,expiries):
            group=sorted([r for r in rows if r['expiration']==expiry],key=lambda r:r['k'])
            t=group[0]['years'];k=np.array([r['k'] for r in group])
            ax.scatter(k,[100*r['observed_iv'] for r in group],s=8,color='#155e75',label='Calculated IV')
            line=np.linspace(min(k),max(k),150)
            ax.plot(line,100*model.evaluate(line,t)['iv'],color='#d97706',label=model.model_type+' fit')
            for r in group:
                if r.get('bid_iv') is not None and r.get('ask_iv') is not None:
                    ax.plot([r['k'],r['k']],[100*float(r['bid_iv']),100*float(r['ask_iv'])],alpha=.15,color='#155e75')
            ax.set_title(expiry);ax.set_xlabel('log(K/F)');ax.set_ylabel('Annualized IV (%)');ax.grid(alpha=.2)
        for ax in list(axes.flat)[len(expiries):]:ax.set_visible(False)
        axes.flat[0].legend(fontsize=8)
        fig.suptitle(label+'\nStable-smile intraday/close approximation; vertical bars show available bid/ask IVs',fontsize=12)
        fig.tight_layout(rect=(0,0,1,.94));fig.savefig(folder/(gid+'_smiles.png'),dpi=130);plt.close(fig)

        fig=plt.figure(figsize=(11,7));ax=fig.add_subplot(projection='3d')
        for t in np.linspace(model.times[0],model.times[-1],35):
            lo,hi=model.domain(t)
            k=np.linspace(lo,hi,80)
            ax.plot(k,np.full_like(k,t),100*model.evaluate(k,t)['iv'],color='#155e75',alpha=.65,lw=.8)
        ax.set_xlabel('log(K/F)');ax.set_ylabel('Years to expiry');ax.set_zlabel('Annualized IV (%)')
        ax.set_title(label+'\nObserved maturity/strike envelope only; no extrapolation')
        fig.tight_layout();fig.savefig(folder/(gid+'_surface.png'),dpi=130);plt.close(fig)

        conditioning=[r for r in result.tables.get('quote_diagnostics',[]) if r['group_id']==gid]
        practical=result.manifest['identity']['config']['schema_version']==3
        panel_rows=1+bool(conditioning)+bool(practical)
        fig,axes=plt.subplots(panel_rows,3,figsize=(16,4.5*panel_rows),squeeze=False)
        axes=axes.ravel()
        axes[0].plot(model.times,model.theta,'o-',color='#155e75')
        axes[0].set(xlabel='Years to expiry',ylabel='ATM total variance',title='Monotone ATM term structure')
        pts=axes[1].scatter([r['k'] for r in rows],[r['scaled_price_error'] for r in rows],
            c=[r['years'] for r in rows],cmap='viridis',s=8)
        axes[1].axhline(0,color='black',lw=.7);axes[1].axhline(1,color='gray',ls='--');axes[1].axhline(-1,color='gray',ls='--')
        axes[1].set(xlabel='log(K/F)',ylabel='Price error / declared error scale',title='Repricing residuals')
        fig.colorbar(pts,ax=axes[1],label='Years')
        metrics=[r for r in result.tables['expiry_metrics'] if r['group_id']==gid]
        axes[2].bar(range(len(metrics)),[r['scaled_price_rmse'] for r in metrics],color='#155e75')
        threshold=result.manifest['identity']['config']['quality']['maximum_scaled_price_rmse']
        axes[2].axhline(threshold,color='#b91c1c',ls='--',label='Acceptance limit')
        axes[2].set_xticks(range(len(metrics)),[r['expiration'] for r in metrics],rotation=75,fontsize=7)
        axes[2].set(ylabel='Scaled price RMSE',title='Errors by expiry');axes[2].legend(fontsize=8)
        if conditioning:
            from datetime import datetime
            quoted=[r for r in conditioning if r['spread_over_vega'] is not None]
            axes[3].scatter([r['k'] for r in quoted],[100*r['spread_over_vega'] for r in quoted],s=8,color='#155e75')
            axes[3].set(xlabel='log(K/F)',ylabel='Full spread / Vega (volatility points)',title='IV uncertainty from the spread')
            stamps=np.array([datetime.fromisoformat(r['capture_timestamp']).timestamp() for r in conditioning])
            points=axes[4].scatter((stamps-min(stamps))/60,[r['scaled_price_error'] for r in conditioning],
                c=[r['k'] for r in conditioning],cmap='coolwarm',s=8)
            axes[4].set(xlabel='Minutes from first capture',ylabel='Scaled price error',title='Timing and strike remain confounded')
            fig.colorbar(points,ax=axes[4],label='log(K/F)')
            comparison=[r for r in result.tables.get('model_comparison',[]) if r['group_id']==gid]
            for offset,name,color in [(-.18,'ssvi','#94a3b8'),(.18,'raw_svi','#155e75')]:
                by_expiry={r['expiration']:r['scaled_price_rmse'] for r in comparison if r['model']==name}
                axes[5].bar(np.arange(len(metrics))+offset,[by_expiry.get(r['expiration'],np.nan) for r in metrics],width=.36,label=name,color=color)
            axes[5].set_xticks(range(len(metrics)),[r['expiration'] for r in metrics],rotation=75,fontsize=7)
            axes[5].set(ylabel='Scaled price RMSE',title='Same quotes: model comparison');axes[5].legend(fontsize=8)
        if practical:
            # Optional research panels may be absent. Pack the essential
            # price-coverage panels immediately after the panels we did draw.
            offset=6 if conditioning else 3
            practical_axes=axes[offset:offset+3]
            quality=result.manifest['identity']['config']['quality']
            live=[r for r in metrics if r['practical_coverage_fraction'] is not None]
            if live:
                x=np.arange(len(live));labels=[r['expiration'] for r in live]
                practical_axes[0].bar(x-.18,[100*r['inside_spread_fraction'] for r in live],width=.36,label='Strict (diagnostic)',color='#94a3b8')
                practical_axes[0].bar(x+.18,[100*r['practical_coverage_fraction'] for r in live],width=.36,label='Practical (acceptance)',color='#155e75')
                practical_axes[0].axhline(100*quality['minimum_practical_coverage'],ls='--',color='#b91c1c')
                practical_axes[0].set(ylabel='Coverage (%)',title='Strict versus practical coverage');practical_axes[0].legend(fontsize=8)
                for ax,key,scale,limit,title,ylabel in [
                    (practical_axes[1],'p95_relative_outside_spread_error',100,quality['maximum_relative_outside_error'],'P95 outside-spread price error','% of midpoint'),
                    (practical_axes[2],'p95_outside_iv_error',100,quality['maximum_outside_iv_error'],'P95 outside-IV error (available bounds)','Volatility percentage points')]:
                    ax.bar(x,[scale*r[key] if r[key] is not None else np.nan for r in live],color='#155e75')
                    ax.axhline(scale*limit,ls='--',color='#b91c1c',label='Per-quote limit (not a P95 gate)')
                    ax.set(title=title,ylabel=ylabel);ax.legend(fontsize=7)
                for ax in practical_axes:ax.set_xticks(x,labels,rotation=75,fontsize=7)
            else:
                for ax in practical_axes:
                    ax.text(.5,.5,'Historical data: interval coverage not applicable',ha='center',va='center',transform=ax.transAxes,fontsize=8)
                    ax.set_axis_off()
        for ax in axes:ax.grid(alpha=.15)
        fig.suptitle(label);fig.tight_layout();fig.savefig(folder/(gid+'_diagnostics.png'),dpi=130);plt.close(fig)
