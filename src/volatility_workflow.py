"""Human-readable overview of the existing volatility pipeline.

Presentation only: no fitting, quote filtering, tolerance changes or downloads.
Raw stage tables remain the authority. Rejected input groups are explained once,
not displayed as independent failures at every downstream stage.
"""
from collections import Counter
from contextlib import contextmanager, redirect_stdout
from html import escape
from io import StringIO
from pathlib import Path
from time import monotonic
import hashlib
import json
import os
import sys
from tempfile import TemporaryDirectory

STAGES = ("Option calibration", "Implied surface", "Local volatility / PDE", "Risk-neutral Monte Carlo")
GATES = {"fit_pass": "surface-price agreement", "pde_pass": "PDE-price agreement",
         "convergence_pass": "time-step convergence", "precision_pass": "sampling precision",
         "forward_pass": "forward expectation"}


class Progress:
    """Keep every solver message, but show at most one heartbeat per 30 seconds.

    This is a UI refresh interval, not a numerical/model parameter. Algorithms
    retain their normal callbacks and do exactly the same calculations.
    """
    def __init__(self, label):
        self.label, self.messages, self.last = label, [], monotonic()

    def __call__(self, message):
        self.messages.append(str(message))
        if monotonic() - self.last >= 30:
            print(f"  {self.label}: working; detailed progress is retained in the execution log.")
            self.last = monotonic()


@contextmanager
def capture_console():
    """Tee console output, including controlled exceptions, into this run's log."""
    buffer, original = StringIO(), sys.stdout
    class Tee:
        def write(self, value):
            original.write(value)
            return buffer.write(value)
        def flush(self):
            original.flush()
    with redirect_stdout(Tee()):
        yield buffer


def label(row):
    # Group identity is shown because the same underlying can have different
    # dates, routes, settlement conventions and quote types.
    return " / ".join(str(row[k]) for k in ("underlying", "currency", "valuation_date", "route", "group_id") if row.get(k))


def mc_checks(result, gid):
    quotes = [r for r in result.tables.get("quotes", []) if r["group_id"] == gid]
    forwards = [r for r in result.tables.get("forwards", []) if r["group_id"] == gid]
    rows = [dict(check=GATES[k], passed=sum(r.get(k) is True for r in quotes), total=len(quotes))
            for k in GATES if k != "forward_pass"]
    rows.append(dict(check=GATES["forward_pass"], passed=sum(r.get("passed") is True for r in forwards), total=len(forwards)))
    return rows


def show_stage(stage, result, target, progress=None):
    """Short console status; never turn missing evidence into a successful test."""
    result.workflow_target = Path(target)
    result.workflow_messages = progress.messages if progress else []
    tables, summary = result.tables, result.manifest["summary"]
    if stage == 0:
        n, accepted = summary["option_records"], summary["iv_calculated"]
        print(f"  IV calculation: {accepted:,}/{n:,} records accepted; {n-accepted:,} excluded (reasons in overview).")
        print("  Observation dates: " + ", ".join(summary["valuation_dates"]))
        for curve in summary["curve_status"]:
            if curve["status"] != "bootstrapped": print(f"  Curve needs attention: {curve}")
    elif stage == 1:
        rows=tables["cohort_status"]
        fitted=[r for r in rows if r.get("model")]
        print(f"  Surface: {len(fitted)} fitted group(s); {len(rows)-len(fitted)} group(s) without a fit (root causes in overview).")
        for row in fitted:
            allowed=row["local_vol_eligibility"]["allowed"]
            print(f"  {label(row)}: {row.get('selected_observations', 0):,} selected quotes, {row.get('expiries', 0)} expiries; "
                  f"{'eligible with documented assumptions' if allowed else 'BLOCKED: '+', '.join(row['local_vol_eligibility']['blockers'])}.")
    elif stage == 2:
        for row in tables["cohorts"]:
            if row.get("reason") == "surface_not_eligible": continue  # upstream cause is already retained
            print(f"  {label(row)}: Dupire {row['status']}; {row.get('validation_passes',0)}/{row.get('requested_points',0)} grid checks.")
        for row in tables["pricing_summary"]:
            if row.get("reason") == "validated_local_volatility_unavailable": continue
            print(f"  PDE repricing: {label(row)}; {row['status']}; {row.get('assessed_quotes',0):,} assessed quotes.")
        for row in tables["state_mapping"]:
            if row.get("reason") != "validated_pricing_required_for_state_mapping" and row["status"] == "unavailable":
                print(f"  Date/spot interface blocked: {label(row)}; {row.get('reason')}")
    else:
        for row in tables["cohorts"]:
            if row.get("reason") == "validated_state_evaluator_unavailable": continue
            print(f"  {label(row)}: {row['status']}; {row.get('paths',0):,} paths.")
            if row.get("start_date"):
                print(f"  MC start (COB + 1): {row['start_date']}; reference instant: {row['start_timestamp']}")
            if row.get("quotes"):
                checks=mc_checks(result,row["group_id"])
                print("  " + "; ".join(f"{r['check']} {r['passed']}/{r['total']}" for r in checks))
                failed=[r["check"] for r in checks if r["passed"] != r["total"] or not r["total"]]
                if failed: print("  Needs attention: " + ", ".join(failed) + ".")
            elif row.get("reason"): print("  Reason: " + row["reason"])


def build_summary(results):
    """Join by group_id, never by ticker alone; trace the first blocking stage.

    Preserve the full original statuses under stage_statuses. An excluded-input
    group is not a second failed fitted surface. Real downstream failures on a
    fitted group remain prominent and are not collapsed into input exclusions.
    """
    calibration, surface, local, mc = results
    stages=[]
    for name,result in zip(STAGES,results):
        stages.append(dict(stage=name, produced=result is not None,
            result_id=result.run_id if result else None,
            summary=result.manifest["summary"] if result else {},
            config=result.manifest["identity"].get("config",{}) if result else {}))
    summary=dict(stages=stages, groups=[], excluded_input_groups=[], calibration_exclusions={},
                 surface_exclusions={}, portfolio_simulation=False)
    if calibration:
        summary["calibration_exclusions"]=calibration.manifest["summary"]["unavailable_reasons"]
    if surface:
        summary["surface_exclusions"]=dict(Counter(r.get("reason") or "unspecified" for r in surface.tables["selection"] if not r.get("selected")))
        indexed=[{r["group_id"]:r for r in table} for table in (
            local.tables["cohorts"] if local else [], local.tables["pricing_summary"] if local else [],
            local.tables["state_mapping"] if local else [], mc.tables["cohorts"] if mc else [])]
        for row in surface.tables["cohort_status"]:
            gid=row["group_id"]; later=[d.get(gid,{}) for d in indexed]
            entry=dict(label=label(row),group_id=gid,stage_statuses=[row]+later,
                       checks=mc_checks(mc,gid) if mc and gid in indexed[-1] else [])
            input_only=row.get("reason")=="no_usable_iv_observations"
            selection=[r for r in surface.tables["selection"] if r["group_id"]==gid]
            entry["input_records"]=len(selection)
            entry["root_cause"]=row.get("reason") if not row.get("model") else None
            if input_only:
                entry["explanation"]="No accepted individual IVs in this input group. Later skipped rows are consequences of this one input rejection, not additional model failures."
                summary["excluded_input_groups"].append(entry)
                continue
            entry["selected_quotes"]=row.get("selected_observations",0)
            entry["expiries"]=row.get("expiries",0)
            entry["permission"]=row.get("local_vol_eligibility",{})
            entry["grid_points"]=later[0].get("requested_points",0)
            entry["pricing_quotes"]=later[1].get("assessed_quotes",0)
            entry["pricing_exclusions"]=later[1].get("excluded_quotes",0)
            entry["timing_diagnostics"]=[{k:r.get(k) for k in ("check","status","decision","required")} for r in surface.tables.get("stability",[]) if r["group_id"]==gid and r.get("category")=="timing"]
            if not entry["permission"].get("allowed"):
                entry["outcome"]="Blocked at implied surface"
                entry["root_cause"]=(", ".join(entry["permission"].get("blockers",[]))
                        if row.get("model") else row.get("reason")) or "missing eligibility evidence"
            else:
                expected=("validated_on_supported_grid","validated_with_boundary_assumptions","prepared_with_boundary_assumptions","validated_with_boundary_assumptions")
                names=("Dupire conversion","PDE repricing","date/spot mapping","risk-neutral Monte Carlo")
                entry["outcome"]="Validated with assumptions"
                for name,state,good in zip(names,later,expected):
                    if state.get("status") != good:
                        entry["outcome"]=("Completed; review required: " if state.get("status")=="review_required" else "Not completed: ") + name
                        if name=="risk-neutral Monte Carlo" and state.get("quotes"):
                            failed=[r["check"] for r in entry["checks"] if r["passed"]!=r["total"] or not r["total"]]
                            entry["root_cause"]=(", ".join(failed) or state.get("reason") or "missing completion evidence")
                        else:entry["root_cause"]=state.get("reason") or "stage result not produced; see execution log"
                        break
            summary["groups"].append(entry)
    return summary


def render(summary, links, guide=None):
    """Self-contained HTML: decision first; technical evidence expandable."""
    e=lambda x:escape(str(x))
    def table(rows, fields):
        if not rows:return "<p>Not applicable / no records.</p>"
        return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+e(k.replace('_',' '))+'</th>' for k in fields)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+e(r.get(k,""))+'</td>' for k in fields)+'</tr>' for r in rows)+'</tbody></table></div>'
    body=['<h1>Volatility workflow overview</h1>',
        '<p>Saved-data study: individual IVs → implied surface → Dupire and boundary completion → PDE validation → risk-neutral underlying Monte Carlo.</p>',
        '<p><strong>Scope:</strong> this is an underlying option-pricing validation, not a portfolio or real-world forecast. A completed calculation can still require review.</p>']
    if guide:body.append('<p><a href="'+e(guide)+'">Detailed implementation and mathematics guide</a></p>')
    body.append('<h2>Results to review</h2>')
    for group in summary['groups']:
        body += ['<section><h3>'+e(group['label'])+'</h3><p><strong>'+e(group['outcome'])+'</strong></p>',
                 '<p>'+e(group.get('root_cause') or 'All required downstream checks passed; approximation and domain limitations still apply.')+'</p>',
                 table([dict(selected_quotes=group['selected_quotes'],expiries=group['expiries'],Dupire_grid_points=group['grid_points'],PDE_MC_quotes=group['pricing_quotes'],outside_supported_strips=group['pricing_exclusions'])],['selected_quotes','expiries','Dupire_grid_points','PDE_MC_quotes','outside_supported_strips']),
                 table(group['checks'],['check','passed','total'])]
        perm=group['permission']
        body.append('<p>Supported Dupire domain: '+e(len(perm.get('domain',[])))+' open expiry intervals. Computational wings and short-end continuation are assumptions, not extra observations.</p>')
        body.append('<ul>'+''.join('<li>'+e(w)+'</li>' for w in perm.get('warnings',[]))+'</ul>')
        if group['timing_diagnostics']:body.append('<details><summary>Capture-window diagnostics (not a synchronization guarantee)</summary>'+table(group['timing_diagnostics'],['check','status','decision','required'])+'</details>')
        body.append('</section>')
    if not summary['groups']:
        body.append('<p>No fitted group reached this overview. Check the input exclusions and stage log; no successful calibration is implied.</p>')
    body += ['<h2>Why some records did not proceed</h2><p>These counts describe different stages; do not add them together. Surface exclusions include IV records rejected earlier.</p>',
        table([dict(reason=k,records=v) for k,v in summary['calibration_exclusions'].items()],['reason','records']),
        '<p>Input-only groups: '+e(len(summary['excluded_input_groups']))+'. They remain in the detailed stage tables for traceability, but are not repeated here as separate local-volatility/PDE/MC failures.</p>',
        table([dict(group=g['label'],records=g['input_records'],reason=g['root_cause']) for g in summary['excluded_input_groups']],['group','records','reason']),
        '<details><summary>Surface selection exclusions</summary>'+table([dict(reason=k,records=v) for k,v in summary['surface_exclusions'].items()],['reason','records'])+'</details>',
        '<h2>Stages and detailed reports</h2><ol>']
    for item,link in zip(summary['stages'],links):
        body.append('<li>'+e(item['stage'])+': '+('<a href="'+e(link)+'">open detailed report</a>' if link else 'not produced; see execution log')+'</li>')
    body+=['</ol><p><a href="progress.log">Full execution log</a> · <a href="summary.json">Structured overview and exact settings</a></p>',
        '<h2>How to read the statuses</h2><ul><li>Excluded input: no usable observation; not a numerical model failure.</li><li>Blocked: a required input or validation is missing or failed.</li><li>Review required: output exists, but one or more acceptance conditions did not pass.</li><li>Validated with assumptions: checks passed within their documented domain and numerical resolution.</li></ul>',
        '<p>Every assessed quote and forward must pass its MC gates; there is no percentage quorum. Sampling precision concerns Monte Carlo uncertainty, not a new failed surface fit. Passing repricing cannot prove economic realism of unobserved tails. Historical statistics/portfolio simulation remain a separate workflow.</p>']
    for group in summary['groups']:
        body.append('<details><summary>Full status and supported domain: '+e(group['label'])+'</summary><pre>'+e(json.dumps(group,indent=2,ensure_ascii=False))+'</pre></details>')
    return '<!doctype html><html lang="en"><meta charset="utf-8"><title>Volatility workflow</title><style>body{font:16px/1.55 system-ui;max-width:1100px;margin:32px auto;padding:0 24px;color:#203047}h1,h2{color:#173b67}section{padding:18px;background:#f2f6fa;margin:20px 0;border-left:4px solid #366da0}table{border-collapse:collapse;width:100%}td,th{padding:10px;border:1px solid #ccd6e0;text-align:left}th{background:#eaf0f6}.scroll,pre{overflow:auto}details{margin:18px 0}summary{cursor:pointer;font-weight:600}pre{font-size:13px}a{color:#145dad}</style>'+''.join(body)+'</html>'


def save_overview(destination, results, console_text):
    """Write a separate, content-addressed overview; never rewrite stage data.

    Changing presentation cannot invalidate a mathematical calibration. The
    report identity contains the presentation code and all numerical stage IDs.
    Corrupt existing outputs fail verification rather than being overwritten.
    """
    summary=build_summary(results)
    root=Path(destination).resolve()
    identity=dict(stages=[r.run_id if r else None for r in results],summary=summary,
                  code_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  console_hash=hashlib.sha256(console_text.encode("utf-8")).hexdigest())
    packed=json.dumps(identity,sort_keys=True,ensure_ascii=False,allow_nan=False).encode('utf-8')
    folder=root/'volatility_review'/hashlib.sha256(packed).hexdigest()[:16]
    if (folder.is_symlink() or not folder.resolve().is_relative_to(root)
            or folder.resolve().parent != (root/'volatility_review').resolve()):
        raise ValueError('unsafe_volatility_overview_path')
    links=[]
    for result in results:
        if result is None:links.append(None);continue
        target=result.workflow_target
        report=target/('report.html' if (target/'report.html').exists() else 'report.md')
        links.append(Path(os.path.relpath(report,folder)).as_posix())
    details=[console_text]
    for stage,result in zip(STAGES,results):
        details += [stage]+(getattr(result,'workflow_messages',[]) if result else [])
    guide=Path(__file__).resolve().parents[1]/'docs'/'volatility_methodology.html'
    guide_link=Path(os.path.relpath(guide,folder)).as_posix() if guide.exists() else None
    files={'summary.json':json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False),
           'report.html':render(summary,links,guide_link),'progress.log':'\n'.join(details)}
    if folder.exists():
        manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        if (not isinstance(manifest,dict) or manifest.get('identity')!=identity
                or not isinstance(manifest.get('files'),dict) or set(manifest['files'])!=set(files)):raise ValueError('volatility_overview_identity_mismatch')
        for name,digest in manifest['files'].items():
            if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=digest:raise ValueError('volatility_overview_output_corrupt:'+name)
        return folder
    folder.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix='.overview-',dir=folder.parent) as temporary:
        work=Path(temporary); hashes={}
        for name,content in files.items():
            (work/name).write_text(content,encoding='utf-8')
            hashes[name]=hashlib.sha256((work/name).read_bytes()).hexdigest()
        (work/'manifest.json').write_text(json.dumps(dict(identity=identity,files=hashes),indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
        folder.mkdir()
        for path in work.iterdir():path.replace(folder/path.name)
    return folder
