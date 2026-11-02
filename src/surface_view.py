"""Self-contained interactive 3D surfaces: no CDN, browser network or new dependency.

Separate meshes preserve maturity gaps and supported strips. The canvas uses
perspective projection and depth-sorted faces, with orbit/zoom and point hover.
Plotting never interpolates across strips or changes numerical model values.
"""
from html import escape
import json
import math
import numpy as np


def render_surface_3d(title,patches,observations=None,notice='Supported domain only; gaps are intentional.'):
    # JSON inside an inert script element must not permit </script> injection.
    payload=json.dumps(dict(patches=patches,observations=observations or []),allow_nan=False,
        separators=(',',':')).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'''+escape(title)+'''</title>
<style>body{margin:0;background:#eff4f8;color:#17354c;font:16px/1.5 system-ui}main{max-width:1250px;margin:auto;padding:24px}h1{font-size:26px;margin:0 0 10px}.note{background:#fff2d8;padding:12px 16px;border-left:4px solid #d48b18}canvas{width:100%;height:650px;background:#fff;border:1px solid #d2deea;border-radius:12px;touch-action:none}button,label{margin:8px 16px 8px 0;padding:8px;font:inherit}button{background:#163c59;color:white;border:0;border-radius:5px;cursor:pointer}#tip{position:fixed;pointer-events:none;background:#132e45;color:white;padding:8px 12px;border-radius:5px;display:none;white-space:pre-line;font-size:13px}small{color:#4c657b}</style>
<main><h1>'''+escape(title)+'''</h1><p class="note">'''+escape(notice)+'''</p><button id="reset">Reset view</button><label><input type="checkbox" id="wire"> Wireframe</label><label><input type="checkbox" id="points" checked> Observed IVs</label><p><small>Drag to rotate · Scroll to zoom · Hover near a grid point for values. T: years from observation; X: underlying/forward (strike/forward for IV); Z: annualized volatility (%).</small></p><canvas id="view" aria-label="Interactive three-dimensional volatility surface"></canvas><div id="legend"></div><div id="tip"></div></main>
<script id="surface-data" type="application/json">'''+payload+r'''</script><script>
'use strict';
const data=JSON.parse(document.getElementById('surface-data').textContent),canvas=document.getElementById('view'),ctx=canvas.getContext('2d'),tip=document.getElementById('tip');
const all=data.patches.flatMap(p=>p.rows.flat()),obs=data.observations;
const values=all.concat(obs), min=[0,0,0],max=[1,1,1];
for(let j=0;j<3;j++){min[j]=Math.min(...values.map(p=>p[j]));max[j]=Math.max(...values.map(p=>p[j]));if(max[j]===min[j])max[j]=min[j]+1;}
// Zero on the volatility axis makes the high downside ridge visible honestly.
min[2]=0;let yaw=-.8,pitch=.65,zoom=1,drag=null,projected=[];
function colour(z){let u=(z-min[2])/(max[2]-min[2]);return `hsl(${235-210*u},68%,${43+8*u}%)`;}
function project(p){let a=(p[0]-min[0])/(max[0]-min[0])*2-1,b=(p[1]-min[1])/(max[1]-min[1])*2-1,z=(p[2]-min[2])/(max[2]-min[2])*1.8-.9;
let x=a*Math.cos(yaw)-b*Math.sin(yaw),v=a*Math.sin(yaw)+b*Math.cos(yaw),y=v*Math.sin(pitch)-z*Math.cos(pitch),depth=v*Math.cos(pitch)+z*Math.sin(pitch),scale=Math.min(canvas.width,canvas.height)*.28*zoom/(1+depth*.14);
return [canvas.width*.5+x*scale,canvas.height*.52+y*scale,depth];}
function line(a,b,text){let p=project(a),q=project(b);ctx.beginPath();ctx.moveTo(p[0],p[1]);ctx.lineTo(q[0],q[1]);ctx.stroke();if(text)ctx.fillText(text,q[0]+5,q[1]-5);}
function draw(){let rect=canvas.getBoundingClientRect();canvas.width=rect.width*devicePixelRatio;canvas.height=rect.height*devicePixelRatio;ctx.clearRect(0,0,canvas.width,canvas.height);ctx.font=`${12*devicePixelRatio}px system-ui`;ctx.fillStyle='#29485f';ctx.strokeStyle='#d9e3ec';ctx.lineWidth=devicePixelRatio;
for(let i=0;i<=4;i++){let t=min[0]+(max[0]-min[0])*i/4,x=min[1]+(max[1]-min[1])*i/4;line([t,min[1],0],[t,max[1],0]);line([min[0],x,0],[max[0],x,0]);}
ctx.strokeStyle='#54758d';line([min[0],min[1],0],[max[0],min[1],0],'T (years)');line([min[0],min[1],0],[min[0],max[1],0],'X / forward');line([min[0],min[1],0],[min[0],min[1],max[2]],'Vol (%)');
for(let i=0;i<=4;i++){let f=i/4;for(let j=0;j<3;j++){let p=[min[0],min[1],0];p[j]=min[j]+(max[j]-min[j])*f;let s=project(p);ctx.fillText(p[j].toFixed(j===2?1:2),s[0]-28*devicePixelRatio,s[1]+14*devicePixelRatio);}}
let faces=[];projected=[];for(const patch of data.patches){let rows=patch.rows;for(const row of rows)for(const p of row)projected.push({screen:project(p),p,label:patch.label});for(let i=0;i<rows.length-1;i++)for(let j=0;j<rows[i].length-1;j++){let pts=[rows[i][j],rows[i+1][j],rows[i+1][j+1],rows[i][j+1]],s=pts.map(project);faces.push({s,depth:s.reduce((a,p)=>a+p[2],0)/4,z:pts.reduce((a,p)=>a+p[2],0)/4});}}
faces.sort((a,b)=>b.depth-a.depth);for(const f of faces){ctx.beginPath();ctx.moveTo(f.s[0][0],f.s[0][1]);for(const p of f.s.slice(1))ctx.lineTo(p[0],p[1]);ctx.closePath();ctx.fillStyle=colour(f.z);ctx.fill();if(document.getElementById('wire').checked){ctx.strokeStyle='#ffffff55';ctx.stroke();}}
if(document.getElementById('points').checked){ctx.fillStyle='#f21b61';for(const p of obs){let s=project(p);ctx.beginPath();ctx.arc(s[0],s[1],2.5*devicePixelRatio,0,7);ctx.fill();projected.push({screen:s,p,label:'Observed IV (intraday approximation)'});}}
}
canvas.onpointerdown=e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);tip.style.display='none';};canvas.onpointerup=()=>drag=null;canvas.onpointercancel=()=>drag=null;
canvas.onpointermove=e=>{if(drag){yaw+=(e.clientX-drag[0])*.007;pitch=Math.max(.08,Math.min(1.5,pitch+(e.clientY-drag[1])*.005));drag=[e.clientX,e.clientY];draw();return;}
let r=canvas.getBoundingClientRect(),x=(e.clientX-r.left)*devicePixelRatio,y=(e.clientY-r.top)*devicePixelRatio,best=null,dist=(16*devicePixelRatio)**2;for(const p of projected){let d=(p.screen[0]-x)**2+(p.screen[1]-y)**2;if(d<dist){dist=d;best=p;}}
if(best){tip.textContent=best.label+'\nT: '+best.p[0].toFixed(5)+' years\nX / forward: '+best.p[1].toFixed(4)+'\nVolatility: '+best.p[2].toFixed(3)+'%';tip.style.display='block';tip.style.left=Math.min(e.clientX+15,innerWidth-270)+'px';tip.style.top=Math.min(e.clientY+15,innerHeight-120)+'px';}else tip.style.display='none';};
canvas.onpointerleave=()=>tip.style.display='none';canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.5,Math.min(2.5,zoom*Math.exp(-e.deltaY*.001)));draw();},{passive:false});
document.getElementById('reset').onclick=()=>{yaw=-.8;pitch=.65;zoom=1;draw();};document.getElementById('wire').onchange=draw;document.getElementById('points').onchange=draw;addEventListener('resize',draw);
document.getElementById('legend').textContent=`Surface colour: ${min[2].toFixed(1)}% to ${max[2].toFixed(1)}%. Separate patches preserve excluded maturity boundaries. Pink points: observed IVs.`;draw();
</script></html>'''


def implied_patches(model,permission):
    patches=[]
    for d in permission.get('domain',[]):
        times=np.linspace(d['start_years'],d['end_years'],14)[1:-1]
        ks=np.linspace(d['k_min'],d['k_max'],45);rows=[]
        for t in times:
            w=np.asarray(model.evaluate(ks,t)['w'],float)
            if not np.isfinite(w).all() or np.any(w<=0):raise ValueError('invalid_plot_variance')
            rows.append([[float(t),math.exp(float(k)),100*math.sqrt(float(v)/t)] for k,v in zip(ks,w)])
        patches.append(dict(label='Supported fitted IV',rows=rows))
    return patches


def write_implied_view(result,gid,path):
    cohort=next(r for r in result.tables['cohort_status'] if r['group_id']==gid)
    permission=cohort.get('local_vol_eligibility',{})
    patches=implied_patches(result.models[gid],permission)
    # Blocked/no-domain cohorts get no fabricated rectangular surface.
    if not patches:
        path.write_text('<!doctype html><meta charset="utf-8"><p>No supported surface domain available.</p>',encoding='utf-8');return
    observations=[[float(r['years']),math.exp(float(r['k'])),100*float(r['observed_iv'])]
        for r in result.tables.get('repricing',[]) if r['group_id']==gid]
    path.write_text(render_surface_3d(str(cohort.get('underlying'))+' - implied volatility',patches,observations,
        'Fitted IV on the bounded conversion domain. Pink observations may lie outside this domain. Intraday approximation and eligibility warnings remain in report.html.'),encoding='utf-8')


def local_patches(result,gid):
    """Connect only adjacent requested cells that have all four valid corners.

    Retain unavailable rows when constructing coordinates. Filtering them out
    first can bridge a missing strike column, even when all row lengths agree.
    Both HTML and PNG consume this same mesh so they cannot disagree on holes.
    """
    rows=[r for r in result.tables['local_vol_grid'] if r['group_id']==gid]
    cohort=next(r for r in result.tables['cohorts'] if r['group_id']==gid)
    label=('Validated interior local volatility' if cohort['status']=='validated_on_supported_grid'
           else 'Diagnostic local volatility (cohort not validated)')
    patches=[]
    for interval in sorted({r['interval'] for r in rows}):
        group=[r for r in rows if r['interval']==interval]
        times=sorted({r['years'] for r in group});ks=sorted({r['k'] for r in group})
        lookup={(r['years'],r['k']):r for r in group}
        for a,b in zip(times[:-1],times[1:]):
            run=[]
            def append_run():
                if len(run)>=2:
                    patches.append(dict(label=label,rows=[[[t,math.exp(k),100*lookup[t,k]['local_volatility']]
                        for k in run] for t in (a,b)]))
            for k in ks:
                corners=[lookup.get((t,k),{}) for t in (a,b)]
                valid=all(r.get('status')=='available' and math.isfinite(r.get('local_volatility',float('nan'))) for r in corners)
                if valid:run.append(k)
                else:append_run();run=[]
            append_run()
    return patches


def write_local_view(result,gid,path):
    patches=local_patches(result,gid)
    if not patches:
        path.write_text('<!doctype html><meta charset="utf-8"><p>No complete local-volatility grid cells available. See report.html.</p>',encoding='utf-8')
        return
    title=next(r.get('underlying') for r in result.tables['cohorts'] if r['group_id']==gid)
    path.write_text(render_surface_3d(str(title)+' - local volatility',patches,notice=
        'Annualized local volatility on available interior samples; cohort validation is stated in the labels and report. Missing cells, short end, wings and exact maturity knots are not plotted as market-supported data. Pricing extensions remain assumptions.'),encoding='utf-8')
