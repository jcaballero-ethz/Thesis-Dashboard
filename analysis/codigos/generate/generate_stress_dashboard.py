"""
Stress Event Calendar Dashboard — Future Dataset
=================================================
Generates an interactive HTML showing all stress events across 60 scenarios
for each α threshold. The calendar view maps each (scenario, day-of-year) cell
to a colour representing the event's cost share. An alpha slider lets the user
switch between thresholds without reloading the page.

Also embeds links from filtered events (α=25%) to the shadow price / flow map.

Inputs:
  CSVs/future/events_global_alpha/events_global_alpha{10..40}.csv  (7 files)

Outputs:
  htmls/htmls_uso/stress_events_multi_alpha.html
"""

# ── Configuration ─────────────────────────────────────────────────────────────
# Output type: HTML interactive
ALPHAS   = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
IN_DIR   = '~/Desktop/Bachelor Thesis/CSVs_and_JSONs/future/events_global_alpha'
OUT_HTML = '~/Desktop/Bachelor Thesis/analysis/htmls_uso/stress_events_multi_alpha.html'
# ──────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import json, os

OUT_HTML = os.path.expanduser(OUT_HTML)
IN_DIR   = os.path.expanduser(IN_DIR)

# ── Read CSVs ──────────────────────────────────────────────────────────────────
ALL_DATA = {}   # alpha_pct -> {sc_num: [[d_start, d_end, share], ...]}
all_shares = []

for alpha in ALPHAS:
    pct  = int(alpha * 100)
    path = os.path.join(IN_DIR, f'events_global_alpha{pct:02d}.csv')
    df = pd.read_csv(path)
    sc_data = {}
    for sc_num in range(1, 61):
        sub = df[df['sc'] == sc_num].copy()
        if sub.empty:
            sc_data[sc_num] = []
        else:
            evs = []
            for _, row in sub.iterrows():
                d1  = int(row['t_start']) // 24 + 1
                d2  = int(row['t_end'])   // 24 + 1
                sh  = round(float(row['cost_share_cross']), 4)
                dur = round(float(row['duration_days']), 1)
                evs.append([d1, d2, sh, dur])
                all_shares.append(sh)
            evs.sort(key=lambda x: x[0])
            sc_data[sc_num] = evs
    ALL_DATA[pct] = sc_data
    print(f'  α={pct}%: {len(df)} events across {df["sc"].nunique()} scenarios')

# ── Compute colour thresholds (percentiles of all non-zero shares) ─────────────
arr = np.array(all_shares)
t1 = round(float(np.percentile(arr, 40)), 4)
t2 = round(float(np.percentile(arr, 70)), 4)
t3 = round(float(np.percentile(arr, 90)), 4)
print(f'Colour thresholds: <{t1}%  {t1}–{t2}%  {t2}–{t3}%  >{t3}%')

# Serialise to JS
js_data   = json.dumps({str(k): {str(s): v for s, v in d.items()}
                        for k, d in ALL_DATA.items()})
alpha_list = json.dumps([int(a*100) for a in ALPHAS])

# ── Build SP flow event lookup (α=25%, cost_share≥0.2, duration≥1d) ───────────
sp_ev_path = os.path.expanduser('~/Desktop/Bachelor Thesis/CSVs_and_JSONs/future/events_global_alpha/events_global_alpha25.csv')
sp_ev = {}
if os.path.exists(sp_ev_path):
    sp_df = pd.read_csv(sp_ev_path)
    sp_df = sp_df[(sp_df['cost_share_cross'] >= 0.2) & (sp_df['duration_days'] >= 1.0)].reset_index(drop=True)
    sp_df = sp_df.sort_values(['sc', 't_start']).reset_index(drop=True)
    for i, row in sp_df.iterrows():
        d1 = int(row['t_start']) // 24 + 1
        d2 = int(row['t_end'])   // 24 + 1
        sc = int(row['sc'])
        sp_ev[f"{sc}_{d1}_{d2}"] = int(i)
sp_ev_js = json.dumps(sp_ev)

# ── Generate HTML ──────────────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Stress Events Dashboard — Global Pool · Multi-α</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  :root {{
    --bg: #F5F3EC; --surface: #FFFFFF; --surface2: #EFEDE6;
    --text: #1A1A18; --text2: #5F5E5A; --text3: #888780;
    --border: rgba(0,0,0,0.12); --border2: rgba(0,0,0,0.22);
    --accent: #BA7517;
  }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          background: linear-gradient(rgba(240,245,255,0.92),rgba(240,245,255,0.92)), url('https://images.unsplash.com/photo-1466611653911-95081537e5b7?w=1600&q=80') center/cover no-repeat fixed; color: var(--text); min-height: 100vh; padding: 2rem 1.5rem; }}
  .container {{ max-width: 1100px; margin: 0 auto; }}
  h1 {{ font-size: 18px; font-weight: 500; margin-bottom: 4px; }}
  .subtitle {{ font-size: 13px; color: var(--text2); margin-bottom: 1.25rem; }}

  /* Alpha tabs */
  .alpha-tabs {{ display: flex; gap: 6px; margin-bottom: 1.5rem; flex-wrap: wrap; }}
  .tab {{ padding: 5px 14px; border-radius: 20px; font-size: 12px; font-weight: 500;
          border: 1px solid var(--border2); cursor: pointer; background: var(--surface);
          color: var(--text2); transition: all .15s; }}
  .tab:hover {{ border-color: var(--accent); color: var(--accent); }}
  .tab.active {{ background: var(--accent); color: #fff; border-color: var(--accent); }}

  /* Metrics */
  .metrics {{ display: grid; grid-template-columns: repeat(6, minmax(0,1fr)); gap: 10px; margin-bottom: 1.5rem; }}
  @media (max-width: 700px) {{ .metrics {{ grid-template-columns: repeat(3,1fr); }} }}
  @media (max-width: 450px) {{ .metrics {{ grid-template-columns: repeat(2,1fr); }} }}
  .metric {{ background: var(--surface2); border-radius: 8px; padding: .8rem 1rem; }}
  .metric-label {{ font-size: 11px; color: var(--text2); margin-bottom: 3px; }}
  .metric-value {{ font-size: 20px; font-weight: 500; }}

  /* Filter row */
  .filter-row {{ display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 1rem; align-items: center; }}
  .filter-row label {{ font-size: 12px; color: var(--text2); }}
  select, input[type=range] {{ font-size: 12px; border: 0.5px solid var(--border2); border-radius: 6px;
                               background: var(--surface); color: var(--text); padding: 4px 8px; outline: none; }}

  /* Heatmap */
  .section-label {{ font-size: 12px; color: var(--text2); margin-bottom: 8px; font-weight: 500; }}
  .heatmap-wrap {{ position: relative; margin-bottom: 10px; }}
  #hm {{ width: 100%; display: block; border-radius: 6px; cursor: crosshair; }}
  #tt {{
    position: fixed; display: none;
    background: var(--surface); border: 0.5px solid var(--border2);
    border-radius: 8px; padding: 7px 11px; font-size: 12px; line-height: 1.7;
    pointer-events: none; z-index: 100; min-width: 165px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
  }}

  /* Legend */
  .legend {{ display: flex; flex-wrap: wrap; align-items: center; gap: 14px; margin: 0 0 1.5rem; font-size: 11px; color: var(--text2); }}
  .swatch {{ width: 10px; height: 10px; border-radius: 2px; display: inline-block; margin-right: 4px; vertical-align: middle; }}

  /* Charts */
  .charts-row {{ display: grid; grid-template-columns: minmax(0,2fr) minmax(0,1fr); gap: 16px; margin-top: 1.25rem; }}
  @media (max-width: 700px) {{ .charts-row {{ grid-template-columns: 1fr; }} }}
  .card {{ background: var(--surface); border: 0.5px solid var(--border); border-radius: 12px; padding: 1rem 1.25rem; }}

  /* Detail panel */
  .detail-panel {{ background: var(--surface); border: 0.5px solid var(--border); border-radius: 12px;
                   padding: 1rem 1.25rem; min-height: 120px; margin-top: 16px; }}
  .event-row {{ display: flex; justify-content: space-between; align-items: center;
                padding: 5px 0; border-bottom: 0.5px solid var(--border); font-size: 12px; }}
  .event-row:last-child {{ border-bottom: none; }}
  .event-row.has-flow {{ cursor: pointer; }}
  .event-row.has-flow:hover {{ background: var(--surface2, rgba(0,0,0,0.04)); border-radius: 4px; }}
  .flow-hint {{ font-size: 10px; color: var(--accent); margin-left: 8px; opacity: 0.7; }}
  .ev-share {{ font-weight: 500; padding: 2px 7px; border-radius: 4px; font-size: 11px; }}
  .sl {{ background: #FDCA6B; color: #6B3A00; }}
  .sm {{ background: #F08C00; color: #fff; }}
  .sh {{ background: #C94A00; color: #fff; }}
  .sv {{ background: #7D0000; color: #fff; }}

  .hint {{ font-size: 11px; color: var(--text3); margin-top: 14px; }}
</style>
</head>
<body>
<div class="container">
  <h1>Stress Events — Global Pool · Multi-α</h1>
  <p class="subtitle">European energy system · ZEN-garden LP · 60 climate scenarios · click a row to inspect</p>

  <div class="alpha-tabs" id="tabs"></div>

  <div class="metrics">
    <div class="metric"><div class="metric-label">total retained events</div><div class="metric-value" id="vt">—</div></div>
    <div class="metric"><div class="metric-label">avg events / scenario</div><div class="metric-value" id="va">—</div></div>
    <div class="metric"><div class="metric-label">avg event duration</div><div class="metric-value" id="vd">—</div></div>
    <div class="metric"><div class="metric-label">peak single-event share</div><div class="metric-value" id="vm">—</div></div>
    <div class="metric"><div class="metric-label">scenarios with events</div><div class="metric-value" id="vw">—</div></div>
    <div class="metric"><div class="metric-label">avg cost share / event</div><div class="metric-value" id="vcs">—</div></div>
  </div>

  <div class="filter-row">
    <label>highlight: dominant event in</label>
    <select id="fSeason">
      <option value="all">all seasons</option>
      <option value="early">early winter (Jan–Feb, days 1–59)</option>
      <option value="late">late winter (Nov–Dec, days 305–365)</option>
      <option value="spring">spring (Mar–May)</option>
      <option value="summer">summer (Jun–Aug)</option>
    </select>
    <label style="margin-left:6px">min cost share ≥</label>
    <input type="range" id="fShare" min="0" max="1000" step="1" value="0" style="width:110px">
    <span id="fShareVal" style="font-size:12px;color:var(--text2);min-width:42px">0.000%</span>
    <span style="font-size:11px;color:var(--text3)" id="shareUnit">(% of cross-sc. cost)</span>
    <label style="margin-left:6px">min duration ≥</label>
    <input type="range" id="fDur" min="0" max="720" step="1" value="0" style="width:110px">
    <input type="number" id="fDurNum" min="0" max="720" value="0" style="width:52px;font-size:12px;border:1px solid #aaa;border-radius:4px;padding:1px 4px;">
    <span style="font-size:12px;color:var(--text2)">h</span>
  </div>

  <p class="section-label">event calendar — 60 scenarios × 365 days · shaded by cost share</p>
  <div class="heatmap-wrap"><canvas id="hm"></canvas></div>
  <div id="tt"></div>

  <div class="legend">
    <span>cost share (cross-sc.):</span>
    <span><span class="swatch" style="background:#FDCA6B"></span>&lt;<span id="leg1">—</span>%</span>
    <span><span class="swatch" style="background:#F08C00"></span><span id="leg2">—</span>%</span>
    <span><span class="swatch" style="background:#C94A00"></span><span id="leg3">—</span>%</span>
    <span><span class="swatch" style="background:#7D0000"></span>top</span>
    <span style="margin-left:6px"><span class="swatch" style="background:rgba(59,109,17,0.12);border:0.5px solid rgba(59,109,17,0.3)"></span>spring/summer</span>
    <span style="color:var(--text3)"><span class="swatch" style="background:rgba(128,128,128,0.15)"></span>filtered out</span>
  </div>

  <div class="charts-row">
    <div class="card">
      <p class="section-label">events per scenario</p>
      <div style="position:relative;height:160px"><canvas id="dc"></canvas></div>
    </div>
    <div class="card">
      <p class="section-label">dominant event by month</p>
      <div style="position:relative;height:160px"><canvas id="tc"></canvas></div>
    </div>
  </div>

  <div class="detail-panel">
    <div id="dt" style="font-size:14px;font-weight:500;margin-bottom:10px;color:var(--text3)">click a scenario row to inspect</div>
    <div id="db"></div>
  </div>

  <p class="hint">thesis: "Characterizing and Interpreting Weather-Driven Stress Events in European Energy Systems" · ETH Zürich RRE Lab</p>
</div>

<script>
const ALL = {js_data};
const ALPHA_LIST = {alpha_list};
const T1={t1}, T2={t2}, T3={t3};   // colour thresholds (% of cross-sc. cost)
const SP_EV = {sp_ev_js};   // (sc_d1_d2) -> ev_id for SP flow map events

const MN=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
const MS=[1,32,60,91,121,152,182,213,244,274,305,335];
const isDark=false;

// Update legend labels
document.getElementById('leg1').textContent=T1.toFixed(3);
document.getElementById('leg2').textContent=T1.toFixed(3)+'–'+T2.toFixed(3);
document.getElementById('leg3').textContent=T2.toFixed(3)+'–'+T3.toFixed(3);

let curAlpha = ALPHA_LIST[0];
let D = {{}};  // current scenario data

function setAlpha(a) {{
  curAlpha = a;
  D = ALL[String(a)];
  // rebuild derived maps
  dmC = {{}};
  sel = null;
  document.getElementById('dt').style.color='var(--text3)';
  document.getElementById('dt').textContent='click a scenario row to inspect';
  document.getElementById('db').innerHTML='';
  updateMetrics();
  draw();
  updateCharts();
  // update slider max based on data range
  const allSh = Object.values(D).flat().map(e=>e[2]);
  const maxSh = allSh.length ? Math.max(...allSh) : 1;
  const sl = document.getElementById('fShare');
  sl.max = Math.round(maxSh * 1000);
}}

// Build tabs
const tabsEl = document.getElementById('tabs');
ALPHA_LIST.forEach(a => {{
  const btn = document.createElement('button');
  btn.className = 'tab' + (a === curAlpha ? ' active' : '');
  btn.textContent = 'α = ' + a + '%';
  btn.addEventListener('click', () => {{
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    btn.classList.add('active');
    setAlpha(a);
  }});
  tabsEl.appendChild(btn);
}});

function d2m(d){{let m=0;for(let i=11;i>=0;i--){{if(MS[i]<=d){{m=i;break;}}}}return{{m,day:d-MS[m]+1}};}}
function colC(cs){{if(!cs)return null;if(cs<T1)return'#FDCA6B';if(cs<T2)return'#F08C00';if(cs<T3)return'#C94A00';return'#7D0000';}}
function scl(cs){{if(cs<T1)return'sl';if(cs<T2)return'sm';if(cs<T3)return'sh';return'sv';}}
let sel=null, fSeas='all', fMin=0.0, fDurMin=0;

function filteredEvs(s){{
  return (D[String(s)]||[]).filter(e=>e[2]>=fMin && e[3]*24>=fDurMin);
}}

function dom(s){{
  const evs=filteredEvs(s);
  if(!evs.length)return null;
  return evs.reduce((b,e)=>e[2]>b[2]?e:b);
}}

function hl(s){{
  const evs=filteredEvs(s); if(!evs.length)return false;
  if(fSeas==='all')return true;
  const e=dom(s); const mid=(e[0]+e[1])/2;
  if(fSeas==='early')return mid<=59; if(fSeas==='late')return mid>=305;
  if(fSeas==='spring')return mid>=60&&mid<=151; if(fSeas==='summer')return mid>=152&&mid<=243;
  return true;
}}

const cv=document.getElementById('hm');
const PL=36,PR=8,PT=8,PB=22;

function draw(){{
  const W=cv.parentElement.clientWidth||900;
  const IH=Math.min(Math.max(220,window.innerHeight*0.36),340);
  const RH=IH/60; const dW=(W-PL-PR)/365;
  cv.width=W; cv.height=IH+PT+PB;
  const cx=cv.getContext('2d');
  cx.fillStyle='#F8F9FB'; cx.fillRect(0,0,W,cv.height);
  cx.fillStyle='rgba(59,109,17,0.08)';
  cx.fillRect(PL+151*dW,PT,92*dW,IH);
  for(let s=1;s<=60;s++){{
    const h=hl(s);
    if(h){{
      const fevs=filteredEvs(s);
      for(const [a,b,c] of fevs){{
        const col=colC(c); if(!col)continue;
        cx.fillStyle=col;
        for(let d=a;d<=b&&d<=365;d++)cx.fillRect(PL+(d-1)*dW,PT+(s-1)*RH,dW,RH-0.4);
      }}
    }}else{{
      cx.fillStyle='rgba(0,0,0,0.06)';cx.fillRect(PL,PT+(s-1)*RH,W-PL-PR,RH);
    }}
    if(s===sel){{cx.strokeStyle='rgba(186,117,23,0.9)';cx.lineWidth=1.5;cx.strokeRect(PL+0.75,PT+(s-1)*RH+0.75,(W-PL-PR)-1.5,RH-1.5);}}
  }}
  cx.strokeStyle='rgba(0,0,0,0.15)';cx.lineWidth=0.5;
  MS.slice(1).forEach(d=>{{const x=PL+(d-1)*dW;cx.beginPath();cx.moveTo(x,PT);cx.lineTo(x,PT+IH);cx.stroke();}});
  const lc=isDark?'#888780':'#5F5E5A';
  cx.fillStyle=lc;cx.font='9px system-ui,sans-serif';cx.textAlign='center';
  MS.forEach((d,i)=>{{const n=i<11?MS[i+1]:366;cx.fillText(MN[i][0],PL+((d-1+(n-1))/2)*dW,PT+IH+PB-5);}});
  cx.textAlign='right';
  for(let s=10;s<=60;s+=10){{cx.fillStyle=lc;cx.fillText('s'+s,PL-3,PT+(s-0.5)*RH+3);}}
}}

const tt=document.getElementById('tt');
cv.addEventListener('mousemove',e=>{{
  const rc=cv.getBoundingClientRect();const sx=cv.width/rc.width;
  const W=cv.width;const IH=cv.height-PT-PB;
  const px=(e.clientX-rc.left)*sx,py=(e.clientY-rc.top)*sx;
  const dW=(W-PL-PR)/365;const RH=IH/60;
  const di=Math.floor((px-PL)/dW)+1,si=Math.floor((py-PT)/RH)+1;
  if(di>=1&&di<=365&&si>=1&&si<=60){{
    const evHere=(D[String(si)]||[]).find(([a,b])=>di>=a&&di<=b);
    const cs=evHere?evHere[2]:0;const{{m,day}}=d2m(di);
    tt.style.display='block';
    tt.style.left=Math.min(e.clientX+14,window.innerWidth-195)+'px';
    tt.style.top=Math.max(e.clientY-55,8)+'px';
    if(cs>0){{
      const ev=(D[String(si)]||[]).find(([a,b])=>di>=a&&di<=b);
      tt.innerHTML=`<strong style="color:var(--text)">Scenario ${{si}}</strong><br><span style="color:var(--text2)">${{MN[m]}} ${{day}} (day ${{di}})</span><br><span style="color:#BA7517;font-weight:500">${{cs.toFixed(4)}}% cost share</span>`+(ev?`<br><span style="color:var(--text2)">event days ${{ev[0]}}–${{ev[1]}} · ${{(ev[1]-ev[0])}} days</span>`:'');
    }} else {{
      tt.innerHTML=`<strong style="color:var(--text)">Scenario ${{si}}</strong><br><span style="color:var(--text2)">${{MN[m]}} ${{day}} (day ${{di}})</span><br><span style="color:var(--text3)">no stress event</span>`;
    }}
  }} else {{ tt.style.display='none'; }}
}});
cv.addEventListener('mouseleave',()=>tt.style.display='none');
cv.addEventListener('click',e=>{{
  const rc=cv.getBoundingClientRect();const sx=cv.width/rc.width;
  const IH=cv.height-PT-PB;const py=(e.clientY-rc.top)*sx;
  const s=Math.floor((py-PT)/(IH/60))+1;
  if(s>=1&&s<=60){{sel=s;draw();showDetail(s);}}
}});

function showDetail(s){{
  const evs=filteredEvs(s);
  if(!evs.length){{
    document.getElementById('dt').textContent=`Scenario ${{s}} — no events matching filters at α=${{curAlpha}}%`;
    document.getElementById('db').innerHTML='';
    return;
  }}
  const d=dom(s),tot=evs.reduce((a,e)=>a+e[2],0);
  document.getElementById('dt').style.color='var(--text)';
  document.getElementById('dt').textContent=`Scenario ${{s}} — ${{evs.length}} matching events · total share: ${{tot.toFixed(4)}}%`;
  let h=`<p style="font-size:12px;color:var(--text2);margin-bottom:10px">dominant: days <strong>${{d[0]}}–${{d[1]}}</strong> (${{(d[1]-d[0])}} days) · <strong>${{d[2].toFixed(4)}}%</strong></p><div>`;
  evs.forEach((ev,i)=>{{
    const{{m:m1,day:d1}}=d2m(ev[0]),{{m:m2,day:d2}}=d2m(ev[1]);
    const spKey=`${{s}}_${{ev[0]}}_${{ev[1]}}`;
    const evId=curAlpha===25?SP_EV[spKey]:undefined;
    const flowCls=evId!==undefined?' has-flow':'';
    const flowAttr=evId!==undefined?' data-evid="'+evId+'"':'';
    const flowHint=evId!==undefined?'<span class="flow-hint">double-click → SP flow</span>':'';
    h+=`<div class="event-row${{flowCls}}"${{flowAttr}}><span><span style="font-size:11px;color:var(--text3);margin-right:6px">ev ${{i+1}}</span>days ${{ev[0]}}–${{ev[1]}} <span style="color:var(--text3);font-size:11px">${{MN[m1]}} ${{d1}}–${{MN[m2]}} ${{d2}}</span></span><span style="color:var(--text2)">${{(ev[1]-ev[0])}}d &nbsp;<span class="ev-share ${{scl(ev[2])}}">${{ev[2].toFixed(4)}}%</span>${{flowHint}}</span></div>`;
  }});
  h+='</div>';
  document.getElementById('db').innerHTML=h;
}}

function updateMetrics(){{
  let tot=0,maxC=0,maxS=1,scWithEv=0,durSum=0,shareSum=0;
  for(let s=1;s<=60;s++){{
    const evs=filteredEvs(s);
    if(evs.length>0){{tot+=evs.length;scWithEv++;}}
    evs.forEach(e=>{{durSum+=e[3];shareSum+=e[2];}});
    const b=dom(s); if(b&&b[2]>maxC){{maxC=b[2];maxS=s;}}
  }}
  document.getElementById('vt').textContent=tot;
  document.getElementById('va').textContent=(tot/60).toFixed(2);
  document.getElementById('vd').textContent=tot>0?(durSum/tot).toFixed(1)+'d':'—';
  document.getElementById('vm').textContent=maxC.toFixed(4)+'% (sc'+maxS+')';
  document.getElementById('vw').textContent=scWithEv+'/60';
  document.getElementById('vcs').textContent=tot>0?(shareSum/tot).toFixed(4)+'%':'—';
}}

let dcChart=null, tcChart=null;
function updateCharts(){{
  const eC=new Array(10).fill(0), timing=new Array(12).fill(0);
  for(let s=1;s<=60;s++){{
    const evs=filteredEvs(s);
    if(evs.length>=1)eC[Math.min(evs.length,10)-1]++;
    const b=dom(s);
    if(b){{
      const mid=(b[0]+b[1])/2;
      for(let i=11;i>=0;i--){{if(MS[i]<=mid){{timing[i]++;break;}}}}
    }}
  }}
  const ticks={{color:isDark?'#888780':'#5F5E5A',font:{{size:11}}}};
  const gridC={{color:isDark?'rgba(255,255,255,0.07)':'rgba(0,0,0,0.06)'}};
  const baseOpts={{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}}}}}};

  if(dcChart)dcChart.destroy();
  dcChart=new Chart(document.getElementById('dc'),{{type:'bar',
    data:{{labels:['1','2','3','4','5','6','7','8','9','10+'],
           datasets:[{{data:eC,backgroundColor:'#C94A00',borderRadius:4}}]}},
    options:{{...baseOpts,plugins:{{...baseOpts.plugins,tooltip:{{callbacks:{{label:c=>`${{c.parsed.y}} scenarios`}}}}}},
      scales:{{x:{{grid:{{display:false}},ticks,title:{{display:true,text:'events per scenario',color:ticks.color,font:{{size:11}}}}}},
               y:{{grid:gridC,ticks,min:0}}}}}}}});

  if(tcChart)tcChart.destroy();
  tcChart=new Chart(document.getElementById('tc'),{{type:'bar',
    data:{{labels:MN.map(m=>m.substring(0,3)),
           datasets:[{{data:timing,backgroundColor:timing.map((_,i)=>(i<=1||i>=10)?'#C94A00':'#639922'),borderRadius:4}}]}},
    options:{{...baseOpts,plugins:{{...baseOpts.plugins,tooltip:{{callbacks:{{label:c=>`${{c.parsed.y}} scenarios`}}}}}},
      scales:{{x:{{grid:{{display:false}},ticks}},y:{{grid:gridC,ticks,min:0}}}}}}}});
}}

// Filter listeners
function applyFilters(){{ draw(); updateMetrics(); updateCharts(); if(sel)showDetail(sel); }}
document.getElementById('fSeason').addEventListener('change',e=>{{fSeas=e.target.value;applyFilters();}});
let _prevShare=0;
document.getElementById('fShare').addEventListener('input',e=>{{
  let v=+e.target.value;
  if((_prevShare<200&&v>200)||(_prevShare>200&&v<200)){{v=200;e.target.value=200;}}
  _prevShare=v;
  fMin=v/1000;
  document.getElementById('fShareVal').textContent=fMin.toFixed(3)+'%';
  applyFilters();
}});
document.getElementById('fDur').addEventListener('input',e=>{{
  fDurMin = +e.target.value;
  document.getElementById('fDurNum').value=fDurMin;
  applyFilters();
}});
['input','change'].forEach(ev=>document.getElementById('fDurNum').addEventListener(ev,e=>{{
  let v=Math.min(720,Math.max(0,parseInt(e.target.value)||0));
  fDurMin=v;
  document.getElementById('fDur').value=v;
  applyFilters();
}}));
window.addEventListener('resize',draw);
document.getElementById('db').addEventListener('dblclick',e=>{{
  const row=e.target.closest('.event-row.has-flow');
  if(!row)return;
  window.parent.postMessage({{type:'openSpFlow',evId:+row.dataset.evid}},'*');
}});

requestAnimationFrame(()=>setAlpha(ALPHA_LIST[0]));
</script>
</body>
</html>"""

out = os.path.expanduser('~/Desktop/Bachelor Thesis/analysis/htmls_uso/stress_events_multi_alpha.html')
with open(out, 'w', encoding='utf-8') as f:
    f.write(html)
print(f'Saved: {out}')
