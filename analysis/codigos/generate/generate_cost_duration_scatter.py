"""
Interactive Scatter — Event Cost vs Duration
=============================================
Generates an interactive HTML scatter plot comparing stress events from both
datasets (future = circles, historical = triangles). Each point represents one
filtered event. Clicking a point shows its details (scenario, period, cluster,
feature anomalies); clicking a second point compares the two.

Inputs:
  CSVs/future/events_global_alpha/events_global_alpha25.csv
  CSVs/historical/events_global_alpha/events_global_historical_alpha25.csv
  CSVs/future/features/features_alpha25_climate_future_clustered.csv
  CSVs/future/features/features_alpha25_operational_future_clustered.csv
  CSVs/historical/features/features_alpha25_climate_historical_clustered.csv
  CSVs/historical/features/features_alpha25_operational_historical_clustered.csv

Outputs:
  htmls/htmls_uso/cost_duration_scatter.html
"""

# ── Configuration ─────────────────────────────────────────────────────────────
# Output type: HTML interactive
THRESH_CS  = 0.2   # minimum cost_share_cross (%) to include an event
THRESH_DUR = 1.0   # minimum duration (days) to include an event
OUT_HTML   = '~/Desktop/Bachelor Thesis/analysis/htmls_uso/cost_duration_scatter.html'
# ──────────────────────────────────────────────────────────────────────────────

import os, json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

OUT_HTML      = os.path.expanduser(OUT_HTML)
FUT_CSV       = os.path.expanduser('~/Desktop/Bachelor Thesis/CSVs_and_JSONs/future/events_global_alpha/events_global_alpha25.csv')
HIST_CSV      = os.path.expanduser('~/Desktop/Bachelor Thesis/CSVs_and_JSONs/historical/events_global_alpha/events_global_historical_alpha25.csv')
FUT_CLIM_CSV  = os.path.expanduser('~/Desktop/Bachelor Thesis/CSVs_and_JSONs/future/features/features_alpha25_climate_future_clustered.csv')
HIST_CLIM_CSV = os.path.expanduser('~/Desktop/Bachelor Thesis/CSVs_and_JSONs/historical/features/features_alpha25_climate_historical_clustered.csv')
FUT_OP_CSV    = os.path.expanduser('~/Desktop/Bachelor Thesis/CSVs_and_JSONs/future/features/features_alpha25_operational_future_clustered.csv')
HIST_OP_CSV   = os.path.expanduser('~/Desktop/Bachelor Thesis/CSVs_and_JSONs/historical/features/features_alpha25_operational_historical_clustered.csv')
OUT_HTML         = os.path.expanduser('~/Desktop/Bachelor Thesis/analysis/htmls_uso/cost_duration_scatter.html')

ZONES_FUT  = ['Nordics_N','Baltics','Nordics_S','British_Isles','NW_Continent',
              'Med_East_N','Med_East_S','Greece','Italy','Iberia','Central_W','East_N','Central_E']
ZONES_HIST = ['Norway','Baltics_FI','Nordics_S','British_Isles','NW_Continent',
              'Med_East_N','Med_East_S','Greece','Italy','Iberia','Central_W','East_N','Central_E']
CLIM_FEATS = ['wind anom','pv anom','heat anom']
OP_FEATS   = ['nettrans anom','stor level anom','stor discharge anom']
FEAT_TYPES = CLIM_FEATS + OP_FEATS

def get_features(df_clim, df_op, sc, t_start, t_end, zones):
    rc = df_clim[(df_clim['sc']==sc) & (df_clim['t_start']==t_start) & (df_clim['t_end']==t_end)]
    ro = df_op  [(df_op['sc'] ==sc) & (df_op['t_start'] ==t_start) & (df_op['t_end'] ==t_end)]
    if rc.empty or ro.empty:
        return None
    rc, ro = rc.iloc[0], ro.iloc[0]
    result = {}
    for ft in CLIM_FEATS:
        result[ft] = {z: round(float(rc[f'{ft} {z}']), 3) for z in zones}
    for ft in OP_FEATS:
        result[ft] = {z: round(float(ro[f'{ft} {z}']), 3) for z in zones}
    return result

def t_to_date(t):
    day = (int(t) - 1) // 24
    return (datetime(2001, 1, 1) + timedelta(days=day)).strftime('%b %d')

df_fut  = pd.read_csv(FUT_CSV);  df_fut['source']  = 'future'
df_hist = pd.read_csv(HIST_CSV); df_hist['source'] = 'historical'
df_fut_clim  = pd.read_csv(FUT_CLIM_CSV)
df_hist_clim = pd.read_csv(HIST_CLIM_CSV)
df_fut_op    = pd.read_csv(FUT_OP_CSV)
df_hist_op   = pd.read_csv(HIST_OP_CSV)

df_all = pd.concat([df_fut, df_hist], ignore_index=True)
df_top = df_all[(df_all['cost_share_cross'] >= 0.2) & (df_all['duration_days'] >= 1.0)].copy()
df_top['global_rank'] = df_top['event_cost'].rank(ascending=False, method='min').astype(int)
df_top['local_rank']  = df_top.groupby('source')['event_cost'].rank(ascending=False, method='min').astype(int)
df_top = df_top.reset_index(drop=True)

events = []
for idx, row in df_top.iterrows():
    is_fut   = row['source'] == 'future'
    df_clim  = df_fut_clim  if is_fut else df_hist_clim
    df_op    = df_fut_op    if is_fut else df_hist_op
    zones    = ZONES_FUT    if is_fut else ZONES_HIST
    events.append({
        'id':               int(idx),
        'source':           row['source'],
        'sc':               int(row['sc']),
        't_start':          int(row['t_start']),
        't_end':            int(row['t_end']),
        'duration_days':    round(float(row['duration_days']), 2),
        'event_cost':       round(float(row['event_cost']), 0),
        'cost_share_cross': round(float(row['cost_share_cross']), 3),
        'period':           f"{t_to_date(row['t_start'])} – {t_to_date(row['t_end'])}",
        'global_rank':      int(row['global_rank']),
        'local_rank':       int(row['local_rank']),
        'features':         get_features(df_clim, df_op, int(row['sc']), int(row['t_start']), int(row['t_end']), zones),
    })

events_json = json.dumps(events)

# scale per feature — climate fixed at 100, operational from 95th percentile
FIXED_SCALE = {'wind anom': 100, 'pv anom': 100, 'heat anom': 100}
feat_scale = {}
for ft in FEAT_TYPES:
    if ft in FIXED_SCALE:
        feat_scale[ft] = float(FIXED_SCALE[ft])
    else:
        cols_fut  = [f'{ft} {z}' for z in ZONES_FUT]
        cols_hist = [f'{ft} {z}' for z in ZONES_HIST]
        vals = np.concatenate([df_fut_op[cols_fut].values.ravel(),
                               df_hist_op[cols_hist].values.ravel()])
        feat_scale[ft] = float(np.ceil(np.percentile(np.abs(vals), 95) / 10) * 10)
feat_scale_json = json.dumps(feat_scale)

x_max = df_top['duration_days'].max()
y_max = df_top['event_cost'].max()
axis_ranges_json = json.dumps({
    'x': [0, round(x_max * 1.06, 1)],
    'y': [0, round(y_max * 1.06, 0)],
})

html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Stress Events — Cost vs Duration</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  :root {{
    --bg:#f4f6fb; --white:#ffffff; --text:#1a1d2e;
    --muted:#6b7280; --border:#e5e7ef; --accent:#4776e6;
    --fut:#3498db; --hist:#e74c3c;
    --sel-a:#4776e6; --sel-b:#f59e0b;
  }}
  html, body {{ height:100%; overflow:hidden; }}
  body {{ font-family:'Inter',sans-serif; background:var(--bg); color:var(--text);
          padding:16px 28px; display:flex; flex-direction:column; }}
  h1 {{ font-size:20px; font-weight:700; margin-bottom:2px; flex-shrink:0; }}
  .subtitle {{ font-size:13px; color:var(--muted); margin-bottom:12px; flex-shrink:0; }}

  .toggle-row {{ display:flex; gap:8px; margin-bottom:12px; flex-wrap:wrap; flex-shrink:0; }}
  .toggle-group {{ display:flex; background:var(--white); border:1px solid var(--border); border-radius:10px; overflow:hidden; }}
  .toggle-btn {{
    padding:8px 18px; font-size:12px; font-weight:500; color:var(--muted);
    cursor:pointer; border:none; background:none; font-family:'Inter',sans-serif; transition:all 0.15s;
  }}
  .toggle-btn:hover {{ background:var(--bg); color:var(--text); }}
  .toggle-btn.active {{ background:var(--accent); color:#fff; font-weight:600; }}

  #main-row {{ display:flex; gap:16px; flex:1; min-height:0; }}
  #plot-container {{
    flex:3; min-width:0; background:var(--white); border:1px solid var(--border);
    border-radius:14px; overflow:hidden; box-shadow:0 2px 8px rgba(0,0,0,0.04);
    padding:8px; display:flex; flex-direction:column;
  }}
  #scatter-plot {{ flex:1; min-height:0; width:100%; }}

  #detail-panel {{
    flex:2; min-width:0; background:var(--white); border:1px solid var(--border);
    border-radius:14px; box-shadow:0 2px 8px rgba(0,0,0,0.04); padding:20px;
    overflow-y:auto; display:none; flex-direction:column; gap:14px;
  }}
  #detail-panel.visible {{ display:flex; }}
  .feat-select-wrap {{ display:flex; align-items:center; gap:8px; }}
  .feat-select-wrap label {{ font-size:11px; font-weight:600; color:var(--muted); white-space:nowrap; }}
  .feat-select-wrap select {{ flex:1; padding:5px 8px; border-radius:6px; border:1px solid var(--border);
                              font-size:12px; font-family:'Inter',sans-serif; }}
  #detail-panel h3 {{ font-size:11px; font-weight:700; color:var(--muted);
                      text-transform:uppercase; letter-spacing:0.8px; margin-bottom:14px; }}
  #detail-empty {{ font-size:12px; color:var(--muted); line-height:1.6; }}
  #detail-content {{ display:none; }}

  .detail-badge {{
    display:inline-block; padding:3px 10px; border-radius:20px;
    font-size:11px; font-weight:600; color:#fff; margin-bottom:14px;
  }}
  .badge-future     {{ background:var(--fut); }}
  .badge-historical {{ background:var(--hist); }}

  .detail-row {{
    display:flex; justify-content:space-between; align-items:baseline;
    padding:7px 0; border-bottom:1px solid var(--border); gap:8px;
  }}
  .detail-row:last-child {{ border-bottom:none; }}
  .detail-key   {{ font-size:11px; color:var(--muted); flex-shrink:0; }}
  .detail-value {{ font-size:12px; font-weight:600; text-align:right; }}
  .panel-title-row {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:14px; }}
  .panel-title-row h3 {{ margin-bottom:0; }}
  #close-btn {{ background:none; border:none; cursor:pointer; color:var(--muted); font-size:18px; line-height:1;
                padding:0 2px; font-family:'Inter',sans-serif; }}
  #close-btn:hover {{ color:var(--text); }}

  .compare-grid {{ display:flex; gap:8px; margin-bottom:4px; }}
  .compare-card {{
    flex:1; min-width:0; border-radius:10px; padding:10px;
    border:1.5px solid var(--border);
  }}
  .compare-card .detail-row {{ padding:5px 0; }}
  .compare-card .detail-key  {{ font-size:10px; }}
  .compare-card .detail-value {{ font-size:11px; }}
  .compare-card .detail-badge {{ margin-bottom:8px; font-size:10px; padding:2px 8px; }}
  .bar-event-label {{
    font-size:10px; font-weight:600; margin-top:8px; margin-bottom:2px;
  }}
</style>
</head>
<body>

<h1>Stress Events — Cost vs Duration</h1>
<p class="subtitle">Events with cost share &ge; 20% and duration &ge; 1 day. Click a point to view details &middot; click a second to compare.</p>

<div class="toggle-row">
  <div class="toggle-group" id="srcGroup">
    <button class="toggle-btn active" onclick="setSource('both')">Both</button>
    <button class="toggle-btn" onclick="setSource('future')">Future</button>
    <button class="toggle-btn" onclick="setSource('historical')">Historical</button>
  </div>
</div>

<div id="main-row">
  <div id="plot-container">
    <div id="scatter-plot"></div>
  </div>
  <div id="detail-panel">
    <div>
      <div class="panel-title-row">
        <h3 id="panel-title">Event detail</h3>
        <button id="close-btn" onclick="deselect()" title="Close">&#x2715;</button>
      </div>
      <div id="detail-content"></div>
    </div>
    <div id="bar-section" style="display:none">
      <div class="feat-select-wrap">
        <label>Feature:</label>
        <select id="featSel" onchange="renderBarChart()">
          <option value="wind anom">Wind CF anomaly (%)</option>
          <option value="pv anom">PV CF anomaly (%)</option>
          <option value="heat anom">Heat demand anomaly (%)</option>
          <option value="nettrans anom">Net transfer anomaly (GWh/h)</option>
          <option value="stor level anom">Storage level anomaly (GWh)</option>
          <option value="stor discharge anom">Storage discharge anomaly (GWh/h)</option>
        </select>
      </div>
      <div id="bar-label-a" class="bar-event-label" style="color:var(--sel-a);display:none">Event A</div>
      <div id="bar-chart" style="display:none"></div>
      <div id="bar-label-b" class="bar-event-label" style="color:var(--sel-b);display:none">Event B</div>
      <div id="bar-chart-b" style="display:none"></div>
    </div>
  </div>
</div>

<script>
const EVENTS      = {events_json};
const AXIS_RANGES = {axis_ranges_json};
const FEAT_SCALE  = {feat_scale_json};
const SEL_COLORS  = ['#4776e6', '#f59e0b'];

let currentSource = 'both';
let selectedIds   = [];

function buildTraces(source) {{
  const fut  = EVENTS.filter(e => e.source === 'future'     && (source === 'both' || source === 'future'));
  const hist = EVENTS.filter(e => e.source === 'historical' && (source === 'both' || source === 'historical'));

  const makeTrace = (evs, color, name) => ({{
    type: 'scatter', mode: 'markers', name,
    x: evs.map(e => e.duration_days),
    y: evs.map(e => e.event_cost),
    customdata: evs.map(e => e.id),
    text: evs.map(e =>
      `<b>Sc${{e.sc}} · ${{e.source}}</b><br>` +
      `Period: ${{e.period}}<br>` +
      `Duration: ${{e.duration_days}} days<br>` +
      `Cost: ${{(e.event_cost/1000).toFixed(1)}} B€<br>` +
      `Cost share: ${{e.cost_share_cross.toFixed(3)}}%`
    ),
    hovertemplate: '%{{text}}<extra></extra>',
    marker: {{ color, size:11, opacity:0.85, line:{{ color:'#fff', width:1.5 }} }},
  }});

  const traces = [];
  if (hist.length) traces.push(makeTrace(hist, '#e74c3c', `Historical (n=${{hist.length}})`));
  if (fut.length)  traces.push(makeTrace(fut,  '#3498db', `Future (n=${{fut.length}})`));

  // highlight up to 2 selected points
  selectedIds.forEach((sid, i) => {{
    const ev = EVENTS[sid];
    if (ev && (source === 'both' || ev.source === source)) {{
      traces.push({{
        type: 'scatter', mode: 'markers',
        name: i === 0 ? 'Event A' : 'Event B',
        x: [ev.duration_days], y: [ev.event_cost],
        hoverinfo: 'skip',
        marker: {{ symbol:'circle-open', color:SEL_COLORS[i], size:22, line:{{color:SEL_COLORS[i], width:3}} }},
        showlegend: false,
      }});
    }}
  }});
  return traces;
}}

function render() {{
  const traces = buildTraces(currentSource);
  const layout = {{
    xaxis: {{
      title: {{ text:'Duration (days)', font:{{size:12,family:'Inter'}} }},
      gridcolor:'#f0f0f0', zeroline:false, tickfont:{{size:11,family:'Inter'}},
      range: AXIS_RANGES.x,
    }},
    yaxis: {{
      title: {{ text:'Event cost (M€)', font:{{size:12,family:'Inter'}} }},
      gridcolor:'#f0f0f0', zeroline:false, tickfont:{{size:11,family:'Inter'}},
      range: AXIS_RANGES.y,
    }},
    legend: {{ font:{{size:11,family:'Inter'}}, bgcolor:'rgba(255,255,255,0.9)', bordercolor:'#e5e7ef', borderwidth:1 }},
    margin: {{t:20, r:20, b:60, l:80}},
    paper_bgcolor:'#ffffff', plot_bgcolor:'#ffffff',
    font: {{family:'Inter'}}, hovermode:'closest',
  }};
  Plotly.react('scatter-plot', traces, layout, {{responsive:true, displayModeBar:false}});
}}

function deselect() {{
  selectedIds = [];
  document.getElementById('detail-panel').classList.remove('visible');
  render();
  requestAnimationFrame(() => Plotly.Plots.resize('scatter-plot'));
}}

function handleClick(id) {{
  const idx = selectedIds.indexOf(id);
  if (idx !== -1) {{
    // clicking already-selected: deselect it
    selectedIds.splice(idx, 1);
  }} else if (selectedIds.length < 2) {{
    selectedIds.push(id);
  }} else {{
    // already 2 selected: replace all with new click
    selectedIds = [id];
  }}
  if (selectedIds.length === 0) {{
    document.getElementById('detail-panel').classList.remove('visible');
  }} else {{
    renderPanel();
    document.getElementById('detail-panel').classList.add('visible');
  }}
  render();
  requestAnimationFrame(() => Plotly.Plots.resize('scatter-plot'));
}}

function renderPanel() {{
  const n = selectedIds.length;
  if (n === 0) return;
  document.getElementById('panel-title').textContent = n === 1 ? 'Event detail' : 'Comparison';
  document.getElementById('detail-content').style.display = 'block';
  document.getElementById('bar-section').style.display = 'block';
  if (n === 1) renderSingleDetail(selectedIds[0]);
  else renderCompareDetail(selectedIds[0], selectedIds[1]);
  renderBarChart();
}}

function renderSingleDetail(id) {{
  const ev = EVENTS[id];
  if (!ev) return;
  const badgeClass = ev.source === 'future' ? 'badge-future' : 'badge-historical';
  const srcLabel   = ev.source.charAt(0).toUpperCase() + ev.source.slice(1);
  document.getElementById('detail-content').innerHTML = `
    <span class="detail-badge ${{badgeClass}}">${{srcLabel}}</span>
    <div class="detail-row"><span class="detail-key">Scenario</span><span class="detail-value">Sc ${{ev.sc}}</span></div>
    <div class="detail-row"><span class="detail-key">Period</span><span class="detail-value">${{ev.period}}</span></div>
    <div class="detail-row"><span class="detail-key">Duration</span><span class="detail-value">${{ev.duration_days}} days</span></div>
    <div class="detail-row"><span class="detail-key">Event cost</span><span class="detail-value">${{(ev.event_cost/1000).toFixed(1)}} B€</span></div>
    <div class="detail-row"><span class="detail-key">Cost share</span><span class="detail-value">${{ev.cost_share_cross.toFixed(3)}}%</span></div>
    <div class="detail-row"><span class="detail-key">Global rank</span><span class="detail-value">#${{ev.global_rank}}</span></div>
    <div class="detail-row"><span class="detail-key">Local rank</span><span class="detail-value">#${{ev.local_rank}}</span></div>
  `;
}}

function renderCompareDetail(id1, id2) {{
  const ev1 = EVENTS[id1], ev2 = EVENTS[id2];
  const makeCard = (ev, ci) => {{
    const src = ev.source === 'future' ? 'Future' : 'Historical';
    const letter = ci === 0 ? 'A' : 'B';
    return `<div class="compare-card" style="border-color:${{SEL_COLORS[ci]}}">
      <span class="detail-badge" style="background:${{SEL_COLORS[ci]}}">${{letter}} &middot; ${{src}}</span>
      <div class="detail-row"><span class="detail-key">Scenario</span><span class="detail-value">Sc ${{ev.sc}}</span></div>
      <div class="detail-row"><span class="detail-key">Period</span><span class="detail-value">${{ev.period}}</span></div>
      <div class="detail-row"><span class="detail-key">Duration</span><span class="detail-value">${{ev.duration_days}} d</span></div>
      <div class="detail-row"><span class="detail-key">Cost</span><span class="detail-value">${{(ev.event_cost/1000).toFixed(1)}} B€</span></div>
      <div class="detail-row"><span class="detail-key">Cost share</span><span class="detail-value">${{ev.cost_share_cross.toFixed(3)}}%</span></div>
      <div class="detail-row"><span class="detail-key">Global rank</span><span class="detail-value">#${{ev.global_rank}}</span></div>
    </div>`;
  }};
  document.getElementById('detail-content').innerHTML =
    `<div class="compare-grid">${{makeCard(ev1,0)}}${{makeCard(ev2,1)}}</div>`;
}}

function renderBarChart() {{
  const feat = document.getElementById('featSel').value;
  if (selectedIds.length === 0) return;

  const labelA = document.getElementById('bar-label-a');
  const labelB = document.getElementById('bar-label-b');
  const chartA = document.getElementById('bar-chart');
  const chartB = document.getElementById('bar-chart-b');
  const yRange = [-FEAT_SCALE[feat], FEAT_SCALE[feat]];

  if (selectedIds.length === 1) {{
    labelA.style.display = 'none';
    labelB.style.display = 'none';
    chartB.style.display = 'none';
    chartA.style.display = 'block';
    const ev = EVENTS[selectedIds[0]];
    if (!ev || !ev.features) return;
    const color = ev.source === 'future' ? '#3498db' : '#e74c3c';
    const x = Object.keys(ev.features[feat]).map(z => z.replace(/_/g,' '));
    const y = Object.values(ev.features[feat]);
    Plotly.react('bar-chart', [{{type:'bar', x, y, marker:{{color, opacity:0.85}}}}], {{
      margin:{{t:10,r:10,b:80,l:40}}, height:260,
      paper_bgcolor:'#ffffff', plot_bgcolor:'#ffffff',
      font:{{family:'Inter',size:10}},
      xaxis:{{tickangle:-45, tickfont:{{size:9}}}},
      yaxis:{{gridcolor:'#f0f0f0', zeroline:true, zerolinecolor:'#ccc', range:yRange}},
      showlegend:false,
    }}, {{responsive:true, displayModeBar:false}});

  }} else {{
    // two events — show two separate bar charts
    labelA.style.display = 'block';
    labelB.style.display = 'block';
    chartA.style.display = 'block';
    chartB.style.display = 'block';

    const plotCommon = {{
      margin:{{t:6,r:10,b:80,l:40}}, height:190,
      paper_bgcolor:'#ffffff', plot_bgcolor:'#ffffff',
      font:{{family:'Inter',size:10}},
      xaxis:{{tickangle:-45, tickfont:{{size:9}}}},
      yaxis:{{gridcolor:'#f0f0f0', zeroline:true, zerolinecolor:'#ccc', range:yRange}},
      showlegend:false,
    }};

    [selectedIds[0], selectedIds[1]].forEach((id, ci) => {{
      const ev = EVENTS[id];
      if (!ev || !ev.features) return;
      const x = Object.keys(ev.features[feat]).map(z => z.replace(/_/g,' '));
      const y = Object.values(ev.features[feat]);
      const divId = ci === 0 ? 'bar-chart' : 'bar-chart-b';
      Plotly.react(divId, [{{type:'bar', x, y, marker:{{color:SEL_COLORS[ci], opacity:0.85}}}}],
        plotCommon, {{responsive:true, displayModeBar:false}});
    }});
  }}
}}

function setSource(src) {{
  currentSource = src;
  document.querySelectorAll('#srcGroup .toggle-btn').forEach(b => {{
    b.classList.toggle('active', b.onclick.toString().includes(`'${{src}}'`));
  }});
  render();
}}

render();

document.getElementById('scatter-plot').on('plotly_click', data => {{
  const id = data.points[0].customdata;
  if (typeof id === 'number') handleClick(id);
}});
</script>
</body>
</html>'''

os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
with open(OUT_HTML, 'w') as f:
    f.write(html)
print(f'Saved: {OUT_HTML}')
