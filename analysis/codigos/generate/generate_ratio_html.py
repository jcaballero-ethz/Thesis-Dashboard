"""
Intensity Ratio Dashboard — Future & Historical
================================================
Generates an HTML visualizing the intensity ratio ρ = cost_share / duration
(% per day) for all events across multiple α thresholds. A higher ρ means the
event concentrates more system cost per day. The dashboard shows:
  - Top-10 most intense events per dataset at each α
  - Ratio distributions and how they shift with α

Inputs:
  CSVs/future/events_global_alpha/events_global_alpha{10..40}.csv
  CSVs/historical/events_global_alpha/events_global_historical_alpha{10..40}.csv

Outputs:
  htmls/htmls_uso/ratio_analysis.html
"""

# ── Configuration ─────────────────────────────────────────────────────────────
# Output type: HTML interactive
ALPHAS      = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
TOP_N       = 10    # number of top events to highlight per dataset
HIGHLIGHT_A = 25    # alpha threshold highlighted by default in the HTML
OUT_HTML    = '~/Desktop/Bachelor Thesis/analysis/htmls_uso/ratio_analysis.html'
# ──────────────────────────────────────────────────────────────────────────────

import os, json
import numpy as np
import pandas as pd

OUT_HTML    = os.path.expanduser(OUT_HTML)
ALPHA_PCTS  = [int(a * 100) for a in ALPHAS]

DATASETS = {
    'future': {
        'label': 'Future',
        'csv': lambda pct: os.path.expanduser(
            f'~/Desktop/Bachelor Thesis/CSVs/future/events_global_alpha/events_global_alpha{pct:02d}.csv'
            if os.path.exists(os.path.expanduser(f'~/Desktop/Bachelor Thesis/CSVs/future/events_global_alpha/events_global_alpha{pct:02d}.csv'))
            else f'~/Desktop/Bachelor Thesis/CSVs/future/events_global_alpha/events_global_alpha{pct:02d}.csv'),
    },
    'historical': {
        'label': 'Historical',
        'csv': lambda pct: os.path.expanduser(f'~/Desktop/Bachelor Thesis/CSVs/historical/events_global_alpha/events_global_historical_alpha{pct:02d}.csv'),
    },
}

COLORS = ['#3498db','#e74c3c','#2ecc71','#9b59b6','#f39c12',
          '#1abc9c','#e67e22','#34495e','#e91e63','#00bcd4']

def load_dfs(cfg):
    dfs = {}
    for alpha in ALPHAS:
        pct  = int(alpha * 100)
        path = cfg['csv'](pct)
        if not os.path.exists(path):
            path = os.path.expanduser(f'~/Desktop/Bachelor Thesis/CSVs/future/events_global_alpha/events_global_alpha{pct:02d}.csv')
        dfs[alpha] = pd.read_csv(path)
    return dfs

def find_persistent(dfs):
    top_per_alpha = {a: dfs[a].nlargest(TOP_N, 'cost_share_cross').reset_index(drop=True)
                     for a in ALPHAS}
    pool = []
    for alpha in ALPHAS:
        for _, row in top_per_alpha[alpha].iterrows():
            sc, ts, te = int(row['sc']), int(row['t_start']), int(row['t_end'])
            already = any(
                ev['sc'] == sc and (
                    (ev['ts'] <= ts and ev['te'] >= te) or
                    (ts <= ev['ts'] and te >= ev['te'])
                ) for ev in pool
            )
            if not already:
                pool.append({'sc': sc, 'ts': ts, 'te': te, 'first_alpha': alpha})

    for ev in pool:
        ev['persistence'] = sum(
            1 for alpha in ALPHAS
            if len(top_per_alpha[alpha][
                (top_per_alpha[alpha]['sc'] == ev['sc']) &
                (top_per_alpha[alpha]['t_start'] <= ev['ts']) &
                (top_per_alpha[alpha]['t_end']   >= ev['te'])
            ]) > 0
        )

    pool.sort(key=lambda x: x['persistence'], reverse=True)
    return pool[:TOP_N]

def compute_rho(pool, dfs):
    events = []
    for ev in pool:
        sc, ts0, te0 = ev['sc'], ev['ts'], ev['te']
        rho_vals, cs_vals, dur_vals = [], [], []
        for alpha in ALPHAS:
            df_a  = dfs[alpha]
            match = df_a[(df_a['sc'] == sc) &
                         (df_a['t_start'] <= te0) &
                         (df_a['t_end']   >= ts0)]
            if len(match) > 0:
                m   = match.iloc[0]
                cs  = round(float(m['cost_share_cross']), 4)
                dur = round(float(m['duration_days']), 2)
                rho_vals.append(round(cs / dur, 5) if dur > 0 else None)
                cs_vals.append(cs)
                dur_vals.append(dur)
            else:
                rho_vals.append(None)
                cs_vals.append(None)
                dur_vals.append(None)

        # % change vs previous alpha
        pct_chg = [None]
        for i in range(1, len(ALPHAS)):
            r_cur, r_prev = rho_vals[i], rho_vals[i-1]
            if r_cur is not None and r_prev is not None and r_prev != 0:
                pct_chg.append(round((r_cur - r_prev) / r_prev * 100, 1))
            else:
                pct_chg.append(None)

        events.append({
            'label':       f'sc{sc}',
            'persistence': ev['persistence'],
            'rho':         rho_vals,
            'cs':          cs_vals,
            'dur':         dur_vals,
            'pct_chg':     pct_chg,
        })
    return events

all_data = {}
for key, cfg in DATASETS.items():
    print(f'Processing {key}...')
    dfs  = load_dfs(cfg)
    pool = find_persistent(dfs)
    all_data[key] = compute_rho(pool, dfs)
    print(f'  Top-10 events: {[e["label"] for e in all_data[key]]}')

data_json   = json.dumps(all_data)
alphas_json = json.dumps(ALPHA_PCTS)
colors_json = json.dumps(COLORS)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Ratio Analysis — ρ = cost share / duration</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  :root {{
    --bg:#f4f6fb; --white:#fff; --text:#1a1d2e; --muted:#6b7280;
    --border:#e5e7ef; --accent:#4776e6; --red:#e74c3c; --amber:#f39c12;
  }}
  html, body {{ height:100%; overflow:hidden; font-family:'Inter',sans-serif;
                background:var(--bg); color:var(--text); }}
  body {{ display:flex; flex-direction:column; padding:16px 24px; gap:12px; }}

  .top-bar {{ display:flex; align-items:baseline; justify-content:space-between; flex-shrink:0; }}
  h1 {{ font-size:18px; font-weight:700; }}
  .subtitle {{ font-size:12px; color:var(--muted); margin-top:2px; flex-shrink:0; }}

  .toggle-group {{ display:flex; background:var(--white); border:1px solid var(--border);
                   border-radius:10px; overflow:hidden; flex-shrink:0; }}
  .toggle-btn {{ padding:7px 20px; font-size:12px; font-weight:500; color:var(--muted);
                 cursor:pointer; border:none; background:none; font-family:'Inter',sans-serif;
                 transition:all 0.15s; }}
  .toggle-btn.active {{ background:var(--accent); color:#fff; font-weight:600; }}

  .main {{ display:flex; gap:16px; flex:1; min-height:0; }}

  #chart-wrap {{ flex:1; min-width:0; background:var(--white); border:1px solid var(--border);
                 border-radius:14px; padding:12px; display:flex; flex-direction:column; }}
  #chart-title {{ font-size:12px; font-weight:600; color:var(--muted); margin-bottom:6px; flex-shrink:0; }}
  #line-chart {{ flex:1; min-height:0; }}

  #table-wrap {{ width:620px; flex-shrink:0; background:var(--white); border:1px solid var(--border);
                 border-radius:14px; padding:16px; overflow-y:auto; display:flex; flex-direction:column; gap:12px; }}
  #bar-section {{ flex-shrink:0; }}
  #bar-section h3 {{ font-size:11px; font-weight:700; color:var(--muted); text-transform:uppercase;
                     letter-spacing:0.8px; margin-bottom:6px; }}
  #drop-bar {{ width:100%; }}
  #table-wrap h3 {{ font-size:11px; font-weight:700; color:var(--muted); text-transform:uppercase;
                    letter-spacing:0.8px; margin-bottom:12px; }}

  table {{ width:100%; border-collapse:collapse; font-size:11px; }}
  th {{ font-size:10px; font-weight:600; color:var(--muted); text-align:center;
        padding:4px 6px; border-bottom:2px solid var(--border); white-space:nowrap; }}
  th.left {{ text-align:left; }}
  td {{ padding:5px 6px; text-align:center; border-bottom:1px solid var(--border); }}
  td.ev-label {{ text-align:left; font-weight:600; font-size:11px; white-space:nowrap; }}
  td.pers {{ color:var(--muted); font-size:10px; }}
  .cell-inner {{ display:flex; flex-direction:column; align-items:center; gap:1px; }}
  .rho-val {{ font-weight:600; font-size:11px; }}
  .pct-val {{ font-size:9px; color:var(--muted); }}
  .drop-big {{ background:#fde8e8; border-radius:4px; }}
  .drop-big .rho-val {{ color:var(--red); }}
  .drop-big .pct-val {{ color:var(--red); font-weight:600; }}
  .drop-mod {{ background:#fef3cd; border-radius:4px; }}
  .drop-mod .pct-val {{ color:var(--amber); font-weight:600; }}
  .alpha-25 {{ background:#e8f0fe; }}
  .alpha-25 .rho-val {{ color:var(--accent); }}
  tr:hover td {{ background:#f8f9ff; }}
  .legend-note {{ font-size:10px; color:var(--muted); margin-top:10px; line-height:1.6; }}
</style>
</head>
<body>

<div class="top-bar">
  <div>
    <h1>Ratio Analysis — ρ = cost share / duration (%/day)</h1>
    <p class="subtitle">Top-10 most persistent events across all α values · vertical line marks α=25% (chosen threshold)</p>
  </div>
  <div class="toggle-group">
    <button class="toggle-btn active" onclick="setDataset('future',this)">Future</button>
    <button class="toggle-btn"        onclick="setDataset('historical',this)">Historical</button>
  </div>
</div>

<div class="main">
  <div id="chart-wrap">
    <div id="chart-title">ρ vs α — one line per event (hover for details)</div>
    <div id="line-chart"></div>
  </div>
  <div id="table-wrap">
    <h3>ρ values &amp; drop vs previous α</h3>
    <table id="ratio-table">
      <thead id="thead"></thead>
      <tbody id="tbody"></tbody>
    </table>
    <p class="legend-note">
      <span style="display:inline-block;width:10px;height:10px;background:#fde8e8;border-radius:2px;margin-right:4px"></span>drop ≥ 30% vs prev α &nbsp;
      <span style="display:inline-block;width:10px;height:10px;background:#fef3cd;border-radius:2px;margin-right:4px"></span>drop 15–30% &nbsp;
      <span style="display:inline-block;width:10px;height:10px;background:#e8f0fe;border-radius:2px;margin-right:4px"></span>α = 25%
    </p>
    <div id="bar-section">
      <h3>Biggest drop concentration by α transition</h3>
      <div id="drop-bar"></div>
    </div>
  </div>
</div>

<script>
const ALL    = {data_json};
const ALPHAS = {alphas_json};
const COLORS = {colors_json};
const HL_A   = {HIGHLIGHT_A};

let currentDs = 'future';
let selectedIdx = null;

function setDataset(ds, btn) {{
  currentDs = ds;
  selectedIdx = null;
  document.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  render();
}}

function selectEvent(idx) {{
  const n      = ALL[currentDs].length;
  const events = ALL[currentDs];
  if (selectedIdx === idx) {{
    selectedIdx = null;
    // restore all traces to full interactivity
    const updates = events.map((ev, i) => ({{
      opacity:       [1],
      hovertemplate: ['%{{text}}<extra></extra>'],
    }}));
    updates.forEach((u, i) => Plotly.restyle('line-chart', u, [i]));
    document.querySelectorAll('#tbody tr').forEach(r => r.style.background = '');
  }} else {{
    selectedIdx = idx;
    events.forEach((ev, i) => {{
      Plotly.restyle('line-chart', {{
        opacity:       [i === idx ? 1 : 0.07],
        hovertemplate: [i === idx ? '%{{text}}<extra></extra>' : '<extra></extra>'],
      }}, [i]);
    }});
    document.querySelectorAll('#tbody tr').forEach((r, i) => {{
      r.style.background = i === idx ? '#eef2ff' : '';
    }});
  }}
}}

function render() {{
  const events = ALL[currentDs];
  renderChart(events);
  renderTable(events);
  renderDropBar(events);
}}

function renderDropBar(events) {{
  // transitions: [10→15, 15→20, 20→25, 25→30, 30→35, 35→40]
  const transitions = [];
  for (let i = 1; i < ALPHAS.length; i++)
    transitions.push({{from: ALPHAS[i-1], to: ALPHAS[i], label: `${{ALPHAS[i-1]}}%→${{ALPHAS[i]}}%`}});

  const counts = transitions.map(() => 0);
  const evNames = transitions.map(() => []);

  events.forEach(ev => {{
    const bd = biggestDrop(ev);
    if (!bd) return;
    const ti = transitions.findIndex(t => t.from === bd.from && t.to === bd.to);
    if (ti >= 0) {{ counts[ti]++; evNames[ti].push(`${{ev.label}} (${{bd.chg.toFixed(1)}}%)`); }}
  }});

  const maxCount = Math.max(...counts, 1);
  const colors = counts.map((c, i) => {{
    const isHL = transitions[i].from === HL_A || transitions[i].to === HL_A;
    return c === maxCount ? '#e74c3c' : isHL ? '#4776e6' : '#94a3b8';
  }});

  Plotly.react('drop-bar', [{{
    type: 'bar',
    x: transitions.map(t => t.label),
    y: counts,
    marker: {{ color: colors, borderRadius: 4 }},
    hovertemplate: '<b>%{{x}}</b><br>%{{y}} event(s)<extra></extra>',
  }}], {{
    height: 160,
    margin: {{t:8, r:8, b:40, l:30}},
    paper_bgcolor: '#ffffff', plot_bgcolor: '#ffffff',
    font: {{family:'Inter', size:10}},
    xaxis: {{ tickfont:{{size:10}}, gridcolor:'#f0f0f0' }},
    yaxis: {{ tickfont:{{size:10}}, gridcolor:'#f0f0f0', zeroline:false,
              dtick:1, range:[0, maxCount + 0.5] }},
    showlegend: false,
  }}, {{responsive:true, displayModeBar:false}});
}}

function renderChart(events) {{
  const traces = events.map((ev, i) => ({{
    type: 'scatter', mode: 'lines+markers', name: ev.label,
    x: ALPHAS,
    y: ev.rho,
    line: {{ color: COLORS[i % COLORS.length], width: 2 }},
    marker: {{ size: 7, color: COLORS[i % COLORS.length] }},
    text: ev.rho.map((r, j) => r !== null
      ? `<b>${{ev.label}}</b> at α=${{ALPHAS[j]}}%<br>ρ = ${{r.toFixed(5)}} %/day<br>cost share = ${{ev.cs[j]}}%<br>duration = ${{ev.dur[j]}}d`
      : `${{ev.label}}: not in top-10 at α=${{ALPHAS[j]}}%`),
    hovertemplate: '%{{text}}<extra></extra>',
    connectgaps: false,
    persistence: ev.persistence,
  }}));

  // vertical line at alpha=25
  const hlIdx  = ALPHAS.indexOf(HL_A);
  const hlX    = HL_A;
  const yVals  = events.flatMap(e => e.rho.filter(v => v !== null));
  const yMax   = yVals.length ? Math.max(...yVals) * 1.08 : 1;

  traces.push({{
    type: 'scatter', mode: 'lines', name: 'α = 25% (chosen)',
    x: [hlX, hlX], y: [0, yMax],
    line: {{ color: '#4776e6', width: 1.5, dash: 'dot' }},
    hoverinfo: 'skip', showlegend: true,
  }});

  const layout = {{
    xaxis: {{
      title: {{ text: 'α (%)', font: {{size:12,family:'Inter'}} }},
      tickvals: ALPHAS, ticktext: ALPHAS.map(a => a+'%'),
      tickfont: {{size:11,family:'Inter'}}, gridcolor:'#f0f0f0', zeroline:false,
    }},
    yaxis: {{
      title: {{ text: 'ρ (%/day)', font: {{size:12,family:'Inter'}} }},
      tickfont: {{size:11,family:'Inter'}}, gridcolor:'#f0f0f0', zeroline:false,
      rangemode: 'tozero',
    }},
    legend: {{ font:{{size:11,family:'Inter'}}, orientation:'v', x:1.01, y:1,
               bgcolor:'rgba(255,255,255,0.9)', bordercolor:'#e5e7ef', borderwidth:1 }},
    margin: {{t:10, r:130, b:50, l:60}},
    paper_bgcolor:'#ffffff', plot_bgcolor:'#ffffff',
    font: {{family:'Inter'}}, hovermode:'closest',
  }};
  Plotly.react('line-chart', traces, layout, {{responsive:true, displayModeBar:false}});
}}

function biggestDrop(ev) {{
  let worstChg = 0, worstJ = -1;
  ev.pct_chg.forEach((chg, j) => {{
    if (chg !== null && chg < worstChg) {{ worstChg = chg; worstJ = j; }}
  }});
  if (worstJ === -1) return null;
  return {{ j: worstJ, from: ALPHAS[worstJ-1], to: ALPHAS[worstJ], chg: worstChg }};
}}

function renderTable(events) {{
  // header
  const thead = document.getElementById('thead');
  thead.innerHTML = '<tr><th class="left">Event</th><th class="left">Pers.</th>' +
    ALPHAS.map(a => `<th${{a===HL_A?' class="alpha-25"':''}}>${{a}}%</th>`).join('') +
    '<th class="left" style="padding-left:8px">Biggest drop</th></tr>';

  const tbody = document.getElementById('tbody');
  tbody.innerHTML = events.map((ev, ei) => {{
    const bd = biggestDrop(ev);
    const cells = ev.rho.map((r, j) => {{
      const a      = ALPHAS[j];
      const chg    = ev.pct_chg[j];
      const isHL   = a === HL_A;
      const isBigDrop = bd && j === bd.j;
      const isBig  = chg !== null && chg <= -30;
      const isMod  = chg !== null && chg > -30 && chg <= -15;
      let cls = isHL ? 'alpha-25' : isBig ? 'drop-big' : isMod ? 'drop-mod' : '';
      if (r === null) return `<td class="${{cls}}" style="color:var(--muted)">—</td>`;
      const chgStr = chg !== null ? (chg >= 0 ? `+${{chg.toFixed(1)}}%` : `${{chg.toFixed(1)}}%`) : '';
      const outline = isBigDrop ? ' outline:2px solid var(--red);outline-offset:-2px;border-radius:4px;' : '';
      return `<td class="${{cls}}" style="${{outline}}"><div class="cell-inner">
        <span class="rho-val">${{r.toFixed(4)}}</span>
        ${{chgStr ? `<span class="pct-val">${{chgStr}}</span>` : ''}}
      </div></td>`;
    }}).join('');
    const bdCell = bd
      ? `<td style="padding-left:8px;white-space:nowrap;font-size:10px">
           <span style="color:var(--red);font-weight:600">${{bd.chg.toFixed(1)}}%</span>
           <span style="color:var(--muted)"> at ${{bd.from}}%→${{bd.to}}%</span>
         </td>`
      : `<td style="color:var(--muted);padding-left:8px">—</td>`;
    return `<tr>
      <td class="ev-label" style="color:${{COLORS[ei % COLORS.length]}}">${{ev.label}}</td>
      <td class="pers">${{ev.persistence}}/7</td>
      ${{cells}}
      ${{bdCell}}
    </tr>`
    .replace('<tr>', `<tr onclick="selectEvent(${{ei}})" style="cursor:pointer">`);
  }}).join('');
}}

render();

// click on a line in the chart to select/deselect that event
document.getElementById('line-chart').on('plotly_click', function(data) {{
  const traceIdx = data.points[0].curveNumber;
  const n = ALL[currentDs].length;
  if (traceIdx < n) {{   // ignore the vertical reference line trace
    selectEvent(traceIdx);
  }}
}});
</script>
</body>
</html>"""

out = os.path.expanduser('~/Desktop/Bachelor Thesis/analysis/ratio_analysis.html')
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, 'w', encoding='utf-8') as f:
    f.write(html)
print(f'Saved: {out}')
