"""
Interactive Correlation Matrix and Zone Map
============================================
Generates an interactive HTML with two views:
  1. Correlation matrix — 27×27 Pearson correlations across stress events,
     selectable by dataset (future/historical) and feature type (wind/PV/heat/combined)
  2. Zone map — Europe coloured by geographic zone, derived from Ward clustering
     on (1 - pairwise correlation) distance at threshold THRESH_CORR

Inputs:
  CSVs_and_JSONs/future/features/features_alpha25_future_per_country.csv
  CSVs_and_JSONs/historical/features/features_historical_per_country.csv

Outputs:
  analysis/htmls_uso/correlation_zones_interactive.html
"""

# ── Configuration ─────────────────────────────────────────────────────────────
# Output type: HTML interactive
THRESH_CORR = 0.65           # minimum pairwise correlation to merge into same zone
OUT_HTML    = '~/Desktop/Bachelor Thesis/analysis/htmls_uso/correlation_zones_interactive.html'
# ──────────────────────────────────────────────────────────────────────────────

import os, json
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

OUT_HTML    = os.path.expanduser(OUT_HTML)
THRESH_DIST = 1 - THRESH_CORR   # Ward distance threshold (= 1 - r_min)

COUNTRIES = ['AT','BE','BG','CH','CZ','DE','DK','EE','EL','ES',
             'FI','FR','HR','HU','IE','IT','LT','LV','NL',
             'NO','PL','PT','RO','SE','SI','SK','UK']

DATASETS = {
    'future':     os.path.expanduser('~/Desktop/Bachelor Thesis/CSVs_and_JSONs/future/features/features_alpha25_future_per_country.csv'),
    'historical': os.path.expanduser('~/Desktop/Bachelor Thesis/CSVs_and_JSONs/historical/features/features_historical_per_country.csv'),
}

OUT_HTML = os.path.expanduser(
    '~/Desktop/Bachelor Thesis/analysis/htmls_uso/correlation_zones_interactive.html')

ISO3 = {
    'AT':'AUT','BE':'BEL','BG':'BGR','CH':'CHE','CZ':'CZE',
    'DE':'DEU','DK':'DNK','EE':'EST','EL':'GRC','ES':'ESP',
    'FI':'FIN','FR':'FRA','HR':'HRV','HU':'HUN','IE':'IRL',
    'IT':'ITA','LT':'LTU','LV':'LVA','NL':'NLD','NO':'NOR',
    'PL':'POL','PT':'PRT','RO':'ROU','SE':'SWE','SI':'SVN',
    'SK':'SVK','UK':'GBR',
}

COUNTRY_NAMES = {
    'AT':'Austria',    'BE':'Belgium',     'BG':'Bulgaria',
    'CH':'Switzerland','CZ':'Czech Rep.',  'DE':'Germany',
    'DK':'Denmark',    'EE':'Estonia',     'EL':'Greece',
    'ES':'Spain',      'FI':'Finland',     'FR':'France',
    'HR':'Croatia',    'HU':'Hungary',     'IE':'Ireland',
    'IT':'Italy',      'LT':'Lithuania',   'LV':'Latvia',
    'NL':'Netherlands','NO':'Norway',      'PL':'Poland',
    'PT':'Portugal',   'RO':'Romania',     'SE':'Sweden',
    'SI':'Slovenia',   'SK':'Slovakia',    'UK':'United Kingdom',
}

# ── helpers ───────────────────────────────────────────────────────────────────

def get_zones(corr_matrix):
    dist = 1 - corr_matrix.values
    np.fill_diagonal(dist, 0)
    dist = (dist + dist.T) / 2
    condensed = squareform(np.clip(dist, 0, None))
    Z = linkage(condensed, method='ward')
    return fcluster(Z, t=THRESH_DIST, criterion='distance')


def compute_panel(corr, labels, show_zones=False):
    order      = np.argsort(labels)
    corr_ord   = corr.iloc[order, :].iloc[:, order]
    sorted_lbl = labels[order]
    countries  = list(corr_ord.columns)

    z = [[round(float(v), 3) for v in row] for row in corr_ord.values.tolist()]

    text = []
    for i, ci in enumerate(countries):
        row = []
        for j, cj in enumerate(countries):
            row.append(f'<b>{ci} × {cj}</b><br>Pearson r: {corr_ord.iloc[i,j]:.3f}')
        text.append(row)

    shapes, annotations, zones = [], [], []
    n_zones  = int(sorted_lbl.max())
    zone_num = 0
    pos      = 0

    for z_id in range(1, n_zones + 1):
        idx     = np.where(sorted_lbl == z_id)[0]
        size    = len(idx)
        members = [countries[i] for i in idx]
        draw    = True if size == 1 else (
            min(corr_ord.iloc[int(a), int(b)]
                for ii, a in enumerate(idx) for b in idx[ii + 1:])
            >= THRESH_CORR
        )
        if draw:
            zone_num += 1
            zones.append({'label': f'Z{zone_num}', 'countries': members})
            if show_zones:
                shapes.append({
                    'type': 'rect',
                    'x0': pos - 0.5, 'x1': pos + size - 0.5,
                    'y0': pos - 0.5, 'y1': pos + size - 0.5,
                    'line': {'color': 'black', 'width': 2.5},
                    'fillcolor': 'rgba(0,0,0,0)',
                })
                center = pos + (size - 1) / 2
                annotations.append({
                    'x': center, 'y': center,
                    'text': f'Z{zone_num}',
                    'showarrow': False,
                    'font': {'size': 9, 'color': '#000', 'family': 'Inter'},
                    'bgcolor': 'rgba(255,255,255,0.75)',
                    'borderpad': 2,
                    'xanchor': 'center', 'yanchor': 'middle',
                })
        pos += size

    return {
        'z': z, 'text': text, 'countries': countries,
        'shapes': shapes, 'annotations': annotations, 'zones': zones,
    }

# ── compute all panels ────────────────────────────────────────────────────────

all_data = {}

for ds_name, csv_path in DATASETS.items():
    if not os.path.exists(csv_path):
        print(f'CSV not found: {csv_path} — skipping')
        continue

    df_feat = pd.read_csv(csv_path)
    print(f'Loaded {ds_name}: {len(df_feat)} rows')

    corr_matrices = {}
    for feat in ['wind anom', 'pv anom', 'heat anom']:
        cols = [f'{feat} {c}' for c in COUNTRIES]
        corr = pd.DataFrame(df_feat[cols].values, columns=COUNTRIES).corr().fillna(0)
        corr_matrices[feat] = corr

    corr_avg = pd.DataFrame(
        np.mean([m.values for m in corr_matrices.values()], axis=0),
        index=COUNTRIES, columns=COUNTRIES
    )

    labels = get_zones(corr_avg)
    print(f'  Zones: {int(labels.max())}')

    all_data[ds_name] = {
        'combined': compute_panel(corr_avg, labels, show_zones=True),
        'wind':     compute_panel(corr_matrices['wind anom'], labels),
        'pv':       compute_panel(corr_matrices['pv anom'], labels),
        'heat':     compute_panel(corr_matrices['heat anom'], labels),
    }

data_json         = json.dumps(all_data)
iso3_json         = json.dumps(ISO3)
country_names_json = json.dumps(COUNTRY_NAMES)

# ── HTML ──────────────────────────────────────────────────────────────────────

html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Correlation Zones — Interactive</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  :root {{
    --bg: #f4f6fb; --white: #ffffff; --text: #1a1d2e;
    --muted: #6b7280; --border: #e5e7ef; --accent: #4776e6;
  }}
  body {{ font-family:'Inter',sans-serif; background:var(--bg); color:var(--text); padding:32px 40px 60px; }}
  h1 {{ font-size:20px; font-weight:700; margin-bottom:4px; }}
  .subtitle {{ font-size:13px; color:var(--muted); margin-bottom:28px; }}
  .toggle-row {{ display:flex; gap:8px; margin-bottom:16px; flex-wrap:wrap; align-items:center; }}
  .toggle-group {{
    display:flex; background:var(--white); border:1px solid var(--border);
    border-radius:10px; overflow:hidden;
  }}
  .toggle-btn {{
    padding:8px 18px; font-size:12px; font-weight:500; color:var(--muted);
    cursor:pointer; border:none; background:none; font-family:'Inter',sans-serif; transition:all 0.15s;
  }}
  .toggle-btn:hover {{ background:var(--bg); color:var(--text); }}
  .toggle-btn.active {{ background:var(--accent); color:#fff; font-weight:600; }}
  #main-row {{ display:flex; gap:16px; align-items:flex-start; }}
  #plot-container {{
    flex:1; min-width:0; background:var(--white); border:1px solid var(--border);
    border-radius:14px; overflow:hidden; box-shadow:0 2px 8px rgba(0,0,0,0.04); padding:8px;
  }}
  #corr-plot, #map-plot {{ width:100%; height:720px; }}
  #map-plot {{ display:none; }}
  #zone-panel {{
    width:190px; flex-shrink:0; background:var(--white); border:1px solid var(--border);
    border-radius:14px; box-shadow:0 2px 8px rgba(0,0,0,0.04); padding:16px;
    max-height:740px; overflow-y:auto; display:none;
  }}
  #zone-panel h3 {{
    font-size:11px; font-weight:700; color:var(--muted);
    text-transform:uppercase; letter-spacing:0.8px; margin-bottom:12px;
  }}
  .zone-item {{ margin-bottom:12px; }}
  .zone-dot {{
    display:inline-block; width:10px; height:10px;
    border-radius:50%; margin-right:5px; flex-shrink:0;
  }}
  .zone-label {{ font-size:12px; font-weight:700; display:flex; align-items:center; margin-bottom:3px; }}
  .zone-countries {{ font-size:11px; color:var(--text); line-height:1.6; padding-left:15px; }}
</style>
</head>
<body>

<h1>Correlation Zones — Country clustering at corr ≥ 0.65</h1>
<p class="subtitle">Ward hierarchical clustering on wind, PV and heat anomaly correlations across 27 European countries.</p>

<div class="toggle-row">
  <div class="toggle-group" id="dsGroup">
    <button class="toggle-btn active" onclick="setDataset('future')">Future</button>
    <button class="toggle-btn" onclick="setDataset('historical')">Historical</button>
  </div>
  <div class="toggle-group" id="viewGroup">
    <button class="toggle-btn active" onclick="setView('matrix')">Correlation Matrix</button>
    <button class="toggle-btn" onclick="setView('map')">Europe Map</button>
  </div>
  <div class="toggle-group" id="featGroup">
    <button class="toggle-btn active" onclick="setFeature('combined')">All features</button>
    <button class="toggle-btn" onclick="setFeature('wind')">Wind</button>
    <button class="toggle-btn" onclick="setFeature('pv')">PV</button>
    <button class="toggle-btn" onclick="setFeature('heat')">Heat</button>
  </div>
</div>

<div id="main-row">
  <div id="plot-container">
    <div id="corr-plot"></div>
    <div id="map-plot"></div>
  </div>
  <div id="zone-panel">
    <h3>Zones</h3>
    <div id="zone-list"></div>
  </div>
</div>

<script>
const allData       = {data_json};
const ISO3          = {iso3_json};
const COUNTRY_NAMES = {country_names_json};

let currentDs   = 'future';
let currentFeat = 'combined';
let currentView = 'matrix';

const FEAT_LABELS = {{
  combined: 'Combined (wind + PV + heat)',
  wind: 'Wind anomaly', pv: 'PV anomaly', heat: 'Heat anomaly',
}};

// Per-country color overrides: {{ dataset: {{ ISO3: [color, label] }} }}
const COLOR_OVERRIDES = {{
  historical: {{
    'ITA': ['#FF6600', 'Italy'],
  }},
}};

const ZONE_PALETTE = [
  '#4776e6','#e96c7c','#8B4513','#f7971e','#8e44ad',
  '#e74c3c','#3498db','#1abc9c','#FF00FF','#c0392b',
  '#27ae60','#d35400','#7f8c8d',
];

// RdYlBu_r
const COLORSCALE = [
  [0.0,'#313695'],[0.1,'#4575B4'],[0.2,'#74ADD1'],
  [0.3,'#ABD9E9'],[0.4,'#E0F3F8'],[0.5,'#FFFFBF'],
  [0.6,'#FEE090'],[0.7,'#FDAE61'],[0.8,'#F46D43'],
  [0.9,'#D73027'],[1.0,'#A50026'],
];

function renderMatrix() {{
  const panel   = allData[currentDs][currentFeat];
  const dsLabel = currentDs.charAt(0).toUpperCase() + currentDs.slice(1);

  const trace = {{
    type: 'heatmap',
    z: panel.z, x: panel.countries, y: panel.countries,
    text: panel.text,
    hovertemplate: '%{{text}}<extra></extra>',
    colorscale: COLORSCALE, zmin: -1, zmax: 1,
    xgap: 1, ygap: 1,
    texttemplate: '%{{z:.2f}}',
    textfont: {{ size: 6, color: '#333' }},
    colorbar: {{
      title: {{ text: 'Pearson r', side: 'right', font: {{ size: 11 }} }},
      thickness: 16, len: 0.9, tickfont: {{ size: 10 }},
    }},
  }};

  const layout = {{
    title: {{ text: `${{dsLabel}} — ${{FEAT_LABELS[currentFeat]}}`, font: {{ size: 14, family: 'Inter', color: '#1a1d2e' }}, x: 0.5 }},
    xaxis: {{ tickfont: {{ size: 10, family: 'Inter' }}, tickangle: -45, side: 'bottom' }},
    yaxis: {{ tickfont: {{ size: 10, family: 'Inter' }}, autorange: 'reversed' }},
    shapes: panel.shapes, annotations: panel.annotations,
    margin: {{ t: 50, r: 90, b: 110, l: 60 }},
    paper_bgcolor: '#ffffff', plot_bgcolor: '#ffffff', font: {{ family: 'Inter' }},
  }};

  Plotly.react('corr-plot', [trace], layout, {{
    responsive: true, displayModeBar: true, displaylogo: false,
    modeBarButtonsToRemove: ['select2d','lasso2d','autoScale2d'],
  }});
}}

function renderMap() {{
  const zones   = allData[currentDs]['combined'].zones;
  const dsLabel = currentDs.charAt(0).toUpperCase() + currentDs.slice(1);

  const overrides = COLOR_OVERRIDES[currentDs] || {{}};

  const traces = zones.map((zone, i) => {{
    const color = ZONE_PALETTE[i % ZONE_PALETTE.length];
    const pairs = zone.countries
      .map(c => [ISO3[c], COUNTRY_NAMES[c] || c])
      .filter(([iso3]) => iso3 && !overrides[iso3]);
    if (!pairs.length) return null;
    const iso3s = pairs.map(p => p[0]);
    const names = pairs.map(p => p[1]);
    return {{
      type: 'choropleth',
      locations: iso3s,
      z: Array(iso3s.length).fill(i + 1),
      text: names.map(n => `<b>${{n}}</b><br>${{zone.label}}`),
      hovertemplate: '%{{text}}<extra></extra>',
      colorscale: [[0, color],[1, color]],
      zmin: 0, zmax: zones.length,
      showscale: false,
      name: zone.label,
      showlegend: true,
      marker: {{ line: {{ color: '#fff', width: 0.8 }} }},
    }};
  }}).filter(Boolean);

  // Add override traces
  Object.entries(overrides).forEach(([iso3, [color, label]]) => {{
    traces.push({{
      type: 'choropleth',
      locations: [iso3],
      z: [0],
      text: [`<b>${{label}}</b><br>(override)`],
      hovertemplate: '%{{text}}<extra></extra>',
      colorscale: [[0, color],[1, color]],
      zmin: 0, zmax: 1,
      showscale: false,
      name: label,
      showlegend: true,
      marker: {{ line: {{ color: '#fff', width: 0.8 }} }},
    }});
  }});

  const layout = {{
    title: {{ text: `${{dsLabel}} — Correlation zones`, font: {{ size: 14, family: 'Inter', color: '#1a1d2e' }}, x: 0.5 }},
    geo: {{
      scope: 'europe',
      showframe: false,
      showcoastlines: true,
      coastlinecolor: '#ccc',
      showland: true,
      landcolor: '#f0f0f0',
      showocean: true,
      oceancolor: '#e8f4f8',
      showcountries: true,
      countrycolor: '#ddd',
      projection: {{ type: 'mercator' }},
      lonaxis: {{ range: [-15, 35] }},
      lataxis: {{ range: [34, 72] }},
    }},
    legend: {{
      title: {{ text: 'Zones', font: {{ size: 11 }} }},
      font: {{ size: 11, family: 'Inter' }},
      bgcolor: 'rgba(255,255,255,0.9)',
      bordercolor: '#e5e7ef',
      borderwidth: 1,
    }},
    margin: {{ t: 50, r: 20, b: 20, l: 20 }},
    paper_bgcolor: '#ffffff',
    font: {{ family: 'Inter' }},
  }};

  Plotly.react('map-plot', traces, layout, {{
    responsive: true, displayModeBar: false,
  }});
}}

function render() {{
  const showMatrix = (currentView === 'matrix');
  document.getElementById('corr-plot').style.display = showMatrix ? 'block' : 'none';
  document.getElementById('map-plot').style.display  = showMatrix ? 'none'  : 'block';
  document.getElementById('featGroup').style.display = showMatrix ? 'flex'  : 'none';

  if (showMatrix) renderMatrix(); else renderMap();

  // Zone side panel — always show for combined matrix or any map view
  const showPanel = (currentView === 'map') || (currentView === 'matrix' && currentFeat === 'combined');
  const zones     = allData[currentDs]['combined'].zones;
  const zonePanel = document.getElementById('zone-panel');
  const zoneList  = document.getElementById('zone-list');
  zonePanel.style.display = showPanel ? 'block' : 'none';
  if (showPanel) {{
    zoneList.innerHTML = zones.map((z, i) => {{
      const color = ZONE_PALETTE[i % ZONE_PALETTE.length];
      return `<div class="zone-item">
        <div class="zone-label">
          <span class="zone-dot" style="background:${{color}}"></span>${{z.label}}
        </div>
        <div class="zone-countries">${{z.countries.map(c => COUNTRY_NAMES[c] || c).join(', ')}}</div>
      </div>`;
    }}).join('');
  }}
}}

function setDataset(ds) {{
  currentDs = ds;
  document.querySelectorAll('#dsGroup .toggle-btn').forEach(b => {{
    b.classList.toggle('active', b.textContent.toLowerCase() === ds);
  }});
  render();
}}

function setView(v) {{
  currentView = v;
  document.querySelectorAll('#viewGroup .toggle-btn').forEach(b => {{
    b.classList.toggle('active', b.onclick.toString().includes(`'${{v}}'`));
  }});
  render();
}}

function setFeature(feat) {{
  currentFeat = feat;
  document.querySelectorAll('#featGroup .toggle-btn').forEach(b => {{
    b.classList.toggle('active', b.onclick.toString().includes(`'${{feat}}'`));
  }});
  render();
}}

// URL param — hide dataset toggle if loaded from parent dashboard
const initDs = new URLSearchParams(window.location.search).get('dataset');
if (initDs && allData[initDs]) {{
  currentDs = initDs;
  document.getElementById('dsGroup').style.display = 'none';
  document.querySelectorAll('#dsGroup .toggle-btn').forEach(b => {{
    b.classList.toggle('active', b.textContent.toLowerCase() === initDs);
  }});
}}
render();
</script>
</body>
</html>'''

os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
with open(OUT_HTML, 'w') as f:
    f.write(html)
print(f'Saved: {OUT_HTML}')
