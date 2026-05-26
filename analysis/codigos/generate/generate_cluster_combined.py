"""
Combined Cluster Map — Climate + Operational Side by Side
==========================================================
Generates 2 HTML files showing, for each stress event, the climate anomaly map
(left panel) and the operational anomaly map (right panel) simultaneously.
The user can select an event and a feature from a dropdown; both maps update.

Inputs:
  CSVs/future/features/features_alpha25_climate_future_clustered.csv
  CSVs/future/features/features_alpha25_operational_future_clustered.csv
  CSVs/historical/features/features_alpha25_climate_historical_clustered.csv
  CSVs/historical/features/features_alpha25_operational_historical_clustered.csv

Outputs:
  htmls/htmls_uso/cluster_combined_future.html
  htmls/htmls_uso/cluster_combined_historical.html
"""

# ── Configuration ─────────────────────────────────────────────────────────────
# Output type: HTML interactive  (2 files: future + historical)
THRESH_CS  = 0.2   # minimum cost_share_cross (%) to include an event
THRESH_DUR = 1.0   # minimum duration (days) to include an event
OUT_DIR    = '~/Desktop/Bachelor Thesis/analysis/htmls_uso'
# ──────────────────────────────────────────────────────────────────────────────

import os, json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

OUT_DIR = os.path.expanduser(OUT_DIR)

ZONES_FUTURE = {
    'Nordics_N':     ['FI', 'NO'],
    'Baltics':       ['EE', 'LV'],
    'Nordics_S':     ['DK', 'SE'],
    'British_Isles': ['IE', 'UK'],
    'NW_Continent':  ['DE', 'BE', 'NL'],
    'Med_East_N':    ['BG', 'RO'],
    'Med_East_S':    ['HR', 'SI'],
    'Greece':        ['EL'],
    'Italy':         ['IT'],
    'Iberia':        ['ES', 'PT'],
    'Central_W':     ['CH', 'FR'],
    'East_N':        ['LT', 'PL'],
    'Central_E':     ['HU', 'SK', 'AT', 'CZ'],
}

ZONES_HISTORICAL = {
    'Norway':        ['NO'],
    'Baltics_FI':    ['EE', 'LV', 'FI'],
    'Nordics_S':     ['DK', 'SE'],
    'British_Isles': ['IE', 'UK'],
    'NW_Continent':  ['DE', 'BE', 'NL'],
    'Med_East_N':    ['BG', 'RO'],
    'Med_East_S':    ['HR', 'SI'],
    'Greece':        ['EL'],
    'Italy':         ['IT'],
    'Iberia':        ['ES', 'PT'],
    'Central_W':     ['CH', 'FR'],
    'East_N':        ['LT', 'PL'],
    'Central_E':     ['HU', 'SK', 'AT', 'CZ'],
}

CODE_MAP = {
    'AT': 'AUT', 'BE': 'BEL', 'BG': 'BGR', 'CH': 'CHE', 'CZ': 'CZE',
    'DE': 'DEU', 'DK': 'DNK', 'EE': 'EST', 'EL': 'GRC', 'ES': 'ESP',
    'FI': 'FIN', 'FR': 'FRA', 'HR': 'HRV', 'HU': 'HUN', 'IE': 'IRL',
    'IT': 'ITA', 'LT': 'LTU', 'LV': 'LVA', 'NL': 'NLD', 'NO': 'NOR',
    'PL': 'POL', 'PT': 'PRT', 'RO': 'ROU', 'SE': 'SWE', 'SI': 'SVN',
    'SK': 'SVK', 'UK': 'GBR',
}

CLUSTER_COLORS = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12',
                  '#1abc9c', '#e67e22', '#34495e', '#e91e63', '#00bcd4']

ALPHA      = 25
THRESH_CS  = 0.2
THRESH_DUR = 1.0

CLIMATE_FEATS = {
    'feat_types':  ['wind anom', 'pv anom', 'heat anom'],
    'feat_labels': {
        'wind anom': 'Wind CF anomaly (%)',
        'pv anom':   'PV CF anomaly (%)',
        'heat anom': 'Heat demand anomaly (%)',
    },
    'feat_units': {'wind anom': '%', 'pv anom': '%', 'heat anom': '%'},
    'fixed_scale': {'wind anom': 100, 'pv anom': 100, 'heat anom': 100},
    'cluster_col': 'cluster_climate',
}

OPERATIONAL_FEATS = {
    'feat_types':  ['nettrans anom', 'stor level anom', 'stor discharge anom'],
    'feat_labels': {
        'nettrans anom':       'Net transfer anomaly (GWh/h)',
        'stor level anom':     'Storage level anomaly (GWh)',
        'stor discharge anom': 'Storage discharge anomaly (GWh/h)',
    },
    'feat_units': {
        'nettrans anom':       'GWh/h',
        'stor level anom':     'GWh',
        'stor discharge anom': 'GWh/h',
    },
    'fixed_scale': {'stor discharge anom': 10},
    'cluster_col': 'cluster_operational',
}

DATASETS = {
    'future': {
        'climate_csv':     os.path.expanduser('~/Desktop/Bachelor Thesis/CSVs/future/features/features_alpha25_climate_future_clustered.csv'),
        'operational_csv': os.path.expanduser('~/Desktop/Bachelor Thesis/CSVs/future/features/features_alpha25_operational_future_clustered.csv'),
        'events_csv':      os.path.expanduser(f'~/Desktop/Bachelor Thesis/CSVs/future/events_global_alpha/events_global_alpha{ALPHA:02d}.csv'),
        'out_html':        os.path.expanduser('~/Desktop/Bachelor Thesis/analysis/htmls_uso/cluster_combined_future.html'),
        'zones':           ZONES_FUTURE,
    },
    'historical': {
        'climate_csv':     os.path.expanduser('~/Desktop/Bachelor Thesis/CSVs/historical/features/features_alpha25_climate_historical_clustered.csv'),
        'operational_csv': os.path.expanduser('~/Desktop/Bachelor Thesis/CSVs/historical/features/features_alpha25_operational_historical_clustered.csv'),
        'events_csv':      os.path.expanduser(f'~/Desktop/Bachelor Thesis/CSVs/historical/events_global_alpha/events_global_historical_alpha{ALPHA:02d}.csv'),
        'out_html':        os.path.expanduser('~/Desktop/Bachelor Thesis/analysis/htmls_uso/cluster_combined_historical.html'),
        'zones':           ZONES_HISTORICAL,
    },
}


def compute_medoids(df_feat, cluster_col, meta_cols):
    excl = set(meta_cols) | {cluster_col}
    feat_cols = [c for c in df_feat.columns if c not in excl]
    X_sc  = StandardScaler().fit_transform(df_feat[feat_cols].values)
    X_pca = PCA(n_components=0.90, random_state=42).fit_transform(X_sc)
    labels = df_feat[cluster_col].values
    k = int(labels.max()) + 1
    medoids = set()
    for c in range(k):
        mask = np.where(labels == c)[0]
        dists = pairwise_distances(X_pca[mask]).sum(axis=1)
        medoids.add(int(mask[np.argmin(dists)]))
    return medoids


def build_events_data(df_feat, df_ev, cfg, medoids, zones):
    feat_types  = cfg['feat_types']
    cluster_col = cfg['cluster_col']
    events = []
    for idx, row in df_feat.iterrows():
        ev_r    = df_ev.iloc[idx]
        cluster = int(row[cluster_col])
        feat_maps = {}
        for ft in feat_types:
            country_vals = {}
            for zone, countries in zones.items():
                col = f'{ft} {zone}'
                val = round(float(row[col]), 3)
                for ctry in countries:
                    iso = CODE_MAP.get(ctry)
                    if iso:
                        country_vals[iso] = val
            feat_maps[ft] = country_vals
        events.append({
            'idx':      idx,
            'cluster':  cluster,
            'sc':       int(ev_r['sc']),
            't_start':  int(ev_r['t_start']),
            't_end':    int(ev_r['t_end']),
            'duration': round(float(ev_r['duration_days']), 1),
            'cost':     round(float(ev_r['cost_share_cross']), 3),
            'features': feat_maps,
            'is_medoid': idx in medoids,
        })
    return events


def build_scale(df_feat, feat_types, fixed_scale, zone_names):
    scale = {}
    for ft in feat_types:
        if ft in fixed_scale:
            scale[ft] = float(fixed_scale[ft])
        else:
            cols = [f'{ft} {z}' for z in zone_names]
            vals = df_feat[cols].values
            scale[ft] = float(np.ceil(np.percentile(np.abs(vals), 95) / 10) * 10)
    return scale


def make_feat_options(feat_types, feat_labels, sel_id):
    return '\n'.join(
        f'        <option value="{ft}">{feat_labels[ft]}</option>'
        for ft in feat_types
    )


def generate_html(ds_name, clim_events, op_events, clim_scale, op_scale):
    clim_json  = json.dumps(clim_events)
    op_json    = json.dumps(op_events)
    clim_scale_json = json.dumps(clim_scale)
    op_scale_json   = json.dumps(op_scale)
    colors_json     = json.dumps(CLUSTER_COLORS)
    clim_units_json = json.dumps(CLIMATE_FEATS['feat_units'])
    op_units_json   = json.dumps(OPERATIONAL_FEATS['feat_units'])

    clim_opts = make_feat_options(CLIMATE_FEATS['feat_types'], CLIMATE_FEATS['feat_labels'], 'climFeat')
    op_opts   = make_feat_options(OPERATIONAL_FEATS['feat_types'], OPERATIONAL_FEATS['feat_labels'], 'opFeat')

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Combined Clustering ({ds_name})</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  body {{ font-family: Arial, sans-serif; margin: 0; background: #f8f9fa; height: 100vh; display: flex; flex-direction: column; overflow: hidden; }}
  h1 {{ color: #2c3e50; font-size: 16px; margin: 10px 16px 8px; flex-shrink: 0; }}
  .split {{ display: flex; flex: 1; gap: 10px; padding: 0 10px 10px; min-height: 0; }}
  .half {{ display: flex; flex-direction: column; flex: 1; min-width: 0; background: white; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.1); padding: 12px; }}
  .half-title {{ font-size: 13px; font-weight: bold; color: #2c3e50; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.05em; }}
  .controls {{ display: flex; gap: 10px; align-items: flex-end; margin-bottom: 10px; flex-wrap: wrap; }}
  .controls label {{ font-size: 12px; font-weight: bold; color: #555; }}
  select {{ padding: 5px 8px; border-radius: 4px; border: 1px solid #ccc; font-size: 12px; max-width: 260px; }}
  .info {{ display: flex; align-items: center; gap: 10px; margin-bottom: 6px; min-height: 40px; }}
  .info p {{ margin: 2px 0; font-size: 12px; color: #444; }}
  .cluster-badge {{ display: inline-block; padding: 3px 10px; border-radius: 10px; color: white; font-weight: bold; font-size: 11px; white-space: nowrap; }}
  .map-wrap {{ flex: 1; min-height: 0; }}
  .link-row {{ text-align: center; font-size: 12px; color: #666; flex-shrink: 0; padding-bottom: 2px; }}
  .link-row label {{ cursor: pointer; }}
  .link-row input {{ cursor: pointer; }}
</style>
</head>
<body>
<h1>Combined Clustering — {ds_name.capitalize()}</h1>
<div class="link-row">
  <label><input type="checkbox" id="linkEvents" onchange="onLinkChange()"> Sync event selection</label>
</div>
<div class="split">
  <!-- CLIMATE half -->
  <div class="half">
    <div class="half-title">Climate</div>
    <div class="controls">
      <div>
        <label>Feature</label><br>
        <select id="climFeat" onchange="renderClim()">
{clim_opts}
        </select>
      </div>
      <div>
        <label>Event</label><br>
        <select id="climEvent" onchange="onClimEventChange()" style="width:230px"></select>
      </div>
    </div>
    <div class="info" id="climInfo"></div>
    <div class="map-wrap" id="climMap"></div>
  </div>
  <!-- OPERATIONAL half -->
  <div class="half">
    <div class="half-title">Operational</div>
    <div class="controls">
      <div>
        <label>Feature</label><br>
        <select id="opFeat" onchange="renderOp()">
{op_opts}
        </select>
      </div>
      <div>
        <label>Event</label><br>
        <select id="opEvent" onchange="onOpEventChange()" style="width:230px"></select>
      </div>
    </div>
    <div class="info" id="opInfo"></div>
    <div class="map-wrap" id="opMap"></div>
  </div>
</div>

<script>
const CLIM_DATA  = {clim_json};
const OP_DATA    = {op_json};
const CLIM_SCALE = {clim_scale_json};
const OP_SCALE   = {op_scale_json};
const COLORS     = {colors_json};
const CLIM_UNITS = {clim_units_json};
const OP_UNITS   = {op_units_json};

const heatColorscale = [
  [0,      '#3182bd'],
  [0.0099, '#3182bd'],
  [0.01,   '#fff7ec'],
  [0.35,   '#fc8d59'],
  [0.7,    '#e31a1c'],
  [1,      '#7f0000'],
];

const dischColorscale = [
  [0,      '#3182bd'],
  [0.0099, '#3182bd'],
  [0.01,   '#fff7ec'],
  [0.35,   '#fc8d59'],
  [0.7,    '#e31a1c'],
  [1,      '#7f0000'],
];

function buildDropdown(selId, data, clusterKey) {{
  const sel = document.getElementById(selId);
  data.forEach((d, i) => {{
    const opt = document.createElement('option');
    opt.value = i;
    const star = d.is_medoid ? ' ★ MEDOID' : '';
    opt.textContent = `sc${{d.sc}}  t=${{d.t_start}}-${{d.t_end}}  (${{d.duration}}d, ${{d.cost}}%)  — C${{d.cluster}}${{star}}`;
    sel.appendChild(opt);
  }});
}}

function renderMap(divId, infoId, data, scale, units, eventIdx, feat) {{
  const d      = data[eventIdx];
  const fmap   = d.features[feat];
  const col    = COLORS[d.cluster % COLORS.length];
  const absMax = scale[feat];
  const unit   = units[feat] || '';

  const medoidTag = d.is_medoid
    ? `<span style="font-size:12px;color:#f39c12;font-weight:bold"> ★ Medoid</span>`
    : '';
  document.getElementById(infoId).innerHTML = `
    <span class="cluster-badge" style="background:${{col}}">Cluster ${{d.cluster}}</span>
    ${{medoidTag}}
    <div>
      <p><b>sc${{d.sc}}</b>  t=${{d.t_start}}–${{d.t_end}}</p>
      <p>${{d.duration}}d &nbsp;|&nbsp; cost ${{d.cost}}%</p>
    </div>`;

  const useHeat  = feat === 'heat anom';
  const useDisch = feat === 'stor discharge anom';
  const reversescale = (feat === 'nettrans anom') ? false : true;

  const trace = {{
    type: 'choropleth',
    locationmode: 'ISO-3',
    locations: Object.keys(fmap),
    z: Object.values(fmap),
    colorscale: useHeat ? heatColorscale : (useDisch ? dischColorscale : 'RdBu'),
    reversescale: useHeat || useDisch ? false : reversescale,
    zmin: useHeat ? -1 : (useDisch ? -1 : -absMax), zmax: absMax,
    colorbar: {{ title: {{ text: unit, side: 'right' }}, thickness: 12 }},
    marker: {{ line: {{ color: 'white', width: 0.8 }} }},
  }};

  const layout = {{
    geo: {{
      scope: 'europe', resolution: 50,
      showland: true,  landcolor: '#eeeeee',
      showocean: true, oceancolor: '#d6eaf8',
      showcoastlines: true, coastlinecolor: '#aaaaaa',
      showframe: false,
      lonaxis: {{ range: [-12, 35] }},
      lataxis: {{ range: [34, 72] }},
    }},
    margin: {{ t: 5, b: 5, l: 5, r: 50 }},
  }};

  Plotly.react(divId, [trace], layout, {{responsive: true, displayModeBar: false}});
}}

function renderClim() {{
  const idx  = parseInt(document.getElementById('climEvent').value);
  const feat = document.getElementById('climFeat').value;
  renderMap('climMap', 'climInfo', CLIM_DATA, CLIM_SCALE, CLIM_UNITS, idx, feat);
}}

function renderOp() {{
  const idx  = parseInt(document.getElementById('opEvent').value);
  const feat = document.getElementById('opFeat').value;
  renderMap('opMap', 'opInfo', OP_DATA, OP_SCALE, OP_UNITS, idx, feat);
}}

function onClimEventChange() {{
  if (document.getElementById('linkEvents').checked) {{
    document.getElementById('opEvent').value = document.getElementById('climEvent').value;
  }}
  renderClim();
  if (document.getElementById('linkEvents').checked) renderOp();
}}

function onOpEventChange() {{
  if (document.getElementById('linkEvents').checked) {{
    document.getElementById('climEvent').value = document.getElementById('opEvent').value;
  }}
  renderOp();
  if (document.getElementById('linkEvents').checked) renderClim();
}}

function onLinkChange() {{
  if (document.getElementById('linkEvents').checked) {{
    document.getElementById('opEvent').value = document.getElementById('climEvent').value;
    renderOp();
  }}
}}

buildDropdown('climEvent', CLIM_DATA);
buildDropdown('opEvent',   OP_DATA);
renderClim();
renderOp();
</script>
</body>
</html>"""


for ds_name, ds in DATASETS.items():
    print(f'\n--- {ds_name} ---')
    for key in ('climate_csv', 'operational_csv'):
        if not os.path.exists(ds[key]):
            print(f'  SKIP: {ds[key]} not found')
            continue

    df_clim = pd.read_csv(ds['climate_csv'])
    df_op   = pd.read_csv(ds['operational_csv'])
    df_ev   = pd.read_csv(ds['events_csv'])
    df_ev   = df_ev[(df_ev['cost_share_cross'] >= THRESH_CS) &
                    (df_ev['duration_days']    >= THRESH_DUR)].reset_index(drop=True)

    zones      = ds['zones']
    zone_names = list(zones.keys())
    meta_cols  = ['sc', 't_start', 't_end']

    clim_medoids = compute_medoids(df_clim, CLIMATE_FEATS['cluster_col'], meta_cols)
    op_medoids   = compute_medoids(df_op,   OPERATIONAL_FEATS['cluster_col'], meta_cols)

    clim_scale = build_scale(df_clim, CLIMATE_FEATS['feat_types'],     CLIMATE_FEATS['fixed_scale'],     zone_names)
    op_scale   = build_scale(df_op,   OPERATIONAL_FEATS['feat_types'], OPERATIONAL_FEATS['fixed_scale'], zone_names)

    clim_events = build_events_data(df_clim, df_ev, CLIMATE_FEATS,     clim_medoids, zones)
    op_events   = build_events_data(df_op,   df_ev, OPERATIONAL_FEATS, op_medoids,   zones)

    html = generate_html(ds_name, clim_events, op_events, clim_scale, op_scale)
    with open(ds['out_html'], 'w') as f:
        f.write(html)
    print(f'  Saved: {ds["out_html"]}')

print('\nDone.')
