"""
Interactive Cluster Maps — Climate and Operational
===================================================
Generates 4 HTML files with interactive choropleth maps of Europe, one per
(feature category × dataset). Each HTML has:
  - A cluster selector (C0–C3) to show the medoid event for that cluster
  - A feature selector (e.g. wind anom, PV anom, heat anom)
  - A Plotly choropleth coloured by the zone-level anomaly value

Run AFTER compute_clustering_climate.py and compute_clustering_operational.py have produced
their *_clustered.csv output files.

Inputs:
  CSVs_and_JSONs/future/features/features_alpha25_climate_future_clustered.csv
  CSVs_and_JSONs/future/features/features_alpha25_operational_future_clustered.csv
  CSVs_and_JSONs/historical/features/features_alpha25_climate_historical_clustered.csv
  CSVs_and_JSONs/historical/features/features_alpha25_operational_historical_clustered.csv

Outputs:
  analysis/htmls_uso/europe_maps_climate_future.html
  analysis/htmls_uso/europe_maps_climate_historical.html
  analysis/htmls_uso/europe_maps_operational_future.html
  analysis/htmls_uso/europe_maps_operational_historical.html
"""

# ── Configuration ─────────────────────────────────────────────────────────────
# Output type: HTML interactive  (4 files: climate/operational × future/historical)
THRESH_CS  = 0.2   # minimum cost_share_cross (%) to include an event
THRESH_DUR = 1.0   # minimum duration (days) to include an event
OUT_DIR    = '~/Desktop/Bachelor Thesis/analysis/htmls_uso'
# ──────────────────────────────────────────────────────────────────────────────

import os, json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

OUT_DIR = os.path.expanduser(OUT_DIR)
from sklearn.metrics import pairwise_distances

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

ALPHA     = 25
THRESH_CS = 0.2
THRESH_DUR = 1.0

# ── Config per cluster type ───────────────────────────────────────────────────
CONFIGS = {
    'climate': {
        'title':       'Climate Clustering',
        'cluster_col': 'cluster_climate',
        'feat_types':  ['wind anom', 'pv anom', 'heat anom'],
        'feat_labels': {
            'wind anom': 'Wind CF anomaly (%)',
            'pv anom':   'PV CF anomaly (%)',
            'heat anom': 'Heat demand anomaly (%)',
        },
        'feat_units': {
            'wind anom': '%',
            'pv anom':   '%',
            'heat anom': '%',
        },
        'fixed_scale': {'wind anom': 100, 'pv anom': 100, 'heat anom': 100},
        'datasets': {
            'future': {
                'feat_csv':   os.path.expanduser('~/Desktop/Bachelor Thesis/CSVs_and_JSONs/future/features/features_alpha25_climate_future_clustered.csv'),
                'events_csv': os.path.expanduser(f'~/Desktop/Bachelor Thesis/CSVs_and_JSONs/future/events_global_alpha/events_global_alpha{ALPHA:02d}.csv'),
                'out_html':   os.path.expanduser('~/Desktop/Bachelor Thesis/analysis/htmls_uso/europe_maps_climate_future.html'),
                'zones':      ZONES_FUTURE,
            },
            'historical': {
                'feat_csv':   os.path.expanduser('~/Desktop/Bachelor Thesis/CSVs_and_JSONs/historical/features/features_alpha25_climate_historical_clustered.csv'),
                'events_csv': os.path.expanduser(f'~/Desktop/Bachelor Thesis/CSVs_and_JSONs/historical/events_global_alpha/events_global_historical_alpha{ALPHA:02d}.csv'),
                'out_html':   os.path.expanduser('~/Desktop/Bachelor Thesis/analysis/htmls_uso/europe_maps_climate_historical.html'),
                'zones':      ZONES_HISTORICAL,
            },
        },
    },
    'operational': {
        'title':       'Operational Clustering',
        'cluster_col': 'cluster_operational',
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
        'datasets': {
            'future': {
                'feat_csv':   os.path.expanduser('~/Desktop/Bachelor Thesis/CSVs_and_JSONs/future/features/features_alpha25_operational_future_clustered.csv'),
                'events_csv': os.path.expanduser(f'~/Desktop/Bachelor Thesis/CSVs_and_JSONs/future/events_global_alpha/events_global_alpha{ALPHA:02d}.csv'),
                'out_html':   os.path.expanduser('~/Desktop/Bachelor Thesis/analysis/htmls_uso/europe_maps_operational_future.html'),
                'zones':      ZONES_FUTURE,
            },
            'historical': {
                'feat_csv':   os.path.expanduser('~/Desktop/Bachelor Thesis/CSVs_and_JSONs/historical/features/features_alpha25_operational_historical_clustered.csv'),
                'events_csv': os.path.expanduser(f'~/Desktop/Bachelor Thesis/CSVs_and_JSONs/historical/events_global_alpha/events_global_historical_alpha{ALPHA:02d}.csv'),
                'out_html':   os.path.expanduser('~/Desktop/Bachelor Thesis/analysis/htmls_uso/europe_maps_operational_historical.html'),
                'zones':      ZONES_HISTORICAL,
            },
        },
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


def build_events_data(df_feat, df_ev, cluster_col, feat_types, medoids, zones):
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


def generate_html(title, events_data, scale, feat_types, feat_labels, feat_units):
    data_json    = json.dumps(events_data)
    colors_json  = json.dumps(CLUSTER_COLORS)
    scale_json   = json.dumps(scale)
    units_json   = json.dumps(feat_units)

    feat_options_html = '\n'.join(
        f'      <option value="{ft}">{feat_labels[ft]}</option>'
        for ft in feat_types
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  body {{ font-family: Arial, sans-serif; margin: 20px; background: #f8f9fa; }}
  h1 {{ color: #2c3e50; font-size: 18px; margin-bottom: 12px; }}
  .top-controls {{ display: flex; align-items: flex-end; gap: 10px; margin-bottom: 15px; flex-wrap: nowrap; }}
  .top-controls label {{ font-weight: bold; font-size: 13px; }}
  select {{ padding: 6px 10px; border-radius: 4px; border: 1px solid #ccc; font-size: 13px; }}
  .panels {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }}
  .panel {{ background: white; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.1); padding: 12px; }}
  .panel.hidden {{ display: none; }}
  .panel-header {{ display: flex; align-items: center; gap: 12px; margin-bottom: 8px; }}
  .panel-header p {{ margin: 2px 0; font-size: 13px; color: #444; }}
  .cluster-badge {{ display: inline-block; padding: 3px 10px; border-radius: 10px;
                    color: white; font-weight: bold; font-size: 12px; white-space: nowrap; }}
</style>
</head>
<body>
<h1>{title} — Side-by-Side Comparison</h1>

<div class="top-controls">
  <div>
    <label>Feature:</label><br>
    <select id="featSel" onchange="render()">
{feat_options_html}
    </select>
  </div>
  <div>
    <label>Event A:</label><br>
    <select id="selA" onchange="render()" style="width:175px"></select>
  </div>
  <div>
    <label>Event B:</label><br>
    <select id="selB" onchange="render()" style="width:175px"></select>
  </div>
  <div>
    <label>Event C:</label><br>
    <select id="selC" onchange="render()" style="width:175px"></select>
  </div>
  <div>
    <label>Event D:</label><br>
    <select id="selD" onchange="render()" style="width:175px"></select>
  </div>
  <div>
    <label>Event E:</label><br>
    <select id="selE" onchange="render()" style="width:175px"></select>
  </div>
  <div>
    <label>Event F:</label><br>
    <select id="selF" onchange="render()" style="width:175px"></select>
  </div>
</div>

<div class="panels" id="panelsGrid">
  <div class="panel" id="panelA">
    <div class="panel-header" id="infoA"></div>
    <div id="mapA"></div>
  </div>
  <div class="panel" id="panelB">
    <div class="panel-header" id="infoB"></div>
    <div id="mapB"></div>
  </div>
  <div class="panel" id="panelC">
    <div class="panel-header" id="infoC"></div>
    <div id="mapC"></div>
  </div>
  <div class="panel hidden" id="panelD">
    <div class="panel-header" id="infoD"></div>
    <div id="mapD"></div>
  </div>
  <div class="panel hidden" id="panelE">
    <div class="panel-header" id="infoE"></div>
    <div id="mapE"></div>
  </div>
  <div class="panel hidden" id="panelF">
    <div class="panel-header" id="infoF"></div>
    <div id="mapF"></div>
  </div>
</div>

<script>
const DATA   = {data_json};
const COLORS = {colors_json};
const SCALE  = {scale_json};
const UNITS  = {units_json};

function buildDropdowns() {{
  ['selA', 'selB', 'selC', 'selD', 'selE', 'selF'].forEach((id, si) => {{
    const sel = document.getElementById(id);
    const none = document.createElement('option');
    none.value = '';
    none.textContent = 'None';
    sel.appendChild(none);
    DATA.forEach((d, i) => {{
      const opt = document.createElement('option');
      opt.value = i;
      const star = d.is_medoid ? ' ★ MEDOID' : '';
      opt.textContent = `sc${{d.sc}}  t=${{d.t_start}}-${{d.t_end}}  (${{d.duration}}d, ${{d.cost}}%)  — C${{d.cluster}}${{star}}`;
      if (i === Math.min(si, DATA.length - 1)) opt.selected = true;
      sel.appendChild(opt);
    }});
    if (si >= 3) sel.value = '';
  }});
}}

function renderMap(divId, infoId, eventIdx, feat) {{
  const d      = DATA[eventIdx];
  const fmap   = d.features[feat];
  const col    = COLORS[d.cluster % COLORS.length];
  const absMax = SCALE[feat];
  const unit   = UNITS[feat] || '';

  const medoidTag = d.is_medoid
    ? `<span style="font-size:13px;color:#f39c12;font-weight:bold"> ★ Medoid</span>`
    : '';
  document.getElementById(infoId).innerHTML = `
    <span class="cluster-badge" style="background:${{col}}">Cluster ${{d.cluster}}</span>
    ${{medoidTag}}
    <div>
      <p><b>sc${{d.sc}}</b>  t=${{d.t_start}}–${{d.t_end}}</p>
      <p>${{d.duration}}d &nbsp;|&nbsp; cost ${{d.cost}}%</p>
    </div>`;

  const reversescale = (feat === 'nettrans anom') ? false : true;

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

  const useHeat   = feat === 'heat anom';
  const useDisch  = feat === 'stor discharge anom';
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
    margin: {{ t: 5, b: 5, l: 5, r: 5 }},
    height: 480,
  }};

  Plotly.react(divId, [trace], layout, {{responsive: true, displayModeBar: false}});
}}

function render() {{
  const feat = document.getElementById('featSel').value;
  const slots = [
    {{ sel: 'selA', panel: 'panelA', map: 'mapA', info: 'infoA' }},
    {{ sel: 'selB', panel: 'panelB', map: 'mapB', info: 'infoB' }},
    {{ sel: 'selC', panel: 'panelC', map: 'mapC', info: 'infoC' }},
    {{ sel: 'selD', panel: 'panelD', map: 'mapD', info: 'infoD' }},
    {{ sel: 'selE', panel: 'panelE', map: 'mapE', info: 'infoE' }},
    {{ sel: 'selF', panel: 'panelF', map: 'mapF', info: 'infoF' }},
  ];
  let visible = 0;
  slots.forEach(s => {{
    const val = document.getElementById(s.sel).value;
    const panel = document.getElementById(s.panel);
    if (val === '') {{
      panel.classList.add('hidden');
    }} else {{
      panel.classList.remove('hidden');
      renderMap(s.map, s.info, parseInt(val), feat);
      visible++;
    }}
  }});
  const cols = visible <= 2 ? Math.max(visible, 1) : 3;
  document.getElementById('panelsGrid').style.gridTemplateColumns = `repeat(${{cols}}, 1fr)`;
}}

buildDropdowns();
render();
</script>
</body>
</html>"""


# ── Main loop ─────────────────────────────────────────────────────────────────
for cl_type, cfg in CONFIGS.items():
    for ds_name, ds in cfg['datasets'].items():
        print(f'\n--- {cfg["title"]} / {ds_name} ---')
        if not os.path.exists(ds['feat_csv']):
            print(f'  SKIP: {ds["feat_csv"]} not found. Run clustering_{cl_type}.py first.')
            continue

        df_feat = pd.read_csv(ds['feat_csv'])
        df_ev   = pd.read_csv(ds['events_csv'])
        df_ev   = df_ev[(df_ev['cost_share_cross'] >= THRESH_CS) &
                        (df_ev['duration_days']    >= THRESH_DUR)].reset_index(drop=True)

        zones     = ds['zones']
        zone_names = list(zones.keys())
        meta_cols = ['sc', 't_start', 't_end']
        medoids   = compute_medoids(df_feat, cfg['cluster_col'], meta_cols)
        scale     = build_scale(df_feat, cfg['feat_types'], cfg['fixed_scale'], zone_names)
        events    = build_events_data(df_feat, df_ev, cfg['cluster_col'],
                                      cfg['feat_types'], medoids, zones)

        html = generate_html(
            title      = f'{cfg["title"]} ({ds_name})',
            events_data = events,
            scale      = scale,
            feat_types = cfg['feat_types'],
            feat_labels = cfg['feat_labels'],
            feat_units  = cfg['feat_units'],
        )

        with open(ds['out_html'], 'w') as f:
            f.write(html)
        print(f'  Saved: {ds["out_html"]}')

print('\nDone.')
