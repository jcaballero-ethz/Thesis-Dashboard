"""
Clustering Comparison Dashboard
================================
Generates an HTML with 4 sub-tabs comparing the clustering results across
datasets and feature types:
  - Climate Future / Climate Historical / Operational Future / Operational Historical

Each tab shows a scatter plot (PCA projection) with cluster colours and a bar
chart of the mean feature anomalies per cluster.

Inputs:
  CSVs_and_JSONs/future/features/features_alpha25_climate_future_clustered.csv
  CSVs_and_JSONs/future/features/features_alpha25_operational_future_clustered.csv
  CSVs_and_JSONs/historical/features/features_alpha25_climate_historical_clustered.csv
  CSVs_and_JSONs/historical/features/features_alpha25_operational_historical_clustered.csv

Outputs:
  analysis/htmls_uso/cluster_comparison.html
"""

# ── Configuration ─────────────────────────────────────────────────────────────
# Output type: HTML interactive
THRESH_CS  = 0.2   # minimum cost_share_cross (%) to include an event
THRESH_DUR = 1.0   # minimum duration (days) to include an event
OUT_HTML   = '~/Desktop/Bachelor Thesis/analysis/htmls_uso/cluster_comparison.html'
# ──────────────────────────────────────────────────────────────────────────────

import json, os
import pandas as pd

OUT_HTML = os.path.expanduser(OUT_HTML)

ZONES_FUTURE = ['Nordics_N','Baltics','Nordics_S','British_Isles','NW_Continent',
                'Med_East_N','Med_East_S','Greece','Italy','Iberia',
                'Central_W','East_N','Central_E']

ZONES_HISTORICAL = ['Norway','Baltics_FI','Nordics_S','British_Isles','NW_Continent',
                    'Med_East_N','Med_East_S','Greece','Italy','Iberia',
                    'Central_W','East_N','Central_E']

THRESH_CS  = 0.2
THRESH_DUR = 1.0

DATASETS = [
    {
        'key':          'climate_future',
        'label':        'Climate · Future',
        'feat_csv':     os.path.expanduser('~/Desktop/Bachelor Thesis/CSVs_and_JSONs/future/features/features_alpha25_climate_future_clustered.csv'),
        'ev_csv':       os.path.expanduser('~/Desktop/Bachelor Thesis/CSVs_and_JSONs/future/events_global_alpha/events_global_alpha25.csv'),
        'cluster_col':  'cluster_climate',
        'feat_prefixes':['wind anom', 'pv anom', 'heat anom'],
        'feat_labels':  ['Wind Anomaly (%)', 'PV Anomaly (%)', 'Heat Anomaly (%)'],
        'zones':        ZONES_FUTURE,
    },
    {
        'key':          'climate_historical',
        'label':        'Climate · Historical',
        'feat_csv':     os.path.expanduser('~/Desktop/Bachelor Thesis/CSVs_and_JSONs/historical/features/features_alpha25_climate_historical_clustered.csv'),
        'ev_csv':       os.path.expanduser('~/Desktop/Bachelor Thesis/CSVs_and_JSONs/historical/events_global_alpha/events_global_historical_alpha25.csv'),
        'cluster_col':  'cluster_climate',
        'feat_prefixes':['wind anom', 'pv anom', 'heat anom'],
        'feat_labels':  ['Wind Anomaly (%)', 'PV Anomaly (%)', 'Heat Anomaly (%)'],
        'zones':        ZONES_HISTORICAL,
    },
    {
        'key':          'operational_future',
        'label':        'Operational · Future',
        'feat_csv':     os.path.expanduser('~/Desktop/Bachelor Thesis/CSVs_and_JSONs/future/features/features_alpha25_operational_future_clustered.csv'),
        'ev_csv':       os.path.expanduser('~/Desktop/Bachelor Thesis/CSVs_and_JSONs/future/events_global_alpha/events_global_alpha25.csv'),
        'cluster_col':  'cluster_operational',
        'feat_prefixes':['nettrans anom', 'stor level anom', 'stor discharge anom'],
        'feat_labels':  ['Net Transfer Anomaly (GWh/h)', 'Storage Level Anomaly (GWh)', 'Storage Discharge Anomaly (GWh/h)'],
        'zones':        ZONES_FUTURE,
    },
    {
        'key':          'operational_historical',
        'label':        'Operational · Historical',
        'feat_csv':     os.path.expanduser('~/Desktop/Bachelor Thesis/CSVs_and_JSONs/historical/features/features_alpha25_operational_historical_clustered.csv'),
        'ev_csv':       os.path.expanduser('~/Desktop/Bachelor Thesis/CSVs_and_JSONs/historical/events_global_alpha/events_global_historical_alpha25.csv'),
        'cluster_col':  'cluster_operational',
        'feat_prefixes':['nettrans anom', 'stor level anom', 'stor discharge anom'],
        'feat_labels':  ['Net Transfer Anomaly (GWh/h)', 'Storage Level Anomaly (GWh)', 'Storage Discharge Anomaly (GWh/h)'],
        'zones':        ZONES_HISTORICAL,
    },
]

all_data = {}
for ds in DATASETS:
    feat_df = pd.read_csv(ds['feat_csv'])
    ev_df   = pd.read_csv(ds['ev_csv'])
    ev_df   = ev_df[
        (ev_df['cost_share_cross'] >= THRESH_CS) &
        (ev_df['duration_days']    >= THRESH_DUR)
    ].reset_index(drop=True)

    merged = feat_df.merge(
        ev_df[['sc','t_start','t_end','duration_days','cost_share_cross']],
        on=['sc','t_start','t_end'], how='inner'
    ).reset_index(drop=True)

    events = []
    for idx, row in merged.iterrows():
        feats = []
        for prefix in ds['feat_prefixes']:
            vals = [round(float(row[f'{prefix} {z}']), 3) for z in ds['zones']]
            feats.append(vals)
        events.append({
            'idx':      int(idx),
            'sc':       int(row['sc']),
            'duration': round(float(row['duration_days']), 2),
            'cost':     round(float(row['cost_share_cross']), 4),
            'cluster':  int(row[ds['cluster_col']]),
            'feats':    feats,
        })

    all_data[ds['key']] = {
        'label':       ds['label'],
        'feat_labels': ds['feat_labels'],
        'zones':       [z.replace('_', ' ') for z in ds['zones']],
        'n_clusters':  max(e['cluster'] for e in events) + 1,
        'events':      events,
    }

data_json = json.dumps(all_data)

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Clustering Comparison</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background: #f8fafc; color: #1e293b;
  height: 100vh; display: flex; flex-direction: column; overflow: hidden;
}

/* ── Tabs ── */
.tab-bar {
  display: flex; background: #fff;
  border-bottom: 1px solid #e2e8f0; flex-shrink: 0; overflow-x: auto;
}
.tab-btn {
  padding: 9px 16px; border: none; border-bottom: 2px solid transparent;
  background: none; cursor: pointer; font-size: 12.5px; font-weight: 500;
  color: #64748b; transition: color .15s, border-color .15s; white-space: nowrap;
}
.tab-btn:hover { color: #1e293b; }
.tab-btn.active { color: #3b82f6; border-bottom-color: #3b82f6; font-weight: 600; }

/* ── Layout ── */
.main { flex: 1; display: flex; min-height: 0; }

/* Left: scatter + pills */
.panel-left {
  width: 36%; min-width: 260px;
  border-right: 1px solid #e2e8f0;
  display: flex; flex-direction: column; background: #fff;
}
.section-title {
  padding: 8px 14px 2px; font-size: 10.5px; font-weight: 600;
  color: #94a3b8; letter-spacing: .06em; text-transform: uppercase; flex-shrink: 0;
}
.scatter-wrap {
  flex: 1; min-height: 0; position: relative; padding: 4px 10px 6px;
}
.scatter-wrap canvas { position: absolute; inset: 4px 10px 6px; }

.sel-area {
  padding: 7px 12px; border-top: 1px solid #e2e8f0;
  min-height: 42px; max-height: 88px; overflow-y: auto;
  display: flex; flex-wrap: wrap; gap: 5px; align-items: center;
}
.sel-hint { font-size: 11px; color: #cbd5e1; font-style: italic; }
.pill {
  display: inline-flex; align-items: center; gap: 4px;
  padding: 3px 9px; border-radius: 10px;
  font-size: 11px; font-weight: 600; color: #fff;
  cursor: pointer; user-select: none; flex-shrink: 0;
  transition: opacity .15s;
}
.pill:hover { opacity: .8; }
.pill .x { font-size: 10px; opacity: .75; }

/* Right: 3 bar charts stacked */
.panel-right {
  flex: 1; display: flex; flex-direction: column; min-height: 0;
}
.bar-section {
  flex: 1; min-height: 0; display: flex; flex-direction: column;
  background: #fff; border-bottom: 1px solid #e2e8f0;
}
.bar-section:last-child { border-bottom: none; }
.bar-header {
  padding: 6px 14px 0; font-size: 10.5px; font-weight: 600;
  color: #94a3b8; letter-spacing: .06em; text-transform: uppercase; flex-shrink: 0;
}
.bar-wrap {
  flex: 1; min-height: 0; position: relative; padding: 2px 10px 6px;
}
.bar-wrap canvas { position: absolute; inset: 2px 10px 6px; }
.empty-msg {
  position: absolute; inset: 0; display: flex;
  align-items: center; justify-content: center;
  font-size: 12px; color: #cbd5e1; font-style: italic; pointer-events: none;
}
</style>
</head>
<body>

<div class="tab-bar" id="tabBar"></div>

<div class="main">
  <div class="panel-left">
    <div class="section-title">Events — click to compare</div>
    <div class="scatter-wrap"><canvas id="scCanvas"></canvas></div>
    <div class="sel-area" id="selArea">
      <span class="sel-hint" id="selHint">click a point to add to comparison</span>
    </div>
  </div>

  <div class="panel-right">
    <div class="bar-section">
      <div class="bar-header" id="bh0"></div>
      <div class="bar-wrap">
        <canvas id="bc0"></canvas>
        <div class="empty-msg" id="em0">select events in the scatter</div>
      </div>
    </div>
    <div class="bar-section">
      <div class="bar-header" id="bh1"></div>
      <div class="bar-wrap">
        <canvas id="bc1"></canvas>
        <div class="empty-msg" id="em1">select events in the scatter</div>
      </div>
    </div>
    <div class="bar-section">
      <div class="bar-header" id="bh2"></div>
      <div class="bar-wrap">
        <canvas id="bc2"></canvas>
        <div class="empty-msg" id="em2">select events in the scatter</div>
      </div>
    </div>
  </div>
</div>

<script>
const DATA  = __DATA_JSON__;

const CLUSTER_COLORS = ['#3b82f6','#f59e0b','#34d399','#f43f5e','#8b5cf6','#06b6d4'];
const SEL_COLORS     = ['#ef4444','#f97316','#a855f7','#0ea5e9','#84cc16','#ec4899'];
const DS_KEYS = ['climate_future','climate_historical','operational_future','operational_historical'];
const MAX_SEL = 6;

let curKey  = DS_KEYS[0];
let selEvts = [];
let scChart = null;
let bCharts = [null, null, null];

// ── Tabs ──────────────────────────────────────────────────────────────────────
const tabBar = document.getElementById('tabBar');
DS_KEYS.forEach((key, i) => {
  const btn = document.createElement('button');
  btn.className = 'tab-btn' + (i === 0 ? ' active' : '');
  btn.textContent = DATA[key].label;
  btn.onclick = () => switchTab(key, btn);
  tabBar.appendChild(btn);
});

function switchTab(key, btn) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  curKey  = key;
  selEvts = [];
  buildScatter();
  updateBars();
  renderPills();
}

// ── Scatter ───────────────────────────────────────────────────────────────────
function buildScatter() {
  const ds = DATA[curKey];
  if (scChart) { scChart.destroy(); scChart = null; }

  const byCluster = {};
  ds.events.forEach(ev => {
    (byCluster[ev.cluster] = byCluster[ev.cluster] || []).push(ev);
  });

  const datasets = Object.keys(byCluster)
    .map(Number).sort((a, b) => a - b)
    .map(ci => {
      const evs = byCluster[ci];
      const col = CLUSTER_COLORS[ci % CLUSTER_COLORS.length];
      return {
        label: 'Cluster ' + ci,
        backgroundColor: col,
        borderColor:     col,
        pointBorderColor: ctx => selEvts.includes(ctx.raw.evIdx) ? '#111' : col,
        pointBorderWidth: ctx => selEvts.includes(ctx.raw.evIdx) ? 2.5 : 0,
        pointRadius:      ctx => selEvts.includes(ctx.raw.evIdx) ? 9 : 6,
        pointHoverRadius: 9,
        data: evs.map(ev => ({ x: ev.duration, y: ev.cost, evIdx: ev.idx })),
      };
    });

  scChart = new Chart(document.getElementById('scCanvas'), {
    type: 'scatter',
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      onClick(evt, els) {
        if (!els.length) return;
        const pt = scChart.data.datasets[els[0].datasetIndex].data[els[0].index];
        toggleEvt(pt.evIdx);
      },
      plugins: {
        legend: { position: 'top', labels: { boxWidth: 10, font: { size: 11 } } },
        tooltip: {
          callbacks: {
            label(ctx) {
              const ev = DATA[curKey].events[ctx.raw.evIdx];
              return [
                'Sc' + ev.sc + '  ·  Cluster ' + ev.cluster,
                'Duration: ' + ev.duration.toFixed(1) + ' d',
                'Cost share: ' + ev.cost.toFixed(3) + ' %',
              ];
            }
          }
        }
      },
      scales: {
        x: {
          title: { display: true, text: 'Duration (days)', font: { size: 11 } },
          grid:  { color: '#f1f5f9' }, ticks: { font: { size: 10 } },
        },
        y: {
          title: { display: true, text: 'Cost share (%)', font: { size: 11 } },
          grid:  { color: '#f1f5f9' }, ticks: { font: { size: 10 } },
        },
      },
    },
  });
}

function toggleEvt(idx) {
  const i = selEvts.indexOf(idx);
  if (i >= 0) { selEvts.splice(i, 1); }
  else {
    if (selEvts.length >= MAX_SEL) selEvts.shift();
    selEvts.push(idx);
  }
  scChart.update('none');
  updateBars();
  renderPills();
}

// ── Bar charts ────────────────────────────────────────────────────────────────
function updateBars() {
  const ds = DATA[curKey];

  for (let fi = 0; fi < 3; fi++) {
    document.getElementById('bh' + fi).textContent = ds.feat_labels[fi];
    const emEl = document.getElementById('em' + fi);

    if (!selEvts.length) {
      emEl.style.display = 'flex';
      if (bCharts[fi]) { bCharts[fi].destroy(); bCharts[fi] = null; }
      continue;
    }
    emEl.style.display = 'none';

    const datasets = selEvts.map((evIdx, si) => {
      const ev  = ds.events[evIdx];
      const col = SEL_COLORS[si % SEL_COLORS.length];
      return {
        label: 'Sc' + ev.sc + ' · ' + ev.duration.toFixed(1) + 'd',
        data:  ev.feats[fi],
        backgroundColor: col + 'bb',
        borderColor:     col,
        borderWidth: 1,
        borderRadius: 2,
      };
    });

    if (bCharts[fi]) { bCharts[fi].destroy(); bCharts[fi] = null; }
    bCharts[fi] = new Chart(document.getElementById('bc' + fi), {
      type: 'bar',
      data: { labels: DATA[curKey].zones, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: {
          legend: {
            display: selEvts.length > 1,
            position: 'top',
            labels: { boxWidth: 10, font: { size: 10 } },
          },
        },
        scales: {
          x: { ticks: { font: { size: 9 }, maxRotation: 40 }, grid: { color: '#f1f5f9' } },
          y: { ticks: { font: { size: 9 } }, grid: { color: '#f1f5f9' } },
        },
      },
    });
  }
}

// ── Pills ─────────────────────────────────────────────────────────────────────
function renderPills() {
  const area = document.getElementById('selArea');
  const hint = document.getElementById('selHint');
  const ds   = DATA[curKey];
  area.querySelectorAll('.pill').forEach(p => p.remove());
  hint.style.display = selEvts.length ? 'none' : '';

  selEvts.forEach((evIdx, si) => {
    const ev  = ds.events[evIdx];
    const col = SEL_COLORS[si % SEL_COLORS.length];
    const p   = document.createElement('span');
    p.className = 'pill';
    p.style.background = col;
    p.title = 'C' + ev.cluster + ' · Sc' + ev.sc + ' · ' + ev.duration.toFixed(1) + 'd  —  click to remove';
    p.innerHTML = 'C' + ev.cluster + ' Sc' + ev.sc + ' <span class="x">×</span>';
    p.onclick = () => toggleEvt(evIdx);
    area.appendChild(p);
  });
}

// ── Init ──────────────────────────────────────────────────────────────────────
buildScatter();
updateBars();
</script>
</body>
</html>
"""

html = HTML.replace('__DATA_JSON__', data_json)

out = os.path.expanduser(
    '~/Desktop/Bachelor Thesis/analysis/htmls_uso/cluster_comparison.html'
)
with open(out, 'w', encoding='utf-8') as f:
    f.write(html)

print(f'Saved: {out}')
for k, v in all_data.items():
    print(f'  {k}: {len(v["events"])} events, {v["n_clusters"]} clusters')
