"""
Shadow Price and Transmission Flow Map — Historical Dataset
============================================================
Generates an interactive hourly map of Europe for each filtered historical
stress event. Identical structure to the future version:
  - Country fill: electricity shadow price (M€/GWh)
  - Arrows: transmission flow direction and magnitude
  - Line width: utilisation fraction (flow / capacity)

Run compute_sp_flow_data.py first to generate the required JSON cache.

Inputs:
  ~/Desktop/Bachelor Thesis/CSVs_and_JSONs/sp_flow_data_historical.json  (pre-computed by compute_sp_flow_data.py)

Outputs:
  analysis/htmls_uso/sp_flow_interactive_historical.html
"""

# ── Configuration ─────────────────────────────────────────────────────────────
# Output type: HTML interactive
JSON_PATH = '~/Desktop/Bachelor Thesis/CSVs_and_JSONs/sp_flow_data_historical.json'
OUT_PATH  = '~/Desktop/Bachelor Thesis/analysis/htmls_uso/sp_flow_interactive_historical.html'
# ──────────────────────────────────────────────────────────────────────────────

import os, json

JSON_PATH = os.path.expanduser(JSON_PATH)
OUT_PATH  = os.path.expanduser(OUT_PATH)

print('Reading cached data...')
with open(JSON_PATH, encoding='utf-8') as f:
    cached = json.load(f)

all_data        = cached['all_data']
sp_max_global   = cached['sp_max_global']
flow_max_global = cached['flow_max_global']
iso_map_js      = cached['iso_map']
COUNTRY_COORDS  = cached['country_coords']

print(f'  Events loaded  : {len(all_data)}')
print(f'  SP max global  : {sp_max_global:.4f} M€/GWh')
print(f'  Flow max global: {flow_max_global:.2f} GWh')
print('Generating HTML...')

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Shadow Price Electricity — Stress Events (Historical)</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{--bg:#F5F3EC;--sur:#FFF;--sur2:#EFEDE6;--t:#1A1A18;--t2:#5F5E5A;--t3:#888780;
      --bd:rgba(0,0,0,0.10);--bd2:rgba(0,0,0,0.20);--accent:#185FA5;--accent2:#C0392B}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:linear-gradient(rgba(240,245,255,0.92),rgba(240,245,255,0.92)),url('https://images.unsplash.com/photo-1466611653911-95081537e5b7?w=1600&q=80') center/cover no-repeat fixed;
     color:var(--t);padding:1.5rem 2rem;max-width:100%;margin:0 auto}}
h1{{font-size:15px;font-weight:500;margin-bottom:3px}}
.sub{{font-size:11px;color:var(--t2);margin-bottom:1.2rem}}
.dropdown{{position:relative;display:inline-block;margin-bottom:1rem}}
.dd-btn{{display:flex;align-items:center;gap:10px;border:0.5px solid var(--bd2);
         background:var(--sur);color:var(--t);border-radius:8px;padding:8px 14px;
         font-size:13px;font-weight:500;cursor:pointer;transition:all .15s;min-width:240px}}
.dd-btn:hover{{background:var(--sur2)}}
.dd-btn .dd-arrow{{margin-left:auto;font-size:10px;color:var(--t3);transition:transform .2s}}
.dropdown.open .dd-arrow{{transform:rotate(180deg)}}
.dd-menu{{position:absolute;top:calc(100% + 6px);left:0;min-width:100%;background:var(--sur);
          border:0.5px solid var(--bd2);border-radius:10px;box-shadow:0 8px 24px rgba(0,0,0,0.10);
          z-index:100;display:none;max-height:320px;overflow-y:auto}}
.dropdown.open .dd-menu{{display:block}}
.dd-item{{padding:8px 14px;font-size:12px;cursor:pointer;color:var(--t2);transition:background .12s;
          border-bottom:0.5px solid var(--bd);white-space:nowrap}}
.dd-item:last-child{{border-bottom:none}}
.dd-item:hover{{background:var(--sur2);color:var(--t)}}
.dd-item.active{{color:var(--accent);font-weight:600;background:var(--sur2)}}
.event-info{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:7px;margin-bottom:1rem}}
.ei{{background:var(--sur2);border-radius:7px;padding:.5rem .75rem}}
.ei-l{{font-size:10px;color:var(--t3);margin-bottom:2px}}
.ei-v{{font-size:13px;font-weight:500}}
.slider-wrap{{background:var(--sur);border:0.5px solid var(--bd);border-radius:11px;
              padding:.85rem 1.05rem;margin-bottom:1rem;display:flex;align-items:center;gap:16px}}
.slider-label{{font-size:11px;font-weight:500;color:var(--t2);min-width:140px}}
#timeSlider{{flex:1;accent-color:var(--accent)}}
.time-display{{font-size:12px;font-weight:500;color:var(--accent);min-width:130px;text-align:right}}
.play-btn{{border:0.5px solid var(--bd2);background:var(--sur2);color:var(--t);border-radius:6px;
           padding:4px 12px;font-size:12px;cursor:pointer;white-space:nowrap}}
.layout{{display:grid;grid-template-columns:minmax(0,1fr) 560px;gap:12px}}
@media(max-width:900px){{.layout{{grid-template-columns:1fr}}}}
#mapDiv{{width:100%;height:900px;border-radius:10px;border:0.5px solid var(--bd)}}
.right-col{{display:flex;flex-direction:row;gap:10px;height:900px}}
.sp-panel{{background:var(--sur);border:0.5px solid var(--bd);border-radius:11px;
           padding:.85rem 1.05rem;overflow-y:auto;flex:0 0 auto;max-height:340px}}
.tr-panel{{background:var(--sur);border:0.5px solid var(--bd);border-radius:11px;
           padding:.85rem 1.05rem;overflow-y:auto;flex:1 1 0;min-height:0}}
.panel-title{{font-size:11px;font-weight:500;color:var(--t2);margin-bottom:.6rem}}
.sp-row{{display:grid;grid-template-columns:28px 1fr 70px;align-items:center;gap:6px;
          padding:3px 0;border-bottom:0.5px solid var(--bd)}}
.sp-flag{{font-size:10px;color:var(--t3)}}
.sp-bar-wrap{{height:8px;background:var(--sur2);border-radius:4px;overflow:hidden}}
.sp-bar{{height:100%;border-radius:4px;transition:width .2s}}
.sp-val{{font-size:11px;font-weight:500;text-align:right}}
.tr-row{{display:grid;grid-template-columns:60px 44px 44px 44px 56px;align-items:center;
          gap:3px;padding:3px 0;border-bottom:0.5px solid var(--bd);font-size:10px}}
.tr-hdr{{font-size:9px;font-weight:600;color:var(--t3);text-align:right}}
.tr-hdr:first-child{{text-align:left}}
.tr-cell{{text-align:right;font-weight:500}}
.tr-edge{{font-size:10px;font-family:monospace;color:var(--t)}}
.util-bar-wrap{{position:relative;height:6px;background:var(--sur2);border-radius:3px;overflow:hidden;margin-top:2px}}
.util-bar{{height:100%;border-radius:3px;transition:width .2s}}
</style>
</head>
<body>
<h1>Shadow Price Electricity + Transmission Flows — Stress Events (Historical, α=25%, cs≥0.2, dur≥1d)</h1>
<p class="sub">Choropleth: electricity shadow price (M€/GWh) · Arrows: net power flow between countries · Slider: hourly resolution</p>

<div class="dropdown" id="evDropdown">
  <button class="dd-btn" onclick="toggleDropdown()">
    <span id="ddLabel">Select event…</span>
    <span class="dd-arrow">▼</span>
  </button>
  <div class="dd-menu" id="ddMenu"></div>
</div>
<div class="event-info" id="eventInfo"></div>

<div class="slider-wrap">
  <span class="slider-label">Hour within event:</span>
  <input type="range" id="timeSlider" min="0" max="0" value="0" step="1">
  <span class="time-display" id="timeDisplay">—</span>
  <button class="play-btn" id="playBtn">&#9654; Play</button>
</div>

<div class="layout">
  <div id="mapDiv"></div>
  <div class="right-col">
    <div class="sp-panel">
      <div class="panel-title">Shadow price by country (M€/GWh)</div>
      <div id="spList"></div>
    </div>
    <div class="tr-panel">
      <div class="panel-title">Transmission lines — active this hour</div>
      <div id="trList"></div>
    </div>
  </div>
</div>

<script>
const allData = {json.dumps(all_data)};
const isoMap  = {json.dumps(iso_map_js)};
const coords  = {json.dumps(COUNTRY_COORDS)};
const spMax   = {round(sp_max_global * 1.05, 4)};
const flowMax = {round(flow_max_global, 2)};

let currentEv = Object.keys(allData).sort((a,b)=>+a-+b)[0];
let currentIdx = 0;
let playing = false;
let playTimer = null;

// ── Build event dropdown ──────────────────────────────────────────────────────
const ddMenu = document.getElementById('ddMenu');
Object.keys(allData).sort((a,b)=>+a-+b).forEach((evId) => {{
    const item = document.createElement('div');
    item.className = 'dd-item';
    item.textContent = allData[evId].event.label;
    item.onclick = () => {{ selectEvent(evId); closeDropdown(); }};
    ddMenu.appendChild(item);
}});

function toggleDropdown() {{
    document.getElementById('evDropdown').classList.toggle('open');
}}
function closeDropdown() {{
    document.getElementById('evDropdown').classList.remove('open');
}}
document.addEventListener('click', e => {{
    if (!document.getElementById('evDropdown').contains(e.target)) closeDropdown();
}});

function selectEvent(evId) {{
    currentEv = evId;
    currentIdx = 0;
    document.getElementById('ddLabel').textContent = allData[evId].event.label;
    document.querySelectorAll('.dd-item').forEach((item, i) => {{
        item.classList.toggle('active', Object.keys(allData).sort((a,b)=>+a-+b)[i] === evId);
    }});
    const d = allData[evId];
    const ev = d.event;

    document.getElementById('eventInfo').innerHTML = `
        <div class="ei"><div class="ei-l">Period</div><div class="ei-v">${{ev.label}}</div></div>
        <div class="ei"><div class="ei-l">Cost share</div><div class="ei-v">${{ev.share}}%</div></div>
        <div class="ei"><div class="ei-l">Duration</div><div class="ei-v">${{ev.n_hours}} hours</div></div>
        <div class="ei"><div class="ei-l">Days</div><div class="ei-v">Day ${{ev.d1}}–${{ev.d2}}</div></div>
    `;

    const slider = document.getElementById('timeSlider');
    slider.max = ev.n_hours - 1;
    slider.value = 0;

    updateMap(0);
}}

// ── Update map for a given hour index ─────────────────────────────────────────
function updateMap(idx) {{
    currentIdx = idx;
    const d = allData[currentEv];
    const hd = d.hours[idx];

    document.getElementById('timeDisplay').textContent = hd.l;
    document.getElementById('timeSlider').value = idx;

    // ── Choropleth (log scale) ────────────────────────────────────────────────
    const locs = [], zvals = [], htexts = [];
    const LOG_MIN = Math.log10(0.01);
    const LOG_MAX = Math.log10(Math.max(spMax, 0.01));
    for (const [code, iso3] of Object.entries(isoMap)) {{
        const sp = hd.sp[code] ?? 0;
        const logVal = sp > 0.001 ? Math.log10(sp) : LOG_MIN;
        locs.push(iso3);
        zvals.push(logVal);
        htexts.push(`${{coords[code]?.name ?? code}}<br>${{sp.toFixed(4)}} M€/GWh`);
    }}

    const tickVals = [-2, -1, 0, 0.5, 1, 1.5, 2].filter(v => v >= LOG_MIN && v <= LOG_MAX);
    const tickText = tickVals.map(v => (Math.pow(10, v)).toFixed(v < 0 ? 3 : v < 1 ? 2 : 1));

    const choropleth = {{
        type: 'choropleth',
        locationmode: 'ISO-3',
        locations: locs,
        z: zvals,
        zmin: LOG_MIN,
        zmax: LOG_MAX,
        colorscale: [
            [0,   'rgb(255,255,255)'],
            [0.3, 'rgb(254,224,182)'],
            [0.6, 'rgb(253,141,60)'],
            [1.0, 'rgb(179,0,0)']
        ],
        showscale: true,
        colorbar: {{
            title: {{text:'M€/GWh',font:{{size:10}}}},
            thickness:12, len:0.6, tickfont:{{size:9}},
            tickvals: tickVals,
            ticktext: tickText
        }},
        hovertemplate: '%{{customdata}}<extra></extra>',
        customdata: htexts,
        marker: {{line: {{color: 'rgba(80,80,80,0.4)', width: 0.5}}}},
        name: 'Shadow price'
    }};

    const lineTraces = [];
    for (const fl of hd.fl) {{
        const src = coords[fl.f];
        const dst = coords[fl.t];
        if (!src || !dst) continue;
        const w = Math.max(1.5, Math.min(9, Math.abs(fl.v) / flowMax * 9));
        const midLat = (src.lat + dst.lat) / 2;
        const midLon = (src.lon + dst.lon) / 2;
        const tip = `${{fl.f}} → ${{fl.t}}<br>${{Math.abs(fl.v).toFixed(2)}} GWh<extra></extra>`;
        lineTraces.push({{
            type: 'scattergeo', mode: 'lines',
            lat: [src.lat, midLat], lon: [src.lon, midLon],
            line: {{width: w, color: '#C0392B'}},
            hovertemplate: tip, showlegend: false
        }});
        lineTraces.push({{
            type: 'scattergeo', mode: 'lines',
            lat: [midLat, dst.lat], lon: [midLon, dst.lon],
            line: {{width: w, color: '#27AE60'}},
            hovertemplate: tip, showlegend: false
        }});
    }}

    const dotLats = [], dotLons = [], dotTips = [];
    for (const code of Object.keys(coords)) {{
        dotLats.push(coords[code].lat);
        dotLons.push(coords[code].lon);
        const sp = hd.sp[code] ?? 0;
        dotTips.push(`${{coords[code].name}}<br>${{sp.toFixed(4)}} M€/GWh<extra></extra>`);
    }}
    lineTraces.push({{
        type: 'scattergeo', mode: 'markers',
        lat: dotLats, lon: dotLons,
        marker: {{size: 5, color: 'white', line: {{color: '#aaa', width: 0.8}}}},
        hovertemplate: dotTips, showlegend: false
    }});

    const layout = {{
        geo: {{
            projection: {{type: 'mercator'}},
            center: {{lat: 52, lon: 10}},
            lataxis: {{range: [36, 71]}},
            lonaxis: {{range: [-15, 35]}},
            showland: true,
            landcolor: 'rgb(245,243,236)',
            coastlinecolor: 'rgb(160,160,160)',
            countrycolor: 'rgb(180,180,180)',
            showocean: true,
            oceancolor: 'rgb(230,240,255)',
            showcountries: true,
            countrywidth: 0.5,
            bgcolor: 'rgba(0,0,0,0)'
        }},
        margin: {{l:0,r:0,t:0,b:0}},
        paper_bgcolor: 'rgba(0,0,0,0)',
        hovermode: 'closest'
    }};

    Plotly.react('mapDiv', [choropleth, ...lineTraces], layout, {{responsive:true, displayModeBar:false}});

    // ── SP sidebar list ───────────────────────────────────────────────────────
    const sorted = Object.entries(hd.sp).sort((a,b) => b[1]-a[1]);
    const listEl = document.getElementById('spList');
    listEl.innerHTML = sorted.map(([code, val]) => {{
        const logVal = val > 0.001 ? Math.log10(val) : LOG_MIN;
        const pct = Math.min(100, Math.max(0, (logVal - LOG_MIN) / (LOG_MAX - LOG_MIN) * 100));
        const r = Math.round(pct / 100 * 179);
        const g = Math.round((1 - pct/100) * 200);
        const color = `rgb(${{r}},${{g}},0)`;
        return `<div class="sp-row">
            <span class="sp-flag">${{code}}</span>
            <div class="sp-bar-wrap"><div class="sp-bar" style="width:${{pct.toFixed(1)}}%;background:${{color}}"></div></div>
            <span class="sp-val" style="color:${{color}}">${{val.toFixed(4)}}</span>
        </div>`;
    }}).join('');

    // ── Transport lines panel ─────────────────────────────────────────────────
    const trEl = document.getElementById('trList');
    const flSorted = [...hd.fl].sort((a,b) => b.dual - a.dual);
    if (flSorted.length === 0) {{
        trEl.innerHTML = '<div style="font-size:10px;color:var(--t3)">No active lines this hour</div>';
    }} else {{
        const hdr = `<div class="tr-row">
            <span class="tr-hdr">Line</span>
            <span class="tr-hdr">Flow</span>
            <span class="tr-hdr">Cap.</span>
            <span class="tr-hdr">Util%</span>
            <span class="tr-hdr">Dual</span>
        </div>`;
        const rows = flSorted.map(fl => {{
            const util = fl.util ?? 0;
            const utilColor = util >= 95 ? '#c0392b' : util >= 70 ? '#e67e22' : '#27ae60';
            const dualColor = fl.dual > 0 ? '#c0392b' : 'var(--t3)';
            return `<div class="tr-row">
                <span class="tr-edge">${{fl.f}}-${{fl.t}}</span>
                <span class="tr-cell">${{Math.abs(fl.v).toFixed(1)}}</span>
                <span class="tr-cell">${{fl.cap.toFixed(1)}}</span>
                <span class="tr-cell" style="color:${{utilColor}}">${{util.toFixed(0)}}%</span>
                <span class="tr-cell" style="color:${{dualColor}}">${{fl.dual.toFixed(3)}}</span>
            </div>`;
        }}).join('');
        trEl.innerHTML = hdr + rows;
    }}
}}

// ── Slider ─────────────────────────────────────────────────────────────────
document.getElementById('timeSlider').addEventListener('input', e => {{
    updateMap(parseInt(e.target.value));
}});

// ── Play / Pause ──────────────────────────────────────────────────────────
document.getElementById('playBtn').addEventListener('click', () => {{
    if (playing) {{
        clearInterval(playTimer);
        playing = false;
        document.getElementById('playBtn').innerHTML = '&#9654; Play';
    }} else {{
        playing = true;
        document.getElementById('playBtn').innerHTML = '&#9646;&#9646; Pause';
        playTimer = setInterval(() => {{
            const max = allData[currentEv].event.n_hours - 1;
            const next = currentIdx >= max ? 0 : currentIdx + 1;
            updateMap(next);
        }}, 85);
    }}
}});

// ── Init ──────────────────────────────────────────────────────────────────
const _h = location.hash.match(/^#ev=(\d+)$/);
selectEvent(_h && allData[_h[1]] ? _h[1] : Object.keys(allData).sort((a,b)=>+a-+b)[0]);

window.addEventListener('hashchange', () => {{
  const m = location.hash.match(/^#ev=(\d+)$/);
  if (m && allData[m[1]]) selectEvent(m[1]);
}});
</script>
</body>
</html>"""

with open(OUT_PATH, 'w', encoding='utf-8') as f:
    f.write(html)

print(f'HTML saved: {OUT_PATH}')
