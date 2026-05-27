"""
Shadow Price and Transmission Flow Extraction
=============================================
Extracts hourly electricity shadow prices (nodal energy balance dual) and
transmission flows for all filtered stress events, for both future and historical
datasets. The output JSONs are read by generate_sp_flow_map.py to build the
interactive map HTML.

Inputs:
  - ZEN-garden output folders (future + historical)
  - events_global_alpha25.csv / events_global_historical_alpha25.csv

Outputs:
  CSVs/sp_flow_data_future.json
  CSVs/sp_flow_data_historical.json
"""

# ── Configuration ─────────────────────────────────────────────────────────────
# Output type: JSON cache (read by generate_sp_flow_map*.py to build the HTML)
ALPHA     = 25
THRESH_CS = 0.2   # minimum cost_share_cross (%) to keep an event
THRESH_DUR = 1.0  # minimum duration (days) to keep an event

DATASETS = {
    'future': {
        'path':       '~/Desktop/ZEN-garden model/outputs/20260202_GF_future_with_cooling',
        'events_csv': f'~/Desktop/Bachelor Thesis/CSVs/future/events_global_alpha/events_global_alpha{ALPHA:02d}.csv',
        'out_json':   '~/Desktop/Bachelor Thesis/CSVs/sp_flow_data_future.json',
    },
    'historical': {
        'path':       '~/Desktop/ZEN-garden model/outputs/20260202_GF_historical_with_cooling',
        'events_csv': f'~/Desktop/Bachelor Thesis/CSVs/historical/events_global_alpha/events_global_historical_alpha{ALPHA:02d}.csv',
        'out_json':   '~/Desktop/Bachelor Thesis/CSVs/sp_flow_data_historical.json',
    },
}
# ──────────────────────────────────────────────────────────────────────────────

import logging
logging.getLogger("pint").setLevel(logging.ERROR)
import sys, os
sys.stderr = open(os.devnull, 'w')
from zen_garden import Results
sys.stderr.close()
sys.stderr = sys.__stderr__

import numpy as np
import json
import pandas as pd
from collections import defaultdict

# ── ZEN-garden API note ───────────────────────────────────────────────────────
# Results(path)                        — loads all scenario outputs
# r.get_dual('constraint_nodal_energy_balance')     — electricity shadow prices
#                                        index: (carrier, node), columns: hours 0..8759
#                                        values: M€/GWh
# r.get_dual('constraint_capacity_factor_transport')— line congestion dual
#                                        index: ('power_line', edge), columns: hours 0..8759
#                                        values: M€/GWh  (>0 when line is at capacity)
# r.get_full_ts('flow_transport')       — power line flows  [GWh/h]
# r.get_full_ts('capacity')             — installed capacity per (technology, node)  [GW]
# ─────────────────────────────────────────────────────────────────────────────

# Expand paths
for ds in DATASETS.values():
    ds['path']       = os.path.expanduser(ds['path'])
    ds['events_csv'] = os.path.expanduser(ds['events_csv'])
    ds['out_json']   = os.path.expanduser(ds['out_json'])

ISO_MAP = {
    'AT':'AUT','BE':'BEL','BG':'BGR','HR':'HRV','CZ':'CZE','DK':'DNK',
    'EE':'EST','FI':'FIN','FR':'FRA','DE':'DEU','EL':'GRC','HU':'HUN',
    'IE':'IRL','IT':'ITA','LV':'LVA','LT':'LTU','LU':'LUX','NL':'NLD',
    'NO':'NOR','PL':'POL','PT':'PRT','RO':'ROU','SK':'SVK','SI':'SVN',
    'ES':'ESP','SE':'SWE','CH':'CHE','UK':'GBR',
}

COUNTRY_COORDS = {
    'AT': {'lat': 47.5,  'lon': 14.5,  'name': 'Austria'},
    'BE': {'lat': 50.5,  'lon': 4.5,   'name': 'Belgium'},
    'BG': {'lat': 42.7,  'lon': 25.5,  'name': 'Bulgaria'},
    'HR': {'lat': 45.1,  'lon': 16.5,  'name': 'Croatia'},
    'CZ': {'lat': 49.8,  'lon': 15.5,  'name': 'Czechia'},
    'DK': {'lat': 56.5,  'lon': 10.0,  'name': 'Denmark'},
    'EE': {'lat': 58.5,  'lon': 25.5,  'name': 'Estonia'},
    'FI': {'lat': 62.0,  'lon': 25.0,  'name': 'Finland'},
    'FR': {'lat': 46.5,  'lon': 2.0,   'name': 'France'},
    'DE': {'lat': 51.5,  'lon': 10.0,  'name': 'Germany'},
    'EL': {'lat': 39.0,  'lon': 22.0,  'name': 'Greece'},
    'HU': {'lat': 47.0,  'lon': 19.0,  'name': 'Hungary'},
    'IE': {'lat': 53.0,  'lon': -8.0,  'name': 'Ireland'},
    'IT': {'lat': 42.5,  'lon': 12.5,  'name': 'Italy'},
    'LV': {'lat': 56.8,  'lon': 24.5,  'name': 'Latvia'},
    'LT': {'lat': 55.0,  'lon': 23.5,  'name': 'Lithuania'},
    'LU': {'lat': 49.6,  'lon': 6.1,   'name': 'Luxembourg'},
    'NL': {'lat': 52.5,  'lon': 5.0,   'name': 'Netherlands'},
    'NO': {'lat': 60.5,  'lon': 10.5,  'name': 'Norway'},
    'PL': {'lat': 52.0,  'lon': 19.0,  'name': 'Poland'},
    'PT': {'lat': 39.5,  'lon': -8.0,  'name': 'Portugal'},
    'RO': {'lat': 45.9,  'lon': 25.0,  'name': 'Romania'},
    'SK': {'lat': 48.7,  'lon': 19.5,  'name': 'Slovakia'},
    'SI': {'lat': 46.1,  'lon': 14.8,  'name': 'Slovenia'},
    'ES': {'lat': 40.0,  'lon': -3.5,  'name': 'Spain'},
    'SE': {'lat': 60.0,  'lon': 18.0,  'name': 'Sweden'},
    'CH': {'lat': 46.8,  'lon': 8.3,   'name': 'Switzerland'},
    'UK': {'lat': 54.0,  'lon': -2.0,  'name': 'United Kingdom'},
}

# Month names and day-of-year offsets for human-readable date labels
_MONTH_NAMES  = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
_MONTH_STARTS = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]

def day_label(day_of_year):
    for i in range(11, -1, -1):
        if _MONTH_STARTS[i] <= day_of_year:
            return f'{_MONTH_NAMES[i]} {day_of_year - _MONTH_STARTS[i] + 1}'
    return f'day {day_of_year}'

def hour_label(t):
    return f'{day_label(t // 24 + 1)} {t % 24:02d}:00'


# ── Main: process each dataset ────────────────────────────────────────────────
for ds_name, ds in DATASETS.items():
    print(f'\n{"="*60}')
    print(f'Processing {ds_name}...')
    print(f'{"="*60}')

    _df = pd.read_csv(ds['events_csv'])
    _df = _df[(_df['cost_share_cross'] >= THRESH_CS) &
              (_df['duration_days']    >= THRESH_DUR)].reset_index(drop=True)
    _df = _df.sort_values(['sc', 't_start']).reset_index(drop=True)

    EVENTS = [(int(i), int(r.sc), int(r.t_start), int(r.t_end), float(r.cost_share_cross))
              for i, r in _df.iterrows()]

    # Group events by scenario to load ZEN data once per scenario
    SC_EVENTS = defaultdict(list)
    for ev_id, sc_num, t_start, t_end, share in EVENTS:
        SC_EVENTS[sc_num].append((ev_id, t_start, t_end, share))

    print('Loading Results...')
    r = Results(path=ds['path'])

    all_data         = {}
    sp_max_global    = 0.0
    flow_max_global  = 0.0

    for sc_num, sc_evs in sorted(SC_EVENTS.items()):
        sc = f'scenario_{sc_num}'
        print(f'\nProcessing {sc} ({len(sc_evs)} event(s))...')

        # Load once per scenario, filter with masks — avoids redundant ZEN-garden reads per event.
        neb     = r.get_dual('constraint_nodal_energy_balance',      scenario_name=sc)
        dual_tr = r.get_dual('constraint_capacity_factor_transport', scenario_name=sc)
        ftr     = r.get_full_ts('flow_transport', scenario_name=sc)
        cap_tr  = r.get_full_ts('capacity',       scenario_name=sc)

        nodes = sorted({idx[1] for idx in neb.index
                        if len(idx) == 2 and idx[0] == 'electricity'
                        and idx[1] in COUNTRY_COORDS})

        edges = [idx for idx in ftr.index
                 if len(idx) == 2 and idx[0] == 'power_line'
                 and '-' in idx[1]
                 and idx[1].split('-')[0] in COUNTRY_COORDS
                 and idx[1].split('-')[1] in COUNTRY_COORDS]

        # Pre-compute line capacities (static within year)
        edge_caps = {}
        for idx in edges:
            edge = idx[1]
            try:
                edge_caps[edge] = float(np.nanmax(cap_tr.loc['power_line', :, edge].values.astype(float)))
            except:
                edge_caps[edge] = 0.0

        for ev_id, t_start, t_end, share in sc_evs:
            d1  = t_start // 24 + 1
            d2  = t_end   // 24 + 1
            hrs = list(range(t_start, t_end + 1))
            print(f'  Event {ev_id}: days {d1}–{d2}')

            hours_data = {}
            for i, t in enumerate(hrs):
                # Shadow prices per node
                sp = {}
                for n in nodes:
                    try:
                        val = float(neb.loc[('electricity', n), t])
                        sp[n] = round(val, 6)
                        sp_max_global = max(sp_max_global, val)
                    except:
                        sp[n] = 0.0

                # Transmission flows with utilization and dual variable
                flows = []
                for idx in edges:
                    edge = idx[1]
                    try:
                        val = float(ftr.loc[idx, t])
                        if abs(val) > 0.001:
                            src, dst = edge.split('-')
                            try:
                                d_tr = round(float(dual_tr.loc[('power_line', edge), t]), 6)
                            except:
                                d_tr = 0.0
                            cap  = edge_caps.get(edge, 0.0)
                            util = round(abs(val) / cap * 100, 1) if cap > 0 else 0.0
                            flows.append({'f': src, 't': dst, 'v': round(val, 3),
                                          'cap': round(cap, 2), 'util': util, 'dual': d_tr})
                            flow_max_global = max(flow_max_global, abs(val))
                    except:
                        pass

                hours_data[i] = {'t': t, 'l': hour_label(t), 'sp': sp, 'fl': flows}

            all_data[ev_id] = {
                'sc': sc_num,
                'event': {
                    't_start': t_start, 't_end': t_end,
                    'd1': d1, 'd2': d2, 'share': round(share, 4),
                    'n_hours': len(hrs),
                    'label': f'Sc{sc_num} · {day_label(d1)}–{day_label(d2)}'
                },
                'nodes': nodes,
                'hours': hours_data,
            }

    out_dict = {
        'all_data':        all_data,
        'sp_max_global':   sp_max_global,
        'flow_max_global': flow_max_global,
        'iso_map':         ISO_MAP,
        'country_coords':  COUNTRY_COORDS,
    }

    with open(ds['out_json'], 'w', encoding='utf-8') as f:
        json.dump(out_dict, f)

    print(f'\nSaved: {ds["out_json"]}')
    print(f'  SP max global  : {sp_max_global:.4f} M€/GWh')
    print(f'  Flow max global: {flow_max_global:.2f} GWh')
    print(f'  Events stored  : {len(all_data)}')
