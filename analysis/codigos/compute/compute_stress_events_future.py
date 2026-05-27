"""
Cross-Scenario Stress Event Identification — Future Dataset
============================================================
Identifies weather-driven stress events from ZEN-garden future climate outputs.

Method:
  1. For each scenario, compute the hourly stress cost Φ_t,s = Σ_{n,c} d_{c,n,t,s} · λ_{c,n,t,s}
  2. Pool all (timestep, scenario) pairs globally and rank by stress cost descending.
     Select pairs until their cumulative cost reaches α × total_cross_scenario_cost.
  3. Within each scenario, group selected timesteps into continuous events (gap < 48 h).
  4. Compute each event's cost share relative to the total cross-scenario cost.

Inputs:
  ZEN-garden future output folder (PATH)

Outputs:
  ~/Desktop/Bachelor Thesis/CSVs_and_JSONs/future/events_global_alpha/events_global_alpha{XX}.csv
  One CSV per alpha value, with columns:
    rank, sc, t_start, t_end, duration_days, event_cost, cost_share_cross
"""

# ── Configuration ─────────────────────────────────────────────────────────────
# Output type: CSV data
PATH          = '~/Desktop/ZEN-garden model/outputs/20260202_GF_future_with_cooling'
SCENARIOS     = list(range(1, 61))       # scenario indices to process
GAP_THRESHOLD = 48                       # max gap (hours) between selected timesteps within one event
ALPHAS        = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]  # cost-coverage fractions to sweep
OUT_DIR       = '~/Desktop/Bachelor Thesis/CSVs_and_JSONs/future/events_global_alpha'
# ──────────────────────────────────────────────────────────────────────────────

import logging
logging.getLogger("pint").setLevel(logging.ERROR)
import sys, os
sys.stderr = open(os.devnull, 'w')
from zen_garden import Results
sys.stderr.close()
sys.stderr = sys.__stderr__

import numpy as np
import csv

PATH    = os.path.expanduser(PATH)
OUT_DIR = os.path.expanduser(OUT_DIR)

# ── ZEN-garden API note ───────────────────────────────────────────────────────
# Results(path)          — loads all scenario outputs from the ZEN-garden folder
# r.get_dual(constraint) — returns a DataFrame of dual variables (shadow prices)
#                          index: (carrier, node), e.g. ('electricity', 'DE')
#                          columns: integer timestep hours 0..8759
#                          values: M€/GWh  (marginal cost of serving one extra GWh)
# r.get_full_ts(var)     — returns a DataFrame of primal time-series values
#                          index: (technology_or_carrier, node)
#                          columns: integer timestep hours 0..8759
# ─────────────────────────────────────────────────────────────────────────────

# Month names and day-of-year offsets for human-readable date labels
_MONTH_NAMES  = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
_MONTH_STARTS = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]

def day_label(day_of_year):
    """Convert a 1-indexed day-of-year to a human-readable label (e.g. 'Jan 3')."""
    for i in range(11, -1, -1):
        if _MONTH_STARTS[i] <= day_of_year:
            return f'{_MONTH_NAMES[i]} {day_of_year - _MONTH_STARTS[i] + 1}'
    return f'day {day_of_year}'

# ── Step 1: Load ZEN-garden outputs and compute hourly stress cost per scenario ─
print('Loading ZEN-garden results...')
r = Results(path=PATH)

scenario_data    = {}   # sc_num -> {'C_t': array of hourly costs, 'timesteps': list}
global_pool      = []   # list of (cost, sc_num, t_idx) for all scenarios and timesteps
total_cross_cost = 0.0

for sc_num in SCENARIOS:
    sc = f'scenario_{sc_num}'
    print(f'  Processing {sc}...', flush=True)

    # Load full DataFrames once per scenario and filter with boolean masks.
    # Using ZEN-garden's index= parameter per event would re-read data for every
    # event in the scenario — much slower when multiple events share the same scenario.
    # shadow prices λ_{c,n,t}: dual of nodal energy balance  [M€/GWh]
    sp     = r.get_dual('constraint_nodal_energy_balance', scenario_name=sc)
    # demand d_{c,n,t}: exogenous demand per carrier and node  [GWh]
    demand = r.get_full_ts('demand', scenario_name=sc)

    timesteps = sorted(sp.columns.tolist())   # list of hour integers 0..8759
    T         = len(timesteps)
    # infer modelled nodes from the 'heat' carrier rows of the shadow price DataFrame
    nodes     = sorted({idx[1] for idx in sp.index if len(idx) == 2 and idx[0] == 'heat'})

    # Compute Φ_t,s = Σ_{n,c} demand_{c,n,t} × shadow_price_{c,n,t}
    C_t = np.zeros(T)
    for t_idx, t in enumerate(timesteps):
        cost = 0.0
        for node in nodes:
            for carrier in ('heat', 'electricity', 'cooling'):
                d, lam = 0.0, 0.0
                try: d   = float(demand.loc[(carrier, node), t])
                except KeyError: pass
                try: lam = float(sp.loc[(carrier, node), t])
                except KeyError: pass
                cost += d * lam
        C_t[t_idx] = cost

    sc_total          = float(np.sum(C_t))
    total_cross_cost += sc_total
    scenario_data[sc_num] = {'C_t': C_t, 'timesteps': timesteps}
    global_pool.extend((cost, sc_num, t_idx) for t_idx, cost in enumerate(C_t))
    print(f'    Annual stress cost: {sc_total:.1f} M€')

print(f'\nTotal cross-scenario cost : {total_cross_cost:.1f} M€')
print(f'Per-scenario mean         : {total_cross_cost / len(SCENARIOS):.1f} M€')
print(f'Total timesteps in pool   : {len(global_pool)}')

# Sort global pool once by cost descending — reused for all alpha values
global_pool.sort(key=lambda x: -x[0])

SEP = '=' * 90

# ── Step 2–4: Sweep alpha values ───────────────────────────────────────────────
for alpha in ALPHAS:
    print(f'\n{SEP}')
    print(f'  α = {alpha:.0%}  — selecting top timestep-scenario pairs globally')
    print(SEP)

    # Select pairs from global pool until cumulative cost >= α × total_cross_cost
    stressful  = {}    # sc_num -> set of selected t_idx
    cumulative = 0.0
    for cost, sc_num, t_idx in global_pool:
        stressful.setdefault(sc_num, set()).add(t_idx)
        cumulative += cost
        if cumulative >= alpha * total_cross_cost:
            break

    print(f'  Scenarios with selected timesteps : {len(stressful)} / {len(SCENARIOS)}')
    print(f'  Total selected timesteps          : {sum(len(v) for v in stressful.values())}')

    # Group selected timesteps into continuous events within each scenario
    all_events = []
    for sc_num in SCENARIOS:
        if sc_num not in stressful:
            continue

        C_t       = scenario_data[sc_num]['C_t']
        ts        = scenario_data[sc_num]['timesteps']
        sel_idx   = np.array(sorted(stressful[sc_num]))

        event_start = sel_idx[0]
        prev_idx    = sel_idx[0]
        for i in range(1, len(sel_idx)):
            if sel_idx[i] - prev_idx > GAP_THRESHOLD:
                # Gap too large — close current event and start a new one
                e_cost = float(np.sum(C_t[event_start:prev_idx + 1]))
                all_events.append({
                    'sc': sc_num, 't_start': ts[event_start], 't_end': ts[prev_idx],
                    'event_cost': e_cost,
                    'cost_share_cross': e_cost / total_cross_cost * 100,
                })
                event_start = sel_idx[i]
            prev_idx = sel_idx[i]
        # Close last event
        e_cost = float(np.sum(C_t[event_start:prev_idx + 1]))
        all_events.append({
            'sc': sc_num, 't_start': ts[event_start], 't_end': ts[prev_idx],
            'event_cost': e_cost,
            'cost_share_cross': e_cost / total_cross_cost * 100,
        })

    all_events.sort(key=lambda x: -x['cost_share_cross'])

    # Save events to CSV
    csv_path = os.path.join(OUT_DIR, f'events_global_alpha{int(alpha * 100):02d}.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['rank','sc','t_start','t_end','duration_days','event_cost','cost_share_cross'])
        writer.writeheader()
        for rank, e in enumerate(all_events, 1):
            dur = (e['t_end'] - e['t_start'] + 1) / 24
            writer.writerow({
                'rank': rank, 'sc': e['sc'],
                't_start': e['t_start'], 't_end': e['t_end'],
                'duration_days': round(dur, 2),
                'event_cost': round(e['event_cost'], 2),
                'cost_share_cross': round(e['cost_share_cross'], 4),
            })
    print(f'  Saved: {csv_path}')
    print(f'  Total events: {len(all_events)}')

    # Print summary table
    print(f'\n  {"#":>3}  {"sc":>4}  {"Period":>22}  {"Timesteps":>16}  {"Dur":>7}  {"Event cost (M€)":>16}  {"Cost share":>11}')
    print(f'  {"-" * 93}')
    for rank, e in enumerate(all_events, 1):
        d1  = e['t_start'] // 24 + 1
        d2  = e['t_end']   // 24 + 1
        dur = (e['t_end'] - e['t_start'] + 1) / 24
        print(f'  {rank:>3}  sc{e["sc"]:>2}  '
              f'{day_label(d1) + "–" + day_label(d2):>22}  '
              f'h {e["t_start"]:>4}–{e["t_end"]:<4}  '
              f'{dur:>5.1f}d  '
              f'{e["event_cost"]:>16.1f}  '
              f'{e["cost_share_cross"]:>9.2f}%')

print(f'\n{SEP}')
print('Done.')
