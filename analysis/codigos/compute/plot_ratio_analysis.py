"""
Ratio Analysis — Static PNGs
==============================
Generates one PNG per dataset showing how the intensity ratio
ρ = cost_share / duration (% per day) evolves across α thresholds for the
top-10 most persistent stress events. Mirrors the content of ratio_analysis.html.

An event is "persistent" if it appears in the top-10 by cost_share_cross
across the most α thresholds. Events are identified by (sc, t_start, t_end).

Inputs:
  CSVs/future/events_global_alpha/events_global_alpha{10..40}.csv
  CSVs/historical/events_global_alpha/events_global_historical_alpha{10..40}.csv

Outputs:
  analysis/fotos/future/ratio_analysis_future.png
  analysis/fotos/historical/ratio_analysis_historical.png
"""

# ── Configuration ─────────────────────────────────────────────────────────────
ALPHAS  = [10, 15, 20, 25, 30, 35, 40]
TOP_N   = 10
OUT_DIR = '~/Desktop/Bachelor Thesis/analysis/fotos'
DATASETS = {
    'future': {
        'label': 'Future',
        'csv': lambda a: f'~/Desktop/Bachelor Thesis/CSVs/future/events_global_alpha/events_global_alpha{a:02d}.csv',
    },
    'historical': {
        'label': 'Historical',
        'csv': lambda a: f'~/Desktop/Bachelor Thesis/CSVs/historical/events_global_alpha/events_global_historical_alpha{a:02d}.csv',
    },
}
COLORS = ['#3498db','#e74c3c','#2ecc71','#9b59b6','#f39c12',
          '#1abc9c','#e67e22','#34495e','#e91e63','#00bcd4']
# ──────────────────────────────────────────────────────────────────────────────

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = os.path.expanduser(OUT_DIR)


def load_dfs(cfg):
    return {a: pd.read_csv(os.path.expanduser(cfg['csv'](a))) for a in ALPHAS}


def find_persistent(dfs):
    top_per = {a: dfs[a].nlargest(TOP_N, 'cost_share_cross') for a in ALPHAS}
    pool = []
    for alpha in ALPHAS:
        for _, row in top_per[alpha].iterrows():
            sc, ts, te = int(row['sc']), int(row['t_start']), int(row['t_end'])
            if not any(e['sc'] == sc and
                       ((e['ts'] <= ts and e['te'] >= te) or
                        (ts <= e['ts'] and te >= e['te']))
                       for e in pool):
                pool.append({'sc': sc, 'ts': ts, 'te': te})
    for ev in pool:
        ev['persistence'] = sum(
            1 for a in ALPHAS
            if len(top_per[a][(top_per[a]['sc'] == ev['sc']) &
                               (top_per[a]['t_start'] <= ev['ts']) &
                               (top_per[a]['t_end']   >= ev['te'])]) > 0
        )
    pool.sort(key=lambda x: x['persistence'], reverse=True)
    return pool[:TOP_N]


def compute_rho(pool, dfs):
    events = []
    for ev in pool:
        sc, ts0, te0 = ev['sc'], ev['ts'], ev['te']
        rho_vals = []
        for a in ALPHAS:
            m = dfs[a][(dfs[a]['sc'] == sc) &
                       (dfs[a]['t_start'] <= te0) &
                       (dfs[a]['t_end']   >= ts0)]
            if len(m) > 0:
                cs  = float(m.iloc[0]['cost_share_cross'])
                dur = float(m.iloc[0]['duration_days'])
                rho_vals.append(cs / dur if dur > 0 else np.nan)
            else:
                rho_vals.append(np.nan)
        events.append({'label': f'sc{sc}', 'rho': rho_vals,
                       'persistence': ev['persistence']})
    return events


for ds_key, cfg in DATASETS.items():
    dfs    = load_dfs(cfg)
    pool   = find_persistent(dfs)
    events = compute_rho(pool, dfs)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, ev in enumerate(events):
        rho = ev['rho']
        valid = [(a, r) for a, r in zip(ALPHAS, rho) if not np.isnan(r)]
        if not valid:
            continue
        xs, ys = zip(*valid)
        ax.plot(xs, ys, marker='o', color=COLORS[i % len(COLORS)],
                linewidth=2, markersize=6,
                label=f'{ev["label"]}  (persist={ev["persistence"]})')

    ax.set_xlabel('α threshold (%)', fontsize=12)
    ax.set_ylabel('ρ = cost share / duration  (% per day)', fontsize=12)
    ax.set_title(f'Intensity ratio ρ vs α — {cfg["label"]}', fontsize=13,
                 fontweight='bold')
    ax.axvline(25, color='gray', linewidth=1.4, linestyle='--', zorder=1, label='α = 25% (main)')
    ax.set_xticks(ALPHAS)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, frameon=True, loc='upper right',
              ncol=2 if len(events) > 6 else 1)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, ds_key, f'ratio_analysis_{ds_key}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')

print('\nDone.')
