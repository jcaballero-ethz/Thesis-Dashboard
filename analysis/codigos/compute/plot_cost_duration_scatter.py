"""
Cost vs Duration Scatter — Static PNGs per Dataset
====================================================
Generates one scatter plot per dataset showing event_cost vs duration_days
for all stress events passing the standard quality filters at α = 25%.
Points are colored by cost_share_cross (%).

Inputs:
  CSVs/future/events_global_alpha/events_global_alpha25.csv
  CSVs/historical/events_global_alpha/events_global_historical_alpha25.csv

Outputs:
  analysis/fotos/future/scatter_plots/cost_duration_scatter_future.png
  analysis/fotos/historical/scatter_plots/cost_duration_scatter_historical.png
"""

# ── Configuration ─────────────────────────────────────────────────────────────
THRESH_CS  = 0.2
THRESH_DUR = 1.0
OUT_DIR    = '~/Desktop/Bachelor Thesis/analysis/fotos'
DATASETS = {
    'future': {
        'label': 'Future',
        'csv':   '~/Desktop/Bachelor Thesis/CSVs/future/events_global_alpha/events_global_alpha25.csv',
        'color': '#3498db',
    },
    'historical': {
        'label': 'Historical',
        'csv':   '~/Desktop/Bachelor Thesis/CSVs/historical/events_global_alpha/events_global_historical_alpha25.csv',
        'color': '#e74c3c',
    },
}
# ──────────────────────────────────────────────────────────────────────────────

import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

OUT_DIR = os.path.expanduser(OUT_DIR)

for ds_key, cfg in DATASETS.items():
    df = pd.read_csv(os.path.expanduser(cfg['csv']))
    df = df[(df['cost_share_cross'] >= THRESH_CS) & (df['duration_days'] >= THRESH_DUR)]

    fig, ax = plt.subplots(figsize=(9, 6))

    sc = ax.scatter(df['duration_days'], df['event_cost'] / 1e3,
                    c=df['cost_share_cross'], cmap='YlOrRd',
                    s=60, alpha=0.85, zorder=3,
                    norm=mcolors.Normalize(vmin=df['cost_share_cross'].min(),
                                           vmax=df['cost_share_cross'].max()))
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label('Cost share (%)', fontsize=10)

    ax.set_xlabel('Duration (days)', fontsize=12)
    ax.set_ylabel('Event cost (B€)', fontsize=12)
    ax.set_title(f'Stress events: cost vs duration — {cfg["label"]}  (α = 25%)',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, ds_key, 'scatter_plots',
                       f'cost_duration_scatter_{ds_key}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')

print('\nDone.')
