"""
Filter Scatter Plots — Cost Share vs Duration
=============================================
Generates scatter plots of cost_share_cross vs duration_days for each alpha value,
for both future and historical datasets. Filter thresholds are shown as dashed lines,
and events are colored dark (passing) or gray (excluded).

Used to visually motivate the choice of filter thresholds (cost_share ≥ 0.2%,
duration ≥ 1 day) and to show how the event set changes with alpha.

Inputs:
  CSVs/future/events_global_alpha/events_global_alpha{XX}.csv   (7 files)
  CSVs/historical/events_global_alpha/events_global_historical_alpha{XX}.csv  (7 files)

Outputs:
  htmls/fotos/future/scatter_plots/filter_scatter_future_alpha{XX}.png      (7 PNGs)
  htmls/fotos/historical/scatter_plots/filter_scatter_historical_alpha{XX}.png  (7 PNGs)
"""

# ── Configuration ─────────────────────────────────────────────────────────────
# Output type: PNG figures only  (14 PNGs: 7 alpha values × 2 datasets)
ALPHAS     = [10, 15, 20, 25, 30, 35, 40]
THRESH_CS  = 0.2   # minimum cost_share_cross (%)
THRESH_DUR = 1.0   # minimum duration (days)
OUT_DIR    = '~/Desktop/Bachelor Thesis/analysis/fotos'
# ──────────────────────────────────────────────────────────────────────────────

import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = os.path.expanduser(OUT_DIR)
os.makedirs(os.path.join(OUT_DIR, 'future',     'scatter_plots'), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, 'historical', 'scatter_plots'), exist_ok=True)

DATASETS = {
    'future': {
        10: os.path.expanduser('~/Desktop/Bachelor Thesis/CSVs/future/events_global_alpha/events_global_alpha10.csv'),
        15: os.path.expanduser('~/Desktop/Bachelor Thesis/CSVs/future/events_global_alpha/events_global_alpha15.csv'),
        20: os.path.expanduser('~/Desktop/Bachelor Thesis/CSVs/future/events_global_alpha/events_global_alpha20.csv'),
        25: os.path.expanduser('~/Desktop/Bachelor Thesis/CSVs/future/events_global_alpha/events_global_alpha25.csv'),
        30: os.path.expanduser('~/Desktop/Bachelor Thesis/CSVs/future/events_global_alpha/events_global_alpha30.csv'),
        35: os.path.expanduser('~/Desktop/Bachelor Thesis/CSVs/future/events_global_alpha/events_global_alpha35.csv'),
        40: os.path.expanduser('~/Desktop/Bachelor Thesis/CSVs/future/events_global_alpha/events_global_alpha40.csv'),
    },
    'historical': {
        10: os.path.expanduser('~/Desktop/Bachelor Thesis/CSVs/historical/events_global_alpha/events_global_historical_alpha10.csv'),
        15: os.path.expanduser('~/Desktop/Bachelor Thesis/CSVs/historical/events_global_alpha/events_global_historical_alpha15.csv'),
        20: os.path.expanduser('~/Desktop/Bachelor Thesis/CSVs/historical/events_global_alpha/events_global_historical_alpha20.csv'),
        25: os.path.expanduser('~/Desktop/Bachelor Thesis/CSVs/historical/events_global_alpha/events_global_historical_alpha25.csv'),
        30: os.path.expanduser('~/Desktop/Bachelor Thesis/CSVs/historical/events_global_alpha/events_global_historical_alpha30.csv'),
        35: os.path.expanduser('~/Desktop/Bachelor Thesis/CSVs/historical/events_global_alpha/events_global_historical_alpha35.csv'),
        40: os.path.expanduser('~/Desktop/Bachelor Thesis/CSVs/historical/events_global_alpha/events_global_historical_alpha40.csv'),
    },
}

COLOR_PASS = '#1a1a1a'   # dark black for retained
COLOR_FAIL = '#bdc3c7'   # grey for excluded

for ds_name, paths in DATASETS.items():
    for alpha in ALPHAS:
        df = pd.read_csv(paths[alpha])
        passed = (df['cost_share_cross'] >= THRESH_CS) & (df['duration_days'] >= THRESH_DUR)
        pct = passed.sum() / len(df) * 100

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.scatter(df.loc[~passed, 'duration_days'], df.loc[~passed, 'cost_share_cross'],
                   color=COLOR_FAIL, s=18, alpha=0.7, zorder=2)
        ax.scatter(df.loc[passed,  'duration_days'], df.loc[passed,  'cost_share_cross'],
                   color=COLOR_PASS, s=18, alpha=0.9, zorder=3)

        ax.axvline(THRESH_DUR, color='firebrick', linewidth=1.4, linestyle='--')
        ax.axhline(THRESH_CS,  color='steelblue', linewidth=1.4, linestyle='--')

        ax.set_title(f'α = {alpha}% — {ds_name}', fontsize=13, fontweight='bold')
        ax.set_xlabel('Duration (days)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Cost share (%)', fontsize=11, fontweight='bold')

        ax.text(0.97, 0.97, f'{passed.sum()}/{len(df)} events\n({pct:.1f}%)',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=11, fontweight='bold', color='black',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.9))

        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        out = os.path.join(OUT_DIR, ds_name, 'scatter_plots', f'filter_scatter_{ds_name}_alpha{alpha:02d}.png')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Saved: {out}')

print('Done.')
