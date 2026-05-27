"""
Scatter Plot — Event Cost vs Duration (Future vs Historical)
=============================================================
Static scatter plot comparing stress events from both datasets on the
event_cost vs duration_days plane. Each point is one stress event; color
indicates dataset (future = blue, historical = red).

Only events passing the standard quality filters are included:
  cost_share_cross >= 0.2%  AND  duration_days >= 1.0

Inputs:
  CSVs/future/events_global_alpha/events_global_alpha25.csv
  CSVs/historical/events_global_alpha/events_global_historical_alpha25.csv

Output:
  analysis/fotos/scatter_fut_vs_hist.png
"""

# ── Configuration ─────────────────────────────────────────────────────────────
FUT_CSV  = '~/Desktop/Bachelor Thesis/CSVs_and_JSONs/future/events_global_alpha/events_global_alpha25.csv'
HIST_CSV = '~/Desktop/Bachelor Thesis/CSVs_and_JSONs/historical/events_global_alpha/events_global_historical_alpha25.csv'
OUT      = '~/Desktop/Bachelor Thesis/analysis/fotos/scatter_fut_vs_hist.png'
THRESH_CS  = 0.2
THRESH_DUR = 1.0
# ──────────────────────────────────────────────────────────────────────────────

import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FUT_CSV  = os.path.expanduser(FUT_CSV)
HIST_CSV = os.path.expanduser(HIST_CSV)
OUT      = os.path.expanduser(OUT)

def load_filtered(path):
    df = pd.read_csv(path)
    return df[(df['cost_share_cross'] >= THRESH_CS) & (df['duration_days'] >= THRESH_DUR)]

fut  = load_filtered(FUT_CSV)
hist = load_filtered(HIST_CSV)

fig, ax = plt.subplots(figsize=(9, 6))

ax.scatter(hist['duration_days'], hist['event_cost'] / 1e3,
           color='#e74c3c', s=60, alpha=0.8, label=f'Historical  (n={len(hist)})', zorder=3)
ax.scatter(fut['duration_days'],  fut['event_cost']  / 1e3,
           color='#3498db', s=60, alpha=0.8, label=f'Future  (n={len(fut)})',      zorder=3)

ax.set_xlabel('Duration (days)', fontsize=12)
ax.set_ylabel('Event cost (B€)', fontsize=12)
ax.set_title('Stress events: cost vs duration — future vs historical', fontsize=13)
ax.legend(fontsize=11, frameon=True)
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {OUT}')
