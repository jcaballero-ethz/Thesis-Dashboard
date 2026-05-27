"""
Pairwise Correlation Matrices — Future Dataset
===============================================
Computes 27×27 Pearson correlation matrices across stress events for wind anomaly,
PV anomaly, and heat demand anomaly, one matrix per feature type plus a combined
average matrix with zone boundaries overlaid.

These matrices are used to visually validate the geographic zone definitions
derived from Ward clustering: countries within the same zone should show r ≥ 0.65.

Inputs:
  CSVs/future/features/features_alpha25_future_per_country.csv  (per-country anomaly features)

Outputs:
  htmls/fotos/future/correlation_matrix/corr_future_wind_anom.png
  htmls/fotos/future/correlation_matrix/corr_future_pv_anom.png
  htmls/fotos/future/correlation_matrix/corr_future_heat_anom.png
  htmls/fotos/future/correlation_matrix/corr_future_combined.png  (average of 3 matrices, with zone boxes)
"""

# ── Configuration ─────────────────────────────────────────────────────────────
# Output type: PNG figures only
CSV     = os.path.expanduser('~/Desktop/Bachelor Thesis/CSVs_and_JSONs/future/features/features_alpha25_future_per_country.csv')
OUT_DIR = os.path.expanduser('~/Desktop/Bachelor Thesis/analysis/fotos/future/correlation_matrix')
# ──────────────────────────────────────────────────────────────────────────────

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

os.makedirs(OUT_DIR, exist_ok=True)

COUNTRIES = ['AT','BE','BG','CH','CZ','DE','DK','EE','EL','ES',
             'FI','FR','HR','HU','IE','IT','LT','LV','NL',
             'NO','PL','PT','RO','SE','SI','SK','UK']

# Zone definitions — countries ordered by zone for matrix visualization
ZONES = {
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

df    = pd.read_csv(CSV)
ordered = [c for zone_countries in ZONES.values() for c in zone_countries]

# ── Per-feature correlation matrices ─────────────────────────────────────────
corr_matrices = {}
for feat in ['wind anom', 'pv anom', 'heat anom']:
    cols = [f'{feat} {c}' for c in COUNTRIES]
    data = df[cols].values

    corr         = pd.DataFrame(data, columns=COUNTRIES).corr().fillna(0)
    corr_matrices[feat] = corr
    corr_ordered = corr.loc[ordered, ordered]

    fig, ax = plt.subplots(figsize=(18, 16))
    sns.heatmap(
        corr_ordered,
        cmap='RdYlBu_r', vmin=-1, vmax=1,
        annot=True, fmt='.2f', annot_kws={'size': 9},
        ax=ax, cbar_kws={'label': 'Pearson correlation'},
        linewidths=0, linecolor='white',
    )
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11, rotation=90, fontweight='bold')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=11, rotation=0,  fontweight='bold')
    ax.set_title(f'Correlation matrix — {feat} (40 stress events, future)', fontsize=13)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, f'corr_future_{feat.replace(" ", "_")}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')

# ── Combined: average of 3 matrices, with zone boundary boxes ─────────────────
corr_avg = pd.DataFrame(
    np.mean([m.values for m in corr_matrices.values()], axis=0),
    index=COUNTRIES, columns=COUNTRIES
)
corr_ordered = corr_avg.loc[ordered, ordered]

fig, ax = plt.subplots(figsize=(18, 16))
sns.heatmap(
    corr_ordered,
    cmap='RdYlBu_r', vmin=-1, vmax=1,
    annot=True, fmt='.2f', annot_kws={'size': 9},
    ax=ax, cbar_kws={'label': 'Mean Pearson correlation'},
    linewidths=0, linecolor='white',
)

start = 0
for zone, countries in ZONES.items():
    n = len(countries)
    rect = mpatches.Rectangle((start, start), n, n,
                               fill=False, edgecolor='black', linewidth=2.5)
    ax.add_patch(rect)
    start += n

ax.set_xticklabels(ax.get_xticklabels(), fontsize=11, rotation=90, fontweight='bold')
ax.set_yticklabels(ax.get_yticklabels(), fontsize=11, rotation=0,  fontweight='bold')

fig.tight_layout()
out = os.path.join(OUT_DIR, 'corr_future_combined.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {out}')
