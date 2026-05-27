"""
Pairwise Correlation Matrices — Historical Dataset
===================================================
Computes 27×27 Pearson correlation matrices across historical stress events for
wind anomaly, PV anomaly, and heat demand anomaly, one matrix per feature type
plus a combined average matrix with zone boundaries overlaid.

These matrices are used to visually validate the geographic zone definitions
derived from Ward clustering: countries within the same zone should show r ≥ 0.65.

If per-country features have not yet been computed, this script loads ZEN-garden
results and computes them (then caches to CSV for future runs).

Inputs:
  ZEN-garden historical output folder  (if cache missing)
  CSVs/historical/features/features_historical_per_country.csv  (cached per-country features)
  CSVs/historical/events_global_alpha/events_global_historical_alpha25.csv

Outputs:
  htmls/fotos/historical/correlation_matrix/corr_historical_wind_anom.png
  htmls/fotos/historical/correlation_matrix/corr_historical_pv_anom.png
  htmls/fotos/historical/correlation_matrix/corr_historical_heat_anom.png
  htmls/fotos/historical/correlation_matrix/corr_historical_combined.png  (average of 3 matrices, with zone boxes)
"""

# ── Configuration ─────────────────────────────────────────────────────────────
# Output type: CSV data (features cache, if not yet computed) + PNG figures
PATH         = '~/Desktop/ZEN-garden model/outputs/20260202_GF_historical_with_cooling'
OUT_DIR      = '~/Desktop/Bachelor Thesis/analysis/fotos/historical/correlation_matrix'
FEATURES_CSV = '~/Desktop/Bachelor Thesis/CSVs/historical/features/features_historical_per_country.csv'
EVENTS_CSV   = '~/Desktop/Bachelor Thesis/CSVs/historical/events_global_alpha/events_global_historical_alpha25.csv'
THRESH_CS    = 0.2   # minimum cost_share_cross (%) to keep an event
THRESH_DUR   = 1.0   # minimum duration (days) to keep an event
# ──────────────────────────────────────────────────────────────────────────────

import logging
logging.getLogger("pint").setLevel(logging.ERROR)
import os, sys
sys.stderr = open(os.devnull, 'w')
from zen_garden import Results
sys.stderr.close()
sys.stderr = sys.__stderr__

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

PATH         = os.path.expanduser(PATH)
OUT_DIR      = os.path.expanduser(OUT_DIR)
FEATURES_CSV = os.path.expanduser(FEATURES_CSV)
EVENTS_CSV   = os.path.expanduser(EVENTS_CSV)
os.makedirs(OUT_DIR, exist_ok=True)

# ── ZEN-garden API note ───────────────────────────────────────────────────────
# Results(path)               — loads all scenario outputs
# r.get_full_ts('max_load')   — capacity factors per (technology, node), cols: hours 0..8759
# r.get_full_ts('demand')     — demand per (carrier, node), cols: hours 0..8759  [GWh]
# ─────────────────────────────────────────────────────────────────────────────

COUNTRIES = ['AT','BE','BG','CH','CZ','DE','DK','EE','EL','ES',
             'FI','FR','HR','HU','IE','IT','LT','LV','NL',
             'NO','PL','PT','RO','SE','SI','SK','UK']

# Zone definitions — used for ordering and boundary boxes in the combined plot
ZONES = {
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

# ── Load and filter stress events ─────────────────────────────────────────────
df_ev = pd.read_csv(EVENTS_CSV)
df_ev = df_ev[(df_ev['cost_share_cross'] >= THRESH_CS) &
              (df_ev['duration_days']    >= THRESH_DUR)].reset_index(drop=True)
n_ev  = len(df_ev)
print(f'{n_ev} historical events after filtering')

# ── Compute per-country features (or load from cache) ─────────────────────────
if os.path.exists(FEATURES_CSV):
    print(f'Loading cached features from {FEATURES_CSV}...')
    df_feat = pd.read_csv(FEATURES_CSV)
else:
    print('Computing features — loading ZEN-garden Results...')
    r    = Results(path=PATH)
    n_c  = len(COUNTRIES)

    bl_wind  = np.zeros((n_ev, n_c))
    bl_pv    = np.zeros((n_ev, n_c))
    bl_heat  = np.zeros((n_ev, n_c))
    ev_wind  = np.zeros((n_ev, n_c))
    ev_pv    = np.zeros((n_ev, n_c))
    ev_heat  = np.zeros((n_ev, n_c))
    bl_count = np.zeros(n_ev)

    for sc_all in range(1, 61):
        print(f'  sc{sc_all}...', flush=True)
        sc_name = f'scenario_{sc_all}'
        # Load once per scenario, filter with masks — avoids redundant ZEN-garden reads per event.
        ml_all  = r.get_full_ts('max_load', scenario_name=sc_name)
        dem_all = r.get_full_ts('demand',   scenario_name=sc_name)

        for ev_idx, row in df_ev.iterrows():
            t_s   = int(row['t_start'])
            t_e   = int(row['t_end'])
            hrs   = list(range(t_s, t_e + 1))
            ev_sc = int(row['sc'])
            bl_count[ev_idx] += 1

            for ci, n in enumerate(COUNTRIES):
                try:
                    w = float(np.mean(ml_all.loc[('wind_onshore', n)].iloc[hrs].values.astype(float)))
                except KeyError:
                    w = 0.0
                bl_wind[ev_idx, ci] += w
                if sc_all == ev_sc:
                    ev_wind[ev_idx, ci] = w

                try:
                    p = float(np.mean(ml_all.loc[('photovoltaics', n)].iloc[hrs].values.astype(float)))
                except KeyError:
                    p = 0.0
                bl_pv[ev_idx, ci] += p
                if sc_all == ev_sc:
                    ev_pv[ev_idx, ci] = p

                try:
                    h = float(np.mean(dem_all.loc[('heat', n)].iloc[hrs].values.astype(float)))
                except KeyError:
                    h = 0.0
                bl_heat[ev_idx, ci] += h
                if sc_all == ev_sc:
                    ev_heat[ev_idx, ci] = h

    # Compute percentage anomaly: (event - baseline_mean) / baseline_mean × 100
    bl = bl_count[:, np.newaxis]
    wind_anom = np.where(bl_wind > 0, (ev_wind - bl_wind/bl) / (bl_wind/bl) * 100, 0.0)
    pv_anom   = np.where(bl_pv   > 0, (ev_pv   - bl_pv  /bl) / (bl_pv  /bl) * 100, 0.0)
    heat_anom = np.where(bl_heat > 0, (ev_heat  - bl_heat/bl) / (bl_heat/bl) * 100, 0.0)

    feat_cols   = ([f'wind anom {c}' for c in COUNTRIES] +
                   [f'pv anom {c}'   for c in COUNTRIES] +
                   [f'heat anom {c}' for c in COUNTRIES])
    feat_arrays = list(wind_anom.T) + list(pv_anom.T) + list(heat_anom.T)
    df_feat = pd.DataFrame(np.column_stack(feat_arrays), columns=feat_cols)
    df_feat['sc']      = df_ev['sc'].values
    df_feat['t_start'] = df_ev['t_start'].values
    df_feat['t_end']   = df_ev['t_end'].values
    df_feat.to_csv(FEATURES_CSV, index=False)
    print(f'Features saved to {FEATURES_CSV}')

# ── Per-feature correlation matrices ─────────────────────────────────────────
ordered = [c for zone_countries in ZONES.values() for c in zone_countries]

corr_matrices = {}
for feat in ['wind anom', 'pv anom', 'heat anom']:
    cols = [f'{feat} {c}' for c in COUNTRIES]
    data = df_feat[cols].values
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
    ax.set_title(f'Correlation matrix — {feat} — historical events (n={n_ev})', fontsize=13)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, f'corr_historical_{feat.replace(" ", "_")}.png')
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
out = os.path.join(OUT_DIR, 'corr_historical_combined.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {out}')
